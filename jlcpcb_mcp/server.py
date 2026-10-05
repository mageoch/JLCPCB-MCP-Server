"""
LCSC MCP Server — official JLCPCB API only.

The local SQLite database is treated as a 24h-membership / 6h-detail cache,
not as a source of truth. Every search will:

  1. If the assembly library membership is older than JLCPCB_MEMBERSHIP_TTL_HOURS,
     paginate getComponentLibraryList, diff against the local DB, fetch full
     details for new codes, and delete codes that disappeared from the library.
  2. After a local search, refetch full details (in one batched API call) for
     any returned row whose last_updated is older than JLCPCB_CACHE_TTL_HOURS.

Required env vars:
  JLCPCB_APP_ID
  JLCPCB_API_KEY
  JLCPCB_API_SECRET

Optional:
  JLCPCB_DB_PATH                — path to SQLite DB (default: ./data/lcsc_parts.db)
  JLCPCB_CACHE_TTL_HOURS        — per-row detail TTL (default: 6)
  JLCPCB_MEMBERSHIP_TTL_HOURS   — library-membership TTL (default: 24)
"""

import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

from .client import JLCPCBClient
from .db import PartsDB

_LOG_FILE = Path(__file__).parent.parent / "data" / "jlcpcb_mcp.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "JLCPCB Parts",
    instructions=(
        "Search and retrieve LCSC/JLCPCB electronic components from the JLCPCB "
        "assembly library (Basic + Extended). The local SQLite database acts as a "
        "short-lived cache: membership is auto-refreshed every "
        "JLCPCB_MEMBERSHIP_TTL_HOURS (default 24), and per-row stock/price details "
        "every JLCPCB_CACHE_TTL_HOURS (default 6)."
    ),
)

CACHE_TTL_HOURS = float(os.getenv("JLCPCB_CACHE_TTL_HOURS", "6"))
MEMBERSHIP_TTL_HOURS = float(os.getenv("JLCPCB_MEMBERSHIP_TTL_HOURS", "24"))


def _db() -> PartsDB:
    return PartsDB(os.getenv("JLCPCB_DB_PATH"))


def _client() -> JLCPCBClient:
    return JLCPCBClient()  # raises EnvironmentError if credentials missing


# ---------------------------------------------------------------------------
# Cache refresh helpers
# ---------------------------------------------------------------------------

LIBRARY_TOTAL_KEY = "library_total_codes"
# Search / get_part answer the user directly: give up after ~6 s (2 + 4 s backoff)
# on a persistent 429/5xx instead of the bulk default of several minutes.
INTERACTIVE_MAX_RETRIES = 2
PROGRESS_EVERY_PAGES = 50


def _stub_progress_logger(db: PartsDB, started_at: float):
    """on_progress callback for iter_library_stubs: logs every PROGRESS_EVERY_PAGES pages.

    The page total is estimated from the last completed refresh; on the first
    load there is none, so only pages / codes / elapsed time are shown.
    """
    last_total = db.get_metadata(LIBRARY_TOTAL_KEY)
    state = {"page": 0, "page_size": None}

    def on_progress(total: int) -> None:
        state["page"] += 1
        if state["page_size"] is None:
            state["page_size"] = total or 1
        if state["page"] % PROGRESS_EVERY_PAGES:
            return
        pages = f"page {state['page']}"
        if last_total:
            est_pages = -(-int(last_total) // state["page_size"])
            pages += f" / ~{est_pages} (est.)"
        logger.info(
            "Library list: %s, %d codes, %.1f min elapsed",
            pages, total, (time.time() - started_at) / 60,
        )

    return on_progress


def _ensure_membership_fresh(db: PartsDB) -> Optional[str]:
    """
    If the library membership is older than MEMBERSHIP_TTL_HOURS or never
    populated, paginate getComponentLibraryList, diff against the local DB,
    enrich new codes via getComponentDetailByCode (batches of 1000), and
    delete codes that are no longer part of the library.

    Blocks the caller for the duration of the refresh — typical cost is ~30 s
    for the stub pagination plus a few seconds per 1000 new codes to enrich.

    Returns a human-readable summary string when a refresh ran, or None when
    the cache was already fresh.
    """
    age = db.library_age_hours()
    if age is not None and age <= MEMBERSHIP_TTL_HOURS:
        return None

    started_at = time.time()
    if age is None:
        logger.info("Library membership not yet populated — refreshing from API…")
        log_age = "never populated"
    else:
        logger.info("Library membership is %.1f h old — refreshing from API…", age)
        log_age = f"{age:.1f}h old"

    try:
        client = _client()

        # 1. Fetch all stubs.
        api_codes = {
            stub["componentCode"]
            for stub in client.iter_library_stubs(
                on_progress=_stub_progress_logger(db, started_at)
            )
            if stub.get("componentCode")
        }
        if not api_codes:
            return "Library refresh skipped: API returned no stubs."

        # 2. Diff against the local DB.
        local_codes = db.get_all_lcsc_codes()
        new_codes = sorted(api_codes - local_codes)

        # 3. Enrich new codes (batches of 1000 handled inside the client).
        if new_codes:
            logger.info("Library refresh: fetching details for %d new codes…", len(new_codes))
            details = client.get_parts_details(new_codes)
            db.import_batch(details)

        # 4. Drop codes that are no longer in the library (purges any
        #    leftover entries from earlier full-catalog imports too).
        removed = db.delete_codes_not_in(api_codes)
        if removed:
            logger.info("Library refresh: removed %d delisted codes", removed)

        # 5. Rebuild FTS once at the end.
        logger.info("Library refresh: rebuilding search index…")
        db.rebuild_fts()

        # 6. Mark cache as fresh; remember the size for the next progress estimate.
        db.set_metadata("basic_library_refreshed_at", str(started_at))
        db.set_metadata(LIBRARY_TOTAL_KEY, str(len(api_codes)))

        elapsed = time.time() - started_at
        msg = (
            f"Library membership refreshed ({log_age}, took {elapsed:.0f}s): "
            f"+{len(new_codes)} new, -{removed} removed."
        )
        logger.info(msg)
        return msg
    except Exception as exc:
        logger.warning("Library refresh failed: %s", exc)
        return f"Library refresh failed ({exc}); falling back to existing local cache."


def _refresh_stale_details(
    db: PartsDB,
    codes: list[str],
    ttl_hours: Optional[float] = None,
) -> int:
    """
    For each code whose row is older than ttl_hours, refetch the full details
    from the API and upsert. Best-effort: API/network errors are logged but
    don't propagate (search results remain available with stale data).

    Returns the number of rows successfully refreshed.
    """
    if ttl_hours is None:
        ttl_hours = CACHE_TTL_HOURS
    stale = db.stale_codes(codes, ttl_hours)
    if not stale:
        return 0
    try:
        client = _client()
        details = client.get_parts_details(stale, max_retries=INTERACTIVE_MAX_RETRIES)
        db.import_batch(details)
        return len(details)
    except Exception as exc:
        logger.warning("Stale detail refresh failed for %d codes: %s", len(stale), exc)
        return 0


def _refreshed_search_results(db: PartsDB, parts: list[dict]) -> list[dict]:
    """
    After a search, refetch any stale rows in `parts` and return rows in the
    original order, but with fresh data. Rows that disappear from the DB after
    the refresh (rare — would mean the part was just dropped from the library)
    are dropped from the result.
    """
    if not parts:
        return parts
    codes = [p["lcsc"] for p in parts if p.get("lcsc")]
    _refresh_stale_details(db, codes)
    fresh: list[dict] = []
    for code in codes:
        row = db.get(code)
        if row:
            fresh.append(row)
    return fresh


def _db_error_response(exc: Exception) -> dict:
    """Map a sqlite3.DatabaseError to a structured tool response.

    Surfaces FTS / DB corruption to the caller (rather than returning an
    empty result) and tells them how to recover.
    """
    msg = str(exc)
    response: dict = {"success": False, "error": msg}
    if "malformed" in msg or "corrupt" in msg.lower():
        response["hint"] = (
            "FTS index appears corrupted. Run "
            "`python -m jlcpcb_mcp.scripts.repair_fts` (or call download_library) "
            "to rebuild it."
        )
    return response


def _safe_search(do_search) -> dict:
    """Run a DB search callable and convert sqlite3 errors to error responses."""
    try:
        return {"success": True, "parts": do_search()}
    except sqlite3.DatabaseError as exc:
        logger.error("DB error during search: %s", exc)
        return _db_error_response(exc)


# ---------------------------------------------------------------------------
# Tool: download_library
# ---------------------------------------------------------------------------

@mcp.tool()
def download_library() -> dict:
    """
    Force a full refresh of the JLCPCB assembly library (Basic + Extended).

    Equivalent to deleting `basic_library_refreshed_at` and triggering a
    membership refresh — but also re-fetches details for *every* code, not
    just new ones. Useful when stock/price data has drifted across the
    whole library and the per-row TTL hasn't expired yet.

    Returns:
        Download statistics.
    """
    db = _db()
    client = _client()

    total = 0

    def on_batch(parts: list) -> None:
        nonlocal total
        total += db.import_batch(parts)

    try:
        client.download_library(on_batch=on_batch)
        db.rebuild_fts()
        # rebuild_specs is safe — _extract_specs is run inside import_batch,
        # but rebuilding ensures specs reflect the latest descriptions.
        db.rebuild_specs()
        db.set_metadata("basic_library_refreshed_at", str(time.time()))
        db.set_metadata(LIBRARY_TOTAL_KEY, str(total))
        stats = db.stats()
        return {
            "success": True,
            "message": f"Library download complete: {total} parts imported",
            **stats,
        }
    except Exception as exc:
        logger.error("Library download failed: %s", exc)
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool: search_parts
# ---------------------------------------------------------------------------

@mcp.tool()
def search_parts(
    query: Optional[str] = None,
    category: Optional[str] = None,
    package: Optional[str] = None,
    library_type: Optional[str] = None,
    manufacturer: Optional[str] = None,
    in_stock: bool = True,
    limit: int = 20,
    skip_refresh: bool = False,
) -> dict:
    """
    Search the local LCSC parts database (assembly library, Basic + Extended).

    The cache is auto-refreshed before the search runs (unless skip_refresh=True):
      - Membership (which codes belong to the library) every JLCPCB_MEMBERSHIP_TTL_HOURS.
      - Per-row stock/price/description for the returned rows, every JLCPCB_CACHE_TTL_HOURS.

    Args:
        query: Free-text search. Multi-word is AND-matched. Tokens with
            punctuation (e.g. '3.3V', 'SOT-23') are treated as exact phrases;
            plain words get a prefix wildcard ('LDO' → matches 'LDO regulators').
        category: Filter by category or subcategory (e.g. 'Resistors', 'Capacitors').
        package: Filter by package (e.g. '0603', 'SOT-23', 'QFN-32').
        library_type: 'Basic' or 'Extended', or None for both.
        manufacturer: Filter by manufacturer name.
        in_stock: Only return parts with stock > 0. Default: True.
        limit: Maximum results. Default: 20.
        skip_refresh: Skip the pre-search membership/details refresh. Use when
            you want the fastest possible response and stale stock/price are OK.

    Returns:
        List of matching parts with LCSC code, description, price breaks, stock.
    """
    db = _db()
    warning = None if skip_refresh else _ensure_membership_fresh(db)
    response = _safe_search(lambda: db.search(
        query=query,
        category=category,
        package=package,
        library_type=library_type,
        manufacturer=manufacturer,
        in_stock=in_stock,
        limit=limit,
    ))
    if not response["success"]:
        return response
    parts = response["parts"] if skip_refresh else _refreshed_search_results(db, response["parts"])
    result: dict = {"success": True, "count": len(parts), "parts": parts}
    if warning:
        result["warning"] = warning
    return result


# ---------------------------------------------------------------------------
# Tool: search_resistors
# ---------------------------------------------------------------------------

@mcp.tool()
def search_resistors(
    value: Optional[str] = None,
    value_min_ohms: Optional[float] = None,
    value_max_ohms: Optional[float] = None,
    package: Optional[str] = None,
    tolerance: Optional[str] = None,
    tolerance_max_pct: Optional[float] = None,
    power_rating: Optional[str] = None,
    power_min_w: Optional[float] = None,
    library_type: Optional[str] = None,
    in_stock: bool = True,
    limit: int = 20,
    skip_refresh: bool = False,
) -> dict:
    """
    Search resistors with parametric filters.

    The cache is auto-refreshed (see search_parts for details).

    Args:
        value: Resistance as text, e.g. '10kΩ', '100R', '4.7k'.
        value_min_ohms: Minimum resistance in Ohms, e.g. 1000 for 1 kΩ.
        value_max_ohms: Maximum resistance in Ohms, e.g. 100000 for 100 kΩ.
        package: SMD/THT package, e.g. '0402', '0603', '0805', 'AXIAL'.
        tolerance: Tolerance as text, e.g. '±1%', '±5%'.
        tolerance_max_pct: Maximum tolerance in percent, e.g. 1.0 for ±1 % or better.
        power_rating: Power rating as text, e.g. '1/10W', '1/4W'.
        power_min_w: Minimum power rating in Watts, e.g. 0.25 for 1/4 W.
        library_type: 'Basic' or 'Extended', or None for both.
        in_stock: Only return parts with stock > 0. Default: True.
        limit: Maximum results. Default: 20.
        skip_refresh: Skip the pre-search membership/details refresh.

    Returns:
        List of matching resistors with LCSC code, description, package, price, stock.
    """
    db = _db()
    warning = None if skip_refresh else _ensure_membership_fresh(db)
    response = _safe_search(lambda: db.search_passive(
        component_type="resistor",
        value=value,
        value_min=value_min_ohms,
        value_max=value_max_ohms,
        package=package,
        tolerance=tolerance,
        tolerance_max_pct=tolerance_max_pct,
        power_rating=power_rating,
        power_min_w=power_min_w,
        library_type=library_type,
        in_stock=in_stock,
        limit=limit,
    ))
    if not response["success"]:
        return response
    parts = response["parts"] if skip_refresh else _refreshed_search_results(db, response["parts"])
    result: dict = {"success": True, "count": len(parts), "parts": parts}
    if warning:
        result["warning"] = warning
    return result


# ---------------------------------------------------------------------------
# Tool: search_capacitors
# ---------------------------------------------------------------------------

@mcp.tool()
def search_capacitors(
    value: Optional[str] = None,
    value_min_farads: Optional[float] = None,
    value_max_farads: Optional[float] = None,
    package: Optional[str] = None,
    voltage_rating: Optional[str] = None,
    voltage_min_v: Optional[float] = None,
    dielectric: Optional[str] = None,
    tolerance: Optional[str] = None,
    library_type: Optional[str] = None,
    in_stock: bool = True,
    limit: int = 20,
    skip_refresh: bool = False,
) -> dict:
    """
    Search capacitors with parametric filters.

    The cache is auto-refreshed (see search_parts for details).

    Args:
        value: Capacitance as text, e.g. '100nF', '10µF', '0.1uF', '1pF'.
        value_min_farads: Minimum capacitance in Farads, e.g. 100e-9 for 100 nF.
        value_max_farads: Maximum capacitance in Farads.
        package: SMD/THT package, e.g. '0402', '0603', '0805'.
        voltage_rating: Voltage as text, e.g. '50V', '100V'.
        voltage_min_v: Minimum voltage rating in Volts, e.g. 50.0.
        dielectric: Dielectric type, e.g. 'X5R', 'X7R', 'C0G', 'NP0', 'Y5V'.
        tolerance: Tolerance as text, e.g. '±10%', '±20%'.
        library_type: 'Basic' or 'Extended', or None for both.
        in_stock: Only return parts with stock > 0. Default: True.
        limit: Maximum results. Default: 20.
        skip_refresh: Skip the pre-search membership/details refresh.

    Returns:
        List of matching capacitors with LCSC code, description, package, price, stock.
    """
    db = _db()
    warning = None if skip_refresh else _ensure_membership_fresh(db)
    response = _safe_search(lambda: db.search_passive(
        component_type="capacitor",
        value=value,
        value_min=value_min_farads,
        value_max=value_max_farads,
        package=package,
        tolerance=tolerance,
        voltage_rating=voltage_rating,
        voltage_min_v=voltage_min_v,
        dielectric=dielectric,
        library_type=library_type,
        in_stock=in_stock,
        limit=limit,
    ))
    if not response["success"]:
        return response
    parts = response["parts"] if skip_refresh else _refreshed_search_results(db, response["parts"])
    result: dict = {"success": True, "count": len(parts), "parts": parts}
    if warning:
        result["warning"] = warning
    return result


# ---------------------------------------------------------------------------
# Tool: search_inductors
# ---------------------------------------------------------------------------

@mcp.tool()
def search_inductors(
    value: Optional[str] = None,
    value_min_henries: Optional[float] = None,
    value_max_henries: Optional[float] = None,
    package: Optional[str] = None,
    current_rating: Optional[str] = None,
    current_min_a: Optional[float] = None,
    tolerance: Optional[str] = None,
    library_type: Optional[str] = None,
    in_stock: bool = True,
    limit: int = 20,
    skip_refresh: bool = False,
) -> dict:
    """
    Search inductors (and ferrite beads) with parametric filters.

    The cache is auto-refreshed (see search_parts for details).

    Args:
        value: Inductance as text, e.g. '10µH', '100nH', '4.7uH'.
        value_min_henries: Minimum inductance in Henries, e.g. 10e-6 for 10 µH.
        value_max_henries: Maximum inductance in Henries.
        package: SMD/THT package, e.g. '0402', '0603', '0805', 'CD43'.
        current_rating: Rated current as text, e.g. '100mA', '1A', '500mA'.
        current_min_a: Minimum current rating in Amperes, e.g. 1.0 for 1 A.
        tolerance: Tolerance as text, e.g. '±10%', '±20%'.
        library_type: 'Basic' or 'Extended', or None for both.
        in_stock: Only return parts with stock > 0. Default: True.
        limit: Maximum results. Default: 20.
        skip_refresh: Skip the pre-search membership/details refresh.

    Returns:
        List of matching inductors with LCSC code, description, package, price, stock.
    """
    db = _db()
    warning = None if skip_refresh else _ensure_membership_fresh(db)
    response = _safe_search(lambda: db.search_passive(
        component_type="inductor",
        value=value,
        value_min=value_min_henries,
        value_max=value_max_henries,
        package=package,
        tolerance=tolerance,
        current_rating=current_rating,
        current_min_a=current_min_a,
        library_type=library_type,
        in_stock=in_stock,
        limit=limit,
    ))
    if not response["success"]:
        return response
    parts = response["parts"] if skip_refresh else _refreshed_search_results(db, response["parts"])
    result: dict = {"success": True, "count": len(parts), "parts": parts}
    if warning:
        result["warning"] = warning
    return result


# ---------------------------------------------------------------------------
# Tool: rebuild_component_specs
# ---------------------------------------------------------------------------

@mcp.tool()
def rebuild_component_specs() -> dict:
    """
    (Re)extract parametric specs for all passives already in the local database.

    Parses every component description to populate the component_specs table
    with structured numeric data (resistance/capacitance/inductance value,
    voltage, current, power, tolerance, dielectric type).

    Run this once after a server upgrade that added the component_specs table
    or improved a parser.

    Returns:
        Number of passive components indexed.
    """
    db = _db()
    try:
        count = db.rebuild_specs()
        return {"success": True, "passives_indexed": count}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool: get_part
# ---------------------------------------------------------------------------

@mcp.tool()
def get_part(lcsc_code: str, live: bool = False) -> dict:
    """
    Get detailed information about a component by its LCSC code.

    Returns the cached row when it exists and is fresher than
    JLCPCB_CACHE_TTL_HOURS (default 6). Otherwise fetches live from the JLCPCB
    API (getComponentDetailByCode) and upserts the result.

    Args:
        lcsc_code: LCSC part number, e.g. 'C25804'.
        live: Force a live API lookup even if the cached row is fresh.

    Returns:
        Component details including description, package, stock, price breaks, datasheet.
    """
    db = _db()

    if not live:
        age = db.part_age_hours(lcsc_code)
        if age is not None and age < CACHE_TTL_HOURS:
            part = db.get(lcsc_code)
            if part:
                return {"success": True, "source": "local_db", "part": part}

    try:
        client = _client()
        raw = client.get_part_detail(lcsc_code, max_retries=INTERACTIVE_MAX_RETRIES)
        if not raw:
            part = db.get(lcsc_code)
            if part:
                return {
                    "success": True,
                    "source": "local_db_stale",
                    "api_error": "API returned no data for this part",
                    "part": part,
                }
            return {"success": False, "error": f"Part {lcsc_code} not found via API"}

        db.import_batch([raw])
        part = db.get(lcsc_code) or raw
        return {"success": True, "source": "api", "part": part}

    except Exception as exc:
        logger.warning("API error for %s: %s", lcsc_code, exc)
        part = db.get(lcsc_code)
        if part:
            return {"success": True, "source": "local_db_stale", "api_error": str(exc), "part": part}
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool: suggest_alternatives
# ---------------------------------------------------------------------------

@mcp.tool()
def suggest_alternatives(lcsc_code: str, limit: int = 5) -> dict:
    """
    Suggest alternative components for a given LCSC part.

    Finds parts in the same category and package, ranked by:
    1. Library type (Basic first — free assembly at JLCPCB)
    2. Unit price (cheapest first)
    3. Stock availability (highest first)

    The reference part and the alternatives are refreshed via the per-row
    detail TTL before being returned.

    Args:
        lcsc_code: Reference LCSC part number, e.g. 'C25804'.
        limit: Maximum number of alternatives. Default: 5.

    Returns:
        List of alternative parts with comparison data.
    """
    db = _db()
    ref = db.get(lcsc_code)
    if not ref:
        return {"success": False, "error": f"Part {lcsc_code} not found in local database"}

    alternatives = db.suggest_alternatives(lcsc_code, limit=limit)

    # Refresh stale rows for the reference and the alternatives so stock/price
    # are current.
    codes = [lcsc_code] + [a["lcsc"] for a in alternatives if a.get("lcsc")]
    _refresh_stale_details(db, codes)
    ref = db.get(lcsc_code) or ref
    alternatives = [db.get(a["lcsc"]) or a for a in alternatives if a.get("lcsc")]

    ref_price = None
    if ref.get("price_breaks"):
        ref_price = ref["price_breaks"][0]["price"]

    return {
        "success": True,
        "reference": {
            "lcsc": lcsc_code,
            "library_type": ref.get("library_type"),
            "price": ref_price,
            "stock": ref.get("stock"),
        },
        "alternatives": alternatives,
    }


# ---------------------------------------------------------------------------
# Tool: get_stats
# ---------------------------------------------------------------------------

@mcp.tool()
def get_stats() -> dict:
    """
    Return statistics about the local LCSC parts database.

    Returns:
        Total parts, basic/extended/preferred counts, in-stock count, DB file info.
    """
    try:
        db = _db()
        return {"success": True, **db.stats()}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool: download_kicad_component
# ---------------------------------------------------------------------------

@mcp.tool()
def download_kicad_component(
    lcsc_code: str,
    output: Optional[str] = None,
    symbol: bool = True,
    footprint: bool = True,
    model_3d: bool = True,
    overwrite: bool = True,
    model_3d_path: Optional[str] = None,
) -> dict:
    """
    Download KiCAD symbol, footprint, and/or 3D model for an LCSC component.

    Uses the public EasyEDA Pro API (no credentials required) to fetch component
    data and convert it to KiCAD format.

    Args:
        lcsc_code: LCSC part number, e.g. 'C25117'.
        output: Base path for the library files, without extension.
                Example: '/path/to/hardware/libs/easyeda/EasyEDA'
                  → symbol  : {output}.kicad_sym
                  → footprint: {output}.pretty/{name}.kicad_mod
                  → 3D model : {output}.3dshapes/{name}.wrl/.step
                Defaults to JLCPCB_EASYEDA_LIB_PATH env var, then './EasyEDA'.
        symbol: Download and add symbol to .kicad_sym library. Default: True.
        footprint: Download and add footprint to .pretty directory. Default: True.
        model_3d: Download 3D model (.wrl/.step) to .3dshapes directory. Default: True.
        overwrite: Replace the component if it already exists. Default: True.
        model_3d_path: Path embedded in the footprint for 3D model references.
                       Supports KiCAD variables, e.g. '${KICAD_3RD_PARTY}/EasyEDA.3dshapes'.
                       Defaults to JLCPCB_EASYEDA_3D_PATH env var, then
                       '${KICAD_3RD_PARTY}/EasyEDA.3dshapes'.

    Returns:
        dict with success status, component name, and created file paths.
    """
    try:
        from easyeda2kicad.easyeda.easyeda_api import EasyedaApi
        from easyeda2kicad.easyeda.easyeda_importer import (
            Easyeda3dModelImporter,
            EasyedaFootprintImporter,
            EasyedaSymbolImporter,
        )
        from easyeda2kicad.kicad.export_kicad_3d_model import Exporter3dModelKicad
        from easyeda2kicad.kicad.export_kicad_footprint import ExporterFootprintKicad
        from easyeda2kicad.kicad.export_kicad_symbol import ExporterSymbolKicad
        from easyeda2kicad.kicad.parameters_kicad_symbol import KicadVersion
        from easyeda2kicad.__main__ import (
            add_component_in_symbol_lib_file,
            fp_already_in_footprint_lib,
            id_already_in_symbol_lib,
            update_component_in_symbol_lib_file,
        )
    except ImportError as exc:
        return {"success": False, "error": f"easyeda2kicad not installed: {exc}"}

    lib_base = output or os.getenv("JLCPCB_EASYEDA_LIB_PATH") or "./EasyEDA"
    lib_base = lib_base.rstrip("/")

    embedded_3d_path = (
        model_3d_path
        or os.getenv("JLCPCB_EASYEDA_3D_PATH")
        or "${KICAD_3RD_PARTY}/EasyEDA.3dshapes"
    )

    # EasyEDA blocks requests with the default "easyeda2kicad" User-Agent;
    # override with a generic browser UA to avoid getting an empty HTML response.
    api = EasyedaApi()
    api.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    cad_data = api.get_cad_data_of_component(lcsc_id=lcsc_code)
    if not cad_data:
        return {"success": False, "error": f"No EasyEDA data found for {lcsc_code}"}

    result: dict = {"success": True, "lcsc_code": lcsc_code, "files": {}}

    if symbol:
        try:
            sym_importer = EasyedaSymbolImporter(easyeda_cp_cad_data=cad_data)
            ee_symbol = sym_importer.get_symbol()
            sym_lib_path = f"{lib_base}.kicad_sym"
            footprint_lib_name = Path(lib_base).name

            sym_lib_file = Path(sym_lib_path)
            sym_lib_file.parent.mkdir(parents=True, exist_ok=True)
            if not sym_lib_file.exists():
                sym_lib_file.write_text(
                    "(kicad_symbol_lib (version 20220914) (generator kicad_symbol_editor)\n)\n",
                    encoding="utf-8",
                )

            already_in_lib = id_already_in_symbol_lib(
                lib_path=sym_lib_path,
                component_name=ee_symbol.info.name,
                kicad_version=KicadVersion.v6,
            )

            exporter = ExporterSymbolKicad(
                symbol=ee_symbol, kicad_version=KicadVersion.v6
            )
            kicad_symbol_lib = exporter.export(footprint_lib_name=footprint_lib_name)

            if already_in_lib:
                if not overwrite:
                    result["files"]["symbol"] = {"skipped": True, "path": sym_lib_path}
                else:
                    update_component_in_symbol_lib_file(
                        lib_path=sym_lib_path,
                        component_name=ee_symbol.info.name,
                        component_content=kicad_symbol_lib,
                        kicad_version=KicadVersion.v6,
                    )
                    result["files"]["symbol"] = {"updated": True, "path": sym_lib_path, "name": ee_symbol.info.name}
            else:
                add_component_in_symbol_lib_file(
                    lib_path=sym_lib_path,
                    component_content=kicad_symbol_lib,
                    kicad_version=KicadVersion.v6,
                )
                result["files"]["symbol"] = {"created": True, "path": sym_lib_path, "name": ee_symbol.info.name}

        except Exception as exc:
            result["files"]["symbol"] = {"error": str(exc)}

    if footprint:
        try:
            fp_importer = EasyedaFootprintImporter(easyeda_cp_cad_data=cad_data)
            ee_footprint = fp_importer.get_footprint()
            fp_dir = f"{lib_base}.pretty"
            fp_filename = f"{ee_footprint.info.name}.kicad_mod"
            fp_full_path = f"{fp_dir}/{fp_filename}"

            already_in_lib = fp_already_in_footprint_lib(
                lib_path=fp_dir,
                package_name=ee_footprint.info.name,
            )

            if already_in_lib and not overwrite:
                result["files"]["footprint"] = {"skipped": True, "path": fp_full_path}
            else:
                Path(fp_dir).mkdir(parents=True, exist_ok=True)
                ki_footprint = ExporterFootprintKicad(footprint=ee_footprint)
                ki_footprint.export(
                    footprint_full_path=fp_full_path,
                    model_3d_path=embedded_3d_path,
                )
                action = "updated" if already_in_lib else "created"
                result["files"]["footprint"] = {action: True, "path": fp_full_path, "name": ee_footprint.info.name}

        except Exception as exc:
            result["files"]["footprint"] = {"error": str(exc)}

    if model_3d:
        try:
            model_importer = Easyeda3dModelImporter(
                easyeda_cp_cad_data=cad_data, download_raw_3d_model=True
            )
            model_exporter = Exporter3dModelKicad(model_3d=model_importer.output)
            Path(f"{lib_base}.3dshapes").mkdir(parents=True, exist_ok=True)
            model_exporter.export(lib_path=lib_base)

            shapes_dir = f"{lib_base}.3dshapes"
            files_created = []
            if model_exporter.output:
                files_created.append(f"{shapes_dir}/{model_exporter.output.name}.wrl")
            if model_exporter.output_step:
                files_created.append(f"{shapes_dir}/{model_exporter.output.name}.step")

            if files_created:
                result["files"]["model_3d"] = {"created": True, "paths": files_created}
            else:
                result["files"]["model_3d"] = {"skipped": True, "reason": "no 3D model available"}

        except Exception as exc:
            result["files"]["model_3d"] = {"error": str(exc)}

    return result


# ---------------------------------------------------------------------------
# Tool: bom_check
# ---------------------------------------------------------------------------

def _unit_price_at_qty(price_breaks: list[dict], qty: int) -> Optional[float]:
    """Return the unit price for the largest break whose qty <= the requested qty.

    JLCPCB price breaks are sorted ascending by qty. Below the smallest break
    quantity, fall back to that smallest break's price (typical real-world cost).
    """
    if not price_breaks:
        return None
    candidate = price_breaks[0].get("price")
    for br in price_breaks:
        if br.get("qty", 0) <= qty:
            candidate = br.get("price", candidate)
        else:
            break
    return candidate


def _check_one_bom_line(
    db: PartsDB,
    lcsc: str,
    qty: int,
    suggest: bool,
) -> dict:
    """Build one row of a bom_check report."""
    part = db.get(lcsc)
    if not part:
        return {
            "lcsc": lcsc,
            "qty": qty,
            "found": False,
            "warnings": ["not in local library — may not be assemblable at JLCPCB"],
        }

    unit = _unit_price_at_qty(part.get("price_breaks") or [], qty)
    line_total = round(unit * qty, 4) if unit is not None else None
    stock = part.get("stock") or 0

    warnings: list[str] = []
    if stock <= 0:
        warnings.append("out of stock")
    elif stock < qty:
        warnings.append(f"insufficient stock ({stock} < {qty})")

    row = {
        "lcsc": lcsc,
        "qty": qty,
        "found": True,
        "mfr_part": part.get("mfr_part"),
        "package": part.get("package"),
        "library_type": part.get("library_type"),
        "stock": stock,
        "unit_price": unit,
        "line_total": line_total,
        "description": part.get("description"),
    }

    if suggest and part.get("library_type") == "Extended":
        try:
            alts = db.suggest_alternatives(lcsc, limit=3)
        except sqlite3.DatabaseError:
            alts = []
        cheaper_basic = None
        for a in alts:
            if a.get("library_type") != "Basic":
                continue
            a_unit = _unit_price_at_qty(a.get("price_breaks") or [], qty)
            if a_unit is None:
                continue
            if unit is None or a_unit < unit:
                cheaper_basic = {
                    "lcsc": a.get("lcsc"),
                    "mfr_part": a.get("mfr_part"),
                    "library_type": "Basic",
                    "unit_price": a_unit,
                    "line_total": round(a_unit * qty, 4),
                    "savings_per_unit": (
                        round(unit - a_unit, 4) if unit is not None else None
                    ),
                }
                break
        if cheaper_basic:
            row["suggested_basic_alternative"] = cheaper_basic

    if warnings:
        row["warnings"] = warnings
    return row


@mcp.tool()
def bom_check(
    items: list,
    qty: int = 1,
    suggest_alternatives: bool = True,
    skip_refresh: bool = False,
) -> dict:
    """Check a Bill of Materials against the JLCPCB assembly library.

    Reports per-code: stock, library type (Basic/Extended), unit price at the
    requested quantity, line total, and (for Extended parts) a cheaper Basic
    drop-in alternative when one is available. Aggregates totals and flags
    out-of-stock or unknown parts.

    Args:
        items: List of LCSC codes (e.g. ['C25804', 'C14663', ...]) OR list of
            dicts {'lcsc': 'C25804', 'qty': 10}. Strings get the global qty.
        qty: Default per-line quantity when items are bare strings. Default: 1.
        suggest_alternatives: When True, propose a cheaper Basic alternative
            for each Extended part (slower; one query per Extended line).
        skip_refresh: Skip the pre-check details refresh. Use for fast checks
            when stale stock/price are acceptable.

    Returns:
        Per-line report + aggregates (basic_count, extended_count, not_found,
        total_cost, out_of_stock).
    """
    db = _db()

    parsed: list[tuple[str, int]] = []
    for item in items:
        if isinstance(item, str):
            parsed.append((item, qty))
        elif isinstance(item, dict) and item.get("lcsc"):
            parsed.append((item["lcsc"], int(item.get("qty", qty))))

    if not parsed:
        return {"success": False, "error": "items list is empty or malformed"}

    if not skip_refresh:
        _refresh_stale_details(db, [code for code, _ in parsed])

    try:
        rows = [_check_one_bom_line(db, code, q, suggest_alternatives) for code, q in parsed]
    except sqlite3.DatabaseError as exc:
        return _db_error_response(exc)

    found = [r for r in rows if r.get("found")]
    not_found = [r["lcsc"] for r in rows if not r.get("found")]
    out_of_stock = [r["lcsc"] for r in found if (r.get("stock") or 0) <= 0]
    insufficient_stock = [
        r["lcsc"] for r in found
        if 0 < (r.get("stock") or 0) < r["qty"]
    ]
    basic = [r for r in found if r.get("library_type") == "Basic"]
    extended = [r for r in found if r.get("library_type") == "Extended"]
    total_cost = round(
        sum(r["line_total"] for r in found if r.get("line_total") is not None),
        4,
    )
    potential_savings = round(
        sum(
            r["suggested_basic_alternative"]["savings_per_unit"] * r["qty"]
            for r in extended
            if r.get("suggested_basic_alternative")
            and r["suggested_basic_alternative"].get("savings_per_unit") is not None
        ),
        4,
    )

    return {
        "success": True,
        "lines": rows,
        "summary": {
            "total_lines": len(rows),
            "found": len(found),
            "not_found": not_found,
            "basic_count": len(basic),
            "extended_count": len(extended),
            "out_of_stock": out_of_stock,
            "insufficient_stock": insufficient_stock,
            "total_cost": total_cost,
            "potential_savings_with_basic_alternatives": potential_savings,
        },
    }


# ---------------------------------------------------------------------------
# Tool: kicad_bom_check
# ---------------------------------------------------------------------------

# Match KiCAD `(property "LCSC Part" "C25804" ...)` lines.
_LCSC_PROP_RE = re.compile(
    r'\(property\s+"(?:LCSC(?:\s+Part)?(?:\s*#)?|JLC\s*PCB)"\s+"([^"]*)"',
    re.IGNORECASE,
)


def _extract_lcsc_codes_from_kicad(text: str) -> list[str]:
    codes: list[str] = []
    for m in _LCSC_PROP_RE.finditer(text):
        code = m.group(1).strip()
        if code:
            codes.append(code)
    return codes


@mcp.tool()
def kicad_bom_check(
    sch_path: str,
    qty: int = 1,
    suggest_alternatives: bool = True,
    skip_refresh: bool = False,
) -> dict:
    """Parse a KiCAD schematic and run bom_check on every LCSC Part property.

    Counts duplicate LCSC codes (one row per code with qty = occurrences × qty).

    Args:
        sch_path: Absolute path to a `.kicad_sch` file (or any text file with
            `(property "LCSC Part" "C…")` entries — also matches 'LCSC#' and
            'JLCPCB' field name variants).
        qty: Multiplier applied to each unique part's occurrence count
            (e.g. qty=10 for a panel of 10 boards). Default: 1.
        suggest_alternatives: See bom_check.
        skip_refresh: See bom_check.

    Returns:
        bom_check output, plus 'sch_path', 'unique_codes', 'total_components'.
    """
    path = Path(sch_path).expanduser()
    if not path.exists():
        return {"success": False, "error": f"file not found: {path}"}

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"success": False, "error": f"cannot read {path}: {exc}"}

    codes = _extract_lcsc_codes_from_kicad(text)
    if not codes:
        return {
            "success": False,
            "error": f"no LCSC Part properties found in {path}",
            "hint": "ensure each symbol has an `LCSC Part` field set in KiCAD",
        }

    counts: dict[str, int] = {}
    for c in codes:
        counts[c] = counts.get(c, 0) + 1

    items = [{"lcsc": c, "qty": n * qty} for c, n in counts.items()]
    result = bom_check(
        items=items,
        suggest_alternatives=suggest_alternatives,
        skip_refresh=skip_refresh,
    )
    if isinstance(result, dict) and result.get("success"):
        result["sch_path"] = str(path)
        result["unique_codes"] = len(counts)
        result["total_components"] = sum(counts.values())
    return result


# ---------------------------------------------------------------------------
# Tool: repair_db
# ---------------------------------------------------------------------------

@mcp.tool()
def repair_db() -> dict:
    """Run integrity checks and rebuild the FTS index.

    Useful after upgrading the server, after an interrupted import, or when
    a search returns the 'database disk image is malformed' / FTS hint.

    Steps:
      1. PRAGMA integrity_check
      2. FTS5 'integrity-check'
      3. FTS5 'rebuild' (always runs — cheap relative to a full re-download)
      4. Reports the result.
    """
    db = _db()
    out: dict = {"success": True, "steps": []}
    try:
        ic = list(db._conn.execute("PRAGMA integrity_check"))
        out["steps"].append({"integrity_check": [r[0] for r in ic]})
    except sqlite3.DatabaseError as exc:
        return _db_error_response(exc)

    try:
        db._conn.execute("INSERT INTO components_fts(components_fts) VALUES('integrity-check')")
        out["steps"].append({"fts_integrity_check": "ok"})
    except sqlite3.DatabaseError as exc:
        out["steps"].append({"fts_integrity_check": str(exc)})

    try:
        db.rebuild_fts()
        out["steps"].append({"fts_rebuild": "ok"})
    except sqlite3.DatabaseError as exc:
        return _db_error_response(exc)

    out["stats"] = db.stats()
    return out


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
