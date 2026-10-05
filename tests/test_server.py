"""Tests for jlcpcb_mcp/server.py — MCP tools."""
import sys

import pytest

import jlcpcb_mcp.server as server


# ---------------------------------------------------------------------------
# _db / _client factories
# ---------------------------------------------------------------------------

def test_db_default_path(monkeypatch):
    monkeypatch.delenv("JLCPCB_DB_PATH", raising=False)
    db = server._db()
    assert "lcsc_parts" in str(db.path)
    db.close()


def test_db_custom_path(monkeypatch, tmp_path):
    custom = str(tmp_path / "custom.db")
    monkeypatch.setenv("JLCPCB_DB_PATH", custom)
    db = server._db()
    assert str(db.path) == custom
    db.close()


def test_client_missing_env(monkeypatch):
    for var in ("JLCPCB_APP_ID", "JLCPCB_API_KEY", "JLCPCB_API_SECRET"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(EnvironmentError):
        server._client()


# ---------------------------------------------------------------------------
# _ensure_membership_fresh
# ---------------------------------------------------------------------------

def test_membership_fresh_returns_none(mocker):
    """Cache age within TTL → no API call, no warning."""
    mock_db = mocker.MagicMock()
    mock_db.library_age_hours.return_value = 1.0  # well under MEMBERSHIP_TTL_HOURS
    mock_client = mocker.patch("jlcpcb_mcp.server._client")
    result = server._ensure_membership_fresh(mock_db)
    assert result is None
    mock_client.assert_not_called()


def test_membership_never_populated_triggers_refresh(mocker):
    """age=None → first-run refresh. New stubs are imported, FTS rebuilt."""
    mock_db = mocker.MagicMock()
    mock_db.library_age_hours.return_value = None
    mock_db.get_all_lcsc_codes.return_value = set()
    mock_db.delete_codes_not_in.return_value = 0

    mock_client = mocker.MagicMock()
    mock_client.iter_library_stubs.return_value = iter([
        {"componentCode": "C1"},
        {"componentCode": "C2"},
    ])
    mock_client.get_parts_details.return_value = [
        {"lcscPart": "C1"},
        {"lcscPart": "C2"},
    ]
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server._ensure_membership_fresh(mock_db)
    assert result is not None
    assert "+2 new" in result
    mock_client.get_parts_details.assert_called_once_with(["C1", "C2"])
    mock_db.import_batch.assert_called_once()
    mock_db.rebuild_fts.assert_called_once()
    keys = [c.args[0] for c in mock_db.set_metadata.call_args_list]
    assert keys == ["basic_library_refreshed_at", server.LIBRARY_TOTAL_KEY]
    assert mock_db.set_metadata.call_args_list[1].args[1] == "2"


def test_stub_progress_logger_first_load(mocker, caplog):
    """No stored total → pages, codes and elapsed time only, every 50 pages."""
    mock_db = mocker.MagicMock()
    mock_db.get_metadata.return_value = None
    cb = server._stub_progress_logger(mock_db, started_at=0)
    mocker.patch("jlcpcb_mcp.server.time.time", return_value=120)
    with caplog.at_level("INFO", logger="jlcpcb_mcp.server"):
        for page in range(1, 101):
            cb(page * 1000)
    assert [r.getMessage() for r in caplog.records] == [
        "Library list: page 50, 50000 codes, 2.0 min elapsed",
        "Library list: page 100, 100000 codes, 2.0 min elapsed",
    ]


def test_stub_progress_logger_with_estimate(mocker, caplog):
    """A stored total from the last refresh gives an estimated page count."""
    mock_db = mocker.MagicMock()
    mock_db.get_metadata.return_value = "120500"
    cb = server._stub_progress_logger(mock_db, started_at=0)
    mocker.patch("jlcpcb_mcp.server.time.time", return_value=60)
    with caplog.at_level("INFO", logger="jlcpcb_mcp.server"):
        for page in range(1, 51):
            cb(page * 1000)
    mock_db.get_metadata.assert_called_once_with(server.LIBRARY_TOTAL_KEY)
    assert [r.getMessage() for r in caplog.records] == [
        "Library list: page 50 / ~121 (est.), 50000 codes, 1.0 min elapsed",
    ]


def test_stub_progress_logger_empty_first_page(mocker, caplog):
    """A first page reporting 0 codes must not divide by zero later."""
    mock_db = mocker.MagicMock()
    mock_db.get_metadata.return_value = "1000"
    cb = server._stub_progress_logger(mock_db, started_at=0)
    with caplog.at_level("INFO", logger="jlcpcb_mcp.server"):
        for _ in range(50):
            cb(0)
    assert "/ ~1000 (est.)" in caplog.records[-1].getMessage()


def test_membership_stale_with_age_in_message(mocker):
    """When age is known, the warning includes how old the cache was."""
    mock_db = mocker.MagicMock()
    mock_db.library_age_hours.return_value = 30.0
    mock_db.get_all_lcsc_codes.return_value = {"C1"}
    mock_db.delete_codes_not_in.return_value = 0

    mock_client = mocker.MagicMock()
    mock_client.iter_library_stubs.return_value = iter([{"componentCode": "C1"}])
    mock_client.get_parts_details.return_value = []
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server._ensure_membership_fresh(mock_db)
    assert "30.0h old" in result


def test_membership_diff_detects_new_and_removed(mocker):
    """new = api - local (enriched); removed = local - api (deleted by db helper)."""
    mock_db = mocker.MagicMock()
    mock_db.library_age_hours.return_value = 100.0
    mock_db.get_all_lcsc_codes.return_value = {"C1", "C2", "C_removed"}
    mock_db.delete_codes_not_in.return_value = 1  # C_removed dropped

    mock_client = mocker.MagicMock()
    mock_client.iter_library_stubs.return_value = iter([
        {"componentCode": "C1"},
        {"componentCode": "C2"},
        {"componentCode": "C_new"},
    ])
    mock_client.get_parts_details.return_value = [{"lcscPart": "C_new"}]
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server._ensure_membership_fresh(mock_db)
    assert result is not None
    assert "+1 new" in result
    assert "-1 removed" in result
    mock_client.get_parts_details.assert_called_once_with(["C_new"])
    mock_db.delete_codes_not_in.assert_called_once_with({"C1", "C2", "C_new"})


def test_membership_no_new_codes_skips_enrichment(mocker):
    """When the diff finds no new codes, get_parts_details/import_batch are skipped."""
    mock_db = mocker.MagicMock()
    mock_db.library_age_hours.return_value = 100.0
    mock_db.get_all_lcsc_codes.return_value = {"C1"}
    mock_db.delete_codes_not_in.return_value = 0

    mock_client = mocker.MagicMock()
    mock_client.iter_library_stubs.return_value = iter([{"componentCode": "C1"}])
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server._ensure_membership_fresh(mock_db)
    assert "+0 new" in result
    mock_client.get_parts_details.assert_not_called()
    mock_db.import_batch.assert_not_called()
    mock_db.rebuild_fts.assert_called_once()


def test_membership_skips_stubs_without_code(mocker):
    """Defensive: stubs missing componentCode are filtered out."""
    mock_db = mocker.MagicMock()
    mock_db.library_age_hours.return_value = None
    mock_db.get_all_lcsc_codes.return_value = set()
    mock_db.delete_codes_not_in.return_value = 0

    mock_client = mocker.MagicMock()
    mock_client.iter_library_stubs.return_value = iter([
        {"componentCode": "C1"},
        {"componentModel": "no_code"},
        {"componentCode": ""},
    ])
    mock_client.get_parts_details.return_value = [{"lcscPart": "C1"}]
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server._ensure_membership_fresh(mock_db)
    assert "+1 new" in result


def test_membership_empty_api_returns_skipped_message(mocker):
    """If the API returns no stubs at all, the cache is left untouched."""
    mock_db = mocker.MagicMock()
    mock_db.library_age_hours.return_value = None
    mock_client = mocker.MagicMock()
    mock_client.iter_library_stubs.return_value = iter([])
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server._ensure_membership_fresh(mock_db)
    assert "skipped" in result.lower()
    mock_db.delete_codes_not_in.assert_not_called()
    mock_db.set_metadata.assert_not_called()


def test_membership_client_error_returns_failure_message(mocker):
    mock_db = mocker.MagicMock()
    mock_db.library_age_hours.return_value = 30.0
    mocker.patch("jlcpcb_mcp.server._client", side_effect=EnvironmentError("no creds"))
    result = server._ensure_membership_fresh(mock_db)
    assert "Library refresh failed" in result


def test_membership_iter_error_returns_failure_message(mocker):
    """Mid-stream error during pagination is caught and reported."""
    mock_db = mocker.MagicMock()
    mock_db.library_age_hours.return_value = 30.0
    mock_client = mocker.MagicMock()

    def _boom():
        raise RuntimeError("network")
        yield  # pragma: no cover — make this a generator

    mock_client.iter_library_stubs.side_effect = lambda: _boom()
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server._ensure_membership_fresh(mock_db)
    assert "Library refresh failed" in result


# ---------------------------------------------------------------------------
# _refresh_stale_details
# ---------------------------------------------------------------------------

def test_refresh_stale_details_empty_codes(mocker):
    """db.stale_codes([], …) returns [] → no API call."""
    mock_db = mocker.MagicMock()
    mock_db.stale_codes.return_value = []
    mock_client = mocker.patch("jlcpcb_mcp.server._client")
    assert server._refresh_stale_details(mock_db, []) == 0
    mock_client.assert_not_called()


def test_refresh_stale_details_all_fresh(mocker):
    """No stale codes → no API call."""
    mock_db = mocker.MagicMock()
    mock_db.stale_codes.return_value = []
    mock_client = mocker.patch("jlcpcb_mcp.server._client")
    assert server._refresh_stale_details(mock_db, ["C1", "C2"]) == 0
    mock_client.assert_not_called()


def test_refresh_stale_details_refetches_and_upserts(mocker):
    mock_db = mocker.MagicMock()
    mock_db.stale_codes.return_value = ["C1", "C2"]
    mock_client = mocker.MagicMock()
    mock_client.get_parts_details.return_value = [{"lcscPart": "C1"}, {"lcscPart": "C2"}]
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)
    assert server._refresh_stale_details(mock_db, ["C1", "C2"]) == 2
    mock_client.get_parts_details.assert_called_once_with(
        ["C1", "C2"], max_retries=server.INTERACTIVE_MAX_RETRIES
    )
    mock_db.import_batch.assert_called_once()


def test_refresh_stale_details_uses_default_ttl(mocker):
    """ttl_hours=None → uses module CACHE_TTL_HOURS."""
    mock_db = mocker.MagicMock()
    mock_db.stale_codes.return_value = []
    mocker.patch("jlcpcb_mcp.server._client")
    server._refresh_stale_details(mock_db, ["C1"])
    args, kwargs = mock_db.stale_codes.call_args
    assert args[1] == server.CACHE_TTL_HOURS or kwargs.get("ttl_hours") == server.CACHE_TTL_HOURS


def test_refresh_stale_details_uses_explicit_ttl(mocker):
    mock_db = mocker.MagicMock()
    mock_db.stale_codes.return_value = []
    mocker.patch("jlcpcb_mcp.server._client")
    server._refresh_stale_details(mock_db, ["C1"], ttl_hours=999.0)
    args, _ = mock_db.stale_codes.call_args
    assert args[1] == 999.0


def test_refresh_stale_details_swallows_api_errors(mocker):
    """API errors don't propagate — search results stay usable with stale data."""
    mock_db = mocker.MagicMock()
    mock_db.stale_codes.return_value = ["C1"]
    mock_client = mocker.MagicMock()
    mock_client.get_parts_details.side_effect = RuntimeError("network down")
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)
    assert server._refresh_stale_details(mock_db, ["C1"]) == 0


# ---------------------------------------------------------------------------
# _refreshed_search_results
# ---------------------------------------------------------------------------

def test_refreshed_search_results_empty_input(mocker):
    """Empty input → returned as-is, no refresh attempted."""
    mock_db = mocker.MagicMock()
    spy = mocker.patch("jlcpcb_mcp.server._refresh_stale_details")
    assert server._refreshed_search_results(mock_db, []) == []
    spy.assert_not_called()


def test_refreshed_search_results_re_reads_from_db(mocker):
    """After refresh, rows are re-read so the caller sees the latest data."""
    mock_db = mocker.MagicMock()
    mock_db.get.side_effect = [{"lcsc": "C1", "stock": 50}, {"lcsc": "C2", "stock": 100}]
    mocker.patch("jlcpcb_mcp.server._refresh_stale_details")
    result = server._refreshed_search_results(mock_db, [
        {"lcsc": "C1", "stock": 0},
        {"lcsc": "C2", "stock": 0},
    ])
    assert [r["stock"] for r in result] == [50, 100]


def test_refreshed_search_results_drops_missing_rows(mocker):
    """If a refresh removes a row entirely, it's dropped from the results."""
    mock_db = mocker.MagicMock()
    mock_db.get.side_effect = [None, {"lcsc": "C2"}]
    mocker.patch("jlcpcb_mcp.server._refresh_stale_details")
    result = server._refreshed_search_results(mock_db, [
        {"lcsc": "C1"},
        {"lcsc": "C2"},
    ])
    assert [r["lcsc"] for r in result] == ["C2"]


def test_refreshed_search_results_skips_rows_without_lcsc(mocker):
    """Defensive: rows with no lcsc are filtered out."""
    mock_db = mocker.MagicMock()
    mocker.patch("jlcpcb_mcp.server._refresh_stale_details")
    result = server._refreshed_search_results(mock_db, [{"description": "broken"}])
    assert result == []


# ---------------------------------------------------------------------------
# download_library
# ---------------------------------------------------------------------------

def test_download_library_success(mocker):
    mock_db = mocker.MagicMock()
    mock_db.stats.return_value = {"total": 5}
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mock_client = mocker.MagicMock()
    mock_client.download_library.return_value = None
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server.download_library()
    assert result["success"] is True
    keys = [c.args[0] for c in mock_db.set_metadata.call_args_list]
    assert keys == ["basic_library_refreshed_at", server.LIBRARY_TOTAL_KEY]
    mock_db.rebuild_fts.assert_called_once()
    mock_db.rebuild_specs.assert_called_once()


def test_download_library_streams_batches(mocker):
    """on_batch callback is wired through to db.import_batch."""
    mock_db = mocker.MagicMock()
    mock_db.import_batch.return_value = 2
    mock_db.stats.return_value = {"total": 2}
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)

    mock_client = mocker.MagicMock()

    def fake_download_library(on_batch=None, on_progress=None):
        if on_batch:
            on_batch([{"lcscPart": "C1"}, {"lcscPart": "C2"}])

    mock_client.download_library.side_effect = fake_download_library
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server.download_library()
    assert result["success"] is True
    assert "2 parts imported" in result["message"]
    mock_db.import_batch.assert_called_once()


def test_download_library_error(mocker):
    mocker.patch("jlcpcb_mcp.server._db", return_value=mocker.MagicMock())
    mock_client = mocker.MagicMock()
    mock_client.download_library.side_effect = RuntimeError("API down")
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server.download_library()
    assert result["success"] is False
    assert "API down" in result["error"]


# ---------------------------------------------------------------------------
# search_parts
# ---------------------------------------------------------------------------

def test_search_parts_success_no_warning(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search.return_value = [{"lcsc": "C1"}]
    mock_db.get.return_value = {"lcsc": "C1", "stock": 100}
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh", return_value=None)
    mocker.patch("jlcpcb_mcp.server._refresh_stale_details")

    result = server.search_parts(query="resistor")
    assert result["success"] is True
    assert result["count"] == 1
    assert result["parts"][0]["stock"] == 100
    assert "warning" not in result


def test_search_parts_propagates_membership_warning(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search.return_value = []
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh", return_value="refreshed: +5 new")

    result = server.search_parts()
    assert result["warning"] == "refreshed: +5 new"
    assert result["count"] == 0


# ---------------------------------------------------------------------------
# search_resistors
# ---------------------------------------------------------------------------

def test_search_resistors_success(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search_passive.return_value = [{"lcsc": "C25744"}]
    mock_db.get.return_value = {"lcsc": "C25744"}
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh", return_value=None)
    mocker.patch("jlcpcb_mcp.server._refresh_stale_details")

    result = server.search_resistors(value="10k", package="0402", library_type="Basic")
    assert result["success"] is True
    assert result["count"] == 1
    assert "warning" not in result


def test_search_resistors_with_warning(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search_passive.return_value = []
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh", return_value="stale → refreshed")

    result = server.search_resistors()
    assert result["warning"] == "stale → refreshed"


def test_search_resistors_passes_filters_through(mocker):
    """Verify all parametric filters are forwarded to db.search_passive."""
    mock_db = mocker.MagicMock()
    mock_db.search_passive.return_value = []
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh", return_value=None)

    server.search_resistors(
        value="10k", value_min_ohms=9000.0, value_max_ohms=11000.0,
        package="0402", tolerance="±1%", tolerance_max_pct=1.0,
        power_rating="1/16W", power_min_w=0.0625,
        library_type="Basic", in_stock=False, limit=50,
    )
    kwargs = mock_db.search_passive.call_args.kwargs
    assert kwargs["component_type"] == "resistor"
    assert kwargs["value_min"] == 9000.0
    assert kwargs["value_max"] == 11000.0
    assert kwargs["limit"] == 50


# ---------------------------------------------------------------------------
# search_capacitors
# ---------------------------------------------------------------------------

def test_search_capacitors_success(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search_passive.return_value = [{"lcsc": "C1525"}]
    mock_db.get.return_value = {"lcsc": "C1525"}
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh", return_value=None)
    mocker.patch("jlcpcb_mcp.server._refresh_stale_details")

    result = server.search_capacitors(value="100nF", dielectric="X7R")
    assert result["success"] is True
    assert "warning" not in result


def test_search_capacitors_with_warning(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search_passive.return_value = []
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh", return_value="stale")

    result = server.search_capacitors()
    assert result["warning"] == "stale"


def test_search_capacitors_passes_filters_through(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search_passive.return_value = []
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh", return_value=None)

    server.search_capacitors(
        value="100nF", value_min_farads=90e-9, value_max_farads=110e-9,
        package="0402", voltage_rating="50V", voltage_min_v=50.0,
        dielectric="X7R", tolerance="±10%", library_type="Basic",
        in_stock=False, limit=15,
    )
    kwargs = mock_db.search_passive.call_args.kwargs
    assert kwargs["component_type"] == "capacitor"
    assert kwargs["dielectric"] == "X7R"
    assert kwargs["voltage_min_v"] == 50.0


# ---------------------------------------------------------------------------
# search_inductors
# ---------------------------------------------------------------------------

def test_search_inductors_success(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search_passive.return_value = [{"lcsc": "C1044"}]
    mock_db.get.return_value = {"lcsc": "C1044"}
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh", return_value=None)
    mocker.patch("jlcpcb_mcp.server._refresh_stale_details")

    result = server.search_inductors(value="10nH")
    assert result["success"] is True
    assert "warning" not in result


def test_search_inductors_with_warning(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search_passive.return_value = []
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh", return_value="stale")

    result = server.search_inductors()
    assert result["warning"] == "stale"


def test_search_inductors_passes_filters_through(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search_passive.return_value = []
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh", return_value=None)

    server.search_inductors(
        value="10µH", value_min_henries=9e-6, value_max_henries=11e-6,
        package="0805", current_rating="1A", current_min_a=1.0,
        tolerance="±20%", library_type="Extended", in_stock=False, limit=8,
    )
    kwargs = mock_db.search_passive.call_args.kwargs
    assert kwargs["component_type"] == "inductor"
    assert kwargs["current_min_a"] == 1.0


# ---------------------------------------------------------------------------
# rebuild_component_specs
# ---------------------------------------------------------------------------

def test_rebuild_specs_success(mocker):
    mock_db = mocker.MagicMock()
    mock_db.rebuild_specs.return_value = 42
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    result = server.rebuild_component_specs()
    assert result == {"success": True, "passives_indexed": 42}


def test_rebuild_specs_error(mocker):
    mock_db = mocker.MagicMock()
    mock_db.rebuild_specs.side_effect = RuntimeError("DB locked")
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    result = server.rebuild_component_specs()
    assert result["success"] is False
    assert "DB locked" in result["error"]


# ---------------------------------------------------------------------------
# get_part
# ---------------------------------------------------------------------------

def test_get_part_local_fresh(mocker):
    mock_db = mocker.MagicMock()
    mock_db.part_age_hours.return_value = 1.0  # fresh
    mock_db.get.return_value = {"lcsc": "C25744"}
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)

    result = server.get_part("C25744", live=False)
    assert result["success"] is True
    assert result["source"] == "local_db"


def test_get_part_local_fresh_no_part_falls_through(mocker):
    """Age < TTL but get() returns None → falls through to API."""
    mock_db = mocker.MagicMock()
    mock_db.part_age_hours.return_value = 1.0
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)

    mock_client = mocker.MagicMock()
    mock_client.get_part_detail.return_value = {"lcscPart": "C25744"}
    mock_db.get.side_effect = [None, {"lcsc": "C25744"}]
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server.get_part("C25744", live=False)
    assert result["source"] == "api"
    mock_client.get_part_detail.assert_called_once_with(
        "C25744", max_retries=server.INTERACTIVE_MAX_RETRIES
    )


def test_get_part_stale_uses_api(mocker):
    mock_db = mocker.MagicMock()
    mock_db.part_age_hours.return_value = 25.0  # stale
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)

    mock_client = mocker.MagicMock()
    mock_client.get_part_detail.return_value = {"lcscPart": "C25744"}
    mock_db.get.return_value = {"lcsc": "C25744"}
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server.get_part("C25744")
    assert result["source"] == "api"


def test_get_part_not_in_db(mocker):
    mock_db = mocker.MagicMock()
    mock_db.part_age_hours.return_value = None  # not in DB
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)

    mock_client = mocker.MagicMock()
    mock_client.get_part_detail.return_value = {"lcscPart": "C25744"}
    mock_db.get.return_value = {"lcsc": "C25744"}
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server.get_part("C25744")
    assert result["source"] == "api"


def test_get_part_api_returns_none_falls_back_to_stale_cache(mocker):
    mock_db = mocker.MagicMock()
    mock_db.part_age_hours.return_value = None
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)

    mock_client = mocker.MagicMock()
    mock_client.get_part_detail.return_value = None
    mock_db.get.return_value = {"lcsc": "C25744"}
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server.get_part("C25744")
    assert result["source"] == "local_db_stale"
    assert "api_error" in result


def test_get_part_api_returns_none_no_cache(mocker):
    mock_db = mocker.MagicMock()
    mock_db.part_age_hours.return_value = None
    mock_db.get.return_value = None
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)

    mock_client = mocker.MagicMock()
    mock_client.get_part_detail.return_value = None
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server.get_part("C99999")
    assert result["success"] is False


def test_get_part_network_error_stale_cache_fallback(mocker):
    mock_db = mocker.MagicMock()
    mock_db.part_age_hours.return_value = None
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)

    mock_client = mocker.MagicMock()
    mock_client.get_part_detail.side_effect = RuntimeError("network down")
    mock_db.get.return_value = {"lcsc": "C25744"}
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server.get_part("C25744")
    assert result["source"] == "local_db_stale"
    assert "network down" in result["api_error"]


def test_get_part_network_error_no_cache(mocker):
    mock_db = mocker.MagicMock()
    mock_db.part_age_hours.return_value = None
    mock_db.get.return_value = None
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)

    mock_client = mocker.MagicMock()
    mock_client.get_part_detail.side_effect = RuntimeError("network down")
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server.get_part("C25744")
    assert result["success"] is False


def test_get_part_live_skips_local_cache(mocker):
    """live=True bypasses the local-cache check entirely."""
    mock_db = mocker.MagicMock()
    mock_db.get.return_value = {"lcsc": "C25744"}
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)

    mock_client = mocker.MagicMock()
    mock_client.get_part_detail.return_value = {"lcscPart": "C25744"}
    mocker.patch("jlcpcb_mcp.server._client", return_value=mock_client)

    result = server.get_part("C25744", live=True)
    assert result["source"] == "api"
    mock_db.part_age_hours.assert_not_called()


# ---------------------------------------------------------------------------
# suggest_alternatives
# ---------------------------------------------------------------------------

def test_suggest_alternatives_found_refreshes_rows(mocker):
    """The reference and each alternative get refreshed before being returned."""
    mock_db = mocker.MagicMock()
    ref = {
        "lcsc": "C25744",
        "library_type": "Basic",
        "stock": 1000000,
        "price_breaks": [{"qty": 20, "price": 0.001}],
    }
    alt = {"lcsc": "C25105", "stock": 200, "price_breaks": []}
    mock_db.get.side_effect = [ref, ref, alt]
    mock_db.suggest_alternatives.return_value = [{"lcsc": "C25105"}]
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    refresh_spy = mocker.patch("jlcpcb_mcp.server._refresh_stale_details")

    result = server.suggest_alternatives("C25744")
    assert result["success"] is True
    assert result["reference"]["price"] == pytest.approx(0.001)
    assert len(result["alternatives"]) == 1
    refresh_spy.assert_called_once()
    refreshed_codes = refresh_spy.call_args[0][1]
    assert refreshed_codes == ["C25744", "C25105"]


def test_suggest_alternatives_no_price_breaks(mocker):
    mock_db = mocker.MagicMock()
    ref = {"lcsc": "C25744", "library_type": "Basic", "stock": 100, "price_breaks": []}
    mock_db.get.side_effect = [ref, ref]
    mock_db.suggest_alternatives.return_value = []
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._refresh_stale_details")

    result = server.suggest_alternatives("C25744")
    assert result["reference"]["price"] is None


def test_suggest_alternatives_unknown_part(mocker):
    mock_db = mocker.MagicMock()
    mock_db.get.return_value = None
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    result = server.suggest_alternatives("C00000")
    assert result["success"] is False


def test_suggest_alternatives_alt_disappears_falls_back(mocker):
    """An alternative removed mid-flight (db.get returns None) falls back to the original row."""
    mock_db = mocker.MagicMock()
    ref = {"lcsc": "C25744", "library_type": "Basic", "stock": 100, "price_breaks": []}
    original_alt = {"lcsc": "C25105", "stock": 50}
    mock_db.get.side_effect = [ref, ref, None]  # alt missing on re-read
    mock_db.suggest_alternatives.return_value = [original_alt]
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._refresh_stale_details")

    result = server.suggest_alternatives("C25744")
    assert len(result["alternatives"]) == 1
    assert result["alternatives"][0] is original_alt


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

def test_get_stats_success(mocker):
    mock_db = mocker.MagicMock()
    mock_db.stats.return_value = {"total": 100, "basic": 50}
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    result = server.get_stats()
    assert result["success"] is True
    assert result["total"] == 100


def test_get_stats_error(mocker):
    mocker.patch("jlcpcb_mcp.server._db", side_effect=RuntimeError("DB error"))
    result = server.get_stats()
    assert result["success"] is False


# ---------------------------------------------------------------------------
# download_kicad_component
# ---------------------------------------------------------------------------

@pytest.fixture
def easyeda_mocks(mocker, tmp_path):
    """Mock all easyeda2kicad imports used inside download_kicad_component."""
    mock_cad_data = mocker.MagicMock()
    mock_api = mocker.MagicMock()
    mock_api.get_cad_data_of_component.return_value = mock_cad_data
    mocker.patch("easyeda2kicad.easyeda.easyeda_api.EasyedaApi", return_value=mock_api)

    mock_symbol = mocker.MagicMock()
    mock_symbol.info.name = "TEST_PART"
    mock_sym_importer = mocker.MagicMock()
    mock_sym_importer.get_symbol.return_value = mock_symbol
    mocker.patch("easyeda2kicad.easyeda.easyeda_importer.EasyedaSymbolImporter", return_value=mock_sym_importer)

    mock_sym_exporter = mocker.MagicMock()
    mock_sym_exporter.export.return_value = "(symbol content)"
    mocker.patch("easyeda2kicad.kicad.export_kicad_symbol.ExporterSymbolKicad", return_value=mock_sym_exporter)
    mocker.patch("easyeda2kicad.kicad.parameters_kicad_symbol.KicadVersion")

    mock_id_check = mocker.patch("easyeda2kicad.__main__.id_already_in_symbol_lib", return_value=False)
    mock_add = mocker.patch("easyeda2kicad.__main__.add_component_in_symbol_lib_file")
    mock_update = mocker.patch("easyeda2kicad.__main__.update_component_in_symbol_lib_file")

    mock_fp = mocker.MagicMock()
    mock_fp.info.name = "TEST_FP"
    mock_fp_importer = mocker.MagicMock()
    mock_fp_importer.get_footprint.return_value = mock_fp
    mocker.patch("easyeda2kicad.easyeda.easyeda_importer.EasyedaFootprintImporter", return_value=mock_fp_importer)

    mock_fp_exporter = mocker.MagicMock()
    mocker.patch("easyeda2kicad.kicad.export_kicad_footprint.ExporterFootprintKicad", return_value=mock_fp_exporter)
    mock_fp_check = mocker.patch("easyeda2kicad.__main__.fp_already_in_footprint_lib", return_value=False)

    mock_3d_output = mocker.MagicMock()
    mock_3d_output.name = "TEST_3D"
    mock_3d_importer = mocker.MagicMock()
    mock_3d_importer.output = mock_3d_output

    mock_3d_exporter = mocker.MagicMock()
    mock_3d_exporter.output = mock_3d_output
    mock_3d_exporter.output_step = mock_3d_output
    mocker.patch("easyeda2kicad.easyeda.easyeda_importer.Easyeda3dModelImporter", return_value=mock_3d_importer)
    mocker.patch("easyeda2kicad.kicad.export_kicad_3d_model.Exporter3dModelKicad", return_value=mock_3d_exporter)

    return {
        "cad_data": mock_cad_data,
        "api": mock_api,
        "symbol": mock_symbol,
        "id_check": mock_id_check,
        "add": mock_add,
        "update": mock_update,
        "fp_check": mock_fp_check,
        "fp_exporter": mock_fp_exporter,
        "3d_exporter": mock_3d_exporter,
        "tmp_path": tmp_path,
    }


def test_download_kicad_import_error(mocker):
    mocker.patch.dict(sys.modules, {"easyeda2kicad.easyeda.easyeda_api": None})
    result = server.download_kicad_component("C25744", output="/tmp/test")
    assert result["success"] is False
    assert "easyeda2kicad" in result["error"]


def test_download_kicad_no_data(easyeda_mocks):
    easyeda_mocks["api"].get_cad_data_of_component.return_value = None
    result = server.download_kicad_component(
        "C00000", output=str(easyeda_mocks["tmp_path"] / "EasyEDA")
    )
    assert result["success"] is False
    assert "No EasyEDA data" in result["error"]


def test_download_kicad_create_all(easyeda_mocks):
    lib_base = str(easyeda_mocks["tmp_path"] / "EasyEDA")
    result = server.download_kicad_component("C25744", output=lib_base)
    assert result["success"] is True
    assert result["files"]["symbol"].get("created") is True
    assert result["files"]["footprint"].get("created") is True
    assert result["files"]["model_3d"].get("created") is True


def test_download_kicad_symbol_update(easyeda_mocks):
    easyeda_mocks["id_check"].return_value = True
    lib_base = str(easyeda_mocks["tmp_path"] / "EasyEDA")
    result = server.download_kicad_component("C25744", output=lib_base, overwrite=True)
    assert result["files"]["symbol"].get("updated") is True
    easyeda_mocks["update"].assert_called_once()


def test_download_kicad_symbol_skip(easyeda_mocks):
    easyeda_mocks["id_check"].return_value = True
    lib_base = str(easyeda_mocks["tmp_path"] / "EasyEDA")
    result = server.download_kicad_component("C25744", output=lib_base, overwrite=False)
    assert result["files"]["symbol"].get("skipped") is True


def test_download_kicad_footprint_update(easyeda_mocks):
    easyeda_mocks["fp_check"].return_value = True
    lib_base = str(easyeda_mocks["tmp_path"] / "EasyEDA")
    result = server.download_kicad_component("C25744", output=lib_base, overwrite=True)
    assert result["files"]["footprint"].get("updated") is True


def test_download_kicad_footprint_skip(easyeda_mocks):
    easyeda_mocks["fp_check"].return_value = True
    lib_base = str(easyeda_mocks["tmp_path"] / "EasyEDA")
    result = server.download_kicad_component("C25744", output=lib_base, overwrite=False)
    assert result["files"]["footprint"].get("skipped") is True


def test_download_kicad_3d_no_output(easyeda_mocks):
    easyeda_mocks["3d_exporter"].output = None
    easyeda_mocks["3d_exporter"].output_step = None
    lib_base = str(easyeda_mocks["tmp_path"] / "EasyEDA")
    result = server.download_kicad_component("C25744", output=lib_base)
    assert result["files"]["model_3d"].get("skipped") is True


def test_download_kicad_3d_no_step(easyeda_mocks):
    easyeda_mocks["3d_exporter"].output_step = None
    lib_base = str(easyeda_mocks["tmp_path"] / "EasyEDA")
    result = server.download_kicad_component("C25744", output=lib_base)
    assert result["files"]["model_3d"].get("created") is True


def test_download_kicad_symbol_error(easyeda_mocks, mocker):
    mocker.patch(
        "easyeda2kicad.easyeda.easyeda_importer.EasyedaSymbolImporter",
        side_effect=RuntimeError("sym error"),
    )
    lib_base = str(easyeda_mocks["tmp_path"] / "EasyEDA")
    result = server.download_kicad_component("C25744", output=lib_base)
    assert "error" in result["files"]["symbol"]


def test_download_kicad_footprint_error(easyeda_mocks, mocker):
    mocker.patch(
        "easyeda2kicad.easyeda.easyeda_importer.EasyedaFootprintImporter",
        side_effect=RuntimeError("fp error"),
    )
    lib_base = str(easyeda_mocks["tmp_path"] / "EasyEDA")
    result = server.download_kicad_component("C25744", output=lib_base)
    assert "error" in result["files"]["footprint"]


def test_download_kicad_3d_error(easyeda_mocks, mocker):
    mocker.patch(
        "easyeda2kicad.easyeda.easyeda_importer.Easyeda3dModelImporter",
        side_effect=RuntimeError("3d error"),
    )
    lib_base = str(easyeda_mocks["tmp_path"] / "EasyEDA")
    result = server.download_kicad_component("C25744", output=lib_base)
    assert "error" in result["files"]["model_3d"]


def test_download_kicad_symbol_false(easyeda_mocks):
    lib_base = str(easyeda_mocks["tmp_path"] / "EasyEDA")
    result = server.download_kicad_component("C25744", output=lib_base, symbol=False)
    assert "symbol" not in result["files"]


def test_download_kicad_footprint_false(easyeda_mocks):
    lib_base = str(easyeda_mocks["tmp_path"] / "EasyEDA")
    result = server.download_kicad_component("C25744", output=lib_base, footprint=False)
    assert "footprint" not in result["files"]


def test_download_kicad_3d_false(easyeda_mocks):
    lib_base = str(easyeda_mocks["tmp_path"] / "EasyEDA")
    result = server.download_kicad_component("C25744", output=lib_base, model_3d=False)
    assert "model_3d" not in result["files"]


def test_download_kicad_default_output(easyeda_mocks, monkeypatch):
    """output=None → uses JLCPCB_EASYEDA_LIB_PATH env var."""
    monkeypatch.setenv("JLCPCB_EASYEDA_LIB_PATH", str(easyeda_mocks["tmp_path"] / "EasyEDA"))
    monkeypatch.delenv("JLCPCB_EASYEDA_3D_PATH", raising=False)
    result = server.download_kicad_component("C25744")
    assert result["success"] is True


def test_download_kicad_model_3d_path(easyeda_mocks):
    lib_base = str(easyeda_mocks["tmp_path"] / "EasyEDA")
    result = server.download_kicad_component(
        "C25744", output=lib_base, model_3d_path="${MY_LIB}/3dshapes"
    )
    assert result["success"] is True


def test_download_kicad_existing_sym_lib(easyeda_mocks, tmp_path):
    """When .kicad_sym already exists, the creation block is skipped."""
    lib_base = str(tmp_path / "EasyEDA")
    sym_lib = tmp_path / "EasyEDA.kicad_sym"
    sym_lib.write_text("(kicad_symbol_lib)", encoding="utf-8")
    result = server.download_kicad_component("C25744", output=lib_base)
    assert result["success"] is True
    assert "symbol" in result["files"]


# ---------------------------------------------------------------------------
# Error wrapping — _safe_search / _db_error_response
# ---------------------------------------------------------------------------

import sqlite3


def test_db_error_response_corruption_includes_repair_hint():
    out = server._db_error_response(sqlite3.DatabaseError("database disk image is malformed"))
    assert out["success"] is False
    assert "repair_fts" in out["hint"]


def test_db_error_response_other_error_no_hint():
    out = server._db_error_response(sqlite3.DatabaseError("syntax error"))
    assert out["success"] is False
    assert "hint" not in out


def test_search_parts_db_error_returned_to_caller(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search.side_effect = sqlite3.DatabaseError("database disk image is malformed")
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh", return_value=None)

    result = server.search_parts(query="x")
    assert result["success"] is False
    assert "malformed" in result["error"]
    assert "hint" in result


def test_search_resistors_db_error_returned_to_caller(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search_passive.side_effect = sqlite3.DatabaseError("malformed")
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh", return_value=None)
    result = server.search_resistors(value="10k")
    assert result["success"] is False


def test_search_capacitors_db_error_returned_to_caller(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search_passive.side_effect = sqlite3.DatabaseError("malformed")
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh", return_value=None)
    result = server.search_capacitors(value="100nF")
    assert result["success"] is False


def test_search_inductors_db_error_returned_to_caller(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search_passive.side_effect = sqlite3.DatabaseError("malformed")
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh", return_value=None)
    result = server.search_inductors(value="10uH")
    assert result["success"] is False


# ---------------------------------------------------------------------------
# skip_refresh
# ---------------------------------------------------------------------------

def test_search_parts_skip_refresh_bypasses_membership_and_details(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search.return_value = [{"lcsc": "C1", "stock": 100}]
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    membership = mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh")
    refresh = mocker.patch("jlcpcb_mcp.server._refresh_stale_details")

    result = server.search_parts(query="x", skip_refresh=True)
    assert result["success"] is True
    membership.assert_not_called()
    refresh.assert_not_called()


def test_search_resistors_skip_refresh(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search_passive.return_value = []
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    membership = mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh")
    refresh = mocker.patch("jlcpcb_mcp.server._refresh_stale_details")
    server.search_resistors(skip_refresh=True)
    membership.assert_not_called()
    refresh.assert_not_called()


def test_search_capacitors_skip_refresh(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search_passive.return_value = []
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    membership = mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh")
    server.search_capacitors(skip_refresh=True)
    membership.assert_not_called()


def test_search_inductors_skip_refresh(mocker):
    mock_db = mocker.MagicMock()
    mock_db.search_passive.return_value = []
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    membership = mocker.patch("jlcpcb_mcp.server._ensure_membership_fresh")
    server.search_inductors(skip_refresh=True)
    membership.assert_not_called()


# ---------------------------------------------------------------------------
# bom_check helpers
# ---------------------------------------------------------------------------

def test_unit_price_at_qty_below_smallest_break_returns_smallest():
    breaks = [{"qty": 100, "price": 0.01}, {"qty": 1000, "price": 0.005}]
    assert server._unit_price_at_qty(breaks, 1) == 0.01


def test_unit_price_at_qty_picks_largest_applicable():
    breaks = [{"qty": 1, "price": 0.1}, {"qty": 100, "price": 0.05}, {"qty": 1000, "price": 0.02}]
    assert server._unit_price_at_qty(breaks, 500) == 0.05
    assert server._unit_price_at_qty(breaks, 5000) == 0.02


def test_unit_price_at_qty_no_breaks_returns_none():
    assert server._unit_price_at_qty([], 10) is None


# ---------------------------------------------------------------------------
# bom_check tool
# ---------------------------------------------------------------------------

def _bom_db(mocker, parts_by_lcsc):
    """Build a mock PartsDB whose .get() looks up lcsc → dict."""
    mock_db = mocker.MagicMock()
    mock_db.get.side_effect = lambda code: parts_by_lcsc.get(code)
    mock_db.suggest_alternatives.return_value = []
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._refresh_stale_details")
    return mock_db


def test_bom_check_basic_aggregates(mocker):
    parts = {
        "C1": {"lcsc": "C1", "mfr_part": "A", "package": "0603", "library_type": "Basic",
               "stock": 1000, "price_breaks": [{"qty": 1, "price": 0.01}], "description": "x"},
        "C2": {"lcsc": "C2", "mfr_part": "B", "package": "0603", "library_type": "Extended",
               "stock": 0, "price_breaks": [{"qty": 1, "price": 0.05}], "description": "y"},
    }
    _bom_db(mocker, parts)
    result = server.bom_check(items=["C1", "C2", "C_NONE"], qty=10, suggest_alternatives=False)
    assert result["success"] is True
    assert result["summary"]["found"] == 2
    assert result["summary"]["not_found"] == ["C_NONE"]
    assert result["summary"]["out_of_stock"] == ["C2"]
    assert result["summary"]["basic_count"] == 1
    assert result["summary"]["extended_count"] == 1
    assert result["summary"]["total_cost"] == pytest.approx(0.6)


def test_bom_check_dict_items_with_per_line_qty(mocker):
    parts = {"C1": {"lcsc": "C1", "mfr_part": "A", "package": "0603",
                    "library_type": "Basic", "stock": 100,
                    "price_breaks": [{"qty": 1, "price": 0.1}], "description": ""}}
    _bom_db(mocker, parts)
    result = server.bom_check(
        items=[{"lcsc": "C1", "qty": 50}, {"lcsc": "C1", "qty": 5}],
        suggest_alternatives=False,
    )
    qtys = [r["qty"] for r in result["lines"]]
    assert qtys == [50, 5]


def test_bom_check_insufficient_stock_warning(mocker):
    parts = {"C1": {"lcsc": "C1", "mfr_part": "A", "package": "", "library_type": "Basic",
                    "stock": 5, "price_breaks": [{"qty": 1, "price": 0.1}], "description": ""}}
    _bom_db(mocker, parts)
    result = server.bom_check(items=["C1"], qty=100, suggest_alternatives=False)
    assert result["summary"]["insufficient_stock"] == ["C1"]


def test_bom_check_suggests_basic_alternative_for_extended(mocker):
    parts = {
        "C_EXT": {"lcsc": "C_EXT", "mfr_part": "Ext", "package": "0603", "library_type": "Extended",
                  "stock": 100, "price_breaks": [{"qty": 1, "price": 0.10}], "description": "x"},
        "C_BASIC": {"lcsc": "C_BASIC", "mfr_part": "Basic", "package": "0603", "library_type": "Basic",
                    "stock": 1000, "price_breaks": [{"qty": 1, "price": 0.02}], "description": "x"},
    }
    mock_db = _bom_db(mocker, parts)
    mock_db.suggest_alternatives.return_value = [parts["C_BASIC"]]

    result = server.bom_check(items=["C_EXT"], qty=10)
    line = result["lines"][0]
    assert line["suggested_basic_alternative"]["lcsc"] == "C_BASIC"
    assert line["suggested_basic_alternative"]["savings_per_unit"] == pytest.approx(0.08)
    assert result["summary"]["potential_savings_with_basic_alternatives"] == pytest.approx(0.8)


def test_bom_check_alternative_swallows_db_error(mocker):
    parts = {"C_EXT": {"lcsc": "C_EXT", "mfr_part": "X", "package": "0603", "library_type": "Extended",
                       "stock": 100, "price_breaks": [{"qty": 1, "price": 0.1}], "description": ""}}
    mock_db = _bom_db(mocker, parts)
    mock_db.suggest_alternatives.side_effect = sqlite3.DatabaseError("bad")
    result = server.bom_check(items=["C_EXT"], qty=1)
    # Alternative lookup failed but main row still reported.
    assert result["success"] is True
    assert "suggested_basic_alternative" not in result["lines"][0]


def test_bom_check_empty_items_returns_error(mocker):
    mocker.patch("jlcpcb_mcp.server._db")
    result = server.bom_check(items=[], qty=1)
    assert result["success"] is False


def test_bom_check_skip_refresh_calls_no_api(mocker):
    parts = {"C1": {"lcsc": "C1", "mfr_part": "", "package": "", "library_type": "Basic",
                    "stock": 1, "price_breaks": [], "description": ""}}
    mock_db = mocker.MagicMock()
    mock_db.get.side_effect = lambda c: parts.get(c)
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    refresh = mocker.patch("jlcpcb_mcp.server._refresh_stale_details")
    server.bom_check(items=["C1"], skip_refresh=True, suggest_alternatives=False)
    refresh.assert_not_called()


def test_bom_check_db_error_during_iteration(mocker):
    mock_db = mocker.MagicMock()
    mock_db.get.side_effect = sqlite3.DatabaseError("malformed")
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    mocker.patch("jlcpcb_mcp.server._refresh_stale_details")
    result = server.bom_check(items=["C1"], qty=1)
    assert result["success"] is False
    assert "hint" in result


def test_bom_check_alternative_skips_non_basic_and_priceless(mocker):
    """Alternatives that aren't Basic, or that have no usable price, must be skipped."""
    parts = {"C_EXT": {"lcsc": "C_EXT", "mfr_part": "X", "package": "0603", "library_type": "Extended",
                       "stock": 100, "price_breaks": [{"qty": 1, "price": 0.10}], "description": ""}}
    mock_db = _bom_db(mocker, parts)
    mock_db.suggest_alternatives.return_value = [
        # Non-Basic alternative — must be skipped (continue branch).
        {"lcsc": "C_OTHER", "library_type": "Extended", "price_breaks": [{"qty": 1, "price": 0.01}]},
        # Basic but no price → skipped.
        {"lcsc": "C_NOPRICE", "library_type": "Basic", "price_breaks": []},
        # Basic + cheaper → wins.
        {"lcsc": "C_BASIC", "mfr_part": "B", "library_type": "Basic",
         "price_breaks": [{"qty": 1, "price": 0.05}]},
    ]
    result = server.bom_check(items=["C_EXT"], qty=1)
    assert result["lines"][0]["suggested_basic_alternative"]["lcsc"] == "C_BASIC"


def test_bom_check_skips_more_expensive_basic_alternative(mocker):
    """A Basic alt that costs >= the Extended reference must NOT be suggested."""
    parts = {"C_EXT": {"lcsc": "C_EXT", "mfr_part": "X", "package": "0603", "library_type": "Extended",
                       "stock": 100, "price_breaks": [{"qty": 1, "price": 0.05}], "description": ""}}
    mock_db = _bom_db(mocker, parts)
    mock_db.suggest_alternatives.return_value = [
        {"lcsc": "C_BASIC_EXPENSIVE", "library_type": "Basic",
         "price_breaks": [{"qty": 1, "price": 0.10}]},
    ]
    result = server.bom_check(items=["C_EXT"], qty=1)
    assert "suggested_basic_alternative" not in result["lines"][0]


def test_bom_check_alternative_used_when_reference_has_no_price(mocker):
    """If the reference Extended part has no price_breaks, any Basic alternative wins."""
    parts = {"C_EXT": {"lcsc": "C_EXT", "mfr_part": "X", "package": "", "library_type": "Extended",
                       "stock": 1, "price_breaks": [], "description": ""}}
    mock_db = _bom_db(mocker, parts)
    mock_db.suggest_alternatives.return_value = [
        {"lcsc": "C_BASIC", "mfr_part": "B", "library_type": "Basic",
         "price_breaks": [{"qty": 1, "price": 0.05}]},
    ]
    result = server.bom_check(items=["C_EXT"], qty=2)
    alt = result["lines"][0]["suggested_basic_alternative"]
    assert alt["lcsc"] == "C_BASIC"
    assert alt["savings_per_unit"] is None


def test_bom_check_ignores_malformed_items(mocker):
    """Non-str / dict-without-lcsc items are silently dropped."""
    parts = {"C1": {"lcsc": "C1", "mfr_part": "", "package": "", "library_type": "Basic",
                    "stock": 1, "price_breaks": [{"qty": 1, "price": 0.1}], "description": ""}}
    _bom_db(mocker, parts)
    result = server.bom_check(
        items=["C1", {"qty": 5}, 42, {"lcsc": ""}],
        qty=1, suggest_alternatives=False,
    )
    # Only the "C1" string item survived parsing.
    assert [r["lcsc"] for r in result["lines"]] == ["C1"]


def test_bom_check_zero_qty_break_omits_unit(mocker):
    """Part with no price_breaks → unit_price=None, line_total=None."""
    parts = {"C1": {"lcsc": "C1", "mfr_part": "", "package": "", "library_type": "Basic",
                    "stock": 1, "price_breaks": [], "description": ""}}
    _bom_db(mocker, parts)
    result = server.bom_check(items=["C1"], qty=10, suggest_alternatives=False)
    assert result["lines"][0]["unit_price"] is None
    assert result["lines"][0]["line_total"] is None


# ---------------------------------------------------------------------------
# kicad_bom_check tool
# ---------------------------------------------------------------------------

def test_extract_lcsc_codes_from_kicad_handles_variants():
    text = """
    (property "LCSC Part" "C25744" (id 9))
    (property "LCSC#" "C1525")
    (property "JLCPCB" "C2827693")
    (property "Reference" "R1")
    (property "LCSC Part" "" (id 9))
    """
    codes = server._extract_lcsc_codes_from_kicad(text)
    assert codes == ["C25744", "C1525", "C2827693"]


def test_kicad_bom_check_missing_file_returns_error(tmp_path):
    result = server.kicad_bom_check(sch_path=str(tmp_path / "missing.kicad_sch"))
    assert result["success"] is False
    assert "not found" in result["error"]


def test_kicad_bom_check_no_codes_returns_error(tmp_path):
    sch = tmp_path / "empty.kicad_sch"
    sch.write_text("(kicad_sch (version 1))", encoding="utf-8")
    result = server.kicad_bom_check(sch_path=str(sch))
    assert result["success"] is False
    assert "no LCSC" in result["error"]


def test_kicad_bom_check_counts_duplicates_and_passes_to_bom_check(mocker, tmp_path):
    sch = tmp_path / "x.kicad_sch"
    sch.write_text(
        '(property "LCSC Part" "C25744")\n'
        '(property "LCSC Part" "C25744")\n'
        '(property "LCSC Part" "C1525")\n',
        encoding="utf-8",
    )
    mock_bom = mocker.patch("jlcpcb_mcp.server.bom_check", return_value={"success": True})
    result = server.kicad_bom_check(sch_path=str(sch), qty=10)
    assert result["unique_codes"] == 2
    assert result["total_components"] == 3
    items = mock_bom.call_args.kwargs["items"]
    qty_by_code = {it["lcsc"]: it["qty"] for it in items}
    assert qty_by_code == {"C25744": 20, "C1525": 10}


def test_kicad_bom_check_propagates_bom_failure(mocker, tmp_path):
    sch = tmp_path / "x.kicad_sch"
    sch.write_text('(property "LCSC Part" "C25744")', encoding="utf-8")
    mocker.patch("jlcpcb_mcp.server.bom_check", return_value={"success": False, "error": "x"})
    result = server.kicad_bom_check(sch_path=str(sch))
    assert result["success"] is False


def test_kicad_bom_check_unreadable_file(tmp_path, mocker):
    sch = tmp_path / "x.kicad_sch"
    sch.write_text("(property \"LCSC Part\" \"C1\")", encoding="utf-8")
    mocker.patch.object(server.Path, "read_text", side_effect=OSError("permission denied"))
    # Use a different path-construction strategy: monkeypatch Path.read_text on the instance
    # by patching pathlib.Path
    import pathlib
    real_read_text = pathlib.Path.read_text
    def boom(self, *a, **k):
        if str(self).endswith("x.kicad_sch"):
            raise OSError("denied")
        return real_read_text(self, *a, **k)
    mocker.patch.object(pathlib.Path, "read_text", boom)
    result = server.kicad_bom_check(sch_path=str(sch))
    assert result["success"] is False
    assert "denied" in result["error"]


# ---------------------------------------------------------------------------
# repair_db tool
# ---------------------------------------------------------------------------

def test_repair_db_runs_all_steps(mocker):
    from jlcpcb_mcp.db import PartsDB
    mocker.patch("jlcpcb_mcp.server._db", return_value=PartsDB(":memory:"))
    result = server.repair_db()
    assert result["success"] is True
    step_keys = [list(s.keys())[0] for s in result["steps"]]
    assert step_keys == ["integrity_check", "fts_integrity_check", "fts_rebuild"]
    assert "stats" in result


def test_repair_db_integrity_check_failure(mocker):
    mock_db = mocker.MagicMock()
    mock_db._conn.execute.side_effect = sqlite3.DatabaseError("malformed")
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    result = server.repair_db()
    assert result["success"] is False
    assert "hint" in result


def test_repair_db_fts_integrity_check_reports_error(mocker):
    """integrity_check passes, FTS integrity-check fails, rebuild succeeds → step records the error."""
    mock_db = mocker.MagicMock()
    calls = {"n": 0}

    def execute(sql, *a):
        calls["n"] += 1
        if calls["n"] == 1:  # PRAGMA integrity_check
            cur = mocker.MagicMock()
            cur.__iter__ = lambda self: iter([("ok",)])
            return cur
        if calls["n"] == 2:  # FTS integrity-check
            raise sqlite3.DatabaseError("fts borked")
        return mocker.MagicMock()

    mock_db._conn.execute.side_effect = execute
    mock_db.rebuild_fts.return_value = None
    mock_db.stats.return_value = {"total": 0}
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    result = server.repair_db()
    assert result["success"] is True
    fts_step = next(s for s in result["steps"] if "fts_integrity_check" in s)
    assert fts_step["fts_integrity_check"] == "fts borked"


def test_repair_db_rebuild_failure(mocker):
    mock_db = mocker.MagicMock()
    cur_ok = mocker.MagicMock()
    cur_ok.__iter__ = lambda self: iter([("ok",)])
    mock_db._conn.execute.side_effect = [cur_ok, mocker.MagicMock()]
    mock_db.rebuild_fts.side_effect = sqlite3.DatabaseError("malformed")
    mocker.patch("jlcpcb_mcp.server._db", return_value=mock_db)
    result = server.repair_db()
    assert result["success"] is False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def test_main(mocker):
    mock_run = mocker.patch.object(server.mcp, "run")
    server.main()
    mock_run.assert_called_once()
