"""
JLCPCB official API client — HMAC-SHA256 authentication.

Env vars required:
  JLCPCB_APP_ID
  JLCPCB_API_KEY
  JLCPCB_API_SECRET
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import string
import time
from typing import Callable, Iterator, Optional

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

BASE_URL = "https://open.jlcpcb.com"
_BASE = "/overseas/openapi/component"

ENDPOINT_DETAIL   = f"{_BASE}/getComponentDetailByCode"  # batch lookup by LCSC code (≤ 1000)
ENDPOINT_LIB_LIST = f"{_BASE}/getComponentLibraryList"   # Basic/Extended library list (cursor)

# getComponentDetailByCode accepts up to 1000 codes per call.
DETAIL_BATCH_MAX = 1000

# Rate limiting: pause between paginated calls, backoff on 429 / 5xx
# (2, 4, 8, … s, capped at 120 s → ~10 min total before giving up).
PAGE_PAUSE_S = 1.0
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 8
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 120.0


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Missing environment variable: {name}. "
            "Set JLCPCB_APP_ID, JLCPCB_API_KEY and JLCPCB_API_SECRET."
        )
    return value


def _nonce() -> str:
    chars = string.ascii_letters + string.digits
    return "".join(secrets.choice(chars) for _ in range(32))


def _sign(secret_key: str, method: str, path: str, timestamp: int, nonce: str, body: str) -> str:
    msg = f"{method}\n{path}\n{timestamp}\n{nonce}\n{body}\n"
    sig = hmac.new(
        secret_key.encode(),
        msg.encode(),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(sig).decode()


def _auth_header(app_id: str, access_key: str, secret_key: str, method: str, path: str, body: str) -> str:
    ts = int(time.time())
    n = _nonce()
    sig = _sign(secret_key, method, path, ts, n, body)
    return (
        f'JOP appid="{app_id}",accesskey="{access_key}",'
        f'nonce="{n}",timestamp="{ts}",signature="{sig}"'
    )


class JLCPCBClient:
    """Thread-safe JLCPCB API client with persistent connection pool."""

    def __init__(self) -> None:
        self.app_id = _require_env("JLCPCB_APP_ID")
        self.access_key = _require_env("JLCPCB_API_KEY")
        self.secret_key = _require_env("JLCPCB_API_SECRET")

        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=2, pool_maxsize=4)
        self._session.mount("https://", adapter)

    def _post(self, endpoint: str, payload: dict, timeout: int = 60) -> dict:
        """Signed POST → returns response['data'].

        Retries 429 / 5xx with exponential backoff (honours Retry-After).
        The signature carries a timestamp + nonce, so it is rebuilt per attempt.
        """
        body = json.dumps(payload, separators=(",", ":"))
        attempt = 0
        while True:
            auth = _auth_header(self.app_id, self.access_key, self.secret_key, "POST", endpoint, body)
            resp = self._session.post(
                f"{BASE_URL}{endpoint}",
                headers={"Authorization": auth, "Content-Type": "application/json"},
                data=body.encode(),
                timeout=timeout,
            )
            if resp.status_code not in RETRY_STATUS or attempt == MAX_RETRIES:
                break
            try:
                delay = float(resp.headers.get("Retry-After", ""))
            except ValueError:
                delay = min(BACKOFF_BASE_S * 2 ** attempt, BACKOFF_MAX_S)
            attempt += 1
            logger.warning(
                "JLCPCB %s → HTTP %d, retry %d/%d in %.0f s",
                endpoint, resp.status_code, attempt, MAX_RETRIES, delay,
            )
            time.sleep(delay)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 200:
            msg = data.get("message") or data.get("msg") or "unknown error"
            raise RuntimeError(f"JLCPCB API error {data.get('code')}: {msg}")
        return data.get("data") or {}

    @staticmethod
    def _normalize_detail(raw: dict) -> dict:
        """Convert getComponentDetailByCode response to import_batch (catalog) format."""
        price_parts = []
        for r in (raw.get("priceRanges") or []):
            end = r["endQuantity"] if r["endQuantity"] > 0 else r["startQuantity"]
            price_parts.append(f"{r['startQuantity']}-{end}:{r['unitPrice']}")
        return {
            "lcscPart":       raw.get("componentCode"),
            "firstCategory":  raw.get("firstTypeName"),
            "secondCategory": raw.get("secondTypeName"),
            "mfrPart":        raw.get("componentModel"),
            "package":        raw.get("componentSpecification"),
            "solderJoint":    raw.get("solderJointCount", 0),
            "manufacturer":   "",
            "libraryType":    raw.get("libraryType", ""),
            "description":    raw.get("description", ""),
            "datasheet":      raw.get("datasheetUrl") or "",
            "stock":          raw.get("stockCount", 0),
            "price":          ",".join(price_parts),
        }

    def get_part_detail(self, lcsc_code: str) -> Optional[dict]:
        """
        Fetch live detail for a single component by LCSC code (e.g. 'C25804').
        Returns the component dict normalized to catalog format, None if not found,
        or raises RuntimeError on API/auth errors.
        """
        results = self.get_parts_details([lcsc_code])
        return results[0] if results else None

    def get_parts_details(self, codes: list[str]) -> list[dict]:
        """
        Batch-fetch full details for a list of LCSC codes.

        Splits the call into chunks of DETAIL_BATCH_MAX (1000) — the API's per-call limit.
        Returns a list of dicts normalized to the catalog format.
        """
        if not codes:
            return []
        out: list[dict] = []
        for i in range(0, len(codes), DETAIL_BATCH_MAX):
            chunk = codes[i:i + DETAIL_BATCH_MAX]
            data = self._post(ENDPOINT_DETAIL, {"componentCodes": chunk})
            items = (
                data.get("componentDetailResponseVOList", [])
                if isinstance(data, dict)
                else (data if isinstance(data, list) else [])
            )
            out.extend(self._normalize_detail(r) for r in items)
        return out

    def get_library_list(
        self,
        last_key: Optional[str] = None,
        page_size: int = 100,
    ) -> tuple[list, Optional[str]]:
        """
        Fetch one page of the assembly library list (cursor-based pagination).

        Args:
            last_key: Cursor from the previous response to fetch the next page.
            page_size: Number of records per page (max 100).

        Returns:
            Tuple of (stubs, next_last_key). next_last_key is None on the last page.
        """
        payload: dict = {"pageSize": page_size}
        if last_key:
            payload["lastKey"] = last_key
        data = self._post(ENDPOINT_LIB_LIST, payload)
        if isinstance(data, dict):
            return data.get("componentLibraryInfoVOS", []), data.get("lastKey")
        return [], None

    def iter_library_stubs(
        self,
        page_size: int = 100,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> Iterator[dict]:
        """
        Yield every stub in the assembly library, paginating through the cursor.

        Stubs are lightweight (componentCode, componentModel, componentSpecification) —
        useful for membership-diff refreshes that only need to know which codes exist
        without re-fetching every detail.

        Args:
            page_size: Records per page (max 100).
            on_progress: Optional callback invoked with the cumulative count after each page.
        """
        last_key: Optional[str] = None
        total = 0
        while True:
            stubs, next_key = self.get_library_list(last_key=last_key, page_size=page_size)
            if not stubs:
                break
            for stub in stubs:
                yield stub
            total += len(stubs)
            if on_progress:
                on_progress(total)
            if not next_key:
                break
            last_key = next_key
            time.sleep(PAGE_PAUSE_S)

    def download_library(
        self,
        on_batch: Optional[Callable[[list], None]] = None,
        on_progress: Optional[Callable[[int, str], None]] = None,
    ) -> list:
        """
        Download the full assembly library list (Basic + Extended).

        Uses getComponentLibraryList cursor pagination, then enriches each batch
        with getComponentDetailByCode to get price, stock, and libraryType.

        Args:
            on_batch: Streaming callback; if None, returns accumulated list.
            on_progress: Optional progress callback.

        Returns:
            All parts (empty list when on_batch is used).
        """
        all_parts: list = []
        total = 0
        page = 0
        last_key: Optional[str] = None

        while True:
            page += 1
            stubs, next_key = self.get_library_list(last_key=last_key, page_size=100)

            if not stubs:
                break

            codes = [s["componentCode"] for s in stubs if s.get("componentCode")]
            parts = self.get_parts_details(codes)

            if parts:
                if on_batch:
                    on_batch(parts)
                else:
                    all_parts.extend(parts)
                total += len(parts)

            if on_progress:
                on_progress(total, f"Library page {page}: {total} parts")
            elif page % 50 == 0:
                logger.info("Library page %d: %d parts", page, total)

            if not next_key:
                break
            last_key = next_key

            time.sleep(PAGE_PAUSE_S)

        return all_parts
