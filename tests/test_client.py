"""Tests for jlcpcb_mcp/client.py."""
import base64
import hashlib
import hmac
import string

import pytest
import requests

from jlcpcb_mcp.client import (
    BACKOFF_MAX_S,
    DETAIL_BATCH_MAX,
    MAX_RETRIES,
    PAGE_PAUSE_MAX_S,
    PAGE_PAUSE_MIN_S,
    JLCPCBClient,
    _auth_header,
    _nonce,
    _require_env,
    _sign,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_require_env_present(monkeypatch):
    monkeypatch.setenv("_TEST_LCSC_VAR", "hello")
    assert _require_env("_TEST_LCSC_VAR") == "hello"


def test_require_env_missing(monkeypatch):
    monkeypatch.delenv("_TEST_LCSC_VAR", raising=False)
    with pytest.raises(EnvironmentError, match="_TEST_LCSC_VAR"):
        _require_env("_TEST_LCSC_VAR")


def test_nonce_length():
    assert len(_nonce()) == 32


def test_nonce_alphanumeric():
    valid = set(string.ascii_letters + string.digits)
    assert all(c in valid for c in _nonce())


def test_sign_deterministic():
    a = _sign("secret", "POST", "/path", 12345, "nonce", "body")
    b = _sign("secret", "POST", "/path", 12345, "nonce", "body")
    assert a == b


def test_sign_correct_hmac():
    secret, method, path, ts, nonce, body = "key", "POST", "/ep", 1, "n", "{}"
    msg = f"{method}\n{path}\n{ts}\n{nonce}\n{body}\n"
    expected = base64.b64encode(
        hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()
    ).decode()
    assert _sign(secret, method, path, ts, nonce, body) == expected


def test_auth_header_format(monkeypatch):
    monkeypatch.setattr("jlcpcb_mcp.client._nonce", lambda: "TESTNONCE")
    monkeypatch.setattr("jlcpcb_mcp.client.time.time", lambda: 9999)
    h = _auth_header("APPID", "ACCESSKEY", "SECRET", "POST", "/ep", "{}")
    assert h.startswith('JOP appid="APPID"')
    assert 'accesskey="ACCESSKEY"' in h
    assert 'nonce="TESTNONCE"' in h
    assert 'timestamp="9999"' in h


# ---------------------------------------------------------------------------
# JLCPCBClient fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(env_vars):
    return JLCPCBClient()


def _mock_response(mocker, json_data):
    resp = mocker.MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

def test_init_missing_env(monkeypatch):
    for var in ("JLCPCB_APP_ID", "JLCPCB_API_KEY", "JLCPCB_API_SECRET"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(EnvironmentError):
        JLCPCBClient()


# ---------------------------------------------------------------------------
# _post
# ---------------------------------------------------------------------------

def test_post_success(client, mocker):
    resp = _mock_response(mocker, {"code": 200, "data": {"key": "val"}})
    mocker.patch.object(client._session, "post", return_value=resp)
    assert client._post("/ep", {}) == {"key": "val"}


def test_post_empty_data(client, mocker):
    resp = _mock_response(mocker, {"code": 200, "data": None})
    mocker.patch.object(client._session, "post", return_value=resp)
    assert client._post("/ep", {}) == {}


def _status_response(mocker, status, headers=None):
    resp = _mock_response(mocker, {"code": 200, "data": {"ok": True}})
    resp.status_code = status
    resp.headers = headers or {}
    return resp


def test_post_retries_429_with_backoff(client, mocker):
    sleep = mocker.patch("jlcpcb_mcp.client.time.sleep")
    responses = [_status_response(mocker, 429), _status_response(mocker, 504), _status_response(mocker, 200)]
    post = mocker.patch.object(client._session, "post", side_effect=responses)
    assert client._post("/ep", {}) == {"ok": True}
    assert post.call_count == 3
    assert [c.args[0] for c in sleep.call_args_list] == [2.0, 4.0]


def test_post_retry_honours_retry_after(client, mocker):
    sleep = mocker.patch("jlcpcb_mcp.client.time.sleep")
    responses = [_status_response(mocker, 429, {"Retry-After": "7"}), _status_response(mocker, 200)]
    mocker.patch.object(client._session, "post", side_effect=responses)
    client._post("/ep", {})
    sleep.assert_called_once_with(7.0)


def test_post_backoff_is_capped(client, mocker):
    sleep = mocker.patch("jlcpcb_mcp.client.time.sleep")
    responses = [_status_response(mocker, 503) for _ in range(MAX_RETRIES)] + [_status_response(mocker, 200)]
    mocker.patch.object(client._session, "post", side_effect=responses)
    client._post("/ep", {})
    assert max(c.args[0] for c in sleep.call_args_list) == BACKOFF_MAX_S


def test_post_gives_up_after_max_retries(client, mocker):
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    last = _status_response(mocker, 429)
    last.raise_for_status.side_effect = requests.HTTPError("429 Too Many Requests")
    responses = [_status_response(mocker, 429) for _ in range(MAX_RETRIES)] + [last]
    post = mocker.patch.object(client._session, "post", side_effect=responses)
    with pytest.raises(requests.HTTPError):
        client._post("/ep", {})
    assert post.call_count == MAX_RETRIES + 1


def test_post_no_retry_on_client_error(client, mocker):
    sleep = mocker.patch("jlcpcb_mcp.client.time.sleep")
    resp = _status_response(mocker, 403)
    resp.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
    post = mocker.patch.object(client._session, "post", return_value=resp)
    with pytest.raises(requests.HTTPError):
        client._post("/ep", {})
    assert post.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize("header, expected", [
    ("7", 7.0),
    ("100000", BACKOFF_MAX_S),   # huge value is capped
    ("-5", 0.0),                 # negative is clamped, time.sleep must not raise
    ("inf", 2.0),                # not finite -> normal backoff (attempt 0)
    ("nan", 2.0),
    ("Wed, 21 Oct 2026 07:28:00 GMT", 2.0),  # HTTP-date form -> normal backoff
])
def test_post_retry_after_is_clamped(client, mocker, header, expected):
    sleep = mocker.patch("jlcpcb_mcp.client.time.sleep")
    responses = [_status_response(mocker, 429, {"Retry-After": header}), _status_response(mocker, 200)]
    mocker.patch.object(client._session, "post", side_effect=responses)
    client._post("/ep", {})
    sleep.assert_called_once_with(expected)


def test_post_max_retries_parameter(client, mocker):
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    last = _status_response(mocker, 503)
    last.raise_for_status.side_effect = requests.HTTPError("503")
    responses = [_status_response(mocker, 503), _status_response(mocker, 503), last]
    post = mocker.patch.object(client._session, "post", side_effect=responses)
    with pytest.raises(requests.HTTPError):
        client._post("/ep", {}, max_retries=2)
    assert post.call_count == 3


def test_post_zero_retries_fails_immediately(client, mocker):
    sleep = mocker.patch("jlcpcb_mcp.client.time.sleep")
    resp = _status_response(mocker, 503)
    resp.raise_for_status.side_effect = requests.HTTPError("503")
    post = mocker.patch.object(client._session, "post", return_value=resp)
    with pytest.raises(requests.HTTPError):
        client._post("/ep", {}, max_retries=0)
    assert post.call_count == 1
    sleep.assert_not_called()


def test_page_pause_starts_fast_and_doubles_on_429(client, mocker):
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    assert client.page_pause == PAGE_PAUSE_MIN_S
    responses = [_status_response(mocker, 429), _status_response(mocker, 200)]
    mocker.patch.object(client._session, "post", side_effect=responses)
    client._post("/ep", {})
    assert client.page_pause == PAGE_PAUSE_MIN_S * 2


def test_page_pause_not_raised_by_5xx(client, mocker):
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    responses = [_status_response(mocker, 503), _status_response(mocker, 200)]
    mocker.patch.object(client._session, "post", side_effect=responses)
    client._post("/ep", {})
    assert client.page_pause == PAGE_PAUSE_MIN_S


def test_page_pause_is_capped(client, mocker):
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    responses = [_status_response(mocker, 429) for _ in range(MAX_RETRIES)] + [_status_response(mocker, 200)]
    mocker.patch.object(client._session, "post", side_effect=responses)
    client._post("/ep", {})
    assert client.page_pause == PAGE_PAUSE_MAX_S


def test_iter_library_stubs_sleeps_current_page_pause(client, mocker):
    sleep = mocker.patch("jlcpcb_mcp.client.time.sleep")
    client.page_pause = 0.8
    pages = [([_make_stub("C1")], "k1"), ([_make_stub("C2")], None)]
    mocker.patch.object(client, "get_library_list", side_effect=pages)
    list(client.iter_library_stubs())
    sleep.assert_called_once_with(0.8)


def test_get_part_detail_passes_max_retries(client, mocker):
    post = mocker.patch.object(client, "_post", return_value={"componentDetailResponseVOList": []})
    client.get_part_detail("C1", max_retries=2)
    assert post.call_args.kwargs["max_retries"] == 2


def test_post_api_error_message(client, mocker):
    resp = _mock_response(mocker, {"code": 400, "message": "bad request"})
    mocker.patch.object(client._session, "post", return_value=resp)
    with pytest.raises(RuntimeError, match="bad request"):
        client._post("/ep", {})


def test_post_api_error_msg_fallback(client, mocker):
    resp = _mock_response(mocker, {"code": 500, "msg": "server error"})
    mocker.patch.object(client._session, "post", return_value=resp)
    with pytest.raises(RuntimeError, match="server error"):
        client._post("/ep", {})


def test_post_api_error_unknown(client, mocker):
    resp = _mock_response(mocker, {"code": 503})
    mocker.patch.object(client._session, "post", return_value=resp)
    with pytest.raises(RuntimeError, match="unknown error"):
        client._post("/ep", {})


# ---------------------------------------------------------------------------
# _normalize_detail
# ---------------------------------------------------------------------------

_SAMPLE_DETAIL = {
    "componentCode": "C25744",
    "firstTypeName": "Resistors",
    "secondTypeName": "Chip Resistor - Surface Mount",
    "componentModel": "0402WGF1002TCE",
    "componentSpecification": "0402",
    "solderJointCount": 2,
    "libraryType": "base",
    "description": "10kΩ ±1% 1/16W",
    "datasheetUrl": "https://example.com/ds.pdf",
    "stockCount": 1000000,
    "priceRanges": [
        {"startQuantity": 1, "endQuantity": 999, "unitPrice": 0.001},
        {"startQuantity": 1000, "endQuantity": -1, "unitPrice": 0.0008},
    ],
}


def test_normalize_detail_fields():
    n = JLCPCBClient._normalize_detail(_SAMPLE_DETAIL)
    assert n["lcscPart"] == "C25744"
    assert n["firstCategory"] == "Resistors"
    assert n["mfrPart"] == "0402WGF1002TCE"
    assert n["package"] == "0402"
    assert n["stock"] == 1000000
    assert n["datasheet"] == "https://example.com/ds.pdf"


def test_normalize_detail_price_string():
    n = JLCPCBClient._normalize_detail(_SAMPLE_DETAIL)
    # priceRanges → "1-999:0.001,1000-1000:0.0008"
    assert "1-999:0.001" in n["price"]
    assert "1000-1000:0.0008" in n["price"]


def test_normalize_detail_datasheet_empty():
    """Missing datasheetUrl → empty string."""
    raw = {**_SAMPLE_DETAIL, "datasheetUrl": ""}
    n = JLCPCBClient._normalize_detail(raw)
    assert n["datasheet"] == ""


def test_normalize_detail_no_price_ranges():
    raw = {**_SAMPLE_DETAIL, "priceRanges": []}
    n = JLCPCBClient._normalize_detail(raw)
    assert n["price"] == ""


# ---------------------------------------------------------------------------
# get_part_detail
# ---------------------------------------------------------------------------

def test_get_part_detail_found(client, mocker):
    mocker.patch.object(client, "_post", return_value={"componentDetailResponseVOList": [_SAMPLE_DETAIL]})
    result = client.get_part_detail("C25744")
    assert result["lcscPart"] == "C25744"


def test_get_part_detail_uses_correct_param(client, mocker):
    """API call uses 'componentCodes' list, not 'componentCode' string."""
    mock_post = mocker.patch.object(client, "_post", return_value=[_SAMPLE_DETAIL])
    client.get_part_detail("C25744")
    _, payload = mock_post.call_args[0]
    assert payload == {"componentCodes": ["C25744"]}


def test_get_part_detail_empty_list(client, mocker):
    mocker.patch.object(client, "_post", return_value={"componentDetailResponseVOList": []})
    assert client.get_part_detail("C99999") is None


def test_get_part_detail_non_list_response(client, mocker):
    """Unexpected non-list response → None."""
    mocker.patch.object(client, "_post", return_value={})
    assert client.get_part_detail("C00000") is None


def test_get_part_detail_runtime_error(client, mocker):
    mocker.patch.object(client, "_post", side_effect=RuntimeError("API error 401: unauthorized"))
    with pytest.raises(RuntimeError, match="401"):
        client.get_part_detail("C00000")


# ---------------------------------------------------------------------------
# get_parts_details (batched)
# ---------------------------------------------------------------------------

def test_get_parts_details_empty(client, mocker):
    """Empty input → no API call, empty result."""
    mock_post = mocker.patch.object(client, "_post")
    assert client.get_parts_details([]) == []
    mock_post.assert_not_called()


def test_get_parts_details_single_batch(client, mocker):
    mock_post = mocker.patch.object(
        client, "_post",
        return_value={"componentDetailResponseVOList": [_SAMPLE_DETAIL]},
    )
    result = client.get_parts_details(["C25744"])
    assert len(result) == 1
    assert result[0]["lcscPart"] == "C25744"
    mock_post.assert_called_once()


def test_get_parts_details_chunks_above_limit(client, mocker):
    """A list of > DETAIL_BATCH_MAX codes is split into multiple API calls."""
    codes = [f"C{i}" for i in range(DETAIL_BATCH_MAX + 5)]
    captured: list[list[str]] = []

    def _fake_post(_endpoint, payload, **_kwargs):
        captured.append(list(payload["componentCodes"]))
        return {"componentDetailResponseVOList": [
            {**_SAMPLE_DETAIL, "componentCode": c} for c in payload["componentCodes"]
        ]}

    mocker.patch.object(client, "_post", side_effect=_fake_post)
    result = client.get_parts_details(codes)

    assert len(result) == DETAIL_BATCH_MAX + 5
    assert len(captured) == 2
    assert len(captured[0]) == DETAIL_BATCH_MAX
    assert len(captured[1]) == 5


def test_get_parts_details_handles_list_response(client, mocker):
    """Some endpoint variants return data as a bare list."""
    mocker.patch.object(client, "_post", return_value=[_SAMPLE_DETAIL])
    result = client.get_parts_details(["C25744"])
    assert len(result) == 1


def test_get_parts_details_handles_unexpected_response(client, mocker):
    """A dict without the expected key returns an empty list, no exception."""
    mocker.patch.object(client, "_post", return_value={"unexpected": "shape"})
    assert client.get_parts_details(["C25744"]) == []


# ---------------------------------------------------------------------------
# get_library_list
# ---------------------------------------------------------------------------

def test_get_library_list_no_filter(client, mocker):
    mock_post = mocker.patch.object(client, "_post", return_value={})
    client.get_library_list()
    _, payload = mock_post.call_args[0]
    assert "libraryType" not in payload
    assert "lastKey" not in payload


def test_get_library_list_with_last_key(client, mocker):
    mock_post = mocker.patch.object(client, "_post", return_value={})
    client.get_library_list(last_key="cursor123", page_size=500)
    _, payload = mock_post.call_args[0]
    assert payload["lastKey"] == "cursor123"
    assert payload["pageSize"] == 500
    assert "libraryType" not in payload
    assert "currentPage" not in payload


def test_get_library_list_non_dict_response(client, mocker):
    """Non-dict response (e.g. None) → returns ([], None)."""
    mocker.patch.object(client, "_post", return_value=None)
    assert client.get_library_list() == ([], None)


# ---------------------------------------------------------------------------
# iter_library_stubs
# ---------------------------------------------------------------------------

def _stub(code: str) -> dict:
    return {"componentCode": code, "componentModel": code, "componentSpecification": "0402"}


def test_iter_library_stubs_paginates(client, mocker):
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    pages = [
        ([_stub("C1"), _stub("C2")], "cursor_a"),
        ([_stub("C3")], None),
    ]
    mocker.patch.object(client, "get_library_list", side_effect=pages)
    result = list(client.iter_library_stubs())
    assert [s["componentCode"] for s in result] == ["C1", "C2", "C3"]


def test_iter_library_stubs_stops_on_empty_page(client, mocker):
    """Empty first page → generator exits cleanly."""
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    mocker.patch.object(client, "get_library_list", return_value=([], None))
    assert list(client.iter_library_stubs()) == []


def test_iter_library_stubs_invokes_progress(client, mocker):
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    pages = [
        ([_stub("C1"), _stub("C2")], "cursor_a"),
        ([_stub("C3")], None),
    ]
    mocker.patch.object(client, "get_library_list", side_effect=pages)
    progress: list[int] = []
    list(client.iter_library_stubs(on_progress=progress.append))
    assert progress == [2, 3]


def test_iter_library_stubs_passes_cursor(client, mocker):
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    mock_get = mocker.patch.object(
        client, "get_library_list",
        side_effect=[
            ([_stub("C1")], "next_key"),
            ([_stub("C2")], None),
        ],
    )
    list(client.iter_library_stubs(page_size=42))
    # Two calls: first with last_key=None, second with last_key="next_key"
    assert mock_get.call_count == 2
    assert mock_get.call_args_list[0].kwargs == {"last_key": None, "page_size": 42}
    assert mock_get.call_args_list[1].kwargs == {"last_key": "next_key", "page_size": 42}


# ---------------------------------------------------------------------------
# download_library
# ---------------------------------------------------------------------------

def _make_stub(code: str) -> dict:
    return {"componentCode": code, "componentModel": code, "componentSpecification": "SMD"}


def _make_detail(code: str) -> dict:
    return {"componentCode": code, "componentModel": code, "componentSpecification": "SMD",
            "firstTypeName": "Resistors", "secondTypeName": "Chip", "libraryType": "base",
            "description": "", "datasheetUrl": "", "solderJointCount": 2,
            "priceRanges": [{"startQuantity": 1, "endQuantity": 9, "unitPrice": 0.01}],
            "stockCount": 100}


def test_download_library_accumulates(client, mocker):
    """Non-null lastKey triggers page 2; null lastKey ends loop."""
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    page1_stubs = [_make_stub(f"C{i}") for i in range(1000)]
    page2_stubs = [_make_stub("C9999")]
    mocker.patch.object(client, "get_library_list", side_effect=[
        (page1_stubs, "cursor_key"),
        (page2_stubs, None),
    ])
    mocker.patch.object(client, "_post", side_effect=[
        {"componentDetailResponseVOList": [_make_detail(s["componentCode"]) for s in page1_stubs]},
        {"componentDetailResponseVOList": [_make_detail(s["componentCode"]) for s in page2_stubs]},
    ])
    result = client.download_library()
    assert len(result) == 1001


def test_download_library_with_callback(client, mocker):
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    stubs = [_make_stub("C1")]
    mocker.patch.object(client, "get_library_list", side_effect=[(stubs, None)])
    mocker.patch.object(client, "_post", return_value={"componentDetailResponseVOList": [_make_detail("C1")]})
    batches = []
    result = client.download_library(on_batch=lambda p: batches.append(p))
    assert result == []
    assert len(batches) == 1


def test_download_library_progress(client, mocker):
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    stubs = [_make_stub("C1")]
    mocker.patch.object(client, "get_library_list", side_effect=[(stubs, None)])
    mocker.patch.object(client, "_post", return_value={"componentDetailResponseVOList": [_make_detail("C1")]})
    calls = []
    client.download_library(on_progress=lambda t, m: calls.append(t))
    assert calls == [1]


def test_download_library_logs_every_50_pages(client, mocker, caplog):
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    pages = [([_make_stub(f"C{i}")], f"k{i}") for i in range(1, 100)] + [([_make_stub("C100")], None)]
    mocker.patch.object(client, "get_library_list", side_effect=pages)
    mocker.patch.object(client, "get_parts_details", side_effect=lambda codes: [{"lcscPart": c} for c in codes])
    with caplog.at_level("INFO", logger="jlcpcb_mcp.client"):
        client.download_library(on_batch=lambda parts: None)
    logged = [r.getMessage() for r in caplog.records if "Library page" in r.getMessage()]
    assert logged == ["Library page 50: 50 parts", "Library page 100: 100 parts"]


def test_download_library_empty_first_page(client, mocker):
    """Empty first page → returns nothing without calling detail endpoint."""
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    mocker.patch.object(client, "get_library_list", return_value=([], None))
    post_mock = mocker.patch.object(client, "_post")
    result = client.download_library()
    assert result == []
    post_mock.assert_not_called()


def test_download_library_non_list_detail_response(client, mocker):
    """_post returns dict without known key for detail endpoint → parts = [], no import."""
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    stubs = [_make_stub("C1")]
    mocker.patch.object(client, "get_library_list", side_effect=[(stubs, None)])
    mocker.patch.object(client, "_post", return_value={"unexpected": "dict"})
    result = client.download_library()
    assert result == []


def test_download_library_detail_enrichment(client, mocker):
    """Verifies codes are passed to ENDPOINT_DETAIL and results are normalized."""
    mocker.patch("jlcpcb_mcp.client.time.sleep")
    stubs = [_make_stub("C580905"), _make_stub("C110499")]
    mocker.patch.object(client, "get_library_list", side_effect=[(stubs, None)])
    detail_raw = [_make_detail("C580905"), _make_detail("C110499")]
    post_mock = mocker.patch.object(client, "_post",
                                    return_value={"componentDetailResponseVOList": detail_raw})
    result = client.download_library()
    assert len(result) == 2
    codes_sent = post_mock.call_args[0][1]["componentCodes"]
    assert set(codes_sent) == {"C580905", "C110499"}
