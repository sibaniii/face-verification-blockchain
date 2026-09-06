"""
Tests for src.search_stage.

Fully offline and deterministic:
    - all network calls (requests.post / requests.get) are mocked
    - no real browser is ever opened (browser_open is always a fake)
    - no real terminal input is read (prompt is always a fake)
    - no real SerpApi search is consumed by running this suite

These tests use small temp files created via pytest's tmp_path fixture
rather than depending on a real sample image, so the whole suite runs
identically on any machine.
"""

import os
import json
from unittest.mock import patch, MagicMock

import pytest

from src.search_stage import (
    search_for_match,
    validate_image_path,
    validate_source_url,
    parse_candidates,
    upload_image_to_serpapi,
    run_google_lens_search,
    load_api_key,
    select_candidate_interactive,
    manual_google_lens_fallback,
    ImageNotFoundError,
    ImageLoadError,
    InvalidURLError,
    SerpApiRequestError,
)

MOCK_LENS_RESPONSE = {
    "search_metadata": {"status": "Success"},
    "visual_matches": [
        {
            "position": 1,
            "title": "Example public post",
            "link": "https://example-social.com/user/post/123",
            "source": "Example Social",
            "thumbnail": "https://example.com/thumb1.jpg",
        },
        {
            "position": 2,
            "title": "Another matching page",
            "link": "https://blog.example.org/photo-feature",
            "thumbnail": "https://example.com/thumb2.jpg",
            # "source" deliberately missing -> parser must fall back to domain
        },
        {
            "position": 3,
            "title": "Thumbnail only, no source page",
            # "link" deliberately missing -> parser must skip this entry
        },
    ],
}


# ---------------------------------------------------------------------
# 1-2: image path validation
# ---------------------------------------------------------------------

def test_valid_image_passes(tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"x" * 1024)
    assert validate_image_path(str(image)) == str(image)


def test_missing_image_raises(tmp_path):
    with pytest.raises(ImageNotFoundError):
        validate_image_path(str(tmp_path / "does_not_exist.jpg"))


def test_empty_image_rejected(tmp_path):
    image = tmp_path / "empty.jpg"
    image.write_bytes(b"")
    with pytest.raises(ImageLoadError):
        validate_image_path(str(image))


def test_oversized_image_rejected(tmp_path):
    image = tmp_path / "big.jpg"
    image.write_bytes(b"0" * (600 * 1024))  # over the 500KB SerpApi limit
    with pytest.raises(ImageLoadError):
        validate_image_path(str(image))


# ---------------------------------------------------------------------
# 3: missing SERPAPI_KEY
# ---------------------------------------------------------------------

def test_missing_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    monkeypatch.setattr("src.search_stage.load_dotenv", lambda *a, **k: None)
    assert load_api_key() is None


def test_present_api_key_returned(monkeypatch):
    monkeypatch.setenv("SERPAPI_KEY", "fake_test_key_123")
    monkeypatch.setattr("src.search_stage.load_dotenv", lambda *a, **k: None)
    assert load_api_key() == "fake_test_key_123"


# ---------------------------------------------------------------------
# 4-7: SerpApi calls and candidate parsing
# ---------------------------------------------------------------------

def test_upload_image_success(tmp_path):
    image = tmp_path / "test.jpg"
    image.write_bytes(b"fake image bytes")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"image_id": "abc123"}

    with patch("src.search_stage.requests.post", return_value=mock_response) as mock_post:
        image_id = upload_image_to_serpapi(str(image), "fake_key")

    assert image_id == "abc123"
    _, kwargs = mock_post.call_args
    assert kwargs["data"]["api_key"] == "fake_key"


def test_run_google_lens_search_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_LENS_RESPONSE

    with patch("src.search_stage.requests.get", return_value=mock_response):
        result = run_google_lens_search("abc123", "fake_key")

    assert result == MOCK_LENS_RESPONSE


def test_parse_candidates_extracts_valid_links():
    candidates = parse_candidates(MOCK_LENS_RESPONSE)

    assert len(candidates) == 2  # entry #3 has no link and must be skipped
    assert candidates[0]["url"] == "https://example-social.com/user/post/123"
    assert candidates[0]["source"] == "Example Social"
    assert candidates[1]["url"] == "https://blog.example.org/photo-feature"
    assert candidates[1]["source"] == "blog.example.org"  # fallback to domain


def test_parse_candidates_handles_no_visual_matches():
    assert parse_candidates({"search_metadata": {"status": "Success"}}) == []


def test_parse_candidates_handles_malformed_response():
    assert parse_candidates("not even a dict") == []
    assert parse_candidates(None) == []
    assert parse_candidates({"visual_matches": "not a list"}) == []
    assert parse_candidates({"visual_matches": [1, 2, "junk"]}) == []


# ---------------------------------------------------------------------
# 8: candidate selection validation
# ---------------------------------------------------------------------

def test_select_candidate_valid_choice():
    candidates = parse_candidates(MOCK_LENS_RESPONSE)
    answers = iter(["1", "y"])
    selected = select_candidate_interactive(candidates, prompt=lambda _: next(answers))
    assert selected["url"] == candidates[0]["url"]


def test_select_candidate_zero_returns_none():
    candidates = parse_candidates(MOCK_LENS_RESPONSE)
    answers = iter(["0"])
    selected = select_candidate_interactive(candidates, prompt=lambda _: next(answers))
    assert selected is None


def test_select_candidate_out_of_range_then_valid():
    candidates = parse_candidates(MOCK_LENS_RESPONSE)
    answers = iter(["99", "2", "y"])
    selected = select_candidate_interactive(candidates, prompt=lambda _: next(answers))
    assert selected["url"] == candidates[1]["url"]


def test_select_candidate_rejects_non_numeric_input():
    candidates = parse_candidates(MOCK_LENS_RESPONSE)
    answers = iter(["not-a-number", "1", "y"])
    selected = select_candidate_interactive(candidates, prompt=lambda _: next(answers))
    assert selected["url"] == candidates[0]["url"]


# ---------------------------------------------------------------------
# 9-12: URL validation
# ---------------------------------------------------------------------

def test_valid_source_url_accepted():
    assert (
        validate_source_url("https://example.com/post/123")
        == "https://example.com/post/123"
    )


def test_empty_url_rejected():
    with pytest.raises(InvalidURLError):
        validate_source_url("")


def test_malformed_url_rejected():
    with pytest.raises(InvalidURLError):
        validate_source_url("http://")


def test_google_search_url_rejected():
    with pytest.raises(InvalidURLError):
        validate_source_url("https://www.google.com/search?q=test")


def test_bing_search_url_rejected():
    with pytest.raises(InvalidURLError):
        validate_source_url("https://www.bing.com/images/search?q=test")


def test_tineye_url_rejected():
    with pytest.raises(InvalidURLError):
        validate_source_url("https://tineye.com/search/abc123")


# ---------------------------------------------------------------------
# 13: API / network error handling
# ---------------------------------------------------------------------

def test_upload_image_http_error(tmp_path):
    image = tmp_path / "test.jpg"
    image.write_bytes(b"fake image bytes")

    mock_response = MagicMock()
    mock_response.status_code = 500

    with patch("src.search_stage.requests.post", return_value=mock_response):
        with pytest.raises(SerpApiRequestError):
            upload_image_to_serpapi(str(image), "fake_key")


def test_run_google_lens_search_api_error_field():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"error": "Invalid API key."}

    with patch("src.search_stage.requests.get", return_value=mock_response):
        with pytest.raises(SerpApiRequestError):
            run_google_lens_search("abc123", "fake_key")


def test_upload_image_network_error_does_not_leak_key(tmp_path):
    import requests as real_requests

    image = tmp_path / "test.jpg"
    image.write_bytes(b"fake image bytes")

    def raise_conn_error(*args, **kwargs):
        raise real_requests.exceptions.ConnectionError(
            "Connection refused (fake_key_should_not_leak)"
        )

    with patch("src.search_stage.requests.post", side_effect=raise_conn_error):
        with pytest.raises(SerpApiRequestError) as exc_info:
            upload_image_to_serpapi(str(image), "fake_key_should_not_leak")

    assert "fake_key_should_not_leak" not in str(exc_info.value)


# ---------------------------------------------------------------------
# 14: API key is never saved in the result JSON
# ---------------------------------------------------------------------

def test_api_key_never_saved_in_result(tmp_path):
    output_path = tmp_path / "search_result.json"
    image = tmp_path / "q.jpg"
    image.write_bytes(b"x" * 1024)

    mock_upload_response = MagicMock()
    mock_upload_response.status_code = 200
    mock_upload_response.json.return_value = {"image_id": "abc123"}

    mock_search_response = MagicMock()
    mock_search_response.status_code = 200
    mock_search_response.json.return_value = MOCK_LENS_RESPONSE

    answers = iter(["1", "y"])

    with patch("src.search_stage.requests.post", return_value=mock_upload_response), \
         patch("src.search_stage.requests.get", return_value=mock_search_response):
        result = search_for_match(
            str(image),
            api_key="super_secret_key_xyz",
            browser_open=lambda _: True,
            prompt=lambda _: next(answers),
            output_path=str(output_path),
        )

    assert "super_secret_key_xyz" not in json.dumps(result)
    assert "super_secret_key_xyz" not in output_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------
# 15: browser fallback can be mocked, never opens a real browser
# ---------------------------------------------------------------------

def test_manual_google_lens_fallback_never_opens_real_browser():
    opened_urls = []
    result = manual_google_lens_fallback(
        browser_open=lambda url: opened_urls.append(url),
        prompt=lambda _: "https://example.com/real-source-page",
    )
    assert opened_urls == ["https://lens.google.com/"]
    assert result["url"] == "https://example.com/real-source-page"


# ---------------------------------------------------------------------
# End-to-end orchestration, fully mocked
# ---------------------------------------------------------------------

def test_search_for_match_full_success_flow(tmp_path):
    output_path = tmp_path / "search_result.json"
    image = tmp_path / "q.jpg"
    image.write_bytes(b"x" * 1024)

    mock_upload_response = MagicMock()
    mock_upload_response.status_code = 200
    mock_upload_response.json.return_value = {"image_id": "abc123"}

    mock_search_response = MagicMock()
    mock_search_response.status_code = 200
    mock_search_response.json.return_value = MOCK_LENS_RESPONSE

    answers = iter(["1", "y"])

    with patch("src.search_stage.requests.post", return_value=mock_upload_response), \
         patch("src.search_stage.requests.get", return_value=mock_search_response):
        result = search_for_match(
            str(image),
            api_key="fake_key",
            browser_open=lambda _: True,
            prompt=lambda _: next(answers),
            output_path=str(output_path),
        )

    assert result["search_engine"] == "serpapi_google_lens"
    assert result["search_method"] == "api"
    assert result["selected_candidate"]["url"] == "https://example-social.com/user/post/123"
    assert output_path.is_file()

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["selected_candidate"]["url"] == result["selected_candidate"]["url"]


def test_search_for_match_missing_key_routes_to_manual_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    monkeypatch.setattr("src.search_stage.load_dotenv", lambda *a, **k: None)

    output_path = tmp_path / "search_result.json"
    image = tmp_path / "q.jpg"
    image.write_bytes(b"x" * 1024)

    opened_urls = []
    answers = iter(["1", "https://example.com/manual-source-page"])

    result = search_for_match(
        str(image),
        api_key=None,
        browser_open=lambda url: opened_urls.append(url),
        prompt=lambda _: next(answers),
        output_path=str(output_path),
    )

    assert result["search_engine"] == "google_lens_manual"
    assert result["search_method"] == "human_in_the_loop"
    assert result["selected_candidate"]["url"] == "https://example.com/manual-source-page"
    assert opened_urls == ["https://lens.google.com/"]


def test_search_for_match_api_failure_routes_to_bing_fallback(tmp_path):
    output_path = tmp_path / "search_result.json"
    image = tmp_path / "q.jpg"
    image.write_bytes(b"x" * 1024)

    mock_upload_response = MagicMock()
    mock_upload_response.status_code = 500  # SerpApi fails

    opened_urls = []
    answers = iter(["2", "https://example.com/found-on-bing"])

    with patch("src.search_stage.requests.post", return_value=mock_upload_response):
        result = search_for_match(
            str(image),
            api_key="fake_key",
            browser_open=lambda url: opened_urls.append(url),
            prompt=lambda _: next(answers),
            output_path=str(output_path),
        )

    assert result["search_engine"] == "bing_visual_search_manual"
    assert result["selected_candidate"]["url"] == "https://example.com/found-on-bing"
    assert opened_urls == ["https://www.bing.com/visualsearch"]


def test_search_for_match_exit_with_no_candidate_returns_none(tmp_path):
    output_path = tmp_path / "search_result.json"
    image = tmp_path / "q.jpg"
    image.write_bytes(b"x" * 1024)

    answers = iter(["3"])  # operator chooses "Exit" at the fallback menu

    result = search_for_match(
        str(image),
        api_key=None,
        browser_open=lambda _: True,
        prompt=lambda _: next(answers),
        output_path=str(output_path),
    )

    assert result is None
    assert not output_path.exists()
