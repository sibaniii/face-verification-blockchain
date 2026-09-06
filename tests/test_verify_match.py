"""
Tests for src.verify_match.

Fully offline and deterministic:
    - no real HTTP requests (fetch_webpage_fn / download_image_fn are
      always fakes injected into verify_match(), or requests.get is
      patched directly for the lower-level fetch/download functions)
    - no real face_recognition model calls (load_image_file /
      face_locations / face_encodings are patched with fakes; only the
      real face_distance -- pure numpy, no model -- is used, so
      distance math is genuinely exercised)
    - process_face (Step 1) is injected as a fake for the query image,
      so these tests don't depend on a real photo or a working dlib
      install either
"""

import os
import json
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.verify_match import (
    verify_match,
    load_search_result,
    extract_image_urls,
    encode_all_faces,
    best_face_distance,
    FACE_DISTANCE_THRESHOLD,
    MAX_CANDIDATE_IMAGES,
    SearchResultMissingError,
    CandidateMissingError,
    WebpageFetchError,
)

QUERY_ENCODING = np.zeros(128)
MATCH_ENCODING = np.zeros(128)          # distance 0.0 from the query -> match
FAR_ENCODING = np.ones(128) * 10.0      # very large distance -> no match

BASE_URL = "https://example.com/team/"


def fake_process_face(_image_path):
    """Stands in for Step 1's process_face() for the query image."""
    return {
        "image_path": _image_path,
        "face_location": (0, 10, 10, 0),
        "face_encoding": QUERY_ENCODING,
        "face_count": 1,
    }


def write_search_result(tmp_path, url="https://example.com/team/", extra=None):
    path = tmp_path / "search_result.json"
    payload = {
        "query_image": "data/input/single_face.jpg",
        "search_engine": "serpapi_google_lens",
        "search_method": "api",
        "selected_candidate": {"title": "Team page", "source": "example.com", "url": url},
        "all_candidates": [],
        "search_completed": True,
        "timestamp_utc": "2026-09-05T10:00:00+00:00",
    }
    if extra is not None:
        payload = extra
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------
# 1-3: search_result.json loading
# ---------------------------------------------------------------------

def test_missing_search_result_file_raises(tmp_path):
    with pytest.raises(SearchResultMissingError):
        load_search_result(str(tmp_path / "does_not_exist.json"))


def test_malformed_search_result_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(SearchResultMissingError):
        load_search_result(str(path))


def test_missing_selected_candidate_raises(tmp_path):
    path = write_search_result(tmp_path, extra={"search_completed": True})
    with pytest.raises(CandidateMissingError):
        load_search_result(path)


def test_selected_candidate_missing_url_raises(tmp_path):
    path = write_search_result(
        tmp_path, extra={"selected_candidate": {"title": "no url here"}}
    )
    with pytest.raises(CandidateMissingError):
        load_search_result(path)


def test_valid_search_result_returns_url(tmp_path):
    path = write_search_result(tmp_path, url="https://real-example.com/page")
    assert load_search_result(path) == "https://real-example.com/page"


# ---------------------------------------------------------------------
# 4-9: webpage image extraction
# ---------------------------------------------------------------------

def test_extract_image_urls_from_meta_and_img_tags():
    html = """
    <html><head>
      <meta property="og:image" content="/images/og.jpg">
      <meta name="twitter:image" content="https://cdn.example.com/twitter.jpg">
    </head><body>
      <img src="/images/photo1.jpg">
      <img data-src="https://cdn.example.com/photo2.jpg">
    </body></html>
    """
    urls = extract_image_urls(html, BASE_URL)
    assert "https://example.com/images/og.jpg" in urls
    assert "https://cdn.example.com/twitter.jpg" in urls
    assert "https://example.com/images/photo1.jpg" in urls
    assert "https://cdn.example.com/photo2.jpg" in urls


def test_extract_image_urls_resolves_relative_urls():
    html = '<html><body><img src="photo.jpg"></body></html>'
    urls = extract_image_urls(html, "https://example.com/team/")
    assert urls == ["https://example.com/team/photo.jpg"]


def test_extract_image_urls_deduplicates():
    html = """
    <html><body>
      <img src="https://cdn.example.com/a.jpg">
      <img src="https://cdn.example.com/a.jpg">
      <img data-src="https://cdn.example.com/a.jpg">
    </body></html>
    """
    urls = extract_image_urls(html, BASE_URL)
    assert urls.count("https://cdn.example.com/a.jpg") == 1


def test_extract_image_urls_rejects_dangerous_schemes():
    html = """
    <html><body>
      <img src="data:image/png;base64,AAAA">
      <img src="javascript:alert(1)">
      <img src="file:///etc/passwd">
      <img src="https://cdn.example.com/real.jpg">
    </body></html>
    """
    urls = extract_image_urls(html, BASE_URL)
    assert urls == ["https://cdn.example.com/real.jpg"]


def test_extract_image_urls_empty_page_returns_empty_list():
    assert extract_image_urls("<html><body></body></html>", BASE_URL) == []


def test_extract_image_urls_respects_max_candidate_images():
    imgs = "".join(f'<img src="https://cdn.example.com/img{i}.jpg">' for i in range(50))
    html = f"<html><body>{imgs}</body></html>"
    urls = extract_image_urls(html, BASE_URL)
    assert len(urls) == MAX_CANDIDATE_IMAGES


def test_extract_image_urls_data_lazy_src_and_data_original():
    html = """
    <html><body>
      <img data-lazy-src="https://cdn.example.com/lazy.jpg">
      <img data-original="https://cdn.example.com/original.jpg">
    </body></html>
    """
    urls = extract_image_urls(html, BASE_URL)
    assert "https://cdn.example.com/lazy.jpg" in urls
    assert "https://cdn.example.com/original.jpg" in urls


def test_extract_image_urls_srcset_and_data_srcset():
    html = """
    <html><body>
      <img srcset="https://cdn.example.com/small.jpg 480w, https://cdn.example.com/large.jpg 1080w">
      <img data-srcset="https://cdn.example.com/lazy-small.jpg 1x, https://cdn.example.com/lazy-large.jpg 2x">
    </body></html>
    """
    urls = extract_image_urls(html, BASE_URL)
    assert "https://cdn.example.com/small.jpg" in urls
    assert "https://cdn.example.com/large.jpg" in urls
    assert "https://cdn.example.com/lazy-small.jpg" in urls
    assert "https://cdn.example.com/lazy-large.jpg" in urls


def test_extract_image_urls_picture_source():
    html = """
    <html><body>
      <picture>
        <source srcset="https://cdn.example.com/webp-version.webp" type="image/webp">
        <source data-srcset="https://cdn.example.com/lazy-fallback.jpg">
        <img src="https://cdn.example.com/fallback.jpg">
      </picture>
    </body></html>
    """
    urls = extract_image_urls(html, BASE_URL)
    assert "https://cdn.example.com/webp-version.webp" in urls
    assert "https://cdn.example.com/lazy-fallback.jpg" in urls
    assert "https://cdn.example.com/fallback.jpg" in urls


def test_extract_image_urls_inline_style_background_image():
    html = """
    <html><body>
      <div class="card" style="background-image: url('https://cdn.example.com/bg-inline.jpg');"></div>
    </body></html>
    """
    urls = extract_image_urls(html, BASE_URL)
    assert "https://cdn.example.com/bg-inline.jpg" in urls


def test_extract_image_urls_style_block_background_image():
    html = """
    <html><head>
      <style>.hero { background-image: url(https://cdn.example.com/bg-style-block.jpg); }</style>
    </head><body></body></html>
    """
    urls = extract_image_urls(html, BASE_URL)
    assert "https://cdn.example.com/bg-style-block.jpg" in urls


def test_extract_image_urls_data_bg_attributes():
    html = """
    <html><body>
      <div class="slide" data-bg="https://cdn.example.com/slide1.jpg"></div>
      <div class="slide" data-background="https://cdn.example.com/slide2.jpg"></div>
      <div class="slide" data-background-image="https://cdn.example.com/slide3.jpg"></div>
    </body></html>
    """
    urls = extract_image_urls(html, BASE_URL)
    assert "https://cdn.example.com/slide1.jpg" in urls
    assert "https://cdn.example.com/slide2.jpg" in urls
    assert "https://cdn.example.com/slide3.jpg" in urls


def test_extract_image_urls_script_embedded_json_fallback():
    html = """
    <html><body>
      <script>
        var carouselData = {"slides": [{"photo": "https://cdn.example.com/from-json.jpg"}]};
      </script>
    </body></html>
    """
    urls = extract_image_urls(html, BASE_URL)
    assert "https://cdn.example.com/from-json.jpg" in urls


def test_extract_image_urls_excludes_svg_but_keeps_generically_named_files():
    html = """
    <html><body>
      <img src="https://cdn.example.com/icons/arrow.svg">
      <img src="https://cdn.example.com/logo.png">
    </body></html>
    """
    urls = extract_image_urls(html, BASE_URL)
    # SVGs are structurally never a photo -> excluded.
    assert not any(u.endswith(".svg") for u in urls)
    # But we deliberately do NOT filter by filename (e.g. "logo") --
    # a real photo could have a generic name, so this must stay in and
    # let face detection (not filename guessing) decide its fate.
    assert "https://cdn.example.com/logo.png" in urls


def test_extract_image_urls_malformed_url_ignored():
    html = """
    <html><body>
      <img src="   ">
      <img src="mailto:someone@example.com">
      <img src="https://cdn.example.com/valid.jpg">
    </body></html>
    """
    urls = extract_image_urls(html, BASE_URL)
    assert urls == ["https://cdn.example.com/valid.jpg"]


def test_extract_image_urls_realistic_carousel_page_multi_mechanism():
    """A single page combining several mechanisms at once (lazy-load
    <img>, srcset, a background-image slide, and a logo via og:image),
    mirroring a real WordPress/page-builder team carousel."""
    html = """
    <html><head>
      <meta property="og:image" content="https://cdn.example.com/logo.png">
    </head><body>
      <div class="carousel">
        <div class="slide" data-bg="https://cdn.example.com/team1.jpg"></div>
        <img data-lazy-src="https://cdn.example.com/team2.jpg" data-srcset="https://cdn.example.com/team2-2x.jpg 2x">
        <picture><source srcset="https://cdn.example.com/team3.jpg"></picture>
      </div>
    </body></html>
    """
    urls = extract_image_urls(html, BASE_URL)
    assert "https://cdn.example.com/logo.png" in urls
    assert "https://cdn.example.com/team1.jpg" in urls
    assert "https://cdn.example.com/team2.jpg" in urls
    assert "https://cdn.example.com/team2-2x.jpg" in urls
    assert "https://cdn.example.com/team3.jpg" in urls
    assert len(urls) == 5  # nothing lost, nothing duplicated


# ---------------------------------------------------------------------
# 10-13: face detection/comparison on candidate images
# (face_locations / face_encodings mocked; face_distance is the real,
#  pure-numpy implementation)
# ---------------------------------------------------------------------

def test_encode_all_faces_no_face():
    with patch("src.verify_match.face_recognition.face_locations", return_value=[]):
        assert encode_all_faces("fake_image_array") == []


def test_encode_all_faces_one_face():
    with patch("src.verify_match.face_recognition.face_locations", return_value=[(0, 10, 10, 0)]), \
         patch("src.verify_match.face_recognition.face_encodings", return_value=[MATCH_ENCODING]):
        faces = encode_all_faces("fake_image_array")
    assert len(faces) == 1
    assert faces[0][0] == (0, 10, 10, 0)


def test_encode_all_faces_multiple_faces():
    locations = [(0, 10, 10, 0), (20, 30, 30, 20), (40, 50, 50, 40)]
    encodings = [FAR_ENCODING, FAR_ENCODING, MATCH_ENCODING]
    with patch("src.verify_match.face_recognition.face_locations", return_value=locations), \
         patch("src.verify_match.face_recognition.face_encodings", return_value=encodings):
        faces = encode_all_faces("fake_image_array")
    assert len(faces) == 3


def test_best_face_distance_picks_smallest():
    distance = best_face_distance(QUERY_ENCODING, [FAR_ENCODING, FAR_ENCODING, MATCH_ENCODING])
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_best_face_distance_empty_returns_none():
    assert best_face_distance(QUERY_ENCODING, []) is None


# ---------------------------------------------------------------------
# Full orchestration (fully mocked): matching, non-matching, mixed
# ---------------------------------------------------------------------

URL_NO_FACE = "https://cdn.example.com/no_face.jpg"
URL_ONE_FACE_NO_MATCH = "https://cdn.example.com/one_face_no_match.jpg"
URL_MULTI_FACE_WITH_MATCH = "https://cdn.example.com/group_photo.jpg"
URL_DOWNLOAD_FAILS = "https://cdn.example.com/unreachable.jpg"
URL_CORRUPT = "https://cdn.example.com/corrupt.jpg"

MARKERS = {
    URL_NO_FACE: b"MARK_NO_FACE",
    URL_ONE_FACE_NO_MATCH: b"MARK_ONE_NO_MATCH",
    URL_MULTI_FACE_WITH_MATCH: b"MARK_MULTI_MATCH",
    URL_DOWNLOAD_FAILS: None,  # simulates a failed download
    URL_CORRUPT: b"MARK_CORRUPT",
}

FACE_DB = {
    b"MARK_NO_FACE": ([], []),
    b"MARK_ONE_NO_MATCH": ([(0, 10, 10, 0)], [FAR_ENCODING]),
    b"MARK_MULTI_MATCH": (
        [(0, 10, 10, 0), (20, 30, 30, 20), (40, 50, 50, 40)],
        [FAR_ENCODING, FAR_ENCODING, MATCH_ENCODING],
    ),
}


def fake_download(url):
    return MARKERS.get(url)


def fake_load_image_file(file_obj):
    data = file_obj.getvalue()
    if data == b"MARK_CORRUPT":
        raise OSError("cannot identify image file (simulated corrupt image)")
    return data  # identity: the "array" is just the marker bytes


def fake_face_locations(image_array):
    return FACE_DB[image_array][0]


def fake_face_encodings(image_array, known_face_locations=None):
    return FACE_DB[image_array][1]


def _html_for(urls):
    imgs = "".join(f'<img src="{u}">' for u in urls)
    return f"<html><body>{imgs}</body></html>"


def _run_verify(tmp_path, urls, threshold=FACE_DISTANCE_THRESHOLD):
    search_result_path = write_search_result(tmp_path, url=BASE_URL)
    output_path = tmp_path / "verification_result.json"
    images_dir = tmp_path / "verification_images"

    html = _html_for(urls)

    with patch("src.verify_match.face_recognition.load_image_file", side_effect=fake_load_image_file), \
         patch("src.verify_match.face_recognition.face_locations", side_effect=fake_face_locations), \
         patch("src.verify_match.face_recognition.face_encodings", side_effect=fake_face_encodings):
        result = verify_match(
            "data/input/single_face.jpg",
            search_result_path=search_result_path,
            output_path=str(output_path),
            images_dir=str(images_dir),
            threshold=threshold,
            process_face_fn=fake_process_face,
            fetch_webpage_fn=lambda _url: html,
            download_image_fn=fake_download,
        )
    return result, str(output_path)


def test_verify_match_finds_match_in_group_photo(tmp_path):
    result, output_path = _run_verify(
        tmp_path, [URL_NO_FACE, URL_ONE_FACE_NO_MATCH, URL_MULTI_FACE_WITH_MATCH]
    )

    assert result["match_found"] is True
    assert result["best_match"]["image_url"] == URL_MULTI_FACE_WITH_MATCH
    assert result["best_match"]["face_distance"] == pytest.approx(0.0, abs=1e-4)
    assert result["images_discovered"] == 3
    assert result["images_analyzed"] == 3  # all three downloaded+decoded fine
    assert os.path.isfile(output_path)


def test_verify_match_no_match_when_nothing_close_enough(tmp_path):
    result, _ = _run_verify(tmp_path, [URL_NO_FACE, URL_ONE_FACE_NO_MATCH])

    assert result["match_found"] is False
    assert result["best_match"] is not None  # still reports the closest attempt
    assert result["best_match"]["face_distance"] > FACE_DISTANCE_THRESHOLD


def test_verify_match_all_no_face_returns_null_best_match(tmp_path):
    result, _ = _run_verify(tmp_path, [URL_NO_FACE])

    assert result["match_found"] is False
    assert result["best_match"] is None


def test_verify_match_threshold_boundary_counts_as_match(tmp_path):
    # Distance is exactly 0.0 (MATCH_ENCODING == QUERY_ENCODING), and
    # threshold=0.0 -> boundary case must still count as a match (<=).
    result, _ = _run_verify(tmp_path, [URL_MULTI_FACE_WITH_MATCH], threshold=0.0)
    assert result["match_found"] is True


def test_verify_match_download_failure_does_not_crash_run(tmp_path):
    result, _ = _run_verify(
        tmp_path, [URL_DOWNLOAD_FAILS, URL_MULTI_FACE_WITH_MATCH]
    )
    assert result["images_discovered"] == 2
    assert result["images_analyzed"] == 1  # the unreachable one was skipped
    assert result["match_found"] is True


def test_verify_match_invalid_image_ignored(tmp_path):
    result, _ = _run_verify(
        tmp_path, [URL_CORRUPT, URL_MULTI_FACE_WITH_MATCH]
    )
    assert result["images_discovered"] == 2
    assert result["images_analyzed"] == 1  # corrupt one was skipped
    assert result["match_found"] is True


# ---------------------------------------------------------------------
# Webpage fetch failure (hard stop, but a controlled/catchable one)
# ---------------------------------------------------------------------

def test_verify_match_webpage_fetch_error_propagates_cleanly(tmp_path):
    search_result_path = write_search_result(tmp_path, url=BASE_URL)

    def failing_fetch(_url):
        raise WebpageFetchError("Candidate webpage returned HTTP 404.")

    with pytest.raises(WebpageFetchError):
        verify_match(
            "data/input/single_face.jpg",
            search_result_path=search_result_path,
            process_face_fn=fake_process_face,
            fetch_webpage_fn=failing_fetch,
        )


# ---------------------------------------------------------------------
# Privacy: no raw embeddings, no API keys, ever saved
# ---------------------------------------------------------------------

def test_no_raw_embeddings_or_secrets_in_saved_result(tmp_path):
    result, output_path = _run_verify(tmp_path, [URL_MULTI_FACE_WITH_MATCH])

    saved_text = open(output_path, encoding="utf-8").read()
    saved = json.loads(saved_text)

    # Only these top-level keys are expected -- no embedding arrays.
    assert set(saved.keys()) == {
        "query_image", "candidate_url", "verification_completed",
        "match_found", "best_match", "images_discovered",
        "images_analyzed", "timestamp_utc",
    }
    assert "face_encoding" not in saved_text
    assert "api_key" not in saved_text.lower()
    # A raw 128-d float array would contain many long decimal numbers;
    # a sanity check that nothing resembling one leaked into the file.
    assert saved_text.count(",") < 20
