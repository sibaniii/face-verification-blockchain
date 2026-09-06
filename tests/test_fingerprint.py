"""
Tests for src.fingerprint.

Fully offline: no real HTTP requests. hash_matched_image() is tested
both via a local cached file (the normal path) and via an injected
fake download_fn (the fallback path) -- never the real network.
"""

import os
import json
import hashlib

import pytest

from src.fingerprint import (
    build_evidence_record,
    canonicalize,
    compute_fingerprint_hash,
    recompute_fingerprint_hash,
    hash_matched_image,
    generate_fingerprint,
    load_verification_result,
    VerificationResultMissingError,
    NoVerifiedMatchError,
)

BASE_VERIFICATION_RESULT = {
    "query_image": "data/input/single_face2.jpg",
    "candidate_url": "https://buildclub.snpsu.edu.in/team/",
    "verification_completed": True,
    "match_found": True,
    "best_match": {
        "image_url": "https://buildclub.snpsu.edu.in/assets/select-team/Tauseef.jpeg",
        "local_image_path": None,  # filled in per-test with a real tmp file
        "face_distance": 0.0,
        "threshold": 0.6,
    },
    "images_discovered": 17,
    "images_analyzed": 17,
    "timestamp_utc": "2026-09-05T10:00:00+00:00",
}


def write_verification_result(tmp_path, local_image_path=None, overrides=None):
    payload = json.loads(json.dumps(BASE_VERIFICATION_RESULT))  # deep copy
    if local_image_path is not None:
        payload["best_match"]["local_image_path"] = local_image_path
    if overrides:
        payload.update(overrides)
    path = tmp_path / "verification_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def write_search_result(tmp_path, search_engine="serpapi_google_lens", search_method="api"):
    payload = {
        "selected_candidate": {"url": "https://buildclub.snpsu.edu.in/team/"},
        "search_engine": search_engine,
        "search_method": search_method,
    }
    path = tmp_path / "search_result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


BASE_RECORD = {
    "matched_url": "https://buildclub.snpsu.edu.in/team/",
    "matched_image_url": "https://buildclub.snpsu.edu.in/assets/select-team/Tauseef.jpeg",
    "matched_image_sha256": "abc123",
    "face_distance": 0.0,
    "verification_threshold": 0.6,
    "verification_status": "VERIFIED",
    "search_engine": "serpapi_google_lens",
    "search_method": "api",
    "timestamp_utc": "2026-09-05T10:00:00+00:00",
}


# ---------------------------------------------------------------------
# 1-5: fingerprint hash determinism / sensitivity
# ---------------------------------------------------------------------

def test_same_record_same_hash():
    h1 = compute_fingerprint_hash(dict(BASE_RECORD))
    h2 = compute_fingerprint_hash(dict(BASE_RECORD))
    assert h1 == h2


def test_changed_matched_url_changes_hash():
    original = compute_fingerprint_hash(dict(BASE_RECORD))
    changed = dict(BASE_RECORD, matched_url="https://different.example.com/page")
    assert compute_fingerprint_hash(changed) != original


def test_changed_matched_image_sha256_changes_hash():
    original = compute_fingerprint_hash(dict(BASE_RECORD))
    changed = dict(BASE_RECORD, matched_image_sha256="def456")
    assert compute_fingerprint_hash(changed) != original


def test_changed_face_distance_changes_hash():
    original = compute_fingerprint_hash(dict(BASE_RECORD))
    changed = dict(BASE_RECORD, face_distance=0.59)
    assert compute_fingerprint_hash(changed) != original


def test_changed_verification_status_changes_hash():
    original = compute_fingerprint_hash(dict(BASE_RECORD))
    changed = dict(BASE_RECORD, verification_status="NOT_VERIFIED")
    assert compute_fingerprint_hash(changed) != original


# ---------------------------------------------------------------------
# 6: canonical JSON is order-independent
# ---------------------------------------------------------------------

def test_canonical_json_independent_of_key_order():
    ordered_a = dict(BASE_RECORD)  # insertion order as defined above
    ordered_b = {k: BASE_RECORD[k] for k in reversed(list(BASE_RECORD.keys()))}

    assert canonicalize(ordered_a) == canonicalize(ordered_b)
    assert compute_fingerprint_hash(ordered_a) == compute_fingerprint_hash(ordered_b)


# ---------------------------------------------------------------------
# 7: timestamp is preserved, never regenerated
# ---------------------------------------------------------------------

def test_build_evidence_record_reuses_stored_timestamp():
    verification_result = json.loads(json.dumps(BASE_VERIFICATION_RESULT))
    record = build_evidence_record(verification_result, "serpapi_google_lens", "api", "abc123")
    assert record["timestamp_utc"] == "2026-09-05T10:00:00+00:00"


def test_generate_fingerprint_does_not_regenerate_timestamp(tmp_path):
    image_bytes = b"fake image bytes for hashing"
    image_path = tmp_path / "cached.jpg"
    image_path.write_bytes(image_bytes)

    vr_path = write_verification_result(tmp_path, local_image_path=str(image_path))
    sr_path = write_search_result(tmp_path)
    evidence_path = tmp_path / "evidence_record.json"
    fp_path = tmp_path / "fingerprint.json"

    record1, hash1 = generate_fingerprint(
        verification_result_path=vr_path,
        search_result_path=sr_path,
        evidence_output_path=str(evidence_path),
        fingerprint_output_path=str(fp_path),
    )
    record2, hash2 = generate_fingerprint(
        verification_result_path=vr_path,
        search_result_path=sr_path,
        evidence_output_path=str(evidence_path),
        fingerprint_output_path=str(fp_path),
    )

    # Same underlying Step 3 result, run twice -> identical evidence
    # timestamp and identical fingerprint hash, even though real time
    # has passed between the two calls.
    assert record1["timestamp_utc"] == record2["timestamp_utc"] == "2026-09-05T10:00:00+00:00"
    assert hash1 == hash2


# ---------------------------------------------------------------------
# 8: image hashing (local cache preferred, download fallback works)
# ---------------------------------------------------------------------

def test_hash_matched_image_uses_local_cache(tmp_path):
    image_bytes = b"some verified image bytes"
    image_path = tmp_path / "cached.jpg"
    image_path.write_bytes(image_bytes)

    expected = hashlib.sha256(image_bytes).hexdigest()
    best_match = {"image_url": "https://example.com/x.jpg", "local_image_path": str(image_path)}

    assert hash_matched_image(best_match) == expected


def test_hash_matched_image_falls_back_to_download(tmp_path):
    image_bytes = b"downloaded bytes since local file is missing"
    expected = hashlib.sha256(image_bytes).hexdigest()

    best_match = {
        "image_url": "https://example.com/x.jpg",
        "local_image_path": str(tmp_path / "does_not_exist.jpg"),
    }

    fake_download = lambda url: image_bytes  # never a real request
    assert hash_matched_image(best_match, download_fn=fake_download) == expected


def test_hash_matched_image_returns_none_when_unobtainable(tmp_path):
    best_match = {
        "image_url": "https://example.com/x.jpg",
        "local_image_path": str(tmp_path / "does_not_exist.jpg"),
    }
    fake_download = lambda url: None
    assert hash_matched_image(best_match, download_fn=fake_download) is None


# ---------------------------------------------------------------------
# Loading verification_result.json: missing file / no verified match
# ---------------------------------------------------------------------

def test_missing_verification_result_raises(tmp_path):
    with pytest.raises(VerificationResultMissingError):
        load_verification_result(str(tmp_path / "does_not_exist.json"))


def test_unverified_match_raises(tmp_path):
    path = write_verification_result(tmp_path, overrides={"match_found": False})
    with pytest.raises(NoVerifiedMatchError):
        load_verification_result(path)


# ---------------------------------------------------------------------
# Full orchestration + no raw embeddings/secrets anywhere in the output
# ---------------------------------------------------------------------

def test_generate_fingerprint_full_flow_matches_build_club_shape(tmp_path):
    image_bytes = b"the actual verified matching image bytes"
    image_path = tmp_path / "Tauseef.jpeg"
    image_path.write_bytes(image_bytes)

    vr_path = write_verification_result(tmp_path, local_image_path=str(image_path))
    sr_path = write_search_result(tmp_path)
    evidence_path = tmp_path / "evidence_record.json"
    fp_path = tmp_path / "fingerprint.json"

    record, fingerprint_hash = generate_fingerprint(
        verification_result_path=vr_path,
        search_result_path=sr_path,
        evidence_output_path=str(evidence_path),
        fingerprint_output_path=str(fp_path),
    )

    assert record["matched_url"] == "https://buildclub.snpsu.edu.in/team/"
    assert record["matched_image_url"] == "https://buildclub.snpsu.edu.in/assets/select-team/Tauseef.jpeg"
    assert record["matched_image_sha256"] == hashlib.sha256(image_bytes).hexdigest()
    assert record["verification_status"] == "VERIFIED"
    assert record["search_engine"] == "serpapi_google_lens"
    assert record["search_method"] == "api"

    assert evidence_path.is_file()
    assert fp_path.is_file()

    saved_fp = json.loads(fp_path.read_text(encoding="utf-8"))
    assert saved_fp["fingerprint_hash"] == fingerprint_hash
    assert saved_fp["algorithm"] == "sha256"

    # No raw embeddings, no API keys, anywhere in either saved file.
    evidence_text = evidence_path.read_text(encoding="utf-8")
    fp_text = fp_path.read_text(encoding="utf-8")
    for text in (evidence_text, fp_text):
        assert "face_encoding" not in text
        assert "api_key" not in text.lower()

    # Recomputing from the saved evidence file reproduces the same hash.
    recomputed_hash, reloaded_record = recompute_fingerprint_hash(str(evidence_path))
    assert recomputed_hash == fingerprint_hash
    assert reloaded_record == record


def test_generate_fingerprint_no_hardcoded_buildclub_url_needed(tmp_path):
    """The same code path, but with a completely different (fake)
    candidate/site, proves nothing about Build Club is hardcoded."""
    image_bytes = b"a totally different person's verified photo bytes"
    image_path = tmp_path / "someone_else.jpg"
    image_path.write_bytes(image_bytes)

    vr_path = write_verification_result(
        tmp_path,
        local_image_path=str(image_path),
        overrides={
            "candidate_url": "https://example-other-site.test/profile/",
            "best_match": {
                "image_url": "https://example-other-site.test/photos/person.jpg",
                "local_image_path": str(image_path),
                "face_distance": 0.21,
                "threshold": 0.6,
            },
        },
    )
    sr_path = write_search_result(tmp_path, search_engine="google_lens_manual", search_method="human_in_the_loop")
    evidence_path = tmp_path / "evidence_record.json"
    fp_path = tmp_path / "fingerprint.json"

    record, _ = generate_fingerprint(
        verification_result_path=vr_path,
        search_result_path=sr_path,
        evidence_output_path=str(evidence_path),
        fingerprint_output_path=str(fp_path),
    )

    assert record["matched_url"] == "https://example-other-site.test/profile/"
    assert record["search_engine"] == "google_lens_manual"
    assert record["search_method"] == "human_in_the_loop"
