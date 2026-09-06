"""
Tests for the pure, file-based helper functions in
src.blockchain_upload (load_fingerprint / load_evidence_reference).
The full main() orchestration talks to a real local Hardhat node and
is validated by actually running it (see README), not by this offline
suite -- that's consistent with how Steps 2/3 handle their own
live-network paths.
"""

import os
import json

import pytest

from src.blockchain_upload import load_fingerprint, load_evidence_reference, UploadInputMissingError


def test_load_fingerprint_missing_file_raises(tmp_path):
    with pytest.raises(UploadInputMissingError):
        load_fingerprint(str(tmp_path / "does_not_exist.json"))


def test_load_fingerprint_missing_hash_field_raises(tmp_path):
    path = tmp_path / "fingerprint.json"
    path.write_text(json.dumps({"algorithm": "sha256"}), encoding="utf-8")
    with pytest.raises(UploadInputMissingError):
        load_fingerprint(str(path))


def test_load_fingerprint_valid(tmp_path):
    path = tmp_path / "fingerprint.json"
    payload = {
        "fingerprint_hash": "a" * 64,
        "algorithm": "sha256",
        "evidence_record_path": "data/output/evidence_record.json",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    fingerprint_hash, evidence_path = load_fingerprint(str(path))
    assert fingerprint_hash == "a" * 64
    assert evidence_path == "data/output/evidence_record.json"


def test_load_evidence_reference_uses_matched_url(tmp_path):
    path = tmp_path / "evidence_record.json"
    path.write_text(json.dumps({"matched_url": "https://buildclub.snpsu.edu.in/team/"}), encoding="utf-8")
    assert load_evidence_reference(str(path)) == "https://buildclub.snpsu.edu.in/team/"


def test_load_evidence_reference_missing_file_raises(tmp_path):
    with pytest.raises(UploadInputMissingError):
        load_evidence_reference(str(tmp_path / "does_not_exist.json"))


def test_load_evidence_reference_falls_back_to_path_if_no_matched_url(tmp_path):
    path = tmp_path / "evidence_record.json"
    path.write_text(json.dumps({"some_other_field": "x"}), encoding="utf-8")
    assert load_evidence_reference(str(path)) == str(path)
