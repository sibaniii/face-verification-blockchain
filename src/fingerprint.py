"""
Step 4 of the pipeline: turn a VERIFIED Step 3 result into a
deterministic, hashable evidence record.

Pipeline:
    data/output/verification_result.json  (Step 3 output, must have
                                            match_found: true)
        + data/output/search_result.json  (Step 2 output, for
                                            search_engine/search_method)
        -> canonical evidence record  (sorted keys, no randomness,
                                        timestamp REUSED from Step 3 --
                                        never regenerated here)
        -> SHA-256(canonical JSON) = fingerprint_hash

DETERMINISM RULE (read before touching this file):
    The evidence record must hash identically every time it is built
    from the same underlying verification/search results. The single
    biggest threat to that is timestamps: if this module generated its
    own "now" timestamp on every run, the same evidence would produce
    a different hash every time, which defeats the entire purpose of a
    fingerprint. So the evidence record's timestamp_utc is copied
    directly from verification_result.json's own timestamp_utc (the
    moment Step 3 actually verified the match) and is never
    regenerated here. The separate fingerprint.json output file may
    carry its own fresh "generated_at" field for bookkeeping, because
    that field is metadata ABOUT the hashing run, not part of what
    gets hashed -- it can change every run without changing the hash.

No raw face embeddings and no API keys are ever included here -- the
evidence record only contains URLs, a distance, a threshold, a status
string, and an image hash.
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timezone

DEFAULT_SEARCH_RESULT_PATH = os.path.join("data", "output", "search_result.json")
DEFAULT_VERIFICATION_RESULT_PATH = os.path.join("data", "output", "verification_result.json")
DEFAULT_EVIDENCE_RECORD_PATH = os.path.join("data", "output", "evidence_record.json")
DEFAULT_FINGERPRINT_PATH = os.path.join("data", "output", "fingerprint.json")


class FingerprintError(Exception):
    """Base exception for all fingerprint-stage errors."""


class VerificationResultMissingError(FingerprintError):
    """Raised when data/output/verification_result.json is missing/unreadable."""


class NoVerifiedMatchError(FingerprintError):
    """Raised when Step 3's result has match_found: false -- nothing to fingerprint."""


# --------------------------------------------------------------------------
# Loading upstream stage output (never hardcoded -- always read from disk)
# --------------------------------------------------------------------------

def load_verification_result(path=DEFAULT_VERIFICATION_RESULT_PATH):
    if not os.path.isfile(path):
        raise VerificationResultMissingError(
            f"{path} not found. Run Step 3 (python -m src.verify_match <image>) first."
        )
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationResultMissingError(f"Could not read {path}: {exc}") from exc

    if not isinstance(data, dict) or not data.get("match_found"):
        raise NoVerifiedMatchError(
            f"{path} does not report a verified match (match_found is not true). "
            "Nothing to fingerprint until Step 3 produces FACE MATCH: YES."
        )
    if not isinstance(data.get("best_match"), dict):
        raise NoVerifiedMatchError(
            f"{path} has match_found true but no best_match details. Re-run Step 3."
        )

    return data


def load_search_engine_info(path=DEFAULT_SEARCH_RESULT_PATH):
    """Best-effort read of search_engine/search_method from Step 2's
    output. Never raises -- if the file is missing/unreadable, returns
    (None, None) and the caller records that honestly rather than
    guessing or hardcoding a value."""
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None

    if not isinstance(data, dict):
        return None, None

    return data.get("search_engine"), data.get("search_method")


# --------------------------------------------------------------------------
# Matched-image hashing
# --------------------------------------------------------------------------

def hash_matched_image(best_match, download_fn=None, timeout=15):
    """
    Return the SHA-256 hex digest of the matched image's bytes, or None
    if it can't be obtained. Prefers the local file Step 3 already
    downloaded and validated (local_image_path); falls back to
    re-fetching the image_url only if that local file is unavailable.
    """
    local_path = best_match.get("local_image_path")
    if local_path and os.path.isfile(local_path):
        with open(local_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    image_url = best_match.get("image_url")
    if not image_url:
        return None

    if download_fn is None:
        import requests
        def download_fn(url):
            try:
                response = requests.get(url, timeout=timeout)
            except requests.exceptions.RequestException:
                return None
            return response.content if response.status_code == 200 else None

    image_bytes = download_fn(image_url)
    if not image_bytes:
        return None

    return hashlib.sha256(image_bytes).hexdigest()


# --------------------------------------------------------------------------
# Canonical evidence record + SHA-256 fingerprint
# --------------------------------------------------------------------------

def build_evidence_record(verification_result, search_engine, search_method, matched_image_sha256):
    """
    Build the canonical evidence record. Field order in this dict does
    not matter for hashing (canonicalize() sorts keys), but is kept
    readable here for humans looking at the source.

    timestamp_utc is copied verbatim from verification_result -- see
    the DETERMINISM RULE in the module docstring.
    """
    best_match = verification_result["best_match"]

    return {
        "matched_url": verification_result.get("candidate_url"),
        "matched_image_url": best_match.get("image_url"),
        "matched_image_sha256": matched_image_sha256,
        "face_distance": best_match.get("face_distance"),
        "verification_threshold": best_match.get("threshold"),
        "verification_status": "VERIFIED" if verification_result.get("match_found") else "NOT_VERIFIED",
        "search_engine": search_engine,
        "search_method": search_method,
        "timestamp_utc": verification_result.get("timestamp_utc"),
    }


def canonicalize(record):
    """Deterministic JSON serialization: sorted keys, fixed compact
    separators, UTF-8-safe (non-ASCII left as literal characters
    rather than \\u-escaped, but consistently either way across runs)."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_fingerprint_hash(record):
    """SHA-256 of the canonical JSON form of record, as a hex string."""
    canonical = canonicalize(record)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def recompute_fingerprint_hash(evidence_record_path=DEFAULT_EVIDENCE_RECORD_PATH):
    """
    Reload a previously saved evidence_record.json exactly as-is and
    recompute its hash. This is the same operation the later
    blockchain re-verification stage will perform: reload the saved
    evidence, hash it again, and compare to what's on-chain. Nothing
    in the record (including timestamp_utc) is regenerated here --
    it's read back verbatim from disk.
    """
    with open(evidence_record_path, encoding="utf-8") as f:
        record = json.load(f)
    return compute_fingerprint_hash(record), record


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_json(data, path):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path


# --------------------------------------------------------------------------
# Terminal output
# --------------------------------------------------------------------------

def _print_banner():
    print("=" * 40)
    print("STEP 4: FINGERPRINT")
    print("=" * 40)
    print()


def _print_result(record, fingerprint_hash, evidence_path, fingerprint_path):
    print("Evidence record:")
    print(f"  Matched URL: {record['matched_url']}")
    print(f"  Matched image URL: {record['matched_image_url']}")
    print(f"  Face distance: {record['face_distance']}")
    print(f"  Threshold: {record['verification_threshold']}")
    print(f"  Verification status: {record['verification_status']}")
    print(f"  Search engine: {record['search_engine']}")
    print(f"  Search method: {record['search_method']}")
    print(f"  Timestamp (reused from Step 3): {record['timestamp_utc']}")
    print()

    print("=" * 40)
    print("SHA-256 FINGERPRINT")
    print("=" * 40)
    print()

    print("INPUT HASH (MATCHED IMAGE):")
    print(record["matched_image_sha256"])
    print()

    print("OUTPUT HASH (EVIDENCE FINGERPRINT):")
    print(fingerprint_hash)
    print()

    print("FINGERPRINT GENERATED: YES")
    print()

    print(f"Evidence record saved to: {evidence_path}")
    print(f"Fingerprint saved to: {fingerprint_path}")
    print()

    print("Status: SUCCESS")
    print("=" * 40)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def generate_fingerprint(
    verification_result_path=DEFAULT_VERIFICATION_RESULT_PATH,
    search_result_path=DEFAULT_SEARCH_RESULT_PATH,
    evidence_output_path=DEFAULT_EVIDENCE_RECORD_PATH,
    fingerprint_output_path=DEFAULT_FINGERPRINT_PATH,
    download_fn=None,
):
    """
    Runs Step 4 end to end. download_fn is injectable purely so tests
    can run fully offline. Returns (record, fingerprint_hash).
    """
    verification_result = load_verification_result(verification_result_path)
    search_engine, search_method = load_search_engine_info(search_result_path)

    matched_image_sha256 = hash_matched_image(
        verification_result["best_match"], download_fn=download_fn
    )
    if matched_image_sha256 is None:
        print("WARNING: could not obtain the matched image's bytes "
              "(local file missing and re-download failed). "
              "matched_image_sha256 will be recorded as null.")
        print()

    record = build_evidence_record(
        verification_result, search_engine, search_method, matched_image_sha256
    )
    fingerprint_hash = compute_fingerprint_hash(record)

    save_json(record, evidence_output_path)
    fingerprint_output = {
        "fingerprint_hash": fingerprint_hash,
        "algorithm": "sha256",
        "evidence_record_path": evidence_output_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(fingerprint_output, fingerprint_output_path)

    _print_banner()
    _print_result(record, fingerprint_hash, evidence_output_path, fingerprint_output_path)

    return record, fingerprint_hash


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if len(argv) not in (0,):
        print("Usage: python -m src.fingerprint")
        return 2

    try:
        generate_fingerprint()
    except FingerprintError as exc:
        print("=" * 40)
        print("STEP 4: FINGERPRINT")
        print("=" * 40)
        print()
        print(f"ERROR: {exc}")
        print("Status: FAILED")
        print("=" * 40)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
