"""
Complete Face Verification + Blockchain Pipeline

Runs all 6 steps:

1. Face detection & encoding
2. Visual web search
3. Independent face verification
4. SHA-256 fingerprint generation
5. Blockchain anchoring
6. Blockchain re-verification
"""

import sys

from src.face_stage import process_face, FaceStageError
from src.search_stage import search_for_match
from src.verify_match import verify_match
from src.fingerprint import generate_fingerprint
from src.blockchain_upload import main as blockchain_upload_main
from src.blockchain_verify import main as blockchain_verify_main


def print_step(number, title):
    print()
    print("=" * 40)
    print(f"STEP {number}: {title}")
    print("=" * 40)
    print()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if len(argv) != 1:
        print("Usage:")
        print("python -m src.pipeline <image_path>")
        return 2

    image_path = argv[0]

    print()
    print("=" * 40)
    print("FACE VERIFICATION PIPELINE")
    print("=" * 40)

    # ============================================================
    # STEP 1
    # ============================================================
    print_step(1, "FACE DETECTION & ENCODING")

    try:
        face_result = process_face(image_path)

        print("✓ Face detected")
        print("✓ Encoding generated")
        print(f"✓ Faces detected: {face_result['face_count']}")

    except FaceStageError as exc:
        print(f"✗ STEP 1 FAILED: {exc}")
        print()
        print("FINAL RESULT")
        print("=" * 40)
        print("STATUS: FAILED")
        print("=" * 40)
        return 1

    # ============================================================
    # STEP 2
    # ============================================================
    print_step(2, "VISUAL WEB SEARCH")

    try:
        search_result = search_for_match(image_path)

        if not search_result:
            print("✗ No candidate was selected.")
            print()
            print("FINAL RESULT")
            print("=" * 40)
            print("STATUS: FAILED")
            print("=" * 40)
            return 1

        print("✓ Search completed")
        print("✓ Candidate discovered")

    except Exception as exc:
        print(f"✗ STEP 2 FAILED: {exc}")
        print()
        print("FINAL RESULT")
        print("=" * 40)
        print("STATUS: FAILED")
        print("=" * 40)
        return 1

    # ============================================================
    # STEP 3
    # ============================================================
    print_step(3, "INDEPENDENT FACE VERIFICATION")

    try:
        verification_result = verify_match(image_path)

        if not verification_result:
            print("✗ Verification returned no result.")
            print()
            print("FINAL RESULT")
            print("=" * 40)
            print("STATUS: FAILED")
            print("=" * 40)
            return 1

        match_found = verification_result.get("match_found", False)

        if not match_found:
            print("✗ Face match: NO")
            print()
            print("FINAL RESULT")
            print("=" * 40)
            print("STATUS: FAILED")
            print("=" * 40)
            return 1

        best_match = verification_result.get("best_match") or {}
        face_distance = best_match.get("face_distance")

        print("✓ Face match: YES")

        if face_distance is not None:
            print(f"✓ Face distance: {face_distance}")

        print("✓ Candidate discovered: VERIFIED")

    except Exception as exc:
        print(f"✗ STEP 3 FAILED: {exc}")
        print()
        print("FINAL RESULT")
        print("=" * 40)
        print("STATUS: FAILED")
        print("=" * 40)
        return 1

    # ============================================================
    # STEP 4
    # ============================================================
    print_step(4, "SHA-256 FINGERPRINT")

    try:
        fingerprint_result = generate_fingerprint()

        if not fingerprint_result:
            print("✗ Fingerprint generation failed.")
            print()
            print("FINAL RESULT")
            print("=" * 40)
            print("STATUS: FAILED")
            print("=" * 40)
            return 1

        print("✓ FINGERPRINT GENERATED")

    except Exception as exc:
        print(f"✗ STEP 4 FAILED: {exc}")
        print()
        print("FINAL RESULT")
        print("=" * 40)
        print("STATUS: FAILED")
        print("=" * 40)
        return 1

    # ============================================================
    # STEP 5
    # ============================================================
    print_step(5, "BLOCKCHAIN ANCHOR")

    try:
        blockchain_upload_status = blockchain_upload_main([])

        if blockchain_upload_status != 0:
            print()
            print("FINAL RESULT")
            print("=" * 40)
            print("STATUS: FAILED")
            print("=" * 40)
            return 1

        print("✓ Fingerprint stored on-chain")

    except Exception as exc:
        print(f"✗ STEP 5 FAILED: {exc}")
        print()
        print("FINAL RESULT")
        print("=" * 40)
        print("STATUS: FAILED")
        print("=" * 40)
        return 1

    # ============================================================
    # STEP 6
    # ============================================================
    print_step(6, "BLOCKCHAIN RE-VERIFICATION")

    try:
        blockchain_verify_status = blockchain_verify_main([])

        if blockchain_verify_status != 0:
            print()
            print("FINAL RESULT")
            print("=" * 40)
            print("STATUS: TAMPER DETECTED")
            print("=" * 40)
            return 1

        print("✓ Local fingerprint matches blockchain")

    except Exception as exc:
        print(f"✗ STEP 6 FAILED: {exc}")
        print()
        print("FINAL RESULT")
        print("=" * 40)
        print("STATUS: FAILED")
        print("=" * 40)
        return 1

    # ============================================================
    # FINAL RESULT
    # ============================================================
    print()
    print("=" * 40)
    print("FINAL RESULT")
    print("=" * 40)
    print()
    print("STATUS: SUCCESS")
    print()
    print("All 6 steps completed successfully.")
    print("The local hash and chain hash were compared and verified")
    print("HENCE, RECORD WAS NOT TAMPERED")
    return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())