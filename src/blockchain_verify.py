"""
Step 6 (verify):

python -m src.blockchain_verify [evidence_record_path]

Recomputes the LOCAL fingerprint using Step 4's own
recompute_fingerprint_hash() function.

Then compares the local fingerprint against the fingerprint stored
on the local Hardhat blockchain for the exact record created by
blockchain_upload.py.

If evidence_record_path is omitted, the real:

    data/output/evidence_record.json

is checked.

For a safe tamper-detection demonstration, a modified COPY of the
evidence file can be supplied instead.
"""

import sys
import json

from src.fingerprint import (
    DEFAULT_EVIDENCE_RECORD_PATH,
    recompute_fingerprint_hash,
)

from src.blockchain_client import (
    DEFAULT_RPC_URL,
    BlockchainClientError,
    load_blockchain_record,
    load_deployment_info,
    load_contract_abi,
    get_web3,
    get_contract,
    get_record,
)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if len(argv) > 1:
        print(
            "Usage: python -m src.blockchain_verify "
            "[evidence_record_path]"
        )
        return 2

    evidence_path = (
        argv[0]
        if argv
        else DEFAULT_EVIDENCE_RECORD_PATH
    )

    print("=" * 40)
    print("STEP 6: BLOCKCHAIN RE-VERIFICATION")
    print("=" * 40)
    print()

    # ============================================================
    # Recompute local fingerprint using Step 4
    # ============================================================
    try:
        local_hash, _ = recompute_fingerprint_hash(
            evidence_path
        )

    except FileNotFoundError:
        print(f"ERROR: {evidence_path} not found.")
        print("=" * 40)
        return 1

    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"ERROR: Could not read {evidence_path}: {exc}"
        )
        print("=" * 40)
        return 1

    # ============================================================
    # Read blockchain record
    # ============================================================
    try:
        blockchain_record = load_blockchain_record()

        deployment = load_deployment_info()

        abi = load_contract_abi()

        w3 = get_web3(
            deployment.get(
                "rpc_url",
                DEFAULT_RPC_URL,
            )
        )

        contract = get_contract(
            w3,
            deployment,
            abi,
        )

        record_id = blockchain_record["record_id"]

        on_chain = get_record(
            contract,
            record_id,
        )

        on_chain_hash = on_chain["fingerprint_hash"]

    except KeyError as exc:
        print(
            f"ERROR: blockchain_record.json is missing "
            f"required field: {exc}"
        )
        print("=" * 40)
        return 1

    except BlockchainClientError as exc:
        print(f"ERROR: {exc}")
        print("=" * 40)
        return 1

    # ============================================================
    # Display comparison
    # ============================================================
    print("Evidence file checked:")
    print(evidence_path)
    print()

    print("Blockchain record ID:")
    print(record_id)
    print()

    print("Local fingerprint:")
    print(local_hash)
    print()

    print("On-chain fingerprint:")
    print(on_chain_hash)
    print()

    # ============================================================
    # Compare fingerprints
    # ============================================================
    verified = local_hash == on_chain_hash

    if verified:
        print("Result:")
        print("VERIFIED")
        print()

        print(
            "The evidence fingerprint matches "
            "the blockchain record."
        )

    else:
        print("Result:")
        print("TAMPER DETECTED")
        print()

        print(
            "The recomputed local fingerprint does NOT match "
            "the fingerprint stored on-chain."
        )

        print(
            "The evidence data may have been modified "
            "since it was anchored."
        )

    print("=" * 40)

    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())