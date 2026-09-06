"""
Step 5 (upload): python -m src.blockchain_upload

Reads (all dynamic, nothing hardcoded):
    data/output/fingerprint.json       [Step 4] -> fingerprint_hash
    data/output/evidence_record.json   [Step 4] -> matched_url (used only
                                        as a short public on-chain
                                        reference string)
    data/output/deployment.json        [scripts/deploy.js] -> contract
                                        address + RPC url (changes every
                                        time the local chain restarts)
    data/output/FingerprintRegistry.abi.json [scripts/deploy.js]

Writes:
    data/output/blockchain_record.json -> tx hash, block number,
                                           record id, contract address --
                                           everything blockchain_verify
                                           needs to check the same record
                                           again later.
"""

import os
import sys
import json

from src.fingerprint import DEFAULT_FINGERPRINT_PATH, DEFAULT_EVIDENCE_RECORD_PATH
from src.blockchain_client import (
    DEFAULT_RPC_URL,
    DEFAULT_BLOCKCHAIN_RECORD_PATH,
    BlockchainClientError,
    load_deployment_info,
    load_contract_abi,
    save_json,
    get_web3,
    get_contract,
    get_default_account,
    store_fingerprint_onchain,
    get_latest_record,
    to_hex_string,
)


class UploadInputMissingError(BlockchainClientError):
    """Raised when Step 4's output files aren't present."""


def load_fingerprint(path=DEFAULT_FINGERPRINT_PATH):
    if not os.path.isfile(path):
        raise UploadInputMissingError(
            f"{path} not found. Run Step 4 (python -m src.fingerprint) first."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    fingerprint_hash = data.get("fingerprint_hash")
    if not fingerprint_hash:
        raise UploadInputMissingError(f"{path} does not contain a fingerprint_hash.")

    evidence_path = data.get("evidence_record_path", DEFAULT_EVIDENCE_RECORD_PATH)
    return fingerprint_hash, evidence_path


def load_evidence_reference(evidence_path):
    """The on-chain 'reference' is the matched public source-page URL
    from Step 4's evidence record -- read fresh, never hardcoded."""
    if not os.path.isfile(evidence_path):
        raise UploadInputMissingError(
            f"{evidence_path} not found. Run Step 4 (python -m src.fingerprint) first."
        )
    with open(evidence_path, encoding="utf-8") as f:
        record = json.load(f)
    return record.get("matched_url") or evidence_path


def _print_banner(contract_address, fingerprint_hash):
    print("=" * 40)
    print("STEP 5: BLOCKCHAIN ANCHOR")
    print("=" * 40)
    print()
    print("Network: Local Hardhat")
    print(f"Contract: {contract_address}")
    print()
    print("Fingerprint:")
    print(fingerprint_hash)
    print()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 0:
        print("Usage: python -m src.blockchain_upload")
        return 2

    try:
        fingerprint_hash, evidence_path = load_fingerprint()
        reference = load_evidence_reference(evidence_path)
        deployment = load_deployment_info()
        abi = load_contract_abi()

        w3 = get_web3(deployment.get("rpc_url", DEFAULT_RPC_URL))
        contract = get_contract(w3, deployment, abi)
        account = get_default_account(w3)

        _print_banner(deployment["contract_address"], fingerprint_hash)

        print("Submitting transaction...")
        print()

        tx_hash, receipt = store_fingerprint_onchain(
            w3, contract, account, fingerprint_hash, reference
        )
        tx_hash_str = to_hex_string(tx_hash)
        block_number = receipt["blockNumber"] if isinstance(receipt, dict) else receipt.blockNumber

        print("Transaction hash:")
        print(tx_hash_str)
        print()
        print("Block number:")
        print(block_number)
        print()

        latest = get_latest_record(contract)
        on_chain_hash = latest["fingerprint_hash"]

        print("Fingerprint stored on-chain:")
        print(on_chain_hash)
        print()

        verified = on_chain_hash == fingerprint_hash
        print(f"Blockchain verification: {'VERIFIED' if verified else 'MISMATCH'}")
        print()
        print("Status: SUCCESS" if verified else "Status: FAILED")
        print("=" * 40)

        blockchain_record = {
            "contract_address": deployment["contract_address"],
            "rpc_url": deployment.get("rpc_url", DEFAULT_RPC_URL),
            "record_id": latest["record_id"],
            "tx_hash": tx_hash_str,
            "block_number": block_number,
            "fingerprint_hash": fingerprint_hash,
            "reference": reference,
            "evidence_record_path": evidence_path,
        }
        save_json(blockchain_record, DEFAULT_BLOCKCHAIN_RECORD_PATH)

        return 0 if verified else 1

    except BlockchainClientError as exc:
        print("=" * 40)
        print("STEP 5: BLOCKCHAIN ANCHOR")
        print("=" * 40)
        print()
        print(f"ERROR: {exc}")
        print("Status: FAILED")
        print("=" * 40)
        return 1


if __name__ == "__main__":
    sys.exit(main())
