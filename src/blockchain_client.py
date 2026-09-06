"""
Step 5 core: a thin, testable client for talking to the local Hardhat
blockchain and the deployed FingerprintRegistry contract.

Nothing here is hardcoded: the contract address and ABI are read from
files written by scripts/deploy.js (data/output/deployment.json and
data/output/FingerprintRegistry.abi.json), which change every time the
local chain is redeployed. web3.py is imported lazily inside the
functions that need it, so this module (and its pure helper functions)
can be imported and unit-tested even in an environment where web3.py
isn't installed.
"""

import os
import json

DEFAULT_RPC_URL = "http://127.0.0.1:8545"
DEFAULT_DEPLOYMENT_PATH = os.path.join("data", "output", "deployment.json")
DEFAULT_ABI_PATH = os.path.join("data", "output", "FingerprintRegistry.abi.json")
DEFAULT_BLOCKCHAIN_RECORD_PATH = os.path.join("data", "output", "blockchain_record.json")


class BlockchainClientError(Exception):
    """Base exception for all blockchain_client errors."""


class DeploymentInfoMissingError(BlockchainClientError):
    """Raised when data/output/deployment.json is missing -- the
    contract hasn't been deployed yet (or the file wasn't written)."""


class ContractAbiMissingError(BlockchainClientError):
    """Raised when data/output/FingerprintRegistry.abi.json is missing."""


class BlockchainRecordMissingError(BlockchainClientError):
    """Raised when data/output/blockchain_record.json is missing --
    blockchain_upload hasn't been run yet."""


class RpcConnectionError(BlockchainClientError):
    """Raised when the local Hardhat node can't be reached."""


# --------------------------------------------------------------------------
# File I/O (deployment info, ABI, and this stage's own saved record)
# --------------------------------------------------------------------------

def load_deployment_info(path=DEFAULT_DEPLOYMENT_PATH):
    if not os.path.isfile(path):
        raise DeploymentInfoMissingError(
            f"{path} not found. Deploy the contract first: "
            "npx hardhat run scripts/deploy.js --network localhost"
        )
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentInfoMissingError(f"Could not read {path}: {exc}") from exc

    if not isinstance(data, dict) or not data.get("contract_address"):
        raise DeploymentInfoMissingError(f"{path} does not contain a contract_address.")

    return data


def load_contract_abi(path=DEFAULT_ABI_PATH):
    if not os.path.isfile(path):
        raise ContractAbiMissingError(
            f"{path} not found. Deploy the contract first: "
            "npx hardhat run scripts/deploy.js --network localhost"
        )
    try:
        with open(path, encoding="utf-8") as f:
            abi = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractAbiMissingError(f"Could not read {path}: {exc}") from exc

    if not isinstance(abi, list):
        raise ContractAbiMissingError(f"{path} does not contain a valid ABI list.")

    return abi


def load_blockchain_record(path=DEFAULT_BLOCKCHAIN_RECORD_PATH):
    if not os.path.isfile(path):
        raise BlockchainRecordMissingError(
            f"{path} not found. Run 'python -m src.blockchain_upload' first."
        )
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise BlockchainRecordMissingError(f"Could not read {path}: {exc}") from exc


def save_json(data, path):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path


# --------------------------------------------------------------------------
# Hash <-> bytes32 conversion
# --------------------------------------------------------------------------

def _validate_sha256_hex(value):
    if not isinstance(value, str) or len(value) != 64:
        raise BlockchainClientError(
            f"Expected a 64-character SHA-256 hex string, got: {value!r}"
        )
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise BlockchainClientError(f"Fingerprint hash is not valid hex: {value!r}") from exc


def hash_to_bytes32(hex_hash):
    """A SHA-256 hex digest is exactly 64 hex chars = 32 bytes, so it
    maps onto Solidity's bytes32 with no truncation or padding."""
    _validate_sha256_hex(hex_hash)
    return bytes.fromhex(hex_hash)


def bytes32_to_hash(value):
    """Inverse of hash_to_bytes32. Accepts raw bytes or anything with
    a .hex() method (e.g. HexBytes)."""
    if hasattr(value, "hex"):
        return value.hex()
    return bytes(value).hex()


def to_hex_string(value):
    """Format a tx hash (HexBytes, bytes, or str) as a '0x...' string."""
    s = value.hex() if hasattr(value, "hex") else str(value)
    if not s.startswith("0x"):
        s = "0x" + s
    return s


# --------------------------------------------------------------------------
# web3 connection (lazy import -- see module docstring)
# --------------------------------------------------------------------------

def get_web3(rpc_url=DEFAULT_RPC_URL):
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise RpcConnectionError(
            f"Could not connect to the local Hardhat node at {rpc_url}. "
            "Make sure 'npx hardhat node' is running in another terminal."
        )
    return w3


def get_contract(w3, deployment_info, abi):
    from web3 import Web3

    address = Web3.to_checksum_address(deployment_info["contract_address"])
    return w3.eth.contract(address=address, abi=abi)


def get_default_account(w3):
    accounts = w3.eth.accounts
    if not accounts:
        raise RpcConnectionError("No accounts available from the local Hardhat node.")
    return accounts[0]


# --------------------------------------------------------------------------
# Contract interaction
# --------------------------------------------------------------------------

def store_fingerprint_onchain(w3, contract, account, fingerprint_hash_hex, reference):
    """Send storeFingerprint(hash, reference) and wait for the receipt.
    Returns (tx_hash, receipt)."""
    fingerprint_bytes = hash_to_bytes32(fingerprint_hash_hex)
    tx_hash = contract.functions.storeFingerprint(fingerprint_bytes, reference).transact(
        {"from": account}
    )
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return tx_hash, receipt


def get_record(contract, record_id):
    fingerprint_bytes, reference, submitter, timestamp = contract.functions.getRecord(
        record_id
    ).call()
    return {
        "record_id": record_id,
        "fingerprint_hash": bytes32_to_hash(fingerprint_bytes),
        "reference": reference,
        "submitter": submitter,
        "timestamp": timestamp,
    }


def get_latest_record(contract):
    record_id, fingerprint_bytes, reference, submitter, timestamp = (
        contract.functions.getLatestRecord().call()
    )
    return {
        "record_id": record_id,
        "fingerprint_hash": bytes32_to_hash(fingerprint_bytes),
        "reference": reference,
        "submitter": submitter,
        "timestamp": timestamp,
    }
