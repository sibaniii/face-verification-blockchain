"""
Tests for src.blockchain_client.

Fully offline: no real web3.py connection, no Node.js/Hardhat needed.
Contract/web3 interaction is tested against small fake objects that
implement just the subset of the web3.py API this module actually
uses (functions.X().transact()/.call(), eth.wait_for_transaction_receipt,
eth.accounts) -- this exercises the real business logic (hash<->bytes32
conversion, tuple-to-dict unpacking, file I/O) without needing web3.py
itself to be installed.
"""

import os
import json

import pytest

from src.blockchain_client import (
    hash_to_bytes32,
    bytes32_to_hash,
    to_hex_string,
    load_deployment_info,
    load_contract_abi,
    load_blockchain_record,
    save_json,
    get_default_account,
    store_fingerprint_onchain,
    get_record,
    get_latest_record,
    BlockchainClientError,
    DeploymentInfoMissingError,
    ContractAbiMissingError,
    BlockchainRecordMissingError,
    RpcConnectionError,
)

SAMPLE_HASH = "a" * 64  # a syntactically valid 64-char hex SHA-256 digest


# ---------------------------------------------------------------------
# hash <-> bytes32
# ---------------------------------------------------------------------

def test_hash_to_bytes32_and_back_roundtrip():
    as_bytes = hash_to_bytes32(SAMPLE_HASH)
    assert len(as_bytes) == 32
    assert bytes32_to_hash(as_bytes) == SAMPLE_HASH


def test_hash_to_bytes32_rejects_wrong_length():
    with pytest.raises(BlockchainClientError):
        hash_to_bytes32("abc123")  # too short


def test_hash_to_bytes32_rejects_non_hex():
    with pytest.raises(BlockchainClientError):
        hash_to_bytes32("z" * 64)  # right length, invalid hex


def test_bytes32_to_hash_accepts_object_with_hex_method():
    class FakeHexBytes:
        def hex(self):
            return SAMPLE_HASH

    assert bytes32_to_hash(FakeHexBytes()) == SAMPLE_HASH


def test_to_hex_string_adds_0x_prefix_when_missing():
    assert to_hex_string("deadbeef") == "0xdeadbeef"
    assert to_hex_string("0xdeadbeef") == "0xdeadbeef"


def test_to_hex_string_handles_hex_method_object():
    class FakeHexBytes:
        def hex(self):
            return "0xcafebabe"

    assert to_hex_string(FakeHexBytes()) == "0xcafebabe"


# ---------------------------------------------------------------------
# File loading: missing files raise clear, specific errors
# ---------------------------------------------------------------------

def test_load_deployment_info_missing_file_raises(tmp_path):
    with pytest.raises(DeploymentInfoMissingError):
        load_deployment_info(str(tmp_path / "does_not_exist.json"))


def test_load_deployment_info_missing_address_raises(tmp_path):
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps({"network": "localhost"}), encoding="utf-8")
    with pytest.raises(DeploymentInfoMissingError):
        load_deployment_info(str(path))


def test_load_deployment_info_valid(tmp_path):
    path = tmp_path / "deployment.json"
    payload = {"contract_address": "0x1234", "rpc_url": "http://127.0.0.1:8545"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_deployment_info(str(path)) == payload


def test_load_contract_abi_missing_file_raises(tmp_path):
    with pytest.raises(ContractAbiMissingError):
        load_contract_abi(str(tmp_path / "does_not_exist.json"))


def test_load_contract_abi_must_be_a_list(tmp_path):
    path = tmp_path / "abi.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ContractAbiMissingError):
        load_contract_abi(str(path))


def test_load_blockchain_record_missing_file_raises(tmp_path):
    with pytest.raises(BlockchainRecordMissingError):
        load_blockchain_record(str(tmp_path / "does_not_exist.json"))


def test_save_json_roundtrip(tmp_path):
    path = tmp_path / "out.json"
    save_json({"a": 1, "b": 2}, str(path))
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}


# ---------------------------------------------------------------------
# get_default_account
# ---------------------------------------------------------------------

class FakeEth:
    def __init__(self, accounts=None, receipt=None):
        self.accounts = accounts or []
        self._receipt = receipt

    def wait_for_transaction_receipt(self, tx_hash):
        return self._receipt


class FakeW3:
    def __init__(self, accounts=None, receipt=None):
        self.eth = FakeEth(accounts=accounts, receipt=receipt)


def test_get_default_account_returns_first():
    w3 = FakeW3(accounts=["0xAAA", "0xBBB"])
    assert get_default_account(w3) == "0xAAA"


def test_get_default_account_raises_when_empty():
    w3 = FakeW3(accounts=[])
    with pytest.raises(RpcConnectionError):
        get_default_account(w3)


# ---------------------------------------------------------------------
# store_fingerprint_onchain / get_record / get_latest_record
# (fake contract implementing just functions.X().transact()/.call())
# ---------------------------------------------------------------------

class FakeBoundFunction:
    def __init__(self, transact_return=None, call_return=None, record=None):
        self._transact_return = transact_return
        self._call_return = call_return
        self._record = record  # list to append (arg1, arg2) calls into, for assertions

    def transact(self, tx_opts):
        if self._record is not None:
            self._record.append(("transact", tx_opts))
        return self._transact_return

    def call(self):
        return self._call_return


class FakeFunctions:
    def __init__(self, store_result=None, get_record_result=None, get_latest_result=None, calls=None):
        self._store_result = store_result
        self._get_record_result = get_record_result
        self._get_latest_result = get_latest_result
        self.calls = calls if calls is not None else []

    def storeFingerprint(self, fingerprint_bytes, reference):
        self.calls.append(("storeFingerprint", fingerprint_bytes, reference))
        return FakeBoundFunction(transact_return="0xFAKE_TX_HASH", record=self.calls)

    def getRecord(self, record_id):
        self.calls.append(("getRecord", record_id))
        return FakeBoundFunction(call_return=self._get_record_result)

    def getLatestRecord(self):
        self.calls.append(("getLatestRecord",))
        return FakeBoundFunction(call_return=self._get_latest_result)


class FakeContract:
    def __init__(self, functions):
        self.functions = functions


def test_store_fingerprint_onchain_sends_correct_bytes_and_waits_for_receipt():
    calls = []
    fake_functions = FakeFunctions(calls=calls)
    contract = FakeContract(fake_functions)
    fake_receipt = {"blockNumber": 7, "status": 1}
    w3 = FakeW3(receipt=fake_receipt)

    tx_hash, receipt = store_fingerprint_onchain(w3, contract, "0xACCOUNT", SAMPLE_HASH, "https://example.com/page")

    assert tx_hash == "0xFAKE_TX_HASH"
    assert receipt == fake_receipt
    assert calls[0][0] == "storeFingerprint"
    assert calls[0][1] == bytes.fromhex(SAMPLE_HASH)  # exact 32-byte payload sent
    assert calls[0][2] == "https://example.com/page"


def test_get_record_unpacks_tuple_into_dict():
    fake_functions = FakeFunctions(
        get_record_result=(bytes.fromhex(SAMPLE_HASH), "https://example.com/page", "0xSUBMITTER", 1234567890)
    )
    contract = FakeContract(fake_functions)

    result = get_record(contract, 3)

    assert result == {
        "record_id": 3,
        "fingerprint_hash": SAMPLE_HASH,
        "reference": "https://example.com/page",
        "submitter": "0xSUBMITTER",
        "timestamp": 1234567890,
    }


def test_get_latest_record_unpacks_tuple_into_dict():
    fake_functions = FakeFunctions(
        get_latest_result=(5, bytes.fromhex(SAMPLE_HASH), "https://example.com/latest", "0xSUBMITTER", 987654321)
    )
    contract = FakeContract(fake_functions)

    result = get_latest_record(contract)

    assert result == {
        "record_id": 5,
        "fingerprint_hash": SAMPLE_HASH,
        "reference": "https://example.com/latest",
        "submitter": "0xSUBMITTER",
        "timestamp": 987654321,
    }
