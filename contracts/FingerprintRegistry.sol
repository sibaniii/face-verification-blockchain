// SPDX-License-Identifier: MIT 
pragma solidity ^0.8.19; 
 
/// @title FingerprintRegistry 
/// @notice Minimal on-chain registry for content-fingerprint hashes. 
/// Stores ONLY a SHA-256 fingerprint hash (as bytes32) and a short 
/// human-readable reference string (e.g. the public source-page URL 
/// the fingerprint corresponds to). Never stores face embeddings, 
/// raw images, or any other private/biometric data. 
contract FingerprintRegistry { 
    struct Record { 
        bytes32 fingerprintHash; 
        string red; 
        address submitter; 
        uint256 timestamp; 
    } 
 
    Record[] private records; 
 
    event FingerprintStored( 
        uint256 indexed recordId, 
        bytes32 indexed fingerprintHash, 
        string red, 
        address indexed submitter, 
        uint256 timestamp 
    ); 
 
    /// @notice Store a new fingerprint record. 
    /// @param fingerprintHash The SHA-256 fingerprint hash (32 bytes). 
    /// @param red A short public reference (e.g. the matched 
    ///        source-page URL) -- for context only, not hashed data. 
    /// @return recordId The index this record was stored at. 
    function storeFingerprint(bytes32 fingerprintHash, string calldata red) 
        external 
        returns (uint256 recordId) 
    { 
        recordId = records.length; 
        records.push( 
            Record({ 
                fingerprintHash: fingerprintHash, 
                red: red, 
                submitter: msg.sender, 
                timestamp: block.timestamp 
            }) 
        ); 
        emit FingerprintStored(recordId, fingerprintHash, red, msg.sender, block.timestamp); 
    } 
 
    /// @notice Total number of records stored so far. 
    function recordCount() external view returns (uint256) { 
        return records.length; 
    } 
 
    /// @notice Fetch a specific record by id. 
    function getRecord(uint256 recordId) 
        external 
        view 
        returns ( 
            bytes32 fingerprintHash, 
            string memory red, 
            address submitter, 
            uint256 timestamp 
        ) 
    { 
        require(recordId < records.length, "FingerprintRegistry: invalid recordId"); 
        Record storage r = records[recordId]; 
        return (r.fingerprintHash, r.red, r.submitter, r.timestamp); 
    } 
 
    /// @notice Convenience: fetch the most recently stored record. 
    function getLatestRecord() 
        external 
        view 
        returns ( 
            uint256 recordId, 
            bytes32 fingerprintHash, 
            string memory red, 
            address submitter, 
            uint256 timestamp 
        ) 
    { 
        require(records.length > 0, "FingerprintRegistry: no records"); 
        recordId = records.length - 1; 
        Record storage r = records[recordId]; 
        return (recordId, r.fingerprintHash, r.red, r.submitter, r.timestamp); 
    } 
}