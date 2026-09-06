// Deploys FingerprintRegistry.sol to whatever network Hardhat is
// pointed at (we only ever use --network localhost, the local
// `npx hardhat node`). Writes the resulting contract address, ABI,
// and deployment tx info to data/output/ so the Python client never
// needs a hardcoded address -- it changes every time the local chain
// is restarted.

const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const FingerprintRegistry = await hre.ethers.getContractFactory("FingerprintRegistry");
  const contract = await FingerprintRegistry.deploy();
  await contract.waitForDeployment();

  const deployTx = contract.deploymentTransaction();
  const receipt = await deployTx.wait();
  const address = await contract.getAddress();

  console.log("========================================");
  console.log("CONTRACT DEPLOYED");
  console.log("========================================");
  console.log("Contract address:", address);
  console.log("Deployment transaction hash:", deployTx.hash);
  console.log("Block number:", receipt.blockNumber);
  console.log("========================================");

  const outputDir = path.join(__dirname, "..", "data", "output");
  fs.mkdirSync(outputDir, { recursive: true });

  fs.writeFileSync(
    path.join(outputDir, "deployment.json"),
    JSON.stringify(
      {
        contract_address: address,
        deployment_tx_hash: deployTx.hash,
        block_number: receipt.blockNumber,
        network: "localhost",
        rpc_url: "http://127.0.0.1:8545",
      },
      null,
      2
    )
  );

  const artifact = await hre.artifacts.readArtifact("FingerprintRegistry");
  fs.writeFileSync(
    path.join(outputDir, "FingerprintRegistry.abi.json"),
    JSON.stringify(artifact.abi, null, 2)
  );

  console.log("Wrote data/output/deployment.json");
  console.log("Wrote data/output/FingerprintRegistry.abi.json");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
