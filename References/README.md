# vLEI Hackathon 2025 Workshop

This workshop demonstrates how to build a complete verifiable Legal Entity Identifier (vLEI) credential system using KERI (Key Event Receipt Infrastructure) and ACDC (Authentic Chained Data Container) technologies. You'll learn to create, issue, and verify organizational identity credentials that establish trust chains between legal entities and their representatives.

## What You'll Learn

**Core Technologies:**
- **KERI**: A decentralized key management system that creates self-certifying identifiers
- **ACDC**: Verifiable credentials built on KERI that can be chained together to form trust relationships
- **vLEI**: Legal entity credentials that prove organizational identity and roles

**Key Concepts:**
- **Trust Chains**: How credentials reference each other to build verifiable organizational hierarchies
- **Delegation**: How organizations can delegate credential issuance authority to trusted parties
- **Role-Based Credentials**: Different types of credentials for official roles vs. engagement contexts

**Practical Skills:**
- Creating and managing KERI identifiers
- Issuing and presenting verifiable credentials
- Building complete organizational identity workflows

## Workshop Modules

This workshop has three separate modules:

1. **Module 1 - vLEI OOR Permissioned Action**
   - vLEI Issue-Hold-Verify Fundamentals
   - Verifying a vLEI Official Organizational Role (OOR) ACDC
2. **(WIP) Module 2 - vLEI-based smart contract access - CCID by Chainlink**
   - Turning a vLEI into a CCID
   - vLEI-based CCID smart contract access
3. **(WIP) Module 3 - Confidentiality with ESSR - Veridian by Cardano**
   - ESSR confidentiality agent to agent with KERI challenge response

## Architecture

This workshop uses a simplified, single signature identifier setup with one identifier for the GEDA, QVI, LE, and Person to show an end-to-end vLEI credential issuance and presentation workflow using both KERIpy and SignifyTS. The GEDA, or GLEIF External Delegated Identifier, is a KERI identifier made with the KERIpy KLI tool. The QVI, LE, and Person AID are created with Signify TS and KERIA. All of this is built on a three witness, one KERIA, and one Sally verifier deployment. The QVI identifier is a delegated identifier that is a delegate of the GEDA delegator.

The workshop demonstrates two types of vLEI credentials: Official Organizational Role (OOR) credentials and Engagement Context Role (ECR) credentials. Both follow similar trust chain patterns but serve different purposes in organizational identity verification.

## How It Works

**The Trust Chain:**
1. **GEDA** (Global Legal Entity Identifier Foundation) creates a **QVI** (Qualified vLEI Issuer) as its delegate
2. **QVI** issues a **Legal Entity** credential to a company, proving the company's identity
3. **Legal Entity** authorizes the QVI to issue role credentials to its employees
4. **QVI** issues **OOR** (official roles) and **ECR** (engagement roles) credentials to people
5. **People** present their credentials to verifiers to prove their organizational roles

**Why This Matters:**
- Organizations can prove their legal identity without relying on centralized authorities
- Role credentials are cryptographically linked to the organization's verified identity
- The entire trust chain can be verified independently by any party
- Credentials can be revoked or updated as organizational structures change

See the `./images/vLEI-Workshop-architecture.png` diagram for a visual representation of the identifiers in green, the delegation between the GEDA and QVI, the credentials in yellow, and the KERIA, Witness, and Verifier (sally) infrastructure at the bottom of the diagram.

## Quick Start

If you want to see the complete vLEI workflow in action immediately:

```bash
# Start the environment and run the complete workflow
./stop.sh

docker compose build # rebuilds builds the gleif/wkshp-tsx-shell image

./deploy.sh 

./run-all.sh
```

This will create all identifiers, issue all credentials, and demonstrate the complete trust chain from GEDA to Person credentials.

## Workshop Instructions

Follow the module specific instructions as

### Module 1 - vLEI OOR Permissioned Action

Abbreviations and names:
- KERI: Key Event Receipt Infrastructure
- ACDC: Authentic Chained Data Containers
- CESR: Composable Event Streaming
- KERIA: KERI Agent server for Signify
- SignifyTS: Typescript implementation of the Signify edge client protocol
- vLEI: verifiable Legal Entity Identifier
- OOBI: Out of Band Identifier URL
- QVI: Qualified vLEI Issuer
- LE: Legal Entity
- OOR: Official Organizational Role
- ECR: Engagement Context Role
- OOR Auth: Official Organizational Role Authorization
- ECR Auth: Engagement Context Role authorization
- AID: Autonomic Identifier
- IPEX: Issuance and Presentation EXchange Protocol

#### vLEI Module Scripts

This section explains the purpose and contents of each script. You may skip to the [instructions](#instructions) section to get started if you are ready.

- `docker compose build` builds the tsx-shell image
- ./deploy.sh sets up the following components:
  - KERIA server
  - Three witnesses
  - vLEI Server (ACDC schema host)
- ./sig-wallet contains the Typescript code for:
  - Setting up each of the Signify Controllers and their KERIA agents.
  - Resolving Schema OOBIs of vLEI schemas (QVI, LE, OOR Auth, ECR Auth, OOR, ECR)
  - Creating KERI AIDs for the QVI, LE, and Person holders.
  - Implementing complete credential workflows for both OOR and ECR credential types.
- ./stop.sh shuts down the components started up by ./deploy.sh  
- ./task-scripts/geda/geda-aid-create.sh
  - Uses the KLI to create the GEDA AID using KERIpy rather than KERIA and SignifyTS. This mirrors what occurs in production.
  - This identifier will delegate to the QVI AID by the QVI creating a delegation request and the GEDA approving it.
- ./task-scripts/qvi/qvi-aid-delegate-create.sh
  - uses the appropriate script in `./sig-wallet/src` to create the QVI AID as a delegated AID from the GEDA AID.
- ./task-scripts/le/le-aid-create.sh
  - uses the appropriate script in `./sig-wallet/src` to create the LE AID.
- ./task-scripts/person/person-aid-create.sh
  - uses the appropriate script in `./sig-wallet/src` to create the person AID that will receive the OOR and ECR credentials.
- ./task-scripts/geda/geda-acdc-issue-qvi.sh  
  - uses the appropriate script in `./sig-wallet/src` to issue the QVI credential from the GEDA AID to the QVI AID.
- ./task-scripts/qvi/qvi-acdc-issue-le.sh
  - uses the appropriate script in `./sig-wallet/src` to issue the LE credential from the QVI AID to the LE AID, chaining the LE to the QVI credential.
- ./task-scripts/le/le-acdc-issue-oor-auth.sh
  - uses the appropriate script in `./sig-wallet/src` to issue the OOR Auth credential from the LE AID to the QVI AID, chaining the OOR Auth credential to the LE credential.
- ./task-scripts/qvi/qvi-acdc-issue-oor.sh
  - uses the appropriate script in `./sig-wallet/src` to issue the OOR credential from the QVI AID to the Person AID, chaining the OOR credential to the OOR Auth credential.
- ./task-scripts/le/le-acdc-issue-ecr-auth.sh
  - uses the appropriate script in `./sig-wallet/src` to issue the ECR Auth credential from the LE AID to the QVI AID, chaining the ECR Auth credential to the LE credential.
- ./task-scripts/qvi/qvi-acdc-issue-ecr.sh
  - uses the appropriate script in `./sig-wallet/src` to issue the ECR credential from the QVI AID to the Person AID, chaining the ECR credential to the ECR Auth credential.


#### Complete Workflow

The workshop demonstrates a complete vLEI trust chain with both OOR and ECR credentials:

1. **GEDA Setup** - Creates GEDA AID and delegates to QVI
2. **QVI Setup** - Creates QVI AID as GEDA delegate and issues QVI credential
3. **LE Setup** - Creates LE AID and issues LE credential with QVI edge
4. **Person Setup** - Creates Person AID for credential receipt
5. **OOR Credentials** - Issues OOR Auth credential from LE to QVI, then OOR credential from QVI to Person
6. **ECR Credentials** - Issues ECR Auth credential from LE to QVI, then ECR credential from QVI to Person
7. **Verification** - Presents all credentials to Sally verifier

#### vLEI Module Instructions

To run the complete workflow:

```bash
./stop.sh

docker compose build # rebuilds builds the gleif/wkshp-tsx-shell image

./deploy.sh

./run-all.sh
```

This will execute all scripts in sequence, demonstrating the complete vLEI credential issuance and presentation process for both OOR and ECR credential types.
