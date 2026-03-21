# SignifyTS Wallet Implementation

This directory contains the TypeScript implementation for the vLEI Hackathon 2025 Workshop, providing SignifyTS-based functionality for creating, managing, and presenting vLEI credentials.

## Overview

The sig-wallet implements a complete vLEI credential workflow using SignifyTS and KERIA, demonstrating both Official Organizational Role (OOR) and Engagement Context Role (ECR) credential types. The implementation follows KERI best practices for decentralized identity management.

## Architecture

The implementation is organized into several key components:

- **Client utilities** (`src/client/`) - Core functionality for KERI operations
- **Task implementations** (`src/tasks/`) - Specific workflow steps organized by entity
- **Common utilities** (`src/tasks/common/`) - Shared functionality across tasks

## Key Components

### Client Utilities (`src/client/`)

- `credentials.ts` - Credential issuance, admission, and presentation functions
- `identifiers.ts` - AID creation and management utilities
- `oobis.ts` - Out-of-band identifier resolution
- `operations.ts` - KERI operation handling and waiting
- `resolve-env.ts` - Environment configuration management

### Task Implementations (`src/tasks/`)

#### GEDA Tasks (`geda/`)
- `geda-aid-create.ts` - Creates the GEDA AID using KERIpy
- `geda-registry-create.ts` - Creates credential registry for GEDA
- `geda-acdc-issue-qvi.ts` - Issues QVI credential from GEDA
- `geda-delegate-approve.ts` - Approves QVI delegation request

#### QVI Tasks (`qvi/`)
- `qvi-aid-delegate-create.ts` - Creates QVI as GEDA delegate
- `qvi-aid-delegate-finish.ts` - Completes QVI delegation
- `qvi-registry-create.ts` - Creates QVI credential registry
- `qvi-acdc-issue-le.ts` - Issues LE credential with QVI edge
- `qvi-acdc-admit-qvi.ts` - Admits QVI credential
- `qvi-acdc-present-qvi.ts` - Presents QVI credential to verifier
- `qvi-acdc-issue-oor.ts` - Issues OOR credential with OOR Auth edge
- `qvi-acdc-admit-oor-auth.ts` - Admits OOR Auth credential
- `qvi-acdc-issue-ecr.ts` - Issues ECR credential with ECR Auth edge
- `qvi-acdc-admit-ecr-auth.ts` - Admits ECR Auth credential

#### LE Tasks (`le/`)
- `le-aid-create.ts` - Creates LE AID
- `le-registry-create.ts` - Creates LE credential registry
- `le-acdc-issue-le.ts` - Issues LE credential
- `le-acdc-admit-le.ts` - Admits LE credential from QVI
- `le-acdc-present-le.ts` - Presents LE credential to verifier
- `le-acdc-issue-oor-auth.ts` - Issues OOR Auth credential with LE edge
- `le-acdc-issue-ecr-auth.ts` - Issues ECR Auth credential with LE edge

#### Person Tasks (`person/`)
- `person-aid-create.ts` - Creates Person AID
- `person-acdc-admit-oor.ts` - Admits OOR credential from QVI
- `person-acdc-present-oor.ts` - Presents OOR credential to verifier
- `person-acdc-admit-ecr.ts` - Admits ECR credential from QVI

#### Common Tasks (`common/`)
- `oobi-resolve.ts` - Generic OOBI resolution functionality
- `oobi-get.ts` - OOBI retrieval utilities

## Credential Types

### Official Organizational Role (OOR) Credentials

OOR credentials represent official positions within an organization:

1. **OOR Auth Credential** - Authorization credential issued by LE to QVI
   - Schema: `EKA57bKBKxr_kN7iN5i7lMUxpMG-s19dRcmov1iDxz-E`
   - Edge: Points to LE credential
   - Rules: Usage, issuance, and privacy disclaimers

2. **OOR Credential** - Official role credential issued by QVI to Person
   - Schema: `EBNaNu-M9P5cgrnfl2Fvymy4E_jvxxyjb70PRtiANlJy`
   - Edge: Points to OOR Auth credential with `I2I` operator
   - Rules: Usage, issuance, and privacy disclaimers

### Engagement Context Role (ECR) Credentials

ECR credentials represent functional roles within organizational contexts:

1. **ECR Auth Credential** - Authorization credential issued by LE to QVI
   - Schema: `EH6ekLjSr8V32WyFbGe1zXjTzFs9PkTYmupJ9H65O14g`
   - Edge: Points to LE credential
   - Rules: Usage, issuance, and privacy disclaimers

2. **ECR Credential** - Engagement role credential issued by QVI to Person
   - Schema: `EEy9PkikFcANV1l7EHukCeXqrzT1hNZjGlUk7wuMO5jw`
   - Edge: Points to ECR Auth credential with `I2I` operator
   - Rules: Usage, issuance, and privacy disclaimers
   - Special: Requires `u` field at both top level and in attributes

## Trust Chain Flow

The implementation demonstrates a complete vLEI trust chain:

1. **GEDA** creates QVI as delegate
2. **QVI** issues LE credential to Legal Entity
3. **LE** issues OOR/ECR Auth credentials to QVI
4. **QVI** issues OOR/ECR credentials to Person
5. **Person** presents credentials to Verifier (Sally)

## Key Functions

### Credential Management
- `issueCredential()` - Issues credentials with edges and rules
- `ipexGrantCredential()` - Grants credentials using IPEX protocol
- `ipexAdmitGrant()` - Admits granted credentials
- `grantCredential()` - Presents credentials to verifiers

### Identifier Management
- `getOrCreateClient()` - Creates or retrieves SignifyTS clients
- `createAid()` - Creates new autonomic identifiers
- `createRegistry()` - Creates credential registries

### OOBI Resolution
- `resolveOobi()` - Resolves out-of-band identifiers
- `resolveEnvironment()` - Configures client environments

## Environment Configuration

The implementation supports both Docker and testnet environments through the `resolveEnvironment()` function, allowing seamless switching between local development and production deployments.

## Schema Support

All vLEI credential schemas are supported:
- QVI Credential: `EBfdlu8R27Fbx-ehrqwImnK-8Cm79sqbAQ4MmvEAYqao`
- LE Credential: `ENPXp1vQzRF6JwIuS-mp2U8Uf1MoADoP_GqQ62VsDZWY`
- OOR Auth Credential: `EKA57bKBKxr_kN7iN5i7lMUxpMG-s19dRcmov1iDxz-E`
- OOR Credential: `EBNaNu-M9P5cgrnfl2Fvymy4E_jvxxyjb70PRtiANlJy`
- ECR Auth Credential: `EH6ekLjSr8V32WyFbGe1zXjTzFs9PkTYmupJ9H65O14g`
- ECR Credential: `EEy9PkikFcANV1l7EHukCeXqrzT1hNZjGlUk7wuMO5jw`

## Usage

The sig-wallet is designed to be executed through shell scripts in the `task-scripts/` directory, which orchestrate the complete vLEI workflow. Each TypeScript file implements a specific step in the credential issuance and presentation process.

For detailed usage instructions, see the main workshop README.md file.
