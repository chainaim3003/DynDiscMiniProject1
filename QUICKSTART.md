# vLEI Workshop - Quick Start Guide

## Overview

This project implements a complete vLEI (Verifiable Legal Entity Identifier) credential workflow including:

- **GEDA** (GLEIF External Delegated AID) - Root of trust
- **QVI** (Qualified vLEI Issuer) - Issues LE and OOR credentials
- **LE** (Legal Entity) - Organizations (Jupiter Knitting, Tommy Hilfiger)
- **OOR** (Official Organizational Role) - Person credentials
- **Agent Delegation** - AI agents delegated from OOR holders
- **Invoice Credentials** - Self-attested invoice credentials with IPEX grant/admit

---

## 🚀 Quick Start (4 Steps)

### Step 1: Stop Any Existing Environment
```bash
./stop.sh
```

### Step 2: Setup Environment
```bash
./setup.sh
```

### Step 3: Deploy Services
```bash
./deploy.sh
```

### Step 4: SAIDify Schema & Run Workflow
```bash
./saidify-and-restart.sh
./run-all-buyerseller-4C-with-agents.sh
```

---

## 📋 One-Liner (Complete Fresh Start)

```bash
./stop.sh && ./setup.sh && ./deploy.sh && ./saidify-and-restart.sh && ./run-all-buyerseller-4C-with-agents.sh
```

---

## 📁 Key Scripts

| Script | Purpose |
|--------|---------|
| `./stop.sh` | Stop containers, remove volumes, clean task data |
| `./setup.sh` | Copy files, fix line endings, build Docker images |
| `./deploy.sh` | Start all services |
| `./saidify-and-restart.sh` | **NEW!** SAIDify schema + restart container (combines 2 steps) |
| `./saidify-with-docker.sh` | SAIDify schema using keripy (called by above) |
| `./run-all-buyerseller-4C-with-agents.sh` | Run the complete workflow |

---

## 🔧 What Each Script Does

### `./stop.sh`
- Stops all Docker containers
- Removes Docker volumes (clears KERIA cache)
- Removes Docker network
- Cleans task-data directory

### `./setup.sh`
- Copies files from Windows to Linux (WSL)
- Installs dependencies (dos2unix, python3, jq)
- Fixes Windows line endings (CRLF → LF)
- Makes scripts executable
- Builds Docker images with `--no-cache`

### `./deploy.sh`
- Creates Docker network
- Starts all services (schema, witnesses, KERIA, verifier, webhook)
- Runs health checks

### `./saidify-and-restart.sh` ⭐ NEW
- Runs SAIDification using keripy in Docker
- Updates `appconfig/schemaSaids.json` with correct SAID
- Restarts schema container
- Waits for container to be ready
- Verifies schema accessibility

### `./run-all-buyerseller-4C-with-agents.sh`
- Creates GEDA and QVI
- Creates organizations (Jupiter, Tommy)
- Issues LE and OOR credentials
- Delegates AI agents (jupiterSellerAgent, tommyBuyerAgent)
- Creates invoice registry
- Issues self-attested invoice credential
- Sends IPEX grant from seller to buyer
- Buyer admits the grant

---

## 📊 Expected Output

When everything works correctly, you should see:

```
✓ Self-attested invoice credential created: E...
  Issuer: E...
  Issuee: E... (same as issuer)
  Self-attested: YES ✓

✓ IPEX grant sent successfully
  Grant result: {"said":"E..."}

✓ IPEX grant admitted by tommyBuyerAgent
✓ Invoice credential now available in tommyBuyerAgent's KERIA storage
```

---

## 🔍 Troubleshooting

### "Additional properties are not allowed" Error
**Cause:** Schema SAID mismatch
**Fix:** Run `./saidify-and-restart.sh`

### IPEX Admit Fails - No Pending Grant Found
**Cause:** Notification propagation delay
**Fix:** Wait a few seconds and run admit again manually:
```bash
docker compose exec -T tsx-shell tsx \
  sig-wallet/src/tasks/invoice/invoice-ipex-admit.ts \
  docker "" tommyBuyerAgent jupiterSellerAgent
```

### TypeScript Changes Not Taking Effect
**Cause:** Docker container using old code
**Fix:** 
```bash
docker compose build --no-cache tsx-shell
docker compose up -d tsx-shell
```

---

## 📊 Service Ports

| Service | Port(s) | Description |
|---------|---------|-------------|
| Schema Server | 7723 | vLEI schema OOBI endpoints |
| Witnesses | 5642-5647 | 6 witnesses |
| KERIA | 3901-3903 | Admin, HTTP, Boot APIs |
| Verifier (Sally) | 9723 | Credential verification |
| Webhook | 9923 | IPEX presentation webhook |

---

## 📝 Configuration Files

| File | Purpose |
|------|---------|
| `appconfig/schemaSaids.json` | **SINGLE SOURCE OF TRUTH** for schema SAIDs |
| `appconfig/configBuyerSellerAIAgent1.json` | Organization and agent configuration |
| `appconfig/invoiceConfig.json` | Invoice data configuration |
| `schemas/self-attested-invoice.json` | Invoice credential schema |

---

## 🔑 Important Notes

1. **Schema SAIDification is Required**: The schema file has an empty `$id` by default. Running `./saidify-and-restart.sh` computes the SAID using keripy and updates all config files.

2. **KERIA Cache**: If you see schema validation errors, run `./stop.sh` to clear KERIA's cache, then do a fresh start.

3. **Notification Propagation**: IPEX grant notifications may take a few seconds to propagate. The updated scripts include retry logic.

4. **BRANs are Secrets**: The task-data directory contains cryptographic secrets. Never commit to version control.

---

*Last updated: December 1, 2025*
