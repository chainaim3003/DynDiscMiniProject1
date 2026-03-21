# Build Instructions for vLEI Configuration-Driven System

## Overview

After modifying TypeScript files to support configuration-driven parameters, you must rebuild the JavaScript files. This document provides step-by-step instructions.

---

## Modified Files

The following TypeScript files have been modified to accept configuration parameters:

### 1. `sig-wallet/src/tasks/le/le-acdc-issue-oor-auth.ts`
**Change**: Added `leLei` parameter (args[10])
- **Before**: LEI was hardcoded as `'254900OPPU84GM83MG36'`
- **After**: LEI is passed as parameter from configuration

### 2. `sig-wallet/src/tasks/qvi/qvi-acdc-issue-oor.ts`
**Change**: Added `leLei` parameter (args[9])
- **Before**: LEI was hardcoded as `'254900OPPU84GM83MG36'`
- **After**: LEI is passed as parameter from configuration

---

## Build Process

### Prerequisites

- Node.js (v18 or higher)
- npm (comes with Node.js)

### Step 1: Navigate to sig-wallet Directory

```bash
cd C:\SATHYA\CHAINAIM3003\mcp-servers\stellarboston\vLEI1\vLEIWorkLinux1\sig-wallet
```

Or on Linux/WSL:
```bash
cd /mnt/c/SATHYA/CHAINAIM3003/mcp-servers/stellarboston/vLEI1/vLEIWorkLinux1/sig-wallet
```

### Step 2: Install Dependencies (if not already installed)

```bash
npm install
```

### Step 3: Build TypeScript Files

```bash
npm run build
```

This command runs `tsc` (TypeScript compiler) and generates JavaScript files in the same directory structure.

### Step 4: Verify Build

Check that JavaScript files were generated:

```bash
ls -la src/tasks/le/le-acdc-issue-oor-auth.js
ls -la src/tasks/qvi/qvi-acdc-issue-oor.js
```

Both files should exist with recent timestamps.

---

## What Gets Generated

The TypeScript compiler (`tsc`) will generate corresponding `.js` files:

```
sig-wallet/src/tasks/
├── le/
│   ├── le-acdc-issue-oor-auth.ts (modified)
│   └── le-acdc-issue-oor-auth.js (generated)
└── qvi/
    ├── qvi-acdc-issue-oor.ts (modified)
    └── qvi-acdc-issue-oor.js (generated)
```

---

## Build Errors and Troubleshooting

### Common Issues

#### 1. TypeScript Version Mismatch
```
Error: Cannot find module 'typescript'
```
**Solution**: 
```bash
npm install typescript@latest --save-dev
```

#### 2. Missing Dependencies
```
Error: Cannot find module 'signify-ts'
```
**Solution**:
```bash
npm install
```

#### 3. Permission Errors on Windows
**Solution**: Run command prompt or PowerShell as Administrator

#### 4. Path Issues on WSL
If building on WSL, ensure you're in the correct mounted directory:
```bash
cd /mnt/c/SATHYA/CHAINAIM3003/mcp-servers/stellarboston/vLEI1/vLEIWorkLinux1/sig-wallet
```

---

## Verification Steps

After building, verify the changes are correctly implemented:

### 1. Check JavaScript File Content

Open the generated JavaScript files and verify they contain the parameter handling:

**le-acdc-issue-oor-auth.js** should have:
```javascript
const leLei = args[10];
// ...
LEI: leLei, // LEI from configuration
```

**qvi-acdc-issue-oor.js** should have:
```javascript
const leLei = args[9];
// ...
LEI: leLei, // LEI from configuration
```

### 2. Test with Configuration

Run the complete flow:
```bash
./run-all-buyerseller-2.sh
```

Look for these success indicators:
- ✓ Green checkmarks indicating configuration usage
- Correct LEI values in output (3358004DXAMRWRUIYJ05 for Jupiter, 54930012QJWZMYHNJW95 for Tommy)
- No errors about undefined parameters

---

## Build Script Details

The `package.json` contains:
```json
{
  "scripts": {
    "build": "tsc"
  }
}
```

The TypeScript configuration (`tsconfig.json`) controls:
- Output directory (same as source)
- Module system (ES modules)
- Target JavaScript version

---

## Alternative: Docker Build

If you're using Docker for development:

```bash
docker compose exec tsx-shell bash
cd /vlei
npm run build
```

---

## Integration with Shell Scripts

After building, the shell scripts will automatically use the updated JavaScript:

### Modified Shell Scripts (Already Updated)

1. **qvi-acdc-issue-le.sh**
   - Now accepts `$1` parameter for LEI
   - Passes LEI to TypeScript: `"${LE_LEI}"`

2. **le-acdc-issue-oor-auth.sh**
   - Now accepts `$1` (person name), `$2` (role), `$3` (LEI)
   - Passes all three to TypeScript: `"${PERSON_NAME}" "${PERSON_OOR}" "${LE_LEI}"`

3. **qvi-acdc-issue-oor.sh**
   - Now accepts `$1` (person name), `$2` (role), `$3` (LEI)
   - Passes all three to TypeScript: `"${PERSON_NAME}" "${PERSON_OOR}" "${LE_LEI}"`

---

## Testing the Changes

### Test 1: Single Organization Flow
```bash
# This should work with Jupiter's LEI: 3358004DXAMRWRUIYJ05
./run-all-buyerseller-2.sh
```

### Test 2: Verify Credential Content
After running, check credential files:
```bash
jq . task-data/le-credential-info.json
jq . task-data/oor-credential-info.json
```

Verify the LEI matches the configuration value.

### Test 3: Sally Verifier
Check Sally verifier logs to ensure credentials are valid:
```bash
docker compose logs verifier | tail -50
```

---

## Continuous Integration

For automated builds, add to your CI/CD pipeline:

```yaml
# Example GitHub Actions
- name: Build TypeScript
  run: |
    cd sig-wallet
    npm install
    npm run build
```

---

## Rollback Procedure

If you need to revert changes:

1. **Restore original TypeScript files** from Git:
   ```bash
   git checkout sig-wallet/src/tasks/le/le-acdc-issue-oor-auth.ts
   git checkout sig-wallet/src/tasks/qvi/qvi-acdc-issue-oor.ts
   ```

2. **Rebuild**:
   ```bash
   cd sig-wallet
   npm run build
   ```

3. **Restore original shell scripts**:
   ```bash
   git checkout task-scripts/qvi/qvi-acdc-issue-le.sh
   git checkout task-scripts/le/le-acdc-issue-oor-auth.sh
   git checkout task-scripts/qvi/qvi-acdc-issue-oor.sh
   ```

---

## Summary

✅ **What Was Changed**:
- TypeScript files now accept configuration parameters
- Shell scripts pass parameters from JSON config
- No more hardcoded LEI, person names, or roles

✅ **What You Need To Do**:
1. Run `npm run build` in sig-wallet directory
2. Test with `./run-all-buyerseller-2.sh`
3. Verify credentials contain correct values

✅ **Expected Result**:
- Jupiter gets LEI: 3358004DXAMRWRUIYJ05
- Tommy gets LEI: 54930012QJWZMYHNJW95
- Person roles match configuration
- All credentials validated by Sally

---

## Questions or Issues?

If you encounter problems:
1. Check TypeScript compilation errors
2. Verify parameter order in shell scripts
3. Review Docker logs
4. Consult official GLEIF documentation: https://github.com/GLEIF-IT/vlei-hackathon-2025-workshop

---

**Document Version**: 1.0  
**Date**: November 10, 2025  
**Related Documents**:
- understanding-3.md
- complete-session.md
- run-all-buyerseller-2.sh
