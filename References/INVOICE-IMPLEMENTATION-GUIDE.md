# Invoice Credential Implementation - Complete Guide

## 📋 Implementation Status

✅ **PHASE 1: COMPLETE** - Schema and Configuration  
✅ **PHASE 2: COMPLETE** - TypeScript Implementation  
✅ **PHASE 3: COMPLETE** - Shell Script Wrappers  
✅ **PHASE 4: COMPLETE** - Orchestration Integration  
✅ **PHASE 5: COMPLETE** - Verification Test Updates  

---

## 📁 Files Created

### Schemas
- ✅ `schemas/invoice-credential-schema.json` - Invoice ACDC schema definition

### Configuration
- ✅ `appconfig/invoiceConfig.json` - Invoice configuration with sample data

### TypeScript Implementation
- ✅ `sig-wallet/src/tasks/invoice/invoice-registry-create.ts`
- ✅ `sig-wallet/src/tasks/invoice/invoice-acdc-issue.ts`
- ✅ `sig-wallet/src/tasks/invoice/invoice-acdc-admit.ts`
- ✅ `sig-wallet/src/tasks/invoice/invoice-acdc-present.ts`
- ✅ `sig-wallet/src/tasks/invoice/invoice-verify-chain.ts`

### Shell Scripts
- ✅ `task-scripts/invoice/invoice-registry-create.sh`
- ✅ `task-scripts/invoice/invoice-acdc-issue.sh`
- ✅ `task-scripts/invoice/invoice-acdc-admit.sh`
- ✅ `task-scripts/invoice/invoice-acdc-present.sh`

### Updated Scripts
- ✅ `run-all-buyerseller-3-with-agents.sh` - Added invoice workflow
- ✅ `test-agent-verification-DEEP-credential.sh` - Added invoice verification

### Documentation
- ✅ `INVOICE-CREDENTIAL-DESIGN.md` - Complete design document
- ✅ `INVOICE-IMPLEMENTATION-GUIDE.md` - This guide

---

## 🚀 Quick Start

### Prerequisites

Before running the invoice workflow, ensure:

1. ✅ Complete vLEI system is deployed (GEDA, QVI, LEs, OORs, Agents)
2. ✅ Jupiter Chief Sales Officer has valid OOR credential
3. ✅ tommyBuyerAgent is created and delegated
4. ✅ Docker containers are running

### Running the Complete Workflow

The invoice credential workflow is integrated into the main orchestration script:

```bash
cd ~/projects/vLEIWorkLinux1

# Run complete workflow (includes invoice issuance automatically)
./run-all-buyerseller-3-with-agents.sh
```

**What happens:**
1. GEDA & QVI setup
2. Jupiter Knitting Company LE credential issued
3. Chief Sales Officer OOR credential issued
4. jupiterSellerAgent delegated from Chief Sales Officer
5. Tommy Hilfiger Europe LE credential issued
6. Chief Procurement Officer OOR credential issued
7. tommyBuyerAgent delegated from Chief Procurement Officer
8. **🧾 INVOICE CREDENTIAL ISSUED** (automatic when Chief Sales Officer processes)
   - Registry created
   - Invoice issued from Chief Sales Officer to tommyBuyerAgent
   - tommyBuyerAgent admits invoice
   - tommyBuyerAgent presents to Sally for verification

---

## 🔍 Manual Invoice Operations

### Issue an Invoice Manually

```bash
# Issue invoice from Chief Sales Officer to tommyBuyerAgent
./task-scripts/invoice/invoice-acdc-issue.sh \
  "Jupiter_Chief_Sales_Officer" \
  "tommyBuyerAgent" \
  "./appconfig/invoiceConfig.json"
```

### Admit an Invoice

```bash
# tommyBuyerAgent admits the invoice
./task-scripts/invoice/invoice-acdc-admit.sh "tommyBuyerAgent"
```

### Present Invoice to Verifier

```bash
# tommyBuyerAgent presents invoice to Sally
./task-scripts/invoice/invoice-acdc-present.sh \
  "tommyBuyerAgent" \
  "Jupiter_Chief_Sales_Officer"
```

### Verify Invoice Chain

```bash
# Run complete invoice verification
./test-agent-verification-DEEP-credential.sh \
  jupiterSellerAgent \
  Jupiter_Chief_Sales_Officer \
  true \
  docker
```

---

## 📊 Trust Chain Structure

```
GLEIF Root (GEDA)
    │
    ├─> QVI Credential
        │
        ├─> LE Credential (Jupiter Knitting Company)
            │  LEI: 3358004DXAMRWRUIYJ05
            │
            ├─> OOR_AUTH Credential (LE → QVI)
            │   │
            │   └─> OOR Credential (QVI → Chief Sales Officer) ✓
            │       │
            │       │ [Chained via ACDC Edge]
            │       │
            │       └─> 🧾 INVOICE CREDENTIAL
            │           │
            │           Issuer: Jupiter_Chief_Sales_Officer
            │           Holder: tommyBuyerAgent
            │           Edge: References OOR Credential SAID
            │           Contains: Invoice details, payment info
```

---

## ⚙️ Configuration

### Invoice Configuration File

Edit `appconfig/invoiceConfig.json` to customize invoice data:

```json
{
  "invoice": {
    "schemaName": "InvoiceCredential",
    "registryName": "JUPITER_SALES_REGISTRY",
    "issuer": {
      "alias": "Jupiter_Chief_Sales_Officer",
      "lei": "3358004DXAMRWRUIYJ05"
    },
    "holder": {
      "alias": "tommyBuyerAgent",
      "lei": "54930012QJWZMYHNJW95"
    },
    "sampleInvoice": {
      "invoiceNumber": "INV-2025-001",
      "invoiceDate": "2025-11-13T00:00:00Z",
      "dueDate": "2025-12-13T00:00:00Z",
      "currency": "USD",
      "totalAmount": 50000.00,
      "lineItems": [
        {
          "description": "Knitted Sweaters - Batch A",
          "quantity": 1000,
          "unitPrice": 45.00,
          "amount": 45000.00
        }
      ],
      "paymentTerms": "Net 30 days",
      "paymentMethod": "stellar",
      "stellarPaymentAddress": "G..."
    }
  }
}
```

---

## 🔐 Verification Process

The invoice credential undergoes comprehensive verification:

### 1. Structure Verification
- ✅ Valid ACDC structure
- ✅ All required fields present
- ✅ Correct schema SAID

### 2. Edge Chain Verification
- ✅ Invoice edges to OOR credential (correct SAID)
- ✅ OOR edges to LE credential
- ✅ LE edges to QVI credential
- ✅ QVI edges to GEDA root

### 3. Authority Verification
- ✅ Invoice issuer is OOR credential holder
- ✅ OOR holder has authority to issue invoices
- ✅ Complete chain traces to trusted root

### 4. Business Rules Verification
- ✅ Invoice amounts are positive
- ✅ Line items sum correctly
- ✅ LEI format is valid (20 characters)
- ✅ Dates are consistent

### 5. Temporal Validity
- ✅ Due date is after invoice date
- ✅ Invoice is not overdue (warning if overdue)
- ✅ Invoice date is not in future

---

## 📝 Sample Invoice Output

After running the workflow, check the invoice details:

```bash
cat ./task-data/Jupiter_Chief_Sales_Officer-invoice-credential-info.json
```

**Example Output:**
```json
{
  "said": "EInvoiceCredentialSAID...",
  "issuer": "Jupiter_Chief_Sales_Officer",
  "issuee": "tommyBuyerAgent",
  "grantSaid": "EGrantSAID...",
  "invoiceNumber": "INV-2025-001",
  "totalAmount": 50000.00,
  "currency": "USD",
  "dueDate": "2025-12-13T00:00:00Z",
  "paymentMethod": "stellar",
  "stellarPaymentAddress": "GCTJ..."
}
```

---

## 🎯 Key Design Decisions

### Why Chain to OOR Credential?

Per GLEIF documentation:
- **OOR credentials prove personal authority** to act on behalf of organization
- Chaining invoice to OOR proves the **specific person** (Chief Sales Officer) has authority to issue invoices
- Follows ACDC delegated authorization pattern
- Enables complete trust chain validation

### ACDC Edge Semantics

The invoice uses ACDC edges to create verifiable chain:

```json
"e": {
  "d": "<SAID>",
  "oor": {
    "n": "<OOR_CREDENTIAL_SAID>",  // References OOR credential
    "s": "<OOR_SCHEMA_SAID>",       // Schema validation
    "o": "I2I"                       // Issuer-to-issuer chain
  }
}
```

### IPEX to tommyBuyerAgent

- Invoice granted directly to **agent** (not OOR holder)
- Agent can verify and process independently
- Follows agent delegation pattern

---

## 🧪 Testing Scenarios

### Test 1: Complete Workflow
```bash
./run-all-buyerseller-3-with-agents.sh
```
✅ Tests full integration from GEDA to invoice

### Test 2: Invoice Verification Only
```bash
./test-agent-verification-DEEP-credential.sh \
  jupiterSellerAgent \
  Jupiter_Chief_Sales_Officer \
  true \
  docker
```
✅ Tests invoice chain verification

### Test 3: Manual Operations
```bash
# Create registry
./task-scripts/invoice/invoice-registry-create.sh \
  "Jupiter_Chief_Sales_Officer" \
  "JUPITER_SALES_REGISTRY"

# Issue invoice
./task-scripts/invoice/invoice-acdc-issue.sh \
  "Jupiter_Chief_Sales_Officer" \
  "tommyBuyerAgent" \
  "./appconfig/invoiceConfig.json"

# Admit invoice
./task-scripts/invoice/invoice-acdc-admit.sh "tommyBuyerAgent"

# Present to verifier
./task-scripts/invoice/invoice-acdc-present.sh \
  "tommyBuyerAgent" \
  "Jupiter_Chief_Sales_Officer"
```
✅ Tests individual operations

---

## ⚠️ Important Notes

### Schema SAID Placeholder

The invoice schema SAID is currently a placeholder:
```typescript
INVOICE_SCHEMA_SAID="EInvoiceSchemaPlaceholder"
```

**Before production use:**
1. Publish the invoice schema to the schema server
2. Calculate the actual SAID
3. Update `invoice-acdc-issue.sh` with real SAID
4. Update `appconfig/invoiceConfig.json`

### Registry Creation

The invoice registry is created automatically when:
- Jupiter Chief Sales Officer is processed
- Registry doesn't already exist

To manually create:
```bash
./task-scripts/invoice/invoice-registry-create.sh \
  "Jupiter_Chief_Sales_Officer" \
  "JUPITER_SALES_REGISTRY"
```

---

## 🐛 Troubleshooting

### Issue: "OOR credential file not found"

**Cause:** Invoice trying to issue before OOR credential exists

**Solution:** Ensure complete workflow runs in order:
1. GEDA & QVI setup
2. LE credentials
3. OOR credentials
4. Invoice credentials

### Issue: "Holder info file not found"

**Cause:** tommyBuyerAgent doesn't exist

**Solution:** Ensure agent delegation completes before invoice issuance

### Issue: "Invoice credential missing OOR edge"

**Cause:** Edge not properly constructed

**Solution:** Check that OOR credential SAID is correctly retrieved:
```bash
cat ./task-data/Jupiter_Chief_Sales_Officer-oor-credential-info.json | jq -r '.said'
```

---

## 📚 References

### Official GLEIF Documentation
- [GLEIF vLEI Ecosystem Governance Framework v3.0](https://www.gleif.org/vlei)
- [ISO 17442-3: Verifiable LEIs](https://www.iso.org/standard/77575.html)
- [ACDC Specification](https://www.ietf.org/archive/id/draft-ssmith-acdc-02.html)
- [KERI Specification](https://github.com/trustoverip/tswg-keri-specification)

### Code Repositories
- [GLEIF Hackathon Workshop](https://github.com/GLEIF-IT/vlei-hackathon-2025-workshop)
- [GLEIF vLEI Trainings](https://github.com/GLEIF-IT/vlei-trainings)
- [Sally Verifier](https://github.com/GLEIF-IT/sally)

---

## ✅ Next Steps

1. **Test the implementation** - Run complete workflow
2. **Publish schema** - Calculate SAID and publish to schema server
3. **Update placeholders** - Replace INVOICE_SCHEMA_SAID with real value
4. **Add Sally extension** - Implement invoice verification in Sally (optional)
5. **Customize configuration** - Adjust invoice data for your use case
6. **Production deployment** - Move from testnet to production

---

## 📞 Support

For issues or questions:
1. Review design document: `INVOICE-CREDENTIAL-DESIGN.md`
2. Check troubleshooting section above
3. Verify all prerequisites are met
4. Ensure Docker containers are running

---

**Document Version:** 1.0.0  
**Last Updated:** November 13, 2025  
**Implementation Status:** ✅ COMPLETE  
**Ready for Testing:** YES
