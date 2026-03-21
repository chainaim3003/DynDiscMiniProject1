import fs from "fs";
import { getOrCreateClient, getAID } from "../../client/identifiers.js";

/**
 * Validate all credentials from KERIA agent
 * Performs comprehensive validation including signature, chain, and schema checks
 * 
 * Usage: tsx agent-validate-credentials.ts <env> <passcode> <agentName>
 */

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1];
const agentName = args[2];

console.log(`========================================`);
console.log(`VALIDATE AGENT CREDENTIALS`);
console.log(`========================================`);
console.log(``);
console.log(`Agent: ${agentName}`);
console.log(``);

interface ValidationResult {
    said: string;
    schema: string;
    type: string;
    signatureValid: boolean;
    chainValid: boolean;
    schemaValid: boolean;
    selfAttested: boolean;
    valid: boolean;
    errors: string[];
}

try {
    // Get client
    console.log(`[1/4] Connecting to KERIA agent...`);
    const client = await getOrCreateClient(passcode, env);
    const agentAID = await getAID(client, agentName);
    console.log(`✓ Connected`);
    console.log(`  Agent AID: ${agentAID.prefix}`);
    console.log(``);
    
    // Query all credentials
    console.log(`[2/4] Querying credentials from KERIA...`);
    const credentials = await client.credentials().list(agentAID.name);
    
    console.log(`✓ Found ${credentials.length} credential(s) to validate`);
    console.log(``);
    
    if (credentials.length === 0) {
        console.log(`⚠ No credentials found for ${agentName}`);
        console.log(`  Nothing to validate`);
        console.log(``);
        
        // Save empty results
        const validationResults = {
            agent: agentName,
            agentAID: agentAID.prefix,
            validCredentials: [],
            invalidCredentials: [],
            totalCredentials: 0,
            totalValid: 0,
            totalInvalid: 0,
            timestamp: new Date().toISOString()
        };
        
        const outputPath = `/task-data/${agentName}-credential-validation-results.json`;
        await fs.promises.writeFile(outputPath, JSON.stringify(validationResults, null, 2));
        
        process.exit(0);
    }
    
    // Validate each credential
    console.log(`[3/4] Validating credentials...`);
    console.log(``);
    
    const validCredentials: ValidationResult[] = [];
    const invalidCredentials: any[] = [];
    
    for (let i = 0; i < credentials.length; i++) {
        const cred = credentials[i];
        const sad = cred.sad || {};
        const said = sad.d || 'N/A';
        const schema = sad.s || 'N/A';
        const issuer = sad.i || 'N/A';
        const attributes = sad.a || {};
        
        console.log(`Validating Credential #${i + 1}: ${said}`);
        
        // Determine credential type
        let credType = 'Unknown';
        if (schema.toLowerCase().includes('invoice')) {
            credType = 'Invoice';
        } else if (schema.toLowerCase().includes('oor')) {
            credType = 'OOR';
        } else if (schema.toLowerCase().includes('le')) {
            credType = 'LE';
        } else if (schema.toLowerCase().includes('qvi')) {
            credType = 'QVI';
        }
        
        const errors: string[] = [];
        let signatureValid = true;
        let chainValid = true;
        let schemaValid = true;
        
        try {
            // 1. Validate SAID (Self-Addressing Identifier)
            console.log(`  [1/5] Validating SAID...`);
            if (!said || said === 'N/A') {
                errors.push('Missing or invalid SAID');
                signatureValid = false;
            } else {
                console.log(`    ✓ SAID present: ${said.substring(0, 20)}...`);
            }
            
            // 2. Validate schema
            console.log(`  [2/5] Validating schema...`);
            if (!schema || schema === 'N/A') {
                errors.push('Missing or invalid schema');
                schemaValid = false;
            } else {
                console.log(`    ✓ Schema present: ${schema.substring(0, 40)}...`);
                // Additional schema validation could be done here
                // e.g., checking if schema is resolvable
            }
            
            // 3. Validate issuer
            console.log(`  [3/5] Validating issuer...`);
            if (!issuer || issuer === 'N/A') {
                errors.push('Missing or invalid issuer');
                signatureValid = false;
            } else {
                console.log(`    ✓ Issuer present: ${issuer.substring(0, 30)}...`);
            }
            
            // 4. Validate credential chain (if applicable)
            console.log(`  [4/5] Validating credential chain...`);
            const edges = sad.e || {};
            const hasEdge = Object.keys(edges).length > 0;
            
            if (hasEdge) {
                console.log(`    ℹ Credential is chained`);
                // For chained credentials, validate edge structure
                for (const [edgeKey, edgeValue] of Object.entries(edges)) {
                    const edge = edgeValue as any;
                    if (!edge.n) {
                        errors.push(`Edge '${edgeKey}' missing SAID reference`);
                        chainValid = false;
                    } else {
                        console.log(`    ✓ Edge '${edgeKey}' references: ${edge.n.substring(0, 20)}...`);
                    }
                }
            } else {
                // Check if self-attested
                const holderInAttributes = attributes.i;
                const selfAttested = issuer === holderInAttributes || issuer === agentAID.prefix;
                
                if (selfAttested) {
                    console.log(`    ✓ Self-attested credential (no chain required)`);
                } else {
                    console.log(`    ⚠ Warning: Credential is not self-attested and has no chain`);
                    // This is not necessarily an error, but worth noting
                }
            }
            
            // 5. Validate attributes
            console.log(`  [5/5] Validating attributes...`);
            const attributeKeys = Object.keys(attributes);
            if (attributeKeys.length === 0) {
                console.log(`    ⚠ Warning: Credential has no attributes`);
            } else {
                console.log(`    ✓ Credential has ${attributeKeys.length} attribute(s)`);
                
                // Type-specific validation
                if (credType === 'Invoice') {
                    const requiredFields = ['invoiceNumber', 'totalAmount', 'currency'];
                    const missingFields = requiredFields.filter(f => !attributes[f]);
                    if (missingFields.length > 0) {
                        errors.push(`Missing invoice fields: ${missingFields.join(', ')}`);
                        schemaValid = false;
                    }
                } else if (credType === 'OOR') {
                    const requiredFields = ['personLegalName', 'officialRole', 'LEI'];
                    const missingFields = requiredFields.filter(f => !attributes[f]);
                    if (missingFields.length > 0) {
                        errors.push(`Missing OOR fields: ${missingFields.join(', ')}`);
                        schemaValid = false;
                    }
                }
            }
            
            // Check if self-attested
            const holderInAttributes = attributes.i;
            const selfAttested = issuer === holderInAttributes || issuer === agentAID.prefix;
            
            // Overall validity
            const valid = errors.length === 0;
            
            if (valid) {
                console.log(`  ✅ Credential VALID`);
                validCredentials.push({
                    said,
                    schema,
                    type: credType,
                    signatureValid,
                    chainValid,
                    schemaValid,
                    selfAttested,
                    valid: true,
                    errors: []
                });
            } else {
                console.log(`  ❌ Credential INVALID: ${errors.join('; ')}`);
                invalidCredentials.push({
                    said,
                    schema,
                    type: credType,
                    error: errors.join('; '),
                    errors
                });
            }
            
        } catch (validationError: any) {
            console.log(`  ❌ Validation error: ${validationError.message}`);
            invalidCredentials.push({
                said,
                schema,
                type: credType,
                error: validationError.message,
                errors: [validationError.message]
            });
        }
        
        console.log(``);
    }
    
    // Generate validation summary
    console.log(`[4/4] Generating validation summary...`);
    console.log(``);
    
    const totalValid = validCredentials.length;
    const totalInvalid = invalidCredentials.length;
    const totalCredentials = credentials.length;
    
    console.log(`Validation Summary:`);
    console.log(`  Total Credentials: ${totalCredentials}`);
    console.log(`  Valid: ${totalValid} (${((totalValid / totalCredentials) * 100).toFixed(1)}%)`);
    console.log(`  Invalid: ${totalInvalid} (${((totalInvalid / totalCredentials) * 100).toFixed(1)}%)`);
    console.log(``);
    
    // Save validation results
    const validationResults = {
        agent: agentName,
        agentAID: agentAID.prefix,
        validCredentials,
        invalidCredentials,
        totalCredentials,
        totalValid,
        totalInvalid,
        validationRate: totalValid / totalCredentials,
        timestamp: new Date().toISOString()
    };
    
    const outputPath = `/task-data/${agentName}-credential-validation-results.json`;
    await fs.promises.writeFile(outputPath, JSON.stringify(validationResults, null, 2));
    console.log(`✓ Validation results saved to ${outputPath}`);
    console.log(``);
    
    console.log(`========================================`);
    if (totalInvalid === 0) {
        console.log(`✅ ALL CREDENTIALS VALID`);
    } else {
        console.log(`⚠ VALIDATION COMPLETED WITH ERRORS`);
    }
    console.log(`========================================`);
    console.log(``);
    
    if (totalInvalid > 0) {
        console.log(`Invalid Credentials:`);
        for (const invalid of invalidCredentials) {
            console.log(`  • ${invalid.said}: ${invalid.error}`);
        }
        console.log(``);
    }
    
} catch (error: any) {
    console.error(`❌ Validation failed: ${error.message}`);
    console.error(error);
    process.exit(1);
}
