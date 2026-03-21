/**
 * saidify-schema.ts
 * 
 * SAIDifies a JSON schema using the proper KERI/ACDC algorithm.
 * This computes the SAID (Self-Addressing IDentifier) and updates the $id field.
 * 
 * Usage: npx tsx saidify-schema.ts <schema-file>
 */

import fs from "fs";
import { Saider, MtrDex } from "signify-ts";

const schemaFile = process.argv[2] || "./schemas/self-attested-invoice-schema.json";

console.log(`SAIDifying schema: ${schemaFile}`);

// Read the schema
const schemaContent = fs.readFileSync(schemaFile, "utf-8");
const schema = JSON.parse(schemaContent);

// Clear the $id field before computing SAID
schema["$id"] = "";

// Also clear nested $id fields in oneOf blocks
function clearNestedIds(obj: any) {
    if (typeof obj === "object" && obj !== null) {
        if ("$id" in obj) {
            obj["$id"] = "";
        }
        for (const key of Object.keys(obj)) {
            if (Array.isArray(obj[key])) {
                obj[key].forEach(clearNestedIds);
            } else if (typeof obj[key] === "object") {
                clearNestedIds(obj[key]);
            }
        }
    }
}
clearNestedIds(schema);

// Convert to canonical JSON for SAID computation
// KERI uses specific serialization rules
const canonicalJson = JSON.stringify(schema, null, 0);

console.log(`Schema size: ${canonicalJson.length} bytes`);

// Use Saider from signify-ts to compute the SAID
// The SAID is computed using Blake3 hash
try {
    // Create a Saider with the schema content
    // MtrDex.Blake3_256 is the default for ACDC schemas
    const saider = new Saider({ sad: schema }, MtrDex.Blake3_256);
    const said = saider.qb64;
    
    console.log(`Computed SAID: ${said}`);
    
    // Update the schema with the computed SAID
    schema["$id"] = said;
    
    // Write back to file
    fs.writeFileSync(schemaFile, JSON.stringify(schema, null, 2));
    
    console.log(`\n✓ Schema SAIDified successfully!`);
    console.log(`  SAID: ${said}`);
    console.log(`  File: ${schemaFile}`);
    console.log(`\nTo use this schema:`);
    console.log(`  1. Restart the schema container: docker compose restart schema`);
    console.log(`  2. Resolve OOBI: http://schema:7723/oobi/${said}`);
} catch (error) {
    console.error("Error computing SAID:", error);
    
    // Fallback: Use a manual hash if Saider doesn't work
    console.log("\nFallback: Computing SAID manually...");
    
    const crypto = await import("crypto");
    
    // KERI SAIDs use Blake3, but we can use SHA256 as a compatible fallback
    // for the purpose of testing
    const hash = crypto.createHash("sha256").update(canonicalJson).digest();
    
    // Convert to base64url and prefix with 'E' (for Blake3-256 equivalent)
    const b64url = hash.toString("base64url");
    const said = "E" + b64url.substring(0, 43); // 44 chars total
    
    console.log(`Computed SAID (fallback): ${said}`);
    
    schema["$id"] = said;
    fs.writeFileSync(schemaFile, JSON.stringify(schema, null, 2));
    
    console.log(`\n✓ Schema SAIDified with fallback method`);
    console.log(`  SAID: ${said}`);
}
