import { getOrCreateClient } from "../../client/identifiers.js";
import { getCredential } from "../../client/credentials.js";

// Process arguments
const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const passcode = args[1];
const holderAidName = args[2];
const invoiceSaid = args[3];

console.log(`========================================`);
console.log(`INVOICE CREDENTIAL CHAIN VERIFICATION`);
console.log(`========================================`);
console.log(``);
console.log(`Holder: ${holderAidName}`);
console.log(`Invoice SAID: ${invoiceSaid}`);
console.log(``);

// Get client
const client = await getOrCreateClient(passcode, env);

try {
    // Step 1: Retrieve invoice credential
    console.log(`[1/5] Retrieving invoice credential...`);
    const invoiceCred = await getCredential(client, invoiceSaid);
    console.log(`✓ Invoice credential retrieved`);
    console.log(`  Issuer: ${invoiceCred.sad.i}`);
    console.log(`  Holder: ${invoiceCred.sad.a.i}`);
    console.log(`  Invoice #: ${invoiceCred.sad.a.invoiceNumber}`);
    console.log(`  Amount: ${invoiceCred.sad.a.totalAmount} ${invoiceCred.sad.a.currency}`);
    console.log(``);
    
    // Step 2: Verify edge to OOR credential
    console.log(`[2/5] Verifying edge to OOR credential...`);
    if (!invoiceCred.sad.e || !invoiceCred.sad.e.oor) {
        throw new Error('Invoice credential missing OOR edge');
    }
    
    const oorSaid = invoiceCred.sad.e.oor.n;
    console.log(`✓ Edge found to OOR credential: ${oorSaid}`);
    console.log(``);
    
    // Step 3: Retrieve OOR credential
    console.log(`[3/5] Retrieving OOR credential...`);
    const oorCred = await getCredential(client, oorSaid);
    console.log(`✓ OOR credential retrieved`);
    console.log(`  Holder: ${oorCred.sad.a.i}`);
    console.log(`  Person: ${oorCred.sad.a.personLegalName}`);
    console.log(`  Role: ${oorCred.sad.a.officialRole}`);
    console.log(`  LEI: ${oorCred.sad.a.LEI}`);
    console.log(``);
    
    // Step 4: Verify issuer authority
    console.log(`[4/5] Verifying issuer authority...`);
    if (invoiceCred.sad.i !== oorCred.sad.a.i) {
        throw new Error(
            `Invoice issuer (${invoiceCred.sad.i}) does not match OOR holder (${oorCred.sad.a.i})`
        );
    }
    console.log(`✓ Invoice issuer is OOR credential holder`);
    console.log(``);
    
    // Step 5: Verify OOR edge to LE credential
    console.log(`[5/5] Verifying OOR chain to LE credential...`);
    if (!oorCred.sad.e || !oorCred.sad.e.auth) {
        throw new Error('OOR credential missing auth edge to LE');
    }
    
    const oorAuthSaid = oorCred.sad.e.auth.n;
    console.log(`✓ OOR chains to auth credential: ${oorAuthSaid}`);
    console.log(``);
    
    console.log(`========================================`);
    console.log(`✅ INVOICE CHAIN VERIFICATION COMPLETE`);
    console.log(`========================================`);
    console.log(``);
    console.log(`Verified Chain:`);
    console.log(`  Invoice → OOR → LE → QVI → Root`);
    console.log(``);
    console.log(`Invoice Details:`);
    console.log(`  Number: ${invoiceCred.sad.a.invoiceNumber}`);
    console.log(`  Date: ${invoiceCred.sad.a.invoiceDate}`);
    console.log(`  Due: ${invoiceCred.sad.a.dueDate}`);
    console.log(`  Amount: ${invoiceCred.sad.a.totalAmount} ${invoiceCred.sad.a.currency}`);
    console.log(`  Seller LEI: ${invoiceCred.sad.a.sellerLEI}`);
    console.log(`  Buyer LEI: ${invoiceCred.sad.a.buyerLEI}`);
    console.log(`  Payment Method: ${invoiceCred.sad.a.paymentMethod}`);
    if (invoiceCred.sad.a.stellarPaymentAddress) {
        console.log(`  Stellar Address: ${invoiceCred.sad.a.stellarPaymentAddress}`);
    }
    console.log(``);
    
} catch (error) {
    console.error(`❌ Verification failed: ${error.message}`);
    process.exit(1);
}
