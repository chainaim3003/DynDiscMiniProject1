// verify-mnemonic.mjs
// Check if the mnemonic in .env generates the correct buyer address

import algosdk from 'algosdk';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Load .env from buyer-agent directory
const envPath = path.join(__dirname, 'src', 'agents', 'buyer-agent', '.env');
dotenv.config({ path: envPath });

const mnemonic = process.env.BUYER_MNEMONIC;
const expectedAddress = process.env.BUYER_ADDRESS;

if (!mnemonic) {
  console.log('❌ BUYER_MNEMONIC not found in .env');
  process.exit(1);
}

console.log('🔍 Verifying mnemonic...\n');

try {
  const account = algosdk.mnemonicToSecretKey(mnemonic);
  
  console.log(`Expected Address: ${expectedAddress}`);
  console.log(`Generated Address: ${account.addr}\n`);
  
  // Trim whitespace for comparison
  const expectedTrimmed = expectedAddress?.trim();
  const generatedTrimmed = account.addr.trim();
  
  if (generatedTrimmed === expectedTrimmed) {
    console.log('✅ SUCCESS! Mnemonic matches the buyer address!');
    console.log('\nYou can now run the invoice test:');
    console.log('  node test-invoice-http.mjs');
  } else {
    console.log('❌ MISMATCH! The mnemonic does NOT match the buyer address.');
    console.log('\n📝 To fix this:');
    console.log('1. Open Lute Wallet');
    console.log('2. Find wallet: ' + expectedAddress);
    console.log('3. Export the mnemonic (seed phrase)');
    console.log('4. Update BUYER_MNEMONIC in: src/agents/buyer-agent/.env');
  }
} catch (error) {
  console.log('❌ Invalid mnemonic format!');
  console.log('Error:', error.message);
  console.log('\nMake sure you have exactly 25 words separated by spaces.');
}
