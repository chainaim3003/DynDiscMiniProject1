#!/usr/bin/env node

// Quick test for invoice flow
import { A2AClient } from '@a2a-js/sdk/client';

const SELLER_URL = 'http://localhost:8080';
const BUYER_URL = 'http://localhost:9090';

async function testInvoiceFlow() {
  console.log('🧪 Testing Invoice Flow...\n');
  
  try {
    // 1. Test seller agent is running
    console.log('1️⃣ Checking seller agent...');
    const sellerClient = new A2AClient(SELLER_URL);
    const sellerCard = await sellerClient.getAgentCard();
    console.log(`   ✅ Seller: ${sellerCard.name}\n`);
    
    // 2. Test buyer agent is running
    console.log('2️⃣ Checking buyer agent...');
    const buyerClient = new A2AClient(BUYER_URL);
    const buyerCard = await buyerClient.getAgentCard();
    console.log(`   ✅ Buyer: ${buyerCard.name}\n`);
    
    // 3. Trigger invoice from seller
    console.log('3️⃣ Sending "send invoice" to seller agent...');
    const response = await sellerClient.sendMessage({
      role: 'user',
      parts: [{ kind: 'text', text: 'send invoice' }]
    });
    console.log(`   ✅ Task created: ${response.taskId}\n`);
    
    console.log('✅ Test complete! Check agent console outputs for validation & payment.\n');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    process.exit(1);
  }
}

testInvoiceFlow();
