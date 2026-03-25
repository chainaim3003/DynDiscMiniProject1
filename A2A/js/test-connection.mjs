// test-connection.mjs
// Diagnostic script to test seller-buyer agent connection

import http from 'http';
import { v4 as uuidv4 } from 'uuid';

console.log('🔍 Testing Buyer Agent Connection\n');

// Test 1: Can we reach the buyer's agent card?
console.log('TEST 1: Fetching buyer agent card...');
const options1 = {
  hostname: '127.0.0.1',
  port: 9090,
  path: '/.well-known/agent-card.json',
  method: 'GET',
};

const req1 = http.request(options1, (res) => {
  console.log(`✅ Agent card responded: ${res.statusCode}`);
  
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => {
    try {
      const card = JSON.parse(data);
      console.log(`   Name: ${card.name}`);
      console.log(`   URL: ${card.url}\n`);
      
      // Test 2: Can we POST a message?
      console.log('TEST 2: Sending POST message to buyer...');
      
      const postData = JSON.stringify({
        jsonrpc: "2.0",
        method: "message/stream",
        params: {
          message: {
            messageId: uuidv4(),
            kind: "message",
            role: "user",
            parts: [{ kind: "text", text: "hello from test" }]
          }
        },
        id: 1
      });
      
      const options2 = {
        hostname: '127.0.0.1',
        port: 9090,
        path: '/',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
          'Content-Length': Buffer.byteLength(postData)
        }
      };
      
      const req2 = http.request(options2, (res) => {
        console.log(`✅ POST responded: ${res.statusCode}`);
        
        let responseData = '';
        res.on('data', chunk => responseData += chunk);
        res.on('end', () => {
          console.log('   Response received\n');
          
          // Test 3: Try with localhost instead of 127.0.0.1
          console.log('TEST 3: Same test with "localhost"...');
          
          const options3 = {
            hostname: 'localhost',
            port: 9090,
            path: '/',
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'text/event-stream',
              'Content-Length': Buffer.byteLength(postData)
            }
          };
          
          const req3 = http.request(options3, (res) => {
            console.log(`✅ localhost POST responded: ${res.statusCode}\n`);
            
            console.log('═══════════════════════════════════════');
            console.log('DIAGNOSIS: All tests passed!');
            console.log('Both 127.0.0.1 and localhost work.');
            console.log('The issue might be with fetch() vs http module.');
            console.log('═══════════════════════════════════════');
          });
          
          req3.on('error', (e) => {
            console.error(`❌ localhost test failed: ${e.message}`);
          });
          
          req3.write(postData);
          req3.end();
        });
      });
      
      req2.on('error', (e) => {
        console.error(`❌ POST test failed: ${e.message}`);
        console.log('\n⚠️  This suggests the buyer agent is not accepting POST requests');
        console.log('    Check if buyer agent is running: npm run agents:buyer');
      });
      
      req2.write(postData);
      req2.end();
      
    } catch (e) {
      console.error(`❌ Failed to parse agent card: ${e.message}`);
    }
  });
});

req1.on('error', (e) => {
  console.error(`❌ Cannot reach buyer agent: ${e.message}`);
  console.log('\n⚠️  Make sure buyer agent is running on port 9090');
  console.log('    Run: npm run agents:buyer');
});

req1.end();
