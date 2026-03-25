import { A2AClient } from "@a2a-js/sdk/client";
import { v4 as uuidv4 } from 'uuid';

async function testInvoice() {
  // Try 127.0.0.1 instead of localhost
  const url = 'http://127.0.0.1:8080';
  console.log(`Connecting to seller agent at ${url}...`);
  
  const client = new A2AClient(url);
  
  console.log('Sending "send invoice" message...');
  
  try {
    const stream = client.sendMessageStream({
      message: {
        kind: "message",
        role: "user",
        messageId: uuidv4(),
        parts: [{ kind: "text", text: "send invoice" }]
      }
    });

    console.log('Waiting for response...\n');
    
    for await (const event of stream) {
      console.log('Event:', event.kind);
      if (event.kind === 'status-update' && event.status.message) {
        const text = event.status.message.parts
          .filter(p => p.kind === 'text')
          .map(p => p.text)
          .join('\n');
        if (text) {
          console.log('\n=== RESPONSE ===');
          console.log(text);
          console.log('================\n');
        }
      }
    }
    
    console.log('✅ Invoice flow completed!');
  } catch (error) {
    console.error('❌ ERROR:', error.message);
    console.error('\n🔍 Debugging info:');
    console.error('- Is seller agent running on port 8080?');
    console.error('- Try: curl http://127.0.0.1:8080/.well-known/agent-card.json');
    console.error('- Check: netstat -ano | findstr :8080');
    if (error.cause) {
      console.error('\nCause:', error.cause);
    }
  }
}

testInvoice();
