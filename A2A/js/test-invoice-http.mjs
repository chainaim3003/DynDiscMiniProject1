// Simple HTTP request without using A2AClient
import http from 'http';
import { v4 as uuidv4 } from 'uuid';

function sendInvoiceViaHttp() {
  const postData = JSON.stringify({
    jsonrpc: "2.0",
    method: "message/stream",
    params: {
      message: {
        messageId: uuidv4(),
        kind: "message",
        role: "user",
        parts: [{ kind: "text", text: "send invoice" }]
      }
    },
    id: 1
  });

  const options = {
    hostname: '127.0.0.1',
    port: 8080,
    path: '/',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
      'Content-Length': Buffer.byteLength(postData)
    }
  };

  console.log('Sending invoice request to seller agent...\n');

  const req = http.request(options, (res) => {
    console.log(`Status: ${res.statusCode}`);
    console.log(`Headers: ${JSON.stringify(res.headers)}\n`);

    res.setEncoding('utf8');
    
    let buffer = '';
    
    res.on('data', (chunk) => {
      buffer += chunk;
      
      // Try to parse SSE events
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.substring(6));
            if (data.result?.kind === 'status-update' && data.result.status?.message) {
              const text = data.result.status.message.parts
                .filter(p => p.kind === 'text')
                .map(p => p.text)
                .join('\n');
              if (text) {
                console.log('\n=== RESPONSE ===');
                console.log(text);
                console.log('================\n');
              }
            }
          } catch (e) {
            // Ignore parse errors
          }
        }
      }
    });

    res.on('end', () => {
      console.log('✅ Stream ended');
    });
  });

  req.on('error', (e) => {
    console.error(`❌ Request error: ${e.message}`);
  });

  req.write(postData);
  req.end();
}

sendInvoiceViaHttp();
