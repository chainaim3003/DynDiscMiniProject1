# vLEI Verification API Server

This API server provides REST endpoints for real vLEI agent verification.

## Setup

### On Linux Machine:

1. **Navigate to api-server directory:**
   ```bash
   cd ~/projects/vLEIWorkLinux1/api-server
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the server:**
   ```bash
   npm start
   ```

   The server will start on port 4000 by default.

## API Endpoints

### Health Check
```bash
GET http://localhost:4000/health
```

### Verify Seller Agent
```bash
POST http://localhost:4000/api/verify/seller
```

Response:
```json
{
  "success": true,
  "output": "verification output...",
  "agent": "jupiterSellerAgent",
  "oorHolder": "Jupiter_Chief_Sales_Officer",
  "timestamp": "2025-11-13T..."
}
```

### Verify Buyer Agent
```bash
POST http://localhost:4000/api/verify/buyer
```

Response:
```json
{
  "success": true,
  "output": "verification output...",
  "agent": "tommyBuyerAgent",
  "oorHolder": "Tommy_Chief_Procurement_Officer",
  "timestamp": "2025-11-13T..."
}
```

## Testing with curl

```bash
# Test health
curl http://localhost:4000/health

# Test seller verification
curl -X POST http://localhost:4000/api/verify/seller

# Test buyer verification
curl -X POST http://localhost:4000/api/verify/buyer
```

## Important Notes

1. **Docker must be running** - The verification scripts use Docker containers
2. **Deploy first** - Run `./deploy.sh` to ensure all services are up
3. **Network access** - The UI (Windows) must be able to reach this server
4. **Port 4000** - Make sure this port is not blocked by firewall

## Connecting from Windows UI

From Windows, you'll need to use the Linux machine's IP address:

```javascript
const API_URL = 'http://<LINUX_IP>:4000';
```

To find your Linux IP:
```bash
hostname -I | awk '{print $1}'
```

## Logs

The server logs all verification attempts to console. Check for:
- `=== SELLER AGENT VERIFICATION REQUEST ===`
- `=== BUYER AGENT VERIFICATION REQUEST ===`
- `Verification result: SUCCESS` or `FAILED`
