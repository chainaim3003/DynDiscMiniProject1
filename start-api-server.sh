#!/bin/bash
#
# start-api-server.sh - Start the vLEI Verification API Server
#

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================"
echo "Starting vLEI Verification API Server"
echo -e "========================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "api-server/package.json" ]; then
    echo -e "${RED}❌ Error: api-server directory not found${NC}"
    echo "Please run this script from the vLEIWorkLinux1 directory"
    exit 1
fi

# Navigate to api-server directory
cd api-server

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}📦 Installing dependencies...${NC}"
    npm install
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to install dependencies${NC}"
        exit 1
    fi
    echo ""
fi

# Get the machine's IP address
IP_ADDRESS=$(hostname -I | awk '{print $1}')

echo -e "${GREEN}✅ Starting API server...${NC}"
echo ""
echo -e "${BLUE}Server Information:${NC}"
echo "  Local:    http://localhost:4000"
echo "  Network:  http://${IP_ADDRESS}:4000"
echo ""
echo -e "${BLUE}Update your UI .env.local with:${NC}"
echo "  NEXT_PUBLIC_API_URL=http://${IP_ADDRESS}:4000"
echo ""
echo "========================================"
echo ""

# Start the server
npm start
