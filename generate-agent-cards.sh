#!/bin/bash

# Agent Card Generator Script
# Generates complete agent cards from vLEI workflow output

echo "════════════════════════════════════════════════════════"
echo "  vLEI Agent Card Generator"
echo "════════════════════════════════════════════════════════"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "✗ Error: Node.js is not installed"
    echo "  Please install Node.js to run this script"
    exit 1
fi

# Check if task-data directory exists
if [ ! -d "./task-data" ]; then
    echo "✗ Error: task-data directory not found"
    echo "  Please run the vLEI workflow first to generate data"
    exit 1
fi

# Check if configuration file exists
if [ ! -f "./appconfig/configBuyerSellerAIAgent1.json" ]; then
    echo "✗ Error: Configuration file not found"
    echo "  Expected: ./appconfig/configBuyerSellerAIAgent1.json"
    exit 1
fi

echo "→ Starting agent card generation..."
echo ""

# Run the Node.js script
node generate-agent-cards.js

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Agent cards generated successfully!"
    echo ""
    echo "📁 Output Location: ./agent-cards/"
    echo ""
    echo "Generated files:"
    ls -lh agent-cards/*.json 2>/dev/null
    echo ""
else
    echo ""
    echo "✗ Agent card generation failed"
    exit 1
fi
