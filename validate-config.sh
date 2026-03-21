#!/bin/bash
# Simple JSON validator using python

CONFIG_FILE="./appconfig/configBuyerSellerAIAgent1.json"

echo "Validating JSON configuration..."

if command -v python3 &> /dev/null; then
    python3 -c "import json; json.load(open('$CONFIG_FILE'))" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ JSON is valid"
        exit 0
    else
        echo "✗ JSON is invalid"
        python3 -c "import json; json.load(open('$CONFIG_FILE'))"
        exit 1
    fi
elif command -v python &> /dev/null; then
    python -c "import json; json.load(open('$CONFIG_FILE'))" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ JSON is valid"
        exit 0
    else
        echo "✗ JSON is invalid"
        python -c "import json; json.load(open('$CONFIG_FILE'))"
        exit 1
    fi
else
    echo "Neither jq nor python found. Cannot validate JSON."
    echo "Please install jq or python to continue."
    exit 1
fi
