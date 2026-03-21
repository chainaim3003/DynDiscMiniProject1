#!/bin/bash
# check-serving-structure.sh - Check setupDoers function structure

echo "=== Checking setupDoers function structure ==="

docker run --rm --entrypoint /bin/sh gleif/sally-custom:latest -c '
echo "Finding setupDoers function and showing first 100 lines:"
echo "=========================================="
grep -n "def setupDoers" /sally/src/sally/core/serving.py
echo ""
echo "Showing lines 200-350 (where app is created and routes added):"
sed -n "200,350p" /sally/src/sally/core/serving.py | cat -n
'
