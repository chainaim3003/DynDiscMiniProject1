#!/bin/bash
# find-server-file.sh - Find Sally's server.py file

echo "=== Finding Sally's server.py file ==="
echo ""

docker run --rm --entrypoint /bin/sh gleif/sally:1.0.2 -c '
echo "1. Finding server.py in Sally source:"
find /sally -name "server.py" -type f 2>/dev/null
echo ""

echo "2. Listing Sally source structure:"
ls -la /sally/src/sally/ 2>/dev/null || echo "Directory not found"
echo ""

echo "3. Finding app directory structure:"
find /sally/src/sally -type d 2>/dev/null | head -20
echo ""

echo "4. Looking for CLI/commands structure:"
find /sally -name "commands" -type d 2>/dev/null
echo ""

echo "5. Finding all .py files in Sally:"
find /sally/src/sally -name "*.py" 2>/dev/null | head -30
'

echo ""
echo "=== Discovery complete ==="
