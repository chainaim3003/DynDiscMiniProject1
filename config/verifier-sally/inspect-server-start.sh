#!/bin/bash
# inspect-server-start.sh - Inspect Sally's server start.py

echo "=== Inspecting Sally's server/start.py ==="
echo ""

docker run --rm --entrypoint /bin/sh gleif/sally:1.0.2 -c '
echo "Content of /sally/src/sally/app/cli/commands/server/start.py:"
echo "================================================================"
cat /sally/src/sally/app/cli/commands/server/start.py
echo ""
echo "================================================================"
echo ""
echo "Also checking core/serving.py (main server logic):"
echo "================================================================"
head -50 /sally/src/sally/core/serving.py
'

echo ""
echo "=== Inspection complete ==="
