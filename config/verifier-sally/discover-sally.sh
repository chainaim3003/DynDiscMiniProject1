#!/bin/bash
# discover-sally.sh - Find Sally's actual installation location

echo "=== Discovering Sally installation in container ==="
echo ""

docker run --rm --entrypoint /bin/sh gleif/sally:1.0.2 -c '
echo "1. Finding Sally directories:"
find /usr -name "sally" -type d 2>/dev/null | head -10
echo ""

echo "2. Finding Sally Python package:"
python3 -c "import sally; import os; print(os.path.dirname(sally.__file__))" 2>/dev/null || echo "Sally not importable as package"
echo ""

echo "3. Finding Sally executable:"
which sally
echo ""

echo "4. Checking Sally structure:"
ls -la $(which sally) 2>/dev/null || echo "Sally executable not found"
echo ""

echo "5. Finding Python site-packages:"
python3 -c "import site; print(site.getsitepackages())"
echo ""

echo "6. Looking for server.py files:"
find /usr -name "server.py" 2>/dev/null | grep -i sally | head -10
echo ""

echo "7. Checking if Sally is installed via pip:"
pip3 list | grep -i sally
echo ""

echo "8. Finding all Python files in likely locations:"
find /usr/local/lib/python* -name "*.py" 2>/dev/null | grep sally | head -20
echo ""

echo "9. Checking sally as a CLI tool (shows install location):"
python3 -c "import sys; print(sys.path)"
'

echo ""
echo "=== Discovery complete ==="
