#!/bin/bash
# Find Sally's actual location in the Docker container

echo "Finding Sally installation location..."
echo ""

# Start a temporary verifier container
echo "Starting temporary Sally container..."
docker run --rm --entrypoint /bin/sh gleif/sally:1.0.2 -c '
echo "=== Python version ==="
python3 --version
echo ""

echo "=== Site packages location ==="
python3 -c "import site; print(site.getsitepackages())"
echo ""

echo "=== Sally package location ==="
python3 -c "import sally; print(sally.__file__)" 2>/dev/null || echo "Sally not importable as module"
echo ""

echo "=== Find all server.py files ==="
find / -name "server.py" 2>/dev/null | grep -i sally || echo "No server.py found in sally paths"
echo ""

echo "=== Find Sally installation directory ==="
find / -type d -name "sally" 2>/dev/null | head -10
echo ""

echo "=== Check common Python paths ==="
ls -la /usr/local/lib/python*/site-packages/ 2>/dev/null | grep sally || echo "Not in /usr/local/lib"
ls -la /usr/lib/python*/site-packages/ 2>/dev/null | grep sally || echo "Not in /usr/lib"
echo ""

echo "=== Sally pip show ==="
pip show sally 2>/dev/null || pip3 show sally 2>/dev/null || echo "pip show not available"
'
