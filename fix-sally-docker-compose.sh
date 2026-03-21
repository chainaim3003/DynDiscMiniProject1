# Quick script to add Sally custom extension to docker-compose.yml
# This creates a backup and updates the verifier service configuration

cat > ~/projects/vLEIWorkLinux1/fix-sally-config.sh << 'EOF'
#!/bin/bash

set -e

echo "=========================================="
echo "ADDING SALLY CUSTOM EXTENSION"
echo "=========================================="
echo ""

# Backup docker-compose.yml
cp docker-compose.yml docker-compose.yml.backup
echo "✓ Backup created: docker-compose.yml.backup"

# Update verifier service to mount custom-sally directory
# We'll add the mount after the existing volume mounts

# Find the line number of the verifier volumes section
LINE_NUM=$(grep -n "verifier-vol:/usr/local/var/keri" docker-compose.yml | cut -d: -f1)

if [ -z "$LINE_NUM" ]; then
    echo "❌ Could not find verifier volumes section"
    exit 1
fi

# Insert custom-sally mount after the verifier-vol line
sed -i "${LINE_NUM}a\      - ./config/verifier-sally/custom-sally:/usr/local/lib/python3.12/site-packages/custom_sally" docker-compose.yml

echo "✓ Added custom-sally volume mount"
echo ""
echo "Modified verifier service to include:"
echo "  - Custom Sally Python modules mounted at runtime"
echo ""
echo "Next steps:"
echo "  1. Restart verifier: docker compose restart verifier"
echo "  2. Check logs: docker compose logs -f verifier"
echo "  3. Test: ./test-agent-verification.sh"
echo ""
EOF

chmod +x ~/projects/vLEIWorkLinux1/fix-sally-config.sh
