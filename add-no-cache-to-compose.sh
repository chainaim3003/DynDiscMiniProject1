#!/bin/bash
# add-no-cache-to-compose.sh - Adds no_cache: true to all build sections

set -e

echo "Adding no_cache: true to docker-compose.yml build sections..."

# Backup first
BACKUP="docker-compose.yml.backup.nocache.$(date +%s)"
cp docker-compose.yml "$BACKUP"
echo "✓ Backup created: $BACKUP"

# Use Python to safely modify YAML
python3 << 'PYTHON_EOF'
import yaml
import sys

try:
    # Read docker-compose.yml
    with open('docker-compose.yml', 'r') as f:
        compose = yaml.safe_load(f)
    
    # Track changes
    modified = False
    
    # Iterate through services
    for service_name, service_config in compose.get('services', {}).items():
        if 'build' in service_config:
            # Check if build is a string (context only) or dict
            if isinstance(service_config['build'], str):
                # Convert to dict format
                context = service_config['build']
                service_config['build'] = {
                    'context': context,
                    'no_cache': True
                }
                print(f"✓ Added no_cache to {service_name} (converted from string)")
                modified = True
            elif isinstance(service_config['build'], dict):
                # Add no_cache if not present
                if 'no_cache' not in service_config['build']:
                    service_config['build']['no_cache'] = True
                    print(f"✓ Added no_cache to {service_name}")
                    modified = True
                else:
                    print(f"  {service_name} already has no_cache")
    
    if modified:
        # Write back
        with open('docker-compose.yml', 'w') as f:
            yaml.dump(compose, f, default_flow_style=False, sort_keys=False)
        print("\n✓ docker-compose.yml updated successfully")
    else:
        print("\n✓ No changes needed - all build sections already have no_cache")
    
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║              no_cache Added Successfully! ✅                ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Services updated:"
    echo "  • tsx-shell (will always build without cache)"
    echo "  • verification (when added, will always build without cache)"
    echo ""
    echo "Now you can just run:"
    echo "  docker compose build      # No --no-cache flag needed"
    echo ""
    echo "To revert:"
    echo "  cp $BACKUP docker-compose.yml"
    echo ""
else
    echo "Failed to update docker-compose.yml"
    echo "Backup available at: $BACKUP"
    exit 1
fi
