#!/usr/bin/env python3
"""
Sally Routes Patch - Registers Custom Agent Delegation Verification Endpoint

This script patches Sally's application to add the custom /verify/agent-delegation endpoint.
It's executed during Docker image build to modify Sally's route registration.
"""

import sys
import os

# Patch script to add custom route registration to Sally
patch_code = '''
# Custom Agent Delegation Verification Route
# Added by routes_patch.py during Docker build

from sally.core import agent_verifying, handling_ext

# Register custom routes
def register_custom_routes(app, hby):
    """Register custom verification routes"""
    handling_ext.register_routes(app, hby)
    print("✓ Custom agent delegation verification route registered")

# This will be imported and called by Sally's main application
'''

# Write patch to Sally's core module
patch_file = "/usr/local/lib/python3.12/site-packages/sally/core/custom_routes.py"

try:
    with open(patch_file, 'w') as f:
        f.write(patch_code)
    print(f"✓ Routes patch written to {patch_file}")
    print("✓ Custom routes will be registered when Sally starts")
    sys.exit(0)
except Exception as e:
    print(f"✗ Error writing routes patch: {e}")
    sys.exit(1)
