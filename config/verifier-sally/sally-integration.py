#!/usr/bin/env python3
"""
Sally Integration Script for Custom Extensions

This script is called by Sally during startup to register custom routes.
It can be invoked manually for testing or automatically by Sally.
"""

import sys
import os

# Add custom_sally to Python path
sys.path.insert(0, '/usr/local/lib/python3.12/site-packages')

try:
    from custom_sally import handling_ext
    print("✓ Custom extensions module loaded successfully")
    
    # This function will be called by Sally when it starts
    def register_custom_routes(app, hby):
        """
        Hook function called by Sally to register custom routes
        
        Args:
            app: Falcon WSGI application instance
            hby: KERI habbing instance
        """
        handling_ext.register_routes(app, hby)
        return True
    
    if __name__ == "__main__":
        print("Custom Sally Extensions - Integration Test")
        print("=" * 50)
        print("This script should be called by Sally during startup")
        print("To test manually, ensure Sally is running and has hby instance")
        
except ImportError as e:
    print(f"✗ Failed to import custom extensions: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error loading custom extensions: {e}")
    sys.exit(1)
