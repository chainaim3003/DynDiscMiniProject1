"""
Sally Startup Script with Agent Verification Extension

This script starts Sally and adds the agent delegation verification endpoint.
"""

import sys
import logging
from sally.app.cli.commands import server

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 60)
print("Starting Sally with Agent Verification Extension")
print("=" * 60)

# Import our agent verification module
try:
    from sally_agent_verification import setup_agent_verification_endpoint
    logger.info("✓ Agent verification module loaded")
except ImportError as e:
    logger.error(f"Failed to load agent verification module: {e}")
    sys.exit(1)

# Patch Sally's server setup to add our endpoint
original_setup = server.setup

def setup_with_agent_verification(hby, *args, **kwargs):
    """
    Enhanced setup that adds agent verification endpoint.
    """
    logger.info("Setting up Sally with agent verification extension...")
    
    # Call original setup to create the Falcon app
    app = original_setup(hby, *args, **kwargs)
    
    # Add our agent verification endpoint
    setup_agent_verification_endpoint(app, hby)
    
    logger.info("✓ Sally setup complete with agent verification")
    return app

# Replace the setup function
server.setup = setup_with_agent_verification

logger.info("✓ Sally patched with agent verification endpoint")
print("=" * 60)
print("Sally is ready with /verify/agent-delegation endpoint")
print("=" * 60)
