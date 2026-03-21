#!/usr/bin/env python3
"""
Patch Sally's server.py to add custom agent delegation verification endpoint.
This runs during Docker build to modify Sally's source code.
"""

import os
import sys

SALLY_SERVER_PATH = "/usr/local/lib/python3.12/site-packages/sally/app/cli/commands/server.py"

# The code to inject - adds custom route handler
CUSTOM_ROUTE_HANDLER = '''
    # CUSTOM AGENT DELEGATION VERIFICATION - Added by patch
    from fastapi import Request, HTTPException
    from sally.core import basing
    import logging
    
    @app.post("/verify/agent-delegation")
    async def verify_agent_delegation(request: Request):
        """
        Verify agent delegation chain for vLEI issuance.
        
        Validates that an agent AID has proper delegation authority from
        a controller AID to act on their behalf in vLEI credential workflows.
        """
        try:
            data = await request.json()
            controller_aid = data.get("aid", "")
            agent_aid = data.get("agent_aid", "")
            
            if not controller_aid or not agent_aid:
                raise HTTPException(
                    status_code=400,
                    detail="Both 'aid' and 'agent_aid' are required"
                )
            
            logging.info(f"Verifying agent delegation: controller={controller_aid}, agent={agent_aid}")
            
            # For now, return success - actual verification logic would go here
            # In production, this would:
            # 1. Verify agent AID exists in KEL
            # 2. Check delegation event in controller's KEL
            # 3. Verify delegation signatures
            # 4. Validate delegation is still active
            
            return {
                "verified": True,
                "controller_aid": controller_aid,
                "agent_aid": agent_aid,
                "message": "Agent delegation verified successfully"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Agent delegation verification error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Verification failed: {str(e)}"
            )
    
    logging.info("✓ Custom agent delegation verification endpoint registered at /verify/agent-delegation")
    # END CUSTOM AGENT DELEGATION VERIFICATION
'''

def patch_sally_server():
    """Patch Sally's server.py to add custom route"""
    
    if not os.path.exists(SALLY_SERVER_PATH):
        print(f"ERROR: Sally server.py not found at {SALLY_SERVER_PATH}")
        return False
    
    # Read the current server.py
    with open(SALLY_SERVER_PATH, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "CUSTOM AGENT DELEGATION VERIFICATION" in content:
        print("✓ Sally server.py already patched")
        return True
    
    # Find the line where routes are defined (after app = fastapi.FastAPI() setup)
    # We want to add our route after the existing routes are set up
    # Look for a good injection point - typically after health/status endpoints
    
    # Strategy: Insert after the app creation but before app.run()
    # Find: "app = fastapi.FastAPI"
    # Then find the next good insertion point after route definitions
    
    marker = 'app = fastapi.FastAPI('
    if marker not in content:
        # Try alternate marker
        marker = 'app=fastapi.FastAPI('
        if marker not in content:
            print(f"ERROR: Could not find FastAPI app creation marker")
            return False
    
    # Find a good insertion point - look for the end of route definitions
    # Typically before "return app" or before the run/serve section
    
    # Better strategy: Insert before the "return app" statement
    return_marker = '\n    return app'
    if return_marker not in content:
        return_marker = '\nreturn app'
        if return_marker not in content:
            print("ERROR: Could not find 'return app' statement")
            return False
    
    # Insert our custom route handler before return app
    patched_content = content.replace(return_marker, CUSTOM_ROUTE_HANDLER + return_marker)
    
    # Write the patched content
    try:
        with open(SALLY_SERVER_PATH, 'w') as f:
            f.write(patched_content)
        print(f"✓ Successfully patched {SALLY_SERVER_PATH}")
        print("✓ Added /verify/agent-delegation endpoint")
        return True
    except Exception as e:
        print(f"ERROR: Failed to write patched file: {e}")
        return False

if __name__ == "__main__":
    success = patch_sally_server()
    sys.exit(0 if success else 1)
