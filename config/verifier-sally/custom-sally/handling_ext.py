"""
HTTP Handler Extension for Agent Delegation Verification

This module adds a custom endpoint to Sally for verifying agent delegations.
Endpoint: POST /verify/agent-delegation

Request body:
{
    "agent_aid": "EAgent...",
    "oor_holder_aid": "EOOR..."
}

Response:
{
    "valid": true/false,
    "agent_aid": "...",
    "oor_holder_aid": "...",
    "oor_credential_said": "...",  (if valid)
    "credential_chain": [...],      (if valid)
    "error": "..."                  (if invalid)
}
"""

import json
import falcon
from typing import Dict, Any
from keri.app import habbing

from custom_sally import agent_verifying


class AgentDelegationVerificationResource:
    """
    Falcon resource for agent delegation verification endpoint
    """
    
    def __init__(self, hby: habbing.Habery):
        """
        Initialize resource with KERI habery
        
        Args:
            hby: KERI habery instance
        """
        self.hby = hby
        self.verifier = agent_verifying.create_verifier(hby)
    
    def on_post(self, req: falcon.Request, resp: falcon.Response):
        """
        Handle POST request to verify agent delegation
        
        Args:
            req: Falcon request object
            resp: Falcon response object
        """
        try:
            # Parse request body
            body = req.bounded_stream.read()
            data = json.loads(body)
            
            # Validate required fields
            agent_aid = data.get("agent_aid")
            oor_holder_aid = data.get("oor_holder_aid")
            
            if not agent_aid or not oor_holder_aid:
                resp.status = falcon.HTTP_400
                resp.media = {
                    "error": "Missing required fields: agent_aid and oor_holder_aid"
                }
                return
            
            # Perform verification
            result = self.verifier.verify_agent_delegation(
                agent_aid=agent_aid,
                oor_holder_aid=oor_holder_aid
            )
            
            # Set response status based on result
            if result["valid"]:
                resp.status = falcon.HTTP_200
            else:
                resp.status = falcon.HTTP_200  # Still 200, but valid=false
            
            resp.media = result
            
        except json.JSONDecodeError:
            resp.status = falcon.HTTP_400
            resp.media = {
                "error": "Invalid JSON in request body"
            }
        except Exception as e:
            resp.status = falcon.HTTP_500
            resp.media = {
                "error": f"Internal server error: {str(e)}"
            }


def register_routes(app: falcon.App, hby: habbing.Habery):
    """
    Register custom routes with Sally's Falcon app
    
    This function should be called from Sally's startup code to add
    the agent delegation verification endpoint.
    
    Args:
        app: Falcon application instance
        hby: KERI habery instance
    """
    # Create resource instance
    agent_verification = AgentDelegationVerificationResource(hby)
    
    # Register route
    app.add_route('/verify/agent-delegation', agent_verification)
    
    print("✓ Custom route registered: POST /verify/agent-delegation")


# Alternative: Middleware approach for automatic registration
class CustomRouteMiddleware:
    """
    Middleware to automatically register custom routes when Sally starts
    """
    
    def __init__(self, hby: habbing.Habery):
        self.hby = hby
        self.registered = False
    
    def process_request(self, req: falcon.Request, resp: falcon.Response):
        """
        Process request - register routes on first call
        
        Args:
            req: Falcon request
            resp: Falcon response
        """
        if not self.registered:
            # Register custom routes
            register_routes(req.context.get('app'), self.hby)
            self.registered = True


def setup_custom_endpoints(app: falcon.App, hby: habbing.Habery):
    """
    Main setup function for custom endpoints
    
    Call this from Sally's initialization code:
    
    from custom_sally.handling_ext import setup_custom_endpoints
    setup_custom_endpoints(app, hby)
    
    Args:
        app: Falcon application instance
        hby: KERI habery instance
    """
    register_routes(app, hby)
    print("✓ Sally custom endpoints initialized")
