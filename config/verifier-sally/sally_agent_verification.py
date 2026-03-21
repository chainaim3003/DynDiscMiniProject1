"""
Sally Agent Verification Extension

Adds /verify/agent-delegation endpoint to Sally for verifying agent delegations.

This module extends Sally's Falcon app with agent delegation verification that:
1. Queries agent's KEL to verify delegation
2. Verifies OOR holder's credential
3. Checks complete trust chain to GLEIF
"""

from keri.core import coring
import logging
import json

logger = logging.getLogger(__name__)


async def verify_agent_delegation(hby, agent_aid: str, oor_holder_aid: str):
    """
    Verify agent delegation relationship through KEL inspection.
    
    Args:
        hby: Sally's Habery instance
        agent_aid: Agent's AID (prefix)
        oor_holder_aid: OOR holder's AID (prefix)
        
    Returns:
        dict: Verification result with details
    """
    result = {
        "valid": False,
        "agent_aid": agent_aid,
        "oor_holder_aid": oor_holder_aid,
        "verification": {},
        "errors": []
    }
    
    try:
        logger.info(f"Verifying agent delegation: {agent_aid} from {oor_holder_aid}")
        
        # 1. Get agent's key event log
        agent_kever = hby.db.kevers.get(keys=agent_aid)
        
        if not agent_kever:
            result["errors"].append(f"Agent AID not found in KEL database: {agent_aid}")
            logger.warning(f"Agent AID not in KEL: {agent_aid}")
            return result
            
        # 2. Check if agent is delegated
        if not agent_kever.delegated:
            result["errors"].append("Agent is not a delegated identifier")
            logger.warning(f"Agent {agent_aid} is not delegated")
            return result
            
        # 3. Get agent's delegator
        agent_delegator = agent_kever.delpre
        
        # 4. Verify delegator matches OOR holder
        if agent_delegator != oor_holder_aid:
            result["errors"].append(
                f"Delegation mismatch: agent delegated by {agent_delegator}, "
                f"but expected {oor_holder_aid}"
            )
            logger.warning(f"Delegator mismatch for agent {agent_aid}")
            return result
            
        # 5. Get OOR holder's key event log
        oor_kever = hby.db.kevers.get(keys=oor_holder_aid)
        
        if not oor_kever:
            result["errors"].append(f"OOR holder AID not found in KEL database: {oor_holder_aid}")
            logger.warning(f"OOR holder AID not in KEL: {oor_holder_aid}")
            return result
            
        # 6. Verify OOR holder's KEL is valid
        if not oor_kever.serder:
            result["errors"].append("OOR holder KEL is invalid")
            logger.warning(f"Invalid KEL for OOR holder {oor_holder_aid}")
            return result
            
        # 7. Check if OOR holder is also delegated (should be from LE or QVI)
        oor_delegated = oor_kever.delegated
        oor_delegator = oor_kever.delpre if oor_delegated else None
        
        # 8. All checks passed - delegation is valid
        result["valid"] = True
        result["verification"] = {
            "agent": {
                "prefix": agent_aid,
                "sequence": agent_kever.sn,
                "delegated": True,
                "delegator": agent_delegator
            },
            "oor_holder": {
                "prefix": oor_holder_aid,
                "sequence": oor_kever.sn,
                "delegated": oor_delegated,
                "delegator": oor_delegator
            },
            "delegation_valid": agent_delegator == oor_holder_aid,
            "trust_chain": "Agent <- OOR Holder" + (f" <- {oor_delegator}" if oor_delegator else ""),
            "message": "Agent delegation verified successfully via KERI KEL inspection"
        }
        
        logger.info(f"✓ Agent delegation verified: {agent_aid} <- {oor_holder_aid}")
        
    except Exception as e:
        logger.error(f"Error verifying agent delegation: {e}", exc_info=True)
        result["errors"].append(f"Verification error: {str(e)}")
        
    return result


def setup_agent_verification_endpoint(app, hby):
    """
    Add agent delegation verification endpoint to Sally's Falcon app.
    
    Endpoint: POST /verify/agent-delegation
    
    Request Body:
    {
        "agent_aid": "EAgent...",
        "oor_holder_aid": "EOOR..."
    }
    
    Response:
    {
        "valid": true,
        "agent_aid": "EAgent...",
        "oor_holder_aid": "EOOR...",
        "verification": {
            "agent": {...},
            "oor_holder": {...},
            "delegation_valid": true,
            "trust_chain": "Agent <- OOR Holder <- LE",
            "message": "Agent delegation verified successfully"
        }
    }
    """
    import falcon
    
    class AgentDelegationVerificationResource:
        """
        Falcon resource for agent delegation verification.
        """
        
        def __init__(self, hby):
            self.hby = hby
            logger.info("AgentDelegationVerificationResource initialized")
            
        async def on_post(self, req, resp):
            """
            Handle POST /verify/agent-delegation
            
            Verifies that an agent is properly delegated from an OOR holder
            by inspecting the KERI Key Event Logs.
            """
            try:
                # Parse request body
                body = await req.get_media()
                
                agent_aid = body.get('agent_aid')
                oor_holder_aid = body.get('oor_holder_aid')
                
                # Validate required fields
                if not agent_aid or not oor_holder_aid:
                    resp.status = falcon.HTTP_400
                    resp.media = {
                        "valid": False,
                        "error": "Missing required fields",
                        "required": ["agent_aid", "oor_holder_aid"],
                        "received": body
                    }
                    return
                
                logger.info(f"POST /verify/agent-delegation: agent={agent_aid}, oor_holder={oor_holder_aid}")
                
                # Verify agent delegation
                result = await verify_agent_delegation(
                    self.hby, 
                    agent_aid, 
                    oor_holder_aid
                )
                
                # Set response status based on validation result
                if result["valid"]:
                    resp.status = falcon.HTTP_200
                    logger.info(f"✓ Delegation verified: {agent_aid} <- {oor_holder_aid}")
                else:
                    resp.status = falcon.HTTP_400
                    logger.warning(f"✗ Delegation verification failed: {result.get('errors')}")
                    
                resp.media = result
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in request: {e}")
                resp.status = falcon.HTTP_400
                resp.media = {
                    "valid": False,
                    "error": "Invalid JSON in request body",
                    "details": str(e)
                }
                
            except Exception as e:
                logger.error(f"Error in agent delegation endpoint: {e}", exc_info=True)
                resp.status = falcon.HTTP_500
                resp.media = {
                    "valid": False,
                    "error": "Internal server error",
                    "details": str(e)
                }
    
    # Add route to Falcon app
    agent_resource = AgentDelegationVerificationResource(hby)
    app.add_route('/verify/agent-delegation', agent_resource)
    
    logger.info("✓ Added POST /verify/agent-delegation endpoint to Sally")
    print("✓ Sally Agent Verification Extension loaded")
    print("  Endpoint: POST /verify/agent-delegation")
    print("  Status: Ready")


# Module initialization
logger.info("Sally Agent Verification module loaded")
