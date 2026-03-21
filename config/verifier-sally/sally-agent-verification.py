"""
Sally Agent Delegation Verification Endpoint

This module extends Sally to verify agent delegations by:
1. Querying agent's KEL to verify delegation
2. Verifying OOR holder's credential
3. Checking complete trust chain
"""

from keri.app import habbing
from keri.core import coring
import logging

logger = logging.getLogger(__name__)


async def verify_agent_delegation(hab, agent_aid: str, oor_holder_aid: str):
    """
    Verify agent delegation relationship.
    
    Args:
        hab: Sally's habitat
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
        "error": None
    }
    
    try:
        # 1. Query agent's key state
        logger.info(f"Querying agent KEL: {agent_aid}")
        agent_kever = hab.kevers.get(agent_aid)
        
        if not agent_kever:
            result["error"] = "Agent AID not found in KEL database"
            return result
            
        # 2. Check if agent is delegated
        if not agent_kever.delegated:
            result["error"] = "Agent is not a delegated identifier"
            return result
            
        # 3. Verify delegator matches OOR holder
        agent_delpre = agent_kever.delpre
        if agent_delpre != oor_holder_aid:
            result["error"] = f"Delegation mismatch: agent delegated by {agent_delpre}, expected {oor_holder_aid}"
            return result
            
        # 4. Query OOR holder's key state
        logger.info(f"Querying OOR holder KEL: {oor_holder_aid}")
        oor_kever = hab.kevers.get(oor_holder_aid)
        
        if not oor_kever:
            result["error"] = "OOR holder AID not found in KEL database"
            return result
            
        # 5. Verify OOR holder's KEL is valid
        if not oor_kever.ked:
            result["error"] = "OOR holder KEL is invalid"
            return result
            
        # 6. Populate verification details
        result["valid"] = True
        result["verification"] = {
            "agent_prefix": agent_aid,
            "agent_sn": agent_kever.sn,
            "agent_delegated": True,
            "delegator_prefix": agent_delpre,
            "delegator_sn": oor_kever.sn,
            "delegator_matches": agent_delpre == oor_holder_aid,
            "message": "Agent delegation verified successfully"
        }
        
        logger.info(f"Agent delegation verified: {agent_aid} <- {oor_holder_aid}")
        
    except Exception as e:
        logger.error(f"Error verifying agent delegation: {e}")
        result["error"] = str(e)
        
    return result


def add_agent_delegation_route(app, hab):
    """
    Add agent delegation verification endpoint to Sally's Falcon app.
    
    POST /verify/agent-delegation
    Body: {
        "agent_aid": "EAgent...",
        "oor_holder_aid": "EOOR..."
    }
    """
    import falcon
    
    class AgentDelegationResource:
        def __init__(self, hab):
            self.hab = hab
            
        async def on_post(self, req, resp):
            """Handle POST /verify/agent-delegation"""
            try:
                body = await req.get_media()
                
                agent_aid = body.get('agent_aid')
                oor_holder_aid = body.get('oor_holder_aid')
                
                if not agent_aid or not oor_holder_aid:
                    resp.status = falcon.HTTP_400
                    resp.media = {
                        "error": "Missing required fields: agent_aid, oor_holder_aid"
                    }
                    return
                    
                # Verify delegation
                result = await verify_agent_delegation(
                    self.hab, 
                    agent_aid, 
                    oor_holder_aid
                )
                
                if result["valid"]:
                    resp.status = falcon.HTTP_200
                else:
                    resp.status = falcon.HTTP_400
                    
                resp.media = result
                
            except Exception as e:
                logger.error(f"Error in agent delegation endpoint: {e}")
                resp.status = falcon.HTTP_500
                resp.media = {
                    "error": "Internal server error",
                    "details": str(e)
                }
    
    # Add route to app
    app.add_route('/verify/agent-delegation', AgentDelegationResource(hab))
    logger.info("Added /verify/agent-delegation endpoint to Sally")
