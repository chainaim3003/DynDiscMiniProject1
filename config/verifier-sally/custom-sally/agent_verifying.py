"""
Agent Delegation Verification Logic

This module implements verification of agent delegation in the vLEI context.
An agent is delegated by an OOR (Organization Organizational Role) holder.

Verification Steps:
1. Verify agent's KEL shows delegation from OOR holder
2. Verify OOR holder's KEL contains delegation seal for agent
3. Retrieve and verify OOR holder's OOR credential
4. Verify complete credential chain (OOR → OOR Auth → LE → QVI → GEDA)
5. Check for revocations at each level
"""

import json
from typing import Dict, Any, Optional
from keri.app import habbing
from keri.core import coring, eventing
from keri.vdr import verifying


class AgentDelegationVerifier:
    """Verifies agent delegation chains in vLEI context"""
    
    def __init__(self, hby: habbing.Habery):
        """
        Initialize verifier with KERI habery
        
        Args:
            hby: KERI habery instance for accessing KELs and credentials
        """
        self.hby = hby
        self.reger = verifying.Reger(name=hby.name, temp=False)
    
    def verify_agent_delegation(
        self, 
        agent_aid: str, 
        oor_holder_aid: str
    ) -> Dict[str, Any]:
        """
        Verify that an agent is properly delegated by an OOR holder
        
        Args:
            agent_aid: Agent's AID (prefix)
            oor_holder_aid: OOR Holder's AID (prefix)
            
        Returns:
            Dictionary with verification result:
            {
                "valid": bool,
                "agent_aid": str,
                "oor_holder_aid": str,
                "oor_credential_said": str (if valid),
                "credential_chain": list (if valid),
                "error": str (if invalid)
            }
        """
        try:
            # Step 1: Verify agent KEL shows delegation
            agent_hab = self.hby.habByName(agent_aid)
            if not agent_hab:
                return {
                    "valid": False,
                    "agent_aid": agent_aid,
                    "oor_holder_aid": oor_holder_aid,
                    "error": f"Agent AID {agent_aid} not found in local KERI database"
                }
            
            # Check agent is delegated
            if not agent_hab.kever.delpre:
                return {
                    "valid": False,
                    "agent_aid": agent_aid,
                    "oor_holder_aid": oor_holder_aid,
                    "error": "Agent is not a delegated AID"
                }
            
            # Verify delegation is from expected OOR holder
            if agent_hab.kever.delpre != oor_holder_aid:
                return {
                    "valid": False,
                    "agent_aid": agent_aid,
                    "oor_holder_aid": oor_holder_aid,
                    "error": f"Agent is delegated by {agent_hab.kever.delpre}, not {oor_holder_aid}"
                }
            
            # Step 2: Verify OOR holder KEL contains delegation seal
            oor_hab = self.hby.habByName(oor_holder_aid)
            if not oor_hab:
                return {
                    "valid": False,
                    "agent_aid": agent_aid,
                    "oor_holder_aid": oor_holder_aid,
                    "error": f"OOR Holder AID {oor_holder_aid} not found"
                }
            
            delegation_seal_found = self._verify_delegation_seal(
                oor_hab, agent_aid
            )
            if not delegation_seal_found:
                return {
                    "valid": False,
                    "agent_aid": agent_aid,
                    "oor_holder_aid": oor_holder_aid,
                    "error": "Delegation seal not found in OOR holder's KEL"
                }
            
            # Step 3: Get OOR holder's OOR credential
            oor_credential = self._get_oor_credential(oor_holder_aid)
            if not oor_credential:
                return {
                    "valid": False,
                    "agent_aid": agent_aid,
                    "oor_holder_aid": oor_holder_aid,
                    "error": "OOR credential not found for OOR holder"
                }
            
            # Step 4: Verify credential chain
            chain_result = self._verify_credential_chain(oor_credential)
            if not chain_result["valid"]:
                return {
                    "valid": False,
                    "agent_aid": agent_aid,
                    "oor_holder_aid": oor_holder_aid,
                    "error": f"Credential chain verification failed: {chain_result['error']}"
                }
            
            # Step 5: Check revocations
            revocation_check = self._check_revocations(chain_result["chain"])
            if not revocation_check["valid"]:
                return {
                    "valid": False,
                    "agent_aid": agent_aid,
                    "oor_holder_aid": oor_holder_aid,
                    "error": f"Revocation found: {revocation_check['error']}"
                }
            
            # All checks passed
            return {
                "valid": True,
                "agent_aid": agent_aid,
                "oor_holder_aid": oor_holder_aid,
                "oor_credential_said": oor_credential.get("sad", {}).get("d"),
                "credential_chain": chain_result["chain"],
                "verification_timestamp": coring.Dater().dts
            }
            
        except Exception as e:
            return {
                "valid": False,
                "agent_aid": agent_aid,
                "oor_holder_aid": oor_holder_aid,
                "error": f"Verification exception: {str(e)}"
            }
    
    def _verify_delegation_seal(
        self, 
        delegator_hab: habbing.Habitat, 
        delegatee_aid: str
    ) -> bool:
        """
        Verify delegation seal exists in delegator's KEL
        
        Args:
            delegator_hab: Delegator's habitat
            delegatee_aid: Delegatee's AID
            
        Returns:
            True if delegation seal found
        """
        # Check for delegation seal in interaction events
        for event in delegator_hab.kever.events:
            if event.get("t") == "ixn":  # Interaction event
                seals = event.get("a", [])
                for seal in seals:
                    if seal.get("i") == delegatee_aid:
                        return True
        return False
    
    def _get_oor_credential(self, oor_holder_aid: str) -> Optional[Dict[str, Any]]:
        """
        Get OOR credential for the OOR holder
        
        Args:
            oor_holder_aid: OOR Holder's AID
            
        Returns:
            OOR credential dict or None
        """
        # Query registry for credentials issued to this AID
        credentials = self.reger.cloneCreds(
            said=None,
            limit=100
        )
        
        for cred in credentials:
            sad = cred.get("sad", {})
            # Check if this is an OOR credential for our AID
            if (sad.get("a", {}).get("i") == oor_holder_aid and
                sad.get("s") and "OORAuthorizationvLEICredential" in sad.get("s")):
                return cred
        
        return None
    
    def _verify_credential_chain(
        self, 
        oor_credential: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify the complete credential chain
        
        Chain: OOR → OOR Auth → LE → QVI → GEDA
        
        Args:
            oor_credential: Starting OOR credential
            
        Returns:
            {"valid": bool, "chain": list, "error": str}
        """
        chain = [oor_credential]
        current_cred = oor_credential
        
        try:
            # Walk up the chain using issuer references
            while True:
                sad = current_cred.get("sad", {})
                issuer_aid = sad.get("i")
                
                if not issuer_aid:
                    break
                
                # Get issuer's credential
                issuer_cred = self._get_credential_for_issuer(issuer_aid)
                if not issuer_cred:
                    # Reached top (GEDA) or chain broken
                    break
                
                chain.append(issuer_cred)
                current_cred = issuer_cred
                
                # Prevent infinite loops
                if len(chain) > 10:
                    return {
                        "valid": False,
                        "error": "Credential chain too long (possible loop)"
                    }
            
            # Verify minimum chain length (should have OOR Auth, LE, QVI at minimum)
            if len(chain) < 3:
                return {
                    "valid": False,
                    "error": f"Credential chain too short ({len(chain)} credentials)"
                }
            
            return {
                "valid": True,
                "chain": chain
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Chain verification error: {str(e)}"
            }
    
    def _get_credential_for_issuer(self, issuer_aid: str) -> Optional[Dict[str, Any]]:
        """
        Get credential issued to an issuer (to walk up chain)
        
        Args:
            issuer_aid: Issuer's AID
            
        Returns:
            Credential dict or None
        """
        credentials = self.reger.cloneCreds(said=None, limit=100)
        
        for cred in credentials:
            sad = cred.get("sad", {})
            if sad.get("a", {}).get("i") == issuer_aid:
                return cred
        
        return None
    
    def _check_revocations(self, credential_chain: list) -> Dict[str, Any]:
        """
        Check if any credential in chain is revoked
        
        Args:
            credential_chain: List of credentials to check
            
        Returns:
            {"valid": bool, "error": str}
        """
        for idx, cred in enumerate(credential_chain):
            sad = cred.get("sad", {})
            cred_said = sad.get("d")
            
            if not cred_said:
                continue
            
            # Check revocation registry
            if self.reger.reger.getTvt(coring.Diger(qb64=cred_said)):
                return {
                    "valid": False,
                    "error": f"Credential at chain position {idx} is revoked"
                }
        
        return {"valid": True}


def create_verifier(hby: habbing.Habery) -> AgentDelegationVerifier:
    """
    Factory function to create agent delegation verifier
    
    Args:
        hby: KERI habery instance
        
    Returns:
        AgentDelegationVerifier instance
    """
    return AgentDelegationVerifier(hby)
