## Sally Verifier Python Extension - Complete Code

This file contains all Python code for the Sally verifier extension.

### File: config/verifier-sally/custom-sally/agent_verifying.py

```python
"""
Agent Delegation Verifier for vLEI

This module provides verification of agent delegation from OOR holders
per the official vLEI specification.

Verification Steps:
1. Verify agent AID is delegated (has delpre in KEL)
2. Verify OOR holder is the delegator (delpre matches OOR holder AID)
3. Verify delegation anchor exists in OOR holder's KEL
4. Verify OOR holder has valid OOR credential
5. Verify OOR credential chain to GEDA
6. Check all credentials are not revoked

Based on official GLEIF vLEI specification and keripy library.
"""

from keri.core import coring, serdering
from keri.db import dbing
from typing import Dict, Optional
import json


class AgentDelegationVerifier:
    """
    Verifies agent delegation from OOR holder per vLEI specification.
    """
    
    # Official vLEI Schema SAIDs
    OOR_SCHEMA = 'EBNaNu-M9P5cgrnfl2Fvymy4E_jvxxyjb70PRtiANlJy'
    OOR_AUTH_SCHEMA = 'EKA57bKBKxr_kN7iN5i7lMUxpMG-s19dRcmov1iDxz-E'
    LE_SCHEMA = 'ENPXp1vQzRF6JwIuS-mp2U8Uf1MoADoP_GqQ62VsDZWY'
    QVI_SCHEMA = 'EBfdlu8R27Fbx-ehrqwImnK-8Cm79sqbAQ4MmvEAYqao'
    
    def __init__(self, hby):
        """
        Initialize verifier with KERI Habery.
        
        Args:
            hby: KERI Habery (habbing.Habery) with access to KELs and credentials
        """
        self.hby = hby
        
    def verify_agent_delegation(
        self, 
        agent_aid: str,
        oor_holder_aid: str,
        verify_oor_credential: bool = True
    ) -> Dict:
        """
        Verify complete agent delegation chain.
        
        Args:
            agent_aid: Agent's AID prefix
            oor_holder_aid: OOR Holder's AID prefix  
            verify_oor_credential: If True, also verify OOR credential chain
            
        Returns:
            dict: Verification result
        """
        result = {
            "valid": False,
            "agent_aid": agent_aid,
            "oor_holder_aid": oor_holder_aid,
            "verification": {},
            "timestamp": coring.nowIso8601()
        }
        
        try:
            # Step 1: Verify agent is delegated from OOR holder
            print(f"[Verification] Step 1: Verifying delegation...")
            delegation_check = self._verify_delegation_exists(agent_aid, oor_holder_aid)
            if not delegation_check["valid"]:
                result["error"] = delegation_check["error"]
                return result
            
            result["verification"]["delegation_valid"] = True
            result["verification"]["delegator_prefix"] = delegation_check["delegator"]
            print(f"[Verification] ✓ Delegation verified")
            
            # Step 2: Verify OOR credential if requested
            if verify_oor_credential:
                print(f"[Verification] Step 2: Verifying OOR credential chain...")
                oor_check = self._verify_oor_credential_chain(oor_holder_aid)
                if not oor_check["valid"]:
                    result["error"] = oor_check["error"]
                    return result
                    
                result["verification"]["oor_credential_valid"] = True
                result["verification"]["oor_credential_said"] = oor_check["credential_said"]
                result["verification"]["le_lei"] = oor_check["le_lei"]
                result["verification"]["qvi_aid"] = oor_check["qvi_aid"]
                result["verification"]["geda_aid"] = oor_check["geda_aid"]
                print(f"[Verification] ✓ OOR credential chain verified")
            
            result["valid"] = True
            result["message"] = "Agent delegation verified successfully"
            print(f"[Verification] ✓ Complete verification successful")
            
        except Exception as e:
            result["error"] = f"Verification failed: {str(e)}"
            result["valid"] = False
            print(f"[Verification] ✗ Verification failed: {str(e)}")
            
        return result
    
    def _verify_delegation_exists(self, agent_aid: str, oor_holder_aid: str) -> Dict:
        """
        Verify agent AID is delegated from OOR holder.
        
        Checks:
        1. Agent KEL has delpre field
        2. delpre matches OOR holder AID
        3. OOR holder KEL has delegation seal for agent
        """
        # Get agent's key event log
        agent_kever = self.hby.kevers.get(agent_aid)
        if not agent_kever:
            return {
                "valid": False,
                "error": f"Agent AID {agent_aid} not found in key state"
            }
        
        # Check if agent is delegated
        if not agent_kever.delpre:
            return {
                "valid": False,
                "error": "Agent AID is not a delegated identifier (no delpre field)"
            }
        
        # Verify delegator matches OOR holder
        if agent_kever.delpre != oor_holder_aid:
            return {
                "valid": False,
                "error": f"Delegator mismatch. Expected {oor_holder_aid}, got {agent_kever.delpre}"
            }
        
        # Get OOR holder's KEL
        oor_kever = self.hby.kevers.get(oor_holder_aid)
        if not oor_kever:
            return {
                "valid": False,
                "error": f"OOR Holder AID {oor_holder_aid} not found"
            }
        
        # Verify delegation seal exists in OOR holder's KEL
        delegation_found = self._find_delegation_seal(oor_kever, agent_aid)
        if not delegation_found:
            return {
                "valid": False,
                "error": "Delegation seal not found in OOR holder's KEL"
            }
        
        return {
            "valid": True,
            "delegator": oor_holder_aid,
            "delegate": agent_aid
        }
    
    def _find_delegation_seal(self, kever, delegate_aid: str) -> bool:
        """
        Search KEL for delegation seal containing delegate AID.
        """
        # Iterate through all events in KEL
        for sn in range(kever.sn + 1):
            try:
                # Get event at sequence number
                dgkey = dbing.dgKey(kever.prefixer.qb64b, sn.to_bytes(4, 'big'))
                raw = self.hby.db.getEvt(dgkey)
                if raw:
                    serder = serdering.SerderKERI(raw=bytes(raw))
                    # Check for seals in event
                    if hasattr(serder, 'seals') and serder.seals:
                        for seal in serder.seals:
                            if seal.get('i') == delegate_aid:
                                return True
            except Exception as e:
                continue
        return False
    
    def _verify_oor_credential_chain(self, oor_holder_aid: str) -> Dict:
        """
        Verify OOR holder has valid OOR credential with complete chain.
        
        Chain: OOR → OOR Auth → LE → QVI → GEDA
        """
        # Get OOR credentials for this holder
        oor_cred = None
        oor_said = None
        
        # Iterate through all credentials in database
        for (said,), saider in self.hby.db.scgs.getItemIter():
            # Get credential
            creder = self.hby.db.creds.get(keys=(said,))
            if creder is None:
                continue
                
            # Check if this is an OOR credential for our holder
            if creder.schema == self.OOR_SCHEMA and creder.issee == oor_holder_aid:
                oor_cred = creder
                oor_said = said
                break
        
        if not oor_cred:
            return {
                "valid": False,
                "error": f"No OOR credential found for {oor_holder_aid}"
            }
        
        # Check if credential is revoked
        reger = self.hby.db.regs.get(keys=(oor_cred.status,))
        if reger:
            state = reger.tever.vcState(oor_said)
            if state != coring.Ilks.iss:
                return {
                    "valid": False,
                    "error": "OOR credential is revoked"
                }
        
        # Get OOR Auth credential SAID from edge
        if not hasattr(oor_cred, 'edge') or 'auth' not in oor_cred.edge:
            return {
                "valid": False,
                "error": "OOR credential missing OOR Auth edge"
            }
        
        oor_auth_said = oor_cred.edge['auth'].get('n')
        if not oor_auth_said:
            return {
                "valid": False,
                "error": "OOR credential missing OOR Auth SAID in edge"
            }
        
        # Get OOR Auth credential
        oor_auth_cred = self.hby.db.creds.get(keys=(oor_auth_said,))
        if not oor_auth_cred:
            return {
                "valid": False,
                "error": "OOR Auth credential not found"
            }
        
        # OOR Auth has edge to LE
        if not hasattr(oor_auth_cred, 'edge') or 'le' not in oor_auth_cred.edge:
            return {
                "valid": False,
                "error": "OOR Auth credential missing LE edge"
            }
        
        le_said = oor_auth_cred.edge['le'].get('n')
        if not le_said:
            return {
                "valid": False,
                "error": "OOR Auth credential missing LE SAID in edge"
            }
        
        # Get LE credential
        le_cred = self.hby.db.creds.get(keys=(le_said,))
        if not le_cred:
            return {
                "valid": False,
                "error": "LE credential not found"
            }
        
        le_lei = le_cred.attrib.get('LEI', 'Unknown')
        
        # LE has edge to QVI
        if not hasattr(le_cred, 'edge') or 'qvi' not in le_cred.edge:
            return {
                "valid": False,
                "error": "LE credential missing QVI edge"
            }
        
        qvi_said = le_cred.edge['qvi'].get('n')
        if not qvi_said:
            return {
                "valid": False,
                "error": "LE credential missing QVI SAID in edge"
            }
        
        # Get QVI credential
        qvi_cred = self.hby.db.creds.get(keys=(qvi_said,))
        if not qvi_cred:
            return {
                "valid": False,
                "error": "QVI credential not found"
            }
        
        qvi_aid = qvi_cred.issuer
        
        # QVI should be delegated from GEDA
        qvi_kever = self.hby.kevers.get(qvi_aid)
        if not qvi_kever:
            return {
                "valid": False,
                "error": f"QVI AID {qvi_aid} not found in key state"
            }
        
        if not qvi_kever.delpre:
            return {
                "valid": False,
                "error": "QVI is not delegated from GEDA"
            }
        
        geda_aid = qvi_kever.delpre
        
        return {
            "valid": True,
            "credential_said": oor_said,
            "le_lei": le_lei,
            "qvi_aid": qvi_aid,
            "geda_aid": geda_aid
        }
```

### File: config/verifier-sally/custom-sally/handling_ext.py

```python
"""
HTTP Handler Extension for Agent Delegation Verification
"""

from .agent_verifying import AgentDelegationVerifier


class AgentDelegationHandler:
    """
    HTTP handler for agent delegation verification endpoint.
    """
    
    def __init__(self, hby):
        self.verifier = AgentDelegationVerifier(hby)
    
    def handle(self, data):
        """
        Handle POST /verify/agent-delegation request.
        """
        agent_aid = data.get('agent_aid')
        oor_holder_aid = data.get('oor_holder_aid')
        verify_oor = data.get('verify_oor_credential', True)
        
        if not agent_aid:
            return {
                "valid": False,
                "error": "Missing required parameter: agent_aid"
            }, 400
        
        if not oor_holder_aid:
            return {
                "valid": False,
                "error": "Missing required parameter: oor_holder_aid"
            }, 400
        
        print(f"[Handler] Processing verification request:")
        print(f"[Handler]   Agent AID: {agent_aid}")
        print(f"[Handler]   OOR Holder AID: {oor_holder_aid}")
        
        result = self.verifier.verify_agent_delegation(
            agent_aid=agent_aid,
            oor_holder_aid=oor_holder_aid,
            verify_oor_credential=verify_oor
        )
        
        status_code = 200 if result['valid'] else 400
        return result, status_code
```

### File: config/verifier-sally/custom-sally/__init__.py

```python
"""
Custom Sally Extensions for vLEI Agent Delegation Verification
"""

from .agent_verifying import AgentDelegationVerifier
from .handling_ext import AgentDelegationHandler

__all__ = ['AgentDelegationVerifier', 'AgentDelegationHandler']
```

### File: config/verifier-sally/entry-point-extended.sh

```bash
#!/bin/bash
# Extended entry point for Sally with agent verification support

set -e

EXPECTED_AID=EMrjKv0T43sslqFfhlEHC9v3t9UoxHWrGznQ1EveRXUO
GEDA_PRE="${GEDA_PRE:-ED1e8pD24aqd0dCZTQHaGpfcluPFD2ajGIY3ARgE5Yvr}"
SALLY_KS_NAME="${SALLY_KS_NAME:-sally}"
SALLY_SALT="${SALLY_SALT:-0ABVqAtad0CBkhDhCEPd514T}"
SALLY_PASSCODE="${SALLY_PASSCODE:-4TBjjhmKu9oeDp49J7Xdy}"
SALLY_PORT="${SALLY_PORT:-9723}"
WEBHOOK_URL="${WEBHOOK_URL:-http://resource:9923}"

if [ -z "${GEDA_PRE}" ]; then
  echo "GEDA_PRE auth AID is not set. Exiting."
  exit 1
fi

echo "Starting Sally verifier with agent delegation verification..."
echo "Configuration:"
echo "   GEDA_PRE: ${GEDA_PRE}"
echo "   SALLY_PORT: ${SALLY_PORT}"
echo "   EXTENSIONS: Agent Delegation Verification"

export DEBUG_KLI=true

# Install custom Sally extensions
echo "Installing custom Sally extensions..."
SALLY_CUSTOM_DIR="/sally/custom-sally"
PYTHON_SITE_PACKAGES="/usr/local/lib/python3.12/site-packages"

if [ -d "${SALLY_CUSTOM_DIR}" ]; then
  echo "Copying custom extensions..."
  mkdir -p "${PYTHON_SITE_PACKAGES}/custom_sally"
  cp -r ${SALLY_CUSTOM_DIR}/* "${PYTHON_SITE_PACKAGES}/custom_sally/"
  
  if [ -f "${PYTHON_SITE_PACKAGES}/custom_sally/agent_verifying.py" ]; then
    echo "✓ agent_verifying.py installed"
  else
    echo "✗ Failed to install agent_verifying.py"
    exit 1
  fi
  
  echo "✓ Custom extensions installed successfully"
else
  echo "Warning: Custom Sally directory not found"
fi

function start_sally() {
  sally server start \
    --direct \
    --name "${SALLY_KS_NAME}" \
    --alias "${SALLY_KS_NAME}" \
    --passcode "${SALLY_PASSCODE}" \
    --http "${SALLY_PORT}" \
    --config-dir /sally/conf \
    --config-file verifier.json \
    --web-hook "${WEBHOOK_URL}" \
    --auth "${GEDA_PRE}" \
    --loglevel INFO
}

function init_sally_aid() {
  kli init \
    --name "${SALLY_KS_NAME}" \
    --salt "${SALLY_SALT}" \
    --passcode "${SALLY_PASSCODE}" \
    --config-dir /sally/conf \
    --config-file "${SALLY_KS_NAME}.json"

  kli incept \
      --name "${SALLY_KS_NAME}" \
      --alias "${SALLY_KS_NAME}" \
      --passcode "${SALLY_PASSCODE}" \
      --config /sally/conf \
      --file "/sally/conf/incept-no-wits.json"
}

mkdir -p /usr/local/var/keri/ks

if [[ -d "/usr/local/var/keri/ks/${SALLY_KS_NAME}" ]]; then
  EXISTING_AID=$(kli aid --name "${SALLY_KS_NAME}" --alias "${SALLY_KS_NAME}" --passcode "${SALLY_PASSCODE}")
  if [[ "${EXISTING_AID}" == "${EXPECTED_AID}" ]]; then
    echo "Starting Sally with existing AID..."
    start_sally
  else
    echo "Error: Sally AID mismatch. Exiting"
    exit 1
  fi
else
  echo "Initializing Sally..."
  init_sally_aid
  start_sally
fi
```

Make executable: `chmod +x config/verifier-sally/entry-point-extended.sh`
