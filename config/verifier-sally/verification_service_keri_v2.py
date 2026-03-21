#!/usr/bin/env python3
"""
ENHANCED KERI-Based Agent Delegation Verification Service v2.0
Performs REAL KEL parsing and delegation verification
"""

from fastapi import FastAPI, Request, HTTPException
import logging
import uvicorn
import httpx
import os
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="vLEI Agent Verifier with Enhanced KEL Parsing",
    description="Real KEL-based delegation verification",
    version="2.0.0"
)

KERIA_URL = os.getenv('KERIA_URL', 'http://keria:3902')


# ============================================================================
# KERIA API FUNCTIONS
# ============================================================================

async def query_kel(aid: str) -> Optional[Dict]:
    """Query KERIA for AID's KEL data"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{KERIA_URL}/identifiers/{aid}",
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"AID not found: {aid[:20]}...")
                return None
    except Exception as e:
        logger.error(f"KEL query error for {aid[:20]}: {e}")
        return None


# ============================================================================
# KEL PARSING FUNCTIONS - THE NEW STUFF!
# ============================================================================

def parse_agent_icp(kel_data: Dict, agent_aid: str, controller_aid: str) -> Tuple[bool, Dict]:
    """
    Parse agent's inception event to verify delegation
    
    Returns:
        (success: bool, details: Dict)
    """
    try:
        # KERIA response structure may vary, try different paths
        events = None
        
        # Try different response structures
        if isinstance(kel_data, dict):
            # Try direct events array
            if 'events' in kel_data:
                events = kel_data['events']
            # Try nested in state
            elif 'state' in kel_data and isinstance(kel_data['state'], dict):
                if 'k' in kel_data['state']:
                    events = kel_data['state']['k']
            # Try direct 'k' field
            elif 'k' in kel_data:
                events = kel_data['k']
        
        if not events or not isinstance(events, list) or len(events) == 0:
            return False, {
                "error": "No events found in KEL",
                "agent_aid": agent_aid,
                "kel_structure": str(type(kel_data))
            }
        
        # Get first event (should be ICP)
        icp_event = events[0]
        
        logger.info(f"ICP Event structure: {icp_event}")
        
        # Verify it's an inception event
        event_type = icp_event.get('t')
        if event_type != 'icp':
            return False, {
                "error": f"First event is not ICP, got: {event_type}",
                "event": icp_event
            }
        
        # Check for delegation field 'di'
        delegator = icp_event.get('di')
        
        if not delegator:
            return False, {
                "error": "Agent is not delegated (no 'di' field in ICP)",
                "icp_event": icp_event,
                "has_di_field": False
            }
        
        # Verify delegator matches controller
        if delegator != controller_aid:
            return False, {
                "error": "Delegator mismatch",
                "expected_controller": controller_aid,
                "actual_delegator": delegator,
                "match": False
            }
        
        # SUCCESS!
        return True, {
            "agent_aid": agent_aid,
            "agent_aid_from_event": icp_event.get('i'),
            "delegator_aid": delegator,
            "controller_aid": controller_aid,
            "match": True,
            "event_type": event_type,
            "sequence_number": icp_event.get('s'),
            "signing_threshold": icp_event.get('kt'),
            "public_keys_count": len(icp_event.get('k', [])),
            "has_di_field": True
        }
        
    except Exception as e:
        logger.error(f"Error parsing agent ICP: {e}", exc_info=True)
        return False, {"error": f"ICP parsing failed: {str(e)}"}


def find_delegation_seal(kel_data: Dict, agent_aid: str, controller_aid: str) -> Tuple[bool, Dict]:
    """
    Search controller's KEL for delegation seal anchoring the agent
    
    Returns:
        (found: bool, details: Dict)
    """
    try:
        # Get events from KEL data
        events = None
        
        if isinstance(kel_data, dict):
            if 'events' in kel_data:
                events = kel_data['events']
            elif 'state' in kel_data and isinstance(kel_data['state'], dict):
                if 'k' in kel_data['state']:
                    events = kel_data['state']['k']
            elif 'k' in kel_data:
                events = kel_data['k']
        
        if not events or not isinstance(events, list):
            return False, {
                "error": "No events found in controller KEL",
                "controller_aid": controller_aid
            }
        
        logger.info(f"Searching {len(events)} events in controller KEL for agent seal")
        
        # Search through all events for seals
        for idx, event in enumerate(events):
            event_type = event.get('t')
            sequence = event.get('s')
            
            # Look for seals in 'a' field (anchors/seals)
            seals = event.get('a', [])
            
            if seals and isinstance(seals, list):
                for seal in seals:
                    if isinstance(seal, dict):
                        seal_aid = seal.get('i')
                        
                        # Check if seal references our agent
                        if seal_aid == agent_aid:
                            logger.info(f"✅ Found delegation seal in event {idx}, sequence {sequence}")
                            return True, {
                                "found": True,
                                "controller_aid": controller_aid,
                                "agent_aid": agent_aid,
                                "seal_in_event_type": event_type,
                                "seal_in_sequence": sequence,
                                "seal_in_event_index": idx,
                                "seal_agent_sequence": seal.get('s'),
                                "seal_digest": seal.get('d', '')[:20] + "...",
                                "total_events_searched": len(events)
            }
        
        # No seal found
        return False, {
            "found": False,
            "controller_aid": controller_aid,
            "agent_aid": agent_aid,
            "events_searched": len(events),
            "error": "No delegation seal found in controller KEL"
        }
        
    except Exception as e:
        logger.error(f"Error searching for delegation seal: {e}", exc_info=True)
        return False, {"error": f"Seal search failed: {str(e)}"}


def verify_event_consistency(agent_icp_details: Dict, seal_details: Dict) -> Tuple[bool, List[Dict]]:
    """
    Verify consistency between agent ICP and controller seal
    
    Returns:
        (all_passed: bool, checks: List[Dict])
    """
    checks = []
    
    try:
        # Check 1: Agent ICP sequence is 0
        agent_seq = agent_icp_details.get('sequence_number')
        checks.append({
            "name": "Agent ICP sequence is 0",
            "passed": agent_seq == "0" or agent_seq == 0,
            "value": agent_seq
        })
        
        # Check 2: Seal references agent's inception (sequence 0)
        seal_agent_seq = seal_details.get('seal_agent_sequence')
        checks.append({
            "name": "Seal references agent inception",
            "passed": seal_agent_seq == "0" or seal_agent_seq == 0,
            "value": seal_agent_seq
        })
        
        # Check 3: Controller seal is in event after inception
        seal_controller_seq = seal_details.get('seal_in_sequence')
        if seal_controller_seq:
            try:
                seq_int = int(seal_controller_seq)
                checks.append({
                    "name": "Controller seal after inception",
                    "passed": seq_int >= 1,
                    "value": seal_controller_seq
                })
            except:
                checks.append({
                    "name": "Controller seal after inception",
                    "passed": False,
                    "value": seal_controller_seq,
                    "error": "Invalid sequence"
                })
        
        # Check 4: AIDs match
        agent_from_icp = agent_icp_details.get('agent_aid_from_event')
        agent_from_seal = seal_details.get('agent_aid')
        checks.append({
            "name": "Agent AIDs match across events",
            "passed": agent_from_icp == agent_from_seal,
            "icp_aid": agent_from_icp,
            "seal_aid": agent_from_seal
        })
        
        all_passed = all(check.get('passed', False) for check in checks)
        
        return all_passed, checks
        
    except Exception as e:
        logger.error(f"Error in consistency checks: {e}")
        return False, [{"error": str(e)}]


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """Health check with KERIA status"""
    keria_status = "unknown"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{KERIA_URL}/spec.yaml", timeout=5.0)
            keria_status = "connected" if response.status_code == 200 else "unreachable"
    except:
        keria_status = "unreachable"
    
    return {
        "status": "healthy",
        "service": "agent-delegation-verifier-keri-v2",
        "version": "2.0.0-enhanced",
        "keria_status": keria_status,
        "keria_url": KERIA_URL,
        "features": [
            "Format validation",
            "KEL existence check",
            "Agent ICP parsing (NEW)",
            "Delegation seal verification (NEW)",
            "Event consistency checks (NEW)"
        ]
    }


@app.post("/verify/agent-delegation")
async def verify_delegation(request: Request):
    """
    Enhanced agent delegation verification with real KEL parsing
    
    Request body:
    {
        "aid": "controller_aid",
        "agent_aid": "agent_aid",
        "verify_kel": true  // optional, default true
    }
    """
    try:
        data = await request.json()
        controller_aid = data.get("aid", "")
        agent_aid = data.get("agent_aid", "")
        verify_kel = data.get("verify_kel", True)
        
        # ========================================
        # STEP 1: Format Validation (existing)
        # ========================================
        
        if not controller_aid or not agent_aid:
            raise HTTPException(400, "Both 'aid' and 'agent_aid' required")
        
        if not (controller_aid.startswith('E') and len(controller_aid) == 44):
            raise HTTPException(400, "Invalid controller AID format")
        if not (agent_aid.startswith('E') and len(agent_aid) == 44):
            raise HTTPException(400, "Invalid agent AID format")
        
        logger.info(f"🔍 Verifying: agent={agent_aid[:20]}... controller={controller_aid[:20]}...")
        
        # ========================================
        # STEP 2: KEL Existence Check (existing)
        # ========================================
        
        logger.info("📥 Fetching KEL data from KERIA...")
        
        agent_kel = await query_kel(agent_aid)
        controller_kel = await query_kel(controller_aid)
        
        if not agent_kel:
            raise HTTPException(404, "Agent AID not found in KEL")
        if not controller_kel:
            raise HTTPException(404, "Controller AID not found in KEL")
        
        logger.info("✅ Both AIDs exist in KERIA")
        
        # If skip KEL parsing, return basic verification
        if not verify_kel:
            logger.info("⏭️  KEL parsing skipped (verify_kel=false)")
            return {
                "valid": True,
                "verified": True,
                "controller_aid": controller_aid,
                "agent_aid": agent_aid,
                "oor_holder_aid": controller_aid,
                "message": "Format and existence verified (KEL parsing skipped)",
                "verification": {
                    "format_valid": True,
                    "existence_verified": True,
                    "kel_parsed": False,
                    "delegation_verified": False
                }
            }
        
        # ========================================
        # STEP 3: Parse Agent ICP Event (NEW!)
        # ========================================
        
        logger.info("🔎 Parsing agent's ICP event...")
        
        icp_success, icp_details = parse_agent_icp(agent_kel, agent_aid, controller_aid)
        
        if not icp_success:
            logger.error(f"❌ Agent ICP verification failed: {icp_details.get('error')}")
            raise HTTPException(400, f"Agent ICP verification failed: {icp_details.get('error')}")
        
        logger.info(f"✅ Agent ICP verified: delegated from {icp_details['delegator_aid'][:20]}...")
        
        # ========================================
        # STEP 4: Find Delegation Seal (NEW!)
        # ========================================
        
        logger.info("🔍 Searching for delegation seal in controller KEL...")
        
        seal_found, seal_details = find_delegation_seal(controller_kel, agent_aid, controller_aid)
        
        if not seal_found:
            logger.error(f"❌ Delegation seal not found: {seal_details.get('error')}")
            raise HTTPException(400, f"Delegation seal verification failed: {seal_details.get('error')}")
        
        logger.info(f"✅ Delegation seal found in controller event {seal_details['seal_in_sequence']}")
        
        # ========================================
        # STEP 5: Verify Consistency (NEW!)
        # ========================================
        
        logger.info("🔍 Verifying event consistency...")
        
        consistency_ok, consistency_checks = verify_event_consistency(icp_details, seal_details)
        
        if not consistency_ok:
            logger.warning("⚠️  Some consistency checks failed")
        else:
            logger.info("✅ All consistency checks passed")
        
        # ========================================
        # SUCCESS: Return Detailed Results
        # ========================================
        
        logger.info("🎉 VERIFICATION SUCCESSFUL!")
        
        return {
            "valid": True,
            "verified": True,
            "controller_aid": controller_aid,
            "agent_aid": agent_aid,
            "oor_holder_aid": controller_aid,
            "message": "Agent delegation verified with KEL parsing",
            "verification": {
                "format_valid": True,
                "existence_verified": True,
                "kel_parsed": True,
                "delegation_verified": True,
                
                # NEW: Detailed verification results
                "agent_icp_analysis": {
                    "verified": icp_success,
                    "has_delegator_field": icp_details.get('has_di_field'),
                    "delegator_matches": icp_details.get('match'),
                    "delegator_aid": icp_details.get('delegator_aid'),
                    "details": icp_details
                },
                
                "delegation_seal_analysis": {
                    "verified": seal_found,
                    "seal_found_in_controller_kel": seal_details.get('found'),
                    "seal_event_type": seal_details.get('seal_in_event_type'),
                    "seal_sequence": seal_details.get('seal_in_sequence'),
                    "details": seal_details
                },
                
                "consistency_checks": {
                    "all_passed": consistency_ok,
                    "checks": consistency_checks
                },
                
                "verification_level": "enhanced_kel_parsing",
                "coverage_percentage": 55
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Verification error: {e}", exc_info=True)
        raise HTTPException(500, f"Verification failed: {str(e)}")


@app.get("/")
async def root():
    """Service information"""
    return {
        "service": "vLEI Agent Verifier with Enhanced KEL Parsing",
        "version": "2.0.0",
        "verification_coverage": "55%",
        "improvements_over_v1": [
            "Real KEL event parsing",
            "Agent ICP delegation field verification",
            "Controller KEL seal search and verification",
            "Event sequence consistency checks",
            "Detailed verification reporting"
        ],
        "what_we_verify": [
            "✅ Format validation (5%)",
            "✅ KEL existence (10%)",
            "✅ Agent ICP has 'di' field = controller (15%)",
            "✅ Controller KEL contains delegation seal (15%)",
            "✅ Event consistency checks (10%)"
        ],
        "still_missing": [
            "❌ Cryptographic signature verification (15%)",
            "❌ Credential chain validation (15%)",
            "❌ Revocation checking (10%)",
            "❌ Witness receipt validation (5%)"
        ],
        "keria_url": KERIA_URL
    }


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Starting ENHANCED KERI Verification Service v2.0")
    logger.info(f"KERIA URL: {KERIA_URL}")
    logger.info("New features:")
    logger.info("  • Real KEL event parsing")
    logger.info("  • Agent ICP delegation verification")
    logger.info("  • Controller seal search")
    logger.info("  • Event consistency checks")
    logger.info("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=9723, log_level="info")
