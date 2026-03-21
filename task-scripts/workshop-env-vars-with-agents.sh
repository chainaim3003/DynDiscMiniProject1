#!/bin/bash
# workshop-env-vars.sh - Environment variables for vLEI workshop module

# Witness AID Prefixes
export WAN_WIT_PRE='BBilc4-L3tFUnfM_wJr4S4OJanAv_VmF_dJNN6vkf2Ha'
export GEDA_PRE='invalid-aid-will-be-replaced'

# GEDA AID details
# Fixed configuration for consistent root of trust
export GEDA_SALT="0AD45YWdzWSwNREuAoitH_CC"
export QVI_SALT='0ABZi_wCWBhxFNEenxtf40mL'
export LE_SALT='0ACzkItDY-F7lqI9ZtTQ1qR5'
export PERSON_SALT='0ADckowyGuNwtJUPLeRqZvTp'

# ===================================================================
# ✨ NEW: Agent-Specific Brans for Cryptographic Signing
# ===================================================================
# These brans enable agents to sign messages with KERIA
# Each delegated agent needs its OOR holder's bran to authenticate

# Jupiter Organization
export JUPITER_LE_SALT='0ACzkItDY-F7lqI9ZtTQ1qR5'  # Reusing LE_SALT for Jupiter
export JUPITER_OOR_SALT='0ADckowyGuNwtJUPLeRqZvTp'  # OOR holder (Chief Sales Officer)
export JUPITER_AGENT_SALT='0AEjupiterAgentSalt01'  # jupiterSellerAgent

# Tommy Organization  
export TOMMY_LE_SALT='0ACzkTommyLE_Salt_Here'
export TOMMY_OOR_SALT='0ADckoTommyOORSalt01'  # OOR holder (Chief Procurement Officer)
export TOMMY_AGENT_SALT='0AEtommyAgentSaltHere'  # tommyBuyerAgent

# ===================================================================

export GEDA_REGISTRY='geda-qvi-registry'

export QVI_SCHEMA_SAID='EBfdlu8R27Fbx-ehrqwImnK-8Cm79sqbAQ4MmvEAYqao'
export LE_SCHEMA_SAID='ENPXp1vQzRF6JwIuS-mp2U8Uf1MoADoP_GqQ62VsDZWY'
export OOR_AUTH_SCHEMA_SAID='EKA57bKBKxr_kN7iN5i7lMUxpMG-s19dRcmov1iDxz-E'
export OOR_SCHEMA_SAID='EBNaNu-M9P5cgrnfl2Fvymy4E_jvxxyjb70PRtiANlJy'
export ECR_AUTH_SCHEMA_SAID='EH6ekLjSr8V32WyFbGe1zXjTzFs9PkTYmupJ9H65O14g'
export ECR_SCHEMA_SAID='EEy9PkikFcANV1l7EHukCeXqrzT1hNZjGlUk7wuMO5jw'
export GEDA_AID_NAME='geda'
export QVI_AID_NAME='qvi'

export VERIFIER_AID='EMrjKv0T43sslqFfhlEHC9v3t9UoxHWrGznQ1EveRXUO'
export VERIFIER_OOBI='http://verifier:9723/oobi'
