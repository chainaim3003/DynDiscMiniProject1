#!/bin/bash
# workshop-env-vars-buyerseller.sh
# Unique salts for buyer-seller scenario with agents

# Witness AID Prefixes  
export WAN_WIT_PRE='BBilc4-L3tFUnfM_wJr4S4OJanAv_VmF_dJNN6vkf2Ha'
export GEDA_PRE='invalid-aid-will-be-replaced'

# ===================================================================
# ROOT & QVI (Reuse existing)
# ===================================================================
export GEDA_SALT="0AD45YWdzWSwNREuAoitH_CC"
export QVI_SALT='0ABZi_wCWBhxFNEenxtf40mL'

# ===================================================================
# JUPITER KNITTING COMPANY (Seller Organization)
# ===================================================================
export JUPITER_LE_SALT='0ACjupiterLE_SaltHere'
export JUPITER_OOR_SALT='0ADjupiterOOR_Salt123'  # Chief Sales Officer
export JUPITER_AGENT_SALT='0AEjupiterAgent_Salt'  # jupiterSellerAgent

# ===================================================================
# TOMMY HILFIGER EUROPE (Buyer Organization)
# ===================================================================
export TOMMY_LE_SALT='0ACtommyLE_Salt_12345'
export TOMMY_OOR_SALT='0ADtommyOOR_Salt_1234'  # Chief Procurement Officer  
export TOMMY_AGENT_SALT='0AEtommyAgent_Salt01'  # tommyBuyerAgent

# ===================================================================
# Legacy compatibility (for scripts that still use these)
# ===================================================================
export LE_SALT='0ACzkItDY-F7lqI9ZtTQ1qR5'
export PERSON_SALT='0ADckowyGuNwtJUPLeRqZvTp'

# ===================================================================
# Registries
# ===================================================================
export GEDA_REGISTRY='geda-qvi-registry'

# ===================================================================
# Schema SAIDs
# ===================================================================
export QVI_SCHEMA_SAID='EBfdlu8R27Fbx-ehrqwImnK-8Cm79sqbAQ4MmvEAYqao'
export LE_SCHEMA_SAID='ENPXp1vQzRF6JwIuS-mp2U8Uf1MoADoP_GqQ62VsDZWY'
export OOR_AUTH_SCHEMA_SAID='EKA57bKBKxr_kN7iN5i7lMUxpMG-s19dRcmov1iDxz-E'
export OOR_SCHEMA_SAID='EBNaNu-M9P5cgrnfl2Fvymy4E_jvxxyjb70PRtiANlJy'
export ECR_AUTH_SCHEMA_SAID='EH6ekLjSr8V32WyFbGe1zXjTzFs9PkTYmupJ9H65O14g'
export ECR_SCHEMA_SAID='EEy9PkikFcANV1l7EHukCeXqrzT1hNZjGlUk7wuMO5jw'

# ===================================================================
# AID Names
# ===================================================================
export GEDA_AID_NAME='geda'
export QVI_AID_NAME='qvi'

# ===================================================================
# Verifier
# ===================================================================
export VERIFIER_AID='EMrjKv0T43sslqFfhlEHC9v3t9UoxHWrGznQ1EveRXUO'
export VERIFIER_OOBI='http://verifier:9723/oobi'

echo "✅ Buyer-Seller environment variables loaded"
echo "   Jupiter LE Salt: ${JUPITER_LE_SALT}"
echo "   Jupiter OOR Salt: ${JUPITER_OOR_SALT}"
echo "   Jupiter Agent Salt: ${JUPITER_AGENT_SALT}"
echo "   Tommy LE Salt: ${TOMMY_LE_SALT}"
echo "   Tommy OOR Salt: ${TOMMY_OOR_SALT}"
echo "   Tommy Agent Salt: ${TOMMY_AGENT_SALT}"
