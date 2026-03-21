cat > task-scripts/subagent/subagent-delete-from-keria.sh << 'EOF'
#!/bin/bash
# subagent-delete-from-keria.sh - Delete sub-agent from KERIA

set -e

SUB_AGENT_ALIAS=$1

if [ -z "$SUB_AGENT_ALIAS" ]; then
    echo "Usage: $0 <subAgentAlias>"
    exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "  Delete Sub-Agent from KERIA"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Sub-Agent: $SUB_AGENT_ALIAS"
echo ""

# Load sub-agent BRAN
BRAN_FILE="task-data/${SUB_AGENT_ALIAS}-bran.txt"
if [ ! -f "$BRAN_FILE" ]; then
    echo "✗ BRAN file not found: $BRAN_FILE"
    exit 1
fi

SUB_BRAN=$(cat "$BRAN_FILE")
echo "✓ Sub-agent BRAN loaded"
echo ""

# Create temporary TypeScript script to delete the agent
cat > /tmp/delete-agent-temp.ts << 'EOTS'
import { getOrCreateClient } from '../../client/identifiers.js';

const args = process.argv.slice(2);
const bran = args[0];
const agentName = args[1];

async function deleteAgent() {
    try {
        console.log(`Connecting to KERIA with sub-agent BRAN...`);
        const client = await getOrCreateClient(bran, 'docker');
        
        console.log(`✓ Connected`);
        console.log(`  Controller: ${client.controller.pre}`);
        console.log(`  Agent: ${client.agent?.pre}`);
        console.log('');
        
        console.log(`Deleting identifier: ${agentName}`);
        await client.identifiers().delete(agentName);
        
        console.log(`✓ Identifier deleted from KERIA`);
        
    } catch (error: any) {
        console.error(`Error: ${error.message}`);
        if (error.message.includes('404') || error.message.includes('not found')) {
            console.log('(Agent may not exist - this is OK)');
            process.exit(0);
        }
        throw error;
    }
}

deleteAgent();
EOTS

# Run the deletion script
docker compose exec tsx-shell tsx /tmp/delete-agent-temp.ts "$SUB_BRAN" "$SUB_AGENT_ALIAS"

echo ""
echo "✓ Sub-agent deleted from KERIA"
echo ""

exit 0
EOF

chmod +x task-scripts/subagent/subagent-delete-from-keria.sh