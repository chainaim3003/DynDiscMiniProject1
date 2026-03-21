import { getOrCreateClient } from '../../client/identifiers.js';
import fs from 'fs';

const env = process.argv[2] as 'docker' | 'testnet';
const agentName = process.argv[3];

(async () => {
    try {
        const branPath = `/task-data/${agentName}-bran.txt`;
        const bran = fs.readFileSync(branPath, 'utf-8').trim();
        console.log(`Connecting with BRAN for ${agentName}...`);
        
        const client = await getOrCreateClient(bran, env);
        console.log('✓ Connected');
        
        console.log(`Deleting identifier: ${agentName}`);
        await client.identifiers().delete(agentName);
        console.log('✓ Deleted successfully');
    } catch (e: any) {
        if (e.message.includes('404') || e.message.includes('not found')) {
            console.log('✓ Already deleted or does not exist');
        } else {
            console.error('Error:', e.message);
            process.exit(1);
        }
    }
})();
