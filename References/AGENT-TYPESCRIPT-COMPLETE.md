## Additional TypeScript Files (Continued from main doc)

### agent-oobi-resolve-qvi.ts
```typescript
import {getOrCreateClient} from "../../client/identifiers.js";
import {resolveOobi} from "../../client/oobis.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const agentPasscode = args[1];
const agentName = args[2];
const qviOobi = args[3];

if (!env || !agentPasscode || !agentName || !qviOobi) {
    console.error('Usage: tsx agent-oobi-resolve-qvi.ts <env> <agentPasscode> <agentName> <qviOobi>');
    process.exit(1);
}

console.log(`Agent ${agentName} resolving QVI OOBI: ${qviOobi}`);
const agentClient = await getOrCreateClient(agentPasscode, env);
await resolveOobi(agentClient, qviOobi, 'qvi');
console.log(`OOBI Resolved: ${qviOobi}`);
```

### agent-oobi-resolve-le.ts
```typescript
import {getOrCreateClient} from "../../client/identifiers.js";
import {resolveOobi} from "../../client/oobis.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const agentPasscode = args[1];
const agentName = args[2];
const leOobi = args[3];

if (!env || !agentPasscode || !agentName || !leOobi) {
    console.error('Usage: tsx agent-oobi-resolve-le.ts <env> <agentPasscode> <agentName> <leOobi>');
    process.exit(1);
}

console.log(`Agent ${agentName} resolving LE OOBI: ${leOobi}`);
const agentClient = await getOrCreateClient(agentPasscode, env);
await resolveOobi(agentClient, leOobi, 'le');
console.log(`OOBI Resolved: ${leOobi}`);
```

### agent-oobi-resolve-verifier.ts
```typescript
import {getOrCreateClient} from "../../client/identifiers.js";
import {resolveOobi} from "../../client/oobis.js";

const args = process.argv.slice(2);
const env = args[0] as 'docker' | 'testnet';
const agentPasscode = args[1];
const agentName = args[2];

if (!env || !agentPasscode || !agentName) {
    console.error('Usage: tsx agent-oobi-resolve-verifier.ts <env> <agentPasscode> <agentName>');
    process.exit(1);
}

const verifierOobi = 'http://verifier:9723/oobi';
console.log(`Agent ${agentName} resolving Verifier (Sally) OOBI: ${verifierOobi}`);
const agentClient = await getOrCreateClient(agentPasscode, env);
await resolveOobi(agentClient, verifierOobi, 'verifier');
console.log(`OOBI Resolved: ${verifierOobi}`);
```

### agent-verify-delegation.ts
```typescript
import fs from 'fs';

const args = process.argv.slice(2);
const dataDir = args[0];
const agentName = args[1];
const oorHolderName = args[2];

if (!dataDir || !agentName || !oorHolderName) {
    console.error('Usage: tsx agent-verify-delegation.ts <dataDir> <agentName> <oorHolderName>');
    process.exit(1);
}

const agentInfoPath = `${dataDir}/${agentName}-info.json`;
if (!fs.existsSync(agentInfoPath)) {
    throw new Error(`Agent info file not found: ${agentInfoPath}`);
}
const agentInfo = JSON.parse(fs.readFileSync(agentInfoPath, 'utf-8'));

const oorHolderInfoPath = `${dataDir}/${oorHolderName}-info.json`;
if (!fs.existsSync(oorHolderInfoPath)) {
    throw new Error(`OOR Holder info file not found: ${oorHolderInfoPath}`);
}
const oorHolderInfo = JSON.parse(fs.readFileSync(oorHolderInfoPath, 'utf-8'));

console.log(`Verifying delegation for agent ${agentName}`);
console.log(`Agent AID: ${agentInfo.aid}`);
console.log(`OOR Holder AID: ${oorHolderInfo.aid}`);

const sallyUrl = 'http://verifier:9723/verify/agent-delegation';
console.log(`Calling Sally verifier at ${sallyUrl}`);

try {
    const response = await fetch(sallyUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            agent_aid: agentInfo.aid,
            oor_holder_aid: oorHolderInfo.aid,
            verify_oor_credential: true
        })
    });

    const result = await response.json();
    console.log('\n' + '='.repeat(60));
    console.log('SALLY VERIFICATION RESULT');
    console.log('='.repeat(60));
    console.log(JSON.stringify(result, null, 2));
    console.log('='.repeat(60) + '\n');

    if (result.valid) {
        console.log('✓ Agent delegation verified successfully');
        console.log(`  Agent: ${agentName} (${agentInfo.aid})`);
        console.log(`  Delegated from: ${oorHolderName} (${oorHolderInfo.aid})`);
        if (result.verification.le_lei) console.log(`  LE LEI: ${result.verification.le_lei}`);
        if (result.verification.qvi_aid) console.log(`  QVI AID: ${result.verification.qvi_aid}`);
        if (result.verification.geda_aid) console.log(`  GEDA AID: ${result.verification.geda_aid}`);
    } else {
        console.error('✗ Verification failed');
        console.error(`  Error: ${result.error}`);
        process.exit(1);
    }
} catch (error) {
    console.error('✗ Failed to call Sally verifier');
    console.error(`  Error: ${error}`);
    process.exit(1);
}
```

## Shell Script Wrappers

All shell scripts follow the same pattern:

```bash
#!/bin/bash
source ./task-scripts/tsx-script-runner.sh
run_task "path/to/typescript-file.ts" "$@"
```

Files to create:
- task-scripts/person/person-delegate-agent-create.sh
- task-scripts/person/person-approve-agent-delegation.sh
- task-scripts/agent/agent-aid-delegate-finish.sh
- task-scripts/agent/agent-oobi-resolve-qvi.sh
- task-scripts/agent/agent-oobi-resolve-le.sh
- task-scripts/agent/agent-oobi-resolve-verifier.sh
- task-scripts/agent/agent-verify-delegation.sh

Make all executable: `chmod +x task-scripts/agent/*.sh task-scripts/person/*.sh`
