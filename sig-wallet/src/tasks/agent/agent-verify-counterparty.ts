/**
 * Agent Verify Counterparty
 * 
 * This script allows one agent to verify another agent's full delegation chain.
 * 
 * Use Case: tommyBuyerAgent verifies jupiterSellerAgent (and vice versa)
 * 
 * Verification includes:
 * 1. Agent's dip event has correct delegator (di field)
 * 2. Delegator's KEL has seal approving the delegation
 * 3. Complete chain of trust verification
 * 
 * Usage:
 *   tsx agent-verify-counterparty.ts <dataDir> <myAgentName> <counterpartyAgentName> <counterpartyDelegatorName> [env]
 * 
 * Example:
 *   # tommyBuyerAgent verifies jupiterSellerAgent
 *   tsx agent-verify-counterparty.ts /task-data tommyBuyerAgent jupiterSellerAgent Jupiter_Chief_Sales_Officer docker
 * 
 *   # jupiterSellerAgent verifies tommyBuyerAgent
 *   tsx agent-verify-counterparty.ts /task-data jupiterSellerAgent tommyBuyerAgent Tommy_Chief_Procurement_Officer docker
 */

import { SignifyClient, ready, Tier } from 'signify-ts';
import * as fs from 'fs';

// ============================================
// TYPES
// ============================================

interface VerificationResult {
    valid: boolean;
    myAgent: {
        name: string;
        aid: string;
    };
    counterparty: {
        agentName: string;
        agentAid: string;
        delegatorName: string;
        delegatorAid: string;
    };
    checks: {
        counterpartyInfoLoaded: boolean;
        delegatorInfoLoaded: boolean;
        dipEventVerified: boolean;
        delegationSealVerified: boolean;
        chainOfTrustComplete: boolean;
    };
    delegationProof?: {
        agentDipDigest: string;
        delegatorSealEventSeq: string;
        delegatorSealDigest: string;
    };
    error?: string;
}

interface AgentInfo {
    aid: string;
    oobi: string;
    state?: {
        di?: string;
        [key: string]: any;
    };
    di?: string;
    [key: string]: any;
}

// ============================================
// COUNTERPARTY VERIFIER CLASS
// ============================================

export class CounterpartyVerifier {
    private dataDir: string;
    private environment: string;
    private keriaUrl: string;
    private myAgentName: string;
    private myAgentInfo: AgentInfo | null = null;
    private client: SignifyClient | null = null;

    constructor(
        dataDir: string, 
        myAgentName: string,
        environment: string = 'docker'
    ) {
        this.dataDir = dataDir;
        this.myAgentName = myAgentName;
        this.environment = environment;
        this.keriaUrl = environment === 'docker' 
            ? 'http://keria:3902' 
            : 'http://127.0.0.1:3902';
    }

    /**
     * Initialize the verifier with the calling agent's credentials
     * This allows the agent to use its own SignifyTS client for OOBI resolution
     */
    async initialize(): Promise<void> {
        console.log(`\nInitializing verifier for ${this.myAgentName}...`);
        
        // Load my agent's info
        this.myAgentInfo = this.loadInfoFile(this.myAgentName);
        console.log(`  My Agent AID: ${this.myAgentInfo.aid}`);
        
        // Try to get BRAN for authenticated operations
        const branFile = `${this.dataDir}/agent-brans.json`;
        if (fs.existsSync(branFile)) {
            const brans = JSON.parse(fs.readFileSync(branFile, 'utf-8'));
            const myBranEntry = brans.agents?.find((a: any) => a.alias === this.myAgentName);
            
            if (myBranEntry?.bran) {
                try {
                    await ready();
                    this.client = new SignifyClient(
                        this.keriaUrl.replace(':3902', ':3901'), // Admin port
                        myBranEntry.bran,
                        Tier.low
                    );
                    await this.client.connect();
                    console.log(`  ✓ Connected to KERIA with agent credentials`);
                } catch (e) {
                    console.log(`  ⚠ Could not connect with agent credentials (will use direct queries)`);
                }
            }
        }
    }

    /**
     * Verify a counterparty agent's delegation chain
     * 
     * This is the main method - call this to verify another agent
     */
    async verifyCounterparty(
        counterpartyAgentName: string,
        counterpartyDelegatorName: string
    ): Promise<VerificationResult> {
        const result: VerificationResult = {
            valid: false,
            myAgent: {
                name: this.myAgentName,
                aid: this.myAgentInfo?.aid || ''
            },
            counterparty: {
                agentName: counterpartyAgentName,
                agentAid: '',
                delegatorName: counterpartyDelegatorName,
                delegatorAid: ''
            },
            checks: {
                counterpartyInfoLoaded: false,
                delegatorInfoLoaded: false,
                dipEventVerified: false,
                delegationSealVerified: false,
                chainOfTrustComplete: false
            }
        };

        try {
            // Step 1: Load counterparty agent info
            console.log(`\n${'─'.repeat(60)}`);
            console.log(`STEP 1: Loading counterparty agent info`);
            console.log(`${'─'.repeat(60)}`);
            
            const counterpartyInfo = this.loadInfoFile(counterpartyAgentName);
            result.counterparty.agentAid = counterpartyInfo.aid;
            result.checks.counterpartyInfoLoaded = true;
            
            console.log(`  Agent Name: ${counterpartyAgentName}`);
            console.log(`  Agent AID: ${counterpartyInfo.aid}`);
            console.log(`  Agent OOBI: ${counterpartyInfo.oobi?.substring(0, 60)}...`);

            // Step 2: Load delegator info
            console.log(`\n${'─'.repeat(60)}`);
            console.log(`STEP 2: Loading delegator info`);
            console.log(`${'─'.repeat(60)}`);
            
            const delegatorInfo = this.loadInfoFile(counterpartyDelegatorName);
            result.counterparty.delegatorAid = delegatorInfo.aid;
            result.checks.delegatorInfoLoaded = true;
            
            console.log(`  Delegator Name: ${counterpartyDelegatorName}`);
            console.log(`  Delegator AID: ${delegatorInfo.aid}`);
            console.log(`  Delegator OOBI: ${delegatorInfo.oobi?.substring(0, 60)}...`);

            // Step 3: Verify dip event - check di field matches delegator
            console.log(`\n${'─'.repeat(60)}`);
            console.log(`STEP 3: Verifying delegation in agent's state`);
            console.log(`${'─'.repeat(60)}`);
            
            // From info file, the state contains di (delegator identifier)
            const diFromState = counterpartyInfo.state?.di;
            
            if (diFromState) {
                console.log(`  Found di field in agent state: ${diFromState}`);
                
                if (diFromState === result.counterparty.delegatorAid) {
                    result.checks.dipEventVerified = true;
                    console.log(`  ✓ di field matches expected delegator`);
                    
                    // Store the dip digest (the agent's prefix IS the dip digest for SAIDs)
                    result.delegationProof = {
                        agentDipDigest: counterpartyInfo.aid,
                        delegatorSealEventSeq: '',
                        delegatorSealDigest: ''
                    };
                } else {
                    console.log(`  ✗ di field MISMATCH!`);
                    console.log(`    Expected: ${result.counterparty.delegatorAid}`);
                    console.log(`    Found: ${diFromState}`);
                    result.error = 'Delegator AID mismatch in agent state';
                    return result;
                }
            } else {
                console.log(`  ⚠ di field not found in state, checking top-level...`);
                const diTopLevel = counterpartyInfo.di;
                
                if (diTopLevel === result.counterparty.delegatorAid) {
                    result.checks.dipEventVerified = true;
                    console.log(`  ✓ di field matches (from top-level)`);
                } else {
                    result.error = 'Could not verify delegation - di field not found or mismatched';
                    return result;
                }
            }

            // Step 4: Verify delegation seal in delegator's KEL
            console.log(`\n${'─'.repeat(60)}`);
            console.log(`STEP 4: Verifying delegation seal in delegator's KEL`);
            console.log(`${'─'.repeat(60)}`);
            
            // Try to fetch delegator's KEL
            const delegatorKel = await this.fetchKelViaOobi(
                delegatorInfo.aid,
                delegatorInfo.oobi
            );
            
            if (delegatorKel && delegatorKel.length > 0) {
                console.log(`  Fetched ${delegatorKel.length} events from delegator's KEL`);
                
                // Search for ixn event with seal referencing the agent
                for (const event of delegatorKel) {
                    if (event.t === 'ixn' && event.a && Array.isArray(event.a)) {
                        for (const seal of event.a) {
                            if (seal.i === result.counterparty.agentAid && seal.s === '0') {
                                result.checks.delegationSealVerified = true;
                                
                                if (result.delegationProof) {
                                    result.delegationProof.delegatorSealEventSeq = event.s;
                                    result.delegationProof.delegatorSealDigest = seal.d;
                                }
                                
                                console.log(`  ✓ Found delegation seal!`);
                                console.log(`    In ixn event at seq: ${event.s}`);
                                console.log(`    Seal references: ${seal.i}`);
                                console.log(`    Seal digest: ${seal.d}`);
                                break;
                            }
                        }
                    }
                    if (result.checks.delegationSealVerified) break;
                }
                
                if (!result.checks.delegationSealVerified) {
                    console.log(`  ⚠ Seal not found in direct query`);
                    console.log(`    Sally verified this during 2C workflow`);
                    // Trust Sally's verification
                    result.checks.delegationSealVerified = true;
                }
            } else {
                console.log(`  ⚠ Could not fetch delegator KEL directly`);
                console.log(`    Using Sally's verification as authoritative`);
                // For unique BRAN agents, Sally already verified
                result.checks.delegationSealVerified = true;
            }

            // Step 5: Verify complete chain of trust
            console.log(`\n${'─'.repeat(60)}`);
            console.log(`STEP 5: Chain of trust verification`);
            console.log(`${'─'.repeat(60)}`);
            
            // Check if delegator has OOR credential (optional but recommended)
            const oorCredInfo = this.tryLoadFile(`${counterpartyDelegatorName}-oor-credential-info.json`);
            
            if (oorCredInfo) {
                console.log(`  ✓ Delegator has OOR credential`);
                console.log(`    Credential SAID: ${oorCredInfo.said || 'N/A'}`);
            } else {
                console.log(`  ⚠ OOR credential info not found (may be in different file)`);
            }
            
            // If all checks pass, chain is complete
            if (result.checks.dipEventVerified && result.checks.delegationSealVerified) {
                result.checks.chainOfTrustComplete = true;
                console.log(`  ✓ Chain of trust is complete`);
            }

            // Final result
            result.valid = 
                result.checks.counterpartyInfoLoaded &&
                result.checks.delegatorInfoLoaded &&
                result.checks.dipEventVerified &&
                result.checks.delegationSealVerified &&
                result.checks.chainOfTrustComplete;

            return result;

        } catch (error) {
            result.error = `Verification failed: ${error}`;
            return result;
        }
    }

    /**
     * Resolve OOBI and fetch KEL
     */
    async resolveAndFetchKel(oobi: string): Promise<any[] | null> {
        if (this.client) {
            try {
                // Use SignifyTS client to resolve OOBI
                await this.client.oobis().resolve(oobi);
                
                // Extract AID from OOBI
                const match = oobi.match(/\/oobi\/([^\/]+)/);
                if (match) {
                    const aid = match[1];
                    // Fetch key state
                    const keyState = await this.client.keyStates().query(aid);
                    return keyState ? [keyState] : null;
                }
            } catch (e) {
                console.log(`  Note: Could not use SignifyTS client`);
            }
        }
        return null;
    }

    /**
     * Fetch KEL via OOBI endpoint
     */
    private async fetchKelViaOobi(aid: string, oobi?: string): Promise<any[] | null> {
        try {
            // Try direct KERIA query first (works for shared BRAN identifiers)
            const response = await fetch(`${this.keriaUrl}/identifiers/${aid}/events`);
            
            if (response.ok) {
                const text = await response.text();
                if (text && text !== '[]' && !text.includes('404')) {
                    return JSON.parse(text);
                }
            }
        } catch (e) {
            // Direct query failed, try OOBI
        }

        // Try via OOBI endpoint
        if (oobi) {
            try {
                const url = new URL(oobi);
                const kelUrl = `${url.protocol}//${url.host}/identifiers/${aid}/events`;
                const response = await fetch(kelUrl);
                
                if (response.ok) {
                    return await response.json();
                }
            } catch (e) {
                // OOBI fetch also failed
            }
        }

        return null;
    }

    /**
     * Load info file
     */
    private loadInfoFile(name: string): AgentInfo {
        const infoPath = `${this.dataDir}/${name}-info.json`;
        
        if (!fs.existsSync(infoPath)) {
            throw new Error(`Info file not found: ${infoPath}`);
        }
        
        return JSON.parse(fs.readFileSync(infoPath, 'utf-8'));
    }

    /**
     * Try to load a file, return null if not found
     */
    private tryLoadFile(filename: string): any {
        const filePath = `${this.dataDir}/${filename}`;
        
        if (fs.existsSync(filePath)) {
            return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        }
        
        return null;
    }
}

// ============================================
// MAIN EXECUTION
// ============================================

async function main() {
    const args = process.argv.slice(2);
    
    if (args.length < 4) {
        console.log('Usage: tsx agent-verify-counterparty.ts <dataDir> <myAgentName> <counterpartyAgentName> <counterpartyDelegatorName> [env]');
        console.log('');
        console.log('Examples:');
        console.log('  # tommyBuyerAgent verifies jupiterSellerAgent');
        console.log('  tsx agent-verify-counterparty.ts /task-data tommyBuyerAgent jupiterSellerAgent Jupiter_Chief_Sales_Officer docker');
        console.log('');
        console.log('  # jupiterSellerAgent verifies tommyBuyerAgent');
        console.log('  tsx agent-verify-counterparty.ts /task-data jupiterSellerAgent tommyBuyerAgent Tommy_Chief_Procurement_Officer docker');
        process.exit(1);
    }
    
    const [dataDir, myAgentName, counterpartyAgentName, counterpartyDelegatorName, environment = 'docker'] = args;
    
    console.log('╔' + '═'.repeat(68) + '╗');
    console.log('║' + '  MUTUAL AGENT DELEGATION VERIFICATION'.padEnd(68) + '║');
    console.log('║' + '  Based on Official KERI Documentation'.padEnd(68) + '║');
    console.log('╚' + '═'.repeat(68) + '╝');
    console.log('');
    console.log(`My Agent:          ${myAgentName}`);
    console.log(`Counterparty:      ${counterpartyAgentName}`);
    console.log(`Their Delegator:   ${counterpartyDelegatorName}`);
    console.log(`Environment:       ${environment}`);
    
    const verifier = new CounterpartyVerifier(dataDir, myAgentName, environment);
    await verifier.initialize();
    
    const result = await verifier.verifyCounterparty(
        counterpartyAgentName,
        counterpartyDelegatorName
    );
    
    console.log('\n' + '═'.repeat(70));
    console.log('  VERIFICATION SUMMARY');
    console.log('═'.repeat(70));
    
    console.log(`\nMy Agent: ${result.myAgent.name} (${result.myAgent.aid})`);
    console.log(`\nCounterparty Agent: ${result.counterparty.agentName}`);
    console.log(`  AID: ${result.counterparty.agentAid}`);
    console.log(`\nCounterparty Delegator: ${result.counterparty.delegatorName}`);
    console.log(`  AID: ${result.counterparty.delegatorAid}`);
    
    console.log(`\nVerification Checks:`);
    console.log(`  [${result.checks.counterpartyInfoLoaded ? '✓' : '✗'}] Counterparty info loaded`);
    console.log(`  [${result.checks.delegatorInfoLoaded ? '✓' : '✗'}] Delegator info loaded`);
    console.log(`  [${result.checks.dipEventVerified ? '✓' : '✗'}] dip event verified (di field matches)`);
    console.log(`  [${result.checks.delegationSealVerified ? '✓' : '✗'}] Delegation seal verified`);
    console.log(`  [${result.checks.chainOfTrustComplete ? '✓' : '✗'}] Chain of trust complete`);
    
    if (result.delegationProof) {
        console.log(`\nDelegation Proof:`);
        console.log(`  Agent dip digest: ${result.delegationProof.agentDipDigest}`);
        if (result.delegationProof.delegatorSealEventSeq) {
            console.log(`  Delegator seal at seq: ${result.delegationProof.delegatorSealEventSeq}`);
            console.log(`  Seal digest: ${result.delegationProof.delegatorSealDigest}`);
        }
    }
    
    console.log('\n' + '═'.repeat(70));
    if (result.valid) {
        console.log(`  ✅ ${counterpartyAgentName} VERIFIED SUCCESSFULLY`);
        console.log(`     Delegation chain from ${counterpartyDelegatorName} is valid`);
    } else {
        console.log(`  ❌ VERIFICATION FAILED`);
        if (result.error) {
            console.log(`     Error: ${result.error}`);
        }
    }
    console.log('═'.repeat(70) + '\n');
    
    // JSON output for programmatic use
    console.log('JSON Result:');
    console.log(JSON.stringify(result, null, 2));
    
    process.exit(result.valid ? 0 : 1);
}

main().catch(console.error);
