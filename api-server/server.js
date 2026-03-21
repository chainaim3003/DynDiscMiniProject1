import express from 'express';
import cors from 'cors';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import { fileURLToPath } from 'url';

const execAsync = promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 4000;

// Enable CORS for all origins (Windows UI can connect)
app.use(cors());
app.use(express.json());

// Helper function to run verification script
// scriptType: 'DEEP', 'DEEP-EXT', or 'DEEP-EXT-CREDENTIAL'
async function runVerification(agentName, oorHolderName, scriptType = 'DEEP') {
  try {
    console.log(`Starting ${scriptType} verification for: ${agentName}`);
    
    let scriptName;
    if (scriptType === 'DEEP-EXT-CREDENTIAL') {
      scriptName = 'test-agent-verification-DEEP-credential.sh';
    } else if (scriptType === 'DEEP-EXT') {
      scriptName = 'test-agent-verification-DEEP-EXT.sh';
    } else {
      scriptName = 'test-agent-verification-DEEP.sh';
    }
    const scriptPath = path.join(__dirname, '..', scriptName);
    // ADD --json flag to get structured output
    const command = `bash ${scriptPath} ${agentName} ${oorHolderName} docker --json`;
    
    console.log(`Executing: ${command}`);
    console.log(`Working directory: ${path.join(__dirname, '..')}`);
    
    const { stdout, stderr } = await execAsync(command, {
      cwd: path.join(__dirname, '..'),
      timeout: 120000, // 120 second timeout (2 minutes)
      maxBuffer: 1024 * 1024 * 10, // 10MB buffer
      env: { ...process.env, GEDA_PRE: '' } // Set GEDA_PRE to avoid warning
    });
    
    console.log('=== SCRIPT OUTPUT START ===');
    console.log('Verification stdout (first 500 chars):', stdout.substring(0, 500));
    console.log('Verification stdout (last 500 chars):', stdout.substring(stdout.length - 500));
    if (stderr) {
      console.log('Verification stderr:', stderr);
    }
    console.log('=== SCRIPT OUTPUT END ===');
    
    // Try to parse JSON output from verification script
    let verificationResult;
    
    try {
      // Extract JSON from output (might have Docker noise)
      const jsonMatch = stdout.match(/\{[\s\S]*"validation"[\s\S]*\}|\{[\s\S]*"success"[\s\S]*\}/);
      
      if (jsonMatch) {
        // Successfully got JSON
        verificationResult = JSON.parse(jsonMatch[0]);
        console.log('Parsed verification JSON successfully');
        
        // Ensure error field exists if not success
        if (!verificationResult.success && !verificationResult.error) {
          verificationResult.error = 'Verification failed (see output for details)';
        }
      } else {
        // Fallback: No JSON found, use old string check
        console.warn('No JSON in output, using fallback');
        const success = stdout.includes('✅ DEEP VERIFICATION PASSED') || 
                       stdout.includes('DEEP VERIFICATION PASSED') ||
                       stdout.includes('✅ DELEGATION VERIFICATION COMPLETE') ||
                       stdout.includes('Delegation is CRYPTOGRAPHICALLY VERIFIED');
        verificationResult = {
          success,
          output: stdout,
          error: success ? null : 'Verification check failed - expected success markers not found in output',
          agent: agentName,
          oorHolder: oorHolderName,
          timestamp: new Date().toISOString()
        };
      }
    } catch (parseError) {
      // Fallback: JSON parsing failed
      console.error('JSON parse failed:', parseError.message);
      const success = stdout.includes('✅ DEEP VERIFICATION PASSED') || 
                     stdout.includes('DEEP VERIFICATION PASSED') ||
                     stdout.includes('✅ DELEGATION VERIFICATION COMPLETE') ||
                     stdout.includes('Delegation is CRYPTOGRAPHICALLY VERIFIED');
      verificationResult = {
        success,
        output: stdout,
        error: success ? null : `JSON parse error: ${parseError.message}`,
        agent: agentName,
        oorHolder: oorHolderName,
        timestamp: new Date().toISOString(),
        parseError: parseError.message
      };
    }
    
    return verificationResult;
    
  } catch (error) {
    console.error(`Verification failed for ${agentName}:`, error);
    
    // Extract useful error info
    const errorMessage = error.stderr || error.message || 'Unknown error occurred';
    const outputText = error.stdout || '';
    
    // Check if it's a "file not found" type error
    let friendlyError = errorMessage;
    if (errorMessage.includes('not found') || errorMessage.includes('No such file')) {
      friendlyError = 'Required task-data files not found. Ensure 2C workflow has completed.';
    } else if (errorMessage.includes('docker') || errorMessage.includes('compose')) {
      friendlyError = 'Docker/compose error. Ensure containers are running.';
    }
    
    return {
      success: false,
      output: outputText,
      error: friendlyError,
      errorDetails: errorMessage,
      agent: agentName,
      oorHolder: oorHolderName,
      timestamp: new Date().toISOString()
    };
  }
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    timestamp: new Date().toISOString(),
    message: 'vLEI Verification API Server is running'
  });
});

// Verify Seller Agent endpoint
// LEGACY: Use /api/buyer/verify/seller instead
app.post('/api/verify/seller', async (req, res) => {
  console.log('=== [LEGACY] SELLER AGENT VERIFICATION REQUEST ===');
  console.log('WARNING: This endpoint is deprecated. Use /api/buyer/verify/seller instead.');
  
  try {
    const agentName = 'jupiterSellerAgent';
    const oorHolderName = 'Jupiter_Chief_Sales_Officer';
    
    const result = await runVerification(agentName, oorHolderName);
    result.deprecated = true;
    result.useInstead = '/api/buyer/verify/seller';
    
    const statusCode = result.success ? 200 : 400;
    console.log(`Verification result: ${result.success ? 'SUCCESS' : 'FAILED'}`);
    
    res.status(statusCode).json(result);
  } catch (error) {
    console.error('Error in seller verification endpoint:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      agent: 'jupiterSellerAgent',
      deprecated: true,
      useInstead: '/api/buyer/verify/seller',
      timestamp: new Date().toISOString()
    });
  }
});

// Verify Buyer Agent endpoint
// LEGACY: Use /api/seller/verify/buyer instead
app.post('/api/verify/buyer', async (req, res) => {
  console.log('=== [LEGACY] BUYER AGENT VERIFICATION REQUEST ===');
  console.log('WARNING: This endpoint is deprecated. Use /api/seller/verify/buyer instead.');
  
  try {
    const agentName = 'tommyBuyerAgent';
    const oorHolderName = 'Tommy_Chief_Procurement_Officer';
    
    const result = await runVerification(agentName, oorHolderName);
    result.deprecated = true;
    result.useInstead = '/api/seller/verify/buyer';
    
    const statusCode = result.success ? 200 : 400;
    console.log(`Verification result: ${result.success ? 'SUCCESS' : 'FAILED'}`);
    
    res.status(statusCode).json(result);
  } catch (error) {
    console.error('Error in buyer verification endpoint:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      agent: 'tommyBuyerAgent',
      deprecated: true,
      useInstead: '/api/seller/verify/buyer',
      timestamp: new Date().toISOString()
    });
  }
});

// ============================================
// STANDARD VERIFICATION ENDPOINTS (DEEP)
// These are for cross-organization verification using DEEP script
// URL Pattern: /api/{caller}/verify/{target}
// - /api/buyer/verify/seller: Buyer verifies seller
// - /api/seller/verify/buyer: Seller verifies buyer
// ============================================

// Buyer verifies Seller Agent (DEEP)
app.post('/api/buyer/verify/seller', async (req, res) => {
  console.log('=== BUYER -> SELLER VERIFICATION (DEEP) ===');
  console.log('Caller: Buyer | Target: Seller');
  
  try {
    const agentName = 'jupiterSellerAgent';
    const oorHolderName = 'Jupiter_Chief_Sales_Officer';
    
    // Use DEEP script for standard verification
    const result = await runVerification(agentName, oorHolderName, 'DEEP');
    
    // Add verification metadata
    result.verificationType = 'STANDARD';
    result.verificationScript = 'DEEP';
    result.caller = 'buyer';
    result.target = 'seller';
    
    const statusCode = result.success ? 200 : 400;
    console.log(`Buyer->Seller Verification (DEEP): ${result.success ? 'SUCCESS' : 'FAILED'}`);
    
    res.status(statusCode).json(result);
  } catch (error) {
    console.error('Error in buyer->seller verification:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      agent: 'jupiterSellerAgent',
      verificationType: 'STANDARD',
      verificationScript: 'DEEP',
      caller: 'buyer',
      target: 'seller',
      timestamp: new Date().toISOString()
    });
  }
});

// Seller verifies Buyer Agent (DEEP)
app.post('/api/seller/verify/buyer', async (req, res) => {
  console.log('=== SELLER -> BUYER VERIFICATION (DEEP) ===');
  console.log('Caller: Seller | Target: Buyer');
  
  try {
    const agentName = 'tommyBuyerAgent';
    const oorHolderName = 'Tommy_Chief_Procurement_Officer';
    
    // Use DEEP script for standard verification
    const result = await runVerification(agentName, oorHolderName, 'DEEP');
    
    // Add verification metadata
    result.verificationType = 'STANDARD';
    result.verificationScript = 'DEEP';
    result.caller = 'seller';
    result.target = 'buyer';
    
    const statusCode = result.success ? 200 : 400;
    console.log(`Seller->Buyer Verification (DEEP): ${result.success ? 'SUCCESS' : 'FAILED'}`);
    
    res.status(statusCode).json(result);
  } catch (error) {
    console.error('Error in seller->buyer verification:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      agent: 'tommyBuyerAgent',
      verificationType: 'STANDARD',
      verificationScript: 'DEEP',
      caller: 'seller',
      target: 'buyer',
      timestamp: new Date().toISOString()
    });
  }
});

// ============================================
// EXTERNAL VERIFICATION ENDPOINTS (DEEP-EXT)
// These are for cross-organization verification
// URL Pattern: /api/{caller}/verify/ext/{target}
// - /api/buyer/verify/ext/seller: Buyer verifies seller externally
// - /api/seller/verify/ext/buyer: Seller verifies buyer externally
// ============================================

// Buyer verifies Seller Agent externally (DEEP-EXT)
app.post('/api/buyer/verify/ext/seller', async (req, res) => {
  console.log('=== BUYER -> SELLER VERIFICATION (DEEP-EXT) ===');
  console.log('Caller: Buyer | Target: Seller');
  
  try {
    const agentName = 'jupiterSellerAgent';
    const oorHolderName = 'Jupiter_Chief_Sales_Officer';
    
    // Use DEEP-EXT script for external verification
    const result = await runVerification(agentName, oorHolderName, 'DEEP-EXT');
    
    // Add external verification metadata
    result.verificationType = 'EXTERNAL';
    result.verificationScript = 'DEEP-EXT';
    result.caller = 'buyer';
    result.target = 'seller';
    
    const statusCode = result.success ? 200 : 400;
    console.log(`Buyer->Seller Verification: ${result.success ? 'SUCCESS' : 'FAILED'}`);
    
    res.status(statusCode).json(result);
  } catch (error) {
    console.error('Error in buyer->seller verification:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      agent: 'jupiterSellerAgent',
      verificationType: 'EXTERNAL',
      verificationScript: 'DEEP-EXT',
      caller: 'buyer',
      target: 'seller',
      timestamp: new Date().toISOString()
    });
  }
});

// Seller verifies Buyer Agent externally (DEEP-EXT)
app.post('/api/seller/verify/ext/buyer', async (req, res) => {
  console.log('=== SELLER -> BUYER VERIFICATION (DEEP-EXT) ===');
  console.log('Caller: Seller | Target: Buyer');
  
  try {
    const agentName = 'tommyBuyerAgent';
    const oorHolderName = 'Tommy_Chief_Procurement_Officer';
    
    // Use DEEP-EXT script for external verification
    const result = await runVerification(agentName, oorHolderName, 'DEEP-EXT');
    
    // Add external verification metadata
    result.verificationType = 'EXTERNAL';
    result.verificationScript = 'DEEP-EXT';
    result.caller = 'seller';
    result.target = 'buyer';
    
    const statusCode = result.success ? 200 : 400;
    console.log(`Seller->Buyer Verification: ${result.success ? 'SUCCESS' : 'FAILED'}`);
    
    res.status(statusCode).json(result);
  } catch (error) {
    console.error('Error in seller->buyer verification:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      agent: 'tommyBuyerAgent',
      verificationType: 'EXTERNAL',
      verificationScript: 'DEEP-EXT',
      caller: 'seller',
      target: 'buyer',
      timestamp: new Date().toISOString()
    });
  }
});

// ============================================
// INVOICE CREDENTIAL VERIFICATION ENDPOINT
// Buyer verifies Seller's Invoice with Credential (DEEP-EXT-CREDENTIAL)
// This endpoint performs:
//   1. Deep agent delegation verification
//   2. Credential query from KERIA
//   3. Credential validation and proof verification
// ============================================
app.post('/api/buyer/verify/sellerInvoice', async (req, res) => {
  console.log('=== BUYER -> SELLER INVOICE CREDENTIAL VERIFICATION (DEEP-EXT-CREDENTIAL) ===');
  console.log('Caller: Buyer | Target: Seller Invoice Credential');
  
  try {
    const agentName = 'jupiterSellerAgent';
    const oorHolderName = 'Jupiter_Chief_Sales_Officer';
    
    // Use DEEP-EXT-CREDENTIAL script for invoice credential verification
    // This includes: agent delegation + credential query + credential validation
    const result = await runVerification(agentName, oorHolderName, 'DEEP-EXT-CREDENTIAL');
    
    // Add invoice credential verification metadata
    result.verificationType = 'INVOICE_CREDENTIAL';
    result.verificationScript = 'DEEP-EXT-CREDENTIAL';
    result.caller = 'buyer';
    result.target = 'sellerInvoice';
    result.verificationSteps = [
      'agent_delegation_verification',
      'credential_query_from_keria',
      'credential_validation_and_proof'
    ];
    
    const statusCode = result.success ? 200 : 400;
    console.log(`Buyer->SellerInvoice Credential Verification: ${result.success ? 'SUCCESS' : 'FAILED'}`);
    
    res.status(statusCode).json(result);
  } catch (error) {
    console.error('Error in buyer->sellerInvoice credential verification:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      agent: 'jupiterSellerAgent',
      verificationType: 'INVOICE_CREDENTIAL',
      verificationScript: 'DEEP-EXT-CREDENTIAL',
      caller: 'buyer',
      target: 'sellerInvoice',
      verificationSteps: [
        'agent_delegation_verification',
        'credential_query_from_keria',
        'credential_validation_and_proof'
      ],
      timestamp: new Date().toISOString()
    });
  }
});

// ============================================
// LEGACY ENDPOINTS (kept for backward compatibility)
// Will be deprecated in future versions
// ============================================

// Legacy: External Verify Seller Agent endpoint
app.post('/api/verify/ext/seller', async (req, res) => {
  console.log('=== [LEGACY] EXTERNAL SELLER VERIFICATION (DEEP-EXT) ===');
  console.log('WARNING: This endpoint is deprecated. Use /api/buyer/verify/ext/seller instead.');
  
  try {
    const agentName = 'jupiterSellerAgent';
    const oorHolderName = 'Jupiter_Chief_Sales_Officer';
    
    const result = await runVerification(agentName, oorHolderName, 'DEEP-EXT');
    
    result.verificationType = 'EXTERNAL';
    result.verificationScript = 'DEEP-EXT';
    result.calledFrom = 'buyer-verifier';
    result.deprecated = true;
    result.useInstead = '/api/buyer/verify/ext/seller';
    
    const statusCode = result.success ? 200 : 400;
    res.status(statusCode).json(result);
  } catch (error) {
    console.error('Error in legacy seller verification:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      agent: 'jupiterSellerAgent',
      deprecated: true,
      useInstead: '/api/buyer/verify/ext/seller',
      timestamp: new Date().toISOString()
    });
  }
});

// Legacy: External Verify Buyer Agent endpoint
app.post('/api/verify/ext/buyer', async (req, res) => {
  console.log('=== [LEGACY] EXTERNAL BUYER VERIFICATION (DEEP-EXT) ===');
  console.log('WARNING: This endpoint is deprecated. Use /api/seller/verify/ext/buyer instead.');
  
  try {
    const agentName = 'tommyBuyerAgent';
    const oorHolderName = 'Tommy_Chief_Procurement_Officer';
    
    const result = await runVerification(agentName, oorHolderName, 'DEEP-EXT');
    
    result.verificationType = 'EXTERNAL';
    result.verificationScript = 'DEEP-EXT';
    result.calledFrom = 'seller-verifier';
    result.deprecated = true;
    result.useInstead = '/api/seller/verify/ext/buyer';
    
    const statusCode = result.success ? 200 : 400;
    res.status(statusCode).json(result);
  } catch (error) {
    console.error('Error in legacy buyer verification:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      agent: 'tommyBuyerAgent',
      deprecated: true,
      useInstead: '/api/seller/verify/ext/buyer',
      timestamp: new Date().toISOString()
    });
  }
});

// Generic verification endpoint (for future use)
app.post('/api/verify/:agentType', async (req, res) => {
  const { agentType } = req.params;
  console.log(`=== GENERIC VERIFICATION REQUEST: ${agentType} ===`);
  
  // Map agent types to their configurations
  const agentConfigs = {
    seller: {
      agentName: 'jupiterSellerAgent',
      oorHolderName: 'Jupiter_Chief_Sales_Officer'
    },
    buyer: {
      agentName: 'tommyBuyerAgent',
      oorHolderName: 'Tommy_Chief_Procurement_Officer'
    }
  };
  
  const config = agentConfigs[agentType.toLowerCase()];
  
  if (!config) {
    return res.status(400).json({
      success: false,
      error: `Unknown agent type: ${agentType}`,
      availableTypes: Object.keys(agentConfigs)
    });
  }
  
  try {
    const result = await runVerification(config.agentName, config.oorHolderName);
    const statusCode = result.success ? 200 : 400;
    
    res.status(statusCode).json(result);
  } catch (error) {
    console.error('Error in generic verification endpoint:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      agent: config.agentName,
      timestamp: new Date().toISOString()
    });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    success: false,
    error: 'Internal server error',
    message: err.message
  });
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log('='.repeat(60));
  console.log('🚀 vLEI Verification API Server Started');
  console.log('='.repeat(60));
  console.log(`📡 Server listening on: http://0.0.0.0:${PORT}`);
  console.log(`🏥 Health check: http://localhost:${PORT}/health`);
  console.log('');
  console.log('Standard Verification (DEEP):');
  console.log(`🔐 Seller verification: POST http://localhost:${PORT}/api/verify/seller`);
  console.log(`🔐 Buyer verification: POST http://localhost:${PORT}/api/verify/buyer`);
  console.log('');
  console.log('Cross-Org Verification (DEEP-EXT) - Pattern: /api/{caller}/verify/ext/{target}');
  console.log(`🔐 Buyer->Seller: POST http://localhost:${PORT}/api/buyer/verify/ext/seller`);
  console.log(`🔐 Seller->Buyer: POST http://localhost:${PORT}/api/seller/verify/ext/buyer`);
  console.log('');
  console.log('Invoice Credential Verification (DEEP-EXT-CREDENTIAL):');
  console.log(`🔐 Buyer->SellerInvoice: POST http://localhost:${PORT}/api/buyer/verify/sellerInvoice`);
  console.log('   → Runs: agent delegation + credential query + credential validation');
  console.log('');
  console.log('Legacy Endpoints (deprecated):');
  console.log(`⚠️  /api/verify/ext/seller -> use /api/buyer/verify/ext/seller`);
  console.log(`⚠️  /api/verify/ext/buyer -> use /api/seller/verify/ext/buyer`);
  console.log('='.repeat(60));
  console.log('Ready to accept verification requests...');
  console.log('');
});
