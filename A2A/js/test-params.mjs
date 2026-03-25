import algosdk from 'algosdk';

const algodToken = '';
const algodServer = 'https://testnet-api.algonode.cloud';
const algodPort = '';

const algodClient = new algosdk.Algodv2(algodToken, algodServer, algodPort);

(async () => {
  try {
    console.log('Fetching transaction parameters...');
    const suggestedParams = await algodClient.getTransactionParams().do();
    
    console.log('\nAll keys in suggestedParams:');
    console.log(Object.keys(suggestedParams));
    
    console.log('\nSuggestedParams type:');
    console.log(typeof suggestedParams);
    console.log(suggestedParams.constructor.name);
    
    console.log('\nChecking values:');
    for (const key of Object.keys(suggestedParams)) {
      const value = suggestedParams[key];
      console.log(`  ${key}: ${value} (type: ${typeof value})`);
    }
    
  } catch (error) {
    console.error('Error:', error);
  }
})();
