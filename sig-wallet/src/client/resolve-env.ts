export type EnvType = 'docker' | 'testnet';

// Workshop environment configuration
export interface WorkshopEnv {
    preset: EnvType;
    adminUrl: string;
    bootUrl: string;
    vleiServerUrl: string;
    witnessUrls: string[];
    witnessIds: string[];
    verifierUrl: string;
    webhookUrl: string;
}

// All six demo witnesses for local
const WAN = 'BBilc4-L3tFUnfM_wJr4S4OJanAv_VmF_dJNN6vkf2Ha';
const WIL = 'BLskRTInXnMxWaGqcpSyMgo0nYbalW99cGZESrz3zapM';
const WES = 'BIKKuvBwpmDVA4Ds-EpL5bt9OqPzWPja2LigFYZN2YfX';
const WIT = 'BM35JN8XeJSEfpxopjn5jr7tAHCE5749f0OobhMLCorE';
const WUB = 'BIj15u5V11bkbtAxMA7gcNJZcax-7TgaBMLsQnMHpYHP';
const WYZ = 'BF2rZTW79z4IXocYRQnjjsOuvFUQv-ptCf8Yltd7PfsM';

// GLEIF Testnet Witnesses for testnet
const TN_WIT1 = 'BJqHtDoLT_K_XyOgr2ejBOqD9276TYMTg2EEqWKs-V0q';
const TN_WIT2 = 'BKifL6vrvwi-im9d6YvCPSJYQ1VPcpYYNBag-eXlx0MM';
const TN_WIT3 = 'BDbh2CJbkQlVSCYzVVyVTT9934yAFn2sFe8tOe-pSVUx';
const TN_WIT4 = 'BKMfsU-HtnUyh911MOUEad2NZqIMzwZpGD3U8hKOWS6M';
const TN_WIT5 = 'BOtIyoeWWJtTUnniPscdUR_C5i_2xv9CnBlKgwsMXc9f';

export function resolveEnvironment(
    input?: EnvType
): WorkshopEnv {
    const preset = input ?? process.env.WORKSHOP_ENV ?? 'docker';
    switch (preset) {
        case 'docker':
            return {
                preset: preset,
                adminUrl: `http://keria:3901`,
                bootUrl: `http://keria:3903`,
                vleiServerUrl: 'http://schema:7723',
                witnessUrls: [
                    'http://witness:5642', // wan
                    'http://witness:5643', // wil
                    'http://witness:5644', // wes
                    'http://witness:5645', // wit
                    'http://witness:5646', // wub
                    'http://witness:5647', // wyz
                ],
                witnessIds: [WAN, WIL, WES, WIT, WUB, WYZ],
                verifierUrl: 'http://verifier:9723',
                webhookUrl: 'http://resource:9923'
            };
        case 'testnet':
            return {
                preset: preset,
                adminUrl: `https://keria.testnet.gleif.org:3901`,
                bootUrl:  `https://keria.testnet.gleif.org:3903`,
                witnessUrls: [
                    'https://wit1.testnet.gleif.org:5641',
                    'https://wit2.testnet.gleif.org:5642',
                    'https://wit3.testnet.gleif.org:5643',
                    'https://wit4.testnet.gleif.org:5644',
                    'https://wit5.testnet.gleif.org:5645',
                ],
                witnessIds: [TN_WIT1, TN_WIT2, TN_WIT3, TN_WIT4, TN_WIT5],
                vleiServerUrl: 'https://schema.testnet.gleif.org:7723',
                verifierUrl: 'https://presentation-handler.testnet.gleif.org:9723',
                webhookUrl: 'https://hook.testnet.gleif.org:9923'
            };
        default:
            throw new Error(`Unknown test environment preset '${preset}'`);
    }
}
