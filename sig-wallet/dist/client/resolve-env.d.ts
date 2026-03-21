export type EnvType = 'docker' | 'testnet';
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
export declare function resolveEnvironment(input?: EnvType): WorkshopEnv;
