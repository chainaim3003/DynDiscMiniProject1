// ================= ACTUS RISK SERVICE + SIMULATION CLIENT =================
// Mirrors the 4-step flow in DynDisc2-Multi-AR-AP-Cash-time-1.json (DD-3 collection)
// but driven programmatically from negotiation outcomes.

export interface ActusConfig {
    riskServiceUrl: string;   // e.g. http://34.203.247.32:8082
    actusUrl:       string;   // e.g. http://34.203.247.32:8083
}

export interface ActusDDParams {
    contractId:           string;   // = invoiceId
    negotiationId:        string;
    invoiceDate:          string;   // ISO date
    dueDate:              string;   // ISO date
    settlementDate:       string;   // ISO date — buyer's chosen early payment date
    notionalAmount:       number;   // full invoice total (incl. tax)
    maxDiscountRate:      number;   // from computeSafeDDRate()
    hurdleRateAnnualized: number;   // default 0.075 (matches DD-3 baseline)
    sellerRevenue:        number;   // used as flat reference index value
}

export interface ActusSubmitResult {
    success:          boolean;
    contractId:       string;
    scenarioId:       string;
    referenceIndexId: string;
    earlySettleId:    string;
    events?:          ActusEvent[];
    error?:           string;
}

export interface ActusEvent {
    type:         string;
    time:         string;
    payoff:       number;
    nominalValue: number;
}

export class ActusClient {
    private riskServiceUrl: string;
    private actusUrl:       string;

    constructor(config?: Partial<ActusConfig>) {
        this.riskServiceUrl = config?.riskServiceUrl
            ?? process.env.ACTUS_RISK_URL
            ?? "http://34.203.247.32:8082";
        this.actusUrl = config?.actusUrl
            ?? process.env.ACTUS_URL
            ?? "http://34.203.247.32:8083";
    }

    // ── Main entry point ──────────────────────────────────────────────────────
    async submitDDContract(params: ActusDDParams): Promise<ActusSubmitResult> {
        // Derive unique IDs from negotiationId suffix (safe for URL paths)
        const suffix      = params.negotiationId.replace(/[^a-zA-Z0-9]/g, "_").slice(-10);
        const refId       = `SELLER_CASH_${suffix}`;
        const earlyId     = `early_settle_${suffix}`;
        const scenarioId  = `dd_scenario_${suffix}`;

        const iso = (d: string): string =>
            d.includes("T") ? d : `${d}T00:00:00`;

        try {
            // ── Step 1: Add Reference Index (seller cash = total revenue) ─────
            await this.post(`${this.riskServiceUrl}/addReferenceIndex`, {
                riskFactorID:     refId,
                marketObjectCode: "SELLER_CASH",
                base: 1,
                data: this.buildFlatSeries(params.invoiceDate, params.dueDate, params.sellerRevenue),
            });

            // ── Step 2: Add Early Settlement Model (LINEAR) ───────────────────
            await this.post(`${this.riskServiceUrl}/addEarlySettlementModel`, {
                riskFactorId:           earlyId,
                invoiceDate:            iso(params.invoiceDate),
                dueDate:                iso(params.dueDate),
                settlementDate:         iso(params.settlementDate),
                notionalAmount:         params.notionalAmount,
                maxDiscountRate:        params.maxDiscountRate,
                discountFunctionType:   "LINEAR",
                decayLambda:            0.0,
                powerAlpha:             1.0,
                stepDiscountSchedule:   null,
                hurdleRateAnnualized:   params.hurdleRateAnnualized,
                buyerCashMOC:           "SELLER_CASH",
                customDiscountMOC:      null,
                monitoringEventTimes:   [iso(params.settlementDate)],
            });

            // ── Step 3: Add Scenario ──────────────────────────────────────────
            await this.post(`${this.riskServiceUrl}/addScenario`, {
                scenarioID: scenarioId,
                riskFactorDescriptors: [
                    { riskFactorID: refId,   riskFactorType: "ReferenceIndex"  },
                    { riskFactorID: earlyId, riskFactorType: "EarlySettlement" },
                ],
            });

            // ── Step 4: Simulate PAM contract (seller = RPA = receivable) ─────
            const simResult = await this.post(`${this.actusUrl}/rf2/scenarioSimulation`, {
                contracts: [{
                    calendar:                         "NC",
                    businessDayConvention:            "SCF",
                    contractType:                     "PAM",
                    statusDate:                       iso(params.invoiceDate),
                    contractRole:                     "RPA",
                    contractID:                       params.contractId,
                    cycleAnchorDateOfInterestPayment: iso(params.dueDate),
                    cycleOfInterestPayment:           "P1ML0",
                    nominalInterestRate:              0,
                    dayCountConvention:               "AA",
                    currency:                         "INR",
                    contractDealDate:                 iso(params.invoiceDate),
                    initialExchangeDate:              iso(params.invoiceDate),
                    maturityDate:                     iso(params.dueDate),
                    notionalPrincipal:                params.notionalAmount,
                    premiumDiscountAtIED:             0,
                    counterpartyID:                   "BUYER",
                    creatorID:                        "SELLER",
                    discountingModels:                [earlyId],
                }],
                scenarioDescriptor: { scenarioID: scenarioId, scenarioType: "scenario" },
                simulateTo:         iso(params.dueDate),
                monitoringTimes:    [],
            });

            const events: ActusEvent[] = Array.isArray(simResult)
                ? (simResult[0]?.events ?? [])
                : [];

            return {
                success:          true,
                contractId:       params.contractId,
                scenarioId,
                referenceIndexId: refId,
                earlySettleId:    earlyId,
                events,
            };

        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : String(err);
            return {
                success:          false,
                contractId:       params.contractId,
                scenarioId,
                referenceIndexId: refId,
                earlySettleId:    earlyId,
                error:            msg,
            };
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private async post(url: string, body: object): Promise<unknown> {
        const response = await fetch(url, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(body),
        });
        if (!response.ok) {
            const text = await response.text().catch(() => "(no body)");
            throw new Error(`ACTUS ${url} → HTTP ${response.status}: ${text}`);
        }
        const text = await response.text();
        try { return JSON.parse(text); } catch { return text; }
    }

    /** Build a flat daily time series from `from` to `to` (inclusive). */
    private buildFlatSeries(
        from:  string,
        to:    string,
        value: number
    ): { time: string; value: number }[] {
        const series: { time: string; value: number }[] = [];
        const cur = new Date(from);
        const end = new Date(to);
        while (cur <= end) {
            series.push({
                time: `${cur.toISOString().split("T")[0]}T00:00:00`,
                value,
            });
            cur.setDate(cur.getDate() + 1);
        }
        return series;
    }
}
