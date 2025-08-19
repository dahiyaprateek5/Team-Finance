// cash_flow.js - Cash Flow Statement Analysis and Processing
// Author: Financial Risk Assessment Platform
// Version: 2.0 - Enhanced Cash Flow Analysis

class CashFlowAnalyzer {
    constructor() {
        this.cashFlowStatements = new Map();
        this.analysisResults = new Map();
        this.predictions = new Map();
        this.init();
    }

    init() {
        console.log('💰 Initializing Cash Flow Analyzer...');
        this.setupEventListeners();
        this.loadCashFlowData();
    }

    setupEventListeners() {
        // Cash flow upload handlers
        const uploadBtn = document.getElementById('upload-cash-flow');
        if (uploadBtn) {
            uploadBtn.addEventListener('click', () => this.handleCashFlowUpload());
        }

        // Generate cash flow button
        const generateBtn = document.getElementById('generate-cash-flow');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => this.generateCashFlowFromBalanceSheet());
        }

        // Analysis trigger
        const analyzeBtn = document.getElementById('analyze-cash-flow');
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => this.performCashFlowAnalysis());
        }
    }

    // ===== DATA LOADING =====
    async loadCashFlowData() {
        try {
            console.log('📊 Loading cash flow data from database...');
            
            const endpoints = [
                '/api/cash-flow-statements',
                '/api/cash_flow',
                '/cash-flow/data',
                '/get_cash_flow_data'
            ];

            for (const endpoint of endpoints) {
                try {
                    const response = await fetch(endpoint);
                    if (response.ok) {
                        const data = await response.json();
                        this.processCashFlowData(data);
                        console.log('✅ Cash flow data loaded successfully');
                        return;
                    }
                } catch (error) {
                    console.log(`❌ Failed to load from ${endpoint}:`, error.message);
                    continue;
                }
            }

            // If no endpoint works, use mock data
            console.log('🎭 Using mock cash flow data...');
            this.loadMockCashFlowData();

        } catch (error) {
            console.error('❌ Error loading cash flow data:', error);
            this.loadMockCashFlowData();
        }
    }

    processCashFlowData(data) {
        let statements = [];

        // Handle different data formats
        if (Array.isArray(data)) {
            statements = data;
        } else if (data.cash_flow_statements) {
            statements = data.cash_flow_statements;
        } else if (data.statements) {
            statements = data.statements;
        } else if (data.data) {
            statements = data.data;
        }

        // Process each statement
        statements.forEach(statement => {
            const normalized = this.normalizeCashFlowStatement(statement);
            this.cashFlowStatements.set(normalized.companyId, normalized);
        });

        console.log(`📊 Processed ${statements.length} cash flow statements`);
    }

    // ===== CASH FLOW STATEMENT NORMALIZATION =====
    normalizeCashFlowStatement(rawStatement) {
        return {
            companyId: rawStatement.company_id || rawStatement.id || Math.random().toString(36).substr(2, 9),
            companyName: rawStatement.company_name || rawStatement.name || 'Unknown Company',
            reportDate: rawStatement.report_date || rawStatement.date || new Date().toISOString().split('T')[0],
            currency: rawStatement.currency || 'USD',
            period: rawStatement.period || 'annual',
            
            // Operating Activities
            operatingActivities: {
                netIncome: parseFloat(rawStatement.net_income || 0),
                adjustments: {
                    depreciation: parseFloat(rawStatement.depreciation || rawStatement.depreciation_amortization || 0),
                    amortization: parseFloat(rawStatement.amortization || 0),
                    stockCompensation: parseFloat(rawStatement.stock_compensation || rawStatement.stock_based_compensation || 0),
                    deferredTax: parseFloat(rawStatement.deferred_tax || 0),
                    gainOnSale: parseFloat(rawStatement.gain_on_sale || 0),
                    lossOnSale: parseFloat(rawStatement.loss_on_sale || 0),
                    other: parseFloat(rawStatement.other_adjustments || 0)
                },
                workingCapitalChanges: {
                    accountsReceivable: parseFloat(rawStatement.accounts_receivable_change || rawStatement.ar_change || 0),
                    inventory: parseFloat(rawStatement.inventory_change || 0),
                    accountsPayable: parseFloat(rawStatement.accounts_payable_change || rawStatement.ap_change || 0),
                    accruedExpenses: parseFloat(rawStatement.accrued_expenses_change || 0),
                    prepaidExpenses: parseFloat(rawStatement.prepaid_expenses_change || 0),
                    other: parseFloat(rawStatement.other_working_capital_changes || 0)
                },
                total: parseFloat(rawStatement.net_cash_from_operating_activities || rawStatement.operating_cash_flow || 0)
            },

            // Investing Activities
            investingActivities: {
                capitalExpenditures: parseFloat(rawStatement.capital_expenditure || rawStatement.capex || 0),
                acquisitions: parseFloat(rawStatement.acquisitions || 0),
                assetSales: parseFloat(rawStatement.asset_sales || rawStatement.proceeds_from_asset_sales || 0),
                investments: parseFloat(rawStatement.investments || 0),
                securitiesPurchases: parseFloat(rawStatement.securities_purchases || 0),
                securitiesSales: parseFloat(rawStatement.securities_sales || 0),
                other: parseFloat(rawStatement.other_investing || 0),
                total: parseFloat(rawStatement.net_cash_from_investing_activities || rawStatement.investing_cash_flow || 0)
            },

            // Financing Activities
            financingActivities: {
                debtIssuance: parseFloat(rawStatement.debt_issuance || rawStatement.new_debt || 0),
                debtRepayment: parseFloat(rawStatement.debt_repayment || rawStatement.debt_payments || 0),
                equityIssuance: parseFloat(rawStatement.equity_issuance || rawStatement.stock_issued || 0),
                shareRepurchases: parseFloat(rawStatement.share_repurchases || rawStatement.stock_repurchased || 0),
                dividends: parseFloat(rawStatement.dividends_paid || rawStatement.dividends || 0),
                other: parseFloat(rawStatement.other_financing || 0),
                total: parseFloat(rawStatement.net_cash_from_financing_activities || rawStatement.financing_cash_flow || 0)
            },

            // Summary
            netCashChange: parseFloat(rawStatement.net_cash_change || 0),
            cashBeginning: parseFloat(rawStatement.cash_beginning || rawStatement.beginning_cash || 0),
            cashEnding: parseFloat(rawStatement.cash_ending || rawStatement.ending_cash || 0),
            freeCashFlow: parseFloat(rawStatement.free_cash_flow || 0)
        };
    }

    // ===== CASH FLOW GENERATION FROM BALANCE SHEET =====
    async generateCashFlowFromBalanceSheet() {
        try {
            console.log('🔄 Generating cash flow from balance sheet...');
            
            // Get balance sheet data
            const balanceSheets = this.getBalanceSheetData();
            if (!balanceSheets || balanceSheets.length === 0) {
                throw new Error('No balance sheet data available');
            }

            // Generate cash flow for each company
            balanceSheets.forEach(bs => {
                const cashFlow = this.convertBalanceSheetToCashFlow(bs);
                this.cashFlowStatements.set(bs.companyName, cashFlow);
            });

            console.log('✅ Cash flow statements generated successfully');
            this.displayCashFlowStatements();

        } catch (error) {
            console.error('❌ Error generating cash flow:', error);
            this.showError(`Failed to generate cash flow: ${error.message}`);
        }
    }

    convertBalanceSheetToCashFlow(balanceSheet, previousBalanceSheet = null) {
        // Use indirect method to derive cash flow from balance sheet changes
        const cashFlow = {
            companyId: balanceSheet.companyName,
            companyName: balanceSheet.companyName,
            reportDate: balanceSheet.reportDate,
            currency: balanceSheet.currency,
            period: 'annual',
            
            operatingActivities: {
                netIncome: this.estimateNetIncome(balanceSheet, previousBalanceSheet),
                adjustments: this.calculateAdjustments(balanceSheet, previousBalanceSheet),
                workingCapitalChanges: this.calculateWorkingCapitalChanges(balanceSheet, previousBalanceSheet),
                total: 0
            },
            
            investingActivities: {
                capitalExpenditures: this.estimateCapEx(balanceSheet, previousBalanceSheet),
                acquisitions: 0,
                assetSales: 0,
                investments: this.calculateInvestmentChanges(balanceSheet, previousBalanceSheet),
                securitiesPurchases: 0,
                securitiesSales: 0,
                other: 0,
                total: 0
            },
            
            financingActivities: {
                debtIssuance: this.calculateDebtChanges(balanceSheet, previousBalanceSheet),
                debtRepayment: 0,
                equityIssuance: this.calculateEquityChanges(balanceSheet, previousBalanceSheet),
                shareRepurchases: 0,
                dividends: this.estimateDividends(balanceSheet, previousBalanceSheet),
                other: 0,
                total: 0
            },
            
            netCashChange: 0,
            cashBeginning: previousBalanceSheet ? previousBalanceSheet.assets.current.cash : 0,
            cashEnding: balanceSheet.assets.current.cash,
            freeCashFlow: 0
        };

        // Calculate totals
        cashFlow.operatingActivities.total = this.calculateOperatingCashFlow(cashFlow.operatingActivities);
        cashFlow.investingActivities.total = this.calculateInvestingCashFlow(cashFlow.investingActivities);
        cashFlow.financingActivities.total = this.calculateFinancingCashFlow(cashFlow.financingActivities);
        
        cashFlow.netCashChange = cashFlow.operatingActivities.total + 
                                cashFlow.investingActivities.total + 
                                cashFlow.financingActivities.total;
        
        cashFlow.freeCashFlow = cashFlow.operatingActivities.total - cashFlow.investingActivities.capitalExpenditures;

        return cashFlow;
    }

    // ===== CASH FLOW CALCULATIONS =====
    calculateOperatingCashFlow(operatingActivities) {
        const adjustmentsTotal = Object.values(operatingActivities.adjustments).reduce((sum, val) => sum + val, 0);
        const wcChangesTotal = Object.values(operatingActivities.workingCapitalChanges).reduce((sum, val) => sum + val, 0);
        
        return operatingActivities.netIncome + adjustmentsTotal - wcChangesTotal;
    }

    calculateInvestingCashFlow(investingActivities) {
        return -investingActivities.capitalExpenditures + 
               investingActivities.assetSales + 
               -investingActivities.acquisitions + 
               -investingActivities.investments +
               investingActivities.securitiesSales - 
               investingActivities.securitiesPurchases +
               investingActivities.other;
    }

    calculateFinancingCashFlow(financingActivities) {
        return financingActivities.debtIssuance - 
               financingActivities.debtRepayment + 
               financingActivities.equityIssuance - 
               financingActivities.shareRepurchases - 
               financingActivities.dividends +
               financingActivities.other;
    }

    // ===== ESTIMATION METHODS =====
    estimateNetIncome(current, previous) {
        if (!previous) return current.totalAssets * 0.05; // Assume 5% ROA
        
        const equityChange = current.totalEquity - previous.totalEquity;
        const estimatedDividends = current.totalEquity * 0.02; // Assume 2% dividend yield
        
        return equityChange + estimatedDividends;
    }

    calculateAdjustments(current, previous) {
        const ppe = current.assets.nonCurrent.ppe;
        const previousPPE = previous ? previous.assets.nonCurrent.ppe : ppe * 0.9;
        
        // Estimate depreciation as 10% of average PPE
        const avgPPE = (ppe + previousPPE) / 2;
        const depreciation = avgPPE * 0.10;
        
        return {
            depreciation: depreciation,
            amortization: current.assets.nonCurrent.intangibleAssets * 0.15,
            stockCompensation: current.totalEquity * 0.01,
            deferredTax: 0,
            gainOnSale: 0,
            lossOnSale: 0,
            other: 0
        };
    }

    calculateWorkingCapitalChanges(current, previous) {
        if (!previous) {
            return {
                accountsReceivable: 0,
                inventory: 0,
                accountsPayable: 0,
                accruedExpenses: 0,
                prepaidExpenses: 0,
                other: 0
            };
        }

        return {
            accountsReceivable: current.assets.current.accountsReceivable - previous.assets.current.accountsReceivable,
            inventory: current.assets.current.inventory - previous.assets.current.inventory,
            accountsPayable: current.liabilities.current.accountsPayable - previous.liabilities.current.accountsPayable,
            accruedExpenses: current.liabilities.current.accruedExpenses - previous.liabilities.current.accruedExpenses,
            prepaidExpenses: current.assets.current.prepaidExpenses - previous.assets.current.prepaidExpenses,
            other: 0
        };
    }

    estimateCapEx(current, previous) {
        const currentPPE = current.assets.nonCurrent.ppe;
        const previousPPE = previous ? previous.assets.nonCurrent.ppe : currentPPE * 0.9;
        const depreciation = (currentPPE + previousPPE) / 2 * 0.10;
        
        // CapEx = Change in PPE + Depreciation
        return Math.max(0, (currentPPE - previousPPE) + depreciation);
    }

    calculateInvestmentChanges(current, previous) {
        if (!previous) return 0;
        
        return current.assets.nonCurrent.investments - previous.assets.nonCurrent.investments;
    }

    calculateDebtChanges(current, previous) {
        if (!previous) return 0;
        
        const currentDebt = current.liabilities.current.shortTermDebt + current.liabilities.nonCurrent.longTermDebt;
        const previousDebt = previous.liabilities.current.shortTermDebt + previous.liabilities.nonCurrent.longTermDebt;
        
        return Math.max(0, currentDebt - previousDebt);
    }

    calculateEquityChanges(current, previous) {
        if (!previous) return 0;
        
        return Math.max(0, current.equity.shareCapital - previous.equity.shareCapital);
    }

    estimateDividends(current, previous) {
        if (!previous) return 0;
        
        // Estimate dividends based on retained earnings change and estimated net income
        const reChange = current.equity.retainedEarnings - previous.equity.retainedEarnings;
        const estimatedNI = this.estimateNetIncome(current, previous);
        
        return Math.max(0, estimatedNI - reChange);
    }

    // ===== CASH FLOW ANALYSIS =====
    analyzeCashFlow(companyId) {
        const cashFlow = this.cashFlowStatements.get(companyId);
        if (!cashFlow) {
            console.error('❌ No cash flow statement found for:', companyId);
            return null;
        }

        const analysis = {
            companyId: companyId,
            companyName: cashFlow.companyName,
            reportDate: cashFlow.reportDate,
            
            // Cash Flow Quality Metrics
            operatingCashFlowQuality: this.assessOperatingCashFlowQuality(cashFlow),
            freeCashFlowAnalysis: this.analyzeFreeCashFlow(cashFlow),
            liquidityAnalysis: this.assessLiquidity(cashFlow),
            
            // Trend Analysis
            cashFlowTrends: this.analyzeCashFlowTrends(cashFlow),
            
            // Risk Assessment
            cashFlowRisks: this.identifyCashFlowRisks(cashFlow),
            
            // Predictions
            predictions: this.generateCashFlowPredictions(cashFlow),
            
            // Overall Scores
            cashFlowScore: 0,
            riskLevel: 'moderate',
            
            // Recommendations
            recommendations: []
        };

        // Calculate overall score
        analysis.cashFlowScore = this.calculateOverallCashFlowScore(analysis);
        analysis.riskLevel = this.determineCashFlowRiskLevel(analysis.cashFlowScore);
        analysis.recommendations = this.generateCashFlowRecommendations(analysis);

        this.analysisResults.set(companyId, analysis);
        return analysis;
    }

    assessOperatingCashFlowQuality(cashFlow) {
        const ocf = cashFlow.operatingActivities.total;
        const netIncome = cashFlow.operatingActivities.netIncome;
        
        return {
            operatingCashFlow: ocf,
            netIncome: netIncome,
            ocfToNetIncomeRatio: netIncome !== 0 ? ocf / netIncome : 0,
            isPositive: ocf > 0,
            isGrowingFasterThanIncome: ocf > netIncome,
            quality: this.rateOCFQuality(ocf, netIncome),
            workingCapitalImpact: this.analyzeWorkingCapitalImpact(cashFlow.operatingActivities.workingCapitalChanges)
        };
    }

    rateOCFQuality(ocf, netIncome) {
        if (ocf <= 0) return 'poor';
        
        const ratio = netIncome !== 0 ? ocf / netIncome : 1;
        
        if (ratio >= 1.2) return 'excellent';
        if (ratio >= 1.0) return 'good';
        if (ratio >= 0.8) return 'fair';
        return 'poor';
    }

    analyzeWorkingCapitalImpact(wcChanges) {
        const totalWCChange = Object.values(wcChanges).reduce((sum, val) => sum + val, 0);
        
        return {
            totalChange: totalWCChange,
            isDrain: totalWCChange > 0,
            isSource: totalWCChange < 0,
            breakdown: wcChanges,
            impact: this.categorizeWCImpact(totalWCChange)
        };
    }

    categorizeWCImpact(wcChange) {
        const absChange = Math.abs(wcChange);
        
        if (absChange < 10000) return 'minimal';
        if (absChange < 50000) return 'moderate';
        if (absChange < 100000) return 'significant';
        return 'major';
    }

    analyzeFreeCashFlow(cashFlow) {
        const fcf = cashFlow.freeCashFlow;
        const ocf = cashFlow.operatingActivities.total;
        const capex = cashFlow.investingActivities.capitalExpenditures;
        
        return {
            freeCashFlow: fcf,
            operatingCashFlow: ocf,
            capitalExpenditures: capex,
            fcfMargin: ocf !== 0 ? fcf / ocf : 0,
            isPositive: fcf > 0,
            isSustainable: this.assessFCFSustainability(fcf, ocf, capex),
            trend: this.analyzeFCFTrend(cashFlow),
            quality: this.rateFCFQuality(fcf, ocf)
        };
    }

    assessFCFSustainability(fcf, ocf, capex) {
        if (fcf <= 0) return false;
        if (ocf <= 0) return false;
        
        // Check if capex is reasonable (typically 3-15% of revenue)
        const capexRatio = capex / ocf;
        return capexRatio < 0.5; // Sustainable if capex is less than 50% of OCF
    }

    rateFCFQuality(fcf, ocf) {
        if (fcf <= 0) return 'poor';
        
        const fcfRatio = fcf / ocf;
        
        if (fcfRatio >= 0.7) return 'excellent';
        if (fcfRatio >= 0.5) return 'good';
        if (fcfRatio >= 0.3) return 'fair';
        return 'poor';
    }

    assessLiquidity(cashFlow) {
        const ocf = cashFlow.operatingActivities.total;
        const cashPosition = cashFlow.cashEnding;
        const netCashChange = cashFlow.netCashChange;
        
        return {
            operatingCashFlow: ocf,
            cashPosition: cashPosition,
            netCashChange: netCashChange,
            liquidityScore: this.calculateLiquidityScore(ocf, cashPosition, netCashChange),
            burnRate: this.calculateBurnRate(cashFlow),
            runway: this.calculateCashRunway(cashFlow)
        };
    }

    calculateLiquidityScore(ocf, cash, netChange) {
        let score = 50; // Base score
        
        // Operating cash flow impact
        if (ocf > 0) score += 30;
        else score -= 40;
        
        // Cash position impact
        if (cash > 100000) score += 10;
        else if (cash < 10000) score -= 20;
        
        // Net cash change impact
        if (netChange > 0) score += 10;
        else if (netChange < -50000) score -= 20;
        
        return Math.max(0, Math.min(100, score));
    }

    calculateBurnRate(cashFlow) {
        const ocf = cashFlow.operatingActivities.total;
        
        if (ocf >= 0) return 0; // No burn if OCF is positive
        
        return Math.abs(ocf) / 12; // Monthly burn rate
    }

    calculateCashRunway(cashFlow) {
        const cash = cashFlow.cashEnding;
        const burnRate = this.calculateBurnRate(cashFlow);
        
        if (burnRate === 0) return 'infinite';
        
        const monthsRemaining = cash / burnRate;
        
        if (monthsRemaining < 6) return 'critical';
        if (monthsRemaining < 12) return 'concerning';
        if (monthsRemaining < 24) return 'moderate';
        return 'healthy';
    }

    // ===== TREND ANALYSIS =====
    analyzeCashFlowTrends(cashFlow) {
        // Note: This would need historical data for proper trend analysis
        // For now, providing structure for future implementation
        
        return {
            operatingTrend: 'stable',
            investingTrend: 'stable',
            financingTrend: 'stable',
            freeCashFlowTrend: 'stable',
            seasonality: this.detectSeasonality(cashFlow),
            cyclicality: this.detectCyclicality(cashFlow)
        };
    }

    detectSeasonality(cashFlow) {
        // Placeholder for seasonality detection
        return {
            hasSeasonality: false,
            peakQuarter: null,
            troughQuarter: null,
            seasonalityStrength: 0
        };
    }

    detectCyclicality(cashFlow) {
        // Placeholder for cyclicality detection
        return {
            hasCyclicality: false,
            cycleLength: null,
            currentPhase: 'unknown'
        };
    }

    // ===== RISK ASSESSMENT =====
    identifyCashFlowRisks(cashFlow) {
        const risks = [];
        
        // Operating cash flow risks
        if (cashFlow.operatingActivities.total <= 0) {
            risks.push({
                type: 'operating_cash_flow',
                severity: 'high',
                description: 'Negative operating cash flow',
                impact: 'Inability to generate cash from core operations'
            });
        }
        
        // Free cash flow risks
        if (cashFlow.freeCashFlow <= 0) {
            risks.push({
                type: 'free_cash_flow',
                severity: 'medium',
                description: 'Negative free cash flow',
                impact: 'No cash available for growth or debt service'
            });
        }
        
        // High capital expenditure risk
        const capexRatio = cashFlow.investingActivities.capitalExpenditures / Math.abs(cashFlow.operatingActivities.total);
        if (capexRatio > 0.8) {
            risks.push({
                type: 'capital_intensity',
                severity: 'medium',
                description: 'High capital expenditure relative to operating cash flow',
                impact: 'Limited financial flexibility'
            });
        }
        
        // Cash position risk
        if (cashFlow.cashEnding < 50000) {
            risks.push({
                type: 'cash_position',
                severity: 'high',
                description: 'Low cash position',
                impact: 'Potential liquidity crisis'
            });
        }
        
        return risks;
    }

    // ===== PREDICTIONS =====
    generateCashFlowPredictions(cashFlow) {
        return {
            nextQuarterOCF: this.predictNextQuarterOCF(cashFlow),
            yearEndCash: this.predictYearEndCash(cashFlow),
            liquidationRisk: this.assessLiquidationRisk(cashFlow),
            growthCapacity: this.assessGrowthCapacity(cashFlow)
        };
    }

    predictNextQuarterOCF(cashFlow) {
        // Simple prediction based on current OCF
        const currentOCF = cashFlow.operatingActivities.total;
        const seasonalFactor = 1.0; // Would be calculated from historical data
        
        return {
            predicted: currentOCF * seasonalFactor * 0.25, // Quarterly
            confidence: 0.7,
            range: {
                low: currentOCF * 0.2,
                high: currentOCF * 0.3
            }
        };
    }

    predictYearEndCash(cashFlow) {
        const currentCash = cashFlow.cashEnding;
        const burnRate = this.calculateBurnRate(cashFlow);
        
        return {
            predicted: currentCash - (burnRate * 12),
            confidence: 0.6,
            scenarios: {
                conservative: currentCash - (burnRate * 15),
                optimistic: currentCash - (burnRate * 9)
            }
        };
    }

    assessLiquidationRisk(cashFlow) {
        const runway = this.calculateCashRunway(cashFlow);
        const ocf = cashFlow.operatingActivities.total;
        
        let riskLevel = 'low';
        let probability = 0.05;
        
        if (runway === 'critical' || ocf < -100000) {
            riskLevel = 'high';
            probability = 0.4;
        } else if (runway === 'concerning' || ocf < 0) {
            riskLevel = 'medium';
            probability = 0.15;
        }
        
        return {
            riskLevel: riskLevel,
            probability: probability,
            timeToLiquidation: runway,
            confidence: 0.75
        };
    }

    assessGrowthCapacity(cashFlow) {
        const fcf = cashFlow.freeCashFlow;
        const ocf = cashFlow.operatingActivities.total;
        
        let capacity = 'limited';
        
        if (fcf > 100000 && ocf > 0) {
            capacity = 'high';
        } else if (fcf > 0 && ocf > 0) {
            capacity = 'moderate';
        }
        
        return {
            capacity: capacity,
            availableCash: Math.max(0, fcf),
            constraints: fcf <= 0 ? ['Negative free cash flow'] : [],
            opportunities: fcf > 0 ? ['Organic growth', 'Acquisitions', 'Debt reduction'] : []
        };
    }

    // ===== SCORING AND RECOMMENDATIONS =====
    calculateOverallCashFlowScore(analysis) {
        let score = 50; // Base score
        
        // Operating cash flow quality (40% weight)
        const ocfQuality = analysis.operatingCashFlowQuality;
        if (ocfQuality.quality === 'excellent') score += 20;
        else if (ocfQuality.quality === 'good') score += 15;
        else if (ocfQuality.quality === 'fair') score += 5;
        else score -= 15;
        
        // Free cash flow analysis (30% weight)
        const fcfAnalysis = analysis.freeCashFlowAnalysis;
        if (fcfAnalysis.quality === 'excellent') score += 15;
        else if (fcfAnalysis.quality === 'good') score += 10;
        else if (fcfAnalysis.quality === 'fair') score += 5;
        else score -= 10;
        
        // Liquidity analysis (20% weight)
        const liquidityScore = analysis.liquidityAnalysis.liquidityScore;
        score += (liquidityScore - 50) * 0.2;
        
        // Risk factors (10% weight)
        const riskCount = analysis.cashFlowRisks.length;
        score -= riskCount * 5;
        
        return Math.max(0, Math.min(100, Math.round(score)));
    }

    determineCashFlowRiskLevel(score) {
        if (score >= 80) return 'low';
        if (score >= 60) return 'moderate';
        if (score >= 40) return 'high';
        return 'critical';
    }

    generateCashFlowRecommendations(analysis) {
        const recommendations = [];
        
        // Operating cash flow recommendations
        if (analysis.operatingCashFlowQuality.quality === 'poor') {
            recommendations.push({
                priority: 'high',
                category: 'operations',
                title: 'Improve Operating Cash Flow',
                description: 'Focus on improving cash generation from core operations',
                actions: [
                    'Accelerate customer collections',
                    'Optimize inventory levels',
                    'Negotiate better payment terms with suppliers',
                    'Review pricing strategy'
                ]
            });
        }
        
        // Free cash flow recommendations
        if (analysis.freeCashFlowAnalysis.freeCashFlow <= 0) {
            recommendations.push({
                priority: 'high',
                category: 'investment',
                title: 'Optimize Capital Allocation',
                description: 'Improve free cash flow generation',
                actions: [
                    'Review capital expenditure priorities',
                    'Consider asset optimization',
                    'Evaluate investment returns',
                    'Delay non-essential investments'
                ]
            });
        }
        
        // Liquidity recommendations
        if (analysis.liquidityAnalysis.liquidityScore < 40) {
            recommendations.push({
                priority: 'critical',
                category: 'liquidity',
                title: 'Address Liquidity Concerns',
                description: 'Immediate action required to improve cash position',
                actions: [
                    'Secure additional funding',
                    'Implement cash conservation measures',
                    'Accelerate collections',
                    'Consider asset sales'
                ]
            });
        }
        
        return recommendations;
    }

    // ===== DISPLAY FUNCTIONS =====
    displayCashFlowStatements() {
        const container = document.getElementById('cash-flow-display');
        if (!container) return;
        
        const statements = Array.from(this.cashFlowStatements.values());
        const html = statements.map(statement => this.generateCashFlowHTML(statement)).join('');
        
        container.innerHTML = html;
    }

    generateCashFlowHTML(cashFlow) {
        return `
            <div class="cash-flow-statement card mb-4">
                <div class="card-header">
                    <h5>${cashFlow.companyName} - Cash Flow Statement</h5>
                    <p class="mb-0">Period ending: ${cashFlow.reportDate}</p>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-4">
                            <h6 class="text-success">Operating Activities</h6>
                            <p>Net Income: ${this.formatCurrency(cashFlow.operatingActivities.netIncome)}</p>
                            <p>Depreciation: ${this.formatCurrency(cashFlow.operatingActivities.adjustments.depreciation)}</p>
                            <p>Working Capital Changes: ${this.formatCurrency(-Object.values(cashFlow.operatingActivities.workingCapitalChanges).reduce((sum, val) => sum + val, 0))}</p>
                            <p><strong>Net Operating Cash Flow: ${this.formatCurrency(cashFlow.operatingActivities.total)}</strong></p>
                        </div>
                        <div class="col-md-4">
                            <h6 class="text-warning">Investing Activities</h6>
                            <p>Capital Expenditures: ${this.formatCurrency(-cashFlow.investingActivities.capitalExpenditures)}</p>
                            <p>Acquisitions: ${this.formatCurrency(-cashFlow.investingActivities.acquisitions)}</p>
                            <p>Asset Sales: ${this.formatCurrency(cashFlow.investingActivities.assetSales)}</p>
                            <p><strong>Net Investing Cash Flow: ${this.formatCurrency(cashFlow.investingActivities.total)}</strong></p>
                        </div>
                        <div class="col-md-4">
                            <h6 class="text-danger">Financing Activities</h6>
                            <p>Debt Issuance: ${this.formatCurrency(cashFlow.financingActivities.debtIssuance)}</p>
                            <p>Debt Repayment: ${this.formatCurrency(-cashFlow.financingActivities.debtRepayment)}</p>
                            <p>Dividends Paid: ${this.formatCurrency(-cashFlow.financingActivities.dividends)}</p>
                            <p><strong>Net Financing Cash Flow: ${this.formatCurrency(cashFlow.financingActivities.total)}</strong></p>
                        </div>
                    </div>
                    <hr>
                    <div class="row">
                        <div class="col-md-6">
                            <p><strong>Net Change in Cash: ${this.formatCurrency(cashFlow.netCashChange)}</strong></p>
                            <p>Cash Beginning: ${this.formatCurrency(cashFlow.cashBeginning)}</p>
                            <p>Cash Ending: ${this.formatCurrency(cashFlow.cashEnding)}</p>
                        </div>
                        <div class="col-md-6">
                            <p><strong>Free Cash Flow: ${this.formatCurrency(cashFlow.freeCashFlow)}</strong></p>
                            <button class="btn btn-primary btn-sm" onclick="cashFlowAnalyzer.analyzeAndDisplay('${cashFlow.companyId}')">
                                Analyze Cash Flow
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    analyzeAndDisplay(companyId) {
        const analysis = this.analyzeCashFlow(companyId);
        if (analysis) {
            this.displayAnalysis(analysis);
        }
    }

    displayAnalysis(analysis) {
        const container = document.getElementById('cash-flow-analysis');
        if (!container) return;

        const html = `
            <div class="cash-flow-analysis">
                <h4>Cash Flow Analysis - ${analysis.companyName}</h4>
                
                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <h6>Overall Assessment</h6>
                            </div>
                            <div class="card-body">
                                <p><strong>Cash Flow Score:</strong> 
                                    <span class="badge bg-${this.getScoreBadgeColor(analysis.cashFlowScore)}">${analysis.cashFlowScore}/100</span>
                                </p>
                                <p><strong>Risk Level:</strong> 
                                    <span class="badge bg-${this.getRiskBadgeColor(analysis.riskLevel)}">${analysis.riskLevel.toUpperCase()}</span>
                                </p>
                                <p><strong>Analysis Date:</strong> ${new Date().toLocaleDateString()}</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-success text-white">
                                <h6>Key Metrics</h6>
                            </div>
                            <div class="card-body">
                                <p><strong>Operating Cash Flow:</strong> ${this.formatCurrency(analysis.operatingCashFlowQuality.operatingCashFlow)}</p>
                                <p><strong>Free Cash Flow:</strong> ${this.formatCurrency(analysis.freeCashFlowAnalysis.freeCashFlow)}</p>
                                <p><strong>Cash Runway:</strong> ${analysis.liquidityAnalysis.runway}</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="row mb-4">
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header bg-info text-white">
                                <h6>Operating Cash Flow Quality</h6>
                            </div>
                            <div class="card-body">
                                <p><strong>Quality Rating:</strong> 
                                    <span class="badge bg-${this.getQualityBadgeColor(analysis.operatingCashFlowQuality.quality)}">${analysis.operatingCashFlowQuality.quality.toUpperCase()}</span>
                                </p>
                                <p><strong>OCF/Net Income:</strong> ${analysis.operatingCashFlowQuality.ocfToNetIncomeRatio.toFixed(2)}</p>
                                <p><strong>Working Capital Impact:</strong> ${analysis.operatingCashFlowQuality.workingCapitalImpact.impact}</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header bg-warning text-dark">
                                <h6>Free Cash Flow Analysis</h6>
                            </div>
                            <div class="card-body">
                                <p><strong>FCF Quality:</strong> 
                                    <span class="badge bg-${this.getQualityBadgeColor(analysis.freeCashFlowAnalysis.quality)}">${analysis.freeCashFlowAnalysis.quality.toUpperCase()}</span>
                                </p>
                                <p><strong>FCF Margin:</strong> ${(analysis.freeCashFlowAnalysis.fcfMargin * 100).toFixed(1)}%</p>
                                <p><strong>Sustainable:</strong> ${analysis.freeCashFlowAnalysis.isSustainable ? 'Yes' : 'No'}</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header bg-secondary text-white">
                                <h6>Liquidity Assessment</h6>
                            </div>
                            <div class="card-body">
                                <p><strong>Liquidity Score:</strong> ${analysis.liquidityAnalysis.liquidityScore}/100</p>
                                <p><strong>Monthly Burn Rate:</strong> ${this.formatCurrency(analysis.liquidityAnalysis.burnRate)}</p>
                                <p><strong>Cash Runway:</strong> ${analysis.liquidityAnalysis.runway}</p>
                            </div>
                        </div>
                    </div>
                </div>

                ${analysis.cashFlowRisks.length > 0 ? `
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header bg-danger text-white">
                                <h6>Identified Risks</h6>
                            </div>
                            <div class="card-body">
                                ${analysis.cashFlowRisks.map(risk => `
                                    <div class="alert alert-${this.getSeverityBadgeColor(risk.severity)} d-flex align-items-center mb-2">
                                        <i class="fas fa-exclamation-triangle me-2"></i>
                                        <div>
                                            <strong>${risk.description}</strong><br>
                                            <small>${risk.impact}</small>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                </div>
                ` : ''}

                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <h6>Predictions</h6>
                            </div>
                            <div class="card-body">
                                <p><strong>Next Quarter OCF:</strong> ${this.formatCurrency(analysis.predictions.nextQuarterOCF.predicted)}</p>
                                <p><strong>Year-end Cash:</strong> ${this.formatCurrency(analysis.predictions.yearEndCash.predicted)}</p>
                                <p><strong>Liquidation Risk:</strong> 
                                    <span class="badge bg-${this.getRiskBadgeColor(analysis.predictions.liquidationRisk.riskLevel)}">${analysis.predictions.liquidationRisk.riskLevel.toUpperCase()}</span>
                                </p>
                                <p><strong>Growth Capacity:</strong> ${analysis.predictions.growthCapacity.capacity}</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-success text-white">
                                <h6>Key Performance Indicators</h6>
                            </div>
                            <div class="card-body">
                                <div class="progress mb-2">
                                    <div class="progress-bar bg-primary" style="width: ${analysis.cashFlowScore}%">
                                        Cash Flow Score: ${analysis.cashFlowScore}%
                                    </div>
                                </div>
                                <div class="progress mb-2">
                                    <div class="progress-bar bg-info" style="width: ${analysis.liquidityAnalysis.liquidityScore}%">
                                        Liquidity Score: ${analysis.liquidityAnalysis.liquidityScore}%
                                    </div>
                                </div>
                                <div class="progress mb-2">
                                    <div class="progress-bar bg-success" style="width: ${this.getQualityPercentage(analysis.operatingCashFlowQuality.quality)}%">
                                        OCF Quality: ${analysis.operatingCashFlowQuality.quality}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                ${analysis.recommendations.length > 0 ? `
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header bg-warning text-dark">
                                <h6>Recommendations</h6>
                            </div>
                            <div class="card-body">
                                ${analysis.recommendations.map(rec => `
                                    <div class="recommendation mb-3">
                                        <h6 class="text-${this.getPriorityColor(rec.priority)}">
                                            <i class="fas fa-lightbulb me-2"></i>${rec.title}
                                        </h6>
                                        <p>${rec.description}</p>
                                        <ul class="list-unstyled">
                                            ${rec.actions.map(action => `<li><i class="fas fa-arrow-right me-2"></i>${action}</li>`).join('')}
                                        </ul>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                </div>
                ` : ''}
            </div>
        `;

        container.innerHTML = html;
    }

    // ===== UTILITY FUNCTIONS =====
    formatCurrency(amount, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency,
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(amount);
    }

    getScoreBadgeColor(score) {
        if (score >= 80) return 'success';
        if (score >= 60) return 'info';
        if (score >= 40) return 'warning';
        return 'danger';
    }

    getRiskBadgeColor(riskLevel) {
        const colors = {
            low: 'success',
            moderate: 'warning',
            high: 'danger',
            critical: 'dark'
        };
        return colors[riskLevel] || 'secondary';
    }

    getQualityBadgeColor(quality) {
        const colors = {
            excellent: 'success',
            good: 'info',
            fair: 'warning',
            poor: 'danger'
        };
        return colors[quality] || 'secondary';
    }

    getSeverityBadgeColor(severity) {
        const colors = {
            low: 'info',
            medium: 'warning',
            high: 'danger',
            critical: 'dark'
        };
        return colors[severity] || 'secondary';
    }

    getPriorityColor(priority) {
        const colors = {
            critical: 'danger',
            high: 'warning',
            medium: 'info',
            low: 'secondary'
        };
        return colors[priority] || 'secondary';
    }

    getQualityPercentage(quality) {
        const percentages = {
            excellent: 90,
            good: 70,
            fair: 50,
            poor: 25
        };
        return percentages[quality] || 0;
    }

    showError(message) {
        console.error('❌ Cash Flow Error:', message);
        const errorContainer = document.getElementById('error-container');
        if (errorContainer) {
            errorContainer.innerHTML = `
                <div class="alert alert-danger alert-dismissible fade show" role="alert">
                    <strong>Error:</strong> ${message}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            `;
        }
    }

    // ===== MOCK DATA =====
    loadMockCashFlowData() {
        const mockStatements = [
            {
                company_id: 'MOCK001',
                company_name: 'TechCorp Inc.',
                net_income: 150000,
                depreciation: 25000,
                accounts_receivable_change: -10000,
                inventory_change: 5000,
                accounts_payable_change: 8000,
                net_cash_from_operating_activities: 178000,
                capital_expenditure: 50000,
                net_cash_from_investing_activities: -50000,
                debt_issuance: 30000,
                dividends_paid: 20000,
                net_cash_from_financing_activities: 10000,
                free_cash_flow: 128000,
                cash_beginning: 100000,
                cash_ending: 238000
            },
            {
                company_id: 'MOCK002',
                company_name: 'Manufacturing Co.',
                net_income: -25000,
                depreciation: 40000,
                accounts_receivable_change: 15000,
                inventory_change: -20000,
                accounts_payable_change: -10000,
                net_cash_from_operating_activities: 20000,
                capital_expenditure: 75000,
                net_cash_from_investing_activities: -75000,
                debt_issuance: 50000,
                dividends_paid: 0,
                net_cash_from_financing_activities: 50000,
                free_cash_flow: -55000,
                cash_beginning: 80000,
                cash_ending: 75000
            }
        ];

        mockStatements.forEach(statement => {
            const normalized = this.normalizeCashFlowStatement(statement);
            this.cashFlowStatements.set(normalized.companyId, normalized);
        });

        console.log('🎭 Mock cash flow data loaded');
    }

    // ===== DATA HELPERS =====
    getBalanceSheetData() {
        // This would integrate with the balance sheet analyzer
        if (window.balanceSheetAnalyzer && window.balanceSheetAnalyzer.balanceSheets) {
            return Array.from(window.balanceSheetAnalyzer.balanceSheets.values());
        }
        return [];
    }

    // ===== PUBLIC API =====
    getCashFlowStatement(companyId) {
        return this.cashFlowStatements.get(companyId);
    }

    getAnalysis(companyId) {
        return this.analysisResults.get(companyId);
    }

    getAllCompanies() {
        return Array.from(this.cashFlowStatements.keys());
    }

    exportCashFlowData(companyId, format = 'json') {
        const cashFlow = this.cashFlowStatements.get(companyId);
        if (!cashFlow) return null;

        if (format === 'json') {
            return JSON.stringify(cashFlow, null, 2);
        } else if (format === 'csv') {
            return this.convertCashFlowToCSV(cashFlow);
        }

        return null;
    }

    convertCashFlowToCSV(cashFlow) {
        const rows = [
            ['Category', 'Item', 'Amount'],
            ['Operating', 'Net Income', cashFlow.operatingActivities.netIncome],
            ['Operating', 'Depreciation', cashFlow.operatingActivities.adjustments.depreciation],
            ['Operating', 'Working Capital Changes', -Object.values(cashFlow.operatingActivities.workingCapitalChanges).reduce((sum, val) => sum + val, 0)],
            ['Operating', 'Net Operating Cash Flow', cashFlow.operatingActivities.total],
            ['Investing', 'Capital Expenditures', -cashFlow.investingActivities.capitalExpenditures],
            ['Investing', 'Net Investing Cash Flow', cashFlow.investingActivities.total],
            ['Financing', 'Debt Issuance', cashFlow.financingActivities.debtIssuance],
            ['Financing', 'Dividends Paid', -cashFlow.financingActivities.dividends],
            ['Financing', 'Net Financing Cash Flow', cashFlow.financingActivities.total],
            ['Summary', 'Free Cash Flow', cashFlow.freeCashFlow],
            ['Summary', 'Net Cash Change', cashFlow.netCashChange],
            ['Summary', 'Cash Ending', cashFlow.cashEnding]
        ];

        return rows.map(row => row.join(',')).join('\n');
    }

    // ===== PERFORMANCE METHODS =====
    performComprehensiveAnalysis(companyId) {
        console.log(`🔍 Performing comprehensive cash flow analysis for ${companyId}...`);
        
        const analysis = this.analyzeCashFlow(companyId);
        if (!analysis) return null;

        // Enhanced analysis with AI insights
        if (window.aiInsights) {
            const cashFlow = this.cashFlowStatements.get(companyId);
            const aiAnalysis = window.aiInsights.analyzeCompany({
                id: companyId,
                name: cashFlow.companyName,
                operating_cash_flow: cashFlow.operatingActivities.total,
                free_cash_flow: cashFlow.freeCashFlow,
                net_income: cashFlow.operatingActivities.netIncome
            });

            // Merge AI insights with cash flow analysis
            analysis.aiInsights = aiAnalysis;
        }

        return analysis;
    }

    generateCashFlowReport(companyId) {
        const analysis = this.performComprehensiveAnalysis(companyId);
        if (!analysis) return null;

        return {
            executiveSummary: this.generateExecutiveSummary(analysis),
            detailedAnalysis: analysis,
            recommendations: analysis.recommendations,
            charts: this.generateChartData(analysis),
            generatedAt: new Date().toISOString()
        };
    }

    generateExecutiveSummary(analysis) {
        const cashFlow = this.cashFlowStatements.get(analysis.companyId);
        
        return {
            companyName: analysis.companyName,
            overallAssessment: analysis.riskLevel,
            keyFindings: [
                `Operating cash flow: ${this.formatCurrency(analysis.operatingCashFlowQuality.operatingCashFlow)}`,
                `Free cash flow: ${this.formatCurrency(analysis.freeCashFlowAnalysis.freeCashFlow)}`,
                `Cash flow score: ${analysis.cashFlowScore}/100`,
                `Risk level: ${analysis.riskLevel}`
            ],
            criticalIssues: analysis.cashFlowRisks.filter(r => r.severity === 'high' || r.severity === 'critical'),
            immediateActions: analysis.recommendations.filter(r => r.priority === 'critical' || r.priority === 'high')
        };
    }

    generateChartData(analysis) {
        const cashFlow = this.cashFlowStatements.get(analysis.companyId);
        
        return {
            cashFlowWaterfall: {
                categories: ['Operating', 'Investing', 'Financing', 'Net Change'],
                values: [
                    cashFlow.operatingActivities.total,
                    cashFlow.investingActivities.total,
                    cashFlow.financingActivities.total,
                    cashFlow.netCashChange
                ]
            },
            scoreGauge: {
                value: analysis.cashFlowScore,
                max: 100,
                thresholds: [40, 60, 80]
            },
            riskBreakdown: {
                labels: analysis.cashFlowRisks.map(r => r.type),
                severities: analysis.cashFlowRisks.map(r => r.severity)
            }
        };
    }
}

// Global instance
window.cashFlowAnalyzer = new CashFlowAnalyzer();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CashFlowAnalyzer;
}