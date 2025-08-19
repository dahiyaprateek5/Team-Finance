// ai-insights.js - AI-Powered Financial Analysis System
// Author: Financial Risk Assessment Platform
// Version: 2.0 - Enhanced AI Analysis

class AIInsights {
    constructor() {
        this.models = {
            riskAssessment: null,
            liquidationPrediction: null,
            healthScoring: null
        };
        this.analysisCache = new Map();
        this.init();
    }

    init() {
        console.log('🤖 Initializing AI Insights Engine...');
        this.loadModels();
    }

    // ===== AI MODEL LOADING =====
    async loadModels() {
        try {
            console.log('📥 Loading AI models...');
            // In a real implementation, you'd load TensorFlow.js models here
            // For now, we'll use rule-based AI algorithms
            this.models.riskAssessment = this.createRiskAssessmentModel();
            this.models.liquidationPrediction = this.createLiquidationModel();
            this.models.healthScoring = this.createHealthScoringModel();
            console.log('✅ AI models loaded successfully');
        } catch (error) {
            console.error('❌ Failed to load AI models:', error);
        }
    }

    // ===== FINANCIAL HEALTH SCORING =====
    calculateFinancialHealthScore(company) {
        try {
            const metrics = this.extractFinancialMetrics(company);
            const weights = {
                liquidity: 0.25,
                profitability: 0.30,
                efficiency: 0.20,
                leverage: 0.15,
                growth: 0.10
            };

            let score = 0;

            // Liquidity Score (0-100)
            const liquidityScore = this.calculateLiquidityScore(metrics);
            score += liquidityScore * weights.liquidity;

            // Profitability Score (0-100)
            const profitabilityScore = this.calculateProfitabilityScore(metrics);
            score += profitabilityScore * weights.profitability;

            // Efficiency Score (0-100)
            const efficiencyScore = this.calculateEfficiencyScore(metrics);
            score += efficiencyScore * weights.efficiency;

            // Leverage Score (0-100)
            const leverageScore = this.calculateLeverageScore(metrics);
            score += leverageScore * weights.leverage;

            // Growth Score (0-100)
            const growthScore = this.calculateGrowthScore(metrics);
            score += growthScore * weights.growth;

            return Math.round(Math.max(0, Math.min(100, score)));
        } catch (error) {
            console.error('Error calculating health score:', error);
            return 50; // Default neutral score
        }
    }

    calculateLiquidityScore(metrics) {
        let score = 50; // Base score

        // Current Ratio Analysis
        if (metrics.currentRatio > 2.0) score += 25;
        else if (metrics.currentRatio > 1.5) score += 15;
        else if (metrics.currentRatio > 1.0) score += 5;
        else if (metrics.currentRatio < 0.5) score -= 30;
        else score -= 15;

        // Operating Cash Flow Analysis
        if (metrics.operatingCashFlow > metrics.revenue * 0.15) score += 20;
        else if (metrics.operatingCashFlow > 0) score += 10;
        else if (metrics.operatingCashFlow < -metrics.revenue * 0.1) score -= 25;
        else score -= 10;

        // Cash Position
        const cashRatio = metrics.cash / (metrics.currentLiabilities || 1);
        if (cashRatio > 0.3) score += 5;
        else if (cashRatio < 0.1) score -= 10;

        return Math.max(0, Math.min(100, score));
    }

    calculateProfitabilityScore(metrics) {
        let score = 50; // Base score

        // Net Profit Margin
        const netMargin = metrics.netIncome / (metrics.revenue || 1);
        if (netMargin > 0.20) score += 25;
        else if (netMargin > 0.10) score += 15;
        else if (netMargin > 0.05) score += 10;
        else if (netMargin > 0) score += 5;
        else if (netMargin < -0.10) score -= 30;
        else score -= 15;

        // ROE (Return on Equity)
        const roe = metrics.netIncome / (metrics.equity || 1);
        if (roe > 0.15) score += 15;
        else if (roe > 0.10) score += 10;
        else if (roe > 0.05) score += 5;
        else if (roe < -0.05) score -= 15;

        // Gross Profit Margin
        const grossMargin = (metrics.revenue - metrics.cogs) / (metrics.revenue || 1);
        if (grossMargin > 0.40) score += 10;
        else if (grossMargin > 0.30) score += 5;
        else if (grossMargin < 0.20) score -= 10;

        return Math.max(0, Math.min(100, score));
    }

    calculateEfficiencyScore(metrics) {
        let score = 50; // Base score

        // Asset Turnover
        const assetTurnover = metrics.revenue / (metrics.totalAssets || 1);
        if (assetTurnover > 1.5) score += 20;
        else if (assetTurnover > 1.0) score += 10;
        else if (assetTurnover > 0.5) score += 5;
        else if (assetTurnover < 0.3) score -= 15;

        // Working Capital Efficiency
        const workingCapital = metrics.currentAssets - metrics.currentLiabilities;
        const wcTurnover = metrics.revenue / (workingCapital || 1);
        if (wcTurnover > 5) score += 15;
        else if (wcTurnover > 3) score += 10;
        else if (wcTurnover < 1) score -= 10;

        // Cash Conversion Cycle
        if (metrics.daysInInventory && metrics.daysInReceivables && metrics.daysInPayables) {
            const cashCycle = metrics.daysInInventory + metrics.daysInReceivables - metrics.daysInPayables;
            if (cashCycle < 30) score += 15;
            else if (cashCycle < 60) score += 10;
            else if (cashCycle > 120) score -= 15;
        }

        return Math.max(0, Math.min(100, score));
    }

    calculateLeverageScore(metrics) {
        let score = 50; // Base score

        // Debt-to-Equity Ratio
        const debtToEquity = metrics.totalDebt / (metrics.equity || 1);
        if (debtToEquity < 0.3) score += 25;
        else if (debtToEquity < 0.6) score += 15;
        else if (debtToEquity < 1.0) score += 5;
        else if (debtToEquity > 2.0) score -= 25;
        else score -= 15;

        // Interest Coverage Ratio
        const interestCoverage = metrics.ebit / (metrics.interestExpense || 1);
        if (interestCoverage > 5) score += 20;
        else if (interestCoverage > 2.5) score += 10;
        else if (interestCoverage > 1.5) score += 5;
        else if (interestCoverage < 1) score -= 20;

        // Debt Service Coverage
        const debtService = metrics.operatingCashFlow / (metrics.debtPayments || 1);
        if (debtService > 2) score += 5;
        else if (debtService < 1) score -= 15;

        return Math.max(0, Math.min(100, score));
    }

    calculateGrowthScore(metrics) {
        let score = 50; // Base score

        // Revenue Growth
        if (metrics.revenueGrowth > 0.20) score += 25;
        else if (metrics.revenueGrowth > 0.10) score += 15;
        else if (metrics.revenueGrowth > 0.05) score += 10;
        else if (metrics.revenueGrowth > 0) score += 5;
        else if (metrics.revenueGrowth < -0.10) score -= 20;
        else score -= 10;

        // Profit Growth
        if (metrics.profitGrowth > 0.15) score += 15;
        else if (metrics.profitGrowth > 0.05) score += 10;
        else if (metrics.profitGrowth < -0.15) score -= 15;

        // Market Share Trends (if available)
        if (metrics.marketShareGrowth > 0.05) score += 10;
        else if (metrics.marketShareGrowth < -0.05) score -= 10;

        return Math.max(0, Math.min(100, score));
    }

    // ===== LIQUIDATION RISK PREDICTION =====
    predictLiquidationRisk(company) {
        try {
            const features = this.extractLiquidationFeatures(company);
            const riskScore = this.calculateLiquidationRiskScore(features);
            const confidence = this.calculatePredictionConfidence(features);
            
            return {
                riskLevel: this.categorizeRiskLevel(riskScore),
                riskScore: riskScore,
                confidence: confidence,
                factors: this.identifyRiskFactors(features),
                timeline: this.estimateTimeToLiquidation(features)
            };
        } catch (error) {
            console.error('Error predicting liquidation risk:', error);
            return {
                riskLevel: 'moderate',
                riskScore: 50,
                confidence: 0.5,
                factors: [],
                timeline: 'unknown'
            };
        }
    }

    calculateLiquidationRiskScore(features) {
        let riskScore = 0;

        // Altman Z-Score components
        const workingCapital = features.currentAssets - features.currentLiabilities;
        const retainedEarnings = features.retainedEarnings || 0;
        const ebit = features.ebit || 0;
        const marketValue = features.marketValue || features.bookValue;
        const sales = features.revenue || 0;
        const totalAssets = features.totalAssets || 1;

        // Altman Z-Score calculation
        const z1 = (workingCapital / totalAssets) * 1.2;
        const z2 = (retainedEarnings / totalAssets) * 1.4;
        const z3 = (ebit / totalAssets) * 3.3;
        const z4 = (marketValue / features.totalLiabilities) * 0.6;
        const z5 = (sales / totalAssets) * 1.0;

        const zScore = z1 + z2 + z3 + z4 + z5;

        // Convert Z-Score to risk percentage
        if (zScore > 2.99) riskScore = 10; // Safe zone
        else if (zScore > 1.81) riskScore = 30; // Grey zone
        else riskScore = 80; // Distress zone

        // Additional risk factors
        if (features.operatingCashFlow < 0) riskScore += 15;
        if (features.netIncome < 0) riskScore += 10;
        if (features.currentRatio < 1) riskScore += 10;
        if (features.debtToEquity > 2) riskScore += 10;

        // Time-based factors
        const negativeEarningsYears = features.consecutiveNegativeEarnings || 0;
        riskScore += negativeEarningsYears * 5;

        return Math.min(100, Math.max(0, riskScore));
    }

    categorizeRiskLevel(riskScore) {
        if (riskScore < 20) return 'low';
        if (riskScore < 40) return 'moderate';
        if (riskScore < 70) return 'high';
        return 'critical';
    }

    identifyRiskFactors(features) {
        const factors = [];

        if (features.operatingCashFlow < 0) {
            factors.push({
                type: 'cash_flow',
                severity: 'high',
                description: 'Negative operating cash flow',
                impact: 'Immediate liquidity concerns'
            });
        }

        if (features.currentRatio < 1) {
            factors.push({
                type: 'liquidity',
                severity: 'high',
                description: 'Current ratio below 1.0',
                impact: 'Unable to meet short-term obligations'
            });
        }

        if (features.debtToEquity > 2) {
            factors.push({
                type: 'leverage',
                severity: 'medium',
                description: 'High debt-to-equity ratio',
                impact: 'Excessive financial leverage'
            });
        }

        if (features.netIncome < 0) {
            factors.push({
                type: 'profitability',
                severity: 'medium',
                description: 'Negative net income',
                impact: 'Operating at a loss'
            });
        }

        return factors;
    }

    estimateTimeToLiquidation(features) {
        const cashBurnRate = Math.abs(features.operatingCashFlow) || 0;
        const availableCash = features.cash || 0;

        if (cashBurnRate === 0 || features.operatingCashFlow >= 0) {
            return 'stable';
        }

        const monthsRemaining = availableCash / (cashBurnRate / 12);

        if (monthsRemaining < 6) return 'immediate';
        if (monthsRemaining < 12) return '6-12 months';
        if (monthsRemaining < 24) return '1-2 years';
        return '2+ years';
    }

    // ===== INDUSTRY BENCHMARKING =====
    compareToIndustry(company, industryData) {
        const companyMetrics = this.extractFinancialMetrics(company);
        const industryBenchmarks = industryData[company.industry] || industryData.default;

        return {
            profitMargin: this.calculatePercentile(companyMetrics.profitMargin, industryBenchmarks.profitMargin),
            currentRatio: this.calculatePercentile(companyMetrics.currentRatio, industryBenchmarks.currentRatio),
            debtToEquity: this.calculatePercentile(companyMetrics.debtToEquity, industryBenchmarks.debtToEquity, true),
            roa: this.calculatePercentile(companyMetrics.roa, industryBenchmarks.roa),
            roe: this.calculatePercentile(companyMetrics.roe, industryBenchmarks.roe)
        };
    }

    calculatePercentile(value, benchmarkArray, lowerIsBetter = false) {
        if (!benchmarkArray || benchmarkArray.length === 0) return 50;

        const sorted = [...benchmarkArray].sort((a, b) => a - b);
        let percentile = 0;

        for (let i = 0; i < sorted.length; i++) {
            if (value >= sorted[i]) {
                percentile = ((i + 1) / sorted.length) * 100;
            }
        }

        return lowerIsBetter ? 100 - percentile : percentile;
    }

    // ===== AI RECOMMENDATIONS =====
    generateRecommendations(company, riskAnalysis) {
        const recommendations = [];

        // Critical recommendations
        if (riskAnalysis.riskLevel === 'critical') {
            recommendations.push({
                priority: 'critical',
                category: 'liquidity',
                title: 'Immediate Cash Flow Management',
                description: 'Urgent action required to improve cash position',
                actions: [
                    'Accelerate accounts receivable collection',
                    'Negotiate payment terms with suppliers',
                    'Consider emergency funding options',
                    'Reduce non-essential expenses immediately'
                ],
                timeframe: 'immediate'
            });
        }

        // Profitability recommendations
        const metrics = this.extractFinancialMetrics(company);
        if (metrics.profitMargin < 0.05) {
            recommendations.push({
                priority: 'high',
                category: 'profitability',
                title: 'Improve Profit Margins',
                description: 'Focus on revenue optimization and cost management',
                actions: [
                    'Review pricing strategy',
                    'Analyze cost structure for reduction opportunities',
                    'Optimize product/service mix',
                    'Improve operational efficiency'
                ],
                timeframe: '3-6 months'
            });
        }

        // Working capital recommendations
        if (metrics.currentRatio < 1.5) {
            recommendations.push({
                priority: 'medium',
                category: 'working_capital',
                title: 'Strengthen Working Capital',
                description: 'Improve short-term financial position',
                actions: [
                    'Optimize inventory levels',
                    'Improve accounts receivable management',
                    'Negotiate better payment terms',
                    'Consider working capital facilities'
                ],
                timeframe: '1-3 months'
            });
        }

        return recommendations.sort((a, b) => {
            const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
            return priorityOrder[a.priority] - priorityOrder[b.priority];
        });
    }

    // ===== UTILITY FUNCTIONS =====
    extractFinancialMetrics(company) {
        return {
            revenue: parseFloat(company.revenue || 0),
            netIncome: parseFloat(company.net_income || 0),
            operatingCashFlow: parseFloat(company.operating_cash_flow || 0),
            freeCashFlow: parseFloat(company.free_cash_flow || 0),
            totalAssets: parseFloat(company.total_assets || 0),
            currentAssets: parseFloat(company.current_assets || 0),
            currentLiabilities: parseFloat(company.current_liabilities || 0),
            totalLiabilities: parseFloat(company.total_liabilities || 0),
            totalDebt: parseFloat(company.total_debt || 0),
            equity: parseFloat(company.equity || 0),
            cash: parseFloat(company.cash || 0),
            cogs: parseFloat(company.cogs || 0),
            ebit: parseFloat(company.ebit || 0),
            interestExpense: parseFloat(company.interest_expense || 0),
            
            // Calculated ratios
            currentRatio: (company.current_assets || 0) / (company.current_liabilities || 1),
            debtToEquity: (company.total_debt || 0) / (company.equity || 1),
            profitMargin: (company.net_income || 0) / (company.revenue || 1),
            roa: (company.net_income || 0) / (company.total_assets || 1),
            roe: (company.net_income || 0) / (company.equity || 1)
        };
    }

    extractLiquidationFeatures(company) {
        const metrics = this.extractFinancialMetrics(company);
        
        return {
            ...metrics,
            retainedEarnings: parseFloat(company.retained_earnings || 0),
            marketValue: parseFloat(company.market_value || company.equity || 0),
            bookValue: parseFloat(company.book_value || company.equity || 0),
            consecutiveNegativeEarnings: parseInt(company.consecutive_negative_earnings || 0),
            debtPayments: parseFloat(company.debt_payments || 0),
            daysInInventory: parseFloat(company.days_in_inventory || 0),
            daysInReceivables: parseFloat(company.days_in_receivables || 0),
            daysInPayables: parseFloat(company.days_in_payables || 0),
            revenueGrowth: parseFloat(company.revenue_growth || 0),
            profitGrowth: parseFloat(company.profit_growth || 0),
            marketShareGrowth: parseFloat(company.market_share_growth || 0)
        };
    }

    calculatePredictionConfidence(features) {
        let confidence = 0.7; // Base confidence

        // Increase confidence based on data completeness
        const requiredFields = ['revenue', 'netIncome', 'operatingCashFlow', 'totalAssets', 'currentRatio'];
        const availableFields = requiredFields.filter(field => features[field] !== undefined && features[field] !== 0);
        
        confidence += (availableFields.length / requiredFields.length) * 0.2;

        // Decrease confidence for extreme outliers
        if (features.currentRatio > 10 || features.debtToEquity > 10) {
            confidence -= 0.1;
        }

        return Math.max(0.3, Math.min(0.95, confidence));
    }

    // ===== MODEL CREATION HELPERS =====
    createRiskAssessmentModel() {
        return {
            name: 'RiskAssessment',
            version: '1.0',
            predict: (features) => this.predictLiquidationRisk(features)
        };
    }

    createLiquidationModel() {
        return {
            name: 'LiquidationPrediction',
            version: '1.0',
            predict: (features) => this.calculateLiquidationRiskScore(features)
        };
    }

    createHealthScoringModel() {
        return {
            name: 'HealthScoring',
            version: '1.0',
            predict: (company) => this.calculateFinancialHealthScore(company)
        };
    }

    // ===== PUBLIC API =====
    async analyzeCompany(company) {
        const cacheKey = `${company.id}-${company.name}`;
        
        if (this.analysisCache.has(cacheKey)) {
            console.log('📋 Using cached analysis for', company.name);
            return this.analysisCache.get(cacheKey);
        }

        console.log('🔍 Performing AI analysis for', company.name);

        const analysis = {
            company: company,
            healthScore: this.calculateFinancialHealthScore(company),
            liquidationRisk: this.predictLiquidationRisk(company),
            recommendations: [],
            timestamp: new Date().toISOString()
        };

        analysis.recommendations = this.generateRecommendations(company, analysis.liquidationRisk);

        // Cache the result
        this.analysisCache.set(cacheKey, analysis);

        return analysis;
    }

    clearCache() {
        this.analysisCache.clear();
        console.log('🗑️ Analysis cache cleared');
    }
}

// Global instance
window.aiInsights = new AIInsights();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AIInsights;
}