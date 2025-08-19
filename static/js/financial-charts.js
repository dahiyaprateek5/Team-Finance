// financial-charts.js - Specialized Financial Charts Engine
// Author: Financial Risk Assessment Platform
// Version: 2.0 - Database-Driven Financial Visualizations

class FinancialCharts {
    constructor() {
        this.charts = new Map();
        this.financialData = new Map();
        this.defaultConfig = {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        };
        this.colors = {
            profit: '#10b981',
            loss: '#ef4444',
            revenue: '#3b82f6',
            expenses: '#f59e0b',
            assets: '#06b6d4',
            liabilities: '#8b5cf6',
            equity: '#10b981',
            cash: '#059669'
        };
        this.init();
    }

    init() {
        console.log('📈 Initializing Financial Charts Engine...');
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Chart interaction handlers
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('financial-chart-refresh')) {
                const chartId = e.target.dataset.chartId;
                const companyId = e.target.dataset.companyId;
                this.refreshFinancialChart(chartId, companyId);
            }
        });

        // Period selector handlers
        document.addEventListener('change', (e) => {
            if (e.target.classList.contains('chart-period-selector')) {
                const chartId = e.target.dataset.chartId;
                const period = e.target.value;
                this.updateChartPeriod(chartId, period);
            }
        });
    }

    // ===== DATABASE INTEGRATION =====
    async fetchFinancialData(endpoint, params = {}) {
        try {
            const url = new URL(endpoint, window.location.origin);
            Object.keys(params).forEach(key => {
                if (params[key] !== null && params[key] !== undefined) {
                    url.searchParams.append(key, params[key]);
                }
            });
            
            console.log(`📊 Fetching financial data from: ${url}`);
            
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log(`✅ Financial data fetched successfully from ${endpoint}`);
            return data;
            
        } catch (error) {
            console.error(`❌ Error fetching financial data from ${endpoint}:`, error);
            throw error;
        }
    }

    // ===== INCOME STATEMENT VISUALIZATION =====
    async createIncomeStatementChart(containerId, companyId, period = 'annual') {
        try {
            const data = await this.fetchFinancialData('/api/financial-charts/income-statement', {
                company_id: companyId,
                period: period
            });

            const chartData = {
                labels: data.periods || [],
                datasets: [
                    {
                        label: 'Revenue',
                        data: data.revenue || [],
                        backgroundColor: this.colors.revenue,
                        borderColor: this.colors.revenue,
                        borderWidth: 2,
                        type: 'line',
                        fill: false,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Gross Profit',
                        data: data.gross_profit || [],
                        backgroundColor: this.hexToRgba(this.colors.profit, 0.7),
                        borderColor: this.colors.profit,
                        borderWidth: 1,
                        type: 'bar',
                        yAxisID: 'y'
                    },
                    {
                        label: 'Net Income',
                        data: data.net_income || [],
                        backgroundColor: data.net_income?.map(val => 
                            val >= 0 ? this.hexToRgba(this.colors.profit, 0.8) : this.hexToRgba(this.colors.loss, 0.8)
                        ),
                        borderColor: data.net_income?.map(val => 
                            val >= 0 ? this.colors.profit : this.colors.loss
                        ),
                        borderWidth: 2,
                        type: 'bar',
                        yAxisID: 'y'
                    }
                ]
            };

            const config = {
                type: 'bar',
                data: chartData,
                options: {
                    ...this.defaultConfig,
                    plugins: {
                        title: {
                            display: true,
                            text: `${data.company_name} - Income Statement Trends`,
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            position: 'top'
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: (context) => {
                                    return `${context.dataset.label}: ${this.formatCurrency(context.parsed.y)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Period'
                            }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Amount ($)'
                            },
                            ticks: {
                                callback: (value) => this.formatCurrency(value)
                            }
                        }
                    },
                    interaction: {
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }
                }
            };

            return this.renderChart(containerId, config, data);

        } catch (error) {
            console.error('❌ Error creating income statement chart:', error);
            this.showChartError(containerId, 'Failed to load income statement data');
        }
    }

    // ===== BALANCE SHEET VISUALIZATION =====
    async createBalanceSheetChart(containerId, companyId, period = 'annual') {
        try {
            const data = await this.fetchFinancialData('/api/financial-charts/balance-sheet', {
                company_id: companyId,
                period: period
            });

            const chartData = {
                labels: data.periods || [],
                datasets: [
                    {
                        label: 'Total Assets',
                        data: data.total_assets || [],
                        backgroundColor: this.hexToRgba(this.colors.assets, 0.7),
                        borderColor: this.colors.assets,
                        borderWidth: 2
                    },
                    {
                        label: 'Total Liabilities',
                        data: data.total_liabilities || [],
                        backgroundColor: this.hexToRgba(this.colors.liabilities, 0.7),
                        borderColor: this.colors.liabilities,
                        borderWidth: 2
                    },
                    {
                        label: 'Total Equity',
                        data: data.total_equity || [],
                        backgroundColor: this.hexToRgba(this.colors.equity, 0.7),
                        borderColor: this.colors.equity,
                        borderWidth: 2
                    }
                ]
            };

            const config = {
                type: 'line',
                data: chartData,
                options: {
                    ...this.defaultConfig,
                    plugins: {
                        title: {
                            display: true,
                            text: `${data.company_name} - Balance Sheet Trends`,
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            position: 'top'
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Period'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Amount ($)'
                            },
                            ticks: {
                                callback: (value) => this.formatCurrency(value)
                            }
                        }
                    },
                    elements: {
                        line: {
                            tension: 0.1
                        },
                        point: {
                            radius: 5,
                            hoverRadius: 8
                        }
                    }
                }
            };

            return this.renderChart(containerId, config, data);

        } catch (error) {
            console.error('❌ Error creating balance sheet chart:', error);
            this.showChartError(containerId, 'Failed to load balance sheet data');
        }
    }

    // ===== CASH FLOW ANALYSIS CHART =====
    async createCashFlowChart(containerId, companyId, period = 'annual') {
        try {
            const data = await this.fetchFinancialData('/api/financial-charts/cash-flow', {
                company_id: companyId,
                period: period
            });

            const chartData = {
                labels: data.periods || [],
                datasets: [
                    {
                        label: 'Operating Cash Flow',
                        data: data.operating_cash_flow || [],
                        backgroundColor: this.colors.profit,
                        borderColor: this.colors.profit,
                        borderWidth: 2,
                        fill: false
                    },
                    {
                        label: 'Investing Cash Flow',
                        data: data.investing_cash_flow || [],
                        backgroundColor: this.colors.expenses,
                        borderColor: this.colors.expenses,
                        borderWidth: 2,
                        fill: false
                    },
                    {
                        label: 'Financing Cash Flow',
                        data: data.financing_cash_flow || [],
                        backgroundColor: this.colors.liabilities,
                        borderColor: this.colors.liabilities,
                        borderWidth: 2,
                        fill: false
                    },
                    {
                        label: 'Free Cash Flow',
                        data: data.free_cash_flow || [],
                        backgroundColor: this.colors.cash,
                        borderColor: this.colors.cash,
                        borderWidth: 3,
                        borderDash: [5, 5],
                        fill: false
                    }
                ]
            };

            const config = {
                type: 'line',
                data: chartData,
                options: {
                    ...this.defaultConfig,
                    plugins: {
                        title: {
                            display: true,
                            text: `${data.company_name} - Cash Flow Analysis`,
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            position: 'top'
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Period'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Cash Flow ($)'
                            },
                            ticks: {
                                callback: (value) => this.formatCurrency(value)
                            }
                        }
                    },
                    elements: {
                        line: {
                            tension: 0.1
                        },
                        point: {
                            radius: 4,
                            hoverRadius: 8
                        }
                    }
                }
            };

            return this.renderChart(containerId, config, data);

        } catch (error) {
            console.error('❌ Error creating cash flow chart:', error);
            this.showChartError(containerId, 'Failed to load cash flow data');
        }
    }

    // ===== FINANCIAL RATIOS CHART =====
    async createFinancialRatiosChart(containerId, companyId, period = 'annual') {
        try {
            const data = await this.fetchFinancialData('/api/financial-charts/financial-ratios', {
                company_id: companyId,
                period: period
            });

            const chartData = {
                labels: data.periods || [],
                datasets: [
                    {
                        label: 'Current Ratio',
                        data: data.current_ratio || [],
                        backgroundColor: this.colors.assets,
                        borderColor: this.colors.assets,
                        borderWidth: 2,
                        yAxisID: 'y',
                        fill: false
                    },
                    {
                        label: 'Debt-to-Equity Ratio',
                        data: data.debt_to_equity || [],
                        backgroundColor: this.colors.liabilities,
                        borderColor: this.colors.liabilities,
                        borderWidth: 2,
                        yAxisID: 'y',
                        fill: false
                    },
                    {
                        label: 'Return on Equity (%)',
                        data: data.return_on_equity || [],
                        backgroundColor: this.colors.profit,
                        borderColor: this.colors.profit,
                        borderWidth: 2,
                        yAxisID: 'y1',
                        fill: false
                    },
                    {
                        label: 'Profit Margin (%)',
                        data: data.profit_margin || [],
                        backgroundColor: this.colors.revenue,
                        borderColor: this.colors.revenue,
                        borderWidth: 2,
                        yAxisID: 'y1',
                        fill: false
                    }
                ]
            };

            const config = {
                type: 'line',
                data: chartData,
                options: {
                    ...this.defaultConfig,
                    plugins: {
                        title: {
                            display: true,
                            text: `${data.company_name} - Financial Ratios Analysis`,
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            position: 'top'
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Period'
                            }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Ratio'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Percentage (%)'
                            },
                            grid: {
                                drawOnChartArea: false
                            }
                        }
                    }
                }
            };

            return this.renderChart(containerId, config, data);

        } catch (error) {
            console.error('❌ Error creating financial ratios chart:', error);
            this.showChartError(containerId, 'Failed to load financial ratios data');
        }
    }

    // ===== PROFITABILITY ANALYSIS =====
    async createProfitabilityChart(containerId, companyId, period = 'annual') {
        try {
            const data = await this.fetchFinancialData('/api/financial-charts/profitability', {
                company_id: companyId,
                period: period
            });

            const chartData = {
                labels: data.periods || [],
                datasets: [
                    {
                        label: 'Gross Profit Margin (%)',
                        data: data.gross_profit_margin || [],
                        backgroundColor: this.hexToRgba(this.colors.profit, 0.2),
                        borderColor: this.colors.profit,
                        borderWidth: 2,
                        fill: true
                    },
                    {
                        label: 'Operating Profit Margin (%)',
                        data: data.operating_profit_margin || [],
                        backgroundColor: this.hexToRgba(this.colors.revenue, 0.2),
                        borderColor: this.colors.revenue,
                        borderWidth: 2,
                        fill: true
                    },
                    {
                        label: 'Net Profit Margin (%)',
                        data: data.net_profit_margin || [],
                        backgroundColor: this.hexToRgba(this.colors.cash, 0.2),
                        borderColor: this.colors.cash,
                        borderWidth: 2,
                        fill: true
                    }
                ]
            };

            const config = {
                type: 'line',
                data: chartData,
                options: {
                    ...this.defaultConfig,
                    plugins: {
                        title: {
                            display: true,
                            text: `${data.company_name} - Profitability Analysis`,
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            position: 'top'
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Period'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Margin (%)'
                            },
                            min: 0,
                            ticks: {
                                callback: (value) => `${value}%`
                            }
                        }
                    },
                    elements: {
                        line: {
                            tension: 0.1
                        },
                        point: {
                            radius: 4,
                            hoverRadius: 8
                        }
                    }
                }
            };

            return this.renderChart(containerId, config, data);

        } catch (error) {
            console.error('❌ Error creating profitability chart:', error);
            this.showChartError(containerId, 'Failed to load profitability data');
        }
    }

    // ===== LIQUIDITY ANALYSIS =====
    async createLiquidityChart(containerId, companyId, period = 'annual') {
        try {
            const data = await this.fetchFinancialData('/api/financial-charts/liquidity', {
                company_id: companyId,
                period: period
            });

            const chartData = {
                labels: data.periods || [],
                datasets: [
                    {
                        label: 'Current Ratio',
                        data: data.current_ratio || [],
                        backgroundColor: this.colors.assets,
                        borderColor: this.colors.assets,
                        borderWidth: 3,
                        type: 'line',
                        yAxisID: 'y'
                    },
                    {
                        label: 'Quick Ratio',
                        data: data.quick_ratio || [],
                        backgroundColor: this.colors.cash,
                        borderColor: this.colors.cash,
                        borderWidth: 3,
                        type: 'line',
                        yAxisID: 'y'
                    },
                    {
                        label: 'Cash Ratio',
                        data: data.cash_ratio || [],
                        backgroundColor: this.colors.profit,
                        borderColor: this.colors.profit,
                        borderWidth: 3,
                        type: 'line',
                        yAxisID: 'y'
                    },
                    {
                        label: 'Working Capital ($M)',
                        data: data.working_capital || [],
                        backgroundColor: this.hexToRgba(this.colors.revenue, 0.6),
                        borderColor: this.colors.revenue,
                        borderWidth: 2,
                        type: 'bar',
                        yAxisID: 'y1'
                    }
                ]
            };

            const config = {
                type: 'bar',
                data: chartData,
                options: {
                    ...this.defaultConfig,
                    plugins: {
                        title: {
                            display: true,
                            text: `${data.company_name} - Liquidity Analysis`,
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            position: 'top'
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Period'
                            }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Ratio'
                            },
                            min: 0
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Working Capital ($M)'
                            },
                            grid: {
                                drawOnChartArea: false
                            },
                            ticks: {
                                callback: (value) => this.formatCurrency(value * 1000000)
                            }
                        }
                    }
                }
            };

            return this.renderChart(containerId, config, data);

        } catch (error) {
            console.error('❌ Error creating liquidity chart:', error);
            this.showChartError(containerId, 'Failed to load liquidity data');
        }
    }

    // ===== INDUSTRY COMPARISON CHART =====
    async createIndustryComparisonChart(containerId, companyId, metric = 'health_score') {
        try {
            const data = await this.fetchFinancialData('/api/financial-charts/industry-comparison', {
                company_id: companyId,
                metric: metric
            });

            const chartData = {
                labels: ['Company', 'Industry Average', 'Top Quartile', 'Bottom Quartile'],
                datasets: [{
                    label: data.metric_name || 'Financial Metric',
                    data: [
                        data.company_value || 0,
                        data.industry_average || 0,
                        data.top_quartile || 0,
                        data.bottom_quartile || 0
                    ],
                    backgroundColor: [
                        this.colors.profit,
                        this.colors.revenue,
                        this.colors.assets,
                        this.colors.expenses
                    ],
                    borderColor: [
                        this.colors.profit,
                        this.colors.revenue,
                        this.colors.assets,
                        this.colors.expenses
                    ],
                    borderWidth: 2
                }]
            };

            const config = {
                type: 'bar',
                data: chartData,
                options: {
                    ...this.defaultConfig,
                    plugins: {
                        title: {
                            display: true,
                            text: `${data.company_name} - Industry Comparison (${data.metric_name})`,
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Comparison Groups'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: data.metric_unit || 'Value'
                            },
                            ticks: {
                                callback: (value) => {
                                    if (data.metric_type === 'currency') {
                                        return this.formatCurrency(value);
                                    } else if (data.metric_type === 'percentage') {
                                        return `${value}%`;
                                    }
                                    return value;
                                }
                            }
                        }
                    }
                }
            };

            return this.renderChart(containerId, config, data);

        } catch (error) {
            console.error('❌ Error creating industry comparison chart:', error);
            this.showChartError(containerId, 'Failed to load industry comparison data');
        }
    }

    // ===== RISK METRICS VISUALIZATION =====
    async createRiskMetricsChart(containerId, companyId, period = 'annual') {
        try {
            const data = await this.fetchFinancialData('/api/financial-charts/risk-metrics', {
                company_id: companyId,
                period: period
            });

            const chartData = {
                labels: data.periods || [],
                datasets: [
                    {
                        label: 'Overall Risk Score',
                        data: data.overall_risk_score || [],
                        backgroundColor: this.hexToRgba(this.colors.loss, 0.2),
                        borderColor: this.colors.loss,
                        borderWidth: 3,
                        fill: true,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Liquidity Risk',
                        data: data.liquidity_risk || [],
                        backgroundColor: this.colors.expenses,
                        borderColor: this.colors.expenses,
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Credit Risk',
                        data: data.credit_risk || [],
                        backgroundColor: this.colors.liabilities,
                        borderColor: this.colors.liabilities,
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Operational Risk',
                        data: data.operational_risk || [],
                        backgroundColor: this.colors.revenue,
                        borderColor: this.colors.revenue,
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y'
                    }
                ]
            };

            const config = {
                type: 'line',
                data: chartData,
                options: {
                    ...this.defaultConfig,
                    plugins: {
                        title: {
                            display: true,
                            text: `${data.company_name} - Risk Metrics Analysis`,
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            position: 'top'
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Period'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Risk Score (0-100)'
                            },
                            min: 0,
                            max: 100
                        }
                    },
                    elements: {
                        line: {
                            tension: 0.1
                        },
                        point: {
                            radius: 4,
                            hoverRadius: 8
                        }
                    }
                }
            };

            return this.renderChart(containerId, config, data);

        } catch (error) {
            console.error('❌ Error creating risk metrics chart:', error);
            this.showChartError(containerId, 'Failed to load risk metrics data');
        }
    }

    // ===== CHART RENDERING AND MANAGEMENT =====
    renderChart(containerId, config, data) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`❌ Container ${containerId} not found`);
            return null;
        }

        // Destroy existing chart
        if (this.charts.has(containerId)) {
            this.destroyChart(containerId);
        }

        try {
            // Create canvas element
            const canvas = document.createElement('canvas');
            container.appendChild(canvas);

            // Create chart instance
            const chartInstance = new Chart(canvas, config);

            // Store chart reference
            this.charts.set(containerId, {
                instance: chartInstance,
                canvas: canvas,
                config: config,
                data: data,
                containerId: containerId
            });

            console.log(`✅ Financial chart created: ${containerId}`);
            return chartInstance;

        } catch (error) {
            console.error(`❌ Error rendering chart ${containerId}:`, error);
            this.showChartError(containerId, 'Failed to render chart');
            return null;
        }
    }

    destroyChart(containerId) {
        const chartInfo = this.charts.get(containerId);
        if (chartInfo) {
            chartInfo.instance.destroy();
            if (chartInfo.canvas && chartInfo.canvas.parentNode) {
                chartInfo.canvas.parentNode.removeChild(chartInfo.canvas);
            }
            this.charts.delete(containerId);
            console.log(`🗑️ Financial chart destroyed: ${containerId}`);
        }
    }

    async refreshFinancialChart(chartId, companyId) {
        console.log(`🔄 Refreshing financial chart: ${chartId}`);
        
        const chartInfo = this.charts.get(chartId);
        if (!chartInfo) return;

        // Determine which chart type to refresh
        const refreshMethods = {
            'income-statement-chart': () => this.createIncomeStatementChart(chartId, companyId),
            'balance-sheet-chart': () => this.createBalanceSheetChart(chartId, companyId),
            'cash-flow-chart': () => this.createCashFlowChart(chartId, companyId),
            'financial-ratios-chart': () => this.createFinancialRatiosChart(chartId, companyId),
            'profitability-chart': () => this.createProfitabilityChart(chartId, companyId),
            'liquidity-chart': () => this.createLiquidityChart(chartId, companyId),
            'industry-comparison-chart': () => this.createIndustryComparisonChart(chartId, companyId),
            'risk-metrics-chart': () => this.createRiskMetricsChart(chartId, companyId)
        };

        const refreshMethod = refreshMethods[chartId];
        if (refreshMethod) {
            await refreshMethod();
        }
    }

    async updateChartPeriod(chartId, period) {
        console.log(`📅 Updating chart period: ${chartId} to ${period}`);
        
        const chartInfo = this.charts.get(chartId);
        if (!chartInfo || !chartInfo.data || !chartInfo.data.company_id) return;

        const companyId = chartInfo.data.company_id;
        
        // Refresh chart with new period
        await this.refreshFinancialChart(chartId, companyId);
    }

    // ===== UTILITY METHODS =====
    hexToRgba(hex, alpha = 1) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    formatCurrency(amount, currency = 'USD') {
        if (amount === null || amount === undefined) return '$0';
        
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency,
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(amount);
    }

    showChartError(containerId, message) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="chart-error text-center p-4">
                    <i class="fas fa-exclamation-triangle text-warning fa-3x mb-3"></i>
                    <h6 class="text-muted">Chart Error</h6>
                    <p class="text-muted small">${message}</p>
                    <button class="btn btn-sm btn-outline-primary financial-chart-refresh" 
                            data-chart-id="${containerId}">
                        <i class="fas fa-refresh me-1"></i> Retry
                    </button>
                </div>
            `;
        }
    }

    // ===== COMPREHENSIVE FINANCIAL DASHBOARD =====
    async loadCompanyFinancialDashboard(companyId, containerSelectors = {}) {
        const defaultSelectors = {
            incomeStatement: 'income-statement-chart',
            balanceSheet: 'balance-sheet-chart',
            cashFlow: 'cash-flow-chart',
            financialRatios: 'financial-ratios-chart',
            profitability: 'profitability-chart',
            liquidity: 'liquidity-chart',
            industryComparison: 'industry-comparison-chart',
            riskMetrics: 'risk-metrics-chart'
        };

        const selectors = { ...defaultSelectors, ...containerSelectors };

        try {
            console.log(`📊 Loading comprehensive financial dashboard for company: ${companyId}`);
            
            // Load all financial charts
            const chartPromises = [
                this.createIncomeStatementChart(selectors.incomeStatement, companyId),
                this.createBalanceSheetChart(selectors.balanceSheet, companyId),
                this.createCashFlowChart(selectors.cashFlow, companyId),
                this.createFinancialRatiosChart(selectors.financialRatios, companyId),
                this.createProfitabilityChart(selectors.profitability, companyId),
                this.createLiquidityChart(selectors.liquidity, companyId),
                this.createIndustryComparisonChart(selectors.industryComparison, companyId),
                this.createRiskMetricsChart(selectors.riskMetrics, companyId)
            ];

            const results = await Promise.allSettled(chartPromises);
            
            // Count successful vs failed charts
            const successful = results.filter(r => r.status === 'fulfilled').length;
            const failed = results.filter(r => r.status === 'rejected').length;
            
            console.log(`✅ Financial dashboard loaded: ${successful} successful, ${failed} failed`);
            
            if (failed > 0) {
                console.warn(`⚠️ ${failed} charts failed to load`);
            }

        } catch (error) {
            console.error(`❌ Error loading financial dashboard for company ${companyId}:`, error);
        }
    }

    // ===== PORTFOLIO FINANCIAL CHARTS =====
    async createPortfolioOverviewChart(containerId) {
        try {
            const data = await this.fetchFinancialData('/api/financial-charts/portfolio-overview');
            
            const chartData = {
                labels: data.company_names || [],
                datasets: [
                    {
                        label: 'Total Assets ($M)',
                        data: data.total_assets || [],
                        backgroundColor: this.hexToRgba(this.colors.assets, 0.7),
                        borderColor: this.colors.assets,
                        borderWidth: 2,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Revenue ($M)',
                        data: data.revenue || [],
                        backgroundColor: this.hexToRgba(this.colors.revenue, 0.7),
                        borderColor: this.colors.revenue,
                        borderWidth: 2,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Health Score',
                        data: data.health_scores || [],
                        backgroundColor: this.hexToRgba(this.colors.profit, 0.7),
                        borderColor: this.colors.profit,
                        borderWidth: 2,
                        type: 'line',
                        yAxisID: 'y1'
                    }
                ]
            };

            const config = {
                type: 'bar',
                data: chartData,
                options: {
                    ...this.defaultConfig,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Portfolio Financial Overview',
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            position: 'top'
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Companies'
                            }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Amount ($M)'
                            },
                            ticks: {
                                callback: (value) => this.formatCurrency(value * 1000000)
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Health Score'
                            },
                            min: 0,
                            max: 100,
                            grid: {
                                drawOnChartArea: false
                            }
                        }
                    }
                }
            };

            return this.renderChart(containerId, config, data);

        } catch (error) {
            console.error('❌ Error creating portfolio overview chart:', error);
            this.showChartError(containerId, 'Failed to load portfolio overview data');
        }
    }

    async createSectorPerformanceChart(containerId) {
        try {
            const data = await this.fetchFinancialData('/api/financial-charts/sector-performance');
            
            const chartData = {
                labels: data.sectors || [],
                datasets: [
                    {
                        label: 'Average Revenue Growth (%)',
                        data: data.revenue_growth || [],
                        backgroundColor: this.colors.revenue,
                        borderColor: this.colors.revenue,
                        borderWidth: 2
                    },
                    {
                        label: 'Average Profit Margin (%)',
                        data: data.profit_margin || [],
                        backgroundColor: this.colors.profit,
                        borderColor: this.colors.profit,
                        borderWidth: 2
                    },
                    {
                        label: 'Average Health Score',
                        data: data.health_score || [],
                        backgroundColor: this.colors.assets,
                        borderColor: this.colors.assets,
                        borderWidth: 2
                    }
                ]
            };

            const config = {
                type: 'radar',
                data: chartData,
                options: {
                    ...this.defaultConfig,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Sector Performance Analysis',
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            position: 'top'
                        }
                    },
                    scales: {
                        r: {
                            angleLines: { display: true },
                            suggestedMin: 0,
                            suggestedMax: 100,
                            ticks: {
                                stepSize: 20,
                                display: true
                            }
                        }
                    }
                }
            };

            return this.renderChart(containerId, config, data);

        } catch (error) {
            console.error('❌ Error creating sector performance chart:', error);
            this.showChartError(containerId, 'Failed to load sector performance data');
        }
    }

    // ===== PREDICTIVE ANALYTICS CHARTS =====
    async createFinancialForecastChart(containerId, companyId, periods = 12) {
        try {
            const data = await this.fetchFinancialData('/api/financial-charts/forecast', {
                company_id: companyId,
                periods: periods
            });
            
            const chartData = {
                labels: data.periods || [],
                datasets: [
                    {
                        label: 'Historical Revenue',
                        data: data.historical_revenue || [],
                        backgroundColor: this.hexToRgba(this.colors.revenue, 0.7),
                        borderColor: this.colors.revenue,
                        borderWidth: 2,
                        pointStyle: 'circle'
                    },
                    {
                        label: 'Predicted Revenue',
                        data: data.predicted_revenue || [],
                        backgroundColor: this.hexToRgba(this.colors.revenue, 0.3),
                        borderColor: this.colors.revenue,
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointStyle: 'triangle'
                    },
                    {
                        label: 'Historical Cash Flow',
                        data: data.historical_cash_flow || [],
                        backgroundColor: this.hexToRgba(this.colors.cash, 0.7),
                        borderColor: this.colors.cash,
                        borderWidth: 2,
                        pointStyle: 'circle'
                    },
                    {
                        label: 'Predicted Cash Flow',
                        data: data.predicted_cash_flow || [],
                        backgroundColor: this.hexToRgba(this.colors.cash, 0.3),
                        borderColor: this.colors.cash,
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointStyle: 'triangle'
                    }
                ]
            };

            const config = {
                type: 'line',
                data: chartData,
                options: {
                    ...this.defaultConfig,
                    plugins: {
                        title: {
                            display: true,
                            text: `${data.company_name} - Financial Forecast`,
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            position: 'top'
                        },
                        annotation: {
                            annotations: {
                                forecastLine: {
                                    type: 'line',
                                    xMin: data.forecast_start_index || 6,
                                    xMax: data.forecast_start_index || 6,
                                    borderColor: this.colors.expenses,
                                    borderWidth: 2,
                                    borderDash: [10, 5],
                                    label: {
                                        content: 'Forecast Starts',
                                        enabled: true,
                                        position: 'top'
                                    }
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Period'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Amount ($)'
                            },
                            ticks: {
                                callback: (value) => this.formatCurrency(value)
                            }
                        }
                    },
                    elements: {
                        line: {
                            tension: 0.1
                        },
                        point: {
                            radius: 4,
                            hoverRadius: 8
                        }
                    }
                }
            };

            return this.renderChart(containerId, config, data);

        } catch (error) {
            console.error('❌ Error creating financial forecast chart:', error);
            this.showChartError(containerId, 'Failed to load forecast data');
        }
    }

    // ===== EXPORT AND REPORTING =====
    async exportChart(chartId, format = 'png') {
        const chartInfo = this.charts.get(chartId);
        if (!chartInfo) {
            console.error(`❌ Chart ${chartId} not found for export`);
            return;
        }

        try {
            const canvas = chartInfo.canvas;
            const dataURL = canvas.toDataURL(`image/${format}`);
            
            // Create download link
            const link = document.createElement('a');
            link.download = `${chartId}-${new Date().toISOString().split('T')[0]}.${format}`;
            link.href = dataURL;
            link.click();
            
            console.log(`📥 Chart exported: ${chartId}.${format}`);

        } catch (error) {
            console.error(`❌ Error exporting chart ${chartId}:`, error);
        }
    }

    async generateFinancialReport(companyId, reportType = 'comprehensive') {
        try {
            console.log(`📊 Generating financial report for company: ${companyId}`);
            
            const reportData = await this.fetchFinancialData('/api/financial-charts/generate-report', {
                company_id: companyId,
                report_type: reportType,
                include_charts: true
            });

            // Create report with embedded charts
            const report = {
                company: reportData.company_info,
                summary: reportData.executive_summary,
                charts: [],
                generated_at: new Date().toISOString()
            };

            // Generate chart images for report
            for (const [chartId, chartInfo] of this.charts) {
                if (chartInfo.data && chartInfo.data.company_id === companyId) {
                    const chartImage = chartInfo.canvas.toDataURL('image/png');
                    report.charts.push({
                        id: chartId,
                        title: chartInfo.config.options.plugins.title.text,
                        image: chartImage
                    });
                }
            }

            return report;

        } catch (error) {
            console.error(`❌ Error generating financial report for ${companyId}:`, error);
            throw error;
        }
    }

    // ===== PUBLIC API =====
    getChart(chartId) {
        return this.charts.get(chartId);
    }

    getAllCharts() {
        return Array.from(this.charts.keys());
    }

    getChartData(chartId) {
        const chartInfo = this.charts.get(chartId);
        return chartInfo ? chartInfo.data : null;
    }

    resizeChart(chartId) {
        const chartInfo = this.charts.get(chartId);
        if (chartInfo && chartInfo.instance) {
            chartInfo.instance.resize();
        }
    }

    resizeAllCharts() {
        this.charts.forEach((chartInfo, chartId) => {
            this.resizeChart(chartId);
        });
    }

    destroyAllCharts() {
        this.charts.forEach((chartInfo, chartId) => {
            this.destroyChart(chartId);
        });
        console.log('🗑️ All financial charts destroyed');
    }

    // ===== BATCH OPERATIONS =====
    async loadPortfolioCharts(containerSelectors = {}) {
        const defaultSelectors = {
            portfolioOverview: 'portfolio-overview-chart',
            sectorPerformance: 'sector-performance-chart'
        };

        const selectors = { ...defaultSelectors, ...containerSelectors };

        try {
            const chartPromises = [
                this.createPortfolioOverviewChart(selectors.portfolioOverview),
                this.createSectorPerformanceChart(selectors.sectorPerformance)
            ];

            await Promise.allSettled(chartPromises);
            console.log('✅ Portfolio charts loaded');

        } catch (error) {
            console.error('❌ Error loading portfolio charts:', error);
        }
    }

    async refreshAllCompanyCharts(companyId) {
        console.log(`🔄 Refreshing all charts for company: ${companyId}`);
        
        const refreshPromises = [];
        
        this.charts.forEach((chartInfo, chartId) => {
            if (chartInfo.data && chartInfo.data.company_id === companyId) {
                refreshPromises.push(this.refreshFinancialChart(chartId, companyId));
            }
        });

        try {
            await Promise.allSettled(refreshPromises);
            console.log(`✅ All charts refreshed for company: ${companyId}`);
        } catch (error) {
            console.error(`❌ Error refreshing charts for company ${companyId}:`, error);
        }
    }

    // ===== PERFORMANCE MONITORING =====
    getChartPerformanceMetrics() {
        return {
            totalCharts: this.charts.size,
            chartTypes: Array.from(this.charts.values()).map(c => c.config.type),
            memoryUsage: this.estimateMemoryUsage(),
            renderTimes: this.getRenderTimes()
        };
    }

    estimateMemoryUsage() {
        let totalDataPoints = 0;
        this.charts.forEach(chartInfo => {
            if (chartInfo.config.data && chartInfo.config.data.datasets) {
                chartInfo.config.data.datasets.forEach(dataset => {
                    if (dataset.data) {
                        totalDataPoints += dataset.data.length;
                    }
                });
            }
        });
        return `~${(totalDataPoints * 8 / 1024).toFixed(2)} KB`; // Rough estimate
    }

    getRenderTimes() {
        // This would need to be implemented with performance monitoring
        return 'Performance monitoring not implemented';
    }
}

// Global instance
window.financialCharts = new FinancialCharts();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FinancialCharts;
}