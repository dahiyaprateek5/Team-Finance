// chart-config.js - Chart Configuration and Rendering Engine
// Author: Financial Risk Assessment Platform
// Version: 2.0 - Enhanced Chart System

class ChartConfigurator {
    constructor() {
        this.charts = new Map();
        this.defaultColors = {
            primary: '#3b82f6',
            success: '#10b981',
            warning: '#f59e0b',
            danger: '#ef4444',
            info: '#06b6d4',
            secondary: '#6b7280'
        };
        this.chartLibraries = {
            chartjs: null,
            plotly: null,
            d3: null
        };
        this.init();
    }

    init() {
        console.log('📊 Initializing Chart Configurator...');
        this.loadChartLibraries();
        this.setupEventListeners();
    }

    // ===== LIBRARY LOADING =====
    async loadChartLibraries() {
        try {
            // Check if Chart.js is available
            if (typeof Chart !== 'undefined') {
                this.chartLibraries.chartjs = Chart;
                console.log('✅ Chart.js loaded');
            }

            // Check if Plotly is available
            if (typeof Plotly !== 'undefined') {
                this.chartLibraries.plotly = Plotly;
                console.log('✅ Plotly loaded');
            }

            // Check if D3 is available
            if (typeof d3 !== 'undefined') {
                this.chartLibraries.d3 = d3;
                console.log('✅ D3.js loaded');
            }

            // Set default library
            this.defaultLibrary = this.chartLibraries.chartjs ? 'chartjs' : 
                                this.chartLibraries.plotly ? 'plotly' : 'canvas';

            console.log(`📊 Default chart library: ${this.defaultLibrary}`);

        } catch (error) {
            console.error('❌ Error loading chart libraries:', error);
            this.defaultLibrary = 'canvas'; // Fallback to canvas-based charts
        }
    }

    setupEventListeners() {
        // Chart refresh button
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('refresh-chart')) {
                const chartId = e.target.dataset.chartId;
                this.refreshChart(chartId);
            }
        });

        // Chart export button
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('export-chart')) {
                const chartId = e.target.dataset.chartId;
                const format = e.target.dataset.format || 'png';
                this.exportChart(chartId, format);
            }
        });

        // Window resize handler
        window.addEventListener('resize', () => {
            this.resizeAllCharts();
        });
    }

    // ===== CHART CREATION METHODS =====
    
    // Financial Health Radar Chart
    createFinancialHealthRadar(containerId, data, options = {}) {
        const config = {
            type: 'radar',
            data: {
                labels: data.labels || ['Liquidity', 'Profitability', 'Efficiency', 'Leverage', 'Growth'],
                datasets: [{
                    label: data.companyName || 'Company',
                    data: data.values || [0, 0, 0, 0, 0],
                    backgroundColor: this.hexToRgba(this.defaultColors.primary, 0.2),
                    borderColor: this.defaultColors.primary,
                    borderWidth: 2,
                    pointBackgroundColor: this.defaultColors.primary,
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: this.defaultColors.primary
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: options.title || 'Financial Health Radar',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        position: 'bottom'
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

        return this.renderChart(containerId, config, 'chartjs');
    }

    // Cash Flow Waterfall Chart
    createCashFlowWaterfall(containerId, data, options = {}) {
        if (this.chartLibraries.plotly) {
            return this.createPlotlyWaterfall(containerId, data, options);
        } else {
            return this.createChartJSWaterfall(containerId, data, options);
        }
    }

    createPlotlyWaterfall(containerId, data, options = {}) {
        const trace = {
            type: 'waterfall',
            orientation: 'v',
            x: data.categories || ['Operating CF', 'Investing CF', 'Financing CF', 'Net Change'],
            y: data.values || [0, 0, 0, 0],
            text: data.values.map(v => this.formatCurrency(v)),
            textposition: 'outside',
            connector: {
                line: {
                    color: 'rgba(63, 63, 63, 0.2)'
                }
            },
            increasing: { marker: { color: this.defaultColors.success } },
            decreasing: { marker: { color: this.defaultColors.danger } },
            totals: { marker: { color: this.defaultColors.info } }
        };

        const layout = {
            title: options.title || 'Cash Flow Waterfall',
            xaxis: { title: 'Cash Flow Components' },
            yaxis: { title: 'Amount ($)' },
            showlegend: false,
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)'
        };

        const config = { responsive: true, displayModeBar: false };

        Plotly.newPlot(containerId, [trace], layout, config);
        
        const chartInstance = {
            id: containerId,
            type: 'waterfall',
            library: 'plotly',
            data: data,
            destroy: () => Plotly.purge(containerId)
        };

        this.charts.set(containerId, chartInstance);
        return chartInstance;
    }

    createChartJSWaterfall(containerId, data, options = {}) {
        // Fallback waterfall using Chart.js bar chart
        const cumulativeValues = this.calculateCumulativeValues(data.values);
        
        const config = {
            type: 'bar',
            data: {
                labels: data.categories || ['Operating CF', 'Investing CF', 'Financing CF', 'Net Change'],
                datasets: [{
                    label: 'Cash Flow',
                    data: cumulativeValues,
                    backgroundColor: data.values.map(v => 
                        v >= 0 ? this.defaultColors.success : this.defaultColors.danger
                    ),
                    borderColor: data.values.map(v => 
                        v >= 0 ? this.defaultColors.success : this.defaultColors.danger
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: options.title || 'Cash Flow Components'
                    },
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return chartConfigurator.formatCurrency(value);
                            }
                        }
                    }
                }
            }
        };

        return this.renderChart(containerId, config, 'chartjs');
    }

    // Risk Assessment Gauge Chart
    createRiskGauge(containerId, data, options = {}) {
        const riskValue = data.riskScore || 0;
        const riskLevel = data.riskLevel || 'moderate';
        
        if (this.chartLibraries.plotly) {
            return this.createPlotlyGauge(containerId, riskValue, riskLevel, options);
        } else {
            return this.createCanvasGauge(containerId, riskValue, riskLevel, options);
        }
    }

    createPlotlyGauge(containerId, value, level, options = {}) {
        const trace = {
            type: 'indicator',
            mode: 'gauge+number+delta',
            value: value,
            domain: { x: [0, 1], y: [0, 1] },
            title: { text: options.title || 'Risk Level' },
            delta: { reference: 50 },
            gauge: {
                axis: { range: [null, 100] },
                bar: { color: this.getRiskColor(level) },
                steps: [
                    { range: [0, 25], color: this.hexToRgba(this.defaultColors.success, 0.3) },
                    { range: [25, 50], color: this.hexToRgba(this.defaultColors.info, 0.3) },
                    { range: [50, 75], color: this.hexToRgba(this.defaultColors.warning, 0.3) },
                    { range: [75, 100], color: this.hexToRgba(this.defaultColors.danger, 0.3) }
                ],
                threshold: {
                    line: { color: 'red', width: 4 },
                    thickness: 0.75,
                    value: 90
                }
            }
        };

        const layout = {
            width: 400,
            height: 300,
            margin: { t: 50, r: 50, l: 50, b: 50 },
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)'
        };

        Plotly.newPlot(containerId, [trace], layout, { responsive: true });
        
        const chartInstance = {
            id: containerId,
            type: 'gauge',
            library: 'plotly',
            data: { value, level },
            destroy: () => Plotly.purge(containerId)
        };

        this.charts.set(containerId, chartInstance);
        return chartInstance;
    }

    // Financial Metrics Comparison Chart
    createMetricsComparison(containerId, data, options = {}) {
        const config = {
            type: 'bar',
            data: {
                labels: data.metrics || [],
                datasets: [{
                    label: data.companyName || 'Company',
                    data: data.values || [],
                    backgroundColor: this.defaultColors.primary,
                    borderColor: this.defaultColors.primary,
                    borderWidth: 1
                }, {
                    label: 'Industry Average',
                    data: data.industryAverages || [],
                    backgroundColor: this.hexToRgba(this.defaultColors.secondary, 0.6),
                    borderColor: this.defaultColors.secondary,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: options.title || 'Financial Metrics Comparison'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        };

        return this.renderChart(containerId, config, 'chartjs');
    }

    // Trend Line Chart
    createTrendChart(containerId, data, options = {}) {
        const config = {
            type: 'line',
            data: {
                labels: data.periods || [],
                datasets: data.datasets.map((dataset, index) => ({
                    label: dataset.label,
                    data: dataset.data,
                    borderColor: this.getColorByIndex(index),
                    backgroundColor: this.hexToRgba(this.getColorByIndex(index), 0.1),
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: options.title || 'Financial Trends'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return chartConfigurator.formatCurrency(value);
                            }
                        }
                    }
                }
            }
        };

        return this.renderChart(containerId, config, 'chartjs');
    }

    // Pie Chart for Portfolio Distribution
    createPortfolioPie(containerId, data, options = {}) {
        const config = {
            type: 'doughnut',
            data: {
                labels: data.categories || [],
                datasets: [{
                    data: data.values || [],
                    backgroundColor: [
                        this.defaultColors.primary,
                        this.defaultColors.success,
                        this.defaultColors.warning,
                        this.defaultColors.danger,
                        this.defaultColors.info,
                        this.defaultColors.secondary
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: options.title || 'Portfolio Distribution'
                    },
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        };

        return this.renderChart(containerId, config, 'chartjs');
    }

    // ===== DATABASE INTEGRATION METHODS =====
    
    async fetchChartData(endpoint, params = {}) {
        try {
            const url = new URL(endpoint, window.location.origin);
            Object.keys(params).forEach(key => url.searchParams.append(key, params[key]));
            
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log(`📊 Chart data fetched from ${endpoint}:`, data);
            return data;
        } catch (error) {
            console.error(`❌ Error fetching chart data from ${endpoint}:`, error);
            throw error;
        }
    }

    async createCompanyHealthRadar(containerId, companyId) {
        try {
            const data = await this.fetchChartData('/api/company-health-metrics', { company_id: companyId });
            
            const chartData = {
                companyName: data.company_name,
                labels: ['Liquidity', 'Profitability', 'Efficiency', 'Leverage', 'Growth'],
                values: [
                    data.liquidity_score || 0,
                    data.profitability_score || 0,
                    data.efficiency_score || 0,
                    data.leverage_score || 0,
                    data.growth_score || 0
                ]
            };

            return this.createFinancialHealthRadar(containerId, chartData);
        } catch (error) {
            console.error('❌ Error creating health radar chart:', error);
            this.showChartError(containerId, 'Failed to load company health data');
        }
    }

    async createCompanyCashFlowWaterfall(containerId, companyId) {
        try {
            const data = await this.fetchChartData('/api/cash-flow-waterfall', { company_id: companyId });
            
            const chartData = {
                categories: ['Operating CF', 'Investing CF', 'Financing CF', 'Net Change'],
                values: [
                    data.operating_cash_flow || 0,
                    data.investing_cash_flow || 0,
                    data.financing_cash_flow || 0,
                    data.net_cash_change || 0
                ]
            };

            return this.createCashFlowWaterfall(containerId, chartData);
        } catch (error) {
            console.error('❌ Error creating cash flow waterfall:', error);
            this.showChartError(containerId, 'Failed to load cash flow data');
        }
    }

    async createRiskAssessmentGauge(containerId, companyId) {
        try {
            const data = await this.fetchChartData('/api/risk-assessment', { company_id: companyId });
            
            const chartData = {
                riskScore: data.risk_score || 0,
                riskLevel: data.risk_level || 'moderate'
            };

            return this.createRiskGauge(containerId, chartData);
        } catch (error) {
            console.error('❌ Error creating risk gauge:', error);
            this.showChartError(containerId, 'Failed to load risk assessment data');
        }
    }

    async createIndustryComparison(containerId, companyId) {
        try {
            const data = await this.fetchChartData('/api/industry-comparison', { company_id: companyId });
            
            const chartData = {
                companyName: data.company_name,
                metrics: data.metrics_names || [],
                values: data.company_values || [],
                industryAverages: data.industry_averages || []
            };

            return this.createMetricsComparison(containerId, chartData);
        } catch (error) {
            console.error('❌ Error creating industry comparison:', error);
            this.showChartError(containerId, 'Failed to load industry comparison data');
        }
    }

    async createFinancialTrends(containerId, companyId, periods = 12) {
        try {
            const data = await this.fetchChartData('/api/financial-trends', { 
                company_id: companyId, 
                periods: periods 
            });
            
            const chartData = {
                periods: data.periods || [],
                datasets: [
                    {
                        label: 'Revenue',
                        data: data.revenue_trend || []
                    },
                    {
                        label: 'Net Income',
                        data: data.net_income_trend || []
                    },
                    {
                        label: 'Operating Cash Flow',
                        data: data.operating_cf_trend || []
                    }
                ]
            };

            return this.createTrendChart(containerId, chartData);
        } catch (error) {
            console.error('❌ Error creating financial trends:', error);
            this.showChartError(containerId, 'Failed to load financial trends data');
        }
    }

    async createPortfolioDistribution(containerId) {
        try {
            const data = await this.fetchChartData('/api/portfolio-distribution');
            
            const chartData = {
                categories: data.categories || [],
                values: data.values || []
            };

            return this.createPortfolioPie(containerId, chartData);
        } catch (error) {
            console.error('❌ Error creating portfolio distribution:', error);
            this.showChartError(containerId, 'Failed to load portfolio data');
        }
    }

    // ===== CHART RENDERING ENGINE =====
    renderChart(containerId, config, library = null) {
        const targetLibrary = library || this.defaultLibrary;
        const container = document.getElementById(containerId);
        
        if (!container) {
            console.error(`❌ Container ${containerId} not found`);
            return null;
        }

        // Destroy existing chart if it exists
        if (this.charts.has(containerId)) {
            this.destroyChart(containerId);
        }

        let chartInstance = null;

        try {
            if (targetLibrary === 'chartjs' && this.chartLibraries.chartjs) {
                chartInstance = new this.chartLibraries.chartjs(container, config);
            } else if (targetLibrary === 'plotly' && this.chartLibraries.plotly) {
                // Plotly charts are handled differently in specific methods
                return null;
            } else {
                // Fallback to canvas-based chart
                chartInstance = this.createCanvasChart(container, config);
            }

            if (chartInstance) {
                this.charts.set(containerId, {
                    id: containerId,
                    instance: chartInstance,
                    library: targetLibrary,
                    config: config,
                    destroy: () => chartInstance.destroy()
                });

                console.log(`✅ Chart created in ${containerId} using ${targetLibrary}`);
            }

            return chartInstance;

        } catch (error) {
            console.error(`❌ Error creating chart in ${containerId}:`, error);
            this.showChartError(containerId, 'Failed to create chart');
            return null;
        }
    }

    // ===== CHART MANAGEMENT =====
    destroyChart(containerId) {
        const chart = this.charts.get(containerId);
        if (chart && chart.destroy) {
            chart.destroy();
            this.charts.delete(containerId);
            console.log(`🗑️ Chart destroyed: ${containerId}`);
        }
    }

    refreshChart(chartId) {
        const chart = this.charts.get(chartId);
        if (chart) {
            console.log(`🔄 Refreshing chart: ${chartId}`);
            
            // Get the refresh function based on chart type
            const refreshFunction = this.getChartRefreshFunction(chart);
            if (refreshFunction) {
                refreshFunction();
            }
        }
    }

    getChartRefreshFunction(chart) {
        // Map chart types to their refresh functions
        const refreshMap = {
            'health-radar': () => this.createCompanyHealthRadar(chart.id, this.getCurrentCompanyId()),
            'cash-flow-waterfall': () => this.createCompanyCashFlowWaterfall(chart.id, this.getCurrentCompanyId()),
            'risk-gauge': () => this.createRiskAssessmentGauge(chart.id, this.getCurrentCompanyId()),
            'industry-comparison': () => this.createIndustryComparison(chart.id, this.getCurrentCompanyId()),
            'financial-trends': () => this.createFinancialTrends(chart.id, this.getCurrentCompanyId()),
            'portfolio-distribution': () => this.createPortfolioDistribution(chart.id)
        };

        return refreshMap[chart.type] || null;
    }

    resizeAllCharts() {
        this.charts.forEach((chart, chartId) => {
            if (chart.instance && chart.instance.resize) {
                chart.instance.resize();
            } else if (chart.library === 'plotly') {
                Plotly.Plots.resize(chartId);
            }
        });
    }

    exportChart(chartId, format = 'png') {
        const chart = this.charts.get(chartId);
        if (!chart) return;

        try {
            if (chart.library === 'chartjs') {
                const canvas = chart.instance.canvas;
                const link = document.createElement('a');
                link.download = `${chartId}-chart.${format}`;
                link.href = canvas.toDataURL(`image/${format}`);
                link.click();
            } else if (chart.library === 'plotly') {
                Plotly.downloadImage(chartId, {
                    format: format,
                    filename: `${chartId}-chart`,
                    height: 600,
                    width: 800
                });
            }

            console.log(`📥 Chart exported: ${chartId}.${format}`);
        } catch (error) {
            console.error(`❌ Error exporting chart ${chartId}:`, error);
        }
    }

    // ===== UTILITY METHODS =====
    calculateCumulativeValues(values) {
        let cumulative = 0;
        return values.map(value => {
            cumulative += value;
            return cumulative;
        });
    }

    hexToRgba(hex, alpha = 1) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    getRiskColor(riskLevel) {
        const riskColors = {
            low: this.defaultColors.success,
            moderate: this.defaultColors.warning,
            high: this.defaultColors.danger,
            critical: '#8b0000'
        };
        return riskColors[riskLevel] || this.defaultColors.secondary;
    }

    getColorByIndex(index) {
        const colors = [
            this.defaultColors.primary,
            this.defaultColors.success,
            this.defaultColors.warning,
            this.defaultColors.danger,
            this.defaultColors.info,
            this.defaultColors.secondary
        ];
        return colors[index % colors.length];
    }

    formatCurrency(value, currency = 'USD') {
        if (value === null || value === undefined) return '$0';
        
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency,
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(value);
    }

    showChartError(containerId, message) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="chart-error text-center p-4">
                    <i class="fas fa-exclamation-triangle text-warning fa-3x mb-3"></i>
                    <h6 class="text-muted">Chart Error</h6>
                    <p class="text-muted small">${message}</p>
                    <button class="btn btn-sm btn-outline-primary refresh-chart" data-chart-id="${containerId}">
                        <i class="fas fa-refresh me-1"></i> Retry
                    </button>
                </div>
            `;
        }
    }

    createCanvasChart(container, config) {
        // Fallback canvas-based chart implementation
        const canvas = document.createElement('canvas');
        canvas.width = container.clientWidth || 400;
        canvas.height = container.clientHeight || 300;
        container.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        
        // Simple bar chart implementation
        if (config.type === 'bar') {
            this.drawCanvasBarChart(ctx, config.data, canvas.width, canvas.height);
        } else if (config.type === 'line') {
            this.drawCanvasLineChart(ctx, config.data, canvas.width, canvas.height);
        } else {
            this.drawCanvasPlaceholder(ctx, canvas.width, canvas.height, 'Chart not supported');
        }

        return {
            canvas: canvas,
            destroy: () => container.removeChild(canvas),
            resize: () => {
                canvas.width = container.clientWidth;
                canvas.height = container.clientHeight;
            }
        };
    }

    drawCanvasBarChart(ctx, data, width, height) {
        // Simple canvas bar chart implementation
        const padding = 40;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;
        
        if (!data.datasets || !data.datasets[0] || !data.datasets[0].data) return;
        
        const values = data.datasets[0].data;
        const maxValue = Math.max(...values);
        const barWidth = chartWidth / values.length * 0.8;
        const barSpacing = chartWidth / values.length * 0.2;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = this.defaultColors.primary;

        values.forEach((value, index) => {
            const barHeight = (value / maxValue) * chartHeight;
            const x = padding + index * (barWidth + barSpacing);
            const y = height - padding - barHeight;

            ctx.fillRect(x, y, barWidth, barHeight);
        });
    }

    drawCanvasLineChart(ctx, data, width, height) {
        // Simple canvas line chart implementation
        const padding = 40;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;
        
        if (!data.datasets || !data.datasets[0] || !data.datasets[0].data) return;
        
        const values = data.datasets[0].data;
        const maxValue = Math.max(...values);
        const pointSpacing = chartWidth / (values.length - 1);

        ctx.clearRect(0, 0, width, height);
        ctx.strokeStyle = this.defaultColors.primary;
        ctx.lineWidth = 2;
        ctx.beginPath();

        values.forEach((value, index) => {
            const x = padding + index * pointSpacing;
            const y = height - padding - (value / maxValue) * chartHeight;

            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();
    }

    drawCanvasPlaceholder(ctx, width, height, message) {
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, width, height);
        
        ctx.fillStyle = '#6c757d';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(message, width / 2, height / 2);
    }

    getCurrentCompanyId() {
        // Get current company ID from URL params or global state
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('company_id') || window.currentCompanyId || null;
    }

    // ===== PUBLIC API =====
    async loadCompanyCharts(companyId, containerSelectors = {}) {
        const defaultSelectors = {
            healthRadar: 'health-radar-chart',
            cashFlowWaterfall: 'cash-flow-waterfall-chart',
            riskGauge: 'risk-gauge-chart',
            industryComparison: 'industry-comparison-chart',
            financialTrends: 'financial-trends-chart'
        };

        const selectors = { ...defaultSelectors, ...containerSelectors };

        // Load all charts for the company
        const chartPromises = [
            this.createCompanyHealthRadar(selectors.healthRadar, companyId),
            this.createCompanyCashFlowWaterfall(selectors.cashFlowWaterfall, companyId),
            this.createRiskAssessmentGauge(selectors.riskGauge, companyId),
            this.createIndustryComparison(selectors.industryComparison, companyId),
            this.createFinancialTrends(selectors.financialTrends, companyId)
        ];

        try {
            await Promise.allSettled(chartPromises);
            console.log(`✅ All charts loaded for company: ${companyId}`);
        } catch (error) {
            console.error(`❌ Error loading charts for company ${companyId}:`, error);
        }
    }

    async loadPortfolioCharts(containerSelectors = {}) {
        const defaultSelectors = {
            portfolioDistribution: 'portfolio-distribution-chart',
            riskOverview: 'portfolio-risk-chart'
        };

        const selectors = { ...defaultSelectors, ...containerSelectors };

        try {
            await this.createPortfolioDistribution(selectors.portfolioDistribution);
            console.log('✅ Portfolio charts loaded');
        } catch (error) {
            console.error('❌ Error loading portfolio charts:', error);
        }
    }

    destroyAllCharts() {
        this.charts.forEach((chart, chartId) => {
            this.destroyChart(chartId);
        });
        console.log('🗑️ All charts destroyed');
    }
}

// Global instance
window.chartConfigurator = new ChartConfigurator();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChartConfigurator;
}