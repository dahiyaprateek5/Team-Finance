// dashboard-analytics.js - Dashboard Analytics Engine
// Author: Financial Risk Assessment Platform
// Version: 2.0 - Database-Driven Analytics

class DashboardAnalytics {
    constructor() {
        this.analytics = new Map();
        this.refreshInterval = null;
        this.refreshRate = 300000; // 5 minutes
        this.cache = new Map();
        this.cacheTimeout = 60000; // 1 minute cache
        this.init();
    }

    init() {
        console.log('📊 Initializing Dashboard Analytics Engine...');
        this.setupEventListeners();
        this.startAutoRefresh();
        this.loadDashboardData();
    }

    setupEventListeners() {
        // Refresh button
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('refresh-dashboard')) {
                this.refreshDashboard();
            }
        });

        // Date range selector
        const dateRangeSelector = document.getElementById('dashboard-date-range');
        if (dateRangeSelector) {
            dateRangeSelector.addEventListener('change', (e) => {
                this.handleDateRangeChange(e.target.value);
            });
        }

        // Company filter
        const companyFilter = document.getElementById('company-filter');
        if (companyFilter) {
            companyFilter.addEventListener('change', (e) => {
                this.handleCompanyFilter(e.target.value);
            });
        }

        // Export buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('export-dashboard')) {
                const format = e.target.dataset.format || 'pdf';
                this.exportDashboard(format);
            }
        });
    }

    // ===== DATA FETCHING FROM DATABASE =====
    async fetchDashboardData(endpoint, params = {}) {
        const cacheKey = `${endpoint}-${JSON.stringify(params)}`;
        
        // Check cache first
        if (this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                console.log(`📋 Using cached data for ${endpoint}`);
                return cached.data;
            }
        }

        try {
            const url = new URL(endpoint, window.location.origin);
            Object.keys(params).forEach(key => {
                if (params[key] !== null && params[key] !== undefined) {
                    url.searchParams.append(key, params[key]);
                }
            });
            
            console.log(`🔍 Fetching dashboard data from: ${url}`);
            
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
            
            // Cache the result
            this.cache.set(cacheKey, {
                data: data,
                timestamp: Date.now()
            });
            
            console.log(`✅ Dashboard data fetched from ${endpoint}`);
            return data;
            
        } catch (error) {
            console.error(`❌ Error fetching dashboard data from ${endpoint}:`, error);
            throw error;
        }
    }

    // ===== MAIN DASHBOARD DATA LOADING =====
    async loadDashboardData() {
        try {
            console.log('📊 Loading dashboard analytics...');
            
            // Load all dashboard components
            await Promise.allSettled([
                this.loadOverviewMetrics(),
                this.loadCompanyStatistics(),
                this.loadRiskAnalytics(),
                this.loadFinancialTrends(),
                this.loadPortfolioPerformance(),
                this.loadRecentActivities(),
                this.loadAlerts()
            ]);
            
            console.log('✅ Dashboard analytics loaded successfully');
            
        } catch (error) {
            console.error('❌ Error loading dashboard data:', error);
            this.showDashboardError('Failed to load dashboard data');
        }
    }

    // ===== OVERVIEW METRICS =====
    async loadOverviewMetrics() {
        try {
            const data = await this.fetchDashboardData('/api/dashboard/overview-metrics');
            
            const metrics = {
                totalCompanies: data.total_companies || 0,
                healthyCompanies: data.healthy_companies || 0,
                atRiskCompanies: data.at_risk_companies || 0,
                criticalCompanies: data.critical_companies || 0,
                totalPortfolioValue: data.total_portfolio_value || 0,
                avgHealthScore: data.average_health_score || 0,
                monthlyGrowth: data.monthly_growth_rate || 0,
                newAlertsCount: data.new_alerts_count || 0
            };

            this.updateOverviewMetrics(metrics);
            this.analytics.set('overview', metrics);
            
        } catch (error) {
            console.error('❌ Error loading overview metrics:', error);
            this.showMetricsError('overview-metrics', 'Failed to load overview metrics');
        }
    }

    updateOverviewMetrics(metrics) {
        // Update overview cards
        this.updateElement('total-companies', metrics.totalCompanies);
        this.updateElement('healthy-companies', metrics.healthyCompanies);
        this.updateElement('at-risk-companies', metrics.atRiskCompanies);
        this.updateElement('critical-companies', metrics.criticalCompanies);
        this.updateElement('total-portfolio-value', this.formatCurrency(metrics.totalPortfolioValue));
        this.updateElement('avg-health-score', metrics.avgHealthScore.toFixed(1));
        this.updateElement('monthly-growth', `${metrics.monthlyGrowth.toFixed(1)}%`);
        this.updateElement('new-alerts-count', metrics.newAlertsCount);

        // Update progress bars and indicators
        this.updateProgressBar('health-score-progress', metrics.avgHealthScore);
        this.updateGrowthIndicator('growth-indicator', metrics.monthlyGrowth);
    }

    // ===== COMPANY STATISTICS =====
    async loadCompanyStatistics() {
        try {
            const data = await this.fetchDashboardData('/api/dashboard/company-statistics');
            
            const stats = {
                byIndustry: data.companies_by_industry || [],
                bySector: data.companies_by_sector || [],
                byRiskLevel: data.companies_by_risk_level || [],
                topPerformers: data.top_performers || [],
                bottomPerformers: data.bottom_performers || []
            };

            this.updateCompanyStatistics(stats);
            this.analytics.set('company_statistics', stats);
            
        } catch (error) {
            console.error('❌ Error loading company statistics:', error);
            this.showMetricsError('company-statistics', 'Failed to load company statistics');
        }
    }

    updateCompanyStatistics(stats) {
        // Update industry distribution chart
        if (window.chartConfigurator) {
            window.chartConfigurator.createPortfolioPie('industry-distribution-chart', {
                categories: stats.byIndustry.map(item => item.industry),
                values: stats.byIndustry.map(item => item.count)
            }, { title: 'Companies by Industry' });

            // Update risk level distribution
            window.chartConfigurator.createPortfoliePie('risk-distribution-chart', {
                categories: stats.byRiskLevel.map(item => item.risk_level),
                values: stats.byRiskLevel.map(item => item.count)
            }, { title: 'Risk Level Distribution' });
        }

        // Update top/bottom performers lists
        this.updatePerformersList('top-performers-list', stats.topPerformers, true);
        this.updatePerformersList('bottom-performers-list', stats.bottomPerformers, false);
    }

    // ===== RISK ANALYTICS =====
    async loadRiskAnalytics() {
        try {
            const data = await this.fetchDashboardData('/api/dashboard/risk-analytics');
            
            const riskData = {
                overallRiskScore: data.overall_portfolio_risk_score || 0,
                riskTrends: data.risk_trends || [],
                highRiskCompanies: data.high_risk_companies || [],
                riskFactors: data.primary_risk_factors || [],
                liquidationProbabilities: data.liquidation_probabilities || []
            };

            this.updateRiskAnalytics(riskData);
            this.analytics.set('risk_analytics', riskData);
            
        } catch (error) {
            console.error('❌ Error loading risk analytics:', error);
            this.showMetricsError('risk-analytics', 'Failed to load risk analytics');
        }
    }

    updateRiskAnalytics(riskData) {
        // Update overall risk score
        this.updateElement('overall-risk-score', riskData.overallRiskScore.toFixed(1));
        this.updateRiskGauge('portfolio-risk-gauge', riskData.overallRiskScore);

        // Update risk trends chart
        if (window.chartConfigurator && riskData.riskTrends.length > 0) {
            window.chartConfigurator.createTrendChart('risk-trends-chart', {
                periods: riskData.riskTrends.map(item => item.period),
                datasets: [{
                    label: 'Portfolio Risk Score',
                    data: riskData.riskTrends.map(item => item.risk_score)
                }]
            }, { title: 'Risk Trends Over Time' });
        }

        // Update high-risk companies list
        this.updateHighRiskCompaniesList(riskData.highRiskCompanies);
        
        // Update risk factors
        this.updateRiskFactorsList(riskData.riskFactors);
    }

    // ===== FINANCIAL TRENDS =====
    async loadFinancialTrends() {
        try {
            const params = {
                periods: 12, // Last 12 months
                metrics: 'revenue,net_income,operating_cash_flow,free_cash_flow'
            };
            
            const data = await this.fetchDashboardData('/api/dashboard/financial-trends', params);
            
            const trends = {
                periods: data.periods || [],
                revenue: data.revenue_trend || [],
                netIncome: data.net_income_trend || [],
                operatingCashFlow: data.operating_cash_flow_trend || [],
                freeCashFlow: data.free_cash_flow_trend || [],
                portfolioGrowth: data.portfolio_growth_rate || 0,
                topGrowthCompanies: data.top_growth_companies || []
            };

            this.updateFinancialTrends(trends);
            this.analytics.set('financial_trends', trends);
            
        } catch (error) {
            console.error('❌ Error loading financial trends:', error);
            this.showMetricsError('financial-trends', 'Failed to load financial trends');
        }
    }

    updateFinancialTrends(trends) {
        // Update portfolio growth rate
        this.updateElement('portfolio-growth-rate', `${trends.portfolioGrowth.toFixed(1)}%`);
        this.updateGrowthIndicator('portfolio-growth-indicator', trends.portfolioGrowth);

        // Update financial trends chart
        if (window.chartConfigurator && trends.periods.length > 0) {
            window.chartConfigurator.createTrendChart('financial-trends-chart', {
                periods: trends.periods,
                datasets: [
                    {
                        label: 'Revenue',
                        data: trends.revenue
                    },
                    {
                        label: 'Net Income',
                        data: trends.netIncome
                    },
                    {
                        label: 'Operating Cash Flow',
                        data: trends.operatingCashFlow
                    },
                    {
                        label: 'Free Cash Flow',
                        data: trends.freeCashFlow
                    }
                ]
            }, { title: 'Portfolio Financial Trends' });
        }

        // Update top growth companies
        this.updateGrowthCompaniesList(trends.topGrowthCompanies);
    }

    // ===== PORTFOLIO PERFORMANCE =====
    async loadPortfolioPerformance() {
        try {
            const data = await this.fetchDashboardData('/api/dashboard/portfolio-performance');
            
            const performance = {
                totalReturn: data.total_return || 0,
                monthlyReturn: data.monthly_return || 0,
                volatility: data.portfolio_volatility || 0,
                sharpeRatio: data.sharpe_ratio || 0,
                benchmarkComparison: data.benchmark_comparison || {},
                sectorPerformance: data.sector_performance || [],
                topContributors: data.top_contributors || [],
                worstPerformers: data.worst_performers || []
            };

            this.updatePortfolioPerformance(performance);
            this.analytics.set('portfolio_performance', performance);
            
        } catch (error) {
            console.error('❌ Error loading portfolio performance:', error);
            this.showMetricsError('portfolio-performance', 'Failed to load portfolio performance');
        }
    }

    updatePortfolioPerformance(performance) {
        // Update performance metrics
        this.updateElement('total-return', `${performance.totalReturn.toFixed(2)}%`);
        this.updateElement('monthly-return', `${performance.monthlyReturn.toFixed(2)}%`);
        this.updateElement('portfolio-volatility', `${performance.volatility.toFixed(2)}%`);
        this.updateElement('sharpe-ratio', performance.sharpeRatio.toFixed(2));

        // Update performance indicators
        this.updatePerformanceIndicator('return-indicator', performance.totalReturn);
        this.updatePerformanceIndicator('monthly-return-indicator', performance.monthlyReturn);

        // Update sector performance chart
        if (window.chartConfigurator && performance.sectorPerformance.length > 0) {
            window.chartConfigurator.createMetricsComparison('sector-performance-chart', {
                metrics: performance.sectorPerformance.map(item => item.sector),
                values: performance.sectorPerformance.map(item => item.return),
                industryAverages: performance.sectorPerformance.map(item => item.benchmark)
            }, { title: 'Sector Performance vs Benchmark' });
        }

        // Update contributors and performers lists
        this.updateContributorsList('top-contributors-list', performance.topContributors);
        this.updateContributorsList('worst-performers-list', performance.worstPerformers);
    }

    // ===== RECENT ACTIVITIES =====
    async loadRecentActivities() {
        try {
            const data = await this.fetchDashboardData('/api/dashboard/recent-activities', {
                limit: 20
            });
            
            const activities = data.activities || [];
            this.updateRecentActivities(activities);
            this.analytics.set('recent_activities', activities);
            
        } catch (error) {
            console.error('❌ Error loading recent activities:', error);
            this.showMetricsError('recent-activities', 'Failed to load recent activities');
        }
    }

    updateRecentActivities(activities) {
        const container = document.getElementById('recent-activities-list');
        if (!container) return;

        if (activities.length === 0) {
            container.innerHTML = '<p class="text-muted text-center">No recent activities</p>';
            return;
        }

        const html = activities.map(activity => `
            <div class="activity-item d-flex align-items-center mb-3">
                <div class="activity-icon me-3">
                    <i class="fas ${this.getActivityIcon(activity.type)} text-${this.getActivityColor(activity.type)}"></i>
                </div>
                <div class="activity-content flex-grow-1">
                    <div class="activity-title">${activity.title}</div>
                    <div class="activity-description text-muted small">${activity.description}</div>
                    <div class="activity-time text-muted small">${this.formatRelativeTime(activity.timestamp)}</div>
                </div>
                ${activity.company_name ? `
                <div class="activity-company">
                    <span class="badge bg-light text-dark">${activity.company_name}</span>
                </div>
                ` : ''}
            </div>
        `).join('');

        container.innerHTML = html;
    }

    // ===== ALERTS MANAGEMENT =====
    async loadAlerts() {
        try {
            const data = await this.fetchDashboardData('/api/dashboard/alerts', {
                status: 'active',
                limit: 10
            });
            
            const alerts = data.alerts || [];
            this.updateAlerts(alerts);
            this.analytics.set('alerts', alerts);
            
        } catch (error) {
            console.error('❌ Error loading alerts:', error);
            this.showMetricsError('alerts', 'Failed to load alerts');
        }
    }

    updateAlerts(alerts) {
        const container = document.getElementById('alerts-list');
        if (!container) return;

        // Update alerts count badge
        const alertsBadge = document.getElementById('alerts-count-badge');
        if (alertsBadge) {
            alertsBadge.textContent = alerts.length;
            alertsBadge.classList.toggle('d-none', alerts.length === 0);
        }

        if (alerts.length === 0) {
            container.innerHTML = '<p class="text-muted text-center">No active alerts</p>';
            return;
        }

        const html = alerts.map(alert => `
            <div class="alert alert-${this.getAlertSeverityClass(alert.severity)} alert-dismissible d-flex align-items-center">
                <i class="fas ${this.getAlertIcon(alert.type)} me-2"></i>
                <div class="flex-grow-1">
                    <div class="alert-title fw-bold">${alert.title}</div>
                    <div class="alert-description small">${alert.description}</div>
                    <div class="alert-meta text-muted small mt-1">
                        ${alert.company_name} • ${this.formatRelativeTime(alert.created_at)}
                    </div>
                </div>
                <button type="button" class="btn-close" data-bs-dismiss="alert" 
                        onclick="dashboardAnalytics.dismissAlert('${alert.id}')"></button>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    // ===== EVENT HANDLERS =====
    async refreshDashboard() {
        console.log('🔄 Refreshing dashboard...');
        
        // Clear cache
        this.cache.clear();
        
        // Show loading indicators
        this.showLoadingIndicators();
        
        try {
            await this.loadDashboardData();
            this.hideLoadingIndicators();
            this.showSuccessMessage('Dashboard refreshed successfully');
        } catch (error) {
            this.hideLoadingIndicators();
            this.showErrorMessage('Failed to refresh dashboard');
        }
    }

    async handleDateRangeChange(range) {
        console.log(`📅 Date range changed to: ${range}`);
        
        // Update analytics based on date range
        this.cache.clear(); // Clear cache when date range changes
        
        try {
            await Promise.allSettled([
                this.loadFinancialTrends(),
                this.loadRiskAnalytics(),
                this.loadPortfolioPerformance()
            ]);
        } catch (error) {
            console.error('❌ Error updating data for date range:', error);
        }
    }

    async handleCompanyFilter(companyId) {
        console.log(`🏢 Company filter changed to: ${companyId}`);
        
        if (companyId === 'all') {
            await this.loadDashboardData();
        } else {
            // Load company-specific dashboard
            await this.loadCompanySpecificDashboard(companyId);
        }
    }

    async loadCompanySpecificDashboard(companyId) {
        try {
            const data = await this.fetchDashboardData('/api/dashboard/company-specific', {
                company_id: companyId
            });
            
            // Update dashboard with company-specific data
            this.updateCompanySpecificMetrics(data);
            
        } catch (error) {
            console.error(`❌ Error loading company-specific dashboard for ${companyId}:`, error);
        }
    }

    async dismissAlert(alertId) {
        try {
            await fetch(`/api/alerts/${alertId}/dismiss`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            // Reload alerts
            await this.loadAlerts();
            
        } catch (error) {
            console.error(`❌ Error dismissing alert ${alertId}:`, error);
        }
    }

    // ===== UTILITY METHODS =====
    updateElement(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
        }
    }

    updateProgressBar(elementId, value, max = 100) {
        const progressBar = document.getElementById(elementId);
        if (progressBar) {
            const percentage = Math.min(100, Math.max(0, (value / max) * 100));
            progressBar.style.width = `${percentage}%`;
            progressBar.setAttribute('aria-valuenow', percentage);
        }
    }

    updateGrowthIndicator(elementId, value) {
        const indicator = document.getElementById(elementId);
        if (indicator) {
            indicator.className = `growth-indicator ${value >= 0 ? 'positive' : 'negative'}`;
            indicator.innerHTML = `
                <i class="fas fa-arrow-${value >= 0 ? 'up' : 'down'}"></i>
                ${Math.abs(value).toFixed(1)}%
            `;
        }
    }

    updatePerformanceIndicator(elementId, value) {
        const indicator = document.getElementById(elementId);
        if (indicator) {
            const isPositive = value >= 0;
            indicator.className = `performance-indicator ${isPositive ? 'positive' : 'negative'}`;
            indicator.innerHTML = `
                <i class="fas fa-${isPositive ? 'trending-up' : 'trending-down'}"></i>
                <span class="text-${isPositive ? 'success' : 'danger'}">${value.toFixed(2)}%</span>
            `;
        }
    }

    updateRiskGauge(elementId, riskScore) {
        if (window.chartConfigurator) {
            window.chartConfigurator.createRiskGauge(elementId, {
                riskScore: riskScore,
                riskLevel: this.getRiskLevel(riskScore)
            });
        }
    }

    updatePerformersList(containerId, performers, isTopPerformers) {
        const container = document.getElementById(containerId);
        if (!container || !performers) return;

        const html = performers.map((company, index) => `
            <div class="performer-item d-flex align-items-center justify-content-between mb-2">
                <div class="d-flex align-items-center">
                    <span class="rank me-2">${index + 1}</span>
                    <div>
                        <div class="company-name fw-bold">${company.company_name}</div>
                        <div class="company-industry text-muted small">${company.industry}</div>
                    </div>
                </div>
                <div class="performance-metric">
                    <span class="badge bg-${isTopPerformers ? 'success' : 'danger'}">
                        ${company.health_score}/100
                    </span>
                </div>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    updateHighRiskCompaniesList(highRiskCompanies) {
        const container = document.getElementById('high-risk-companies-list');
        if (!container || !highRiskCompanies) return;

        const html = highRiskCompanies.map(company => `
            <div class="risk-company-item d-flex align-items-center justify-content-between mb-3">
                <div>
                    <div class="company-name fw-bold">${company.company_name}</div>
                    <div class="risk-factors text-muted small">${company.primary_risk_factors.join(', ')}</div>
                </div>
                <div class="risk-score">
                    <span class="badge bg-danger">${company.risk_score}/100</span>
                </div>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    updateRiskFactorsList(riskFactors) {
        const container = document.getElementById('risk-factors-list');
        if (!container || !riskFactors) return;

        const html = riskFactors.map(factor => `
            <div class="risk-factor-item d-flex align-items-center justify-content-between mb-2">
                <div class="d-flex align-items-center">
                    <i class="fas fa-exclamation-triangle text-warning me-2"></i>
                    <span>${factor.factor_name}</span>
                </div>
                <div>
                    <span class="badge bg-light text-dark">${factor.affected_companies} companies</span>
                </div>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    updateGrowthCompaniesList(growthCompanies) {
        const container = document.getElementById('growth-companies-list');
        if (!container || !growthCompanies) return;

        const html = growthCompanies.map(company => `
            <div class="growth-company-item d-flex align-items-center justify-content-between mb-2">
                <div>
                    <div class="company-name fw-bold">${company.company_name}</div>
                    <div class="company-sector text-muted small">${company.sector}</div>
                </div>
                <div class="growth-rate">
                    <span class="badge bg-success">+${company.growth_rate.toFixed(1)}%</span>
                </div>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    updateContributorsList(containerId, contributors) {
        const container = document.getElementById(containerId);
        if (!container || !contributors) return;

        const html = contributors.map(company => `
            <div class="contributor-item d-flex align-items-center justify-content-between mb-2">
                <div>
                    <div class="company-name fw-bold">${company.company_name}</div>
                    <div class="contribution text-muted small">${company.contribution_type}</div>
                </div>
                <div class="contribution-value">
                    <span class="badge bg-${company.contribution > 0 ? 'success' : 'danger'}">
                        ${company.contribution > 0 ? '+' : ''}${company.contribution.toFixed(2)}%
                    </span>
                </div>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    // ===== HELPER METHODS =====
    formatCurrency(amount, currency = 'USD') {
        if (amount === null || amount === undefined) return '$0';
        
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency,
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(amount);
    }

    formatRelativeTime(timestamp) {
        const now = new Date();
        const time = new Date(timestamp);
        const diffInHours = (now - time) / (1000 * 60 * 60);

        if (diffInHours < 1) {
            return 'Just now';
        } else if (diffInHours < 24) {
            return `${Math.floor(diffInHours)} hours ago`;
        } else if (diffInHours < 168) {
            return `${Math.floor(diffInHours / 24)} days ago`;
        } else {
            return time.toLocaleDateString();
        }
    }

    getRiskLevel(riskScore) {
        if (riskScore < 25) return 'low';
        if (riskScore < 50) return 'moderate';
        if (riskScore < 75) return 'high';
        return 'critical';
    }

    getActivityIcon(activityType) {
        const icons = {
            'analysis': 'fa-chart-line',
            'alert': 'fa-exclamation-triangle',
            'company_added': 'fa-plus-circle',
            'report_generated': 'fa-file-alt',
            'risk_assessment': 'fa-shield-alt',
            'data_update': 'fa-sync-alt'
        };
        return icons[activityType] || 'fa-info-circle';
    }

    getActivityColor(activityType) {
        const colors = {
            'analysis': 'primary',
            'alert': 'warning',
            'company_added': 'success',
            'report_generated': 'info',
            'risk_assessment': 'danger',
            'data_update': 'secondary'
        };
        return colors[activityType] || 'secondary';
    }

    getAlertIcon(alertType) {
        const icons = {
            'liquidity': 'fa-tint',
            'profitability': 'fa-chart-line',
            'cash_flow': 'fa-money-bill-wave',
            'risk': 'fa-exclamation-triangle',
            'compliance': 'fa-gavel'
        };
        return icons[alertType] || 'fa-bell';
    }

    getAlertSeverityClass(severity) {
        const classes = {
            'low': 'info',
            'medium': 'warning',
            'high': 'danger',
            'critical': 'dark'
        };
        return classes[severity] || 'secondary';
    }

    showLoadingIndicators() {
        document.querySelectorAll('.loading-indicator').forEach(indicator => {
            indicator.classList.remove('d-none');
        });
    }

    hideLoadingIndicators() {
        document.querySelectorAll('.loading-indicator').forEach(indicator => {
            indicator.classList.add('d-none');
        });
    }

    showSuccessMessage(message) {
        this.showToast(message, 'success');
    }

    showErrorMessage(message) {
        this.showToast(message, 'danger');
    }

    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toast-container') || this.createToastContainer();
        
        const toastHtml = `
            <div class="toast align-items-center text-white bg-${type} border-0" role="alert">
                <div class="d-flex">
                    <div class="toast-body">${message}</div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        
        const toastElement = toastContainer.lastElementChild;
        const toast = new bootstrap.Toast(toastElement);
        toast.show();
        
        // Remove toast element after it's hidden
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    }

    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
        return container;
    }

    showDashboardError(message) {
        const errorContainer = document.getElementById('dashboard-error-container');
        if (errorContainer) {
            errorContainer.innerHTML = `
                <div class="alert alert-danger alert-dismissible fade show" role="alert">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    <strong>Dashboard Error:</strong> ${message}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            `;
        }
    }

    showMetricsError(containerId, message) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="metrics-error text-center p-4">
                    <i class="fas fa-exclamation-triangle text-warning fa-2x mb-2"></i>
                    <p class="text-muted">${message}</p>
                    <button class="btn btn-sm btn-outline-primary refresh-dashboard">
                        <i class="fas fa-refresh me-1"></i> Retry
                    </button>
                </div>
            `;
        }
    }

    // ===== AUTO REFRESH =====
    startAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }

        this.refreshInterval = setInterval(() => {
            console.log('🔄 Auto-refreshing dashboard...');
            this.loadDashboardData();
        }, this.refreshRate);

        console.log(`⏰ Auto-refresh enabled (${this.refreshRate / 1000}s interval)`);
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
            console.log('⏹️ Auto-refresh disabled');
        }
    }

    setRefreshRate(rate) {
        this.refreshRate = rate;
        this.startAutoRefresh();
        console.log(`⏰ Refresh rate updated to ${rate / 1000}s`);
    }

    // ===== EXPORT FUNCTIONALITY =====
    async exportDashboard(format = 'pdf') {
        try {
            console.log(`📥 Exporting dashboard as ${format}...`);
            
            const exportData = {
                overview: this.analytics.get('overview'),
                company_statistics: this.analytics.get('company_statistics'),
                risk_analytics: this.analytics.get('risk_analytics'),
                financial_trends: this.analytics.get('financial_trends'),
                portfolio_performance: this.analytics.get('portfolio_performance'),
                export_timestamp: new Date().toISOString()
            };

            const response = await fetch('/api/dashboard/export', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    format: format,
                    data: exportData
                })
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `dashboard-report-${new Date().toISOString().split('T')[0]}.${format}`;
                link.click();
                window.URL.revokeObjectURL(url);
                
                this.showSuccessMessage(`Dashboard exported as ${format.toUpperCase()}`);
            } else {
                throw new Error(`Export failed: ${response.statusText}`);
            }

        } catch (error) {
            console.error('❌ Error exporting dashboard:', error);
            this.showErrorMessage('Failed to export dashboard');
        }
    }

    // ===== PUBLIC API =====
    getAnalytics(key) {
        return this.analytics.get(key);
    }

    getAllAnalytics() {
        return Object.fromEntries(this.analytics);
    }

    clearCache() {
        this.cache.clear();
        console.log('🗑️ Dashboard cache cleared');
    }

    destroy() {
        this.stopAutoRefresh();
        this.clearCache();
        this.analytics.clear();
        console.log('🗑️ Dashboard Analytics destroyed');
    }
}

// Global instance
window.dashboardAnalytics = new DashboardAnalytics();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DashboardAnalytics;
}