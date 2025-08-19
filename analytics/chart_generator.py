import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import json

class ChartGenerator:
    """
    Chart and visualization generator for financial data
    
    Creates interactive charts using Plotly for the Financial Risk Assessment Platform.
    Generates various chart types including line charts, bar charts, radar charts,
    and custom financial visualizations.
    """
    
    def __init__(self):
        """Initialize Chart Generator with default styling"""
        self.default_colors = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
            '#9b59b6', '#34495e', '#16a085', '#e67e22'
        ]
        
        self.chart_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'responsive': True
        }
        
        # Chart styling templates
        self.layout_template = {
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
            'showlegend': True,
            'legend': {
                'orientation': 'h',
                'yanchor': 'top',
                'y': -0.1,
                'xanchor': 'center',
                'x': 0.5
            }
        }

    def create_financial_summary_chart(self, financial_data):
        """
        Create financial summary overview chart
        
        Args:
            financial_data (dict): Financial data with balance sheets and cash flows
            
        Returns:
            dict: Plotly chart JSON
        """
        try:
            balance_sheets = financial_data.get('balance_sheets', [])
            cash_flows = financial_data.get('cash_flows', [])
            
            if not balance_sheets or not cash_flows:
                return self.create_error_chart("Insufficient data for financial summary")
            
            # Prepare data
            years = [bs['year'] for bs in balance_sheets]
            total_assets = [bs.get('total_assets', 0) or 0 for bs in balance_sheets]
            total_equity = [bs.get('total_equity', 0) or 0 for bs in balance_sheets]
            net_income = [cf.get('net_income', 0) or 0 for cf in cash_flows]
            
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]],
                subplot_titles=["Financial Summary Overview"]
            )
            
            # Add assets and equity bars
            fig.add_trace(
                go.Bar(
                    x=years,
                    y=total_assets,
                    name='Total Assets',
                    marker_color=self.default_colors[0],
                    yaxis='y'
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Bar(
                    x=years,
                    y=total_equity,
                    name='Total Equity',
                    marker_color=self.default_colors[1],
                    yaxis='y'
                ),
                secondary_y=False
            )
            
            # Add net income line
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=net_income,
                    mode='lines+markers',
                    name='Net Income',
                    line=dict(color=self.default_colors[2], width=3),
                    yaxis='y2'
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title="Financial Performance Overview",
                **self.layout_template
            )
            
            fig.update_yaxes(title_text="Assets & Equity ($)", secondary_y=False)
            fig.update_yaxes(title_text="Net Income ($)", secondary_y=True)
            fig.update_xaxes(title_text="Year")
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Error creating financial summary chart: {e}")
            return self.create_error_chart("Error creating financial summary chart")

    def create_cash_flow_analysis_chart(self, cash_flows):
        """
        Create cash flow analysis chart
        
        Args:
            cash_flows (list): Cash flow data
            
        Returns:
            dict: Plotly chart JSON
        """
        try:
            if not cash_flows:
                return self.create_error_chart("No cash flow data available")
            
            years = [cf['year'] for cf in cash_flows]
            operating_cf = [cf.get('net_cash_from_operating_activities', 0) or 0 for cf in cash_flows]
            investing_cf = [cf.get('net_cash_from_investing_activities', 0) or 0 for cf in cash_flows]
            financing_cf = [cf.get('net_cash_from_financing_activities', 0) or 0 for cf in cash_flows]
            free_cf = [cf.get('free_cash_flow', 0) or 0 for cf in cash_flows]
            
            # Create stacked bar chart for cash flow components
            fig = go.Figure()
            
            # Add cash flow components
            fig.add_trace(go.Bar(
                x=years,
                y=operating_cf,
                name='Operating Cash Flow',
                marker_color=self.default_colors[0]
            ))
            
            fig.add_trace(go.Bar(
                x=years,
                y=investing_cf,
                name='Investing Cash Flow',
                marker_color=self.default_colors[1]
            ))
            
            fig.add_trace(go.Bar(
                x=years,
                y=financing_cf,
                name='Financing Cash Flow',
                marker_color=self.default_colors[2]
            ))
            
            # Add free cash flow line
            fig.add_trace(go.Scatter(
                x=years,
                y=free_cf,
                mode='lines+markers',
                name='Free Cash Flow',
                line=dict(color=self.default_colors[3], width=3),
                yaxis='y2'
            ))
            
            # Update layout
            fig.update_layout(
                title="Cash Flow Analysis",
                xaxis_title="Year",
                yaxis_title="Cash Flow ($)",
                barmode='relative',
                **self.layout_template
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Error creating cash flow chart: {e}")
            return self.create_error_chart("Error creating cash flow chart")

    def create_financial_ratios_radar(self, ratios):
        """
        Create radar chart for financial ratios
        
        Args:
            ratios (dict): Financial ratios
            
        Returns:
            dict: Plotly chart JSON
        """
        try:
            # Define ratio categories and their values
            categories = [
                'Current Ratio',
                'Quick Ratio', 
                'ROA',
                'ROE',
                'Debt/Equity',
                'Profit Margin'
            ]
            
            # Get values and normalize them for radar chart
            values = [
                min(ratios.get('current_ratio', 0), 3),  # Cap at 3 for better visualization
                min(ratios.get('quick_ratio', 0), 3),
                ratios.get('roa', 0) * 10,  # Scale up for visibility
                ratios.get('roe', 0) * 10,
                max(0, 3 - ratios.get('debt_to_equity', 0)),  # Invert (lower debt = better)
                ratios.get('profit_margin', 0) * 10
            ]
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Current Performance',
                line=dict(color=self.default_colors[0])
            ))
            
            # Add industry benchmark (example values)
            benchmark_values = [2.0, 1.5, 1.0, 1.2, 2.0, 1.5]  # Example benchmarks
            
            fig.add_trace(go.Scatterpolar(
                r=benchmark_values,
                theta=categories,
                fill='toself',
                name='Industry Benchmark',
                line=dict(color=self.default_colors[1], dash='dash'),
                opacity=0.6
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 3]
                    )
                ),
                title="Financial Ratios Analysis",
                **self.layout_template
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Error creating radar chart: {e}")
            return self.create_error_chart("Error creating radar chart")

    def create_trend_analysis_chart(self, trends_data):
        """
        Create trend analysis chart
        
        Args:
            trends_data (dict): Trend analysis data
            
        Returns:
            dict: Plotly chart JSON
        """
        try:
            # Create trend indicators chart
            categories = []
            scores = []
            colors = []
            
            # Revenue trend
            revenue_trend = trends_data.get('revenue_trend', {})
            if revenue_trend:
                categories.append('Revenue Trend')
                direction = revenue_trend.get('direction', 'stable')
                if direction == 'increasing':
                    scores.append(3)
                    colors.append('#2ecc71')
                elif direction == 'stable':
                    scores.append(2)
                    colors.append('#f39c12')
                else:
                    scores.append(1)
                    colors.append('#e74c3c')
            
            # Asset growth
            asset_trend = trends_data.get('asset_growth', {})
            if asset_trend:
                categories.append('Asset Growth')
                direction = asset_trend.get('direction', 'stable')
                if direction == 'growing':
                    scores.append(3)
                    colors.append('#2ecc71')
                elif direction == 'stable':
                    scores.append(2)
                    colors.append('#f39c12')
                else:
                    scores.append(1)
                    colors.append('#e74c3c')
            
            # Cash flow trend
            cf_trend = trends_data.get('cash_flow_trend', {})
            if cf_trend:
                categories.append('Cash Flow Trend')
                direction = cf_trend.get('direction', 'stable')
                if direction == 'improving':
                    scores.append(3)
                    colors.append('#2ecc71')
                elif direction == 'stable':
                    scores.append(2)
                    colors.append('#f39c12')
                else:
                    scores.append(1)
                    colors.append('#e74c3c')
            
            # Debt trend
            debt_trend = trends_data.get('debt_trend', {})
            if debt_trend:
                categories.append('Debt Management')
                direction = debt_trend.get('direction', 'stable')
                if direction == 'decreasing':
                    scores.append(3)
                    colors.append('#2ecc71')
                elif direction == 'stable':
                    scores.append(2)
                    colors.append('#f39c12')
                else:
                    scores.append(1)
                    colors.append('#e74c3c')
            
            if not categories:
                return self.create_error_chart("No trend data available")
            
            # Create horizontal bar chart
            fig = go.Figure(go.Bar(
                x=scores,
                y=categories,
                orientation='h',
                marker_color=colors,
                text=[f"Score: {score}/3" for score in scores],
                textposition='inside'
            ))
            
            fig.update_layout(
                title="Financial Trends Analysis",
                xaxis_title="Trend Score (1=Poor, 2=Stable, 3=Good)",
                yaxis_title="Financial Metrics",
                xaxis=dict(range=[0, 3]),
                **self.layout_template
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Error creating trend chart: {e}")
            return self.create_error_chart("Error creating trend chart")

    def create_risk_assessment_chart(self, risk_factors):
        """
        Create risk assessment visualization
        
        Args:
            risk_factors (list): List of risk factors
            
        Returns:
            dict: Plotly chart JSON
        """
        try:
            if not risk_factors:
                return self.create_no_risk_chart()
            
            # Categorize risks by severity
            severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
            risk_types = {}
            
            for risk in risk_factors:
                severity = risk.get('severity', 'medium')
                risk_type = risk.get('type', 'unknown')
                
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                risk_types[risk_type] = risk_types.get(risk_type, 0) + 1
            
            # Create donut chart for risk severity
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "pie"}, {"type": "pie"}]],
                subplot_titles=["Risk Severity", "Risk Types"]
            )
            
            # Risk severity donut
            severity_labels = []
            severity_values = []
            severity_colors = []
            
            color_map = {
                'low': '#2ecc71',
                'medium': '#f39c12', 
                'high': '#e67e22',
                'critical': '#e74c3c'
            }
            
            for severity, count in severity_counts.items():
                if count > 0:
                    severity_labels.append(severity.title())
                    severity_values.append(count)
                    severity_colors.append(color_map[severity])
            
            fig.add_trace(go.Pie(
                labels=severity_labels,
                values=severity_values,
                marker_colors=severity_colors,
                hole=0.4,
                name="Risk Severity"
            ), row=1, col=1)
            
            # Risk types donut
            type_labels = list(risk_types.keys())
            type_values = list(risk_types.values())
            
            fig.add_trace(go.Pie(
                labels=[t.title() for t in type_labels],
                values=type_values,
                hole=0.4,
                name="Risk Types"
            ), row=1, col=2)
            
            fig.update_layout(
                title="Risk Assessment Overview",
                **self.layout_template
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Error creating risk chart: {e}")
            return self.create_error_chart("Error creating risk assessment chart")

    def create_industry_comparison_chart(self, company_data, industry_benchmarks):
        """
        Create industry comparison chart
        
        Args:
            company_data (dict): Company financial data
            industry_benchmarks (dict): Industry benchmark data
            
        Returns:
            dict: Plotly chart JSON
        """
        try:
            metrics = ['Current Ratio', 'Debt/Equity', 'ROA', 'Profit Margin']
            company_values = [
                company_data.get('current_ratio', 0),
                company_data.get('debt_to_equity', 0),
                company_data.get('roa', 0) * 100,  # Convert to percentage
                company_data.get('profit_margin', 0) * 100
            ]
            
            industry_values = [
                industry_benchmarks.get('median_current_ratio', 1.5),
                industry_benchmarks.get('median_debt_to_equity', 0.4),
                industry_benchmarks.get('median_roa', 0.08) * 100,
                industry_benchmarks.get('median_profit_margin', 0.1) * 100
            ]
            
            fig = go.Figure()
            
            # Company bars
            fig.add_trace(go.Bar(
                x=metrics,
                y=company_values,
                name='Company',
                marker_color=self.default_colors[0]
            ))
            
            # Industry benchmark bars
            fig.add_trace(go.Bar(
                x=metrics,
                y=industry_values,
                name='Industry Median',
                marker_color=self.default_colors[1]
            ))
            
            fig.update_layout(
                title="Company vs Industry Benchmarks",
                xaxis_title="Financial Metrics",
                yaxis_title="Values",
                barmode='group',
                **self.layout_template
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Error creating comparison chart: {e}")
            return self.create_error_chart("Error creating industry comparison chart")

    def create_error_chart(self, message):
        """Create error message chart"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Chart Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            **self.layout_template
        )
        return json.loads(fig.to_json())

    def create_no_risk_chart(self):
        """Create chart when no risks are identified"""
        fig = go.Figure()
        fig.add_annotation(
            text="✓ No significant risks identified",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=18, color="green")
        )
        fig.update_layout(
            title="Risk Assessment - All Clear",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            **self.layout_template
        )
        return json.loads(fig.to_json())

    def generate_company_charts(self, financial_data):
        """
        Generate all charts for a company dashboard
        
        Args:
            financial_data (dict): Complete financial data
            
        Returns:
            dict: Collection of charts
        """
        try:
            charts = {}
            
            # Financial summary chart
            charts['financial_summary'] = self.create_financial_summary_chart(financial_data)
            
            # Cash flow analysis
            cash_flows = financial_data.get('cash_flows', [])
            if cash_flows:
                charts['cash_flow_analysis'] = self.create_cash_flow_analysis_chart(cash_flows)
            
            # Financial ratios radar
            ratios = financial_data.get('ratios', {})
            if ratios:
                charts['financial_ratios'] = self.create_financial_ratios_radar(ratios)
            
            # Trend analysis
            trends = financial_data.get('trends', {})
            if trends:
                charts['trend_analysis'] = self.create_trend_analysis_chart(trends)
            
            # Risk assessment
            risk_factors = financial_data.get('risk_factors', [])
            charts['risk_assessment'] = self.create_risk_assessment_chart(risk_factors)
            
            return charts
            
        except Exception as e:
            print(f"Error generating company charts: {e}")
            return {'error': 'Failed to generate charts'}

    def generate_industry_charts(self, industry_data):
        """
        Generate charts for industry analysis
        
        Args:
            industry_data (dict): Industry analysis data
            
        Returns:
            dict: Collection of industry charts
        """
        try:
            charts = {}
            
            # Industry performance distribution
            charts['performance_distribution'] = self.create_industry_performance_chart(industry_data)
            
            # Industry trends
            charts['industry_trends'] = self.create_industry_trends_chart(industry_data)
            
            # Risk factors by industry
            charts['industry_risks'] = self.create_industry_risk_chart(industry_data)
            
            return charts
            
        except Exception as e:
            print(f"Error generating industry charts: {e}")
            return {'error': 'Failed to generate industry charts'}

    def create_industry_performance_chart(self, industry_data):
        """Create industry performance distribution chart"""
        try:
            # Sample data - in real implementation, this would use actual industry data
            companies = industry_data.get('companies', [])
            if not companies:
                return self.create_error_chart("No industry data available")
            
            # Extract performance metrics
            performance_scores = [comp.get('health_score', 50) for comp in companies]
            company_names = [comp.get('name', f'Company {i}') for i, comp in enumerate(companies)]
            
            # Create histogram of performance scores
            fig = go.Figure(data=[go.Histogram(
                x=performance_scores,
                nbinsx=20,
                marker_color=self.default_colors[0],
                opacity=0.7
            )])
            
            fig.update_layout(
                title="Industry Performance Distribution",
                xaxis_title="Financial Health Score",
                yaxis_title="Number of Companies",
                **self.layout_template
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Error creating industry performance chart: {e}")
            return self.create_error_chart("Error creating industry performance chart")

    def create_industry_trends_chart(self, industry_data):
        """Create industry trends chart"""
        try:
            # Sample trend data
            quarters = ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024']
            avg_revenue_growth = [2.1, 3.2, 2.8, 4.1, 3.5]
            avg_profit_margin = [8.2, 8.5, 8.1, 8.8, 9.1]
            
            fig = make_subplots(
                specs=[[{"secondary_y": True}]],
                subplot_titles=["Industry Trends"]
            )
            
            # Revenue growth line
            fig.add_trace(
                go.Scatter(
                    x=quarters,
                    y=avg_revenue_growth,
                    name="Avg Revenue Growth (%)",
                    line=dict(color=self.default_colors[0], width=3)
                ),
                secondary_y=False
            )
            
            # Profit margin line
            fig.add_trace(
                go.Scatter(
                    x=quarters,
                    y=avg_profit_margin,
                    name="Avg Profit Margin (%)",
                    line=dict(color=self.default_colors[1], width=3)
                ),
                secondary_y=True
            )
            
            fig.update_layout(
                title="Industry Financial Trends",
                **self.layout_template
            )
            
            fig.update_yaxes(title_text="Revenue Growth (%)", secondary_y=False)
            fig.update_yaxes(title_text="Profit Margin (%)", secondary_y=True)
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Error creating industry trends chart: {e}")
            return self.create_error_chart("Error creating industry trends chart")

    def create_industry_risk_chart(self, industry_data):
        """Create industry risk analysis chart"""
        try:
            # Sample risk data
            risk_categories = ['Market Risk', 'Credit Risk', 'Operational Risk', 'Regulatory Risk', 'Liquidity Risk']
            risk_levels = [65, 45, 30, 55, 25]  # Risk scores out of 100
            
            # Color coding based on risk level
            colors = []
            for level in risk_levels:
                if level >= 70:
                    colors.append('#e74c3c')  # High risk - red
                elif level >= 50:
                    colors.append('#f39c12')  # Medium risk - orange
                else:
                    colors.append('#2ecc71')  # Low risk - green
            
            fig = go.Figure(go.Bar(
                x=risk_categories,
                y=risk_levels,
                marker_color=colors,
                text=[f"{level}%" for level in risk_levels],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Industry Risk Assessment",
                xaxis_title="Risk Categories",
                yaxis_title="Risk Level (%)",
                yaxis=dict(range=[0, 100]),
                **self.layout_template
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Error creating industry risk chart: {e}")
            return self.create_error_chart("Error creating industry risk chart")

    def create_waterfall_chart(self, data, title="Waterfall Analysis"):
        """
        Create waterfall chart for financial analysis
        
        Args:
            data (dict): Data with 'categories' and 'values' lists
            title (str): Chart title
            
        Returns:
            dict: Plotly chart JSON
        """
        try:
            categories = data.get('categories', [])
            values = data.get('values', [])
            
            if len(categories) != len(values):
                return self.create_error_chart("Invalid waterfall data")
            
            # Calculate cumulative values for waterfall effect
            cumulative = [0]
            for i, val in enumerate(values[:-1]):
                cumulative.append(cumulative[-1] + val)
            
            fig = go.Figure()
            
            # Add bars for each category
            for i, (cat, val) in enumerate(zip(categories, values)):
                if i == 0:  # Starting value
                    fig.add_trace(go.Bar(
                        x=[cat],
                        y=[val],
                        name=cat,
                        marker_color=self.default_colors[0]
                    ))
                elif i == len(categories) - 1:  # Ending value
                    fig.add_trace(go.Bar(
                        x=[cat],
                        y=[cumulative[i] + val],
                        name=cat,
                        marker_color=self.default_colors[1]
                    ))
                else:  # Intermediate values
                    color = self.default_colors[2] if val >= 0 else self.default_colors[3]
                    fig.add_trace(go.Bar(
                        x=[cat],
                        y=[val],
                        base=cumulative[i],
                        name=cat,
                        marker_color=color
                    ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Categories",
                yaxis_title="Value",
                showlegend=False,
                **self.layout_template
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Error creating waterfall chart: {e}")
            return self.create_error_chart("Error creating waterfall chart")

    def create_heatmap(self, data, title="Financial Metrics Heatmap"):
        """
        Create heatmap for financial metrics comparison
        
        Args:
            data (dict): Data with 'x', 'y', and 'z' values
            title (str): Chart title
            
        Returns:
            dict: Plotly chart JSON
        """
        try:
            x_labels = data.get('x', [])
            y_labels = data.get('y', [])
            z_values = data.get('z', [])
            
            fig = go.Figure(data=go.Heatmap(
                z=z_values,
                x=x_labels,
                y=y_labels,
                colorscale='RdYlGn',
                reversescale=True,
                showscale=True
            ))
            
            fig.update_layout(
                title=title,
                **self.layout_template
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            return self.create_error_chart("Error creating heatmap")

    def create_gauge_chart(self, value, title="Financial Health Score", max_value=100):
        """
        Create gauge chart for financial health score
        
        Args:
            value (float): Current value
            title (str): Chart title
            max_value (float): Maximum value for gauge
            
        Returns:
            dict: Plotly chart JSON
        """
        try:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': title},
                delta={'reference': max_value * 0.7},  # Reference point at 70%
                gauge={
                    'axis': {'range': [None, max_value]},
                    'bar': {'color': self.default_colors[0]},
                    'steps': [
                        {'range': [0, max_value * 0.4], 'color': "lightgray"},
                        {'range': [max_value * 0.4, max_value * 0.7], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': max_value * 0.9
                    }
                }
            ))
            
            fig.update_layout(
                **self.layout_template,
                height=400
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"Error creating gauge chart: {e}")
            return self.create_error_chart("Error creating gauge chart")