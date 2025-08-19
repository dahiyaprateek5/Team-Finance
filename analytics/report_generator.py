import pandas as pd
import numpy as np
from datetime import datetime
import json
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAnalyzer:
    """Real Database Financial Analyzer"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None
    
    def get_db_connection(self):
        """Get database connection"""
        try:
            if not self.connection or self.connection.closed:
                self.connection = psycopg2.connect(**self.db_config)
            return self.connection
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def fetch_company_data(self, company_id):
        """Fetch company data from database"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Fetch balance sheet data
            cursor.execute("""
                SELECT * FROM balance_sheet_1 
                WHERE company_id = %s 
                ORDER BY year DESC 
                LIMIT 3
            """, (company_id,))
            balance_sheets = cursor.fetchall()
            
            # Fetch cash flow data
            cursor.execute("""
                SELECT * FROM cash_flow_statement 
                WHERE company_id = %s 
                ORDER BY year DESC 
                LIMIT 3
            """, (company_id,))
            cash_flows = cursor.fetchall()
            
            # Fetch income statement data
            cursor.execute("""
                SELECT * FROM income_statement 
                WHERE company_id = %s 
                ORDER BY year DESC 
                LIMIT 3
            """, (company_id,))
            income_statements = cursor.fetchall()
            
            cursor.close()
            
            return {
                'balance_sheets': balance_sheets,
                'cash_flows': cash_flows,
                'income_statements': income_statements
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for company {company_id}: {e}")
            return None
    
    def calculate_financial_ratios(self, company_data):
        """Calculate financial ratios from real data"""
        try:
            if not company_data['balance_sheets']:
                return None
                
            latest_bs = company_data['balance_sheets'][0]
            latest_income = company_data['income_statements'][0] if company_data['income_statements'] else {}
            latest_cf = company_data['cash_flows'][0] if company_data['cash_flows'] else {}
            
            # Liquidity ratios
            current_assets = float(latest_bs.get('current_assets', 0) or 0)
            current_liabilities = float(latest_bs.get('current_liabilities', 0) or 0)
            cash = float(latest_bs.get('cash_and_equivalents', 0) or 0)
            inventory = float(latest_bs.get('inventory', 0) or 0)
            
            current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0
            quick_ratio = (current_assets - inventory) / current_liabilities if current_liabilities > 0 else 0
            cash_ratio = cash / current_liabilities if current_liabilities > 0 else 0
            
            # Profitability ratios
            net_income = float(latest_income.get('net_income', 0) or 0)
            total_assets = float(latest_bs.get('total_assets', 0) or 0)
            total_equity = float(latest_bs.get('total_equity', 0) or 0)
            revenue = float(latest_income.get('revenue', 0) or 0)
            
            roa = net_income / total_assets if total_assets > 0 else 0
            roe = net_income / total_equity if total_equity > 0 else 0
            profit_margin = net_income / revenue if revenue > 0 else 0
            
            # Leverage ratios
            total_debt = float(latest_bs.get('total_debt', 0) or 0)
            interest_expense = float(latest_income.get('interest_expense', 0) or 0)
            ebit = net_income + interest_expense + float(latest_income.get('tax_expense', 0) or 0)
            
            debt_to_equity = total_debt / total_equity if total_equity > 0 else 0
            debt_ratio = total_debt / total_assets if total_assets > 0 else 0
            interest_coverage = ebit / interest_expense if interest_expense > 0 else 0
            
            # Efficiency ratios
            asset_turnover = revenue / total_assets if total_assets > 0 else 0
            cogs = float(latest_income.get('cost_of_goods_sold', 0) or 0)
            accounts_receivable = float(latest_bs.get('accounts_receivable', 0) or 0)
            
            inventory_turnover = cogs / inventory if inventory > 0 else 0
            receivables_turnover = revenue / accounts_receivable if accounts_receivable > 0 else 0
            
            return {
                'liquidity': {
                    'current_ratio': round(current_ratio, 2),
                    'quick_ratio': round(quick_ratio, 2),
                    'cash_ratio': round(cash_ratio, 2)
                },
                'profitability': {
                    'roa': round(roa, 4),
                    'roe': round(roe, 4),
                    'profit_margin': round(profit_margin, 4)
                },
                'leverage': {
                    'debt_to_equity': round(debt_to_equity, 2),
                    'debt_ratio': round(debt_ratio, 2),
                    'interest_coverage': round(interest_coverage, 2)
                },
                'efficiency': {
                    'asset_turnover': round(asset_turnover, 2),
                    'inventory_turnover': round(inventory_turnover, 2),
                    'receivables_turnover': round(receivables_turnover, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating ratios: {e}")
            return None
    
    def assess_financial_health(self, ratios):
        """Assess overall financial health from ratios"""
        try:
            if not ratios:
                return None
                
            scores = {}
            
            # Liquidity score
            liquidity_score = 0
            if ratios['liquidity']['current_ratio'] >= 2.0:
                liquidity_score += 30
            elif ratios['liquidity']['current_ratio'] >= 1.5:
                liquidity_score += 20
            elif ratios['liquidity']['current_ratio'] >= 1.0:
                liquidity_score += 10
            
            if ratios['liquidity']['quick_ratio'] >= 1.0:
                liquidity_score += 20
            elif ratios['liquidity']['quick_ratio'] >= 0.5:
                liquidity_score += 10
            
            scores['liquidity'] = min(liquidity_score, 100)
            
            # Profitability score
            profitability_score = 0
            if ratios['profitability']['roa'] >= 0.15:
                profitability_score += 40
            elif ratios['profitability']['roa'] >= 0.10:
                profitability_score += 30
            elif ratios['profitability']['roa'] >= 0.05:
                profitability_score += 20
            elif ratios['profitability']['roa'] > 0:
                profitability_score += 10
            
            if ratios['profitability']['profit_margin'] >= 0.15:
                profitability_score += 30
            elif ratios['profitability']['profit_margin'] >= 0.10:
                profitability_score += 20
            elif ratios['profitability']['profit_margin'] >= 0.05:
                profitability_score += 10
            
            scores['profitability'] = min(profitability_score, 100)
            
            # Leverage score (lower is better)
            leverage_score = 100
            if ratios['leverage']['debt_to_equity'] > 2.0:
                leverage_score -= 40
            elif ratios['leverage']['debt_to_equity'] > 1.0:
                leverage_score -= 20
            elif ratios['leverage']['debt_to_equity'] > 0.5:
                leverage_score -= 10
            
            if ratios['leverage']['interest_coverage'] < 2.0:
                leverage_score -= 30
            elif ratios['leverage']['interest_coverage'] < 5.0:
                leverage_score -= 15
            
            scores['leverage'] = max(leverage_score, 0)
            
            # Efficiency score
            efficiency_score = 0
            if ratios['efficiency']['asset_turnover'] >= 1.0:
                efficiency_score += 30
            elif ratios['efficiency']['asset_turnover'] >= 0.5:
                efficiency_score += 20
            elif ratios['efficiency']['asset_turnover'] >= 0.25:
                efficiency_score += 10
            
            if ratios['efficiency']['inventory_turnover'] >= 6.0:
                efficiency_score += 25
            elif ratios['efficiency']['inventory_turnover'] >= 4.0:
                efficiency_score += 15
            elif ratios['efficiency']['inventory_turnover'] >= 2.0:
                efficiency_score += 10
            
            scores['efficiency'] = min(efficiency_score, 100)
            
            # Overall score
            overall_score = sum(scores.values()) / len(scores)
            
            # Rating
            if overall_score >= 85:
                rating = 'Excellent'
            elif overall_score >= 70:
                rating = 'Good'
            elif overall_score >= 55:
                rating = 'Fair'
            elif overall_score >= 40:
                rating = 'Poor'
            else:
                rating = 'Critical'
            
            return {
                'overall_score': round(overall_score, 1),
                'overall_rating': rating,
                'category_scores': scores
            }
            
        except Exception as e:
            logger.error(f"Error assessing financial health: {e}")
            return None
    
    def analyze_company_performance(self, company_id):
        """Analyze company financial performance from database"""
        try:
            logger.info(f"Analyzing company {company_id} from database")
            
            # Fetch real data
            company_data = self.fetch_company_data(company_id)
            if not company_data:
                return {'error': 'Company data not found'}
            
            # Calculate ratios
            ratios = self.calculate_financial_ratios(company_data)
            if not ratios:
                return {'error': 'Unable to calculate financial ratios'}
            
            # Assess health
            health_assessment = self.assess_financial_health(ratios)
            if not health_assessment:
                return {'error': 'Unable to assess financial health'}
            
            # Risk assessment
            risk_assessment = self.assess_risk_levels(ratios, health_assessment)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(ratios, health_assessment)
            
            return {
                'company_id': company_id,
                'analysis_date': datetime.now().isoformat(),
                'summary': health_assessment,
                'ratios': ratios,
                'risk_assessment': risk_assessment,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing company {company_id}: {e}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    def assess_risk_levels(self, ratios, health_assessment):
        """Assess risk levels based on financial ratios"""
        try:
            risk_factors = []
            risk_levels = {}
            
            # Liquidity risk
            if ratios['liquidity']['current_ratio'] < 1.0:
                risk_levels['liquidity_risk'] = 'high'
                risk_factors.append('Current ratio below 1.0 indicates liquidity stress')
            elif ratios['liquidity']['current_ratio'] < 1.5:
                risk_levels['liquidity_risk'] = 'medium'
                risk_factors.append('Current ratio below 1.5 may indicate tight liquidity')
            else:
                risk_levels['liquidity_risk'] = 'low'
            
            # Solvency risk
            if ratios['leverage']['debt_to_equity'] > 2.0:
                risk_levels['solvency_risk'] = 'high'
                risk_factors.append('High debt-to-equity ratio indicates solvency risk')
            elif ratios['leverage']['debt_to_equity'] > 1.0:
                risk_levels['solvency_risk'] = 'medium'
                risk_factors.append('Moderate debt levels require monitoring')
            else:
                risk_levels['solvency_risk'] = 'low'
            
            # Profitability risk
            if ratios['profitability']['roa'] < 0:
                risk_levels['profitability_risk'] = 'high'
                risk_factors.append('Negative return on assets indicates profitability issues')
            elif ratios['profitability']['roa'] < 0.05:
                risk_levels['profitability_risk'] = 'medium'
                risk_factors.append('Low return on assets may impact future growth')
            else:
                risk_levels['profitability_risk'] = 'low'
            
            # Operational risk
            if ratios['efficiency']['asset_turnover'] < 0.25:
                risk_levels['operational_risk'] = 'high'
                risk_factors.append('Low asset turnover indicates operational inefficiency')
            elif ratios['efficiency']['asset_turnover'] < 0.5:
                risk_levels['operational_risk'] = 'medium'
                risk_factors.append('Asset utilization could be improved')
            else:
                risk_levels['operational_risk'] = 'low'
            
            # Overall risk
            high_risks = sum(1 for level in risk_levels.values() if level == 'high')
            medium_risks = sum(1 for level in risk_levels.values() if level == 'medium')
            
            if high_risks >= 2:
                risk_levels['overall_risk'] = 'high'
            elif high_risks >= 1 or medium_risks >= 2:
                risk_levels['overall_risk'] = 'medium'
            else:
                risk_levels['overall_risk'] = 'low'
            
            # Risk score (0-100, higher is worse)
            risk_score = 100 - health_assessment['overall_score']
            
            return {
                'risk_score': round(risk_score, 1),
                'risk_levels': risk_levels,
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {
                'risk_score': 50,
                'risk_levels': {'overall_risk': 'medium'},
                'risk_factors': ['Unable to assess risk accurately']
            }
    
    def generate_recommendations(self, ratios, health_assessment):
        """Generate recommendations based on analysis"""
        try:
            recommendations = []
            
            # Liquidity recommendations
            if ratios['liquidity']['current_ratio'] < 1.5:
                recommendations.append({
                    'category': 'liquidity',
                    'priority': 'High' if ratios['liquidity']['current_ratio'] < 1.0 else 'Medium',
                    'recommendation': 'Improve working capital management and cash flow',
                    'impact': 'Better short-term financial stability'
                })
            
            # Profitability recommendations
            if ratios['profitability']['roa'] < 0.05:
                recommendations.append({
                    'category': 'profitability',
                    'priority': 'High' if ratios['profitability']['roa'] < 0 else 'Medium',
                    'recommendation': 'Focus on cost reduction and revenue optimization',
                    'impact': 'Improved profitability and return on assets'
                })
            
            # Leverage recommendations
            if ratios['leverage']['debt_to_equity'] > 1.0:
                recommendations.append({
                    'category': 'leverage',
                    'priority': 'High' if ratios['leverage']['debt_to_equity'] > 2.0 else 'Medium',
                    'recommendation': 'Reduce debt burden and improve debt management',
                    'impact': 'Lower financial risk and interest costs'
                })
            
            # Efficiency recommendations
            if ratios['efficiency']['asset_turnover'] < 0.5:
                recommendations.append({
                    'category': 'efficiency',
                    'priority': 'Medium',
                    'recommendation': 'Improve asset utilization and operational efficiency',
                    'impact': 'Better resource utilization and revenue generation'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def predict_financial_health(self, company_id, months=12):
        """Predict future financial health based on trends"""
        try:
            company_data = self.fetch_company_data(company_id)
            if not company_data or len(company_data['balance_sheets']) < 2:
                return {'error': 'Insufficient data for prediction'}
            
            # Calculate trend from last 2 years
            current_data = company_data['balance_sheets'][0]
            previous_data = company_data['balance_sheets'][1]
            
            current_ratios = self.calculate_financial_ratios({'balance_sheets': [current_data], 'income_statements': company_data['income_statements'][:1], 'cash_flows': company_data['cash_flows'][:1]})
            previous_ratios = self.calculate_financial_ratios({'balance_sheets': [previous_data], 'income_statements': company_data['income_statements'][1:2], 'cash_flows': company_data['cash_flows'][1:2]})
            
            if not current_ratios or not previous_ratios:
                return {'error': 'Unable to calculate ratios for prediction'}
            
            current_health = self.assess_financial_health(current_ratios)
            previous_health = self.assess_financial_health(previous_ratios)
            
            if not current_health or not previous_health:
                return {'error': 'Unable to assess health for prediction'}
            
            # Simple trend analysis
            trend_change = current_health['overall_score'] - previous_health['overall_score']
            predicted_score = current_health['overall_score'] + (trend_change * 0.5)  # Conservative prediction
            
            # Determine trend
            if trend_change > 5:
                trend = 'improving'
                confidence = 'High'
            elif trend_change < -5:
                trend = 'declining'
                confidence = 'High'
            else:
                trend = 'stable'
                confidence = 'Medium'
            
            return {
                'current_health_score': current_health['overall_score'],
                'predicted_health_score': round(max(0, min(100, predicted_score)), 1),
                'confidence_level': confidence,
                'prediction_horizon_months': months,
                'trend': trend
            }
            
        except Exception as e:
            logger.error(f"Error predicting for company {company_id}: {e}")
            return {'error': f'Prediction failed: {str(e)}'}

class PDFReportGenerator:
    """
    PDF-focused Financial Report Generator
    Generates comprehensive financial reports in PDF format only
    """
    
    def __init__(self, db_config=None):
        """
        Initialize PDF Report Generator
        
        Args:
            db_config (dict): Database configuration (optional for demo)
        """
        self.db_config = db_config or {}
        self.analyzer = FinancialAnalyzer(db_config)
        
        # PDF styling configuration
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Setup custom styles for PDF reports"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1f4e79'),
            alignment=1  # Center alignment
        )
        
        # Header style
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#2e75b6')
        )
        
        # Subheader style
        self.subheader_style = ParagraphStyle(
            'CustomSubHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#70ad47')
        )
        
        # Executive summary style
        self.exec_style = ParagraphStyle(
            'ExecutiveStyle',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            backgroundColor=colors.HexColor('#f0f8ff')
        )

    def generate_executive_report(self, company_id):
        """
        Generate executive summary PDF report
        
        Args:
            company_id (str): Company identifier
            
        Returns:
            dict: Generated report data with base64 PDF
        """
        try:
            logger.info(f"Generating executive report for company {company_id}")
            
            # Get analysis data
            analysis = self.analyzer.analyze_company_performance(company_id)
            if 'error' in analysis:
                return {'error': 'Unable to generate report - analysis failed'}
            
            predictions = self.analyzer.predict_financial_health(company_id)
            
            # Create PDF
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
            story = []
            
            # Title page
            story.append(Paragraph("Financial Health Report", self.title_style))
            story.append(Paragraph(f"Company: {company_id}", self.styles['Normal']))
            story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}", self.styles['Normal']))
            story.append(Spacer(1, 30))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", self.header_style))
            
            overall_score = analysis['summary']['overall_score']
            overall_rating = analysis['summary']['overall_rating']
            
            exec_summary = f"""
            <para>
            <b>Overall Financial Health Score:</b> {overall_score}/100 ({overall_rating})<br/>
            <b>Risk Level:</b> {analysis['risk_assessment']['risk_levels']['overall_risk'].title()}<br/>
            <b>Future Outlook:</b> {predictions.get('trend', 'stable').title()}<br/>
            </para>
            """
            story.append(Paragraph(exec_summary, self.exec_style))
            story.append(Spacer(1, 20))
            
            # Key Metrics Table
            story.append(Paragraph("Key Financial Metrics", self.header_style))
            metrics_data = self.prepare_metrics_table(analysis['ratios'])
            metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch, 1.5*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e75b6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f8ff')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 20))
            
            # Financial Health Scores
            story.append(Paragraph("Financial Health Breakdown", self.header_style))
            scores_data = [['Category', 'Score', 'Status']]
            
            for category, score in analysis['summary']['category_scores'].items():
                if score >= 80:
                    status = 'Excellent'
                elif score >= 65:
                    status = 'Good'
                elif score >= 50:
                    status = 'Fair'
                else:
                    status = 'Poor'
                
                scores_data.append([
                    category.title(),
                    f"{score}/100",
                    status
                ])
            
            scores_table = Table(scores_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            scores_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#70ad47')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fff8')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))
            story.append(scores_table)
            story.append(Spacer(1, 20))
            
            # Risk Analysis
            story.append(Paragraph("Risk Assessment", self.header_style))
            risk_data = analysis['risk_assessment']
            
            risk_summary = f"""
            <para>
            <b>Overall Risk Level:</b> {risk_data['risk_levels']['overall_risk'].title()}<br/>
            <b>Risk Score:</b> {risk_data['risk_score']}/100<br/>
            </para>
            """
            story.append(Paragraph(risk_summary, self.styles['Normal']))
            story.append(Spacer(1, 10))
            
            # Risk Factors
            if risk_data['risk_factors']:
                story.append(Paragraph("Key Risk Factors:", self.subheader_style))
                for factor in risk_data['risk_factors']:
                    story.append(Paragraph(f"• {factor}", self.styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Recommendations
            story.append(Paragraph("Key Recommendations", self.header_style))
            
            high_priority = [r for r in analysis['recommendations'] if r.get('priority') == 'High']
            medium_priority = [r for r in analysis['recommendations'] if r.get('priority') == 'Medium']
            
            if high_priority:
                story.append(Paragraph("High Priority Actions:", self.subheader_style))
                for rec in high_priority:
                    rec_text = f"<b>{rec['category'].title()}:</b> {rec['recommendation']}<br/><i>Expected Impact:</i> {rec['impact']}"
                    story.append(Paragraph(rec_text, self.styles['Normal']))
                    story.append(Spacer(1, 8))
            
            if medium_priority:
                story.append(Paragraph("Medium Priority Actions:", self.subheader_style))
                for rec in medium_priority:
                    rec_text = f"<b>{rec['category'].title()}:</b> {rec['recommendation']}<br/><i>Expected Impact:</i> {rec['impact']}"
                    story.append(Paragraph(rec_text, self.styles['Normal']))
                    story.append(Spacer(1, 8))
            
            # Future Outlook
            if not predictions.get('error'):
                story.append(Spacer(1, 20))
                story.append(Paragraph("Future Outlook", self.header_style))
                
                outlook_text = f"""
                <para>
                <b>Current Health Score:</b> {predictions.get('current_health_score', 'N/A')}/100<br/>
                <b>Predicted Score (12 months):</b> {predictions.get('predicted_health_score', 'N/A')}/100<br/>
                <b>Trend:</b> {predictions.get('trend', 'stable').title()}<br/>
                <b>Confidence Level:</b> {predictions.get('confidence_level', 'Medium')}<br/>
                </para>
                """
                story.append(Paragraph(outlook_text, self.exec_style))
            
            # Footer
            story.append(Spacer(1, 30))
            footer_text = f"<para><i>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Financial Risk Assessment Platform</i></para>"
            story.append(Paragraph(footer_text, self.styles['Normal']))
            
            # Build PDF
            doc.build(story)
            pdf_data = buffer.getvalue()
            buffer.close()
            
            return {
                'success': True,
                'report_type': 'Executive Summary',
                'company_id': company_id,
                'generated_at': datetime.now().isoformat(),
                'pdf_base64': base64.b64encode(pdf_data).decode('utf-8'),
                'filename': f"executive_report_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
            
        except Exception as e:
            logger.error(f"Error generating executive report: {e}")
            return {'error': f'Report generation failed: {str(e)}'}
    
    def generate_detailed_report(self, company_id):
        """
        Generate detailed financial analysis PDF report
        
        Args:
            company_id (str): Company identifier
            
        Returns:
            dict: Generated report data with base64 PDF
        """
        try:
            logger.info(f"Generating detailed report for company {company_id}")
            
            # Get analysis data
            analysis = self.analyzer.analyze_company_performance(company_id)
            if 'error' in analysis:
                return {'error': 'Unable to generate detailed report - analysis failed'}
            
            predictions = self.analyzer.predict_financial_health(company_id)
            
            # Create PDF
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
            story = []
            
            # Title page
            story.append(Paragraph("Detailed Financial Analysis Report", self.title_style))
            story.append(Paragraph(f"Company: {company_id}", self.styles['Normal']))
            story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}", self.styles['Normal']))
            story.append(Paragraph(f"Analysis Period: Multi-year financial assessment", self.styles['Normal']))
            story.append(Spacer(1, 30))
            
            # Table of Contents
            story.append(Paragraph("Table of Contents", self.header_style))
            toc_items = [
                "1. Executive Summary",
                "2. Financial Ratios Analysis", 
                "3. Liquidity Analysis",
                "4. Profitability Analysis",
                "5. Leverage Analysis",
                "6. Efficiency Analysis",
                "7. Risk Assessment",
                "8. Recommendations",
                "9. Future Outlook"
            ]
            for item in toc_items:
                story.append(Paragraph(item, self.styles['Normal']))
            story.append(PageBreak())
            
            # 1. Executive Summary
            story.append(Paragraph("1. Executive Summary", self.header_style))
            
            overall_score = analysis['summary']['overall_score']
            overall_rating = analysis['summary']['overall_rating']
            risk_level = analysis['risk_assessment']['risk_levels']['overall_risk']
            
            exec_summary = f"""
            <para>
            This comprehensive financial analysis of {company_id} reveals the following key findings:<br/><br/>
            <b>Overall Financial Health:</b> {overall_score}/100 ({overall_rating})<br/>
            <b>Risk Assessment:</b> {risk_level.title()} Risk Level<br/>
            <b>Primary Strengths:</b> {self.identify_strengths(analysis['summary']['category_scores'])}<br/>
            <b>Areas of Concern:</b> {self.identify_concerns(analysis['summary']['category_scores'])}<br/>
            </para>
            """
            story.append(Paragraph(exec_summary, self.exec_style))
            story.append(Spacer(1, 20))
            
            # 2. Financial Ratios Analysis
            story.append(Paragraph("2. Financial Ratios Analysis", self.header_style))
            story.append(Paragraph("The following table presents key financial ratios calculated from the company's financial statements:", self.styles['Normal']))
            story.append(Spacer(1, 10))
            
            ratios_data = self.prepare_detailed_ratios_table(analysis['ratios'])
            ratios_table = Table(ratios_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
            ratios_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e75b6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f8ff')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 1), (-1, -1), 10)
            ]))
            story.append(ratios_table)
            story.append(PageBreak())
            
            # 3. Liquidity Analysis
            story.append(Paragraph("3. Liquidity Analysis", self.header_style))
            liquidity_ratios = analysis['ratios']['liquidity']
            liquidity_score = analysis['summary']['category_scores']['liquidity']
            
            liquidity_text = f"""
            <para>
            <b>Liquidity Score: {liquidity_score}/100</b><br/><br/>
            
            Liquidity ratios measure the company's ability to meet short-term obligations:<br/><br/>
            
            <b>Current Ratio: {liquidity_ratios['current_ratio']}</b><br/>
            {self.interpret_current_ratio(liquidity_ratios['current_ratio'])}<br/><br/>
            
            <b>Quick Ratio: {liquidity_ratios['quick_ratio']}</b><br/>
            {self.interpret_quick_ratio(liquidity_ratios['quick_ratio'])}<br/><br/>
            
            <b>Cash Ratio: {liquidity_ratios['cash_ratio']}</b><br/>
            {self.interpret_cash_ratio(liquidity_ratios['cash_ratio'])}<br/>
            </para>
            """
            story.append(Paragraph(liquidity_text, self.styles['Normal']))
            story.append(Spacer(1, 20))
            
            # 4. Profitability Analysis
            story.append(Paragraph("4. Profitability Analysis", self.header_style))
            profitability_ratios = analysis['ratios']['profitability']
            profitability_score = analysis['summary']['category_scores']['profitability']
            
            profitability_text = f"""
            <para>
            <b>Profitability Score: {profitability_score}/100</b><br/><br/>
            
            Profitability ratios assess the company's ability to generate profits:<br/><br/>
            
            <b>Return on Assets (ROA): {profitability_ratios['roa']:.1%}</b><br/>
            {self.interpret_roa(profitability_ratios['roa'])}<br/><br/>
            
            <b>Return on Equity (ROE): {profitability_ratios['roe']:.1%}</b><br/>
            {self.interpret_roe(profitability_ratios['roe'])}<br/><br/>
            
            <b>Profit Margin: {profitability_ratios['profit_margin']:.1%}</b><br/>
            {self.interpret_profit_margin(profitability_ratios['profit_margin'])}<br/>
            </para>
            """
            story.append(Paragraph(profitability_text, self.styles['Normal']))
            story.append(Spacer(1, 20))
            
            # 5. Leverage Analysis
            story.append(Paragraph("5. Leverage Analysis", self.header_style))
            leverage_ratios = analysis['ratios']['leverage']
            leverage_score = analysis['summary']['category_scores']['leverage']
            
            leverage_text = f"""
            <para>
            <b>Leverage Score: {leverage_score}/100</b><br/><br/>
            
            Leverage ratios evaluate the company's debt levels and financial risk:<br/><br/>
            
            <b>Debt-to-Equity Ratio: {leverage_ratios['debt_to_equity']}</b><br/>
            {self.interpret_debt_to_equity(leverage_ratios['debt_to_equity'])}<br/><br/>
            
            <b>Interest Coverage Ratio: {leverage_ratios['interest_coverage']}</b><br/>
            {self.interpret_interest_coverage(leverage_ratios['interest_coverage'])}<br/>
            </para>
            """
            story.append(Paragraph(leverage_text, self.styles['Normal']))
            story.append(Spacer(1, 20))
            
            # 6. Efficiency Analysis
            story.append(Paragraph("6. Efficiency Analysis", self.header_style))
            efficiency_ratios = analysis['ratios']['efficiency']
            efficiency_score = analysis['summary']['category_scores']['efficiency']
            
            efficiency_text = f"""
            <para>
            <b>Efficiency Score: {efficiency_score}/100</b><br/><br/>
            
            Efficiency ratios measure how effectively the company uses its assets:<br/><br/>
            
            <b>Asset Turnover: {efficiency_ratios['asset_turnover']}</b><br/>
            {self.interpret_asset_turnover(efficiency_ratios['asset_turnover'])}<br/><br/>
            
            <b>Inventory Turnover: {efficiency_ratios['inventory_turnover']}</b><br/>
            {self.interpret_inventory_turnover(efficiency_ratios['inventory_turnover'])}<br/>
            </para>
            """
            story.append(Paragraph(efficiency_text, self.styles['Normal']))
            story.append(PageBreak())
            
            # 7. Risk Assessment
            story.append(Paragraph("7. Risk Assessment", self.header_style))
            risk_data = analysis['risk_assessment']
            
            risk_table_data = [['Risk Category', 'Level', 'Impact']]
            for risk_type, level in risk_data['risk_levels'].items():
                if risk_type != 'overall_risk':
                    impact = self.get_risk_impact(risk_type, level)
                    risk_table_data.append([
                        risk_type.replace('_', ' ').title(),
                        level.title(),
                        impact
                    ])
            
            risk_table = Table(risk_table_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffeaea')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 1), (-1, -1), 10)
            ]))
            story.append(risk_table)
            story.append(Spacer(1, 15))
            
            # Risk Factors Detail
            if risk_data['risk_factors']:
                story.append(Paragraph("Key Risk Factors:", self.subheader_style))
                for i, factor in enumerate(risk_data['risk_factors'], 1):
                    story.append(Paragraph(f"{i}. {factor}", self.styles['Normal']))
                story.append(Spacer(1, 20))
            
            # 8. Recommendations
            story.append(Paragraph("8. Recommendations", self.header_style))
            
            recommendations = analysis['recommendations']
            if recommendations:
                # Group by priority
                high_priority = [r for r in recommendations if r.get('priority') == 'High']
                medium_priority = [r for r in recommendations if r.get('priority') == 'Medium']
                low_priority = [r for r in recommendations if r.get('priority') == 'Low']
                
                if high_priority:
                    story.append(Paragraph("High Priority Recommendations:", self.subheader_style))
                    for i, rec in enumerate(high_priority, 1):
                        rec_text = f"""
                        <para>
                        <b>{i}. {rec['category'].title()} Improvement</b><br/>
                        <b>Action:</b> {rec['recommendation']}<br/>
                        <b>Expected Impact:</b> {rec['impact']}<br/>
                        </para>
                        """
                        story.append(Paragraph(rec_text, self.styles['Normal']))
                        story.append(Spacer(1, 10))
                
                if medium_priority:
                    story.append(Paragraph("Medium Priority Recommendations:", self.subheader_style))
                    for i, rec in enumerate(medium_priority, 1):
                        rec_text = f"""
                        <para>
                        <b>{i}. {rec['category'].title()} Enhancement</b><br/>
                        <b>Action:</b> {rec['recommendation']}<br/>
                        <b>Expected Impact:</b> {rec['impact']}<br/>
                        </para>
                        """
                        story.append(Paragraph(rec_text, self.styles['Normal']))
                        story.append(Spacer(1, 10))
            else:
                story.append(Paragraph("No specific recommendations at this time. Continue monitoring financial performance.", self.styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # 9. Future Outlook
            story.append(Paragraph("9. Future Outlook", self.header_style))
            
            if not predictions.get('error'):
                outlook_text = f"""
                <para>
                Based on historical trends and current financial position:<br/><br/>
                
                <b>Current Health Score:</b> {predictions.get('current_health_score', 'N/A')}/100<br/>
                <b>Predicted Score (12 months):</b> {predictions.get('predicted_health_score', 'N/A')}/100<br/>
                <b>Trend Direction:</b> {predictions.get('trend', 'stable').title()}<br/>
                <b>Confidence Level:</b> {predictions.get('confidence_level', 'Medium')}<br/><br/>
                
                {self.get_outlook_interpretation(predictions)}
                </para>
                """
                story.append(Paragraph(outlook_text, self.exec_style))
            else:
                story.append(Paragraph("Insufficient historical data for reliable future outlook prediction.", self.styles['Normal']))
            
            # Report Footer
            story.append(Spacer(1, 40))
            footer_text = f"""
            <para>
            <b>Disclaimer:</b> This report is based on financial data analysis and should be used for informational purposes only. 
            Financial conditions can change rapidly, and this analysis should be considered alongside other factors when making business decisions.<br/><br/>
            <i>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Financial Risk Assessment Platform</i>
            </para>
            """
            story.append(Paragraph(footer_text, self.styles['Normal']))
            
            # Build PDF
            doc.build(story)
            pdf_data = buffer.getvalue()
            buffer.close()
            
            return {
                'success': True,
                'report_type': 'Detailed Analysis',
                'company_id': company_id,
                'generated_at': datetime.now().isoformat(),
                'pdf_base64': base64.b64encode(pdf_data).decode('utf-8'),
                'filename': f"detailed_report_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
            
        except Exception as e:
            logger.error(f"Error generating detailed report: {e}")
            return {'error': f'Detailed report generation failed: {str(e)}'}
    
    def generate_risk_report(self, company_id):
        """
        Generate risk-focused PDF report
        
        Args:
            company_id (str): Company identifier
            
        Returns:
            dict: Generated risk report data with base64 PDF
        """
        try:
            logger.info(f"Generating risk report for company {company_id}")
            
            # Get analysis data
            analysis = self.analyzer.analyze_company_performance(company_id)
            if 'error' in analysis:
                return {'error': 'Unable to generate risk report - analysis failed'}
            
            # Create PDF
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
            story = []
            
            # Title page with risk styling
            story.append(Paragraph("Financial Risk Assessment Report", self.title_style))
            story.append(Paragraph(f"Company: {company_id}", self.styles['Normal']))
            story.append(Paragraph(f"Assessment Date: {datetime.now().strftime('%B %d, %Y')}", self.styles['Normal']))
            
            risk_level = analysis['risk_assessment']['risk_levels']['overall_risk']
            risk_score = analysis['risk_assessment']['risk_score']
            
            risk_header = f"""
            <para>
            <b>RISK LEVEL: {risk_level.upper()}</b><br/>
            <b>RISK SCORE: {risk_score}/100</b>
            </para>
            """
            story.append(Spacer(1, 20))
            story.append(Paragraph(risk_header, self.exec_style))
            story.append(Spacer(1, 30))
            
            # Risk Summary
            story.append(Paragraph("Risk Assessment Summary", self.header_style))
            
            risk_summary = f"""
            <para>
            This risk assessment evaluates {company_id}'s financial stability and identifies potential areas of concern 
            that could impact the company's ability to meet its financial obligations and maintain operations.<br/><br/>
            
            <b>Overall Risk Classification:</b> {risk_level.title()}<br/>
            <b>Risk Score:</b> {risk_score}/100 (Higher scores indicate greater risk)<br/>
            <b>Assessment Basis:</b> Multi-factor analysis of financial ratios and performance indicators<br/>
            </para>
            """
            story.append(Paragraph(risk_summary, self.styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Detailed Risk Breakdown
            story.append(Paragraph("Detailed Risk Analysis", self.header_style))
            
            risk_levels = analysis['risk_assessment']['risk_levels']
            
            # Create detailed risk table
            detailed_risk_data = [['Risk Category', 'Level', 'Score Impact', 'Assessment']]
            
            risk_descriptions = {
                'liquidity_risk': 'Ability to meet short-term obligations',
                'solvency_risk': 'Long-term financial stability and debt management',
                'profitability_risk': 'Capacity to generate sustainable profits',
                'operational_risk': 'Efficiency in asset utilization and operations'
            }
            
            for risk_type, level in risk_levels.items():
                if risk_type != 'overall_risk':
                    impact = 'High' if level == 'high' else 'Medium' if level == 'medium' else 'Low'
                    assessment = risk_descriptions.get(risk_type, 'Financial risk assessment')
                    
                    detailed_risk_data.append([
                        risk_type.replace('_', ' ').title(),
                        level.title(),
                        impact,
                        assessment
                    ])
            
            detailed_risk_table = Table(detailed_risk_data, colWidths=[1.8*inch, 1*inch, 1*inch, 2.2*inch])
            detailed_risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffeaea')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 1), (-1, -1), 9)
            ]))
            story.append(detailed_risk_table)
            story.append(Spacer(1, 20))
            
            # Risk Factors
            story.append(Paragraph("Identified Risk Factors", self.header_style))
            risk_factors = analysis['risk_assessment']['risk_factors']
            
            if risk_factors:
                for i, factor in enumerate(risk_factors, 1):
                    story.append(Paragraph(f"{i}. {factor}", self.styles['Normal']))
                    story.append(Spacer(1, 5))
            else:
                story.append(Paragraph("No significant risk factors identified at this time.", self.styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Risk Mitigation Recommendations
            story.append(Paragraph("Risk Mitigation Recommendations", self.header_style))
            
            # Filter high-priority recommendations for risk mitigation
            high_risk_recs = [r for r in analysis['recommendations'] if r.get('priority') == 'High']
            
            if high_risk_recs:
                story.append(Paragraph("Immediate Actions Required:", self.subheader_style))
                for i, rec in enumerate(high_risk_recs, 1):
                    mitigation_text = f"""
                    <para>
                    <b>{i}. {rec['category'].title()} Risk Mitigation</b><br/>
                    <b>Action:</b> {rec['recommendation']}<br/>
                    <b>Risk Reduction Impact:</b> {rec['impact']}<br/>
                    </para>
                    """
                    story.append(Paragraph(mitigation_text, self.styles['Normal']))
                    story.append(Spacer(1, 10))
            
            # General risk mitigation strategies
            story.append(Paragraph("General Risk Management Strategies:", self.subheader_style))
            
            general_strategies = [
                "Regular monitoring of key financial ratios and performance indicators",
                "Maintenance of adequate cash reserves for operational flexibility",
                "Diversification of revenue sources to reduce dependency risks",
                "Regular review and optimization of cost structure",
                "Proactive management of debt levels and payment schedules"
            ]
            
            for strategy in general_strategies:
                story.append(Paragraph(f"• {strategy}", self.styles['Normal']))
                story.append(Spacer(1, 3))
            
            story.append(Spacer(1, 20))
            
            # Monitoring Recommendations
            story.append(Paragraph("Ongoing Monitoring Recommendations", self.header_style))
            
            monitoring_text = f"""
            <para>
            To maintain effective risk management, the following monitoring schedule is recommended:<br/><br/>
            
            <b>Monthly Monitoring:</b><br/>
            • Cash flow position and liquidity ratios<br/>
            • Key operational metrics and efficiency indicators<br/>
            • Debt service coverage and payment schedules<br/><br/>
            
            <b>Quarterly Review:</b><br/>
            • Comprehensive financial ratio analysis<br/>
            • Risk level reassessment<br/>
            • Strategy adjustment based on performance trends<br/><br/>
            
            <b>Annual Assessment:</b><br/>
            • Full financial health evaluation<br/>
            • Risk management strategy review<br/>
            • Long-term financial planning and forecasting<br/>
            </para>
            """
            story.append(Paragraph(monitoring_text, self.styles['Normal']))
            
            # Report Footer
            story.append(Spacer(1, 30))
            footer_text = f"""
            <para>
            <b>Important Notice:</b> This risk assessment is based on financial data analysis at a specific point in time. 
            Financial conditions and risk levels can change rapidly due to internal and external factors. 
            Regular reassessment is essential for effective risk management.<br/><br/>
            <i>Risk Assessment generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Financial Risk Assessment Platform</i>
            </para>
            """
            story.append(Paragraph(footer_text, self.styles['Normal']))
            
            # Build PDF
            doc.build(story)
            pdf_data = buffer.getvalue()
            buffer.close()
            
            return {
                'success': True,
                'report_type': 'Risk Assessment',
                'company_id': company_id,
                'risk_level': risk_level,
                'risk_score': risk_score,
                'generated_at': datetime.now().isoformat(),
                'pdf_base64': base64.b64encode(pdf_data).decode('utf-8'),
                'filename': f"risk_report_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {'error': f'Risk report generation failed: {str(e)}'}
    
    # Helper methods for interpretations
    def prepare_metrics_table(self, ratios):
        """Prepare metrics table data for PDF"""
        data = [['Metric Category', 'Ratio', 'Value']]
        
        for category, category_ratios in ratios.items():
            for ratio_name, value in category_ratios.items():
                formatted_value = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                data.append([
                    category.title(),
                    ratio_name.replace('_', ' ').title(),
                    formatted_value
                ])
        
        return data
    
    def prepare_detailed_ratios_table(self, ratios):
        """Prepare detailed ratios table for comprehensive analysis"""
        data = [['Financial Ratio', 'Value', 'Interpretation']]
        
        # Liquidity ratios
        data.append(['Current Ratio', f"{ratios['liquidity']['current_ratio']:.2f}", self.interpret_current_ratio(ratios['liquidity']['current_ratio'])])
        data.append(['Quick Ratio', f"{ratios['liquidity']['quick_ratio']:.2f}", self.interpret_quick_ratio(ratios['liquidity']['quick_ratio'])])
        data.append(['Cash Ratio', f"{ratios['liquidity']['cash_ratio']:.2f}", self.interpret_cash_ratio(ratios['liquidity']['cash_ratio'])])
        
        # Profitability ratios
        data.append(['Return on Assets', f"{ratios['profitability']['roa']:.1%}", self.interpret_roa(ratios['profitability']['roa'])])
        data.append(['Return on Equity', f"{ratios['profitability']['roe']:.1%}", self.interpret_roe(ratios['profitability']['roe'])])
        data.append(['Profit Margin', f"{ratios['profitability']['profit_margin']:.1%}", self.interpret_profit_margin(ratios['profitability']['profit_margin'])])
        
        # Leverage ratios
        data.append(['Debt-to-Equity', f"{ratios['leverage']['debt_to_equity']:.2f}", self.interpret_debt_to_equity(ratios['leverage']['debt_to_equity'])])
        data.append(['Interest Coverage', f"{ratios['leverage']['interest_coverage']:.2f}", self.interpret_interest_coverage(ratios['leverage']['interest_coverage'])])
        
        # Efficiency ratios
        data.append(['Asset Turnover', f"{ratios['efficiency']['asset_turnover']:.2f}", self.interpret_asset_turnover(ratios['efficiency']['asset_turnover'])])
        data.append(['Inventory Turnover', f"{ratios['efficiency']['inventory_turnover']:.2f}", self.interpret_inventory_turnover(ratios['efficiency']['inventory_turnover'])])
        
        return data
    
    def identify_strengths(self, category_scores):
        """Identify company strengths from category scores"""
        strengths = []
        for category, score in category_scores.items():
            if score >= 80:
                strengths.append(f"Excellent {category}")
            elif score >= 70:
                strengths.append(f"Strong {category}")
        
        return ", ".join(strengths) if strengths else "Balanced performance across categories"
    
    def identify_concerns(self, category_scores):
        """Identify areas of concern from category scores"""
        concerns = []
        for category, score in category_scores.items():
            if score < 50:
                concerns.append(f"Weak {category}")
            elif score < 65:
                concerns.append(f"Below-average {category}")
        
        return ", ".join(concerns) if concerns else "No significant concerns identified"
    
    # Ratio interpretation methods
    def interpret_current_ratio(self, ratio):
        """Interpret current ratio"""
        if ratio >= 2.0:
            return "Excellent liquidity position"
        elif ratio >= 1.5:
            return "Good liquidity position"
        elif ratio >= 1.0:
            return "Adequate liquidity"
        else:
            return "Liquidity concerns"
    
    def interpret_quick_ratio(self, ratio):
        """Interpret quick ratio"""
        if ratio >= 1.0:
            return "Strong immediate liquidity"
        elif ratio >= 0.5:
            return "Moderate liquidity"
        else:
            return "Limited immediate liquidity"
    
    def interpret_cash_ratio(self, ratio):
        """Interpret cash ratio"""
        if ratio >= 0.5:
            return "Strong cash position"
        elif ratio >= 0.2:
            return "Adequate cash reserves"
        else:
            return "Limited cash reserves"
    
    def interpret_roa(self, ratio):
        """Interpret return on assets"""
        if ratio >= 0.15:
            return "Excellent asset utilization"
        elif ratio >= 0.10:
            return "Good asset efficiency"
        elif ratio >= 0.05:
            return "Moderate asset efficiency"
        elif ratio > 0:
            return "Low asset efficiency"
        else:
            return "Negative returns on assets"
    
    def interpret_roe(self, ratio):
        """Interpret return on equity"""
        if ratio >= 0.20:
            return "Excellent returns to shareholders"
        elif ratio >= 0.15:
            return "Good shareholder returns"
        elif ratio >= 0.10:
            return "Moderate shareholder returns"
        elif ratio > 0:
            return "Low shareholder returns"
        else:
            return "Negative shareholder returns"
    
    def interpret_profit_margin(self, ratio):
        """Interpret profit margin"""
        if ratio >= 0.15:
            return "Excellent profitability"
        elif ratio >= 0.10:
            return "Good profit margins"
        elif ratio >= 0.05:
            return "Moderate profitability"
        elif ratio > 0:
            return "Low profit margins"
        else:
            return "Operating at a loss"
    
    def interpret_debt_to_equity(self, ratio):
        """Interpret debt to equity ratio"""
        if ratio <= 0.3:
            return "Conservative debt levels"
        elif ratio <= 0.6:
            return "Moderate debt levels"
        elif ratio <= 1.0:
            return "High debt levels"
        else:
            return