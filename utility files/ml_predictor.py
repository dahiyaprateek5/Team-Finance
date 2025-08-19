import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier, MLPRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CashFlowGenerator:
    """Generate cash flow statements from balance sheet data using ML and financial formulas"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        
    def generate_from_balance_sheet(self, balance_sheet_data: Dict[str, Any], 
                                  previous_year_data: Dict[str, Any] = None,
                                  method: str = 'indirect') -> Dict[str, Any]:
        """Generate cash flow statement from balance sheet data"""
        
        try:
            if method == 'indirect':
                return self._generate_indirect_cash_flow(balance_sheet_data, previous_year_data)
            else:
                return self._generate_direct_cash_flow(balance_sheet_data, previous_year_data)
        except Exception as e:
            logger.error(f"Error generating cash flow: {e}")
            return self._get_default_cash_flow_template()
    
    def _generate_indirect_cash_flow(self, current_data: Dict[str, Any], 
                                   previous_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate cash flow using indirect method"""
        
        # Extract key values
        net_income = current_data.get('net_income', 0)
        if net_income == 0:
            # Estimate net income as percentage of total assets (conservative estimate)
            net_income = current_data.get('total_assets', 0) * 0.05
        
        # Calculate depreciation (estimate if not provided)
        depreciation = current_data.get('depreciation_and_amortization', 0)
        if depreciation == 0:
            # Estimate as percentage of PPE
            ppe = current_data.get('property_plant_equipment', 0)
            depreciation = ppe * 0.10 if ppe > 0 else current_data.get('total_assets', 0) * 0.03
        
        # Calculate changes in working capital
        working_capital_changes = self._calculate_working_capital_changes(current_data, previous_data)
        
        # Operating Cash Flow
        operating_cash_flow = (
            net_income + 
            depreciation + 
            current_data.get('stock_based_compensation', 0) +
            working_capital_changes['total_change']
        )
        
        # Investing Activities
        capex = self._estimate_capital_expenditure(current_data, previous_data)
        acquisitions = current_data.get('acquisitions', 0)
        investing_cash_flow = -(capex + acquisitions)
        
        # Financing Activities
        financing_cash_flow = self._calculate_financing_activities(current_data, previous_data)
        
        # Free Cash Flow
        free_cash_flow = operating_cash_flow - capex
        
        # Financial Ratios
        ocf_to_ni_ratio = operating_cash_flow / net_income if net_income != 0 else 0
        debt_to_equity = self._calculate_debt_to_equity(current_data)
        interest_coverage = self._calculate_interest_coverage(current_data)
        
        # Liquidation Risk Assessment
        liquidation_label = self._assess_liquidation_risk(current_data, {
            'net_income': net_income,
            'operating_cash_flow': operating_cash_flow,
            'free_cash_flow': free_cash_flow
        })
        
        return {
            'net_income': net_income,
            'depreciation_and_amortization': depreciation,
            'stock_based_compensation': current_data.get('stock_based_compensation', 0),
            'changes_in_working_capital': working_capital_changes['total_change'],
            'accounts_receivable': working_capital_changes['accounts_receivable'],
            'inventory': working_capital_changes['inventory'],
            'accounts_payable': working_capital_changes['accounts_payable'],
            'net_cash_from_operating_activities': operating_cash_flow,
            'capital_expenditures': capex,
            'acquisitions': acquisitions,
            'net_cash_from_investing_activities': investing_cash_flow,
            'dividends_paid': current_data.get('dividends_paid', 0),
            'share_repurchases': current_data.get('share_repurchases', 0),
            'net_cash_from_financing_activities': financing_cash_flow,
            'free_cash_flow': free_cash_flow,
            'ocf_to_net_income_ratio': ocf_to_ni_ratio,
            'liquidation_label': liquidation_label,
            'debt_to_equity_ratio': debt_to_equity,
            'interest_coverage_ratio': interest_coverage
        }
    
    def _calculate_working_capital_changes(self, current_data: Dict[str, Any], 
                                         previous_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate changes in working capital components"""
        
        if previous_data is None:
            # Estimate changes based on industry averages
            total_assets = current_data.get('total_assets', 0)
            return {
                'accounts_receivable': -(total_assets * 0.02),  # Increase in AR reduces cash
                'inventory': -(total_assets * 0.01),  # Increase in inventory reduces cash
                'accounts_payable': total_assets * 0.015,  # Increase in AP increases cash
                'total_change': -(total_assets * 0.015)  # Net working capital increase
            }
        
        # Calculate actual changes
        ar_change = (current_data.get('accounts_receivable', 0) - 
                    previous_data.get('accounts_receivable', 0))
        inventory_change = (current_data.get('inventory', 0) - 
                           previous_data.get('inventory', 0))
        ap_change = (current_data.get('accounts_payable', 0) - 
                    previous_data.get('accounts_payable', 0))
        
        total_change = -ar_change - inventory_change + ap_change
        
        return {
            'accounts_receivable': -ar_change,
            'inventory': -inventory_change,
            'accounts_payable': ap_change,
            'total_change': total_change
        }
    
    def _estimate_capital_expenditure(self, current_data: Dict[str, Any], 
                                    previous_data: Dict[str, Any] = None) -> float:
        """Estimate capital expenditures"""
        
        if previous_data is None:
            # Estimate as percentage of PPE or total assets
            ppe = current_data.get('property_plant_equipment', 0)
            if ppe > 0:
                return ppe * 0.08  # 8% of PPE as maintenance capex
            else:
                return current_data.get('total_assets', 0) * 0.04
        
        # Calculate from PPE changes + depreciation
        current_ppe = current_data.get('property_plant_equipment', 0)
        previous_ppe = previous_data.get('property_plant_equipment', 0)
        depreciation = current_data.get('depreciation_and_amortization', 0)
        
        capex = (current_ppe - previous_ppe) + depreciation
        return max(capex, 0)  # Ensure non-negative
    
    def _calculate_financing_activities(self, current_data: Dict[str, Any], 
                                      previous_data: Dict[str, Any] = None) -> float:
        """Calculate net cash from financing activities"""
        
        dividends = current_data.get('dividends_paid', 0)
        share_repurchases = current_data.get('share_repurchases', 0)
        
        if previous_data is None:
            # Estimate debt changes
            debt_change = 0
        else:
            current_debt = (current_data.get('short_term_debt', 0) + 
                           current_data.get('long_term_debt', 0))
            previous_debt = (previous_data.get('short_term_debt', 0) + 
                            previous_data.get('long_term_debt', 0))
            debt_change = current_debt - previous_debt
        
        return debt_change - dividends - share_repurchases
    
    def _calculate_debt_to_equity(self, data: Dict[str, Any]) -> float:
        """Calculate debt-to-equity ratio"""
        total_debt = data.get('short_term_debt', 0) + data.get('long_term_debt', 0)
        total_equity = data.get('total_equity', 1)  # Avoid division by zero
        
        return total_debt / total_equity if total_equity != 0 else 0
    
    def _calculate_interest_coverage(self, data: Dict[str, Any]) -> float:
        """Calculate interest coverage ratio"""
        net_income = data.get('net_income', 0)
        interest_expense = data.get('interest_expense', 0)
        
        if interest_expense == 0:
            # Estimate interest expense based on debt
            total_debt = data.get('short_term_debt', 0) + data.get('long_term_debt', 0)
            interest_expense = total_debt * 0.05  # Assume 5% interest rate
        
        ebit = net_income + interest_expense
        return ebit / interest_expense if interest_expense != 0 else float('inf')
    
    def _assess_liquidation_risk(self, balance_sheet_data: Dict[str, Any], 
                               cash_flow_data: Dict[str, Any]) -> int:
        """Assess liquidation risk (0 = low risk, 1 = high risk)"""
        
        risk_factors = 0
        
        # Check negative net income
        if cash_flow_data.get('net_income', 0) < 0:
            risk_factors += 1
        
        # Check negative operating cash flow
        if cash_flow_data.get('operating_cash_flow', 0) < 0:
            risk_factors += 1
        
        # Check negative free cash flow
        if cash_flow_data.get('free_cash_flow', 0) < 0:
            risk_factors += 1
        
        # Check high debt-to-equity ratio
        debt_to_equity = self._calculate_debt_to_equity(balance_sheet_data)
        if debt_to_equity > 2.0:
            risk_factors += 1
        
        # Check low current ratio
        current_ratio = (balance_sheet_data.get('current_assets', 0) / 
                        balance_sheet_data.get('current_liabilities', 1))
        if current_ratio < 1.0:
            risk_factors += 1
        
        # Return 1 if 3 or more risk factors present
        return 1 if risk_factors >= 3 else 0
    
    def _generate_direct_cash_flow(self, current_data: Dict[str, Any], 
                                 previous_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate cash flow using direct method (simplified)"""
        # For direct method, we would need detailed transaction data
        # This is a simplified version that estimates based on balance sheet
        
        # Start with indirect method and adjust
        indirect_cf = self._generate_indirect_cash_flow(current_data, previous_data)
        
        # Direct method shows actual cash receipts and payments
        revenue = current_data.get('revenue', 0)
        if revenue == 0:
            # Estimate revenue based on assets
            revenue = current_data.get('total_assets', 0) * 0.8
        
        # Estimate cash collections (revenue - change in AR)
        ar_change = indirect_cf['accounts_receivable']
        cash_from_customers = revenue + ar_change
        
        # Estimate cash payments to suppliers
        cogs = revenue * 0.6  # Assume 60% COGS
        inventory_change = indirect_cf['inventory']
        ap_change = indirect_cf['accounts_payable']
        cash_to_suppliers = cogs - inventory_change - ap_change
        
        # Operating expenses
        operating_expenses = revenue * 0.2  # Assume 20% operating expenses
        
        # Net operating cash flow (direct method)
        operating_cash_flow = cash_from_customers - cash_to_suppliers - operating_expenses
        
        # Update the cash flow with direct method calculation
        indirect_cf['net_cash_from_operating_activities'] = operating_cash_flow
        indirect_cf['free_cash_flow'] = operating_cash_flow - indirect_cf['capital_expenditures']
        
        return indirect_cf
    
    def _get_default_cash_flow_template(self) -> Dict[str, Any]:
        """Return default cash flow template with zero values"""
        return {
            'net_income': 0,
            'depreciation_and_amortization': 0,
            'stock_based_compensation': 0,
            'changes_in_working_capital': 0,
            'accounts_receivable': 0,
            'inventory': 0,
            'accounts_payable': 0,
            'net_cash_from_operating_activities': 0,
            'capital_expenditures': 0,
            'acquisitions': 0,
            'net_cash_from_investing_activities': 0,
            'dividends_paid': 0,
            'share_repurchases': 0,
            'net_cash_from_financing_activities': 0,
            'free_cash_flow': 0,
            'ocf_to_net_income_ratio': 0,
            'liquidation_label': 0,
            'debt_to_equity_ratio': 0,
            'interest_coverage_ratio': 0
        }

class LiquidationPredictor:
    """Predict liquidation risk using machine learning models"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42),
            'lightgbm': lgb.LGBMClassifier(random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        }
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
    
    def prepare_features(self, cash_flow_data: Dict[str, Any], 
                        balance_sheet_data: Dict[str, Any] = None) -> np.ndarray:
        """Prepare features for liquidation prediction"""
        
        features = []
        
        # Cash flow features
        features.extend([
            cash_flow_data.get('net_income', 0),
            cash_flow_data.get('net_cash_from_operating_activities', 0),
            cash_flow_data.get('free_cash_flow', 0),
            cash_flow_data.get('ocf_to_net_income_ratio', 0),
            cash_flow_data.get('debt_to_equity_ratio', 0),
            cash_flow_data.get('interest_coverage_ratio', 0)
        ])
        
        # Balance sheet features (if available)
        if balance_sheet_data:
            current_ratio = (balance_sheet_data.get('current_assets', 0) / 
                           max(balance_sheet_data.get('current_liabilities', 1), 1))
            quick_ratio = ((balance_sheet_data.get('current_assets', 0) - 
                           balance_sheet_data.get('inventory', 0)) / 
                          max(balance_sheet_data.get('current_liabilities', 1), 1))
            
            features.extend([
                current_ratio,
                quick_ratio,
                balance_sheet_data.get('total_assets', 0),
                balance_sheet_data.get('total_liabilities', 0) / 
                max(balance_sheet_data.get('total_assets', 1), 1)  # Debt ratio
            ])
        else:
            features.extend([0, 0, 0, 0])  # Default values
        
        self.feature_names = [
            'net_income', 'operating_cash_flow', 'free_cash_flow', 
            'ocf_to_ni_ratio', 'debt_to_equity', 'interest_coverage',
            'current_ratio', 'quick_ratio', 'total_assets', 'debt_ratio'
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train_models(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train liquidation prediction models"""
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for data_point in training_data:
                features = self.prepare_features(
                    data_point.get('cash_flow', {}),
                    data_point.get('balance_sheet', {})
                )
                X.append(features.flatten())
                y.append(data_point.get('liquidation_label', 0))
            
            X = np.array(X)
            y = np.array(y)
            
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train and evaluate models
            model_scores = {}
            
            for name, model in self.models.items():
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    model_scores[name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                    
                    logger.info(f"{name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    continue
            
            # Select best model based on F1 score
            if model_scores:
                best_model_name = max(model_scores.keys(), 
                                    key=lambda x: model_scores[x]['f1'])
                self.best_model = self.models[best_model_name]
                self.is_trained = True
                
                logger.info(f"Best model: {best_model_name}")
                return model_scores
            else:
                logger.error("No models trained successfully")
                return {}
                
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return {}
    
    def predict_liquidation_risk(self, cash_flow_data: Dict[str, Any], 
                               balance_sheet_data: Dict[str, Any] = None) -> Tuple[int, float]:
        """Predict liquidation risk"""
        
        try:
            # Prepare features
            features = self.prepare_features(cash_flow_data, balance_sheet_data)
            
            if self.is_trained and self.best_model is not None:
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Predict
                prediction = self.best_model.predict(features_scaled)[0]
                
                # Get probability if available
                if hasattr(self.best_model, 'predict_proba'):
                    probability = self.best_model.predict_proba(features_scaled)[0][1]
                else:
                    probability = float(prediction)
                
                return int(prediction), float(probability)
            else:
                # Fallback to rule-based prediction
                return self._rule_based_prediction(cash_flow_data, balance_sheet_data)
                
        except Exception as e:
            logger.error(f"Error in liquidation prediction: {e}")
            return self._rule_based_prediction(cash_flow_data, balance_sheet_data)
    
    def _rule_based_prediction(self, cash_flow_data: Dict[str, Any], 
                             balance_sheet_data: Dict[str, Any] = None) -> Tuple[int, float]:
        """Rule-based liquidation prediction as fallback"""
        
        risk_score = 0
        max_score = 5
        
        # Negative net income
        if cash_flow_data.get('net_income', 0) < 0:
            risk_score += 1
        
        # Negative operating cash flow
        if cash_flow_data.get('net_cash_from_operating_activities', 0) < 0:
            risk_score += 1
        
        # Negative free cash flow
        if cash_flow_data.get('free_cash_flow', 0) < 0:
            risk_score += 1
        
        # High debt-to-equity ratio
        if cash_flow_data.get('debt_to_equity_ratio', 0) > 2:
            risk_score += 1
        
        # Low interest coverage
        if cash_flow_data.get('interest_coverage_ratio', float('inf')) < 2:
            risk_score += 1
        
        probability = risk_score / max_score
        prediction = 1 if probability >= 0.6 else 0
        
        return prediction, probability

class FinancialRatioCalculator:
    """Calculate various financial ratios"""
    
    @staticmethod
    def calculate_liquidity_ratios(balance_sheet_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate liquidity ratios"""
        
        current_assets = balance_sheet_data.get('current_assets', 0)
        current_liabilities = max(balance_sheet_data.get('current_liabilities', 1), 1)
        inventory = balance_sheet_data.get('inventory', 0)
        cash = balance_sheet_data.get('cash_and_equivalents', 0)
        
        return {
            'current_ratio': current_assets / current_liabilities,
            'quick_ratio': (current_assets - inventory) / current_liabilities,
            'cash_ratio': cash / current_liabilities
        }
    
    @staticmethod
    def calculate_leverage_ratios(balance_sheet_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate leverage ratios"""
        
        total_debt = (balance_sheet_data.get('short_term_debt', 0) + 
                     balance_sheet_data.get('long_term_debt', 0))
        total_equity = max(balance_sheet_data.get('total_equity', 1), 1)
        total_assets = max(balance_sheet_data.get('total_assets', 1), 1)
        
        return {
            'debt_to_equity': total_debt / total_equity,
            'debt_to_assets': total_debt / total_assets,
            'equity_ratio': total_equity / total_assets
        }
    
    @staticmethod
    def calculate_efficiency_ratios(balance_sheet_data: Dict[str, Any], 
                                  income_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate efficiency ratios"""
        
        if income_data is None:
            return {'asset_turnover': 0, 'inventory_turnover': 0, 'receivables_turnover': 0}
        
        revenue = income_data.get('revenue', 0)
        total_assets = max(balance_sheet_data.get('total_assets', 1), 1)
        inventory = max(balance_sheet_data.get('inventory', 1), 1)
        accounts_receivable = max(balance_sheet_data.get('accounts_receivable', 1), 1)
        cogs = income_data.get('cost_of_goods_sold', revenue * 0.6)  # Estimate if not available
        
        return {
            'asset_turnover': revenue / total_assets,
            'inventory_turnover': cogs / inventory,
            'receivables_turnover': revenue / accounts_receivable
        }
    
    @staticmethod
    def calculate_profitability_ratios(balance_sheet_data: Dict[str, Any], 
                                     income_data: Dict[str, Any] = None,
                                     cash_flow_data: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate profitability ratios"""
        
        if income_data is None:
            return {'roa': 0, 'roe': 0, 'operating_margin': 0}
        
        net_income = income_data.get('net_income', 0)
        revenue = max(income_data.get('revenue', 1), 1)
        total_assets = max(balance_sheet_data.get('total_assets', 1), 1)
        total_equity = max(balance_sheet_data.get('total_equity', 1), 1)
        operating_income = income_data.get('operating_income', net_income)
        
        ratios = {
            'roa': net_income / total_assets,
            'roe': net_income / total_equity,
            'operating_margin': operating_income / revenue,
            'net_margin': net_income / revenue
        }
        
        # Add cash flow ratios if available
        if cash_flow_data:
            operating_cf = cash_flow_data.get('net_cash_from_operating_activities', 0)
            ratios['operating_cf_margin'] = operating_cf / revenue
            ratios['ocf_to_net_income'] = operating_cf / max(net_income, 1)
        
        return ratios

class ModelManager:
    """Manage ML models - save, load, and update"""
    
    @staticmethod
    def save_model(model, scaler, feature_names: List[str], model_path: str) -> bool:
        """Save trained model and preprocessing objects"""
        try:
            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_names': feature_names,
                'timestamp': datetime.now(),
                'version': '1.0'
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    @staticmethod
    def load_model(model_path: str) -> Tuple[Any, Any, List[str]]:
        """Load trained model and preprocessing objects"""
        try:
            model_data = joblib.load(model_path)
            
            return (
                model_data['model'],
                model_data['scaler'],
                model_data['feature_names']
            )
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None, []
    
    @staticmethod
    def get_model_info(model_path: str) -> Dict[str, Any]:
        """Get model information"""
        try:
            model_data = joblib.load(model_path)
            
            return {
                'timestamp': model_data.get('timestamp'),
                'version': model_data.get('version'),
                'feature_names': model_data.get('feature_names'),
                'model_type': type(model_data['model']).__name__
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}

def generate_financial_insights(cash_flow_data: Dict[str, Any], 
                              balance_sheet_data: Dict[str, Any] = None) -> List[Dict[str, str]]:
    """Generate AI-powered financial insights"""
    
    insights = []
    
    # Operating Cash Flow Analysis
    ocf = cash_flow_data.get('net_cash_from_operating_activities', 0)
    net_income = cash_flow_data.get('net_income', 0)
    
    if ocf > net_income and net_income > 0:
        insights.append({
            'type': 'positive',
            'title': 'Strong Cash Generation',
            'description': f'Operating cash flow (${ocf:,.0f}) exceeds net income (${net_income:,.0f}), indicating high-quality earnings and strong cash conversion.'
        })
    elif ocf < 0:
        insights.append({
            'type': 'negative',
            'title': 'Negative Operating Cash Flow',
            'description': f'Operating cash flow is negative (${ocf:,.0f}), indicating the core business is consuming cash. Immediate attention required.'
        })
    
    # Free Cash Flow Analysis
    fcf = cash_flow_data.get('free_cash_flow', 0)
    if fcf > 0:
        insights.append({
            'type': 'positive',
            'title': 'Positive Free Cash Flow',
            'description': f'Free cash flow of ${fcf:,.0f} provides flexibility for debt reduction, dividends, or growth investments.'
        })
    elif fcf < 0:
        insights.append({
            'type': 'warning',
            'title': 'Negative Free Cash Flow',
            'description': f'Negative free cash flow (${fcf:,.0f}) indicates capital investments exceed operating cash generation.'
        })
    
    # Debt Analysis
    debt_to_equity = cash_flow_data.get('debt_to_equity_ratio', 0)
    if debt_to_equity > 2:
        insights.append({
            'type': 'warning',
            'title': 'High Leverage',
            'description': f'Debt-to-equity ratio of {debt_to_equity:.2f} indicates high financial leverage. Monitor debt service capabilities.'
        })
    elif debt_to_equity < 0.5:
        insights.append({
            'type': 'positive',
            'title': 'Conservative Capital Structure',
            'description': f'Low debt-to-equity ratio of {debt_to_equity:.2f} indicates conservative financing and low financial risk.'
        })
    
    # Liquidity Analysis (if balance sheet data available)
    if balance_sheet_data:
        current_ratio = (balance_sheet_data.get('current_assets', 0) / 
                        max(balance_sheet_data.get('current_liabilities', 1), 1))
        
        if current_ratio > 2:
            insights.append({
                'type': 'positive',
                'title': 'Strong Liquidity Position',
                'description': f'Current ratio of {current_ratio:.2f} indicates excellent ability to meet short-term obligations.'
            })
        elif current_ratio < 1:
            insights.append({
                'type': 'negative',
                'title': 'Liquidity Concern',
                'description': f'Current ratio of {current_ratio:.2f} indicates potential difficulty meeting short-term obligations.'
            })
    
    return insights

def validate_cash_flow_data(cash_flow_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate cash flow data for consistency"""
    
    errors = []
    
    # Check for required fields
    required_fields = ['net_income', 'net_cash_from_operating_activities', 'free_cash_flow']
    for field in required_fields:
        if field not in cash_flow_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate free cash flow calculation
    ocf = cash_flow_data.get('net_cash_from_operating_activities', 0)
    capex = cash_flow_data.get('capital_expenditures', 0)
    calculated_fcf = ocf - capex
    reported_fcf = cash_flow_data.get('free_cash_flow', 0)
    
    if abs(calculated_fcf - reported_fcf) > max(abs(reported_fcf) * 0.1, 1000):
        errors.append("Free cash flow calculation inconsistency")
    
    # Check for unreasonable values
    total_assets = cash_flow_data.get('total_assets', 1)
    if abs(ocf) > total_assets * 2:
        errors.append("Operating cash flow seems unreasonably high relative to assets")
    
    return len(errors) == 0, errors

# Global model instances
cash_flow_generator = CashFlowGenerator()
liquidation_predictor = LiquidationPredictor()