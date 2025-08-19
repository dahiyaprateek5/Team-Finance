"""
Performance Tracking and Monitoring
"""

import numpy as np  # type: ignore
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

class PerformanceTracker:
    """
    Track and monitor model performance over time
    """
    
    def __init__(self):
        self.performance_history = []
        self.model_registry = {}
        self.alerts = []
        self.thresholds = {}
    
    def register_model(self, model_id: str, model_info: Dict[str, Any]):
        """Register a model for tracking"""
        self.model_registry[model_id] = {
            **model_info,
            'registered_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        print(f"✅ Model {model_id} registered for tracking")
    
    def log_performance(self, model_id: str, metrics: Dict[str, Any], 
                       dataset_info: Optional[Dict[str, Any]] = None):
        """Log performance metrics for a model"""
        try:
            performance_record = {
                'model_id': model_id,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'dataset_info': dataset_info or {},
                'performance_hash': self._calculate_performance_hash(metrics)
            }
            
            self.performance_history.append(performance_record)
            
            # Update model registry
            if model_id in self.model_registry:
                self.model_registry[model_id]['last_updated'] = datetime.now().isoformat()
                self.model_registry[model_id]['performance_count'] = self.model_registry[model_id].get('performance_count', 0) + 1
            
            # Check for alerts
            self._check_performance_alerts(model_id, metrics)
            
            print(f"📊 Performance logged for model {model_id}")
            
        except Exception as e:
            print(f"❌ Failed to log performance: {e}")
    
    def set_alert_threshold(self, model_id: str, metric_name: str, 
                           threshold_value: float, condition: str = 'below'):
        """Set performance alert threshold"""
        if model_id not in self.thresholds:
            self.thresholds[model_id] = {}
        
        self.thresholds[model_id][metric_name] = {
            'threshold': threshold_value,
            'condition': condition,  # 'below', 'above', 'change'
            'created_at': datetime.now().isoformat()
        }
        
        print(f"🚨 Alert threshold set: {model_id}.{metric_name} {condition} {threshold_value}")
    
    def _check_performance_alerts(self, model_id: str, metrics: Dict[str, Any]):
        """Check if performance metrics trigger any alerts"""
        if model_id not in self.thresholds:
            return
        
        for metric_name, threshold_config in self.thresholds[model_id].items():
            if metric_name in metrics:
                current_value = metrics[metric_name]
                threshold_value = threshold_config['threshold']
                condition = threshold_config['condition']
                
                alert_triggered = False
                
                if condition == 'below' and current_value < threshold_value:
                    alert_triggered = True
                elif condition == 'above' and current_value > threshold_value:
                    alert_triggered = True
                elif condition == 'change':
                    # Check for significant change from baseline
                    baseline = self._get_baseline_metric(model_id, metric_name)
                    if baseline and abs(current_value - baseline) > threshold_value:
                        alert_triggered = True
                
                if alert_triggered:
                    alert = {
                        'model_id': model_id,
                        'metric_name': metric_name,
                        'current_value': current_value,
                        'threshold_value': threshold_value,
                        'condition': condition,
                        'timestamp': datetime.now().isoformat(),
                        'severity': self._determine_alert_severity(current_value, threshold_value, condition)
                    }
                    
                    self.alerts.append(alert)
                    print(f"🚨 ALERT: {model_id}.{metric_name} = {current_value} ({condition} {threshold_value})")
    
    def _get_baseline_metric(self, model_id: str, metric_name: str) -> Optional[float]:
        """Get baseline value for a metric"""
        model_records = [r for r in self.performance_history if r['model_id'] == model_id]
        
        if len(model_records) >= 5:  # Need at least 5 records for baseline
            recent_values = [r['metrics'].get(metric_name) for r in model_records[-5:]]
            recent_values = [v for v in recent_values if v is not None]
            
            if recent_values:
                return np.mean(recent_values)
        
        return None
    
    def _determine_alert_severity(self, current_value: float, threshold_value: float, 
                                 condition: str) -> str:
        """Determine alert severity based on how far from threshold"""
        try:
            if condition == 'below':
                deviation = (threshold_value - current_value) / (threshold_value + 1e-10)
            elif condition == 'above':
                deviation = (current_value - threshold_value) / (threshold_value + 1e-10)
            else:
                deviation = abs(current_value - threshold_value) / (threshold_value + 1e-10)
            
            if deviation > 0.2:  # 20% deviation
                return 'critical'
            elif deviation > 0.1:  # 10% deviation
                return 'warning'
            else:
                return 'info'
                
        except:
            return 'info'
    
    def _calculate_performance_hash(self, metrics: Dict[str, Any]) -> str:
        """Calculate hash for performance fingerprinting"""
        try:
            # Create a simplified hash based on key metrics
            key_metrics = ['accuracy', 'f1_weighted', 'r2_score', 'rmse']
            hash_string = ""
            
            for metric in key_metrics:
                if metric in metrics:
                    hash_string += f"{metric}:{metrics[metric]:.4f}|"
            
            return str(hash(hash_string))
            
        except:
            return str(hash(str(metrics)))
    
    def get_performance_trend(self, model_id: str, metric_name: str, 
                            days: int = 30) -> Dict[str, Any]:
        """Get performance trend for a specific metric"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter records for this model and time period
            filtered_records = []
            for record in self.performance_history:
                if record['model_id'] == model_id:
                    try:
                        record_date = datetime.fromisoformat(record['timestamp'])
                        if record_date >= cutoff_date:
                            if metric_name in record['metrics']:
                                filtered_records.append({
                                    'timestamp': record['timestamp'],
                                    'value': record['metrics'][metric_name]
                                })
                    except:
                        continue
            
            if not filtered_records:
                return {'error': 'No data found for the specified period'}
            
            # Calculate trend statistics
            values = [r['value'] for r in filtered_records]
            timestamps = [r['timestamp'] for r in filtered_records]
            
            trend_analysis = {
                'model_id': model_id,
                'metric_name': metric_name,
                'period_days': days,
                'data_points': len(values),
                'current_value': values[-1] if values else None,
                'mean_value': float(np.mean(values)),
                'std_value': float(np.std(values)),
                'min_value': float(np.min(values)),
                'max_value': float(np.max(values)),
                'trend_direction': self._calculate_trend_direction(values),
                'volatility': float(np.std(values) / (np.mean(values) + 1e-10)),
                'timestamps': timestamps,
                'values': values
            }
            
            return trend_analysis
            
        except Exception as e:
            print(f"❌ Trend analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate overall trend direction"""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
            
            if abs(slope) < 0.001:  # Very small slope
                return 'stable'
            elif slope > 0:
                return 'improving'
            else:
                return 'declining'
        except:
            return 'unknown'
    
    def get_model_comparison(self, metric_name: str, days: int = 30) -> Dict[str, Any]:
        """Compare performance across all models"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            model_performance = {}
            
            # Collect recent performance for each model
            for record in self.performance_history:
                try:
                    record_date = datetime.fromisoformat(record['timestamp'])
                    if record_date >= cutoff_date:
                        model_id = record['model_id']
                        
                        if metric_name in record['metrics']:
                            if model_id not in model_performance:
                                model_performance[model_id] = []
                            
                            model_performance[model_id].append(record['metrics'][metric_name])
                except:
                    continue
            
            # Calculate comparison statistics
            comparison = {}
            for model_id, values in model_performance.items():
                comparison[model_id] = {
                    'mean_performance': float(np.mean(values)),
                    'std_performance': float(np.std(values)),
                    'best_performance': float(np.max(values)),
                    'worst_performance': float(np.min(values)),
                    'data_points': len(values),
                    'trend': self._calculate_trend_direction(values)
                }
            
            # Rank models
            if comparison:
                ranked_models = sorted(
                    comparison.items(),
                    key=lambda x: x[1]['mean_performance'],
                    reverse=True  # Assume higher is better
                )
                
                comparison_result = {
                    'metric_name': metric_name,
                    'period_days': days,
                    'models_compared': len(comparison),
                    'ranking': [{'model_id': model_id, **stats} for model_id, stats in ranked_models],
                    'best_model': ranked_models[0][0] if ranked_models else None,
                    'worst_model': ranked_models[-1][0] if ranked_models else None
                }
                
                return comparison_result
            
            return {'error': 'No performance data found for comparison'}
            
        except Exception as e:
            print(f"❌ Model comparison failed: {e}")
            return {'error': str(e)}
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active performance alerts"""
        if severity:
            return [alert for alert in self.alerts if alert['severity'] == severity]
        return self.alerts.copy()
    
    def clear_alerts(self, model_id: Optional[str] = None):
        """Clear alerts for a specific model or all alerts"""
        if model_id:
            self.alerts = [alert for alert in self.alerts if alert['model_id'] != model_id]
            print(f"🧹 Cleared alerts for model {model_id}")
        else:
            self.alerts = []
            print("🧹 Cleared all alerts")
    
    def export_performance_data(self, model_id: Optional[str] = None, 
                               format: str = 'json') -> str:
        """Export performance data"""
        try:
            if model_id:
                data = [r for r in self.performance_history if r['model_id'] == model_id]
            else:
                data = self.performance_history
            
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'model_id_filter': model_id,
                'total_records': len(data),
                'performance_data': data,
                'model_registry': self.model_registry,
                'active_alerts': self.alerts
            }
            
            if format == 'json':
                return json.dumps(export_data, indent=2, default=str)
            else:
                return str(export_data)
                
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return ""
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        try:
            report = {
                'report_generated_at': datetime.now().isoformat(),
                'total_models_tracked': len(self.model_registry),
                'total_performance_records': len(self.performance_history),
                'active_alerts': len(self.alerts),
                'models_overview': {},
                'alert_summary': {
                    'critical': len([a for a in self.alerts if a['severity'] == 'critical']),
                    'warning': len([a for a in self.alerts if a['severity'] == 'warning']),
                    'info': len([a for a in self.alerts if a['severity'] == 'info'])
                }
            }
            
            # Model overview
            for model_id, model_info in self.model_registry.items():
                model_records = [r for r in self.performance_history if r['model_id'] == model_id]
                
                report['models_overview'][model_id] = {
                    'registered_at': model_info['registered_at'],
                    'last_updated': model_info['last_updated'],
                    'performance_records': len(model_records),
                    'model_type': model_info.get('model_type', 'unknown'),
                    'latest_performance': model_records[-1]['metrics'] if model_records else None
                }
            
            return report
            
        except Exception as e:
            print(f"❌ Summary report generation failed: {e}")
            return {'error': str(e)}

# =====================================
# USAGE EXAMPLE AND TESTING
# =====================================

if __name__ == "__main__":
    print("🚀 Testing Model Evaluation System...")
    
    # Test data generation
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic financial data
    X = np.random.randn(n_samples, 10)  # 10 financial features
    
    # For regression: predict financial health score
    y_regression = (X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1)
    y_pred_regression = y_regression + np.random.randn(n_samples) * 0.2
    
    # For classification: predict risk categories
    y_classification = np.random.randint(0, 4, n_samples)  # 4 risk categories
    y_pred_classification = y_classification + np.random.randint(-1, 2, n_samples)
    y_pred_classification = np.clip(y_pred_classification, 0, 3)
    
    # Generate probabilities for classification
    y_prob = np.random.dirichlet(np.ones(4), n_samples)
    
    print("\n📊 Testing Regression Metrics...")
    reg_metrics = RegressionMetrics()  # type: ignore
    reg_results = reg_metrics.calculate_all_metrics(y_regression, y_pred_regression)
    print(f"✅ R² Score: {reg_results.get('r2_score', 'N/A'):.4f}")
    print(f"✅ RMSE: {reg_results.get('rmse', 'N/A'):.4f}")
    print(f"✅ Financial Accuracy: {reg_results.get('financial_accuracy', 'N/A'):.2f}%")
    
    print("\n📊 Testing Classification Metrics...")
    clf_metrics = ClassificationMetrics()  # type: ignore
    clf_results = clf_metrics.calculate_all_metrics(
        y_classification, y_pred_classification, y_prob,
        class_names=['Poor', 'Fair', 'Good', 'Excellent']
    )
    print(f"✅ Accuracy: {clf_results.get('accuracy', 'N/A'):.4f}")
    print(f"✅ F1 Weighted: {clf_results.get('f1_weighted', 'N/A'):.4f}")
    print(f"✅ Risk Precision: {clf_results.get('risk_precision', 'N/A'):.4f}")
    
    print("\n📊 Testing Cross Validation...")
    cv = CrossValidator() # type: ignore
    
    # Mock model for testing
    class MockModel:
        def fit(self, X, y): pass
        def predict(self, X): return np.random.randint(0, 4, len(X))
        def predict_proba(self, X): return np.random.dirichlet(np.ones(4), len(X))
    
    mock_model = MockModel()
    cv_results = cv._manual_cross_validation(mock_model, X, y_classification, n_splits=3)
    print(f"✅ CV Mean Score: {cv_results.get('manual_cv_mean', 'N/A'):.4f}")
    
    print("\n📊 Testing Performance Tracker...")
    tracker = PerformanceTracker()
    
    # Register models
    tracker.register_model('xgboost_model', {'model_type': 'classification', 'version': '1.0'})
    tracker.register_model('rf_model', {'model_type': 'classification', 'version': '1.0'})
    
    # Log performance
    tracker.log_performance('xgboost_model', clf_results)
    tracker.log_performance('rf_model', {**clf_results, 'accuracy': clf_results.get('accuracy', 0) - 0.05})
    
    # Set alerts
    tracker.set_alert_threshold('xgboost_model', 'accuracy', 0.7, 'below')
    
    # Get comparison
    comparison = tracker.get_model_comparison('accuracy', days=1)
    print(f"✅ Best Model: {comparison.get('best_model', 'N/A')}")
    
    # Generate summary
    summary = tracker.get_summary_report()
    print(f"✅ Total Models Tracked: {summary.get('total_models_tracked', 0)}")
    print(f"✅ Total Records: {summary.get('total_performance_records', 0)}")
    
    print("\n🎉 Model Evaluation System Test Completed Successfully!")
    print("=" * 60)