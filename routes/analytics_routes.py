# =====================================
# File: routes/analytics_routes.py
# Analytics Routes with Confidence Score Support
# =====================================

"""
Analytics routes for confidence score analysis and visualization
"""

from flask import Blueprint, request, jsonify, render_template # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import json
from datetime import datetime
import traceback
from typing import Dict, List, Any, Optional

# Import ML predictor
from utils.ml_predictor import MLPredictor, format_confidence_response, get_confidence_recommendations

# Create blueprint
analytics_bp = Blueprint('analytics', __name__, url_prefix='/analytics')

# Initialize ML predictor
ml_predictor = MLPredictor()

@analytics_bp.route('/')
def analytics_dashboard():
    """Analytics dashboard page"""
    return render_template('analytics/dashboard.html')

@analytics_bp.route('/confidence')
def confidence_dashboard():
    """Confidence analysis dashboard"""
    return render_template('analytics/confidence_dashboard.html')

@analytics_bp.route('/api/predict_with_confidence', methods=['POST'])
def predict_with_confidence():
    """
    Make predictions with detailed confidence analysis
    Enhanced endpoint specifically for confidence scoring
    """
    try:
        data = request.json
        model_name = data.get('model_name')
        
        # Get models from app context (you'll need to import this)
        from app import models
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 404
        
        model_info = models[model_name]
        
        if not model_info['is_trained']:
            return jsonify({'error': f'Model {model_name} is not trained'}), 400
        
        # Get prediction data
        X_pred = np.array(data.get('X_pred'))
        feature_names = data.get('feature_names')
        confidence_threshold = data.get('confidence_threshold', 0.7)
        include_details = data.get('include_details', True)
        
        # Convert to DataFrame if feature names provided
        if feature_names:
            X_pred = pd.DataFrame(X_pred, columns=feature_names)
        
        model = model_info['model']
        
        # Enhanced model info
        enhanced_model_info = {
            'name': model_name,
            'type': model_info['type'],
            'algorithm': model_info['algorithm'],
            'scenario': model_info.get('scenario', 'unknown'),
            'created_at': model_info.get('created_at'),
            'trained_at': model_info.get('trained_at')
        }
        
        # Get predictions with confidence
        result = ml_predictor.predict_with_confidence(
            model, enhanced_model_info, X_pred, confidence_threshold
        )
        
        # Format response
        formatted_result = format_confidence_response(result, include_details)
        
        # Add recommendations
        if 'confidence_summary' in result:
            formatted_result['recommendations'] = get_confidence_recommendations(
                result['confidence_summary']
            )
        
        # Add timestamp
        formatted_result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(formatted_result)
        
    except Exception as e:
        return jsonify({
            'error': str(e), 
            'traceback': traceback.format_exc()
        }), 500

@analytics_bp.route('/api/confidence_analysis/<model_name>', methods=['POST'])
def confidence_analysis(model_name):
    """
    Detailed confidence calibration analysis
    """
    try:
        from app import models
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 404
        
        model_info = models[model_name]
        
        if not model_info['is_trained']:
            return jsonify({'error': f'Model {model_name} is not trained'}), 400
        
        data = request.json
        X_test = np.array(data.get('X_test'))
        y_test = np.array(data.get('y_test')) if data.get('y_test') else None
        feature_names = data.get('feature_names')
        
        # Convert to DataFrame if feature names provided
        if feature_names:
            X_test = pd.DataFrame(X_test, columns=feature_names)
        
        model = model_info['model']
        
        # Enhanced model info
        enhanced_model_info = {
            'name': model_name,
            'type': model_info['type'],
            'algorithm': model_info['algorithm'],
            'scenario': model_info.get('scenario', 'unknown')
        }
        
        # Perform confidence analysis
        if y_test is not None:
            # With ground truth - full calibration analysis
            analysis = ml_predictor.analyze_confidence_calibration(
                model, enhanced_model_info, X_test, y_test
            )
        else:
            # Without ground truth - basic confidence analysis
            confidence_data = ml_predictor.predict_with_confidence(
                model, enhanced_model_info, X_test
            )
            
            analysis = {
                'model_name': model_name,
                'model_type': model_info['type'],
                'algorithm': model_info['algorithm'],
                'n_samples': len(X_test),
                'confidence_summary': confidence_data.get('confidence_summary', {}),
                'note': 'Ground truth not provided. Limited analysis available.'
            }
        
        # Add recommendations
        if 'confidence_summary' in analysis:
            analysis['recommendations'] = get_confidence_recommendations(
                analysis['confidence_summary']
            )
        elif 'confidence_statistics' in analysis:
            analysis['recommendations'] = get_confidence_recommendations(
                analysis['confidence_statistics']
            )
        
        analysis['timestamp'] = datetime.now().isoformat()
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@analytics_bp.route('/api/batch_confidence_analysis', methods=['POST'])
def batch_confidence_analysis():
    """
    Analyze confidence for multiple models at once
    """
    try:
        from app import models
        
        data = request.json
        model_names = data.get('model_names', [])
        X_test = np.array(data.get('X_test'))
        y_test = np.array(data.get('y_test')) if data.get('y_test') else None
        feature_names = data.get('feature_names')
        confidence_threshold = data.get('confidence_threshold', 0.7)
        
        if not model_names:
            return jsonify({'error': 'model_names list is required'}), 400
        
        # Convert to DataFrame if feature names provided
        if feature_names:
            X_test = pd.DataFrame(X_test, columns=feature_names)
        
        batch_results = {
            'timestamp': datetime.now().isoformat(),
            'confidence_threshold': confidence_threshold,
            'n_samples': len(X_test),
            'models_analyzed': len(model_names),
            'results': {}
        }
        
        for model_name in model_names:
            if model_name not in models:
                batch_results['results'][model_name] = {
                    'error': f'Model {model_name} not found'
                }
                continue
            
            model_info = models[model_name]
            
            if not model_info['is_trained']:
                batch_results['results'][model_name] = {
                    'error': f'Model {model_name} is not trained'
                }
                continue
            
            try:
                model = model_info['model']
                
                # Enhanced model info
                enhanced_model_info = {
                    'name': model_name,
                    'type': model_info['type'],
                    'algorithm': model_info['algorithm'],
                    'scenario': model_info.get('scenario', 'unknown')
                }
                
                # Get confidence analysis
                if y_test is not None:
                    analysis = ml_predictor.analyze_confidence_calibration(
                        model, enhanced_model_info, X_test, y_test
                    )
                else:
                    confidence_data = ml_predictor.predict_with_confidence(
                        model, enhanced_model_info, X_test, confidence_threshold
                    )
                    analysis = {
                        'confidence_summary': confidence_data.get('confidence_summary', {}),
                        'mean_confidence': confidence_data.get('confidence_summary', {}).get('mean_confidence', 0)
                    }
                
                batch_results['results'][model_name] = analysis
                
            except Exception as e:
                batch_results['results'][model_name] = {
                    'error': str(e)
                }
        
        # Add comparative analysis
        successful_results = {k: v for k, v in batch_results['results'].items() 
                            if 'error' not in v}
        
        if successful_results:
            # Compare mean confidences
            confidences = {}
            for model_name, result in successful_results.items():
                if 'confidence_summary' in result:
                    confidences[model_name] = result['confidence_summary'].get('mean_confidence', 0)
                elif 'confidence_statistics' in result:
                    confidences[model_name] = result['confidence_statistics'].get('mean', 0)
            
            if confidences:
                best_model = max(confidences.items(), key=lambda x: x[1])
                worst_model = min(confidences.items(), key=lambda x: x[1])
                
                batch_results['comparative_analysis'] = {
                    'best_confidence_model': {
                        'name': best_model[0],
                        'confidence': best_model[1]
                    },
                    'worst_confidence_model': {
                        'name': worst_model[0],
                        'confidence': worst_model[1]
                    },
                    'confidence_range': best_model[1] - worst_model[1],
                    'mean_confidence_across_models': float(np.mean(list(confidences.values())))
                }
        
        return jsonify(batch_results)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@analytics_bp.route('/api/confidence_trends/<model_name>')
def confidence_trends(model_name):
    """
    Get confidence trends over time (if performance tracker available)
    """
    try:
        from app import performance_tracker, models
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 404
        
        if not performance_tracker:
            return jsonify({'error': 'Performance tracker not available'}), 400
        
        # Get historical performance data
        try:
            model_history = performance_tracker.get_model_history(model_name)
            
            if not model_history:
                return jsonify({'message': 'No historical data available for this model'})
            
            # Extract confidence-related metrics over time
            trends = {
                'model_name': model_name,
                'data_points': len(model_history),
                'trends': {
                    'timestamps': [],
                    'accuracy': [],
                    'confidence_metrics': []
                }
            }
            
            for entry in model_history:
                trends['trends']['timestamps'].append(entry.get('timestamp'))
                trends['trends']['accuracy'].append(entry.get('accuracy', 0))
                
                # Extract confidence-related metrics if available
                confidence_metric = entry.get('mean_confidence', entry.get('confidence_score', 0))
                trends['trends']['confidence_metrics'].append(confidence_metric)
            
            return jsonify(trends)
            
        except Exception as e:
            return jsonify({'error': f'Failed to get trends: {e}'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analytics_bp.route('/api/model_comparison_confidence', methods=['POST'])
def model_comparison_confidence():
    """
    Compare confidence scores across multiple models on same dataset
    """
    try:
        from app import models
        
        data = request.json
        model_names = data.get('model_names', [])
        X_test = np.array(data.get('X_test'))
        feature_names = data.get('feature_names')
        comparison_metrics = data.get('metrics', ['mean_confidence', 'high_confidence_percentage'])
        
        if not model_names:
            return jsonify({'error': 'model_names list is required'}), 400
        
        if len(model_names) < 2:
            return jsonify({'error': 'At least 2 models required for comparison'}), 400
        
        # Convert to DataFrame if feature names provided
        if feature_names:
            X_test = pd.DataFrame(X_test, columns=feature_names)
        
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X_test),
            'models_compared': len(model_names),
            'comparison_metrics': comparison_metrics,
            'results': {},
            'rankings': {}
        }
        
        # Get confidence data for each model
        model_confidences = {}
        
        for model_name in model_names:
            if model_name not in models:
                comparison_results['results'][model_name] = {
                    'error': f'Model {model_name} not found'
                }
                continue
            
            model_info = models[model_name]
            
            if not model_info['is_trained']:
                comparison_results['results'][model_name] = {
                    'error': f'Model {model_name} is not trained'
                }
                continue
            
            try:
                model = model_info['model']
                
                # Enhanced model info
                enhanced_model_info = {
                    'name': model_name,
                    'type': model_info['type'],
                    'algorithm': model_info['algorithm'],
                    'scenario': model_info.get('scenario', 'unknown')
                }
                
                # Get confidence data
                confidence_data = ml_predictor.predict_with_confidence(
                    model, enhanced_model_info, X_test
                )
                
                # Extract comparison metrics
                summary = confidence_data.get('confidence_summary', {})
                
                model_result = {
                    'model_type': model_info['type'],
                    'algorithm': model_info['algorithm'],
                    'scenario': model_info.get('scenario'),
                    'confidence_summary': summary,
                    'predictions_count': len(confidence_data.get('predictions', [])),
                    'metrics': {}
                }
                
                # Calculate specific metrics for comparison
                for metric in comparison_metrics:
                    if metric == 'mean_confidence':
                        model_result['metrics'][metric] = summary.get('mean_confidence', 0)
                    elif metric == 'high_confidence_percentage':
                        model_result['metrics'][metric] = summary.get('high_confidence_percentage', 0)
                    elif metric == 'std_confidence':
                        model_result['metrics'][metric] = summary.get('std_confidence', 0)
                    elif metric == 'min_confidence':
                        model_result['metrics'][metric] = summary.get('min_confidence', 0)
                    elif metric == 'max_confidence':
                        model_result['metrics'][metric] = summary.get('max_confidence', 0)
                
                comparison_results['results'][model_name] = model_result
                model_confidences[model_name] = model_result['metrics']
                
            except Exception as e:
                comparison_results['results'][model_name] = {
                    'error': str(e)
                }
        
        # Create rankings for each metric
        for metric in comparison_metrics:
            metric_values = {name: data['metrics'].get(metric, 0) 
                           for name, data in model_confidences.items()}
            
            if metric_values:
                # Sort by metric value (higher is better for confidence metrics)
                sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                
                comparison_results['rankings'][metric] = [
                    {
                        'rank': i + 1,
                        'model_name': model_name,
                        'value': value,
                        'model_type': comparison_results['results'][model_name]['model_type']
                    }
                    for i, (model_name, value) in enumerate(sorted_models)
                ]
        
        # Add overall best model recommendation
        if model_confidences:
            # Calculate composite score (weighted average of metrics)
            weights = {
                'mean_confidence': 0.4,
                'high_confidence_percentage': 0.3,
                'std_confidence': -0.2,  # Lower std is better
                'min_confidence': 0.1
            }
            
            composite_scores = {}
            for model_name, metrics in model_confidences.items():
                score = 0
                weight_sum = 0
                
                for metric, weight in weights.items():
                    if metric in metrics:
                        if metric == 'std_confidence':
                            # Invert std_confidence (lower is better)
                            score += weight * (1 - min(metrics[metric], 1))
                        else:
                            score += weight * metrics[metric]
                        weight_sum += abs(weight)
                
                if weight_sum > 0:
                    composite_scores[model_name] = score / weight_sum
            
            if composite_scores:
                best_model = max(composite_scores.items(), key=lambda x: x[1])
                comparison_results['recommendation'] = {
                    'best_overall_model': best_model[0],
                    'composite_score': best_model[1],
                    'reasoning': 'Based on weighted combination of confidence metrics',
                    'weights_used': weights
                }
        
        return jsonify(comparison_results)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@analytics_bp.route('/api/confidence_report/<model_name>', methods=['POST'])
def generate_confidence_report(model_name):
    """
    Generate comprehensive confidence report for a model
    """
    try:
        from app import models
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 404
        
        model_info = models[model_name]
        
        if not model_info['is_trained']:
            return jsonify({'error': f'Model {model_name} is not trained'}), 400
        
        data = request.json
        X_test = np.array(data.get('X_test'))
        y_test = np.array(data.get('y_test')) if data.get('y_test') else None
        feature_names = data.get('feature_names')
        confidence_threshold = data.get('confidence_threshold', 0.7)
        include_recommendations = data.get('include_recommendations', True)
        
        # Convert to DataFrame if feature names provided
        if feature_names:
            X_test = pd.DataFrame(X_test, columns=feature_names)
        
        model = model_info['model']
        
        # Enhanced model info
        enhanced_model_info = {
            'name': model_name,
            'type': model_info['type'],
            'algorithm': model_info['algorithm'],
            'scenario': model_info.get('scenario', 'unknown'),
            'created_at': model_info.get('created_at'),
            'trained_at': model_info.get('trained_at')
        }
        
        # Generate comprehensive report
        report = {
            'report_metadata': {
                'model_name': model_name,
                'generated_at': datetime.now().isoformat(),
                'report_type': 'Confidence Analysis Report',
                'data_samples': len(X_test),
                'confidence_threshold': confidence_threshold
            },
            'model_information': enhanced_model_info
        }
        
        # Basic confidence analysis
        confidence_data = ml_predictor.predict_with_confidence(
            model, enhanced_model_info, X_test, confidence_threshold
        )
        
        report['confidence_analysis'] = {
            'summary': confidence_data.get('confidence_summary', {}),
            'distribution': confidence_data.get('confidence_summary', {}).get('confidence_distribution', {}),
            'sample_count_by_confidence': {
                'high_confidence': confidence_data.get('confidence_summary', {}).get('high_confidence_count', 0),
                'low_confidence': confidence_data.get('confidence_summary', {}).get('low_confidence_count', 0)
            }
        }
        
        # Calibration analysis if ground truth available
        if y_test is not None:
            calibration_analysis = ml_predictor.analyze_confidence_calibration(
                model, enhanced_model_info, X_test, y_test
            )
            report['calibration_analysis'] = calibration_analysis
        
        # Get model performance if available
        try:
            performance_metrics = model.evaluate(X_test, y_test) if y_test is not None else {}
            report['performance_metrics'] = performance_metrics
        except Exception as e:
            report['performance_metrics'] = {'error': f'Could not evaluate: {e}'}
        
        # Feature importance if available
        try:
            if hasattr(model, 'get_feature_importance'):
                feature_importance = model.get_feature_importance()
                report['feature_importance'] = dict(list(feature_importance.items())[:10])  # Top 10
        except Exception as e:
            report['feature_importance'] = {'error': f'Could not get importance: {e}'}
        
        # Recommendations
        if include_recommendations:
            recommendations = []
            
            # Confidence-based recommendations
            conf_summary = confidence_data.get('confidence_summary', {})
            recommendations.extend(get_confidence_recommendations(conf_summary))
            
            # Performance-based recommendations
            if y_test is not None and 'performance_metrics' in report:
                perf_metrics = report['performance_metrics']
                
                if model_info['algorithm'] == 'classifier':
                    accuracy = perf_metrics.get('accuracy', 0)
                    if accuracy < 0.7:
                        recommendations.append("📉 Model accuracy is below 70%. Consider retraining with more data or feature engineering.")
                    elif accuracy > 0.9:
                        recommendations.append("🎯 Excellent model accuracy. Consider this model for production use.")
                
                elif model_info['algorithm'] == 'regressor':
                    r2_score = perf_metrics.get('r2_score', 0)
                    if r2_score < 0.5:
                        recommendations.append("📉 Low R² score. Model may need improvement through feature engineering or algorithm tuning.")
                    elif r2_score > 0.8:
                        recommendations.append("🎯 Good R² score. Model shows strong predictive performance.")
            
            # Model-specific recommendations
            model_type = model_info['type']
            if model_type == 'neural_networks':
                recommendations.append("🧠 Neural network detected. Consider using uncertainty estimation for better confidence scores.")
            elif model_type == 'xgboost':
                recommendations.append("🚀 XGBoost model. Feature importance analysis can provide valuable insights.")
            
            report['recommendations'] = recommendations
        
        # Report summary
        report['executive_summary'] = {
            'overall_confidence': conf_summary.get('mean_confidence', 0),
            'high_confidence_predictions': f"{conf_summary.get('high_confidence_percentage', 0):.1f}%",
            'model_reliability': 'High' if conf_summary.get('mean_confidence', 0) > 0.8 else 
                               'Medium' if conf_summary.get('mean_confidence', 0) > 0.6 else 'Low',
            'ready_for_production': conf_summary.get('mean_confidence', 0) > 0.7 and 
                                   conf_summary.get('high_confidence_percentage', 0) > 60
        }
        
        return jsonify(report)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Helper routes for confidence visualization data
@analytics_bp.route('/api/confidence_chart_data/<model_name>', methods=['POST'])
def get_confidence_chart_data(model_name):
    """
    Get data formatted for confidence visualization charts
    """
    try:
        from app import models
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 404
        
        model_info = models[model_name]
        
        if not model_info['is_trained']:
            return jsonify({'error': f'Model {model_name} is not trained'}), 400
        
        data = request.json
        X_test = np.array(data.get('X_test'))
        feature_names = data.get('feature_names')
        
        # Convert to DataFrame if feature names provided
        if feature_names:
            X_test = pd.DataFrame(X_test, columns=feature_names)
        
        model = model_info['model']
        
        # Enhanced model info
        enhanced_model_info = {
            'name': model_name,
            'type': model_info['type'],
            'algorithm': model_info['algorithm']
        }
        
        # Get confidence data
        confidence_data = ml_predictor.predict_with_confidence(
            model, enhanced_model_info, X_test
        )
        
        # Format for charts
        chart_data = {
            'confidence_histogram': {
                'bins': ['0-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'],
                'counts': list(confidence_data.get('confidence_summary', {}).get('confidence_distribution', {}).values())
            },
            'confidence_scores': confidence_data.get('confidence_scores', []),
            'predictions': confidence_data.get('predictions', [])
        }
        
        # Add probability data for classification
        if model_info['algorithm'] == 'classifier':
            chart_data['probabilities'] = confidence_data.get('probabilities', [])
            
            # Class probability distribution
            if 'probabilities' in confidence_data:
                probs = np.array(confidence_data['probabilities'])
                n_classes = probs.shape[1]
                
                chart_data['class_probability_distribution'] = {}
                for i in range(n_classes):
                    class_name = f'class_{i}'
                    if hasattr(model, 'class_names') and model.class_names and i < len(model.class_names):
                        class_name = model.class_names[i]
                    
                    chart_data['class_probability_distribution'][class_name] = {
                        'mean': float(np.mean(probs[:, i])),
                        'std': float(np.std(probs[:, i])),
                        'values': probs[:, i].tolist()
                    }
        
        # Add uncertainty data for regression
        elif model_info['algorithm'] == 'regressor':
            if 'uncertainties' in confidence_data:
                chart_data['uncertainties'] = confidence_data['uncertainties']
                chart_data['uncertainty_vs_confidence'] = [
                    {'confidence': conf, 'uncertainty': unc}
                    for conf, unc in zip(
                        confidence_data.get('confidence_scores', []),
                        confidence_data.get('uncertainties', [])
                    )
                ]
        
        return jsonify(chart_data)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500