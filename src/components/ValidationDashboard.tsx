import React, { useState, useEffect } from 'react';
import { toast } from 'sonner';

interface PerformanceMetrics {
  annual_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  trades_per_day: number;
  profit_factor: number;
  volatility: number;
}

interface ValidationResult {
  passed: boolean;
  metrics: PerformanceMetrics;
  recommendations: string[];
  risk_warnings: string[];
}

interface DriftAlert {
  timestamp: string;
  model_name: string;
  drift_type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  drift_score: number;
  affected_metrics: string[];
  recommended_action: string;
}

interface WalkForwardResult {
  total_periods: number;
  successful_periods: number;
  average_sharpe: number;
  consistency_score: number;
  best_period: {
    start_date: string;
    end_date: string;
    sharpe_ratio: number;
  };
  worst_period: {
    start_date: string;
    end_date: string;
    sharpe_ratio: number;
  };
}

const ValidationDashboard: React.FC = () => {
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [driftAlerts, setDriftAlerts] = useState<DriftAlert[]>([]);
  const [walkForwardResult, setWalkForwardResult] = useState<WalkForwardResult | null>(null);
  const [loading, setLoading] = useState({
    validation: false,
    drift: false,
    walkForward: false,
    optimization: false
  });
  const [walkForwardDates, setWalkForwardDates] = useState({
    startDate: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    endDate: new Date().toISOString().split('T')[0]
  });
  const [validationDays, setValidationDays] = useState(30);

  useEffect(() => {
    loadValidationData();
    loadDriftData();
  }, []);

  const loadValidationData = async () => {
    setLoading(prev => ({ ...prev, validation: true }));
    try {
      const response = await fetch(`http://localhost:8000/api/models/performance-validation?days=${validationDays}`);
      const data = await response.json();
      
      if (data.status === 'success') {
        setValidationResult(data.validation_result);
      } else {
        toast.error('Failed to load validation data');
      }
    } catch (error) {
      console.error('Error loading validation data:', error);
      toast.error('Error loading validation data');
    } finally {
      setLoading(prev => ({ ...prev, validation: false }));
    }
  };

  const loadDriftData = async () => {
    setLoading(prev => ({ ...prev, drift: true }));
    try {
      const response = await fetch('http://localhost:8000/api/models/drift-detection');
      const data = await response.json();
      
      if (data.status === 'success') {
        setDriftAlerts(data.recent_alerts);
      } else {
        toast.error('Failed to load drift data');
      }
    } catch (error) {
      console.error('Error loading drift data:', error);
      toast.error('Error loading drift data');
    } finally {
      setLoading(prev => ({ ...prev, drift: false }));
    }
  };

  const runWalkForwardTest = async () => {
    setLoading(prev => ({ ...prev, walkForward: true }));
    try {
      const response = await fetch('http://localhost:8000/api/models/walk-forward-test', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          start_date: walkForwardDates.startDate,
          end_date: walkForwardDates.endDate
        })
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setWalkForwardResult(data.results);
        toast.success('Walk-forward test completed successfully');
      } else {
        toast.error('Walk-forward test failed');
      }
    } catch (error) {
      console.error('Error running walk-forward test:', error);
      toast.error('Error running walk-forward test');
    } finally {
      setLoading(prev => ({ ...prev, walkForward: false }));
    }
  };

  const optimizeEnsemble = async () => {
    setLoading(prev => ({ ...prev, optimization: true }));
    try {
      const response = await fetch('http://localhost:8000/api/models/optimize-ensemble', {
        method: 'POST'
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        toast.success('Ensemble weights optimized successfully');
        // Reload validation data to see updated performance
        await loadValidationData();
      } else {
        toast.error('Ensemble optimization failed');
      }
    } catch (error) {
      console.error('Error optimizing ensemble:', error);
      toast.error('Error optimizing ensemble');
    } finally {
      setLoading(prev => ({ ...prev, optimization: false }));
    }
  };

  const formatPercentage = (value: number) => `${(value * 100).toFixed(2)}%`;
  const formatNumber = (value: number, decimals: number = 2) => value.toFixed(decimals);

  if (loading.validation) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-700 p-6">
        <h1 className="text-2xl font-bold text-white mb-4">System Validation Dashboard</h1>
        <div className="text-gray-400">Loading validation data...</div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">System Validation Dashboard</h1>
        <div className="flex gap-2">
          <button
            onClick={loadValidationData}
            disabled={loading.validation}
            className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm"
          >
            Refresh
          </button>
          <button
            onClick={optimizeEnsemble}
            disabled={loading.optimization}
            className="px-3 py-1 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 text-sm"
          >
            Optimize Ensemble
          </button>
        </div>
      </div>

      {/* Performance Validation */}
      <div className="space-y-4">
        <div className="flex items-center gap-4 mb-4">
          <div className="flex items-center gap-2">
            <label className="text-white text-sm">Days to validate:</label>
            <input
              type="number"
              value={validationDays}
              onChange={(e) => setValidationDays(Number(e.target.value))}
              className="w-20 px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white text-sm"
              min="1"
              max="365"
            />
            <button 
              onClick={loadValidationData} 
              disabled={loading.validation}
              className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 text-sm"
            >
              Update
            </button>
          </div>
          <div className="text-white">
            Status: <span className={validationResult?.passed ? 'text-green-400' : 'text-red-400'}>
              {validationResult?.passed ? 'PASSED' : 'FAILED'}
            </span>
          </div>
        </div>

        {validationResult && (
          <div className="bg-gray-800 rounded border border-gray-700 p-4">
            <h3 className="text-white font-medium mb-3">Performance Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-gray-400">Annual Return</div>
                <div className={`font-medium ${validationResult.metrics.annual_return > 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatPercentage(validationResult.metrics.annual_return)}
                </div>
              </div>
              <div>
                <div className="text-gray-400">Sharpe Ratio</div>
                <div className="text-white font-medium">{formatNumber(validationResult.metrics.sharpe_ratio)}</div>
              </div>
              <div>
                <div className="text-gray-400">Max Drawdown</div>
                <div className="text-red-400 font-medium">{formatPercentage(validationResult.metrics.max_drawdown)}</div>
              </div>
              <div>
                <div className="text-gray-400">Win Rate</div>
                <div className="text-white font-medium">{formatPercentage(validationResult.metrics.win_rate)}</div>
              </div>
              <div>
                <div className="text-gray-400">Trades/Day</div>
                <div className="text-white font-medium">{formatNumber(validationResult.metrics.trades_per_day)}</div>
              </div>
              <div>
                <div className="text-gray-400">Profit Factor</div>
                <div className="text-white font-medium">{formatNumber(validationResult.metrics.profit_factor)}</div>
              </div>
              <div>
                <div className="text-gray-400">Volatility</div>
                <div className="text-white font-medium">{formatPercentage(validationResult.metrics.volatility)}</div>
              </div>
            </div>
          </div>
        )}

        {validationResult?.recommendations && validationResult.recommendations.length > 0 && (
          <div className="bg-gray-800 rounded border border-gray-700 p-4">
            <h3 className="text-white font-medium mb-3">Recommendations</h3>
            <ul className="space-y-1 text-sm text-gray-300">
              {validationResult.recommendations.map((rec, index) => (
                <li key={index}>• {rec}</li>
              ))}
            </ul>
          </div>
        )}

        {validationResult?.risk_warnings && validationResult.risk_warnings.length > 0 && (
          <div className="bg-gray-800 rounded border border-gray-700 p-4">
            <h3 className="text-orange-400 font-medium mb-3">Risk Warnings</h3>
            <ul className="space-y-1 text-sm text-orange-300">
              {validationResult.risk_warnings.map((warning, index) => (
                <li key={index}>⚠ {warning}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Concept Drift Detection */}
        <div className="bg-gray-800 rounded border border-gray-700 p-4">
          <h3 className="text-white font-medium mb-3">Concept Drift Detection</h3>
          <div className="space-y-3">
            {driftAlerts.length === 0 ? (
              <div className="text-center py-4 text-gray-400">
                No drift alerts detected
              </div>
            ) : (
              driftAlerts.map((alert, index) => (
                <div key={index} className="bg-gray-700 rounded p-3 border-l-4 border-orange-500">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-white">{alert.model_name}</span>
                      <span className={`px-2 py-1 rounded text-xs ${
                        alert.severity === 'critical' ? 'bg-red-600 text-white' :
                        alert.severity === 'high' ? 'bg-orange-600 text-white' :
                        alert.severity === 'medium' ? 'bg-yellow-600 text-black' :
                        'bg-blue-600 text-white'
                      }`}>
                        {alert.severity.toUpperCase()}
                      </span>
                    </div>
                    <span className="text-xs text-gray-400">
                      {new Date(alert.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="space-y-1 text-sm text-gray-300">
                    <p><strong>Drift Type:</strong> {alert.drift_type}</p>
                    <p><strong>Score:</strong> {formatNumber(alert.drift_score)}</p>
                    <p><strong>Affected Metrics:</strong> {alert.affected_metrics.join(', ')}</p>
                    <p><strong>Recommended Action:</strong> {alert.recommended_action}</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Walk-Forward Testing */}
        <div className="bg-gray-800 rounded border border-gray-700 p-4">
          <h3 className="text-white font-medium mb-3">Walk-Forward Testing</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="text-white text-sm block mb-1">Start Date</label>
              <input
                type="date"
                value={walkForwardDates.startDate}
                onChange={(e) => setWalkForwardDates(prev => ({ ...prev, startDate: e.target.value }))}
                className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
              />
            </div>
            <div>
              <label className="text-white text-sm block mb-1">End Date</label>
              <input
                type="date"
                value={walkForwardDates.endDate}
                onChange={(e) => setWalkForwardDates(prev => ({ ...prev, endDate: e.target.value }))}
                className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm"
              />
            </div>
          </div>
          
          <button 
            onClick={runWalkForwardTest} 
            disabled={loading.walkForward}
            className="w-full mb-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {loading.walkForward ? 'Running...' : 'Run Walk-Forward Test'}
          </button>

          {walkForwardResult && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="text-gray-400">Total Periods</div>
                <div className="text-white font-medium">{walkForwardResult.total_periods}</div>
              </div>
              <div>
                <div className="text-gray-400">Success Rate</div>
                <div className="text-white font-medium">
                  {formatPercentage(walkForwardResult.successful_periods / walkForwardResult.total_periods)}
                </div>
              </div>
              <div>
                <div className="text-gray-400">Avg Sharpe</div>
                <div className="text-white font-medium">{formatNumber(walkForwardResult.average_sharpe)}</div>
              </div>
              <div>
                <div className="text-gray-400">Consistency</div>
                <div className="text-white font-medium">{formatPercentage(walkForwardResult.consistency_score)}</div>
              </div>
              <div className="md:col-span-2">
                <div className="text-gray-400">Best Period</div>
                <div className="text-sm text-gray-300">
                  {walkForwardResult.best_period.start_date} to {walkForwardResult.best_period.end_date}
                </div>
                <div className="text-green-400 font-medium">
                  Sharpe: {formatNumber(walkForwardResult.best_period.sharpe_ratio)}
                </div>
              </div>
              <div className="md:col-span-2">
                <div className="text-gray-400">Worst Period</div>
                <div className="text-sm text-gray-300">
                  {walkForwardResult.worst_period.start_date} to {walkForwardResult.worst_period.end_date}
                </div>
                <div className="text-red-400 font-medium">
                  Sharpe: {formatNumber(walkForwardResult.worst_period.sharpe_ratio)}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Live Trading Readiness */}
        <div className="bg-gray-800 rounded border border-gray-700 p-4">
          <h3 className="text-white font-medium mb-3">Live Trading Readiness</h3>
          <div className="text-center py-4">
            <p className="text-gray-400 mb-4">
              Comprehensive system validation for live trading readiness
            </p>
            <button 
              onClick={async () => {
                try {
                  const response = await fetch('/api/system/validate-readiness');
                  const data = await response.json();
                  
                  if (data.status === 'success') {
                    toast.success('Live readiness validation completed');
                    console.log('Readiness result:', data.validation_result);
                  } else {
                    toast.error('Live readiness validation failed');
                  }
                } catch (error) {
                  console.error('Error validating readiness:', error);
                  toast.error('Error validating readiness');
                }
              }}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            >
              Validate Live Readiness
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ValidationDashboard;