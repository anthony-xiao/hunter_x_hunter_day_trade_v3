interface SystemHealthProps {
  status: {
    status: string
    timestamp: string
    trading_active: boolean
    components: {
      data_pipeline: boolean
      feature_engineer: boolean
      model_trainer: boolean
      signal_generator: boolean
      execution_engine: boolean
      risk_manager: boolean
    }
  } | null
}

interface StatusItemProps {
  label: string
  status: string
  description?: string
}

const StatusItem = ({ label, status, description }: StatusItemProps) => {
  const getStatusConfig = (status: string) => {
    if (!status) {
      return {
        color: 'text-gray-400',
        text: 'Unknown'
      }
    }
    
    switch (status.toLowerCase()) {
      case 'connected':
      case 'active':
      case 'loaded':
      case 'running':
        return {
          color: 'text-green-400',
          text: 'Healthy'
        }
      case 'disconnected':
      case 'inactive':
      case 'error':
      case 'failed':
        return {
          color: 'text-red-400',
          text: 'Error'
        }
      case 'loading':
      case 'initializing':
      case 'starting':
        return {
          color: 'text-yellow-400',
          text: 'Loading'
        }
      default:
        return {
          color: 'text-gray-400',
          text: 'Unknown'
        }
    }
  }

  const config = getStatusConfig(status)

  return (
    <div className="flex items-center justify-between p-3 bg-gray-800 rounded border border-gray-700">
      <div>
        <h4 className="text-sm font-medium text-white">{label}</h4>
        {description && (
          <p className="text-xs text-gray-400">{description}</p>
        )}
      </div>
      <div className="flex items-center space-x-2">
        <span className={`text-sm font-medium ${config.color}`}>
          {config.text}
        </span>
        <span className="text-xs text-gray-500 capitalize">
          {status}
        </span>
      </div>
    </div>
  )
}

export const SystemHealth = ({ status }: SystemHealthProps) => {
  if (!status) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-700 p-6">
        <h2 className="text-lg font-medium text-white mb-4">System Health</h2>
        <div className="flex items-center justify-center py-8">
          <span className="text-gray-400">Checking system status...</span>
        </div>
      </div>
    )
  }



  const getOverallHealth = () => {
    if (!status) {
      return {
        status: 'unknown' as const,
        color: 'text-gray-400',
        message: 'System status unknown'
      }
    }
    
    const statuses = [
      status.components.data_pipeline,
      status.components.execution_engine,
      status.components.signal_generator,
      status.components.model_trainer
    ]

    const healthyCount = statuses.filter(s => s === true).length
    const errorCount = statuses.filter(s => s === false).length

    if (errorCount > 0) {
      return {
        status: 'degraded',
        color: 'text-yellow-400',
        message: `${errorCount} component(s) need attention`
      }
    }

    if (healthyCount === statuses.length) {
      return {
        status: 'healthy',
        color: 'text-green-400',
        message: 'All systems operational'
      }
    }

    return {
      status: 'unknown',
      color: 'text-gray-400',
      message: 'System status unclear'
    }
  }

  const overallHealth = getOverallHealth()

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-medium text-white">System Health</h2>
        <div className="text-xs text-gray-400">
          Last updated: {new Date(status.timestamp).toLocaleTimeString()}
        </div>
      </div>

      {/* Overall Status */}
      <div className="mb-6 p-4 bg-gray-800 rounded border border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-medium text-white">Overall System Status</h3>
            <p className="text-sm text-gray-400">{overallHealth.message}</p>
          </div>
          <span className={`text-sm font-medium ${overallHealth.color} capitalize`}>
            {overallHealth.status}
          </span>
        </div>
      </div>

      {/* Component Status */}
      <div className="space-y-3">
        <StatusItem
          label="Data Pipeline"
          status={status.components.data_pipeline ? 'active' : 'inactive'}
          description="Polygon.io data ingestion and processing"
        />
        
        <StatusItem
          label="Execution Engine"
          status={status.components.execution_engine ? 'active' : 'inactive'}
          description="Alpaca API integration and order execution"
        />
        
        <StatusItem
          label="Signal Generator"
          status={status.components.signal_generator ? 'active' : 'inactive'}
          description="Trading signal generation and analysis"
        />
        
        <StatusItem
          label="Model Trainer"
          status={status.components.model_trainer ? 'active' : 'inactive'}
          description="Machine learning model training and optimization"
        />
        
        <StatusItem
          label="Feature Engineer"
          status={status.components.feature_engineer ? 'active' : 'inactive'}
          description="Feature extraction and engineering"
        />
        
        <StatusItem
          label="Risk Manager"
          status={status.components.risk_manager ? 'active' : 'inactive'}
          description="Portfolio risk assessment and management"
        />
      </div>

      {/* Quick Actions */}
      <div className="mt-6 pt-4 border-t border-gray-700">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-400">System Actions</span>
          <div className="flex space-x-2">
            <button 
              onClick={() => window.location.reload()}
              className="text-xs px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
            >
              Refresh Status
            </button>
            <button 
              onClick={() => window.open('http://localhost:8000/docs', '_blank')}
              className="text-xs px-3 py-1 bg-gray-700 text-gray-300 rounded hover:bg-gray-600 transition-colors"
            >
              API Docs
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}