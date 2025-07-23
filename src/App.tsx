import { useState, useEffect } from 'react'
import { TradingDashboard } from './components/TradingDashboard'
import { SystemHealth } from './components/SystemHealth'
import { PositionsTable } from './components/PositionsTable'
import { ModelPerformance } from './components/ModelPerformance'
import { TradingControls } from './components/TradingControls'
import ValidationDashboard from './components/ValidationDashboard'

type TabType = 'control' | 'positions' | 'models' | 'data' | 'execution'

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('control')
  const [systemStatus, setSystemStatus] = useState<any>(null)

  useEffect(() => {
    // Check system health on startup
    checkSystemHealth()
    
    // Set up periodic health checks
    const interval = setInterval(checkSystemHealth, 30000) // Every 30 seconds
    
    return () => clearInterval(interval)
  }, [])

  const checkSystemHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/health/detailed')
      const data = await response.json()
      setSystemStatus(data)
    } catch (error) {
      console.error('Failed to check system health:', error)
      setSystemStatus({ error: 'Backend not available' })
    }
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case 'control':
        return (
          <div className="space-y-6">
            <SystemHealth status={systemStatus} />
            <TradingControls />
            <TradingDashboard />
          </div>
        )
      case 'positions':
        return <PositionsTable />
      case 'models':
        return <ModelPerformance />
      case 'data':
        return <ValidationDashboard />
      case 'execution':
        return (
          <div className="bg-[#333333] border border-[#666666] p-6">
            <h2 className="text-xl font-semibold mb-4 text-white">Position Monitor</h2>
            <PositionsTable />
          </div>
        )
      default:
        return null
    }
  }

  return (
    <div className="min-h-screen bg-[#1a1a1a]">
      {/* Header */}
      <header className="bg-[#333333] border-b border-[#666666]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <h1 className="text-xl font-bold text-white">
                Algorithmic Trading System
              </h1>
            </div>
            
            <div className="flex items-center space-x-2">
              <div className={`h-3 w-3 rounded-full ${
                systemStatus?.database_status === 'connected' && 
                systemStatus?.trading_engine_status === 'active'
                  ? 'bg-[#00ff88]' 
                  : 'bg-[#ff4444]'
              }`}></div>
              <span className="text-sm text-white">
                {systemStatus?.database_status === 'connected' ? 'Online' : 'Offline'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-[#333333] border-b border-[#666666]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            <button
              onClick={() => setActiveTab('control')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'control'
                  ? 'border-[#00ff88] text-[#00ff88]'
                  : 'border-transparent text-white hover:text-[#00ff88]'
              }`}
            >
              Control Panel
            </button>
            
            <button
              onClick={() => setActiveTab('positions')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'positions'
                  ? 'border-[#00ff88] text-[#00ff88]'
                  : 'border-transparent text-white hover:text-[#00ff88]'
              }`}
            >
              Portfolio Summary
            </button>
            
            <button
              onClick={() => setActiveTab('models')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'models'
                  ? 'border-[#00ff88] text-[#00ff88]'
                  : 'border-transparent text-white hover:text-[#00ff88]'
              }`}
            >
              Model Training
            </button>
            
            <button
              onClick={() => setActiveTab('data')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'data'
                  ? 'border-[#00ff88] text-[#00ff88]'
                  : 'border-transparent text-white hover:text-[#00ff88]'
              }`}
            >
              Data Pipeline
            </button>
            
            <button
              onClick={() => setActiveTab('execution')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'execution'
                  ? 'border-[#00ff88] text-[#00ff88]'
                  : 'border-transparent text-white hover:text-[#00ff88]'
              }`}
            >
              Trading Execution
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderTabContent()}
      </main>
    </div>
  )
}

export default App
