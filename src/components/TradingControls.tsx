import { useState } from 'react'

interface TradingControlsProps {
  onStatusChange?: () => void
}

export const TradingControls = ({ onStatusChange }: TradingControlsProps) => {
  const [isStarting, setIsStarting] = useState(false)
  const [isStopping, setIsStopping] = useState(false)
  const [tradingMode, setTradingMode] = useState<'paper' | 'live'>('paper')
  const [showConfirmStop, setShowConfirmStop] = useState(false)



  const handleStartTrading = async () => {
    setIsStarting(true)
    try {
      const response = await fetch('http://localhost:8000/trading/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })

      if (!response.ok) {
        throw new Error(`Failed to start trading: ${response.statusText}`)
      }

      const data = await response.json()
      console.log('Trading started:', data)
      
      // Notify parent component to refresh status
      if (onStatusChange) {
        onStatusChange()
      }
    } catch (error) {
      console.error('Error starting trading:', error)
      alert('Failed to start trading. Please check the system status.')
    } finally {
      setIsStarting(false)
    }
  }

  const handleStopTrading = async () => {
    setIsStopping(true)
    try {
      const response = await fetch('http://localhost:8000/trading/stop', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })

      if (!response.ok) {
        throw new Error(`Failed to stop trading: ${response.statusText}`)
      }

      const data = await response.json()
      console.log('Trading stopped:', data)
      
      // Notify parent component to refresh status
      if (onStatusChange) {
        onStatusChange()
      }
    } catch (error) {
      console.error('Error stopping trading:', error)
      alert('Failed to stop trading. Please check the system status.')
    } finally {
      setIsStopping(false)
      setShowConfirmStop(false)
    }
  }



  return (
    <div className="bg-[#333333] border border-[#666666] p-6">
      <h2 className="text-lg font-medium text-white mb-6">Trading Engine</h2>
      
      <div className="space-y-6">
        {/* Large Start/Stop Buttons */}
        <div className="flex space-x-4">
          <button
            onClick={handleStartTrading}
            disabled={isStarting}
            className="flex-1 px-8 py-6 bg-[#00ff88] text-black text-xl font-bold hover:bg-[#00cc6a] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isStarting ? 'STARTING...' : 'START TRADING'}
          </button>
          
          <button
            onClick={() => setShowConfirmStop(true)}
            disabled={isStopping}
            className="flex-1 px-8 py-6 bg-[#ff4444] text-white text-xl font-bold hover:bg-[#cc3333] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isStopping ? 'STOPPING...' : 'STOP TRADING'}
          </button>
        </div>
        
        {/* Simple Mode Toggle */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-white">Trading Mode</label>
          <div className="flex space-x-4">
            <button
              onClick={() => setTradingMode('paper')}
              className={`px-4 py-2 text-sm font-medium ${
                tradingMode === 'paper'
                  ? 'bg-[#00ff88] text-black'
                  : 'bg-[#666666] text-white hover:bg-[#777777]'
              } transition-colors`}
            >
              Paper Trading
            </button>
            <button
              onClick={() => setTradingMode('live')}
              className={`px-4 py-2 text-sm font-medium ${
                tradingMode === 'live'
                  ? 'bg-[#ff4444] text-white'
                  : 'bg-[#666666] text-white hover:bg-[#777777]'
              } transition-colors`}
            >
              Live Trading
            </button>
          </div>
        </div>
          
        {/* Stop Confirmation Modal */}
        {showConfirmStop && (
          <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
            <div className="bg-[#333333] border border-[#666666] p-6 max-w-md w-full mx-4">
              <h3 className="text-lg font-medium text-white mb-4">Confirm Stop Trading</h3>
              <p className="text-white mb-6">
                Are you sure you want to stop trading? This will halt all automated trading operations.
              </p>
              <div className="flex space-x-3">
                <button
                  onClick={handleStopTrading}
                  disabled={isStopping}
                  className="flex-1 px-4 py-2 bg-[#ff4444] text-white hover:bg-[#cc3333] disabled:opacity-50 transition-colors"
                >
                  {isStopping ? 'Stopping...' : 'Yes, Stop Trading'}
                </button>
                <button
                  onClick={() => setShowConfirmStop(false)}
                  className="flex-1 px-4 py-2 bg-[#666666] text-white hover:bg-[#777777] transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
        
        {/* Basic Status Text */}
        <div className="text-sm text-white">
          <p>Mode: {tradingMode === 'paper' ? 'Paper Trading' : 'Live Trading'}</p>
          <p>Status: System Ready</p>
        </div>
      </div>
    </div>
  )
}