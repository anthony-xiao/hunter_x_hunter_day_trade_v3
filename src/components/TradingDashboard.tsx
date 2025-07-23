import { useState, useEffect } from 'react'

interface TradingStatus {
  is_trading: boolean
  portfolio_value: number
  cash_balance: number
  positions_count: number
  daily_pnl: number
  total_trades: number
}

export const TradingDashboard = () => {
  const [tradingStatus, setTradingStatus] = useState<TradingStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchTradingStatus()
    
    // Set up periodic updates
    const interval = setInterval(fetchTradingStatus, 10000) // Every 10 seconds
    
    return () => clearInterval(interval)
  }, [])

  const fetchTradingStatus = async () => {
    try {
      // Fetch both trading status and portfolio data
      const [statusResponse, portfolioResponse] = await Promise.all([
        fetch('http://localhost:8000/trading/status'),
        fetch('http://localhost:8000/portfolio')
      ])
      
      if (!statusResponse.ok || !portfolioResponse.ok) {
        throw new Error(`HTTP error! status: ${statusResponse.status} or ${portfolioResponse.status}`)
      }
      
      const statusData = await statusResponse.json()
      const portfolioData = await portfolioResponse.json()
      
      // Combine the data to match the expected interface
      const combinedData = {
        is_trading: statusData.trading_active || statusData.is_trading || false,
        portfolio_value: portfolioData.equity || portfolioData.portfolio_value || 0,
        cash_balance: portfolioData.cash || portfolioData.buying_power || 0,
        positions_count: statusData.positions || portfolioData.positions_count || 0,
        daily_pnl: statusData.daily_pnl || portfolioData.daily_pnl || 0,
        total_trades: statusData.daily_trades || portfolioData.total_trades || 0
      }
      
      setTradingStatus(combinedData)
      setError(null)
    } catch (err) {
      console.error('Failed to fetch trading status:', err)
      setError('Failed to fetch trading data')
    } finally {
      setLoading(false)
    }
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value)
  }

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat('en-US').format(value)
  }

  if (loading) {
    return (
      <div className="bg-[#333333] border border-[#666666] p-6">
        <div className="text-white">Loading portfolio data...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-[#333333] border border-[#666666] p-6">
        <div className="text-[#ff4444]">Error: {error}</div>
      </div>
    )
  }

  if (!tradingStatus) {
    return (
      <div className="bg-[#333333] border border-[#666666] p-6">
        <div className="text-white">No trading data available</div>
      </div>
    )
  }

  const pnlChange = tradingStatus.portfolio_value > 0 
    ? (tradingStatus.daily_pnl / tradingStatus.portfolio_value) * 100 
    : 0

  return (
    <div className="bg-[#333333] border border-[#666666] p-6">
      <h2 className="text-lg font-medium text-white mb-6">Portfolio Summary</h2>
      
      {/* Plain Text Display of Key Metrics */}
      <div className="space-y-4 text-white font-mono">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-sm text-white">Portfolio Value:</div>
            <div className="text-xl">{formatCurrency(tradingStatus.portfolio_value)}</div>
          </div>
          <div>
            <div className="text-sm text-white">Cash Balance:</div>
            <div className="text-xl">{formatCurrency(tradingStatus.cash_balance)}</div>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-sm text-white">Daily P&L:</div>
            <div className={`text-xl ${
              tradingStatus.daily_pnl >= 0 ? 'text-[#00ff88]' : 'text-[#ff4444]'
            }`}>
              {tradingStatus.daily_pnl >= 0 ? '+' : ''}{formatCurrency(tradingStatus.daily_pnl)}
            </div>
          </div>
          <div>
            <div className="text-sm text-white">Active Positions:</div>
            <div className="text-xl">{tradingStatus.positions_count}</div>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-sm text-white">Daily Return:</div>
            <div className={`text-xl ${
              pnlChange >= 0 ? 'text-[#00ff88]' : 'text-[#ff4444]'
            }`}>
              {pnlChange >= 0 ? '+' : ''}{pnlChange.toFixed(2)}%
            </div>
          </div>
          <div>
            <div className="text-sm text-white">Total Trades:</div>
            <div className="text-xl">{tradingStatus.total_trades}</div>
          </div>
        </div>
        
        <div className="pt-4 border-t border-[#666666]">
          <div className="text-sm text-white">
            Status: {tradingStatus.is_trading ? 'TRADING ACTIVE' : 'TRADING INACTIVE'}
          </div>
          <div className="text-sm text-white">
            Last Updated: {new Date().toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  )
}