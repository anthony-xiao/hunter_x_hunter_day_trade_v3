import { useState, useEffect } from 'react'

interface Position {
  symbol: string
  quantity: number
  avg_price: number
  market_value: number
  unrealized_pnl: number
  pnl_percent: number
}

export const PositionsTable = () => {
  const [positions, setPositions] = useState<Position[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)

  useEffect(() => {
    fetchPositions()
    
    // Set up periodic updates
    const interval = setInterval(fetchPositions, 15000) // Every 15 seconds
    
    return () => clearInterval(interval)
  }, [])

  const fetchPositions = async () => {
    try {
      const response = await fetch('http://localhost:8000/positions')
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      setPositions(data)
      setLastUpdate(new Date())
      setError(null)
    } catch (err) {
      console.error('Failed to fetch positions:', err)
      setError('Failed to fetch positions')
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

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
  }

  const getTotalValue = () => {
    return positions.reduce((sum, pos) => sum + pos.market_value, 0)
  }

  const getTotalPnL = () => {
    return positions.reduce((sum, pos) => sum + pos.unrealized_pnl, 0)
  }

  const getTotalPnLPercent = () => {
    const totalValue = getTotalValue()
    const totalPnL = getTotalPnL()
    return totalValue > 0 ? (totalPnL / (totalValue - totalPnL)) * 100 : 0
  }

  if (loading) {
    return (
      <div className="bg-[#333333] border border-[#666666] p-6">
        <h2 className="text-lg font-medium text-white mb-6">Current Positions</h2>
        <div className="text-white font-mono">Loading positions...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-[#333333] border border-[#666666] p-6">
        <h2 className="text-lg font-medium text-white mb-6">Current Positions</h2>
        <div className="text-[#ff4444] font-mono">Error: {error}</div>
      </div>
    )
  }

  return (
    <div className="bg-[#333333] border border-[#666666] p-6">
      <h2 className="text-lg font-medium text-white mb-6">Current Positions</h2>
      
      {/* Summary */}
      <div className="mb-6 text-white font-mono">
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div>
            <div className="text-sm">Total Positions:</div>
            <div className="text-xl">{positions.length}</div>
          </div>
          <div>
            <div className="text-sm">Total Value:</div>
            <div className="text-xl">{formatCurrency(getTotalValue())}</div>
          </div>
          <div>
            <div className="text-sm">Total P&L:</div>
            <div className={`text-xl ${
              getTotalPnL() >= 0 ? 'text-[#00ff88]' : 'text-[#ff4444]'
            }`}>
              {formatCurrency(getTotalPnL())} ({formatPercent(getTotalPnLPercent())})
            </div>
          </div>
        </div>
        
        {lastUpdate && (
          <div className="text-sm text-white border-t border-[#666666] pt-2">
            Last Updated: {lastUpdate.toLocaleTimeString()}
          </div>
        )}
      </div>

      {/* Positions List */}
      {positions.length === 0 ? (
        <div className="text-white font-mono">
          No active positions. Start trading to see positions here.
        </div>
      ) : (
        <div className="space-y-2">
          {positions.map((position) => (
            <div key={position.symbol} className="bg-[#444444] border border-[#666666] p-4">
              <div className="grid grid-cols-6 gap-4 text-white font-mono">
                <div>
                  <div className="text-sm text-white">Symbol:</div>
                  <div className="text-lg font-bold">{position.symbol}</div>
                </div>
                <div>
                  <div className="text-sm text-white">Quantity:</div>
                  <div>{position.quantity.toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-sm text-white">Avg Price:</div>
                  <div>{formatCurrency(position.avg_price)}</div>
                </div>
                <div>
                  <div className="text-sm text-white">Market Value:</div>
                  <div>{formatCurrency(position.market_value)}</div>
                </div>
                <div>
                  <div className="text-sm text-white">Unrealized P&L:</div>
                  <div className={position.unrealized_pnl >= 0 ? 'text-[#00ff88]' : 'text-[#ff4444]'}>
                    {formatCurrency(position.unrealized_pnl)}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-white">P&L %:</div>
                  <div className={position.pnl_percent >= 0 ? 'text-[#00ff88]' : 'text-[#ff4444]'}>
                    {formatPercent(position.pnl_percent)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}