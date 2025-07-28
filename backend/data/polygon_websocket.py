import asyncio
import json
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass
from loguru import logger
import websockets
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, EquityTrade, EquityQuote, EquityAgg, Feed, Market

from config import settings

@dataclass
class RealTimeData:
    symbol: str
    timestamp: datetime
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    data_type: str = "trade"  # "trade", "quote", "agg"
    
    # OHLCV fields for aggregate data
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    vwap: Optional[float] = None
    accumulated_volume: Optional[int] = None
    opening_price: Optional[float] = None
    average_trade_size: Optional[int] = None
    transactions: Optional[int] = None
    
class PolygonWebSocketManager:
    """Manages real-time data streaming from Polygon.io WebSocket API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.polygon_api_key
        self.client: Optional[WebSocketClient] = None
        self.is_connected = False
        self.subscribed_symbols: set = set()
        self.connection_task: Optional[asyncio.Task] = None
        
        # Data handlers
        self.trade_handlers: List[Callable] = []
        self.quote_handlers: List[Callable] = []
        self.agg_handlers: List[Callable] = []
        
        # Data cache for latest values
        self.latest_trades: Dict[str, RealTimeData] = {}
        self.latest_quotes: Dict[str, RealTimeData] = {}
        self.latest_aggs: Dict[str, RealTimeData] = {}
        
        # Connection management
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds
        
    async def connect(self) -> bool:
        """Establish WebSocket connection to Polygon.io"""
        try:
            logger.info(f"Attempting to connect to Polygon WebSocket with API key: {self.api_key[:10]}...")
            
            # Initialize WebSocketClient with subscriptions parameter
            self.client = WebSocketClient(
                api_key=self.api_key,
                feed=Feed.RealTime,
                market=Market.Stocks,
                subscriptions=[]  # Start with empty subscriptions
            )
            
            logger.info("WebSocketClient initialized, starting connection task...")
            
            # Start the connection in a background task to avoid blocking
            self.connection_task = asyncio.create_task(self.client.connect(self._handle_message))
            
            # Give it a moment to establish connection
            await asyncio.sleep(0.5)
            
            self.is_connected = True
            self.reconnect_attempts = 0
            
            logger.info("Successfully started Polygon WebSocket connection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Polygon WebSocket: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.connection_task and not self.connection_task.done():
            self.connection_task.cancel()
            try:
                await self.connection_task
            except asyncio.CancelledError:
                pass
        
        if self.client and self.is_connected:
            try:
                await self.client.close()
                self.is_connected = False
                logger.info("Disconnected from Polygon WebSocket")
            except Exception as e:
                logger.error(f"Error disconnecting from WebSocket: {e}")
        
        self.client = None
        self.subscribed_symbols.clear()
        self.connection_task = None
    
    async def subscribe_trades(self, symbols: List[str]):
        """Subscribe to real-time trade data for symbols"""
        if not self.is_connected:
            logger.warning("Not connected to WebSocket. Attempting to connect...")
            if not await self.connect():
                return False
        
        try:
            # Subscribe to trades using correct format: T.{symbol}
            subscriptions = [f"T.{symbol}" for symbol in symbols]
            self.client.subscribe(*subscriptions)
            self.subscribed_symbols.update(symbols)
            logger.info(f"Subscribed to trades for: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to trades: {e}")
            return False
    
    async def subscribe_quotes(self, symbols: List[str]):
        """Subscribe to real-time quote data for symbols"""
        if not self.is_connected:
            logger.warning("Not connected to WebSocket. Attempting to connect...")
            if not await self.connect():
                return False
        
        try:
            # Subscribe to quotes using correct format: Q.{symbol}
            subscriptions = [f"Q.{symbol}" for symbol in symbols]
            self.client.subscribe(*subscriptions)
            self.subscribed_symbols.update(symbols)
            logger.info(f"Subscribed to quotes for: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to quotes: {e}")
            return False
    
    async def subscribe_minute_aggs(self, symbols: List[str]):
        """Subscribe to real-time minute aggregates for symbols"""
        if not self.is_connected:
            logger.warning("Not connected to WebSocket. Attempting to connect...")
            if not await self.connect():
                return False
        
        try:
            # Subscribe to minute aggregates using correct format: AM.{symbol}
            subscriptions = [f"AM.{symbol}" for symbol in symbols]
            self.client.subscribe(*subscriptions)
            self.subscribed_symbols.update(symbols)
            logger.info(f"Subscribed to minute aggregates for: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to minute aggregates: {e}")
            return False
    
    def add_trade_handler(self, handler: Callable[[RealTimeData], None]):
        """Add a handler function for trade data"""
        self.trade_handlers.append(handler)
    
    def add_quote_handler(self, handler: Callable[[RealTimeData], None]):
        """Add a handler function for quote data"""
        self.quote_handlers.append(handler)
    
    def add_agg_handler(self, handler: Callable[[RealTimeData], None]):
        """Add a handler function for aggregate data"""
        self.agg_handlers.append(handler)
    
    async def _handle_message(self, messages: List[WebSocketMessage]):
        """Handle incoming WebSocket messages"""
        try:
            for message in messages:
                if isinstance(message, EquityTrade):
                    await self._handle_trade(message)
                elif isinstance(message, EquityQuote):
                    await self._handle_quote(message)
                elif isinstance(message, EquityAgg):
                    await self._handle_agg(message)
                else:
                    logger.debug(f"Received unknown message type: {type(message)}")
                    
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _handle_trade(self, trade: EquityTrade):
        """Handle trade data"""
        try:
            data = RealTimeData(
                symbol=trade.symbol,
                timestamp=datetime.fromtimestamp(trade.timestamp / 1000),
                volume=int(trade.size) if trade.size else None,
                data_type="trade",
                close=float(trade.price)  # Store trade price as close
            )
            
            # Update cache
            self.latest_trades[trade.symbol] = data
            
            # Call handlers
            for handler in self.trade_handlers:
                try:
                    await handler(data) if asyncio.iscoroutinefunction(handler) else handler(data)
                except Exception as e:
                    logger.error(f"Error in trade handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
    
    async def _handle_quote(self, quote: EquityQuote):
        """Handle quote data"""
        try:
            # Use mid-price as the close price
            mid_price = (float(quote.bid_price) + float(quote.ask_price)) / 2
            
            data = RealTimeData(
                symbol=quote.symbol,
                timestamp=datetime.fromtimestamp(quote.timestamp / 1000),
                bid=float(quote.bid_price),
                ask=float(quote.ask_price),
                data_type="quote",
                close=mid_price  # Store mid-price as close
            )
            
            # Update cache
            self.latest_quotes[quote.symbol] = data
            
            # Call handlers
            for handler in self.quote_handlers:
                try:
                    await handler(data) if asyncio.iscoroutinefunction(handler) else handler(data)
                except Exception as e:
                    logger.error(f"Error in quote handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing quote data: {e}")
    
    async def _handle_agg(self, agg: EquityAgg):
        """Handle aggregate data"""
        try:
            # Use start_timestamp (s) for the timestamp - this is the start of the minute bar
            timestamp_ms = getattr(agg, 'start_timestamp', None) or getattr(agg, 's', None)
            if timestamp_ms is None:
                logger.error(f"No timestamp found in aggregate data for {agg.symbol}")
                return
                
            # Extract all available OHLCV fields from the EquityAgg object
            data = RealTimeData(
                symbol=agg.symbol,
                timestamp=datetime.fromtimestamp(timestamp_ms / 1000),
                volume=int(agg.volume) if agg.volume else None,
                data_type="agg",
                
                # Complete OHLCV data
                open=float(agg.open) if hasattr(agg, 'open') and agg.open is not None else None,
                high=float(agg.high) if hasattr(agg, 'high') and agg.high is not None else None,
                low=float(agg.low) if hasattr(agg, 'low') and agg.low is not None else None,
                close=float(agg.close) if hasattr(agg, 'close') and agg.close is not None else None,
                vwap=float(getattr(agg, 'vwap', None)) if getattr(agg, 'vwap', None) is not None else None,
                accumulated_volume=int(getattr(agg, 'accumulated_volume', None)) if getattr(agg, 'accumulated_volume', None) is not None else None,
                opening_price=float(getattr(agg, 'opening_price', None)) if getattr(agg, 'opening_price', None) is not None else None,
                average_trade_size=int(getattr(agg, 'average_trade_size', None)) if getattr(agg, 'average_trade_size', None) is not None else None,
                transactions=int(getattr(agg, 'transactions', None)) if getattr(agg, 'transactions', None) is not None else None
            )
            
            # Add debug logging to track WebSocket data reception
            logger.info(f"WebSocket received complete aggregate data for {agg.symbol}: O={data.open}, H={data.high}, L={data.low}, C={data.close}, V={data.volume}, VWAP={data.vwap}, timestamp={data.timestamp}")
            
            # Update cache
            self.latest_aggs[agg.symbol] = data
            
            # Call handlers
            logger.debug(f"Calling {len(self.agg_handlers)} aggregate handlers for {agg.symbol}")
            for handler in self.agg_handlers:
                try:
                    await handler(data) if asyncio.iscoroutinefunction(handler) else handler(data)
                except Exception as e:
                    logger.error(f"Error in agg handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing aggregate data: {e}")
    
    async def _handle_error(self, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        self.is_connected = False
        
        # Attempt reconnection
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}")
            await asyncio.sleep(self.reconnect_delay)
            await self.connect()
    
    async def _handle_close(self, close_status):
        """Handle WebSocket close"""
        logger.info(f"WebSocket connection closed: {close_status}")
        self.is_connected = False
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol from any data source"""
        # Priority: trades > quotes > aggs
        if symbol in self.latest_trades:
            return self.latest_trades[symbol].close
        elif symbol in self.latest_quotes:
            return self.latest_quotes[symbol].close
        elif symbol in self.latest_aggs:
            return self.latest_aggs[symbol].close
        return None
    
    def get_latest_quote(self, symbol: str) -> Optional[RealTimeData]:
        """Get the latest quote data for a symbol"""
        return self.latest_quotes.get(symbol)
    
    def get_latest_trade(self, symbol: str) -> Optional[RealTimeData]:
        """Get the latest trade data for a symbol"""
        return self.latest_trades.get(symbol)
    
    async def run_forever(self):
        """Keep the WebSocket connection alive"""
        while True:
            if not self.is_connected:
                logger.info("WebSocket not connected. Attempting to connect...")
                await self.connect()
            
            await asyncio.sleep(1)  # Check connection status every second

# Global WebSocket manager instance
websocket_manager = PolygonWebSocketManager()

# Convenience functions
async def start_real_time_data(symbols: List[str]) -> bool:
    """Start real-time data streaming for given symbols"""
    try:
        # Connect to WebSocket
        if not await websocket_manager.connect():
            return False
        
        # Subscribe to all data types
        await websocket_manager.subscribe_trades(symbols)
        await websocket_manager.subscribe_quotes(symbols)
        await websocket_manager.subscribe_minute_aggs(symbols)
        
        logger.info(f"Started real-time data streaming for {len(symbols)} symbols")
        return True
        
    except Exception as e:
        logger.error(f"Failed to start real-time data streaming: {e}")
        return False

async def stop_real_time_data():
    """Stop real-time data streaming"""
    await websocket_manager.disconnect()
    logger.info("Stopped real-time data streaming")

def get_current_price(symbol: str) -> Optional[float]:
    """Get current price for a symbol from WebSocket data"""
    return websocket_manager.get_latest_price(symbol)

def get_current_quote(symbol: str) -> Optional[RealTimeData]:
    """Get current quote for a symbol from WebSocket data"""
    return websocket_manager.get_latest_quote(symbol)

def get_current_trade(symbol: str) -> Optional[RealTimeData]:
    """Get current trade for a symbol from WebSocket data"""
    return websocket_manager.get_latest_trade(symbol)