-- Create trading system database tables for Supabase
-- This migration creates all tables needed for the algorithmic trading system

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create market_data table
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(10, 4) NOT NULL,
    high DECIMAL(10, 4) NOT NULL,
    low DECIMAL(10, 4) NOT NULL,
    close DECIMAL(10, 4) NOT NULL,
    volume BIGINT NOT NULL,
    vwap DECIMAL(10, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create features table
CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    features JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    prediction DECIMAL(10, 6) NOT NULL,
    confidence DECIMAL(5, 4),
    features_used JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create trades table
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(15, 6) NOT NULL,
    price DECIMAL(10, 4) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    order_id VARCHAR(100),
    strategy VARCHAR(50),
    pnl DECIMAL(15, 4),
    commission DECIMAL(10, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create positions table
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    quantity DECIMAL(15, 6) NOT NULL,
    avg_price DECIMAL(10, 4) NOT NULL,
    market_value DECIMAL(15, 4),
    unrealized_pnl DECIMAL(15, 4),
    side VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short')),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create model_performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    symbol VARCHAR(10),
    date DATE NOT NULL,
    accuracy DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    sharpe_ratio DECIMAL(8, 4),
    total_return DECIMAL(8, 4),
    max_drawdown DECIMAL(8, 4),
    win_rate DECIMAL(5, 4),
    profit_factor DECIMAL(8, 4),
    metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);

CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp ON features(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_features_timestamp ON features(timestamp);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol_timestamp ON predictions(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_model_name ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);

CREATE INDEX IF NOT EXISTS idx_model_performance_model_name ON model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_model_performance_date ON model_performance(date);
CREATE INDEX IF NOT EXISTS idx_model_performance_symbol ON model_performance(symbol);

-- Grant permissions to authenticated users
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO authenticated;

-- Grant permissions to anon users (for read-only access)
GRANT SELECT ON ALL TABLES IN SCHEMA public TO anon;

-- Enable Row Level Security (RLS) for all tables
ALTER TABLE market_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE features ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_performance ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (allow all operations for authenticated users)
CREATE POLICY "Allow all operations for authenticated users" ON market_data
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON features
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON predictions
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON trades
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON positions
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON model_performance
    FOR ALL USING (auth.role() = 'authenticated');

-- Create read-only policies for anon users
CREATE POLICY "Allow read access for anon users" ON market_data
    FOR SELECT USING (auth.role() = 'anon');

CREATE POLICY "Allow read access for anon users" ON features
    FOR SELECT USING (auth.role() = 'anon');

CREATE POLICY "Allow read access for anon users" ON predictions
    FOR SELECT USING (auth.role() = 'anon');

CREATE POLICY "Allow read access for anon users" ON trades
    FOR SELECT USING (auth.role() = 'anon');

CREATE POLICY "Allow read access for anon users" ON positions
    FOR SELECT USING (auth.role() = 'anon');

CREATE POLICY "Allow read access for anon users" ON model_performance
    FOR SELECT USING (auth.role() = 'anon');