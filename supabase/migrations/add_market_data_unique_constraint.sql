-- Add unique constraint on symbol and timestamp for market_data table
ALTER TABLE market_data ADD CONSTRAINT market_data_symbol_timestamp_unique UNIQUE (symbol, timestamp);

-- Grant permissions to anon and authenticated roles
GRANT SELECT, INSERT, UPDATE ON market_data TO anon;
GRANT ALL PRIVILEGES ON market_data TO authenticated;