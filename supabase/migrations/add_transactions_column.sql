-- Add transactions column to market_data table
ALTER TABLE market_data ADD COLUMN transactions INTEGER;

-- Add accumulated_volume column as well since it's also referenced in the code
ALTER TABLE market_data ADD COLUMN accumulated_volume BIGINT;

-- Grant permissions to anon and authenticated roles
GRANT SELECT, INSERT, UPDATE, DELETE ON market_data TO anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON market_data TO authenticated;