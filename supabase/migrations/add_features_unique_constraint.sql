-- Add unique constraint on symbol and timestamp for features table
-- This constraint is required by the SQLAlchemy model definition
ALTER TABLE features ADD CONSTRAINT uq_features_symbol_timestamp UNIQUE (symbol, timestamp);

-- Grant permissions to anon and authenticated roles
GRANT SELECT, INSERT, UPDATE ON features TO anon;
GRANT ALL PRIVILEGES ON features TO authenticated;