-- Create model_predictions table for storing model prediction results
CREATE TABLE IF NOT EXISTS public.model_predictions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    prediction NUMERIC NOT NULL,
    confidence NUMERIC,
    actual_value NUMERIC,
    features_used JSONB,
    model_version VARCHAR,
    training_date TIMESTAMPTZ,
    prediction_horizon INTEGER, -- minutes ahead
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_model_predictions_symbol_timestamp ON public.model_predictions(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_model_predictions_model_name ON public.model_predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_model_predictions_created_at ON public.model_predictions(created_at);

-- Enable RLS
ALTER TABLE public.model_predictions ENABLE ROW LEVEL SECURITY;

-- Grant permissions to anon and authenticated roles
GRANT SELECT ON public.model_predictions TO anon;
GRANT ALL PRIVILEGES ON public.model_predictions TO authenticated;
GRANT USAGE ON SEQUENCE public.model_predictions_id_seq TO authenticated;