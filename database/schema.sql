-- Adaptive LoRA Framework Database Schema
-- PostgreSQL schema for production deployment
-- Compatible with Supabase (FREE tier: 500MB)

-- =============================================
-- USER MANAGEMENT
-- =============================================

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    api_key VARCHAR(64) UNIQUE NOT NULL,
    tier VARCHAR(20) DEFAULT 'free' CHECK (tier IN ('free', 'pro', 'enterprise')),
    rate_limit_per_minute INTEGER DEFAULT 60,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for API key lookups
CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- =============================================
-- REQUEST LOGGING
-- =============================================

CREATE TABLE IF NOT EXISTS requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    request_id VARCHAR(64) UNIQUE NOT NULL,
    prompt TEXT NOT NULL,
    adapter_used VARCHAR(50) NOT NULL,
    complexity VARCHAR(20),
    router_confidence DECIMAL(4, 3),
    latency_ms INTEGER NOT NULL,
    tokens_generated INTEGER,
    cached BOOLEAN DEFAULT false,
    status VARCHAR(20) DEFAULT 'success' CHECK (status IN ('success', 'error', 'timeout')),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for analytics queries
CREATE INDEX IF NOT EXISTS idx_requests_user_id ON requests(user_id);
CREATE INDEX IF NOT EXISTS idx_requests_adapter ON requests(adapter_used);
CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_requests_cached ON requests(cached);

-- =============================================
-- EVALUATION RESULTS
-- =============================================

CREATE TABLE IF NOT EXISTS evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID REFERENCES requests(id) ON DELETE CASCADE,
    overall_score DECIMAL(4, 3) NOT NULL,
    coherence_score DECIMAL(4, 3),
    relevance_score DECIMAL(4, 3),
    factuality_score DECIMAL(4, 3),
    safety_score DECIMAL(4, 3),
    uncertainty_score DECIMAL(4, 3),
    is_failure BOOLEAN DEFAULT false,
    failure_type VARCHAR(50),
    failure_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evaluations_request_id ON evaluations(request_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_is_failure ON evaluations(is_failure);
CREATE INDEX IF NOT EXISTS idx_evaluations_overall_score ON evaluations(overall_score);

-- =============================================
-- USER FEEDBACK
-- =============================================

CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID REFERENCES requests(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    thumbs_up BOOLEAN,
    comment TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_request_id ON feedback(request_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);

-- =============================================
-- ADAPTER PERFORMANCE TRACKING
-- =============================================

CREATE TABLE IF NOT EXISTS adapter_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    adapter_name VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    total_requests INTEGER DEFAULT 0,
    avg_latency_ms DECIMAL(10, 2),
    avg_quality_score DECIMAL(4, 3),
    failure_count INTEGER DEFAULT 0,
    cache_hits INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(adapter_name, date)
);

CREATE INDEX IF NOT EXISTS idx_adapter_metrics_date ON adapter_metrics(date);
CREATE INDEX IF NOT EXISTS idx_adapter_metrics_adapter ON adapter_metrics(adapter_name);

-- =============================================
-- MATERIALIZED VIEW FOR DAILY ANALYTICS
-- =============================================

CREATE MATERIALIZED VIEW IF NOT EXISTS daily_metrics AS
SELECT 
    DATE(created_at) as date,
    user_id,
    COUNT(*) as total_requests,
    COUNT(*) FILTER (WHERE cached = true) as cached_requests,
    AVG(latency_ms)::INTEGER as avg_latency_ms,
    SUM(tokens_generated) as total_tokens,
    COUNT(*) FILTER (WHERE status = 'error') as error_count
FROM requests
GROUP BY DATE(created_at), user_id;

-- Refresh function (call periodically)
-- SELECT refresh_daily_metrics();

CREATE OR REPLACE FUNCTION refresh_daily_metrics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW daily_metrics;
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- ROUTER TRAINING DATA
-- =============================================

CREATE TABLE IF NOT EXISTS router_training_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query TEXT NOT NULL,
    adapter_label VARCHAR(50) NOT NULL,
    complexity_label VARCHAR(20),
    confidence DECIMAL(4, 3),
    is_validated BOOLEAN DEFAULT false,
    source VARCHAR(50) DEFAULT 'production', -- 'production', 'synthetic', 'manual'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_router_training_source ON router_training_data(source);
CREATE INDEX IF NOT EXISTS idx_router_training_validated ON router_training_data(is_validated);

-- =============================================
-- API RATE LIMITING (if not using Redis)
-- =============================================

CREATE TABLE IF NOT EXISTS rate_limits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    request_count INTEGER DEFAULT 1,
    UNIQUE(user_id, window_start)
);

CREATE INDEX IF NOT EXISTS idx_rate_limits_user_window ON rate_limits(user_id, window_start);

-- =============================================
-- CACHE METADATA (optional, if not using Redis)
-- =============================================

CREATE TABLE IF NOT EXISTS cache_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_key VARCHAR(128) UNIQUE NOT NULL,
    prompt_hash VARCHAR(64) NOT NULL,
    adapter_used VARCHAR(50) NOT NULL,
    response TEXT NOT NULL,
    hit_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cache_entries_key ON cache_entries(cache_key);
CREATE INDEX IF NOT EXISTS idx_cache_entries_expires ON cache_entries(expires_at);

-- =============================================
-- HELPER FUNCTIONS
-- =============================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Generate API key
CREATE OR REPLACE FUNCTION generate_api_key()
RETURNS VARCHAR(64) AS $$
BEGIN
    RETURN 'sk_' || encode(gen_random_bytes(30), 'hex');
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- SAMPLE DATA (for testing)
-- =============================================

-- Create a test user
-- INSERT INTO users (email, api_key, tier)
-- VALUES ('test@example.com', generate_api_key(), 'free');

-- =============================================
-- VIEWS FOR COMMON QUERIES
-- =============================================

CREATE OR REPLACE VIEW request_summary AS
SELECT 
    r.id,
    r.request_id,
    r.prompt,
    r.adapter_used,
    r.latency_ms,
    r.cached,
    e.overall_score,
    e.is_failure,
    u.email as user_email,
    r.created_at
FROM requests r
LEFT JOIN evaluations e ON r.id = e.request_id
LEFT JOIN users u ON r.user_id = u.id;

CREATE OR REPLACE VIEW adapter_performance AS
SELECT 
    adapter_used,
    COUNT(*) as total_requests,
    AVG(latency_ms)::INTEGER as avg_latency,
    AVG(e.overall_score)::DECIMAL(4,3) as avg_quality,
    COUNT(*) FILTER (WHERE e.is_failure = true) as failures,
    COUNT(*) FILTER (WHERE cached = true) as cache_hits
FROM requests r
LEFT JOIN evaluations e ON r.id = e.request_id
GROUP BY adapter_used;

-- =============================================
-- GRANTS (adjust as needed for your setup)
-- =============================================

-- For Supabase, these are typically handled automatically
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO authenticated;
