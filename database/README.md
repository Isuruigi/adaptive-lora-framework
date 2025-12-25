# Database Setup

This directory contains the PostgreSQL database schema for the Adaptive LoRA Framework.

## Quick Setup with Supabase (FREE)

1. **Create Supabase Project**
   - Go to https://supabase.com
   - Create new project (FREE tier: 500MB, 50K rows)
   - Get your connection string from Settings → Database

2. **Run Schema**
   ```bash
   # Option 1: Supabase SQL Editor
   # Copy contents of schema.sql and paste into SQL Editor
   
   # Option 2: psql
   psql "postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres" -f schema.sql
   ```

3. **Configure Environment**
   ```bash
   # Add to your .env file
   DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres
   ```

## Tables

| Table | Purpose |
|-------|---------|
| `users` | User accounts and API keys |
| `requests` | Request logging with adapter, latency, tokens |
| `evaluations` | Quality scores per request |
| `feedback` | User ratings and comments |
| `adapter_metrics` | Daily adapter performance aggregates |
| `router_training_data` | Data for router improvement |
| `cache_entries` | Optional SQL-based cache |

## Views

| View | Purpose |
|------|---------|
| `request_summary` | Joined request + evaluation + user data |
| `adapter_performance` | Per-adapter aggregate statistics |
| `daily_metrics` | Materialized view for analytics |

## Maintenance

```sql
-- Refresh daily metrics (run via cron/scheduled job)
SELECT refresh_daily_metrics();

-- Clean old cache entries
DELETE FROM cache_entries WHERE expires_at < NOW();

-- Clean old rate limit records
DELETE FROM rate_limits WHERE window_start < NOW() - INTERVAL '1 hour';
```

## Supabase FREE Tier Limits

- 500MB database
- 50,000 rows (soft limit)
- 500MB file storage
- 2GB bandwidth

For 50K requests/month, this is more than sufficient!
