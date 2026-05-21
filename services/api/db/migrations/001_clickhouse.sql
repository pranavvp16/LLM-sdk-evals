CREATE TABLE IF NOT EXISTS inference_logs (
    trace_id            String,
    session_id          String,
    conversation_id     String,
    provider            LowCardinality(String),
    model               LowCardinality(String),
    api_protocol        LowCardinality(String),
    started_at          DateTime64(3),
    ended_at            DateTime64(3),
    latency_ms          Float64,
    input_tokens        UInt32,
    output_tokens       UInt32,
    cache_read_tokens   UInt32,
    cache_write_tokens  UInt32,
    total_tokens        UInt32,
    cost_usd            Float64,
    input_preview       String,
    output_preview      String,
    status              LowCardinality(String),
    error_message       Nullable(String),
    stop_reason         LowCardinality(String),
    stream              Bool,
    temperature         Nullable(Float32),
    max_tokens          Nullable(UInt32),
    date                Date DEFAULT toDate(started_at)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (provider, started_at)
TTL date + INTERVAL 90 DAY;
