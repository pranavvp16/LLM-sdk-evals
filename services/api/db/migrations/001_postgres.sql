CREATE TABLE sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_agent  TEXT,
    ip_hash     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE conversations (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id   UUID REFERENCES sessions(id),
    title        TEXT DEFAULT '',
    provider     TEXT NOT NULL,
    model        TEXT NOT NULL,
    status       TEXT DEFAULT 'active',
    created_at   TIMESTAMPTZ DEFAULT now(),
    updated_at   TIMESTAMPTZ DEFAULT now(),
    cancelled_at TIMESTAMPTZ
);

CREATE TABLE messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,          -- user | assistant | tool_result
    content         TEXT NOT NULL,
    tool_call_id    TEXT,                   -- non-null for role='tool_result'
    token_count     INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT now()
);
