-- Additive migration for existing installs: ensures the tool_call_id column
-- is present on the messages table. Safe to run multiple times.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS tool_call_id TEXT;
