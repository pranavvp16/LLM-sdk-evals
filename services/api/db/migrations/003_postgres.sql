-- Persist full tool round-trips so multi-turn context can be replayed.
--   * `tool_calls`  — JSON array of {id, name, arguments} on assistant rows.
--   * `is_error`    — flags failed tool executions on tool_result rows.

ALTER TABLE messages ADD COLUMN IF NOT EXISTS tool_calls JSONB;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS is_error BOOLEAN DEFAULT FALSE;
