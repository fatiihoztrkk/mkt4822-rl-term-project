"""SQLite schema migrations."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  repo_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  user_request TEXT NOT NULL,
  task_type TEXT NOT NULL,
  target_files_json TEXT NOT NULL,
  strategy_name TEXT NOT NULL,
  model_name TEXT NOT NULL,
  status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER NOT NULL,
  file_path TEXT NOT NULL,
  before_hash TEXT NOT NULL,
  after_hash TEXT NOT NULL,
  diff_text TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES tasks(id)
);
CREATE TABLE IF NOT EXISTS validation_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER NOT NULL,
  attempt_index INTEGER NOT NULL,
  syntax_ok INTEGER NOT NULL,
  import_ok INTEGER NOT NULL,
  ruff_ok INTEGER NOT NULL,
  pyright_ok INTEGER NOT NULL,
  pytest_ok INTEGER NOT NULL,
  semantic_score REAL NOT NULL,
  total_score REAL NOT NULL,
  failure_type TEXT NOT NULL,
  failure_summary TEXT NOT NULL,
  duration_sec REAL NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES tasks(id)
);
CREATE TABLE IF NOT EXISTS reflections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER NOT NULL,
  failure_type TEXT NOT NULL,
  root_cause TEXT NOT NULL,
  lesson TEXT NOT NULL,
  future_rule TEXT NOT NULL,
  applicable_task_types_json TEXT NOT NULL,
  applicable_files_json TEXT NOT NULL,
  reward_after_reuse REAL DEFAULT 0,
  created_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES tasks(id)
);
CREATE TABLE IF NOT EXISTS repo_summaries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  repo_id TEXT NOT NULL,
  file_path TEXT NOT NULL,
  symbol_summary TEXT NOT NULL,
  last_hash TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS strategy_stats (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  strategy_name TEXT NOT NULL,
  task_type TEXT NOT NULL,
  model_name TEXT NOT NULL,
  attempts INTEGER NOT NULL,
  successes INTEGER NOT NULL,
  mean_reward REAL NOT NULL,
  last_reward REAL NOT NULL,
  last_used_at TEXT NOT NULL,
  UNIQUE(strategy_name, task_type, model_name)
);
CREATE TABLE IF NOT EXISTS bandit_arms (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  context_key TEXT NOT NULL,
  arm_id TEXT NOT NULL,
  alpha REAL NOT NULL DEFAULT 1.0,
  beta REAL NOT NULL DEFAULT 1.0,
  pulls INTEGER NOT NULL DEFAULT 0,
  cumulative_reward REAL NOT NULL DEFAULT 0.0,
  last_reward REAL NOT NULL DEFAULT 0.0,
  last_success INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL,
  UNIQUE(context_key, arm_id)
);
CREATE INDEX IF NOT EXISTS idx_bandit_arms_ctx ON bandit_arms(context_key);
"""
