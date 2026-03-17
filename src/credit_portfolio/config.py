"""Loads config.yaml with environment variable overrides."""

import os
from pathlib import Path

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: str | None = None) -> dict:
    """Load config from YAML, with env var overrides.

    Environment variable overrides:
        CREDIT_PORTFOLIO_CSV       -> paths.csv
        CREDIT_PORTFOLIO_OUTPUT    -> paths.output
        CREDIT_PORTFOLIO_LOG_LEVEL -> logging.level
        CREDIT_PORTFOLIO_LLM_MODEL -> llm.model
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if path.exists():
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    # Env var overrides
    env_csv = os.environ.get("CREDIT_PORTFOLIO_CSV")
    env_out = os.environ.get("CREDIT_PORTFOLIO_OUTPUT")
    env_log_level = os.environ.get("CREDIT_PORTFOLIO_LOG_LEVEL")
    env_llm_model = os.environ.get("CREDIT_PORTFOLIO_LLM_MODEL")
    if env_csv:
        cfg.setdefault("paths", {})["csv"] = env_csv
    if env_out:
        cfg.setdefault("paths", {})["output"] = env_out
    if env_log_level:
        cfg.setdefault("logging", {})["level"] = env_log_level
    if env_llm_model:
        cfg.setdefault("llm", {})["model"] = env_llm_model

    return cfg


def resolve_csv_path(cfg: dict) -> Path:
    """Resolve the CSV path relative to project root."""
    raw = cfg.get("paths", {}).get("csv", "data/fred_credit_spreads.csv")
    p = Path(raw)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


def resolve_output_dir(cfg: dict) -> Path:
    """Resolve the output directory relative to project root."""
    raw = cfg.get("paths", {}).get("output", "output")
    p = Path(raw)
    if p.is_absolute():
        out = p
    else:
        out = _PROJECT_ROOT / p
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_backtest_config(cfg: dict | None = None):
    """Build a BacktestConfig from the YAML config."""
    from credit_portfolio.backtests.bucket_backtest import BacktestConfig
    if cfg is None:
        cfg = load_config()
    bt = cfg.get("backtest", {})
    fac = cfg.get("factors", {})
    return BacktestConfig(
        tilt_strength=bt.get("tilt_strength", 0.10),
        tc_bps=bt.get("transaction_cost_bps", 5.0),
        momentum_window=fac.get("momentum_window", 6),
        min_weight=bt.get("min_weight", 0.01),
        factor_weights=fac.get("weights", {"z_dts": 0.50, "z_value": 0.25, "z_momentum": 0.25}),
    )
