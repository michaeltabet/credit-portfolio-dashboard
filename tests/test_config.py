"""Tests for config loading, path resolution, and env var overrides."""

import os
import pytest
from pathlib import Path

from credit_portfolio.config import load_config, resolve_csv_path, resolve_output_dir


class TestLoadConfig:
    def test_returns_dict(self):
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_has_paths_section(self):
        cfg = load_config()
        assert "paths" in cfg

    def test_nonexistent_path_returns_dict(self, tmp_path):
        cfg = load_config(str(tmp_path / "nonexistent.yaml"))
        assert isinstance(cfg, dict)

    def test_env_var_override_csv(self, monkeypatch):
        monkeypatch.setenv("CREDIT_PORTFOLIO_CSV", "/tmp/custom.csv")
        cfg = load_config()
        assert cfg["paths"]["csv"] == "/tmp/custom.csv"

    def test_env_var_override_output(self, monkeypatch):
        monkeypatch.setenv("CREDIT_PORTFOLIO_OUTPUT", "/tmp/custom_out")
        cfg = load_config()
        assert cfg["paths"]["output"] == "/tmp/custom_out"


class TestResolveCsvPath:
    def test_default_path_ends_with_csv(self):
        p = resolve_csv_path({})
        assert str(p).endswith("fred_credit_spreads.csv")

    def test_absolute_path_passthrough(self):
        p = resolve_csv_path({"paths": {"csv": "/absolute/path.csv"}})
        assert p == Path("/absolute/path.csv")

    def test_relative_path_resolved_to_project_root(self):
        p = resolve_csv_path({"paths": {"csv": "data/test.csv"}})
        assert p.is_absolute()
        assert str(p).endswith("data/test.csv")


class TestResolveOutputDir:
    def test_creates_directory(self, tmp_path):
        target = tmp_path / "new_output"
        cfg = {"paths": {"output": str(target)}}
        result = resolve_output_dir(cfg)
        assert result.exists()
        assert result.is_dir()

    def test_default_returns_output(self):
        p = resolve_output_dir({})
        assert str(p).endswith("output")
