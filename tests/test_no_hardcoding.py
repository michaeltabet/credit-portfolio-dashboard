"""Guard tests: no hardcoded magic numbers or print() in library code."""

import ast
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "credit_portfolio"


class TestNoPrintInSrc:
    """No bare print() calls in library modules (use logger instead)."""

    EXCLUDED = {"__main__.py"}

    def test_no_print_calls(self):
        violations = []
        for py in _SRC.rglob("*.py"):
            if py.name in self.EXCLUDED:
                continue
            try:
                tree = ast.parse(py.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    violations.append(
                        f"{py.relative_to(_SRC.parent.parent)}:{node.lineno}"
                    )
        assert violations == [], (
            "Bare print() found in src/:\n" + "\n".join(violations)
        )


class TestConfigSections:
    """config.yaml has all required sections."""

    def test_required_sections(self):
        import yaml

        config_path = _SRC.parents[1] / "config.yaml"
        assert config_path.exists(), "config.yaml not found"
        cfg = yaml.safe_load(config_path.read_text())
        for section in ("logging", "llm", "backtest"):
            assert section in cfg, f"Missing config section: {section}"

    def test_logging_has_level(self):
        import yaml

        config_path = _SRC.parents[1] / "config.yaml"
        cfg = yaml.safe_load(config_path.read_text())
        assert "level" in cfg["logging"]

    def test_llm_has_model(self):
        import yaml

        config_path = _SRC.parents[1] / "config.yaml"
        cfg = yaml.safe_load(config_path.read_text())
        assert "model" in cfg["llm"]


class TestConstantsExist:
    """Key constants are defined in constants.py."""

    def test_key_constants_importable(self):
        from credit_portfolio.data.constants import (
            BL_RIDGE_PENALTY,
            HMM_CONVERGENCE_TOL,
            OPT_CONSTRAINT_TOL,
            RP_VOL_FLOOR,
            ML_NUMERIC_TOL,
            ML_RANDOM_SEED,
            ML_MIN_TRAIN_SAMPLES,
            ML_MIN_TEST_SAMPLES,
            ML_MIN_QUINTILE_SIZE,
            ML_TARGET_COL,
            LLM_MODEL_ID,
            LLM_MAX_TOKENS,
            OAS_PCT_TO_BP,
            MONTHS_PER_YEAR,
            QUARTERLY_TO_ANNUAL,
            ATTRIBUTION_TOP_N,
            CHART_ROLLING_WINDOW,
            HMM_BAR_WIDTH,
            HISTORICAL_EVENTS,
            CHART_FILENAMES,
            ML_MODEL_CHOICES,
        )

        assert isinstance(BL_RIDGE_PENALTY, float)
        assert isinstance(ML_RANDOM_SEED, int)
        assert isinstance(LLM_MODEL_ID, str)
        assert isinstance(LLM_MAX_TOKENS, int)
        assert isinstance(CHART_FILENAMES, dict)
        assert isinstance(ML_MODEL_CHOICES, list)
        assert isinstance(HISTORICAL_EVENTS, list)
