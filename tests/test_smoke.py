"""Post-refactoring guard tests.

Verify no legacy root-level scripts remain and no src/ module
imports from them.
"""

from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestNoLegacyFiles:
    def test_no_root_python_files(self):
        """No legacy .py files at project root."""
        root_py = sorted(f.name for f in _PROJECT_ROOT.glob("*.py"))
        assert root_py == [], f"Unexpected root .py files: {root_py}"

    def test_canonical_csv_in_data_dir(self):
        """FRED CSV must exist in data/ (the canonical location)."""
        data_csv = _PROJECT_ROOT / "data" / "fred_credit_spreads.csv"
        assert data_csv.exists(), "Canonical CSV missing from data/"


class TestNoForbiddenImports:
    LEGACY_MODULES = [
        "from black_litterman import",
        "from hmm_regime import",
        "from pipeline import",
        "from run_pipeline import",
        "from optimizer import",
        "from fred_data import",
        "from data_loader import",
        "from commentary import",
        "from attribution import",
        "from universe import",
        "from make_charts_bl import",
        "from prophet_views import",
        "from risk_parity import",
        "from optimizer_bl import",
    ]

    def test_no_legacy_imports_in_src(self):
        """No src/ file imports from legacy root-level modules."""
        src = _PROJECT_ROOT / "src"
        violations = []
        for py in src.rglob("*.py"):
            content = py.read_text()
            for bad in self.LEGACY_MODULES:
                if bad in content:
                    violations.append(f"{py.relative_to(_PROJECT_ROOT)}: {bad}")
        assert violations == [], f"Forbidden imports found:\n" + "\n".join(violations)


class TestPackageImports:
    def test_import_data(self):
        from credit_portfolio.data import load, build_universe, JUNE_SHOCK
        assert callable(load)
        assert callable(build_universe)

    def test_import_models(self):
        from credit_portfolio.models import fit_hmm, BLResult
        assert callable(fit_hmm)

    def test_import_optimizers(self):
        from credit_portfolio.optimizers import optimise, optimise_bl, optimise_risk_parity
        assert callable(optimise)
        assert callable(optimise_bl)
        assert callable(optimise_risk_parity)

    def test_import_analytics(self):
        from credit_portfolio.analytics import attribute, generate_commentary_mock
        assert callable(attribute)
        assert callable(generate_commentary_mock)

    def test_import_charts(self):
        from credit_portfolio.charts import chart_value_signal, chart_hmm_regimes
        assert callable(chart_value_signal)
        assert callable(chart_hmm_regimes)
