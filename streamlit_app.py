"""Streamlit Cloud entry point."""
import importlib
import sys
from pathlib import Path

# Ensure the src directory is on the path so credit_portfolio can be imported
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run the dashboard (Streamlit executes module top-level code on import)
import credit_portfolio.dashboard  # noqa: F401
