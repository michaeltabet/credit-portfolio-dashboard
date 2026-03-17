"""Tests for analytics/commentary.py"""

import pytest
from unittest.mock import patch, MagicMock

from credit_portfolio.analytics.commentary import (
    generate_commentary_mock, generate_commentary, SYSTEM_PROMPT,
)
from credit_portfolio.analytics.attribution import attribute


@pytest.fixture
def mock_report(sample_universe_pair, sample_opt_result_pair):
    df_t0, df_t1 = sample_universe_pair
    r0, r1 = sample_opt_result_pair
    return attribute(df_t0, df_t1, r0, r1)


class TestMockCommentary:
    def test_returns_string(self, mock_report):
        text = generate_commentary_mock(mock_report)
        assert isinstance(text, str)
        assert len(text) > 50

    def test_has_two_paragraphs(self, mock_report):
        text = generate_commentary_mock(mock_report)
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        assert len(paragraphs) == 2

    def test_mentions_sector_names(self, mock_report):
        text = generate_commentary_mock(mock_report)
        # Should mention at least one sector
        sectors = ["Financials", "Healthcare", "Technology", "Energy",
                   "Industrials", "Consumer Staples", "Utilities", "Materials"]
        mentioned = any(s in text for s in sectors)
        assert mentioned

    def test_no_optimizer_jargon(self, mock_report):
        text = generate_commentary_mock(mock_report)
        jargon = ["CVXPY", "KKT", "dual variable", "solver", "CLARABEL"]
        for word in jargon:
            assert word.lower() not in text.lower()


class TestSystemPrompt:
    def test_system_prompt_guidelines(self):
        assert "CVXPY" in SYSTEM_PROMPT
        assert "two paragraphs" in SYSTEM_PROMPT.lower()


class TestAPICall:
    def test_api_call_shape(self, mock_report):
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Mock API commentary.")]
        mock_client.messages.create.return_value = mock_message

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = generate_commentary(mock_report)
            assert result == "Mock API commentary."
            mock_client.messages.create.assert_called_once()
            call_kwargs = mock_client.messages.create.call_args
            assert call_kwargs.kwargs["system"] == SYSTEM_PROMPT

    def test_model_from_config(self, mock_report):
        """LLM model and max_tokens should come from config, not hardcoded."""
        from credit_portfolio.config import load_config
        from credit_portfolio.data.constants import LLM_MODEL_ID, LLM_MAX_TOKENS

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Config test.")]
        mock_client.messages.create.return_value = mock_message

        cfg = load_config()
        expected_model = cfg.get("llm", {}).get("model", LLM_MODEL_ID)
        expected_tokens = cfg.get("llm", {}).get("max_tokens", LLM_MAX_TOKENS)

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            generate_commentary(mock_report)
            call_kwargs = mock_client.messages.create.call_args
            assert call_kwargs.kwargs["model"] == expected_model
            assert call_kwargs.kwargs["max_tokens"] == expected_tokens
