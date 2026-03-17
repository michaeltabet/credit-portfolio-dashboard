"""Statistical models: HMM regime detection, Black-Litterman, Prophet views."""

from credit_portfolio.models.hmm_regime import fit_hmm, HMMResult
from credit_portfolio.models.black_litterman import (
    BLResult, run_black_litterman, black_litterman_posterior,
)
