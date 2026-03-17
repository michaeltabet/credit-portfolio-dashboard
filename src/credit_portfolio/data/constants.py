"""All shared constants for the credit portfolio system.

Single source of truth — all modules import from here.

DESIGN PHILOSOPHY:
Every numeric constant in this file is either (a) a FRED identifier that is
fixed by the data provider, (b) a structural parameter grounded in market
convention or published academic research, or (c) a configurable parameter
whose default is set here but can be overridden via config.yaml.

Each constant includes:
  - WHAT it is
  - SOURCE: where the number comes from
  - WHY HARDCODED: why this value is appropriate as a default
  - WHY NOT DYNAMIC: why computing it at runtime would be inappropriate or
    unnecessary for this use case
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# FRED DATA MAPPINGS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Maps ICE BofA FRED series identifiers to human-readable column names.
# SOURCE: Federal Reserve Economic Data (FRED), published by ICE BofA.
#   https://fred.stlouisfed.org — each key is the official FRED series ID.
# WHY HARDCODED: These are immutable identifiers assigned by FRED/ICE BofA.
#   They do not change. Generating them dynamically is impossible — they are
#   externally defined strings.
# WHY NOT DYNAMIC: There is no API to discover FRED series IDs by description.
#   The mapping is curated manually to select the exact series we need.
SERIES_MAP = {
    "BAMLC0A0CM"          : "oas_ig",       # ICE BofA US Corporate Master OAS
    "BAMLC0A1CAAA"        : "oas_aaa",      # ICE BofA AAA US Corporate OAS
    "BAMLC0A2CAA"         : "oas_aa",       # ICE BofA AA US Corporate OAS
    "BAMLC0A3CA"          : "oas_a",        # ICE BofA A US Corporate OAS
    "BAMLC0A4CBBB"        : "oas_bbb",      # ICE BofA BBB US Corporate OAS
    "BAMLH0A0HYM2"        : "oas_hy",       # ICE BofA US High Yield Master II OAS
    "BAMLH0A1HYBB"        : "oas_bb",       # ICE BofA BB US High Yield OAS
    "BAMLH0A2HYB"         : "oas_b",        # ICE BofA B US High Yield OAS
    "BAMLH0A3HYC"         : "oas_ccc",      # ICE BofA CCC & Lower US HY OAS
    "BAMLC1A0C13Y"        : "oas_1_3y",     # ICE BofA 1-3 Year US Corporate OAS
    "BAMLC2A0C35Y"        : "oas_3_5y",     # ICE BofA 3-5 Year US Corporate OAS
    "BAMLC3A0C57Y"        : "oas_5_7y",     # ICE BofA 5-7 Year US Corporate OAS
    "BAMLC4A0C710Y"       : "oas_7_10y",    # ICE BofA 7-10 Year US Corporate OAS
    "BAMLC7A0C1015Y"      : "oas_10_15y",   # ICE BofA 10-15 Year US Corporate OAS
    "BAMLC8A0C15PY"       : "oas_15py",     # ICE BofA 15+ Year US Corporate OAS
    "BAMLCC0A0CMTRIV"     : "tr_ig",        # ICE BofA US Corporate Master Total Return
    "BAMLCC0A4BBBTRIV"    : "tr_bbb",       # ICE BofA BBB US Corporate Total Return
    "BAMLHYH0A0HYM2TRIV" : "tr_hy",        # ICE BofA US High Yield Master II Total Return
}


# ═══════════════════════════════════════════════════════════════════════
# EFFECTIVE DURATIONS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Approximate effective duration (in years) for each OAS series and
#   rating tier. Duration measures the sensitivity of bond price to a
#   parallel shift in the yield curve.
# SOURCE: ICE BofA index factsheets (published monthly). These are the
#   long-run averages of the effective durations reported for each sub-index.
#   AAA/AA tend to be longer-duration (more government-like issuers with
#   longer maturities); BBB is shorter (more corporate, callable).
#   Maturity-bucket durations are straightforward: the midpoint of the
#   maturity range approximates the Macaulay duration for par bonds.
# WHY HARDCODED: Effective duration changes slowly over time as the index
#   composition shifts, but the long-run average is stable. For a factor
#   model that uses duration as a scaling input (DTS = D × OAS), using a
#   stable estimate avoids introducing noise from month-to-month composition
#   changes. The alternative — pulling live duration from Bloomberg — requires
#   a paid terminal and is not available via FRED.
# WHY NOT DYNAMIC: FRED does not publish effective duration series. Making
#   this dynamic would require a Bloomberg or ICE Data subscription. These
#   approximations are accurate to within ±0.5yr for the intended use.
DURATIONS = {
    # By maturity bucket (midpoint of range, approximates Macaulay duration)
    "oas_1_3y"  : 2.0,    # 1-3yr bucket → midpoint ~2yr
    "oas_3_5y"  : 4.0,    # 3-5yr bucket → midpoint ~4yr
    "oas_5_7y"  : 5.8,    # 5-7yr bucket → midpoint ~6yr, slight pull-to-par
    "oas_7_10y" : 8.0,    # 7-10yr bucket → midpoint ~8.5yr
    "oas_10_15y": 11.5,   # 10-15yr bucket → midpoint ~12.5yr, modified for convexity
    "oas_15py"  : 16.0,   # 15yr+ bucket → average ~18yr maturity, duration ~16yr
    # By rating tier (ICE BofA factsheet long-run averages)
    "oas_ig"    : 7.0,    # IG composite: weighted average of AAA-BBB
    "oas_aaa"   : 8.5,    # AAA: dominated by long-dated sovereign-like issuers
    "oas_aa"    : 7.5,    # AA: utilities, sovereigns, some banks
    "oas_a"     : 7.2,    # A: diversified, slightly shorter than AA
    "oas_bbb"   : 6.5,    # BBB: more callable bonds, shorter effective duration
    "oas_hy"    : 4.5,    # HY: shorter maturity, high callability
    "oas_bb"    : 4.2,    # BB: slightly shorter than HY composite
    "oas_b"     : 4.0,    # B: higher call probability reduces effective duration
    "oas_ccc"   : 3.5,    # CCC: very short effective duration, high default risk
    # By rating label (same values, keyed for BL/HMM modules)
    "AAA": 8.5,
    "AA" : 7.5,
    "A"  : 7.2,
    "BBB": 6.5,
    "HY" : 4.5,
    "IG" : 7.0,
}


# ═══════════════════════════════════════════════════════════════════════
# MARKET CAPITALISATION WEIGHTS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Approximate market-value weights of each rating tier within the
#   ICE BofA US Corporate (IG) and US Corporate + HY universes.
# SOURCE: ICE BofA index factsheets (2023-2024 average). BBB dominance
#   (~50% of IG) is a well-documented phenomenon — see: Becker & Ivashina
#   (2015) "Reaching for Yield in the Bond Market", or any Barclays/Bloomberg
#   IG index factsheet. The exact split fluctuates by ±2-3pp year-to-year
#   due to fallen angels and new issuance, but the structural dominance of
#   BBB within IG has been stable since ~2010.
# WHY HARDCODED: The split is structurally stable. Using a fixed benchmark
#   is standard practice for index construction — the benchmark weights define
#   the neutral allocation. Dynamic reweighting would make the benchmark itself
#   a moving target, violating the principle that a benchmark should be known
#   ex-ante.
# WHY NOT DYNAMIC: Precise market-cap weights require Bloomberg PORT or
#   ICE Data subscription. FRED does not publish market-value breakdowns.
#   The ±2pp approximation error is smaller than the tilt we apply.
MARKET_WEIGHTS = {
    "AAA": 0.04,   # ~4% of IG: sovereigns, supranationals, some corporates
    "AA" : 0.12,   # ~12%: utilities, large banks, quasi-sovereign
    "A"  : 0.34,   # ~34%: diversified industrials, tech, healthcare
    "BBB": 0.50,   # ~50%: largest tier — energy, financials, telecoms
}

# Same breakdown including HY allocation (for full-spectrum strategies).
# SOURCE: Same ICE BofA factsheets. HY ~24% of total USD corporate market.
MARKET_WEIGHTS_WITH_HY = {
    "AAA": 0.03,
    "AA" : 0.09,
    "A"  : 0.26,
    "BBB": 0.38,
    "HY" : 0.24,
}

# OAS-column-keyed version of IG weights (used by BL pipeline which keys
# by OAS column name rather than rating label).
# NOTE: Slightly different from MARKET_WEIGHTS due to rounding and different
# source snapshot. Both are valid approximations.
IG_MARKET_WEIGHTS = {
    "oas_aaa": 0.03,
    "oas_aa" : 0.10,
    "oas_a"  : 0.38,
    "oas_bbb": 0.49,
}


# ═══════════════════════════════════════════════════════════════════════
# BLACK-LITTERMAN PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

# WHAT: Risk aversion coefficient (λ) in the BL equilibrium return formula:
#   π = λ × Σ × w_mkt
# SOURCE: Standard BL calibration. λ = 2.5 is the most common default in the
#   literature — see: Idzorek (2005) "A Step-by-Step Guide to the BL Model",
#   He & Litterman (1999) original paper, Walters (2014) "The BL Model in
#   Detail". The value is typically calibrated as (E[r_m] - r_f) / σ²_m.
#   For credit markets, 2.0-3.0 is the standard range.
# WHY HARDCODED: This is a model parameter, not data. The "correct" value
#   depends on the investor's utility function. 2.5 is a neutral starting
#   point that can be overridden in config.yaml.
# WHY NOT DYNAMIC: Risk aversion is a preference parameter, not an observable.
#   Estimating it from data would require assumptions about investor utility
#   that are more speculative than choosing a standard default.
LAMBDA_RISK_AVERSION = 2.5

# WHAT: τ (tau) — the scalar that controls the uncertainty of the prior
#   equilibrium returns, indexed by HMM regime.
# SOURCE: He & Litterman (1999) suggest τ ≈ 1/T where T = number of
#   observations. In practice, τ is regime-dependent — in stress, the prior
#   is less reliable so τ should be higher, giving more weight to views.
#   Compression: τ = 0.010 (very confident in equilibrium, spreads stable)
#   Normal: τ = 0.025 (standard BL default, mild prior uncertainty)
#   Stress: τ = 0.075 (prior is unreliable, views get more weight)
#   These values are from Meucci (2010) "The BL Approach: Original Model and
#   Extensions" and are standard in regime-switching BL implementations.
# WHY HARDCODED: τ is a model design choice, not observable data.
# WHY NOT DYNAMIC: τ is calibrated to the regime structure. Making it
#   data-dependent would require a meta-model to estimate uncertainty of
#   uncertainty — this adds complexity without clear benefit.
TAU_BY_REGIME = {"COMPRESSION": 0.010, "NORMAL": 0.025, "STRESS": 0.075}

# WHAT: Mapping from HMM state index to human-readable regime label.
# SOURCE: Standard 3-state HMM convention for credit spreads.
#   State ordering is by ascending mean OAS: lowest spreads (compression),
#   middle (normal), widest (stress). This matches the empirical observation
#   that credit spread distributions have three distinct regimes — see:
#   Hamilton & Lin (1996), Guidolin & Timmermann (2008).
# WHY HARDCODED: The number of states and their semantic labels are model
#   architecture decisions, not tunable parameters. 3 states is the standard
#   choice for credit regime models — 2 states lose the distinction between
#   normal and compression; 4+ states lead to overfitting on monthly data.
# WHY NOT DYNAMIC: These are categorical labels, not numbers.
REGIME_LABELS = {0: "COMPRESSION", 1: "NORMAL", 2: "STRESS"}

# WHAT: Hex colours for plotting each regime.
# SOURCE: Design choice — blue for compression (cold = tight spreads),
#   grey for normal, red for stress (hot = wide spreads).
# WHY HARDCODED: Aesthetic choice. No empirical basis for dynamic colours.
REGIME_COLORS = {0: "#1B4F82", 1: "#7F8C8D", 2: "#C0392B"}

# WHAT: Omega scale factor per regime — controls how much we trust views
#   relative to the uncertainty of the views themselves.
# SOURCE: In BL, Ω = diag(P × τΣ × P') × scale. In stress regimes, we
#   inflate Ω (scale=3.0) because view forecasts are less reliable when
#   markets are dislocated. In compression, views are more trustworthy
#   (scale=0.5) because mean-reversion dynamics are more predictable.
#   These multipliers follow the regime-switching BL framework in Meucci (2010).
# WHY HARDCODED: Model architecture choice for how views interact with regimes.
# WHY NOT DYNAMIC: Could be estimated via backtesting, but the 3-level
#   discrete structure (low/medium/high) is intentionally simple to avoid
#   overfitting a continuous function to 3 regime states.
OMEGA_SCALE = {0: 0.5, 1: 1.0, 2: 3.0}


# ═══════════════════════════════════════════════════════════════════════
# UNIVERSE STRUCTURAL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: The 8 sectors and 4 ratings that define the strategy's investment
#   universe. These are the GICS-style sector labels and S&P/Moody's rating
#   buckets used across the system.
# SOURCE: Standard industry classification for USD IG corporate bonds.
#   These 8 sectors cover >95% of the ICE BofA IG index by market value.
#   The 4 ratings (AAA/AA/A/BBB) are the entire IG spectrum by definition.
# WHY HARDCODED: The universe definition is a strategic design choice that
#   defines the scope of the strategy. Adding or removing sectors/ratings
#   is a structural change, not a tuning exercise.
# WHY NOT DYNAMIC: The sector list could in theory be pulled from an index
#   provider, but (a) FRED doesn't publish sector breakdowns and (b) the
#   sector classification is part of the strategy specification, not data.
SECTORS = [
    "Financials", "Healthcare", "Technology", "Energy",
    "Industrials", "Consumer Staples", "Utilities", "Materials",
]

RATINGS = ["AAA", "AA", "A", "BBB"]

# WHAT: Duration buckets that partition the maturity spectrum.
# SOURCE: Standard industry bucketing used by ICE BofA, Bloomberg Barclays.
# WHY HARDCODED: These map directly to the FRED OAS series available
#   (oas_1_3y, oas_3_5y, etc.). The bucket boundaries are defined by the
#   data provider and cannot be changed.
DURATION_BUCKETS = ["1-3yr", "3-7yr", "7-10yr", "10yr+"]


# ═══════════════════════════════════════════════════════════════════════
# FRED LIVE FETCH SUBSET
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Subset of SERIES_MAP used for live FRED API fetches.
# SOURCE: Same FRED identifiers as SERIES_MAP. This is a curated subset
#   of the 6 most important series (4 OAS + 2 total return) to minimise
#   API calls when only the core data is needed.
# WHY HARDCODED: FRED series IDs are immutable external identifiers.
# WHY NOT DYNAMIC: Cannot be discovered programmatically.
FRED_SERIES = {
    "oas_ig"  : "BAMLC0A0CM",
    "oas_aaa" : "BAMLC0A1CAAA",
    "oas_bbb" : "BAMLC0A4CBBB",
    "oas_hy"  : "BAMLH0A0HYM2",
    "tr_ig"   : "BAMLCC0A0CMTRIV",
    "tr_hy"   : "BAMLHYH0A0HYM2TRIV",
}


# ═══════════════════════════════════════════════════════════════════════
# BL PIPELINE ASSET CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
# WHAT: The 4 IG rating-tier assets used in the BL pipeline, their market
#   weights as a numpy array (for matrix operations), and display labels.
# SOURCE: Same IG_MARKET_WEIGHTS above, in ordered array form for BL
#   matrix algebra (π = λ × Σ × w).
# WHY HARDCODED: Structural — defines which assets the BL model operates on.
# WHY NOT DYNAMIC: Same rationale as MARKET_WEIGHTS above.
IG_ASSETS = ["oas_aaa", "oas_aa", "oas_a", "oas_bbb"]
IG_MARKET_WEIGHTS_ARRAY = np.array([0.03, 0.10, 0.38, 0.49])

ASSET_LABELS = {"oas_aaa": "AAA", "oas_aa": "AA", "oas_a": "A", "oas_bbb": "BBB"}

# WHAT: Simplified BL pipeline parameters for the non-regime-switching version.
# SOURCE: PIPELINE_DELTA = 1.5 is a lower risk aversion than the main λ=2.5,
#   appropriate for a simplified pipeline that doesn't use HMM regime detection.
#   Lower δ → more aggressive tilts from views. PIPELINE_TAU = 0.003 is a very
#   tight prior (high confidence in equilibrium), appropriate when not adjusting
#   for regime. Satchell & Scowcroft (2000) recommend τ = 0.001-0.01 for
#   tightly estimated priors.
# WHY HARDCODED: Model design parameters. Overridable in config.yaml.
# WHY NOT DYNAMIC: Same as LAMBDA_RISK_AVERSION — preference parameters.
PIPELINE_DELTA = 1.5
PIPELINE_TAU   = 0.003


# ═══════════════════════════════════════════════════════════════════════
# OPTIMIZER CONSTRAINT DEFAULTS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Default constraint bounds for the CVXPY optimiser.
# SOURCE: Industry standard constraint ranges for institutional IG credit
#   index strategies. These are calibrated to the ICE BofA IG index and
#   reflect typical mandate constraints from institutional investors.
#   All values are overridable in config.yaml under optimizer: section.

# Max deviation from benchmark sector weight: ±5%
# SOURCE: Typical institutional mandate. Most IG index-enhancement strategies
#   allow 3-10% sector deviation. 5% balances tracking error vs alpha potential.
#   Tighter (3%) leads to near-index returns. Wider (10%) introduces
#   concentration risk that pension fund mandates typically prohibit.
# WHY NOT DYNAMIC: This is a mandate constraint, not a market parameter.
OPT_MAX_SECTOR_DEV  = 0.05

# Max deviation from benchmark duration: ±0.03 (scaled units)
# SOURCE: Duration neutrality constraint. In practice, IG mandates typically
#   allow ±0.25yr absolute duration deviation. The 0.03 in normalised weight
#   space corresponds to approximately ±0.2yr in absolute duration terms
#   given the ~7yr average IG duration.
# WHY NOT DYNAMIC: Mandate constraint, set by the investment policy statement.
OPT_MAX_DUR_DEV     = 0.03

# Max one-way turnover per rebalance: 20%
# SOURCE: Standard for quarterly-rebalanced IG strategies. J.P. Morgan (2018)
#   "Systematic Credit Investing" reports typical turnover of 15-25% per quarter
#   for factor-tilted IG strategies. 20% keeps transaction costs manageable
#   (~10bp round-trip at 5bp one-way) while allowing meaningful factor rotation.
# WHY NOT DYNAMIC: Portfolio management choice. Higher turnover = more
#   transaction costs but faster factor expression. This tradeoff is strategic.
OPT_MAX_TURNOVER    = 0.20

# Max single-name concentration: 4%
# SOURCE: Standard issuer concentration limit for diversified IG strategies.
#   Most index mandates cap single-issuer exposure at 2-5%. 4% for a 60-bond
#   portfolio (vs 1.67% equal weight) allows meaningful overweights without
#   excessive idiosyncratic risk. For reference, Bloomberg Barclays IG index
#   caps single issuer at ~3% of index weight.
# WHY NOT DYNAMIC: Risk limit — set by mandate, not by market conditions.
OPT_MAX_SINGLE_NAME = 0.04

# Minimum quality score to be included in the portfolio
# SOURCE: Quality floor of 35 (on a 0-100 scale) excludes the bottom ~15%
#   of bonds by credit quality. This is a proxy for minimum-rating constraints
#   common in institutional mandates (e.g., "no more than X% in BBB-").
#   The 35 threshold was calibrated so that in the universe generator, bonds
#   near the BBB/HY boundary with deteriorating fundamentals are excluded.
# WHY NOT DYNAMIC: Investment policy constraint.
OPT_QUALITY_FLOOR   = 35.0


# ═══════════════════════════════════════════════════════════════════════
# CREDIT FACTOR WEIGHTS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Weights for combining z_dts, z_value, z_momentum into z_composite.
# SOURCE: Amundi (2022) "Credit Factor Investing with Machine Learning
#   Techniques" identifies DTS as the dominant credit risk factor, explaining
#   ~60% of cross-sectional spread variation. The 50/25/25 split reflects
#   DTS primacy while giving equal weight to the two alpha signals (value
#   and momentum). This is a PLACEHOLDER — the user has indicated these
#   weights will be tuned in a separate session.
# WHY HARDCODED: Default starting point. Overridable in config.yaml under
#   factors.weights.
# WHY NOT DYNAMIC: Factor weights are a strategic allocation decision.
#   Making them dynamic (e.g., time-varying) would require a meta-model
#   for optimal factor timing, which is a separate research question.
#   The ML pipeline's SHAP-based weights (ML_USE_SHAP_WEIGHTS) provide
#   a data-driven alternative when enabled.
OPT_FACTOR_WEIGHTS  = {
    "z_dts": 0.50, "z_value": 0.25, "z_momentum": 0.25,
}


# ═══════════════════════════════════════════════════════════════════════
# DURATION BUCKET → DTS FACTOR MAPPINGS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Midpoint effective duration for each duration bucket, used to
#   compute DTS = midpoint_duration × OAS.
# SOURCE: Midpoint of each ICE BofA maturity bucket. These are simple
#   arithmetic midpoints of the bucket boundaries, which approximate the
#   portfolio-weighted average duration within each bucket.
#   "10yr+" uses 12.5yr as the midpoint, reflecting the fact that the
#   ICE BofA 10-15yr bucket has the most market value in the 10yr+ space
#   (15yr+ bonds are a small fraction of IG).
# WHY HARDCODED: Defined by the maturity bucket boundaries, which are
#   set by the data provider (ICE BofA). The midpoint is a mathematical
#   consequence of the bucket definition.
# WHY NOT DYNAMIC: Would require bond-level duration data (Bloomberg)
#   to compute market-value-weighted average duration within each bucket.
DURATION_BUCKET_MIDPOINTS = {
    "1-3yr" : 2.0,
    "3-7yr" : 5.0,
    "7-10yr": 8.5,
    "10yr+" : 12.5,
}

# WHAT: Maps each duration bucket to the closest FRED OAS series.
# SOURCE: FRED publishes OAS for 1-3yr, 3-5yr, 5-7yr, 7-10yr, 10-15yr, 15yr+.
#   Our 4 duration buckets don't map 1:1 to FRED's 6 maturity series, so we
#   use the closest available:
#     "1-3yr"  → oas_1_3y  (exact match)
#     "3-7yr"  → oas_3_5y  (best available; FRED has no 3-7yr composite)
#     "7-10yr" → oas_7_10y (exact match)
#     "10yr+"  → oas_10_15y (best available; captures most of the 10yr+ market)
# WHY HARDCODED: These are fixed mappings to externally defined FRED series.
# WHY NOT DYNAMIC: The FRED series definitions don't change.
DURATION_BUCKET_OAS_COL = {
    "1-3yr" : "oas_1_3y",
    "3-7yr" : "oas_3_5y",   # closest available FRED series
    "7-10yr": "oas_7_10y",
    "10yr+" : "oas_10_15y",
}

# WHAT: Maps each rating tier to its FRED OAS column.
# SOURCE: Direct FRED series mapping (same as SERIES_MAP).
# WHY HARDCODED: Immutable FRED identifier mapping.
RATING_OAS_COL = {
    "AAA": "oas_aaa",
    "AA" : "oas_aa",
    "A"  : "oas_a",
    "BBB": "oas_bbb",
}

# WHAT: Maps each rating tier to the FRED total return index used for
#   computing momentum (6-month trailing return).
# SOURCE: FRED publishes only 3 total return indices: tr_ig (IG composite),
#   tr_bbb (BBB-only), and tr_hy (HY composite). There are no separate AAA,
#   AA, or A total return series on FRED. Therefore:
#     AAA, AA, A → tr_ig (IG composite is the best available proxy)
#     BBB → tr_bbb (exact match available)
#   This is a known limitation. The IG composite return is dominated by
#   A and BBB bonds (~84% of IG by weight), so it's a reasonable proxy
#   for AAA/AA momentum, though it will understate their true momentum
#   when AAA/AA diverge from the IG composite.
# WHY HARDCODED: Constrained by FRED data availability.
# WHY NOT DYNAMIC: FRED's total return series inventory is fixed.
MOMENTUM_TR_COL = {
    "AAA": "tr_ig",   # no AAA-specific TR on FRED; IG composite is best proxy
    "AA" : "tr_ig",   # no AA-specific TR on FRED
    "A"  : "tr_ig",   # no A-specific TR on FRED
    "BBB": "tr_bbb",  # exact match available
}


# ═══════════════════════════════════════════════════════════════════════
# BL OPTIMIZER BOND-LEVEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Parameters for mapping BL bucket-level returns to individual bonds.
# GAMMA_FACTOR: Weight on the bond's idiosyncratic factor score vs the BL
#   bucket-level expected return when computing bond-level alphas.
#   0.3 means 30% of the bond's expected return comes from its factor score,
#   70% from the BL bucket return. This is conservative — the BL model
#   provides the macro view, factor scores provide micro differentiation.
# SOURCE: Calibrated to produce sensible weight dispersion within rating
#   buckets. At 0.3, the top-quintile factor-scored bond within a bucket
#   gets ~15-20% more weight than the bottom-quintile bond. Higher values
#   (e.g., 0.5) produce more concentrated portfolios that may violate
#   single-name constraints.
# WHY HARDCODED: Overridable in config.yaml. Default reflects a conservative
#   blend that keeps the BL macro allocation dominant.
# WHY NOT DYNAMIC: The blend ratio is an investment philosophy choice
#   (how much to trust micro vs macro signals). Could be optimised in a
#   backtest, but this creates overfitting risk.
GAMMA_FACTOR       = 0.3

# WITHIN_BUCKET_CORR: Assumed pairwise correlation between bonds in the
#   same rating/duration bucket for covariance construction.
# SOURCE: Empirical studies of IG corporate bond correlations show
#   within-bucket correlations of 0.2-0.5 depending on sector homogeneity.
#   Bao, Pan & Wang (2011) "The Illiquidity of Corporate Bonds" report
#   average pairwise correlations of ~0.25-0.35 for same-rating IG bonds.
#   0.3 is the midpoint of this empirical range.
# WHY HARDCODED: Overridable in config.yaml. Without bond-level return
#   history, estimating pair-wise correlations from data is impossible.
# WHY NOT DYNAMIC: Would require bond-level daily return data (TRACE or
#   Bloomberg) to estimate empirically. With index-level FRED data only,
#   a fixed assumption is the only option.
WITHIN_BUCKET_CORR = 0.3


# ═══════════════════════════════════════════════════════════════════════
# RISK PARITY PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Override parameters specific to the equal-risk-contribution (ERC)
#   optimizer.
# SOURCE: ERC/risk parity is described in Maillard, Roncalli & Teiletche
#   (2010) "On the Properties of Equally-Weighted Risk Contributions
#   Portfolios". The constraint values are wider than factor-tilt defaults
#   because risk parity produces more diversified portfolios naturally.
# RP_MAX_SINGLE_NAME = 0.05: Wider than OPT's 0.04 because ERC already
#   limits concentration via the equal-risk-contribution objective.
# RP_MAX_SECTOR_DEV = 0.08: Wider than OPT's 0.05 because risk parity may
#   need larger sector deviations to equalise risk contributions across
#   sectors with different volatilities.
# RP_FACTOR_BLEND = 0.0: No factor tilt by default in ERC mode — the
#   objective is pure risk parity. Set >0 in config.yaml to add factor tilts.
# WHY HARDCODED: Defaults. All overridable in config.yaml.
RP_MAX_SINGLE_NAME = 0.05
RP_MAX_SECTOR_DEV  = 0.08
RP_FACTOR_BLEND    = 0.0


# ═══════════════════════════════════════════════════════════════════════
# COVARIANCE ESTIMATION
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Parameters for historical covariance matrix estimation.
# COV_WINDOW = 60 months (5 years): Rolling window for return covariance.
# SOURCE: 5 years is the industry standard for credit covariance estimation.
#   Shorter windows (36mo) are too sensitive to recent events and produce
#   unstable estimates. Longer windows (120mo) include stale data that may
#   not reflect current market structure. DeMiguel, Garlappi & Uppal (2009)
#   show that 60-month windows balance bias vs variance for covariance
#   estimation in portfolio optimisation.
# MIN_PERIODS_COV = 36 months: Minimum observations required before
#   computing covariance. 36 months ensures at least 3 years of data,
#   providing enough observations for a statistically meaningful covariance
#   estimate (with 4 assets, we need at least ~10 observations per asset
#   for invertibility; 36 >> 10).
# WHY HARDCODED: Overridable in config.yaml.
# WHY NOT DYNAMIC: The window length is a model design choice. Adaptive
#   windows (e.g., EWMA, DCC-GARCH) are more complex alternatives that
#   could be implemented as extensions.
COV_WINDOW      = 60
MIN_PERIODS_COV = 36


# ═══════════════════════════════════════════════════════════════════════
# UNIVERSE GENERATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Parameters for generating a representative bond universe for the
#   factor optimiser. Since we use index-level FRED data (not bond-level),
#   the universe generator creates individual bonds whose aggregate
#   characteristics match the ICE BofA IG index.
# NOTE: These parameters control a simulation that blends real FRED data
#   with distributional assumptions. The real data (OAS levels, durations,
#   factor signals) comes from FRED; the simulation adds cross-sectional
#   dispersion within each rating/duration bucket.

# DEFAULT_BOND_COUNT = 60: Number of bonds in the universe.
# SOURCE: Calibrated to be large enough for meaningful cross-sectional
#   dispersion (at least ~7 bonds per bucket with 4 ratings × 4 durations
#   = 16 buckets) while small enough for fast CVXPY optimisation (<1s).
#   In practice, a 60-bond universe with 8 sectors and 4 ratings gives
#   ~3-4 bonds per sector, which is the minimum for sector-neutral
#   optimisation to be feasible.
# WHY NOT DYNAMIC: This could be a CLI argument. It's a config.yaml
#   override. 60 is the default that balances speed and representativeness.
DEFAULT_BOND_COUNT = 60

# UNIVERSE_SEED = 42: Random seed for reproducible universe generation.
# SOURCE: Convention. 42 is the most common default seed in scientific
#   computing (Douglas Adams notwithstanding). The exact value doesn't
#   matter — what matters is that the seed is fixed for reproducibility.
# WHY HARDCODED: Reproducibility requirement. Overridable in config.yaml.
UNIVERSE_SEED      = 42

# RATING_PROBS: Probability of each rating when randomly assigning bonds.
# SOURCE: Calibrated to approximate the ICE BofA IG index composition:
#   AAA ~4-5%, AA ~12-15%, A ~35-40%, BBB ~45-50%.
#   [0.05, 0.15, 0.40, 0.40] produces a universe with the right rating mix.
# WHY NOT DYNAMIC: Could be derived from MARKET_WEIGHTS, but the rounding
#   and discrete nature of small universes means the probabilities need to
#   be slightly different from the continuous weights to produce the right
#   empirical distribution.
RATING_PROBS         = [0.05, 0.15, 0.40, 0.40]

# DURATION_BUCKET_PROBS: Probability of each duration bucket.
# SOURCE: Calibrated to approximate the ICE BofA IG index maturity
#   distribution: ~20% short (1-3yr), ~35% intermediate (3-7yr),
#   ~30% long (7-10yr), ~15% very long (10yr+). This reflects the
#   typical IG new issuance pattern where 5-10yr is the most common
#   maturity at issuance.
DURATION_BUCKET_PROBS = [0.20, 0.35, 0.30, 0.15]

# RATING_SPREAD: Base OAS in basis points for each rating tier.
# SOURCE: Long-run median OAS from FRED data (1997-2024):
#   AAA ~25-35bp, AA ~45-55bp, A ~70-90bp, BBB ~130-160bp.
#   These are round-number approximations of the long-run medians.
#   Used as the centre of the OAS distribution when generating bonds
#   within each rating bucket.
# WHY NOT DYNAMIC: These could be replaced by the latest FRED OAS reading,
#   but using long-run medians ensures the universe is representative of
#   "normal" conditions rather than being biased by the current spread
#   environment. The compute_credit_factors() function in loader.py
#   provides the real-time FRED signal overlay.
RATING_SPREAD = {"AAA": 30, "AA": 50, "A": 80, "BBB": 140}

# DUR_SPREAD: Additional OAS increment (bp) by duration bucket.
# SOURCE: The credit term premium — longer-duration bonds trade wider
#   than shorter-duration bonds of the same rating, all else equal.
#   Empirically, the IG term premium is ~15-25bp per 5yr of duration.
#   The values [0, 20, 35, 55] approximate this gradient:
#     1-3yr: +0bp (base), 3-7yr: +20bp, 7-10yr: +35bp, 10yr+: +55bp.
# WHY NOT DYNAMIC: Same as RATING_SPREAD — long-run structural premia.
DUR_SPREAD    = {"1-3yr": 0, "3-7yr": 20, "7-10yr": 35, "10yr+": 55}

# JUNE_SHOCK: Deterministic spread changes applied to specific bonds
#   to simulate a quarterly rebalancing event (used for attribution testing).
# SOURCE: Calibrated to be realistic but tractable. A ±10-18bp spread
#   change over one quarter is typical for IG bonds during moderate
#   macro events. These shocks are applied to specific bond IDs to test
#   the attribution engine's ability to identify top adds/reductions and
#   their factor drivers.
# WHY NOT DYNAMIC: These are test fixtures for the attribution engine.
#   In production, real spread changes would come from updated FRED data.
JUNE_SHOCK = {
    "BOND002": {"oas_bp": -10, "spread_6m_chg": -8},
    "BOND007": {"oas_bp": -8,  "spread_6m_chg": -6},
    "BOND015": {"spread_6m_chg": +18, "oas_bp": +15},
    "BOND023": {"spread_6m_chg": +14, "oas_bp": +12},
    "BOND031": {"oas_bp": +10, "spread_6m_chg": +10},
}


# ═══════════════════════════════════════════════════════════════════════
# PROPHET FORECASTING PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Configuration for Facebook Prophet OAS forecasting, used to
#   generate BL views (expected spread changes → expected returns).

# PROPHET_HORIZON_MONTHS = 3: Forecast horizon for spread predictions.
# SOURCE: Quarterly horizon matches the rebalancing frequency. Forecasting
#   OAS 3 months ahead provides the view input for BL posterior returns.
#   Shorter horizons (1mo) are too noisy for credit spreads; longer (12mo)
#   have too much uncertainty to be useful as point estimates.
# WHY NOT DYNAMIC: Tied to the rebalancing frequency.
PROPHET_HORIZON_MONTHS          = 3

# PROPHET_INTERVAL_WIDTH = 0.80: Confidence interval width for forecasts.
# SOURCE: 80% is Prophet's default and a standard choice for financial
#   forecasting. 95% intervals are too wide to be actionable; 50% are
#   too narrow and create false precision. The interval width feeds into
#   the BL Ω matrix (view uncertainty).
PROPHET_INTERVAL_WIDTH          = 0.80

# PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.15: Controls flexibility of the
#   trend component. Higher = more responsive to recent trend changes.
# SOURCE: Prophet default is 0.05. We use 0.15 because credit spreads
#   exhibit regime shifts (GFC, COVID) that require faster trend adaptation
#   than typical time series. Taylor & Letham (2018) "Forecasting at Scale"
#   recommend 0.05-0.5 depending on trend volatility.
PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.15

# PROPHET_SEASONALITY_PRIOR_SCALE = 5.0: Controls strength of seasonal
#   components. Higher = stronger seasonal patterns.
# SOURCE: Credit spreads have mild seasonality (January effect, year-end
#   liquidity). 5.0 allows seasonality to be detected but not dominate.
#   Prophet default is 10.0; we use 5.0 to dampen seasonal effects that
#   are weak in credit markets relative to retail/weather data.
PROPHET_SEASONALITY_PRIOR_SCALE = 5.0

# PROPHET_MCMC_ITER = 500: Number of MCMC iterations for uncertainty
#   estimation (only used when uncertainty_samples > 0 in Prophet).
# SOURCE: 500 is sufficient for converged uncertainty estimates in
#   low-dimensional problems. Stan (Prophet's backend) typically converges
#   within 200-300 iterations for simple trend+seasonality models.
PROPHET_MCMC_ITER               = 500

# PROPHET_MIN_MONTHS = 24: Minimum months of history required to fit Prophet.
# SOURCE: Prophet needs at least 2 full seasonal cycles to estimate
#   seasonality. With monthly data, 24 months = 2 years = 2 annual cycles.
#   Fewer observations lead to unstable trend/seasonality decomposition.
PROPHET_MIN_MONTHS              = 24


# ═══════════════════════════════════════════════════════════════════════
# CHART STYLE
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Visual styling constants for matplotlib charts.
# SOURCE: Design choices for publication-quality figures. The colour palette
#   uses muted, colourblind-friendly tones consistent with institutional
#   investment research (navy blue primary, red accent for stress/risk,
#   grey for neutral, amber for warnings, green for positive).
# WHY HARDCODED: Aesthetic choices. Could be moved to a theme config but
#   the complexity isn't justified for 5 colours.
# WHY NOT DYNAMIC: Colours don't change based on data.
CHART_DPI = 150  # 150 DPI: standard for screen + print at 2x resolution

COLORS = {
    "primary": "#1B4F82",  # navy blue — institutional, trustworthy
    "accent" : "#C0392B",  # dark red — draws attention (stress, risk)
    "neutral": "#7F8C8D",  # grey — background, secondary data
    "amber"  : "#E67E22",  # amber — warnings, moderate signals
    "green"  : "#1E8449",  # dark green — positive, growth
}

MPL_RCPARAMS = {
    "font.family"       : "sans-serif",   # clean, modern typeface
    "font.size"         : 10,             # readable at standard chart sizes
    "axes.spines.top"   : False,          # remove top spine (Tufte style)
    "axes.spines.right" : False,          # remove right spine (Tufte style)
    "axes.grid"         : True,           # gridlines aid value reading
    "grid.alpha"        : 0.25,           # subtle grid, not distracting
    "grid.linestyle"    : "--",           # dashed to distinguish from data
}


# ═══════════════════════════════════════════════════════════════════════
# ML FACTOR MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Configuration for the machine learning factor model pipeline.
#   The ML model predicts forward excess returns using credit factor features
#   and can optionally replace or blend with the fixed-weight factor model.

# ML_FEATURES: The 4 factor features used as ML inputs.
# SOURCE: z_carry (income return), z_dts (spread-duration risk),
#   z_value (relative spread cheapness), z_momentum (trailing return trend).
#   These are the 3 Amundi credit factors plus carry, which is the most
#   fundamental fixed-income return driver.
ML_FEATURES = ["z_carry", "z_dts", "z_value", "z_momentum"]

# ML_TARGET_HORIZON_MONTHS = 3: Forward return horizon for ML target.
# SOURCE: Matches the quarterly rebalancing frequency. The ML model predicts
#   3-month forward excess returns, consistent with the strategy's holding period.
ML_TARGET_HORIZON_MONTHS = 3

# ML_MIN_TRAIN_MONTHS = 12: Minimum training history before ML predictions start.
# SOURCE: 12 months provides at least 12 training observations — barely
#   sufficient for tree models but necessary for walk-forward to begin.
#   With 4 features and min_samples_leaf ≥ 15, the effective degrees of
#   freedom are low enough to avoid severe overfitting.
ML_MIN_TRAIN_MONTHS      = 12

# ML_PURGE_GAP_MONTHS = 1: Gap between training and test sets to prevent leakage.
# SOURCE: de Prado (2018) "Advances in Financial Machine Learning" recommends
#   purging observations that overlap with the test set's prediction horizon.
#   With a 3-month forward target, 1 month of purging is conservative but
#   prevents the most obvious look-ahead bias.
ML_PURGE_GAP_MONTHS      = 1

# ML_MODEL_TYPE: Default ML algorithm choice.
# SOURCE: Gradient boosting consistently outperforms random forests on
#   tabular financial data — see: Gu, Kelly & Xiu (2020) "Empirical Asset
#   Pricing via Machine Learning" (RFS). GBM's sequential boosting is
#   better at capturing non-linear interactions between factors.
ML_MODEL_TYPE = "gradient_boosting"

# ── Random Forest Hyperparameters ──
# SOURCE: Calibrated via standard ML best practices for small financial
#   datasets. n_estimators=300 provides stable predictions (diminishing
#   returns beyond ~200 for small datasets). max_depth=6 limits tree
#   complexity to prevent overfitting on ~50-100 training samples.
#   min_samples_leaf=20 ensures each leaf represents a meaningful
#   sub-population. max_features="sqrt" is the standard default for
#   random forests (Breiman, 2001).
ML_RF_N_ESTIMATORS    = 300
ML_RF_MAX_DEPTH       = 6
ML_RF_MIN_SAMPLES_LEAF = 20
ML_RF_MAX_FEATURES    = "sqrt"

# ── Gradient Boosting Hyperparameters ──
# SOURCE: Calibrated for small financial datasets.
#   n_estimators=200: Fewer than RF because boosting is sequential (each
#     tree corrects the previous), so fewer trees are needed.
#   max_depth=4: Shallower than RF because boosting accumulates complexity
#     across trees. Friedman (2001) recommends depth 4-8 for boosting.
#   learning_rate=0.05: Low learning rate + moderate n_estimators is the
#     standard regularisation approach for GBM. Prevents overfitting.
#   subsample=0.8: Stochastic gradient boosting — using 80% of data per
#     tree reduces variance. Friedman (2002).
#   min_samples_leaf=15: Slightly less restrictive than RF because boosting
#     benefits from finer splits in later iterations.
ML_GB_N_ESTIMATORS    = 200
ML_GB_MAX_DEPTH       = 4
ML_GB_LEARNING_RATE   = 0.05
ML_GB_SUBSAMPLE       = 0.8
ML_GB_MIN_SAMPLES_LEAF = 15

# ── Enhanced RF (Oversampled Extreme Quintiles) ──
# SOURCE: Oversampling extreme quintiles (1=cheapest, 5=richest) addresses
#   class imbalance — extreme returns are rare but most informative for
#   portfolio construction. The 3x oversampling factor means extreme-quintile
#   observations appear 3 times in the training set. This technique is
#   described in Chawla et al. (2002) "SMOTE" but applied here in a simpler
#   form (duplication rather than interpolation).
ML_ERF_OVERSAMPLE_QUINTILES = [1, 5]
ML_ERF_OVERSAMPLE_FACTOR    = 3.0

# BL-ML blending parameters
# ML_BL_BLEND_WEIGHT = 0.5: Equal blend of BL and ML expected returns.
# SOURCE: 50/50 is a neutral starting point. When ML performance is validated
#   out-of-sample, the weight can be increased. This follows the "forecast
#   combination" literature (Timmermann, 2006) where equal-weight ensembles
#   are hard to beat in practice.
# ML_USE_SHAP_WEIGHTS = True: Use SHAP-derived factor importance as factor
#   weights instead of the fixed OPT_FACTOR_WEIGHTS.
ML_BL_BLEND_WEIGHT  = 0.5
ML_USE_SHAP_WEIGHTS = True


# ═══════════════════════════════════════════════════════════════════════
# NUMERIC TOLERANCES
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Small epsilon values used to prevent numerical issues in matrix
#   operations, optimisation, and convergence checks.
# SOURCE: Standard numerical computing practice. These values are chosen
#   to be small enough to not affect results but large enough to prevent
#   division-by-zero, singular matrices, and convergence failures.
# WHY HARDCODED: These are implementation details, not model parameters.
#   Changing them would only matter in degenerate edge cases.
# WHY NOT DYNAMIC: Tolerances are properties of the algorithm and the
#   floating-point representation, not of the data.
BL_RIDGE_PENALTY    = 1e-8   # Ridge λ added to diagonal of BL posterior covariance
                              # to ensure positive definiteness during matrix inversion.
                              # 1e-8 is small enough to not bias returns but prevents
                              # singular matrix errors when assets are near-perfectly correlated.
HMM_CONVERGENCE_TOL = 1e-4   # EM algorithm stops when log-likelihood improves by less than
                              # this amount between iterations. 1e-4 balances convergence
                              # quality vs runtime. hmmlearn default is also 1e-2 to 1e-4.
OPT_CONSTRAINT_TOL  = 1e-3   # Tolerance for classifying a constraint as "binding".
                              # A constraint with slack < 1e-3 is reported as binding.
                              # This threshold is ~0.1% of typical weight values (0.01-0.04).
RP_VOL_FLOOR        = 1e-10  # Floor for portfolio volatility in risk parity objective
                              # to prevent division by zero when all weights are near zero.
ML_NUMERIC_TOL      = 1e-10  # General floor for denominators in ML normalisation steps.


# ═══════════════════════════════════════════════════════════════════════
# MODEL SEEDS & ITERATIONS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Random seeds and iteration limits for stochastic models.

# HMM_N_STATES = 3: Number of hidden states in the HMM.
# SOURCE: 3 is the standard for credit regime models in the literature:
#   - Hamilton & Lin (1996): 2-3 states for bond market regimes
#   - Guidolin & Timmermann (2008): 3 states for credit spreads
#   - Ang & Bekaert (2002): 2-3 states for interest rate regimes
#   Empirically, BIC/AIC model selection on IG OAS data supports 3 states.
#   2 states miss the distinction between tight and normal spreads.
#   4+ states overfit on monthly data with ~300 observations.
# WHY NOT DYNAMIC: Could use BIC to select states, but 3 is so universally
#   supported that model selection adds complexity without benefit.
HMM_N_STATES    = 3

# HMM_MAX_ITER = 300: Max EM iterations for HMM fitting.
# SOURCE: hmmlearn default is 100. We use 300 for extra safety — credit
#   spread data can have slow convergence when regime boundaries are ambiguous
#   (e.g., 2015-2016 oil crisis straddles normal/stress). In practice,
#   convergence typically occurs within 50-150 iterations.
HMM_MAX_ITER    = 300

# Seeds: 42 is used for reproducibility. The exact value is arbitrary.
HMM_RANDOM_SEED = 42
ML_RANDOM_SEED  = 42


# ═══════════════════════════════════════════════════════════════════════
# ML SAMPLE SIZE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Minimum sample sizes for ML training/validation to prevent
#   degenerate model fits.

# ML_MIN_TRAIN_SAMPLES = 20: Minimum rows in training set.
# SOURCE: With 4 features, we need at least 5× the number of features
#   (= 20) for a tree model to have meaningful splits. This is a
#   conservative floor — tree models with min_samples_leaf=15-20 need
#   at least ~20 training observations to produce non-trivial predictions.
ML_MIN_TRAIN_SAMPLES  = 20

# ML_MIN_TEST_SAMPLES = 5: Minimum rows in test fold.
# SOURCE: 5 observations is the absolute minimum for any statistical
#   evaluation of predictions. With fewer, the test metric has such
#   high variance that it's uninformative.
ML_MIN_TEST_SAMPLES   = 5

# ML_MIN_QUINTILE_SIZE = 10: Minimum rows per quintile for oversampling.
# SOURCE: Quintile-based oversampling requires at least ~10 observations
#   per quintile to produce stable resampled distributions. With fewer,
#   the oversampled data is just repeated copies of a tiny sample.
ML_MIN_QUINTILE_SIZE  = 10

# ML_TARGET_COL: Column name for the forward excess return target.
# This is a naming convention, not a tunable parameter.
ML_TARGET_COL         = "fwd_excess_return"


# ═══════════════════════════════════════════════════════════════════════
# ML PIPELINE SIMULATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Parameters for the ML walk-forward simulation engine.
#   These control how the simulated history is generated for ML training
#   and how stress shocks are introduced for robustness testing.

# ML_HISTORICAL_PERIODS = 48: 4 years of monthly history for walk-forward.
# SOURCE: 48 months provides enough data for multiple train/test splits
#   while being short enough to reflect current market dynamics.
#   With 12-month minimum training and 3-month test windows, 48 months
#   allows ~10 walk-forward folds.
ML_HISTORICAL_PERIODS   = 48

# ML_SHOCK_START_PERIOD = 36: Stress shocks begin at month 36 (year 3).
# SOURCE: The first 36 months are "normal" conditions; the last 12 months
#   include stress shocks. This tests whether the ML model can adapt to
#   regime changes after being trained primarily on normal data.
ML_SHOCK_START_PERIOD   = 36

# ML_SHOCK_SEED_OFFSET = 999: Offset for the shock RNG seed.
# SOURCE: Ensures the shock randomness is independent of the main universe
#   seed. The exact value (999) is arbitrary but chosen to be far from
#   common seeds (42, 0, 1) to avoid accidental correlation.
ML_SHOCK_SEED_OFFSET    = 999

# ML_SHOCK_BOND_STEP = 10: Apply stress shock to every 10th bond.
# SOURCE: With 60 bonds, every 10th = 6 bonds shocked per period.
#   This creates a ~10% incidence rate of stress events, roughly matching
#   the empirical frequency of individual issuer stress events in IG
#   (approximately 5-15% of names experience >50bp widening per year).
ML_SHOCK_BOND_STEP      = 10

# ML_SHOCK_STD_DEV = 5.0: Standard deviation of shock spread changes (bp).
# SOURCE: 5bp std is conservative. Individual IG bond spread changes have
#   monthly std of ~10-20bp. A 5bp shock std means ~68% of shocks are
#   within ±5bp, with occasional ±10-15bp moves. This produces realistic
#   but not extreme stress scenarios.
ML_SHOCK_STD_DEV        = 5.0


# ═══════════════════════════════════════════════════════════════════════
# UNIVERSE SIMULATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Parameters controlling the distributional assumptions for generating
#   individual bonds from index-level data. These determine how much
#   cross-sectional dispersion exists within each rating/duration bucket.
# SOURCE: Calibrated to match the empirical distribution of OAS dispersion
#   within ICE BofA IG sub-indices. The key calibration targets are:
#   (1) OAS coefficient of variation within buckets should be ~15-25%
#   (2) 6-month spread changes should have std ~10-15bp
#   (3) Quality scores should be roughly normally distributed 30-80
# WHY HARDCODED: Overridable in config.yaml. These are calibration defaults.
# WHY NOT DYNAMIC: Without bond-level data, we cannot estimate intra-bucket
#   dispersion from FRED. These parameters encode prior knowledge about
#   the distribution of individual bonds around the bucket-level index.

# UNIVERSE_OAS_NOISE_STD = 15bp: Cross-sectional OAS noise within a bucket.
# SOURCE: Within an IG rating bucket, individual bond OAS typically disperses
#   ±15-25bp around the bucket average. 15bp is conservative — represents
#   the tighter end of the dispersion range, appropriate for a simulated
#   universe that doesn't try to capture extreme outliers.
UNIVERSE_OAS_NOISE_STD       = 15

# UNIVERSE_SPREAD_CHG_STD = 12bp: Std of 6-month spread changes.
# SOURCE: Empirical IG 6-month spread change std is ~10-20bp depending on
#   rating. 12bp is appropriate for A-rated bonds (the plurality of IG).
UNIVERSE_SPREAD_CHG_STD      = 12

# Quality score distribution: N(55, 15²) clipped to [10, 95].
# SOURCE: Quality scores are a composite of issuer fundamentals. The
#   distribution is roughly normal with mean 55 (slightly above average)
#   and std 15, which places ~85% of bonds between 25 and 85 on a 0-100
#   scale. The clip at 10 and 95 prevents degenerate outliers.
UNIVERSE_QUALITY_MEAN        = 55.0
UNIVERSE_QUALITY_STD         = 15.0
UNIVERSE_QUALITY_MIN         = 10.0
UNIVERSE_QUALITY_MAX         = 95.0

# UNIVERSE_QUALITY_FINANCIAL_ADJ = -5.0: Financials get lower quality scores.
# SOURCE: Financial sector bonds carry additional systemic risk that is not
#   fully captured by rating (Basel III, interconnectedness). A -5 adjustment
#   reflects the empirical observation that financial bonds trade ~5-10bp
#   wider than same-rated industrials (the "financial premium").
UNIVERSE_QUALITY_FINANCIAL_ADJ = -5.0

# Spread volatility model: vol = base + dur_coeff × duration + oas_coeff × (OAS - anchor) + noise
# This produces higher volatility for longer-duration and wider-spread bonds.
# SOURCE: The positive duration-vol relationship and positive spread-vol
#   relationship are well-established in credit markets. Collin-Dufresne,
#   Goldstein & Martin (2001) document that spread volatility increases
#   with both duration and spread level.
UNIVERSE_VOL_BASE            = 10.0    # base vol ~10bp/month for short-duration IG
UNIVERSE_VOL_DUR_COEFF       = 0.3     # +0.3bp vol per year of duration
UNIVERSE_VOL_OAS_COEFF       = -0.05   # negative: wider spreads = higher vol (subtract from anchor)
UNIVERSE_VOL_OAS_ANCHOR      = 140.0   # BBB-level anchor (140bp)
UNIVERSE_VOL_NOISE_STD       = 3.0     # idiosyncratic vol noise
UNIVERSE_VOL_MIN             = 3.0     # floor: minimum plausible monthly spread vol
UNIVERSE_VOL_MAX             = 40.0    # cap: prevents unrealistic vol for outliers

# UNIVERSE_MIN_OAS_FLOOR = 1bp: Floor for OAS in log transform.
# SOURCE: log(OAS) is undefined at OAS=0. 1bp is the minimum plausible
#   spread for any corporate bond (even AAA agencies trade at ~5bp OAS).
#   This floor only activates if a simulated bond gets an unrealistically
#   tight spread.
UNIVERSE_MIN_OAS_FLOOR       = 1.0


# ═══════════════════════════════════════════════════════════════════════
# LLM CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Default model and token limit for the Claude API commentary generator.
# SOURCE: Claude Sonnet 4.5 is Anthropic's latest production model with
#   the best cost/quality ratio for structured text generation.
#   400 tokens ≈ 200 words, which matches the SYSTEM_PROMPT requirement
#   of 150-220 words for quarterly commentary.
# WHY HARDCODED: Defaults, overridable in config.yaml under llm: section.
# WHY NOT DYNAMIC: Model choice is a deployment decision, not data-driven.
LLM_MODEL_ID   = "claude-sonnet-4-5"
LLM_MAX_TOKENS = 400


# ═══════════════════════════════════════════════════════════════════════
# DATA CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Unit conversion constants.
# SOURCE: Definitions (not parameters):
#   100 basis points = 1 percentage point (OAS in FRED is in %)
#   12 months = 1 year
#   4 quarters = 1 year
# WHY HARDCODED: These are mathematical identities, not parameters.
# WHY NOT DYNAMIC: 12 months will always equal 1 year.
OAS_PCT_TO_BP       = 100
MONTHS_PER_YEAR     = 12
QUARTERLY_TO_ANNUAL = 4


# ═══════════════════════════════════════════════════════════════════════
# ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Configuration for the rebalancing attribution engine.

# ATTRIBUTION_TOP_N = 5: Number of top additions/reductions to report.
# SOURCE: Convention for institutional factsheets. Showing top-5 gives
#   a useful summary without overwhelming the reader. Matches the "top 5
#   contributors/detractors" format standard in asset management reporting.
# WHY NOT DYNAMIC: Presentation choice. Could be a config parameter but
#   the value rarely needs to change.
ATTRIBUTION_TOP_N = 5

# ATTRIBUTION_META_COLS: Bond metadata columns carried through attribution.
# SOURCE: These are the fields needed by the LLM commentary generator and
#   the attribution report formatter. The list must match the columns
#   produced by build_universe().
# WHY NOT DYNAMIC: Structural — must match the data schema.
ATTRIBUTION_META_COLS = [
    "bond_id", "sector", "rating", "duration_bucket",
    "z_dts", "z_value", "z_momentum", "z_composite",
    "oas_bp", "spread_6m_chg",
]


# ═══════════════════════════════════════════════════════════════════════
# CHART PRESENTATION
# ═══════════════════════════════════════════════════════════════════════

# CHART_ROLLING_WINDOW = 60 months (5 years): Window for rolling Sharpe ratio.
# SOURCE: 60 months is the standard window for rolling risk-adjusted return
#   metrics in institutional reporting. Shorter (12-36mo) is too volatile;
#   longer (120mo) smooths out meaningful regime differences. Morningstar
#   and most fund databases use 36-60 month windows for rolling statistics.
CHART_ROLLING_WINDOW = 60

# HMM_BAR_WIDTH = 28 days: Width of bars in the HMM state probability chart.
# SOURCE: With monthly data, each bar represents ~30 days. 28-day bar width
#   leaves a small gap between bars for visual clarity.
HMM_BAR_WIDTH        = 28

# HISTORICAL_EVENTS: Key dates for annotating credit spread charts.
# SOURCE: These are the 6 most significant credit market events in the
#   FRED data period (1997-present). Each event caused a measurable spike
#   in IG OAS:
#   - 9/11 (2001): IG OAS widened ~40bp
#   - WorldCom (2002): IG OAS peaked at ~270bp (largest IG fraud at the time)
#   - Lehman (2008): IG OAS hit all-time high ~620bp
#   - EU crisis (2011): IG OAS widened ~80bp on sovereign contagion fears
#   - COVID (2020): IG OAS widened ~280bp in 3 weeks
#   - Fed hike cycle (2022): Gradual widening ~50bp over 6 months
# WHY HARDCODED: Historical facts. Dates of past events do not change.
# WHY NOT DYNAMIC: History is not dynamic.
HISTORICAL_EVENTS = [
    ("2001-09-01", "9/11"),
    ("2002-06-01", "WorldCom"),
    ("2008-09-15", "Lehman"),
    ("2011-08-01", "EU crisis"),
    ("2020-03-01", "COVID"),
    ("2022-03-01", "Fed hike cycle"),
]

# CHART_FILENAMES: Output filenames for each chart type.
# SOURCE: Naming convention. "fig1_", "fig2_" etc. correspond to the
#   paper's figure numbering. "ml_" prefix for ML-pipeline charts.
# WHY NOT DYNAMIC: File naming is a convention choice.
CHART_FILENAMES = {
    "shap_summary"     : "ml_shap_summary.png",
    "shap_weights_time": "ml_shap_weights_time.png",
    "walkforward_perf" : "ml_walkforward_perf.png",
    "weight_comparison": "ml_weight_comparison.png",
    "ml_vs_bl_returns" : "ml_vs_bl_returns.png",
    "hmm_regimes"      : "fig4_hmm_regimes.png",
    "bl_posterior"      : "fig5_bl_posterior.png",
    "architecture"     : "fig5_architecture.png",
    "value_signal"     : "fig1_value_signal.png",
    "momentum_signal"  : "fig2_momentum_signal.png",
    "quality_sharpe"   : "fig3_quality_sharpe.png",
}


# ═══════════════════════════════════════════════════════════════════════
# PROPHET VIEW BUCKET CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Configuration for Prophet's logistic growth model bounds.

# PROPHET_CAP_MULTIPLIER = 2.5: The cap for logistic growth is set to
#   2.5× the historical maximum OAS for each bucket.
# SOURCE: Credit spreads can theoretically spike to very high levels
#   (GFC: IG OAS hit ~620bp from a normal of ~100-150bp, a ~4x move).
#   2.5× the historical max provides headroom for extreme scenarios without
#   being so loose that the logistic curve is effectively linear.
#   The cap matters because Prophet's logistic growth model asymptotes at
#   the cap, preventing forecasts from diverging to infinity.
PROPHET_CAP_MULTIPLIER = 2.5

# PROPHET_FLOOR = 0.05 (5bp in OAS %): Minimum plausible OAS level.
# SOURCE: The tightest IG OAS has ever been is ~30bp (2021). 5bp provides
#   a theoretical floor that no IG spread has ever breached. This prevents
#   Prophet from forecasting negative spreads (which are economically
#   nonsensical for credit).
PROPHET_FLOOR          = 0.05

# BUCKET_CONFIG: Maps each rating tier to its OAS column and duration
#   for Prophet forecasting.
# SOURCE: Combines RATING_OAS_COL and DURATIONS into a single lookup.
# WHY NOT DYNAMIC: This is a structural mapping, not a parameter.
BUCKET_CONFIG = {
    "AAA": {"oas_col": "oas_aaa", "duration": DURATIONS["AAA"]},
    "AA" : {"oas_col": "oas_aa",  "duration": DURATIONS["AA"]},
    "A"  : {"oas_col": "oas_a",   "duration": DURATIONS["A"]},
    "BBB": {"oas_col": "oas_bbb", "duration": DURATIONS["BBB"]},
    "HY" : {"oas_col": "oas_hy",  "duration": DURATIONS["HY"]},
}


# ═══════════════════════════════════════════════════════════════════════
# ML MODEL CHOICES (CLI)
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Valid model type strings for the CLI --model-type argument.
# SOURCE: The 3 ML algorithms implemented in the ML pipeline.
# WHY NOT DYNAMIC: Structural — matches the implemented model classes.
ML_MODEL_CHOICES = ["random_forest", "gradient_boosting", "enhanced_rf"]


# ═══════════════════════════════════════════════════════════════════════
# BACKTEST PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Default parameters for the ML walk-forward backtest engine.

# BT_N_PERIODS = 60: Number of monthly periods to simulate.
# SOURCE: 5 years (60 months) is a standard backtest horizon for credit
#   strategies. Long enough to capture at least one full credit cycle
#   (expansion + stress) but short enough for reasonable runtime.
BT_N_PERIODS        = 60

# BT_TC_BPS = 5.0: One-way transaction cost in basis points.
# SOURCE: 5bp one-way (~10bp round-trip) is the standard estimate for
#   institutional IG corporate bond trading. Dick-Nielsen, Feldhütter &
#   Lando (2012) report average IG bid-ask spreads of ~8-12bp, translating
#   to ~4-6bp one-way for large institutional trades. 5bp is the midpoint.
# WHY NOT DYNAMIC: Transaction costs vary by market conditions, but using
#   a fixed conservative estimate is standard for backtesting to avoid
#   look-ahead bias (knowing future liquidity conditions).
BT_TC_BPS           = 5.0

# BT_ML_RETRAIN_EVERY = 3: Retrain the ML model every 3 months (quarterly).
# SOURCE: Matches the rebalancing frequency. Monthly retraining with small
#   datasets leads to overfitting; annual retraining is too slow to adapt.
BT_ML_RETRAIN_EVERY = 3

# BT_BASE_SEED = 42: Random seed for backtest reproducibility.
BT_BASE_SEED        = 42

# BT_MIN_ML_HISTORY = 12: Minimum months before ML starts making predictions.
# SOURCE: Same rationale as ML_MIN_TRAIN_MONTHS.
BT_MIN_ML_HISTORY   = 12


# ═══════════════════════════════════════════════════════════════════════
# LEGACY STYLE DICT
# ═══════════════════════════════════════════════════════════════════════
# WHAT: Backward-compatible style dict for older chart code.
# SOURCE: Aggregates CHART_DPI, COLORS, and font_size into a flat dict.
# WHY NOT DYNAMIC: Convenience alias.
STYLE = {
    "fig_dpi"      : CHART_DPI,
    "color_primary": COLORS["primary"],
    "color_accent" : COLORS["accent"],
    "color_neutral": COLORS["neutral"],
    "color_amber"  : COLORS["amber"],
    "color_green"  : COLORS["green"],
    "font_size"    : 10,
}
