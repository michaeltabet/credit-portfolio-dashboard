# Factor Credit Optimizer + LLM Commentary Engine

## What this is

A working implementation of the research paper:
**"Factor Investing in Corporate Credit: Structural Challenges, Methodology, and an LLM-Powered Commentary Engine"**

The optimizer builds a factor-tilted investment-grade credit portfolio.
The commentary engine uses Claude to generate quarterly client-facing
commentary directly from the attribution output.

## Files

| File | Purpose |
|------|---------|
| `universe.py` | Synthetic IG bond universe with factor scoring |
| `optimizer.py` | CVXPY factor-tilted portfolio optimizer |
| `attribution.py` | Rebalancing attribution: what changed and why |
| `commentary.py` | LLM commentary generator (Claude API or mock) |
| `fred_data.py` | ICE BofA empirical charts from FRED |
| `main.py` | End-to-end pipeline runner |

## Run

```bash
pip install -r requirements.txt
python main.py
```

For live Claude commentary:
```bash
export ANTHROPIC_API_KEY=your_key_here
python main.py
```

## Design note on data governance

The LLM receives only derived attribution summaries (factor Z-score deltas,
sector weight changes). No raw holdings, position sizes, or confidential
issuer data are transmitted. For production in a regulated environment,
route through an enterprise API endpoint or on-premises LLM deployment.
