# Tuning Notes

## Threshold Guide
| Score | Meaning |
|-------|---------|
| < 0.20 | Strong genuine match |
| 0.20 – 0.38 | Genuine match — accepted |
| > 0.38 | Rejected |
| > 0.45 | Clearly different person |

## Key Constants (search_engine.py)
- L1_THRESHOLD = 0.25  — Layer 1 signature pre-filter distance
- TOP_K = 80           — max candidates passed to Layer 2
- Fallback = top-10    — used when L1 passes nothing (small DB)
- Aggregation: 0.7 * min + 0.3 * mean per user

## Enrollment Quality Targets
- Self-match (same user, different samples): all pairs < 0.35
- Cross-match (different users): all pairs > 0.45

## CLAHE Parameters
- clipLimit = 2.5       — less noise amplification than default 3.5
- tileGridSize = 16x16  — matches 32x32 Gabor block spatial scale
