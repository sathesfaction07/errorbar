# CAP-M7 · `evaluation-platform`

One evaluation system where **every number carries its uncertainty and its
provenance**, and where "promoted on a gain inside noise" is not a policy violation
but a type error.

**Read [`FINDINGS.md`](FINDINGS.md) for the measured numbers**, the four mandated
crosschecks, the eight defects found in this code while building it, and — §7 — an
explicit list of what is **not built yet**. The composition seam
(`gates/promotion.py`) now exists; S7/S8, the version-scoped leaderboard, and any
real judge model or pinned dataset do not. This capstone is not finished.

## The teaching core

```python
Estimate(point=0.82, interval=..., provenance=..., metric=...)
```

Four things are impossible by construction:

| Attempt | Result |
|---|---|
| `Estimate(0.82, None, prov, metric)` | refused — a point estimate is not a result |
| `Estimate(0.82, iv, None, metric)` | refused — a number of unknown origin cannot be compared |
| `Metric("accuracy")` | refused — `higher_is_better` has no safe default |
| `estimate_a > estimate_b` | refused — use `compare()`, which can say INDISTINGUISHABLE |

`compare()` returns **four** states. `INDISTINGUISHABLE` and `INCOMPARABLE` are kept
apart deliberately: the first is a completed measurement, the second is a *missing*
one and a pipeline defect.

`PromotionDecision` carries the same discipline one level up: `promote=True` beside
any blocking gate raises, an empty gate list raises, and there is no override
parameter — because an override is how every fail-closed gate becomes advisory.

## The one statistical subtlety worth knowing

`compare()` never checks whether the two intervals overlap. Overlap is the intuitive
test and it is **wrong in both directions** — two 95% intervals can overlap while
their difference is significant, and paired data can show near-identical marginals
with a large, tight difference. So it requires an interval on the **difference**,
computed paired where possible, and records which method was used.

Measured on the S1b fixture: the marginals overlap, and the paired difference CI is
**26× tighter** and excludes zero. The overlap test would have blocked a real
improvement.

## Run it

Everything runs locally. No cloud, no API key, no network after setup.

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy statsmodels choix krippendorff scikit-learn pytest

.venv/bin/python -m pytest tests/ -q        # 83 tests
.venv/bin/python run_capstone.py; echo $?   # measured numbers + 3-state exit
```

### Exit codes are three-state

```
0  every REQUIRED claim was conclusive and passed
1  a conclusive FAIL      -- we measured, and it is bad
2  a REQUIRED claim was never verified -- we did not measure
```

All three are verified end-to-end, not just unit-tested: injecting a `FAIL` yields 1
and injecting a `COULD_NOT_RUN` yields 2. In the exit-2 run the report reads
`pass_rate: 100% of 19 conclusive (21 claims total)` — it *looks* perfect, and the
exit code is still 2 with the unverified claim named. That asymmetry is the point.

> Use `run_capstone.py; echo $?` — **not** `run_capstone.py | tail; echo $?`. A pipe
> returns the last command's status and reports 0 over a real 2.

## The four mandated crosschecks are real, not skipped

Installed into `.venv` before any code was written, so none could quietly become
`COULD_NOT_RUN`:

| Quantity | Reference | Agreement |
|---|---|---|
| BH q-values | `statsmodels` `fdr_bh` | 1.11e-16 |
| Krippendorff α | `krippendorff` | **0.00e+00** (nominal, interval, ragged) |
| Bradley-Terry θ | `choix.ilsr_pairwise` | 1.65e-09 |
| Bootstrap CI | `scipy.stats.bootstrap` | within 1% of interval width |

Plus `scipy.stats.false_discovery_control` and `sklearn.metrics.cohen_kappa_score` as
independent second opinions.

## Layout

```
evalplat/
  core.py                4-state Verdict/Result/Report; exit-code logic
  numbers/estimate.py    THE TEACHING CORE -- Estimate, Metric, Provenance, compare()
  numbers/ci.py          analytic-t, Wilson, bootstrap; the naive CI as counter-example
  audit/provenance.py    canonical hashing, EvalSet, JudgeBundle
  stats/multiplicity.py  Benjamini-Hochberg, Bonferroni, unadjusted counter-example
  judges/agreement.py    Krippendorff alpha, Cohen kappa, the agreement floor
  preference/bt.py       Bradley-Terry with unbounded-MLE detection
  gates/null_judge.py    the signature test: fail-closed scoring
  gates/slice.py         disaggregated slice gate; BH-adjusted within the family
  gates/promotion.py     THE SEAM -- measurements become one four-state decision
tests/test_evalplat.py   83 tests: refusals, reverse tests, crosschecks, worst input
run_capstone.py          scenario runner; 3-state exit; writes findings.json
```

## Why the package is nested under `evalplat/`

**Not a style choice.** The spec's layout puts `numbers/` at the top level, and
`numbers` is a **Python standard library module**. Measured: adding an ordinary
`numbers/__init__.py` breaks `fractions`, `decimal` and `statistics` with
`AttributeError: module 'numbers' has no attribute 'Number'`. Worse, the trap is
latent — without `__init__.py` the directory is a namespace package and loses
precedence, so everything works until someone adds one. `kb-capstones.md` has been
corrected; details in `FINDINGS.md` §4.1.

## Counter-examples are retained on purpose

`_normal_approx_ci_DO_NOT_USE`, `unadjusted_DO_NOT_USE`, `win_rate_ranking`,
`naive_score_eval_set_DO_NOT_USE`, `aggregate_only_gate_DO_NOT_USE`,
`naive_promotion_DO_NOT_USE`, `percent_agreement`. Each is what real eval code
contains — plausible, short, and wrong. Each has a **necessity pre-check** proving it
actually breaks the clean case (`FINDINGS.md` §5), because a planted defect that
passes anyway is not a fixture.
