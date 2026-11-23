"""Test suite for CAP-M7.

Structure follows the standing bar rather than module boundaries:

  1. REFUSALS       the type system rejects what it must
  2. REVERSE TESTS  every gate can also PASS -- a gate that cannot pass is not a gate
  3. CROSSCHECKS    four mandated external references, plus two extra
  4. PLANTED        each defect FAILS, and the necessity pre-check shows it bites
  5. EXIT CODES     asserted numerically, not via report text
  6. WORST INPUT    empty, degenerate, non-finite, adversarial
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalplat.audit.provenance import (  # noqa: E402
    EvalSet,
    JudgeBundle,
    ProvenanceError,
    canonical_hash,
)
from evalplat.core import Report, Result, Verdict  # noqa: E402
from evalplat.gates.null_judge import (  # noqa: E402
    FlakyJudge,
    UnreachableJudge,
    WorkingJudge,
    naive_score_eval_set_DO_NOT_USE,
    null_judge_gate,
    score_eval_set,
)
from evalplat.gates.promotion import (  # noqa: E402
    PromotionDecision,
    PromotionError,
    naive_promotion_DO_NOT_USE,
    promotion_gate,
)
from evalplat.gates.slice import (  # noqa: E402
    SliceError,
    SliceSpec,
    aggregate_only_gate_DO_NOT_USE,
    slice_gate,
)
from evalplat.judges.agreement import (  # noqa: E402
    MeasurementLevel,
    cohen_kappa,
    krippendorff_alpha,
    percent_agreement,
)
from evalplat.numbers.ci import (  # noqa: E402
    _normal_approx_ci_DO_NOT_USE,
    analytic_mean_ci,
    bootstrap_ci,
    estimate_mean,
    estimate_proportion,
    wilson_ci,
)
from evalplat.numbers.estimate import (  # noqa: E402
    EXACT_JUDGE,
    ComparisonVerdict,
    Estimate,
    EstimateError,
    Interval,
    Metric,
    Provenance,
    compare,
)
from evalplat.preference.bt import (  # noqa: E402
    bradley_terry,
    rank_displacement,
    win_rate_ranking,
)
from evalplat.stats.multiplicity import (  # noqa: E402
    benjamini_hochberg,
    bonferroni,
    paired_pvalue,
    unadjusted_DO_NOT_USE,
)

ACC = Metric("accuracy", higher_is_better=True)
LAT = Metric("p95_latency", higher_is_better=False, unit="ms")
PROV = Provenance("v1", "judge_v1", "set_a")
IV = Interval(0.76, 0.84, 0.95, "bootstrap", 200)

BUNDLE = JudgeBundle("llama3:8b", "Rate 0-1.", "1=correct", 0.0, "parser-v2")
ITEMS = tuple({"id": f"i{k}", "gold": 0.8} for k in range(50))
EVALSET = EvalSet("smoke", ITEMS, "v1")


# ---------------------------------------------------------------- 1. REFUSALS

@pytest.mark.parametrize(
    "make, why",
    [
        (lambda: Estimate(0.8, None, PROV, ACC), "no interval"),
        (lambda: Estimate(0.8, IV, None, ACC), "no provenance"),
        (lambda: Estimate(0.8, IV, PROV, "accuracy"), "metric as bare string"),
        (lambda: Estimate(0.99, IV, PROV, ACC), "point outside interval"),
        (lambda: Estimate(float("nan"), IV, PROV, ACC), "nan point"),
        (lambda: Estimate(True, IV, PROV, ACC), "bool point"),
        (lambda: Metric("acc", higher_is_better=None), "direction missing"),
        (lambda: Metric("", higher_is_better=True), "empty metric name"),
        (lambda: Interval(0.84, 0.76, 0.95, "b", 10), "inverted interval"),
        (lambda: Interval(0.1, 0.2, 95, "b", 10), "level as 95 not 0.95"),
        (lambda: Interval(0.1, 0.2, 0.95, "b", 0), "n=0"),
        (lambda: Interval(float("inf"), 0.2, 0.95, "b", 10), "infinite bound"),
        (lambda: Interval(0.1, 0.2, 0.95, "", 10), "no method recorded"),
        (lambda: Provenance("", "j", "s"), "blank system_version"),
        (lambda: Provenance("v1", "", "s"), "blank judge hash"),
        (lambda: Estimate(0.8, IV, PROV, ACC, per_item=(0.1, 0.2)), "per_item length != n"),
    ],
)
def test_refusals(make, why):
    with pytest.raises(EstimateError):
        make()


def test_ordering_operators_all_refuse():
    a = Estimate(0.80, IV, PROV, ACC)
    b = Estimate(
        0.82, Interval(0.78, 0.86, 0.95, "b", 200), Provenance("v2", "judge_v1", "set_a"), ACC
    )
    for op in ("__gt__", "__lt__", "__ge__", "__le__"):
        with pytest.raises(EstimateError):
            getattr(a, op)(b)


def test_render_always_includes_interval():
    e = Estimate(0.8, IV, PROV, ACC)
    out = e.render()
    assert "0.76" in out and "0.84" in out and "95%" in out and "n=200" in out


# ------------------------------------------------- 2. COMPARISON: FOUR STATES

def _est(point, lo, hi, version, metric=ACC, judge="judge_v1", eset="set_a", per=None, n=200):
    return Estimate(point, Interval(lo, hi, 0.95, "analytic-t", n if per is None else len(per)),
                    Provenance(version, judge, eset), metric, per)


def test_gain_inside_noise_is_indistinguishable():
    c = compare(_est(0.80, 0.76, 0.84, "v1"), _est(0.82, 0.78, 0.86, "v2"))
    assert c.verdict is ComparisonVerdict.INDISTINGUISHABLE
    assert not c.verdict.permits_promotion


def test_judge_change_is_incomparable_not_a_regression():
    c = compare(_est(0.80, 0.76, 0.84, "v1"), _est(0.82, 0.78, 0.86, "v2", judge="judge_v2"))
    assert c.verdict is ComparisonVerdict.INCOMPARABLE
    assert "judge bundle differs" in c.reason


def test_eval_set_change_is_incomparable():
    c = compare(_est(0.80, 0.76, 0.84, "v1"), _est(0.82, 0.78, 0.86, "v2", eset="set_b"))
    assert c.verdict is ComparisonVerdict.INCOMPARABLE


def test_lower_is_better_metric_gets_the_sign_right():
    """REVERSE-SIGN TEST. A latency DECREASE must read BETTER, and an increase WORSE."""
    high = (420, 400, 440)
    low = (300, 290, 310)
    fast = compare(_est(*high, "v1", LAT, EXACT_JUDGE), _est(*low, "v2", LAT, EXACT_JUDGE))
    slow = compare(_est(*low, "v1", LAT, EXACT_JUDGE), _est(*high, "v2", LAT, EXACT_JUDGE))
    assert fast.verdict is ComparisonVerdict.BETTER
    assert slow.verdict is ComparisonVerdict.WORSE


def test_comparison_can_return_better_reverse_test():
    """A comparator that can only ever say INDISTINGUISHABLE is useless."""
    c = compare(_est(0.50, 0.46, 0.54, "v1"), _est(0.90, 0.86, 0.94, "v2"))
    assert c.verdict is ComparisonVerdict.BETTER and c.verdict.permits_promotion


def test_paired_is_more_powerful_than_overlap_test():
    """The headline statistical claim: marginal intervals can OVERLAP while the
    paired difference decisively excludes zero."""
    rng = np.random.default_rng(7)
    base = rng.normal(0.80, 0.15, 400).clip(0, 1)
    cand = (base + 0.02).clip(0, 1)
    eb = estimate_mean(base, PROV, ACC)
    ec = estimate_mean(cand, Provenance("v2", "judge_v1", "set_a"), ACC)
    assert eb.interval.overlaps(ec.interval), "fixture invalid: intervals must overlap"
    c = compare(eb, ec)
    assert c.verdict is ComparisonVerdict.BETTER
    assert c.diff_interval is not None and c.diff_interval.excludes_zero()
    assert c.diff_interval.width < eb.interval.width, "paired must be tighter"


# ------------------------------------------------------------- 3. CROSSCHECKS

def test_crosscheck_analytic_ci_vs_scipy():
    from scipy import stats as st
    x = np.random.default_rng(42).normal(0.8, 0.12, 60)
    mine = analytic_mean_ci(x, 0.95)
    lo, hi = st.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=st.sem(x))
    assert abs(mine.lo - lo) < 1e-12 and abs(mine.hi - hi) < 1e-12


def test_crosscheck_bootstrap_vs_scipy():
    from scipy import stats as st
    x = np.random.default_rng(42).normal(0.8, 0.12, 60)
    mine = bootstrap_ci(x, level=0.95, n_boot=10000, seed=7)
    ref = st.bootstrap((x,), np.mean, confidence_level=0.95, n_resamples=10000,
                       method="percentile", random_state=7)
    assert abs(mine.lo - ref.confidence_interval.low) < 0.01 * mine.width
    assert abs(mine.hi - ref.confidence_interval.high) < 0.01 * mine.width


def test_crosscheck_bh_vs_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = np.concatenate([np.random.default_rng(11).uniform(0, 0.02, 6),
                        np.random.default_rng(12).uniform(0, 1, 90)])
    mine = benjamini_hochberg(p, 0.05)
    rej, q, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    assert np.max(np.abs(np.array(mine.qvalues) - q)) < 1e-12
    assert np.array_equal(np.array(mine.rejected), rej)


def test_crosscheck_bh_vs_scipy():
    from scipy.stats import false_discovery_control
    p = np.random.default_rng(13).uniform(0, 1, 50)
    mine = benjamini_hochberg(p, 0.05)
    assert np.max(np.abs(np.array(mine.qvalues) - false_discovery_control(p, method="bh"))) < 1e-12


@pytest.mark.parametrize("level,name", [(MeasurementLevel.NOMINAL, "nominal"),
                                        (MeasurementLevel.INTERVAL, "interval")])
def test_crosscheck_alpha_vs_krippendorff_package(level, name):
    import krippendorff as kd
    data = [[1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3]]
    mine = krippendorff_alpha(data, level)
    ref = kd.alpha(reliability_data=[[np.nan if v is None else v for v in r] for r in data],
                   level_of_measurement=name)
    assert abs(mine.value - ref) < 1e-12


def test_crosscheck_alpha_ragged_three_raters():
    import krippendorff as kd
    rag = [[1, 1, None, 2, 2, None, 3, 3, 1, None, 2, 2],
           [1, 2, 2, 2, None, 1, 3, 3, None, 1, 2, None],
           [None, 1, 2, None, 2, 1, None, 3, 1, 1, None, 2]]
    mine = krippendorff_alpha(rag, MeasurementLevel.NOMINAL)
    ref = kd.alpha(reliability_data=[[np.nan if v is None else v for v in r] for r in rag],
                   level_of_measurement="nominal")
    assert abs(mine.value - ref) < 1e-12


def test_crosscheck_kappa_vs_sklearn():
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(3)
    h = rng.integers(0, 2, 200)
    j = np.where(rng.random(200) < 0.85, h, 1 - h)
    assert abs(cohen_kappa(h, j).value - cohen_kappa_score(h, j)) < 1e-12


def test_crosscheck_bt_vs_choix():
    import choix
    items = ["A", "B", "C", "D", "E"]
    true_t = {"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0, "E": -2.0}
    rng = np.random.default_rng(5)
    games = []
    for a in items:
        for b in items:
            if a < b:
                p = 1 / (1 + math.exp(-(true_t[a] - true_t[b])))
                games += [(a, b) if rng.random() < p else (b, a) for _ in range(40)]
    mine = bradley_terry(games, items, n_boot=0)
    idx = {it: k for k, it in enumerate(items)}
    ref = choix.ilsr_pairwise(
        len(items), [(idx[winner], idx[loser]) for winner, loser in games], alpha=1e-9
    )
    ref = ref - ref.mean()
    assert np.max(np.abs(np.array(mine.theta) - ref)) < 1e-6
    assert mine.ranking == ("A", "B", "C", "D", "E")


# ------------------------------------- 4. PLANTED DEFECTS + NECESSITY (rule 11)

def test_planted_normal_approx_produces_impossible_bounds():
    lo, hi = _normal_approx_ci_DO_NOT_USE(49, 50)
    assert hi > 1.0, "necessity: the defect must actually produce an impossible bound"
    assert wilson_ci(49, 50).hi <= 1.0


def test_planted_normal_approx_zero_width_at_boundary():
    lo, hi = _normal_approx_ci_DO_NOT_USE(50, 50)
    assert hi == lo == 1.0, "necessity: defect must claim perfect certainty"
    w = wilson_ci(50, 50)
    assert w.width > 0.05, "wilson must retain honest uncertainty at p=1"


def test_planted_unadjusted_multiplicity_manufactures_winners():
    """NECESSITY PRE-CHECK: 8 models x 12 tasks, all nulls true."""
    m, trials = 96, 400
    un = bh = 0
    for t in range(trials):
        p = np.random.default_rng(1000 + t).uniform(0, 1, m)
        un += unadjusted_DO_NOT_USE(p, 0.05).n_rejected
        bh += benjamini_hochberg(p, 0.05).n_rejected
    assert un / trials > 3.0, f"necessity failed: only {un/trials:.2f} false discoveries"
    assert bh / trials < 0.5
    assert bonferroni(np.random.default_rng(1).uniform(0, 1, m), 0.05).n_rejected == 0


def test_planted_win_rate_misranks_under_unbalanced_pairings():
    true_t = {"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0, "E": -2.0}
    items, truth = list(true_t), ("A", "B", "C", "D", "E")
    rng = np.random.default_rng(1)
    sched = {("A", "B"): 60, ("B", "C"): 40, ("C", "D"): 20,
             ("D", "E"): 60, ("C", "E"): 20, ("B", "E"): 10}
    games = []
    for (a, b), n in sched.items():
        p = 1 / (1 + math.exp(-(true_t[a] - true_t[b])))
        games += [(a, b) if rng.random() < p else (b, a) for _ in range(n)]
    wr = win_rate_ranking(games, items)
    wr_rank = tuple(sorted(items, key=lambda i: (-wr[i], i)))
    bt = bradley_terry(games, items, n_boot=0)
    assert rank_displacement(truth, wr_rank) > 0, "necessity: win-rate must misrank here"
    assert rank_displacement(truth, bt.ranking) == 0


def test_planted_base_rate_makes_raw_agreement_meaningless():
    humans = np.array([1] * 170 + [0] * 30)
    lazy = np.ones(200, dtype=int)
    assert percent_agreement(humans, lazy) == pytest.approx(0.85)
    a = krippendorff_alpha([humans.tolist(), lazy.tolist()], MeasurementLevel.NOMINAL)
    assert a.value < 0.05, "necessity: chance-corrected must expose the lazy judge"
    assert krippendorff_alpha([humans.tolist(), lazy.tolist()],
                              MeasurementLevel.NOMINAL, floor=0.667).verdict is Verdict.FAIL


def test_planted_naive_scorer_hides_a_dead_judge():
    """NECESSITY: the naive version really does report success from total failure."""
    scores = naive_score_eval_set_DO_NOT_USE(UnreachableJudge(BUNDLE), EVALSET)
    assert scores == [], "necessity: naive path must silently produce an empty list"
    assert len(scores) == 0  # ...which a caller reads as '0 failures'
    gate = null_judge_gate(UnreachableJudge(BUNDLE), EVALSET)
    assert gate.verdict is Verdict.COULD_NOT_RUN
    assert gate.verdict.blocks_promotion


def test_planted_judge_upgrade_changes_the_hash():
    one_char = BUNDLE.with_changes(prompt=BUNDLE.prompt + " ")
    assert one_char.hash != BUNDLE.hash
    for field, value in [("model_id", "llama3:70b"), ("rubric", "1=good"),
                         ("temperature", 0.1), ("parser_version", "parser-v3")]:
        assert BUNDLE.with_changes(**{field: value}).hash != BUNDLE.hash, field


# ----------------------------------------------------- 5. GATES: REVERSE TESTS

def test_null_judge_gate_can_pass():
    """REVERSE TEST -- a gate that cannot pass is not a gate, it is a blocker."""
    g = null_judge_gate(WorkingJudge(BUNDLE), EVALSET)
    assert g.verdict is Verdict.PASS and not g.verdict.blocks_promotion


def test_null_judge_gate_catches_partial_outage():
    g = null_judge_gate(FlakyJudge(BUNDLE, fail_every=5), EVALSET, coverage_floor=0.98)
    assert g.verdict is Verdict.COULD_NOT_RUN
    assert g.detail["coverage"] == pytest.approx(0.8)


def test_null_judge_gate_tolerates_coverage_above_floor():
    g = null_judge_gate(FlakyJudge(BUNDLE, fail_every=100), EVALSET, coverage_floor=0.95)
    assert g.verdict is Verdict.PASS


def test_non_numeric_judge_output_is_not_a_score():
    class Garbage(WorkingJudge):
        def score(self, item):
            return "excellent"  # type: ignore[return-value]

    out = score_eval_set(Garbage(BUNDLE), EVALSET)
    assert out.scored_ids == () and len(out.failed_ids) == len(EVALSET)
    assert null_judge_gate(Garbage(BUNDLE), EVALSET).verdict is Verdict.COULD_NOT_RUN


# ------------------------------------------------------------- 6. EXIT CODES

def test_exit_code_zero_only_when_conclusive_and_passing():
    r = Report()
    r.require("a")
    r.add(Result("a", Verdict.PASS, "ok"))
    assert r.exit_code() == 0


def test_exit_code_one_on_real_failure():
    r = Report()
    r.require("a")
    r.add(Result("a", Verdict.FAIL, "measured and bad"))
    assert r.exit_code() == 1


def test_exit_code_two_when_required_claim_unverified():
    """THE SIGNATURE ASSERTION -- COULD_NOT_RUN must not exit 0."""
    r = Report()
    r.require("a")
    r.add(Result("a", Verdict.COULD_NOT_RUN, "judge unreachable"))
    assert r.exit_code() == 2


def test_exit_code_two_when_required_claim_never_attempted():
    r = Report()
    r.require("never_ran")
    r.add(Result("other", Verdict.PASS, "ok"))
    assert r.exit_code() == 2
    assert [x.claim for x in r.unverified_required] == ["never_ran"]


def test_exit_code_two_on_empty_report():
    assert Report().exit_code() == 2


def test_pass_rate_denominator_excludes_non_conclusive():
    r = Report()
    r.add(Result("a", Verdict.PASS, "ok"))
    r.add(Result("b", Verdict.COULD_NOT_RUN, "nope"))
    r.add(Result("c", Verdict.SKIPPED, "n/a"))
    rate, n = r.pass_rate()
    assert (rate, n) == (1.0, 1)
    assert "of 1 conclusive (3 claims total)" in r.render()


# -------------------------------------------------------------- 7. WORST INPUT

@pytest.mark.parametrize("bad", [
    lambda: analytic_mean_ci([0.5]),
    lambda: analytic_mean_ci([]),
    lambda: analytic_mean_ci([0.5, float("nan")]),
    lambda: wilson_ci(51, 50),
    lambda: wilson_ci(-1, 50),
    lambda: wilson_ci(5, 1),
    lambda: wilson_ci(True, 50),
])
def test_worst_input_refused(bad):
    with pytest.raises(EstimateError):
        bad()


def test_zero_variance_interval_is_labelled_degenerate():
    iv = analytic_mean_ci([0.8] * 20)
    assert iv.width == 0.0 and "degenerate" in iv.method


def test_empty_pvalue_family_is_refused_not_treated_as_clean():
    from evalplat.stats.multiplicity import MultiplicityError
    with pytest.raises(MultiplicityError):
        benjamini_hochberg([])


def test_provenance_refuses_unstable_types():
    with pytest.raises(ProvenanceError):
        canonical_hash({"judge": object()})
    with pytest.raises(ProvenanceError):
        canonical_hash({"t": float("nan")})
    with pytest.raises(ProvenanceError):
        canonical_hash({1: "int key"})


def test_canonical_hash_is_order_independent_and_stable():
    a = canonical_hash({"x": 1, "y": [1, 2], "z": {"p": 0.5}}, prefix="j")
    b = canonical_hash({"z": {"p": 0.5}, "y": [1, 2], "x": 1}, prefix="j")
    assert a == b
    assert canonical_hash({"x": 1}, prefix="judge") != canonical_hash({"x": 1}, prefix="evalset")


def test_eval_set_refuses_empty_and_duplicate_ids():
    with pytest.raises(ProvenanceError):
        EvalSet("e", (), "v1")
    with pytest.raises(ProvenanceError):
        EvalSet("e", ({"id": "a"}, {"id": "a"}), "v1")
    with pytest.raises(ProvenanceError):
        EvalSet("e", ({"no_id": 1},), "v1")


def test_bt_refuses_or_flags_degenerate_arenas():
    from evalplat.preference.bt import PreferenceError
    with pytest.raises(PreferenceError):
        bradley_terry([])
    with pytest.raises(PreferenceError):
        bradley_terry([("A", "A")], ["A", "B"])
    assert bradley_terry([("A", "B")] * 5 + [("B", "C")] * 5 + [("C", "B")] * 3,
                         ["A", "B", "C"], n_boot=0).verdict is Verdict.COULD_NOT_RUN
    assert bradley_terry([("A", "B")] * 5 + [("C", "D")] * 5,
                         ["A", "B", "C", "D"], n_boot=0).verdict is Verdict.COULD_NOT_RUN


def test_paired_pvalue_refuses_unpaired_and_incomparable():
    a = _est(0.8, 0.76, 0.84, "v1")
    b = _est(0.82, 0.78, 0.86, "v2")
    with pytest.raises(EstimateError):
        paired_pvalue(a, b)
    pa = estimate_mean([0.7, 0.8, 0.9, 0.85], PROV, ACC)
    pb = estimate_mean([0.75, 0.85, 0.95, 0.9], Provenance("v2", "judge_v2", "set_a"), ACC)
    with pytest.raises(EstimateError):
        paired_pvalue(pa, pb)


def test_proportion_estimate_at_boundaries():
    for k in (0, 25, 50):
        e = estimate_proportion(k, 50, Provenance("v1", EXACT_JUDGE, "s"), ACC)
        assert 0.0 <= e.interval.lo <= e.point <= e.interval.hi <= 1.0
        assert e.interval.width > 0


# ------------------------------------------- 8. SLICE + PROMOTION (the seam, S9)

def _sliced_fixture(seed=11, n=500, n_minority=50, en_delta=0.03, de_delta=-0.15):
    """Aggregate improves while a minority slice is harmed.

    The per-item deltas carry noise on purpose. An earlier version applied a constant
    shift, which made every paired difference identical and produced a ZERO-WIDTH
    difference interval -- arithmetically correct but an unrealistic fixture that
    would have masked any bug in the interval arithmetic.
    """
    rng = np.random.default_rng(seed)
    lang = np.array(["en"] * (n - n_minority) + ["de"] * n_minority)
    base = rng.normal(0.80, 0.10, n).clip(0, 1)
    cand = base.copy()
    jitter = rng.normal(0, 0.02, n)
    cand[lang == "en"] += en_delta + jitter[lang == "en"]
    cand[lang == "de"] += de_delta + jitter[lang == "de"]
    cand = cand.clip(0, 1)
    eb = estimate_mean(base, PROV, ACC)
    ec = estimate_mean(cand, Provenance("v2", "judge_v1", "set_a"), ACC)
    return eb, ec, SliceSpec("language", tuple(lang))


def _agreement(seed=3, concord=0.90):
    rng = np.random.default_rng(seed)
    h = rng.integers(0, 2, 200)
    j = np.where(rng.random(200) < concord, h, 1 - h)
    return krippendorff_alpha([h.tolist(), j.tolist()], MeasurementLevel.NOMINAL, floor=0.667)


COVERAGE_OK = Result("s6.null_judge", Verdict.PASS, "judge scored 500/500 (100%)")


def test_planted_aggregate_only_gate_approves_a_harmful_change():
    """NECESSITY PRE-CHECK for S9 -- the defect must actually approve the bad change."""
    eb, ec, _ = _sliced_fixture()
    assert ec.point > eb.point, "fixture invalid: the aggregate must improve"
    assert aggregate_only_gate_DO_NOT_USE(eb, ec).verdict is Verdict.PASS
    assert naive_promotion_DO_NOT_USE(eb, ec) is True


def test_slice_regression_blocks_despite_aggregate_improvement():
    """THE SIGNATURE TEST FOR S9."""
    eb, ec, spec = _sliced_fixture()
    rep = slice_gate(eb, ec, spec)
    assert rep.verdict is Verdict.FAIL
    assert rep.regressed == ("de",)
    assert rep.aggregate.verdict is ComparisonVerdict.BETTER
    assert rep.outcomes[0].qvalue is not None, "slices must be BH-adjusted"


def test_slice_gate_can_pass_reverse_test():
    """A slice gate that can never pass gets switched off within a month."""
    eb, ec, spec = _sliced_fixture(en_delta=0.05, de_delta=0.05)
    rep = slice_gate(eb, ec, spec)
    assert rep.verdict is Verdict.PASS and rep.regressed == ()


def test_underpowered_slice_blocks_rather_than_passing():
    eb, ec, _ = _sliced_fixture(n=500, n_minority=5, de_delta=0.0)
    lang = tuple(["en"] * 495 + ["de"] * 5)
    rep = slice_gate(eb, ec, SliceSpec("language", lang), min_slice_n=20)
    assert rep.verdict is Verdict.COULD_NOT_RUN
    assert "de" in rep.unevaluable and rep.verdict.blocks_promotion


def test_slice_spec_refuses_misalignment_and_blank_labels():
    eb, ec, _ = _sliced_fixture()
    with pytest.raises(SliceError):
        slice_gate(eb, ec, SliceSpec("language", ("en",) * 3))
    with pytest.raises(SliceError):
        SliceSpec("language", ("en", "", "de"))
    with pytest.raises(SliceError):
        SliceSpec("", ("en",))


def test_slice_gate_requires_per_item():
    eb, ec, spec = _sliced_fixture()
    bare = Estimate(eb.point, eb.interval, eb.provenance, ACC)
    with pytest.raises(SliceError):
        slice_gate(bare, ec, spec)


def test_promotion_decision_cannot_contradict_its_evidence():
    """The teaching invariant of gates/promotion.py."""
    for gates, verdict, why in [
        ((Result("g", Verdict.COULD_NOT_RUN, "unknown"),), Verdict.PASS, "blocking CNR"),
        ((Result("g", Verdict.FAIL, "bad"),), Verdict.PASS, "blocking FAIL"),
        ((), Verdict.PASS, "no gates at all"),
        ((Result("g", Verdict.PASS, "ok"),), Verdict.COULD_NOT_RUN, "verdict mismatch"),
    ]:
        with pytest.raises(PromotionError):
            PromotionDecision(True, verdict, gates, why)


def test_promotion_blocks_on_slice_regression():
    eb, ec, spec = _sliced_fixture()
    dec, reps = promotion_gate(primary=(eb, ec), slices=[spec],
                               judge_coverage=COVERAGE_OK, judge_agreement=_agreement())
    assert dec.verdict is Verdict.FAIL and dec.promote is False
    assert any("slice" in b for b in dec.blocking)
    assert len(reps) == 1


def test_promotion_can_promote_reverse_test():
    eb, ec, spec = _sliced_fixture(en_delta=0.05, de_delta=0.05)
    dec, _ = promotion_gate(primary=(eb, ec), slices=[spec],
                            judge_coverage=COVERAGE_OK, judge_agreement=_agreement())
    assert dec.verdict is Verdict.PASS and dec.promote is True and dec.blocking == ()


@pytest.mark.parametrize("override,why", [
    ({"judge_coverage": None}, "missing coverage must block"),
    ({"judge_agreement": None}, "judge-scored without agreement must block"),
    ({"slices": []}, "no slice breakdown must block"),
])
def test_promotion_fails_closed_on_missing_evidence(override, why):
    eb, ec, spec = _sliced_fixture(en_delta=0.05, de_delta=0.05)
    call = {"primary": (eb, ec), "slices": [spec],
            "judge_coverage": COVERAGE_OK, "judge_agreement": _agreement()}
    call.update(override)
    dec, _ = promotion_gate(**call)
    assert dec.verdict is Verdict.COULD_NOT_RUN, why
    assert dec.promote is False


def test_promotion_blocks_when_judge_below_agreement_floor():
    eb, ec, spec = _sliced_fixture(en_delta=0.05, de_delta=0.05)
    lazy = krippendorff_alpha([[1] * 170 + [0] * 30, [1] * 200],
                              MeasurementLevel.NOMINAL, floor=0.667)
    dec, _ = promotion_gate(primary=(eb, ec), slices=[spec],
                            judge_coverage=COVERAGE_OK, judge_agreement=lazy)
    assert dec.verdict is Verdict.FAIL and "gate.instrument" in dec.blocking


def test_promotion_blocks_on_incomparable_provenance():
    eb, ec, spec = _sliced_fixture(en_delta=0.05, de_delta=0.05)
    moved = Estimate(ec.point, ec.interval, Provenance("v2", "judge_v2", "set_a"),
                     ACC, ec.per_item)
    dec, _ = promotion_gate(primary=(eb, moved), slices=[spec],
                            judge_coverage=COVERAGE_OK, judge_agreement=_agreement())
    assert dec.verdict is Verdict.FAIL and "gate.comparability" in dec.blocking


def test_promotion_blocks_on_guardrail_regression():
    """A latency regression must block even when accuracy genuinely improves."""
    eb, ec, spec = _sliced_fixture(en_delta=0.05, de_delta=0.05)
    rng = np.random.default_rng(5)
    fast = rng.gamma(9.0, 30.0, 400)
    slow = fast + rng.gamma(4.0, 30.0, 400)
    lb = estimate_mean(fast, Provenance("v1", EXACT_JUDGE, "set_a"), LAT)
    lc = estimate_mean(slow, Provenance("v2", EXACT_JUDGE, "set_a"), LAT)
    dec, _ = promotion_gate(primary=(eb, ec), guardrails=[(lb, lc)], slices=[spec],
                            judge_coverage=COVERAGE_OK, judge_agreement=_agreement())
    assert dec.verdict is Verdict.FAIL
    assert any("guardrail" in b for b in dec.blocking)


def test_promotion_coverage_shortfall_blocks():
    eb, ec, spec = _sliced_fixture(en_delta=0.05, de_delta=0.05)
    short = Result("s6.null_judge", Verdict.COULD_NOT_RUN, "coverage 80% below floor 98%")
    dec, _ = promotion_gate(primary=(eb, ec), slices=[spec],
                            judge_coverage=short, judge_agreement=_agreement())
    assert dec.verdict is Verdict.COULD_NOT_RUN and dec.promote is False


def test_exact_metric_skips_instrument_gate_without_blocking():
    """SKIPPED must not block -- but it must still appear, so the gate list is the
    same shape every run and an absent gate is visible rather than inferred."""
    rng = np.random.default_rng(9)
    fast = rng.gamma(9.0, 40.0, 300)
    lb = estimate_mean(fast, Provenance("v1", EXACT_JUDGE, "s"), LAT)
    lc = estimate_mean(fast * 0.7, Provenance("v2", EXACT_JUDGE, "s"), LAT)
    dec, _ = promotion_gate(primary=(lb, lc), judge_coverage=COVERAGE_OK,
                            require_slice_breakdown=False)
    inst = [g for g in dec.gates if g.claim == "gate.instrument"]
    assert len(inst) == 1 and inst[0].verdict is Verdict.SKIPPED
    assert dec.promote is True


def test_slice_module_does_not_shadow_the_slice_builtin():
    """`gates/slice.py` is safe because `slice` is a builtin, not a module -- but
    after the `numbers/` stdlib incident this is asserted, not assumed."""
    assert slice(0, 3).stop == 3
    import evalplat.gates.slice as m
    assert m.MIN_SLICE_N == 20
