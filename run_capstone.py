#!/usr/bin/env python3
"""CAP-M7 capstone runner. Produces the measured numbers the spec demands.

EXIT CODES ARE THREE-STATE AND THIS IS THE POINT OF THE ARTIFACT:

    0  every REQUIRED claim was conclusive and passed
    1  a conclusive FAIL -- we measured, and it is bad
    2  a REQUIRED claim was never verified -- we did not measure

Use `python run_capstone.py; echo $?`. Do NOT pipe into `tail`/`head` before
reading `$?`: a pipeline returns the LAST command's status, so `run | tail; echo $?`
will report 0 over a real 2. That mistake was made while building the CAP-A2/S1
spike and is recorded in its FINDINGS.md section 6.

Run with the project venv, which carries the four mandated crosscheck references:
    .venv/bin/python run_capstone.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evalplat.audit.provenance import EvalSet, JudgeBundle  # noqa: E402
from evalplat.core import Report, Result, Verdict  # noqa: E402
from evalplat.gates.null_judge import (  # noqa: E402
    FlakyJudge,
    UnreachableJudge,
    WorkingJudge,
    naive_score_eval_set_DO_NOT_USE,
    null_judge_gate,
)
from evalplat.gates.promotion import (  # noqa: E402
    naive_promotion_DO_NOT_USE,
    promotion_gate,
)
from evalplat.gates.slice import (  # noqa: E402
    SliceSpec,
    aggregate_only_gate_DO_NOT_USE,
    slice_gate,
)
from evalplat.judges.agreement import (  # noqa: E402
    MeasurementLevel,
    krippendorff_alpha,
    percent_agreement,
)
from evalplat.numbers.ci import (  # noqa: E402
    _normal_approx_ci_DO_NOT_USE,
    bootstrap_ci,
    estimate_mean,
    wilson_ci,
)
from evalplat.numbers.estimate import (  # noqa: E402
    EXACT_JUDGE,
    ComparisonVerdict,
    Estimate,
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
    unadjusted_DO_NOT_USE,
)

ACC = Metric("accuracy", higher_is_better=True)
LAT = Metric("p95_latency", higher_is_better=False, unit="ms")
BUNDLE = JudgeBundle("llama3:8b", "Rate the answer 0-1.", "1 = fully correct", 0.0, "parser-v2")
EVALSET = EvalSet("support-qa", tuple({"id": f"q{k}", "gold": 0.8} for k in range(200)), "v1")


def _try(rep: Report, claim: str, fn) -> None:
    """Any unexpected exception becomes COULD_NOT_RUN, never a silent pass."""
    try:
        rep.add(fn())
    except Exception as exc:  # noqa: BLE001
        rep.add(Result(claim, Verdict.COULD_NOT_RUN, f"{type(exc).__name__}: {exc}"))


def s1_every_number_has_an_interval(rep: Report) -> None:
    print("\nS1  every number carries an interval, and a gain inside noise is not a gain")
    base = Estimate(0.800, Interval(0.760, 0.840, 0.95, "analytic-t", 200),
                    Provenance("v1", BUNDLE.hash, EVALSET.hash), ACC)
    cand = Estimate(0.820, Interval(0.780, 0.860, 0.95, "analytic-t", 200),
                    Provenance("v2", BUNDLE.hash, EVALSET.hash), ACC)
    c = compare(base, cand)
    print(f"    baseline  {base.render()}")
    print(f"    candidate {cand.render()}")
    print(f"    -> {c.render()}")
    rep.add(Result(
        "s1.interval_mandatory",
        Verdict.PASS if c.verdict is ComparisonVerdict.INDISTINGUISHABLE else Verdict.FAIL,
        f"a +2.0pt point gain resolves to {c.verdict.value}; promotion permitted="
        f"{c.verdict.permits_promotion}",
        {"difference": c.difference, "diff_ci": [c.diff_interval.lo, c.diff_interval.hi],
         "method": c.method},
    ))

    print("\n    the naive proportion interval, measured against Wilson:")
    rows = []
    for k, n in [(49, 50), (50, 50), (98, 100)]:
        lo, hi = _normal_approx_ci_DO_NOT_USE(k, n)
        w = wilson_ci(k, n)
        bad = "IMPOSSIBLE >1" if hi > 1.0 else ("ZERO WIDTH" if hi == lo else "")
        print(f"      p={k/n:.2f} n={n}: naive=[{lo:.4f},{hi:.4f}] {bad:<13} "
              f"wilson=[{w.lo:.4f},{w.hi:.4f}] width={w.width:.4f}")
        rows.append({"p": k / n, "n": n, "naive_hi": hi, "wilson_width": w.width})
    worst = max(r["naive_hi"] for r in rows)
    rep.add(Result(
        "s1.wilson_required", Verdict.PASS if worst > 1.0 else Verdict.FAIL,
        f"the naive interval reaches {worst:.4f} (>1 is impossible) while Wilson stays "
        "inside [0,1] and retains width at the boundary", {"rows": rows},
    ))


def s1b_overlap_is_the_wrong_test(rep: Report) -> None:
    print("\nS1b the interval-overlap test is wrong; the difference interval is not")
    rng = np.random.default_rng(7)
    base_v = rng.normal(0.80, 0.15, 400).clip(0, 1)
    cand_v = (base_v + 0.02).clip(0, 1)
    eb = estimate_mean(base_v, Provenance("v1", BUNDLE.hash, EVALSET.hash), ACC)
    ec = estimate_mean(cand_v, Provenance("v2", BUNDLE.hash, EVALSET.hash), ACC)
    overlap = eb.interval.overlaps(ec.interval)
    c = compare(eb, ec)
    ratio = eb.interval.width / c.diff_interval.width
    print(f"    marginal intervals overlap: {overlap}  -> naive verdict would be 'no difference'")
    print(f"    paired difference CI: [{c.diff_interval.lo:+.5f}, {c.diff_interval.hi:+.5f}]"
          f" via {c.method}")
    print(f"    correct verdict: {c.verdict.value}  (difference CI is {ratio:.0f}x tighter"
          f" than either marginal)")
    rep.add(Result(
        "s1b.overlap_vs_difference",
        Verdict.PASS if (overlap and c.verdict is ComparisonVerdict.BETTER) else Verdict.FAIL,
        f"intervals overlap yet the paired difference excludes zero; the difference CI is "
        f"{ratio:.0f}x tighter, so the overlap test would have blocked a real improvement",
        {"overlap": overlap, "verdict": c.verdict.value, "tightness_ratio": ratio},
    ))


def s2_judge_versioning(rep: Report) -> None:
    print("\nS2  scores are bound to (judge_bundle_hash, eval_set_hash)")
    upgraded = BUNDLE.with_changes(prompt=BUNDLE.prompt + " Be strict.")
    print(f"    judge v1 hash {BUNDLE.hash[:16]}")
    print(f"    judge v2 hash {upgraded.hash[:16]}  (prompt edited: +10 chars)")
    old = Estimate(0.80, Interval(0.76, 0.84, 0.95, "analytic-t", 200),
                   Provenance("v1", BUNDLE.hash, EVALSET.hash), ACC)
    new = Estimate(0.86, Interval(0.82, 0.90, 0.95, "analytic-t", 200),
                   Provenance("v2", upgraded.hash, EVALSET.hash), ACC)
    c = compare(old, new)
    print(f"    -> {c.render()}")
    changed = sum(
        BUNDLE.with_changes(**{f: v}).hash != BUNDLE.hash
        for f, v in [("model_id", "llama3:70b"), ("prompt", "x"), ("rubric", "y"),
                     ("temperature", 0.1), ("parser_version", "p3")]
    )
    rep.add(Result(
        "s2.judge_versioning",
        (
            Verdict.PASS
            if c.verdict is ComparisonVerdict.INCOMPARABLE and changed == 5
            else Verdict.FAIL
        ),
        f"a judge edit makes the comparison {c.verdict.value}; {changed}/5 identity "
        "fields independently change the hash",
        {"verdict": c.verdict.value, "fields_covered": changed},
    ))


def s3_judge_human_agreement(rep: Report) -> None:
    print("\nS3  a judge is unusable until it agrees with humans")
    rng = np.random.default_rng(3)
    humans = rng.integers(0, 2, 200)
    good = np.where(rng.random(200) < 0.90, humans, 1 - humans)
    imbalanced = np.array([1] * 170 + [0] * 30)
    lazy = np.ones(200, dtype=int)

    g = krippendorff_alpha([humans.tolist(), good.tolist()], MeasurementLevel.NOMINAL, floor=0.667)
    b = krippendorff_alpha(
        [imbalanced.tolist(), lazy.tolist()], MeasurementLevel.NOMINAL, floor=0.667
    )
    raw = percent_agreement(imbalanced, lazy)
    print(f"    genuine judge : {g.render()}")
    print(f"    lazy judge    : raw agreement {raw:.1%} -> alpha {b.value:+.4f}")
    print(f"                    {b.render()}")
    rep.add(Result(
        "s3.judge_human_anchor",
        Verdict.PASS if (g.verdict is Verdict.PASS and b.verdict is Verdict.FAIL) else Verdict.FAIL,
        f"agreement floor passes a real judge (alpha={g.value:.3f}) and fails a "
        f"majority-class judge that scores {raw:.0%} raw agreement (alpha={b.value:+.3f})",
        {"good_alpha": g.value, "lazy_alpha": b.value, "lazy_raw_agreement": raw},
    ))


def s4_preference_modelling(rep: Report) -> None:
    print("\nS4  Bradley-Terry vs win rate under unbalanced pairings")
    true_t = {"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0, "E": -2.0}
    items, truth = list(true_t), tuple(true_t)
    rng = np.random.default_rng(1)
    sched = {("A", "B"): 60, ("B", "C"): 40, ("C", "D"): 20,
             ("D", "E"): 60, ("C", "E"): 20, ("B", "E"): 10}
    games = []
    for (a, b), n in sched.items():
        p = 1 / (1 + math.exp(-(true_t[a] - true_t[b])))
        games += [(a, b) if rng.random() < p else (b, a) for _ in range(n)]
    wr = win_rate_ranking(games, items)
    wr_rank = tuple(sorted(items, key=lambda i: (-wr[i], i)))
    bt = bradley_terry(games, items, n_boot=400)
    d_wr, d_bt = rank_displacement(truth, wr_rank), rank_displacement(truth, bt.ranking)
    print("    win rates: " + "  ".join(f"{i}={wr[i]:.3f}" for i in items))
    print(f"    win-rate ranking {wr_rank}  displacement={d_wr}")
    print(f"    BT ranking       {bt.ranking}  displacement={d_bt}")
    print(f"    truth            {truth}")

    wrong_wr = wrong_bt = 0
    trials = 200
    for t in range(trials):
        r = np.random.default_rng(9000 + t)
        pairs = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("C", "E"), ("B", "E"), ("A", "C")]
        gg = []
        for (a, b), n in zip(pairs, r.integers(8, 70, len(pairs))):
            p = 1 / (1 + math.exp(-(true_t[a] - true_t[b])))
            gg += [(a, b) if r.random() < p else (b, a) for _ in range(int(n))]
        w = win_rate_ranking(gg, items)
        wrong_wr += rank_displacement(truth, tuple(sorted(items, key=lambda i: (-w[i], i)))) > 0
        bb = bradley_terry(gg, items, n_boot=0)
        if bb.theta[0] == bb.theta[0]:
            wrong_bt += rank_displacement(truth, bb.ranking) > 0
    print(f"    over {trials} random unbalanced schedules: win-rate misranks "
          f"{wrong_wr/trials:.1%}, BT misranks {wrong_bt/trials:.1%}")
    rep.add(Result(
        "s4.preference_modelling",
        Verdict.PASS if (d_bt == 0 and d_wr > 0 and wrong_wr > wrong_bt) else Verdict.FAIL,
        f"BT recovers the true order (displacement {d_bt}) where win rate does not "
        f"({d_wr}); across {trials} schedules win rate misranks {wrong_wr/trials:.0%} "
        f"vs BT {wrong_bt/trials:.0%}",
        {"displacement_winrate": d_wr, "displacement_bt": d_bt,
         "misrank_rate_winrate": wrong_wr / trials, "misrank_rate_bt": wrong_bt / trials},
    ))


def s5_multiplicity(rep: Report) -> None:
    print("\nS5  multiplicity control across models x tasks")
    m, trials = 96, 500
    un = bh = bo = 0
    any_un = any_bh = 0
    for t in range(trials):
        p = np.random.default_rng(1000 + t).uniform(0, 1, m)
        u, b, o = (unadjusted_DO_NOT_USE(p, .05), benjamini_hochberg(p, .05), bonferroni(p, .05))
        un += u.n_rejected
        bh += b.n_rejected
        bo += o.n_rejected
        any_un += u.n_rejected > 0
        any_bh += b.n_rejected > 0
    print(f"    8 models x 12 tasks = {m} comparisons, ALL nulls true, {trials} trials")
    print(f"    mean false discoveries: unadjusted={un/trials:.3f}  BH={bh/trials:.3f}  "
          f"bonferroni={bo/trials:.3f}")
    print(
        f"    P(at least one false winner): unadjusted={any_un/trials:.1%}"
        f"  BH={any_bh/trials:.1%}"
    )
    rep.add(Result(
        "s5.multiplicity",
        Verdict.PASS if (un / trials > 3.0 and bh / trials < 0.5) else Verdict.FAIL,
        f"unadjusted thresholding manufactures {un/trials:.2f} false winners per "
        f"leaderboard and declares a winner in {any_un/trials:.0%} of runs where every "
        f"model is identical; BH holds that to {bh/trials:.2f} and {any_bh/trials:.0%}",
        {"false_discoveries_unadjusted": un / trials, "false_discoveries_bh": bh / trials,
         "p_any_false_winner_unadjusted": any_un / trials, "family_size": m},
    ))


def s6_null_judge(rep: Report) -> None:
    print("\nS6  null-judge fail-closed  [SIGNATURE TEST]")
    naive = naive_score_eval_set_DO_NOT_USE(UnreachableJudge(BUNDLE), EVALSET)
    print(f"    the naive harness on a DEAD judge returns a list of len={len(naive)}")
    print("    ...which a caller reads as '0 failures' and a pipeline reads as green")
    dead = null_judge_gate(UnreachableJudge(BUNDLE), EVALSET)
    partial = null_judge_gate(FlakyJudge(BUNDLE, fail_every=5), EVALSET, coverage_floor=0.98)
    working = null_judge_gate(WorkingJudge(BUNDLE), EVALSET)
    for label, r in [("dead judge", dead), ("20% failures", partial), ("working judge", working)]:
        print(f"    {label:<15} -> {r.verdict.value:<13} blocks={r.verdict.blocks_promotion}")
    ok = (dead.verdict is Verdict.COULD_NOT_RUN and dead.verdict.blocks_promotion
          and partial.verdict is Verdict.COULD_NOT_RUN and working.verdict is Verdict.PASS)
    rep.add(Result(
        "s6.null_judge_fail_closed", Verdict.PASS if ok else Verdict.FAIL,
        f"the naive path yields {len(naive)} scores and no signal; the gate returns "
        "COULD_NOT_RUN (which blocks) for both total and partial outage, and PASS for a "
        "working judge",
        {"naive_len": len(naive), "dead": dead.verdict.value,
         "partial_coverage": partial.detail.get("coverage"), "working": working.verdict.value},
    ))


def s10_cost_latency(rep: Report) -> None:
    print("\nS10 cost and latency are first-class axes, with direction in the type")
    rng = np.random.default_rng(21)
    slow = rng.gamma(9.0, 40.0, 500)
    fast = rng.gamma(9.0, 28.0, 500)
    def p95(a):
        return float(np.quantile(a, 0.95))
    e_slow = Estimate(p95(slow), bootstrap_ci(slow, p95, seed=1),
                      Provenance("v1", EXACT_JUDGE, EVALSET.hash), LAT)
    e_fast = Estimate(p95(fast), bootstrap_ci(fast, p95, seed=1),
                      Provenance("v2", EXACT_JUDGE, EVALSET.hash), LAT)
    c = compare(e_slow, e_fast)
    print(f"    v1 {e_slow.render()}")
    print(f"    v2 {e_fast.render()}")
    print(f"    -> {c.render()}")
    rep.add(Result(
        "s10.cost_latency_axes",
        Verdict.PASS if c.verdict is ComparisonVerdict.BETTER else Verdict.FAIL,
        f"a p95 latency DECREASE of {abs(c.difference):.1f}ms reads as "
        f"{c.verdict.value} because direction travels in the Metric type",
        {"verdict": c.verdict.value, "difference_ms": c.difference},
    ))


def _sliced_case(en_delta: float, de_delta: float, seed: int = 11):
    """500 items, 90% 'en' / 10% 'de', with noise on the per-item deltas so the
    paired differences are not all identical."""
    rng = np.random.default_rng(seed)
    lang = np.array(["en"] * 450 + ["de"] * 50)
    base = rng.normal(0.80, 0.10, 500).clip(0, 1)
    cand = base.copy()
    jit = rng.normal(0, 0.02, 500)
    cand[lang == "en"] += en_delta + jit[lang == "en"]
    cand[lang == "de"] += de_delta + jit[lang == "de"]
    cand = cand.clip(0, 1)
    eb = estimate_mean(base, Provenance("v1", BUNDLE.hash, EVALSET.hash), ACC)
    ec = estimate_mean(cand, Provenance("v2", BUNDLE.hash, EVALSET.hash), ACC)
    return eb, ec, SliceSpec("language", tuple(lang)), lang, base, cand


def s9_slices(rep: Report) -> None:
    print("\nS9  a slice regression blocks promotion even when the aggregate improves")
    eb, ec, spec, lang, base, cand = _sliced_case(0.03, -0.15)
    de_delta = float(cand[lang == "de"].mean() - base[lang == "de"].mean())
    print(f"    aggregate  {eb.point:.4f} -> {ec.point:.4f}  ({ec.point - eb.point:+.4f})")
    print(f"      en (450) {base[lang=='en'].mean():.4f} -> {cand[lang=='en'].mean():.4f}")
    de_base, de_cand = base[lang == 'de'].mean(), cand[lang == 'de'].mean()
    print(f"      de ( 50) {de_base:.4f} -> {de_cand:.4f}   <-- harmed")
    naive = aggregate_only_gate_DO_NOT_USE(eb, ec)
    print(f"    aggregate-only gate (the defect) -> {naive.verdict.value}")
    print(f"    naive_promotion_DO_NOT_USE       -> {naive_promotion_DO_NOT_USE(eb, ec)}")
    slices = slice_gate(eb, ec, spec)
    print(slices.render())
    cb, cc, cspec, *_ = _sliced_case(0.05, 0.05)
    clean = slice_gate(cb, cc, cspec)
    print(f"    REVERSE TEST (both slices improve) -> {clean.verdict.value}")
    ok = (naive.verdict is Verdict.PASS and slices.verdict is Verdict.FAIL
          and "de" in slices.regressed and clean.verdict is Verdict.PASS)
    rep.add(Result(
        "s9.slice_gate", Verdict.PASS if ok else Verdict.FAIL,
        f"the aggregate reads BETTER ({ec.point - eb.point:+.4f}) and the aggregate-only "
        f"gate APPROVES, while the slice gate blocks on 'de' ({de_delta:+.4f}); a clean "
        "change still passes",
        {"aggregate_delta": float(ec.point - eb.point), "de_delta": de_delta,
         "naive_gate": naive.verdict.value, "slice_gate": slices.verdict.value,
         "regressed": list(slices.regressed), "reverse_test": clean.verdict.value},
    ))


def s_promotion(rep: Report) -> None:
    print("\nPROMOTION  the composition layer -- what must be true simultaneously")
    rng = np.random.default_rng(3)
    h = rng.integers(0, 2, 200)
    j = np.where(rng.random(200) < 0.90, h, 1 - h)
    agree = krippendorff_alpha([h.tolist(), j.tolist()], MeasurementLevel.NOMINAL, floor=0.667)
    cov = null_judge_gate(WorkingJudge(BUNDLE), EVALSET)

    hb, hc, hspec, *_ = _sliced_case(0.03, -0.15)
    gb, gc, gspec, *_ = _sliced_case(0.05, 0.05)
    blocked, _ = promotion_gate(primary=(hb, hc), slices=[hspec],
                                judge_coverage=cov, judge_agreement=agree)
    print(blocked.render())
    promoted, _ = promotion_gate(primary=(gb, gc), slices=[gspec],
                                 judge_coverage=cov, judge_agreement=agree)
    print("\n    REVERSE TEST -- a genuinely good change:")
    print(f"    {promoted.render().splitlines()[0]}")

    missing, _ = promotion_gate(primary=(gb, gc), judge_coverage=cov, judge_agreement=agree)
    dead, _ = promotion_gate(
        primary=(gb, gc), slices=[gspec], judge_agreement=agree,
        judge_coverage=null_judge_gate(UnreachableJudge(BUNDLE), EVALSET),
    )
    print(f"    no slice breakdown supplied  -> {missing.verdict.value}, promote={missing.promote}")
    print(f"    judge unreachable            -> {dead.verdict.value}, promote={dead.promote}")

    ok = (blocked.promote is False and promoted.promote is True
          and missing.verdict is Verdict.COULD_NOT_RUN
          and dead.verdict is Verdict.COULD_NOT_RUN)
    rep.add(Result(
        "promotion.composition", Verdict.PASS if ok else Verdict.FAIL,
        f"{len(blocked.gates)} gates compose into one four-state decision: it blocks a "
        "slice-harming change, promotes a clean one, and returns COULD_NOT_RUN -- never "
        "promote -- when the slice breakdown or the judge is missing",
        {"blocked": blocked.verdict.value, "blocking_gates": list(blocked.blocking),
         "promoted": promoted.promote, "n_gates": len(blocked.gates),
         "missing_slices": missing.verdict.value, "dead_judge": dead.verdict.value},
    ))


def main() -> int:
    print("CAP-M7 evaluation-platform -- capstone run")
    print("=" * 78)
    rep = Report()
    rep.require(
        "s1.interval_mandatory", "s1.wilson_required", "s1b.overlap_vs_difference",
        "s2.judge_versioning", "s3.judge_human_anchor", "s4.preference_modelling",
        "s5.multiplicity", "s6.null_judge_fail_closed", "s9.slice_gate",
        "promotion.composition", "s10.cost_latency_axes",
    )
    _try(rep, "s1.interval_mandatory", lambda: s1_every_number_has_an_interval(rep) or
         Result("s1.runner", Verdict.PASS, "scenario completed"))
    _try(rep, "s1b.overlap_vs_difference", lambda: s1b_overlap_is_the_wrong_test(rep) or
         Result("s1b.runner", Verdict.PASS, "scenario completed"))
    _try(rep, "s2.judge_versioning", lambda: s2_judge_versioning(rep) or
         Result("s2.runner", Verdict.PASS, "scenario completed"))
    _try(rep, "s3.judge_human_anchor", lambda: s3_judge_human_agreement(rep) or
         Result("s3.runner", Verdict.PASS, "scenario completed"))
    _try(rep, "s4.preference_modelling", lambda: s4_preference_modelling(rep) or
         Result("s4.runner", Verdict.PASS, "scenario completed"))
    _try(rep, "s5.multiplicity", lambda: s5_multiplicity(rep) or
         Result("s5.runner", Verdict.PASS, "scenario completed"))
    _try(rep, "s6.null_judge_fail_closed", lambda: s6_null_judge(rep) or
         Result("s6.runner", Verdict.PASS, "scenario completed"))
    _try(rep, "s9.slice_gate", lambda: s9_slices(rep) or
         Result("s9.runner", Verdict.PASS, "scenario completed"))
    _try(rep, "promotion.composition", lambda: s_promotion(rep) or
         Result("promotion.runner", Verdict.PASS, "scenario completed"))
    _try(rep, "s10.cost_latency_axes", lambda: s10_cost_latency(rep) or
         Result("s10.runner", Verdict.PASS, "scenario completed"))

    print("\n" + "=" * 78)
    print("VERDICTS")
    print(rep.render())
    rep.to_json("findings.json")
    code = rep.exit_code()
    print("\nwrote findings.json")
    print({0: "EXIT 0 -- all required claims conclusive and passing",
           1: "EXIT 1 -- conclusive failure(s)",
           2: "EXIT 2 -- required claim(s) unverified"}[code])
    return code


if __name__ == "__main__":
    sys.exit(main())
