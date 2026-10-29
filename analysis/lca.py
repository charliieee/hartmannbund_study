"""
lca.py  --  latent class analysis of digital application use (npj Digital Medicine article)

Diagnostics for the latent class analysis of the 19 digital-application items:
  A  model selection k = 1..5 (LL, parameters, AIC, BIC, aBIC, entropy, class sizes)
  B  convergence: log-likelihood of every random start (k = 3, 4) under the reference
     specification (20 starts, seed 42) and under 100 starts (seeds 42, 1, 2026)
  C  class assignment certainty (posterior probabilities) for k = 3 and k = 4
  D  cross-classification of the 3- and 4-class solutions, class profiles
  E  sector: class x sector, class characteristics within hospital / non-hospital,
     logistic regressions (class + sector, class x sector)
  G  optional parametric bootstrap likelihood-ratio tests (--blrt N), run last
  H  Fig. 2 and Figure S8-1 (3- and 4-class solution)
  I  checksums against the article / Supplementary Note 8

Usage:
  python lca.py <analysis_dataset.csv> <outdir> [--blrt N] [--blrt-starts S]

Reference specification of the 19-item model (identical to results_R6.py): StepMix 3.0, categorical
measurement model, 19 items on three levels, n_init = 20, random_state = 42,
abs_tol = 1e-10, max_iter = 1000, n = 356 respondents with >= 15 valid items.

Primary model:
  python lca.py analysis_dataset.csv <outdir> --coding four --starts 100 --drop 12,15,16 --labels primary --blrt 200
  (four response levels: not available | unclear what this is | rarely-monthly | weekly-daily; 16 items after
  exclusion of tele-rounds, digital pre-registration and triage tools; 100 random starts; three classes)
Options: --coding primary|four, --starts N, --drop i,j,k (item numbers 1-19), --labels primary|four, --blrt N, --blrt-starts S
"""

import copy
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import adjusted_rand_score
from stepmix import StepMix

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- arguments
INFILE = sys.argv[1]
OUTDIR = sys.argv[2]
N_BLRT = 0
BLRT_STARTS = 10
if "--blrt" in sys.argv:
    N_BLRT = int(sys.argv[sys.argv.index("--blrt") + 1])
if "--blrt-starts" in sys.argv:
    BLRT_STARTS = int(sys.argv[sys.argv.index("--blrt-starts") + 1])
CODING = "primary"            # "primary": 3 levels as in the 19-item model; "four": not available | unclear | rarely-monthly | weekly-daily
if "--coding" in sys.argv:
    CODING = sys.argv[sys.argv.index("--coding") + 1]
assert CODING in ("primary", "four"), CODING
REF_STARTS_ARG = int(sys.argv[sys.argv.index("--starts") + 1]) if "--starts" in sys.argv else 20
DROP = []                       # item numbers (1-19) to exclude, e.g. --drop 15,16
if "--drop" in sys.argv:
    DROP = [int(v) for v in sys.argv[sys.argv.index("--drop") + 1].split(",")]
os.makedirs(OUTDIR, exist_ok=True)

LOG = open(os.path.join(OUTDIR, "lca_log.txt"), "w", encoding="utf-8")


def log(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def save(df, name, **kw):
    df.to_csv(os.path.join(OUTDIR, name), **kw)
    log(f"  -> written {name}")


REF_STARTS, REF_SEED = REF_STARTS_ARG, 42      # reference specification: 20 starts (19-item model); --starts 100 for the four-level coding
ITEM_COLS = [f"lca_item_{i:02d}" for i in range(1, 20)]
ITEM_LABELS = ["Practice/hospital IS", "Electronic patient record (ePA)",
               "e-medication list", "e-sick note (eAU)", "PACS / imaging",
               "Laboratory IS", "Clinical decision support", "DiGA",
               "Health apps", "Phone assistance", "Video consultation",
               "Tele-rounds", "Tele-/home monitoring", "Speech recognition",
               "Digital pre-registration", "Triage tools",
               "Medication safety (AMTS)", "Automated duty scheduling",
               "Knowledge management"]
FREQ = {"täglich": 4, "wöchentlich": 3, "monatlich": 2, "seltener": 1,
        "möchte ich, aber nicht verfügbar": 0, "unklar, was das ist": 0}

# Provisional class labels (ordered by mean digital index, descending). The final
# neutral labels are decided in the package document; change here and re-run.
LABELS_BY_CODING = {
    "primary": {
        3: ["Wide range, routine use", "Wide range, occasional use", "Core set, routine use"],
        4: ["Wide range, routine use", "Wide range, occasional use",
            "Core set, routine use (ambulatory profile)", "Core set, routine use (hospital profile)"],
    },
    "four": {   # provisional labels for the four-level coding (ordered by mean index)
        3: ["Wide range", "Core set, others not available", "Core set, others unfamiliar"],
        4: ["Wide range, routine use", "Wide range, occasional use",
            "Core set, others not available", "Core set, others unfamiliar"],
    },
}
LABEL_SET = CODING              # --labels primary|four: which provisional label set to use (default: follows the coding)
if "--labels" in sys.argv:
    LABEL_SET = sys.argv[sys.argv.index("--labels") + 1]
LABELS = LABELS_BY_CODING[LABEL_SET]
# Level coding of the six response options (missing -> level 0 in both codings)
LEVEL_MAP = {
    "primary": {"möchte ich, aber nicht verfügbar": 0, "unklar, was das ist": 0,
                "seltener": 1, "monatlich": 1, "wöchentlich": 2, "täglich": 2},
    "four": {"möchte ich, aber nicht verfügbar": 0, "unklar, was das ist": 1,
             "seltener": 2, "monatlich": 2, "wöchentlich": 3, "täglich": 3},
}[CODING]
LEVEL_NAMES = {"primary": ["not available/unclear", "rarely-monthly", "weekly-daily"],
               "four": ["not available", "unclear", "rarely-monthly", "weekly-daily"]}[CODING]
N_LEVELS = len(LEVEL_NAMES)
OPTIONS = ["täglich", "wöchentlich", "monatlich", "seltener",
           "möchte ich, aber nicht verfügbar", "unklar, was das ist"]
OPT_EN = {"täglich": "daily", "wöchentlich": "weekly", "monatlich": "monthly", "seltener": "rarely",
          "möchte ich, aber nicht verfügbar": "not_available", "unklar, was das ist": "unclear"}
# Three-level-coding labels, kept for the cross-reference with Supplementary Note 8
OLD_LABELS = {3: ["Frontrunner", "Intermediate", "Basic"]}


def fit(X, k, n_init=REF_STARTS, seed=REF_SEED, save_inits=False):
    m = StepMix(n_components=k, measurement="categorical", n_init=n_init,
                random_state=seed, verbose=0, progress_bar=False,
                save_param_init=save_inits)
    m.fit(X)
    return m


def fit_stats(m, X):
    n = len(X)
    ll = m.score(X) * n
    k = m.n_components
    p = int(m.n_parameters)
    assert p == (k - 1) + k * N_ITEMS * (N_LEVELS - 1), (p, k)   # primary, 19 items: same count as results_R6.py
    pr = m.predict_proba(X)
    entropy = 1 + (pr * np.log(np.clip(pr, 1e-12, None))).sum() / (n * np.log(k)) if k > 1 else np.nan
    sizes = sorted(np.bincount(m.predict(X), minlength=k), reverse=True)
    return dict(k=k, logLik=ll, n_par=p, AIC=-2 * ll + 2 * p,
                BIC=-2 * ll + p * np.log(n),
                aBIC=-2 * ll + p * np.log((n + 2) / 24),
                entropy=entropy, smallest_class_n=min(sizes),
                smallest_class_pct=100 * min(sizes) / n,
                class_sizes="/".join(str(s) for s in sizes))


def ordered_classes(m, X, index, k):
    """Modal class, ordered by mean digital index (descending) -> 0..k-1."""
    raw = pd.Series(m.predict(X), index=X.index)
    order = index.groupby(raw).mean().sort_values(ascending=False).index
    remap = {c: i for i, c in enumerate(order)}
    return raw.map(remap), remap


# ============================================================================= load
t0 = time.time()
df = pd.read_csv(INFILE)
keep = df[ITEM_COLS].notna().all(axis=1)          # >= 15 valid items (of all 19), as in prepare_data.py
sub = df[keep].copy()
raw_opts = sub[df.columns[18:37]].copy()
raw_opts.columns = ITEM_LABELS
ALL_ITEM_LABELS = list(ITEM_LABELS)
if DROP:
    dropped = [ALL_ITEM_LABELS[i - 1] for i in DROP]
    raw_opts = raw_opts.drop(columns=dropped)
    ITEM_LABELS = [l for l in ALL_ITEM_LABELS if l not in dropped]
    ITEM_COLS_USED = [c for i, c in enumerate(ITEM_COLS, 1) if i not in DROP]
else:
    dropped = []
    ITEM_COLS_USED = list(ITEM_COLS)
N_ITEMS = len(ITEM_LABELS)
X = raw_opts.apply(lambda col: col.map(LEVEL_MAP)).fillna(0).astype(int)
X.columns = ITEM_COLS_USED
if CODING == "primary" and not DROP:
    assert (X.values == df.loc[keep, ITEM_COLS].astype(int).values).all(), "coding differs from lca_item columns"
n = len(X)
log(f"Coding: {CODING}; levels: {LEVEL_NAMES}; items used: {N_ITEMS}" + (f"; dropped: {dropped}" if DROP else ""))
log(f"LCA sample: n = {n} (>= 15 valid items), {X.shape[1]} items, "
    f"levels {sorted(pd.unique(X.values.ravel()))}")
opt_ct = pd.crosstab(raw_opts.fillna("(missing)").stack(future_stack=True),
                     X.stack(future_stack=True).values)
log("Response option x coded level (cells):\n" + opt_ct.to_string())
save(opt_ct, "0_response_option_x_level.csv")

setting = sub[df.columns[11]]
sub["sector3"] = np.select(
    [setting.isin(["Krankenhaus (stationär)", "Krankenhaus (ambulant)"]),
     setting.isin(["Praxis (Einzelpraxis)",
                   "Praxis (Gemeinschaftspraxis/Berufsausübungsgemeinschaft)",
                   "MVZ (Medizinisches Versorgungszentrum)"])],
    ["hospital", "ambulatory"], default="other")
sub["hospital"] = (sub["sector3"] == "hospital").astype(int)
sub["prov_yes"] = (sub["provision"] == "yes").astype(int)
defined = sub["ai_use"].notna()
log("Sector in LCA sample:", sub["sector3"].value_counts().to_dict())
log(f"Defined AI-use status in LCA sample: n = {int(defined.sum())}")

raw_items = raw_opts.apply(lambda col: col.map(FREQ)).fillna(0)      # mean-frequency heatmap input (0-4), items as used

# ============================================================================= A  model selection
log(f"\n=== A  Model selection, reference specification ({REF_STARTS} starts, seed {REF_SEED}) ===")
models, rows = {}, []
for k in range(1, 6):
    m = fit(X, k, save_inits=(k in (3, 4)))
    models[k] = m
    rows.append(fit_stats(m, X))
sel = pd.DataFrame(rows).set_index("k")
log(sel.round(3).to_string())
save(sel.round(4), "A_model_selection.csv")
log("BIC minimum at k =", sel["BIC"].idxmin(), "| aBIC minimum at k =", sel["aBIC"].idxmin(),
    "| AIC minimum at k =", sel["AIC"].idxmin(), "| entropy maximum at k =", sel["entropy"].idxmax())

cls3, _ = ordered_classes(models[3], X, sub["digital_index"], 3)
cls4, _ = ordered_classes(models[4], X, sub["digital_index"], 4)
sub["cls3"] = cls3
sub["cls4"] = cls4

# ============================================================================= B  convergence
log("\n=== B  Convergence: log-likelihood across random starts ===")
TOL = 0.01   # LL units; starts within TOL of the best start count as having reached it
conv_rows = []
start_tables = {}
for k in (3, 4):
    m = models[k]
    ll_starts = np.array(m.lower_bound_buffer_) * n
    best = ll_starts.max()
    at_max = np.abs(ll_starts - best) < TOL
    # partition agreement of every start with the retained solution
    ref_lab = m.predict(X)
    aris = []
    for params in m.param_buffer_:
        mm = copy.deepcopy(m)
        mm.set_parameters(params)
        aris.append(adjusted_rand_score(ref_lab, mm.predict(X)))
    aris = np.array(aris)
    tab = pd.DataFrame({"start": np.arange(1, len(ll_starts) + 1), "logLik": ll_starts,
                        "delta_to_best": ll_starts - best, "reached_best": at_max,
                        "ARI_vs_retained": aris})
    start_tables[k] = tab
    save(tab.round(4), f"B_starts_k{k}_{REF_STARTS}starts_seed{REF_SEED}.csv", index=False)
    distinct = np.unique(np.round(ll_starts, 2))
    near5 = np.unique(np.round(ll_starts, 1))
    near5 = near5[near5 > best - 5]
    conv_rows.append(dict(k=k, spec=f"{REF_STARTS} starts, seed {REF_SEED} (reference)", n_starts=len(ll_starts),
                          best_logLik=best, starts_at_best=int(at_max.sum()),
                          share_at_best=at_max.mean(),
                          starts_same_partition=int((aris > 0.999).sum()),
                          n_distinct_maxima=len(distinct),
                          n_distinct_maxima_within_5LL=len(near5),
                          worst_logLik=ll_starts.min(),
                          median_logLik=np.median(ll_starts)))
    log(f"k = {k}: best LL {best:.2f}; {at_max.sum()}/{len(ll_starts)} starts within {TOL} of best; "
        f"{(aris > 0.999).sum()} starts yield the identical partition; "
        f"distinct LL values (2 dp): {len(distinct)}; worst start {ll_starts.min():.2f}")
    log("   LL of the 20 starts, sorted:", np.sort(ll_starts)[::-1].round(2).tolist())

    # more starts: 100 starts with the reference seed (the first 20 draws are identical
    # to the reference run because the RandomState stream is shared) and two other seeds
    for seed in (42, 1, 2026):
        m100 = fit(X, k, n_init=100, seed=seed)
        ll100 = np.array(m100.lower_bound_buffer_) * n
        best100 = ll100.max()
        ari = adjusted_rand_score(ref_lab, m100.predict(X))
        same_sizes = sorted(np.bincount(m100.predict(X), minlength=k)) == sorted(np.bincount(ref_lab, minlength=k))
        near5_100 = np.unique(np.round(ll100, 1))
        near5_100 = near5_100[near5_100 > best100 - 5]
        conv_rows.append(dict(k=k, spec=f"100 starts, seed {seed}", n_starts=100,
                              best_logLik=best100,
                              starts_at_best=int((np.abs(ll100 - best100) < TOL).sum()),
                              share_at_best=(np.abs(ll100 - best100) < TOL).mean(),
                              starts_same_partition=np.nan,
                              n_distinct_maxima=len(np.unique(np.round(ll100, 2))),
                              n_distinct_maxima_within_5LL=len(near5_100),
                              worst_logLik=ll100.min(), median_logLik=np.median(ll100),
                              best_minus_reference=best100 - best,
                              ARI_vs_reference_solution=ari, same_class_sizes=same_sizes))
        log(f"   100 starts, seed {seed}: best LL {best100:.2f} (diff to reference {best100 - best:+.4f}); "
            f"{(np.abs(ll100 - best100) < TOL).sum()}/100 at best; distinct local maxima within 5 LL units "
            f"of the best: {len(near5_100)}; ARI vs reference solution {ari:.4f}; same class sizes: {same_sizes}")
conv = pd.DataFrame(conv_rows)
save(conv.round(4), "B_convergence_summary.csv", index=False)

# ============================================================================= C  assignment certainty
log("\n=== C  Class assignment certainty ===")
cert_rows, avepp_tables = [], {}
for k in (3, 4):
    m = models[k]
    pr = m.predict_proba(X)
    cl = sub[f"cls{k}"].values
    # reorder posterior columns to the ordered classes
    _, remap = ordered_classes(m, X, sub["digital_index"], k)
    inv = {v: kk for kk, v in remap.items()}
    pr_o = pr[:, [inv[i] for i in range(k)]]
    maxp = pr_o.max(axis=1)
    assert np.all(pr_o.argmax(axis=1) == cl)
    # AvePP matrix: mean posterior of class j among respondents assigned to class i
    avepp = pd.DataFrame([[pr_o[cl == i, j].mean() for j in range(k)] for i in range(k)],
                         index=[f"assigned: {LABELS[k][i]}" for i in range(k)],
                         columns=[f"P({LABELS[k][j]})" for j in range(k)])
    avepp_tables[k] = avepp
    save(avepp.round(3), f"C_avepp_matrix_k{k}.csv")
    for i in range(k):
        sel_i = cl == i
        cert_rows.append(dict(k=k, class_=LABELS[k][i], n=int(sel_i.sum()),
                              mean_max_posterior=maxp[sel_i].mean(),
                              median_max_posterior=np.median(maxp[sel_i]),
                              min_max_posterior=maxp[sel_i].min(),
                              share_ge_0_8=(maxp[sel_i] >= 0.8).mean(),
                              share_ge_0_9=(maxp[sel_i] >= 0.9).mean(),
                              share_lt_0_7=(maxp[sel_i] < 0.7).mean()))
    cert_rows.append(dict(k=k, class_="all", n=n, mean_max_posterior=maxp.mean(),
                          median_max_posterior=np.median(maxp), min_max_posterior=maxp.min(),
                          share_ge_0_8=(maxp >= 0.8).mean(), share_ge_0_9=(maxp >= 0.9).mean(),
                          share_lt_0_7=(maxp < 0.7).mean()))
    log(f"k = {k}: AvePP matrix\n{avepp.round(3).to_string()}")
cert = pd.DataFrame(cert_rows)
log(cert.round(3).to_string())
save(cert.round(4), "C_assignment_certainty.csv", index=False)

# ============================================================================= D  3 vs 4 classes
log("\n=== D  Cross-classification 3-class x 4-class and class profiles ===")
ct34 = pd.crosstab(sub["cls4"].map(dict(enumerate(LABELS[4]))),
                   sub["cls3"].map(dict(enumerate(LABELS[3]))))
ct34 = ct34.loc[LABELS[4], LABELS[3]]
log(ct34.to_string())
save(ct34, "D_crosstab_4class_x_3class.csv")


def profile(cl_series, labels):
    g = cl_series.map(dict(enumerate(labels)))
    out = pd.DataFrame({
        "n": g.value_counts(),
        "share_pct": g.value_counts() / n * 100,
        "digital_index_mean": sub["digital_index"].groupby(g).mean(),
        "ai_use_pct": sub.loc[defined, "ai_use"].groupby(g[defined]).mean() * 100,
        "ai_use_n_defined": g[defined].value_counts(),
        "provision_pct": sub["prov_yes"].groupby(g).mean() * 100,
        "trained_pct": sub["trained"].groupby(g).mean() * 100,
        "hospital_pct": sub["hospital"].groupby(g).mean() * 100,
        "ambulatory_pct": (sub["sector3"] == "ambulatory").groupby(g).mean() * 100,
    }).loc[labels]
    # composition of the six response options among the 19 x n item cells of each class
    for opt in OPTIONS:
        out[f"cells_{OPT_EN[opt]}_pct"] = [(raw_opts[cl_series == i] == opt).values.mean() * 100
                                           for i in range(len(labels))]
    out["cells_missing_pct"] = [raw_opts[cl_series == i].isna().values.mean() * 100 for i in range(len(labels))]
    return out


def chi2_by_class(cl_series, labels, mask_defined):
    res = {}
    for name, var, mask in [("ai_use", sub["ai_use"], mask_defined),
                            ("provision", sub["prov_yes"], pd.Series(True, index=sub.index)),
                            ("trained", sub["trained"], pd.Series(True, index=sub.index))]:
        ct = pd.crosstab(cl_series[mask], var[mask])
        chi2, p, dof, exp = stats.chi2_contingency(ct, correction=False)
        res[name] = dict(chi2=chi2, df=dof, p=p, n=int(mask.sum()), min_expected=exp.min())
    return pd.DataFrame(res).T


for k in (3, 4):
    prof = profile(sub[f"cls{k}"], LABELS[k])
    log(f"\nClass profiles, k = {k}:\n{prof.round(1).to_string()}")
    save(prof.round(2), f"D_profiles_k{k}.csv")
    tests = chi2_by_class(sub[f"cls{k}"], LABELS[k], defined)
    log(f"chi-square tests across classes, k = {k}:\n{tests.round(4).to_string()}")
    save(tests.round(5), f"D_chi2_k{k}.csv")
    heat = pd.DataFrame({LABELS[k][i]: raw_items[sub[f"cls{k}"] == i].mean() for i in range(k)})
    save(heat.round(2), f"D_item_means_k{k}.csv")
    # conditional response probabilities (item level x class) for the Supplement
    crp = []
    for i in range(k):
        for lvl in range(N_LEVELS):
            crp.append(pd.Series((X[sub[f"cls{k}"] == i] == lvl).mean().values,
                                 index=ITEM_LABELS, name=f"{LABELS[k][i]} | P({LEVEL_NAMES[lvl]})"))
    save(pd.concat(crp, axis=1).round(3), f"D_conditional_response_probs_k{k}.csv")

# --- D2: pairwise comparison of the four classes on the outcomes (which pairs differ?)
log("\nD2  Pairwise comparison of the four classes (k = 4) on AI use, employer provision, prior training")
sub_rows = []
for a, b in [(i, j) for i in range(4) for j in range(i + 1, 4)]:
    for name, var, mask in [("ai_use", sub["ai_use"], defined),
                            ("provision", sub["prov_yes"], pd.Series(True, index=sub.index)),
                            ("trained", sub["trained"], pd.Series(True, index=sub.index))]:
        mm = mask & sub["cls4"].isin([a, b])
        ct = pd.crosstab(sub.loc[mm, "cls4"], var[mm])
        chi2, p, dof, exp = stats.chi2_contingency(ct, correction=False)
        p_f = stats.fisher_exact(ct.values)[1]
        pa = sub.loc[mm & (sub["cls4"] == a), var.name].mean() * 100
        pb = sub.loc[mm & (sub["cls4"] == b), var.name].mean() * 100
        sub_rows.append(dict(class_a=LABELS[4][a], class_b=LABELS[4][b], outcome=name,
                             pct_a=pa, n_a=int((mm & (sub["cls4"] == a)).sum()),
                             pct_b=pb, n_b=int((mm & (sub["cls4"] == b)).sum()),
                             diff_pp=pb - pa, chi2=chi2, p_chi2=p, p_fisher=p_f))
d2 = pd.DataFrame(sub_rows)
log(d2.round(3).to_string(index=False))
save(d2.round(4), "D2_pairwise_class_comparison_k4.csv", index=False)

# ============================================================================= E  sector
log("\n=== E  Sector: class x sector, stratified class characteristics, regressions ===")
for k in (3, 4):
    g = sub[f"cls{k}"].map(dict(enumerate(LABELS[k])))
    ct = pd.crosstab(g, sub["sector3"]).loc[LABELS[k], ["hospital", "ambulatory", "other"]]
    ct["total"] = ct.sum(axis=1)
    rowpct = (ct[["hospital", "ambulatory", "other"]].div(ct["total"], axis=0) * 100).round(1)
    chi2, p, dof, _ = stats.chi2_contingency(ct[["hospital", "ambulatory"]], correction=False)
    log(f"\nk = {k}: class x sector (counts)\n{ct.to_string()}\nrow %:\n{rowpct.to_string()}")
    log(f"   chi-square hospital vs ambulatory x class: chi2({dof}) = {chi2:.1f}, p = {p:.2g}")
    out = ct.copy()
    for c in ["hospital", "ambulatory", "other"]:
        out[f"{c}_rowpct"] = rowpct[c]
    save(out, f"E_class_x_sector_k{k}.csv")

    # stratified characteristics
    strat_rows = []
    for stratum, mask_s in [("hospital", sub["hospital"] == 1), ("non-hospital", sub["hospital"] == 0),
                            ("ambulatory only", sub["sector3"] == "ambulatory")]:
        for i in range(k):
            mk = mask_s & (sub[f"cls{k}"] == i)
            md = mk & defined
            strat_rows.append(dict(k=k, stratum=stratum, class_=LABELS[k][i], n=int(mk.sum()),
                                   ai_use_pct=sub.loc[md, "ai_use"].mean() * 100, ai_use_n=int(md.sum()),
                                   provision_pct=sub.loc[mk, "prov_yes"].mean() * 100,
                                   trained_pct=sub.loc[mk, "trained"].mean() * 100,
                                   digital_index_mean=sub.loc[mk, "digital_index"].mean()))
        # chi-square within stratum
        for name, var, mask in [("ai_use", sub["ai_use"], defined),
                                ("provision", sub["prov_yes"], pd.Series(True, index=sub.index)),
                                ("trained", sub["trained"], pd.Series(True, index=sub.index))]:
            mm = mask & mask_s
            ct_s = pd.crosstab(sub.loc[mm, f"cls{k}"], var[mm])
            if ct_s.shape[0] > 1 and ct_s.shape[1] > 1:
                chi2, p, dof, exp = stats.chi2_contingency(ct_s, correction=False)
                strat_rows.append(dict(k=k, stratum=stratum, class_=f"chi2 test: {name}", n=int(mm.sum()),
                                       ai_use_pct=np.nan, ai_use_n=np.nan, provision_pct=np.nan,
                                       trained_pct=np.nan, digital_index_mean=np.nan,
                                       chi2=chi2, df=dof, p=p, min_expected=exp.min()))
    strat = pd.DataFrame(strat_rows)
    log(f"\nk = {k}: class characteristics within sector strata\n"
        f"{strat.drop(columns=['k']).round(3).to_string(index=False)}")
    save(strat.round(4), f"E_stratified_k{k}.csv", index=False)

    # logistic regressions: outcome ~ class + hospital, and interaction (LR tests)
    reg_rows = []
    for outcome, mask in [("ai_use", defined), ("prov_yes", pd.Series(True, index=sub.index)),
                          ("trained", pd.Series(True, index=sub.index))]:
        d = sub.loc[mask, [outcome, f"cls{k}", "hospital"]].dropna().copy()
        d["cls"] = d[f"cls{k}"].astype(int)
        d[outcome] = d[outcome].astype(int)
        m0 = smf.logit(f"{outcome} ~ hospital", d).fit(disp=0)
        m1 = smf.logit(f"{outcome} ~ C(cls) + hospital", d).fit(disp=0)
        m1b = smf.logit(f"{outcome} ~ C(cls)", d).fit(disp=0)
        try:
            m2 = smf.logit(f"{outcome} ~ C(cls) * hospital", d).fit(disp=0, maxiter=200)
            lr_int = 2 * (m2.llf - m1.llf)
            p_int = stats.chi2.sf(lr_int, k - 1)
            conv2 = m2.mle_retvals.get("converged", True)
        except Exception as e:  # separation etc.
            lr_int, p_int, conv2 = np.nan, np.nan, False
        lr_cls = 2 * (m1.llf - m0.llf)
        lr_sec = 2 * (m1.llf - m1b.llf)
        ors = np.exp(m1.params)
        ci = np.exp(m1.conf_int())
        row = dict(k=k, outcome=outcome, n=len(d),
                   LR_class_adj_sector=lr_cls, df_class=k - 1, p_class=stats.chi2.sf(lr_cls, k - 1),
                   LR_sector_adj_class=lr_sec, p_sector=stats.chi2.sf(lr_sec, 1),
                   OR_hospital=ors["hospital"], OR_hospital_ci=f"{ci.loc['hospital', 0]:.2f}-{ci.loc['hospital', 1]:.2f}",
                   LR_interaction=lr_int, df_interaction=k - 1, p_interaction=p_int,
                   interaction_model_converged=conv2)
        for i in range(1, k):
            key = f"C(cls)[T.{i}]"
            row[f"OR_{LABELS[k][i]}_vs_{LABELS[k][0]}"] = ors[key]
            row[f"CI_{LABELS[k][i]}"] = f"{ci.loc[key, 0]:.2f}-{ci.loc[key, 1]:.2f}"
        reg_rows.append(row)
    reg = pd.DataFrame(reg_rows)
    log(f"\nk = {k}: logistic regressions (reference class = {LABELS[k][0]})\n"
        f"{reg.drop(columns=['k']).round(4).to_string(index=False)}")
    save(reg.round(5), f"E_regressions_k{k}.csv", index=False)

# ============================================================================= H  Figure 4
log("\n=== H  Figure 4 drafts ===")


SHORT_TICKS = {   # abbreviated axis labels for the four-class figures
    "Core set, routine use (ambulatory profile)": "Core set,\nroutine use\n(ambulatory)",
    "Core set, routine use (hospital profile)": "Core set,\nroutine use\n(hospital)",
    "Core set, others not available": "Core set,\nothers not\navailable",
    "Core set, others unfamiliar": "Core set,\nothers\nunfamiliar",
}


def tick(l):
    if l in SHORT_TICKS:
        return SHORT_TICKS[l]
    return l.replace(", ", ",\n").replace(" (", "\n(")


def make_fig4(k, fname, sector_panel=False):
    labels = LABELS[k]
    prof = profile(sub[f"cls{k}"], labels)
    heat = np.array([[raw_items.loc[sub[f"cls{k}"] == i, c].mean() for i in range(k)]
                     for c in ITEM_LABELS])
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    if sector_panel:
        fig = plt.figure(figsize=(7.09 if k == 3 else 8.8, 8.2 if k == 3 else 8.6))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.15 if k == 3 else 1.25, 1], height_ratios=[1, 1, 1],
                              wspace=0.55 if k == 3 else 0.45, hspace=0.75)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, 1])
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.09 if k == 3 else 9.4, 5.8),
                                       gridspec_kw={"width_ratios": [1.15 if k == 3 else 1.35, 1]})
    ax1.imshow(heat, cmap="Blues", vmin=0, vmax=4, aspect="auto")
    xt = [tick(l) for l in labels]
    ax1.set_xticks(range(k), xt, fontsize=6.5 if k == 4 else 7)
    ax1.set_yticks(range(N_ITEMS), ITEM_LABELS)
    for i in range(N_ITEMS):
        for j in range(k):
            ax1.text(j, i, f"{heat[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                     color="white" if heat[i, j] >= 2 else "#0C447C")
    ax1.text(-0.62, 1.01, f"a  Mean use frequency of {N_ITEMS} digital applications (0-4)",
             transform=ax1.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold")
    ax1.spines[["top", "right", "left", "bottom"]].set_visible(False)
    x = np.arange(k)
    width = 0.26
    for i, (lab, col_) in enumerate([("AI use*", "ai_use_pct"), ("Employer provision", "provision_pct"),
                                     ("Prior AI training", "trained_pct")]):
        ax2.bar(x + (i - 1) * width, prof[col_], width, label=lab,
                color=["#185FA5", "#378ADD", "#B5D4F4"][i])
    ax2.set_xticks(x, [f"{xt[i]}\nn = {int(prof.loc[labels[i], 'n'])}" for i in range(k)],
                   fontsize=6 if k == 4 else 6.5)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("%")
    ax2.set_title("b  Profile characteristics", loc="left", fontsize=9, fontweight="bold")
    ax2.legend(frameon=False, fontsize=7)
    if sector_panel:
        strat = pd.read_csv(os.path.join(OUTDIR, f"E_stratified_k{k}.csv"))
        strat = strat[~strat["class_"].str.startswith("chi2")]
        w2 = 0.36
        for i, (stratum, col_) in enumerate([("hospital", "#185FA5"), ("non-hospital", "#9DC4EA")]):
            s = strat[strat["stratum"] == stratum].set_index("class_").loc[labels]
            bars = ax3.bar(x + (i - 0.5) * w2, s["ai_use_pct"], w2, label=stratum.capitalize(), color=col_)
            for b, nn in zip(bars, s["ai_use_n"]):
                ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"n={int(nn)}",
                         ha="center", va="bottom", fontsize=5.5, color="#333333")
        ax3.set_xticks(x, xt, fontsize=6.5)
        ax3.set_ylim(0, 100)
        ax3.set_ylabel("% using AI*")
        ax3.set_title("c  AI use by profile and sector", loc="left", fontsize=9, fontweight="bold")
        ax3.legend(frameon=False, fontsize=7, loc="upper right")
        # panel d: AI use by profile and number of surveyed applications the respondent did not recognise
        n_un = (raw_opts == "unklar, was das ist").sum(axis=1)
        cat = pd.cut(n_un, [-1, 0, 2, 4, 99], labels=["0", "1-2", "3-4", "5+"])
        w4 = 0.2
        cols4 = ["#D6E6F7", "#9DC4EA", "#378ADD", "#0C447C"]   # sequential blues, same family as panels a-c
        drows = []
        for ci, (lev, col_) in enumerate(zip(["0", "1-2", "3-4", "5+"], cols4)):
            vals, ns = [], []
            for i in range(k):
                m = defined & (sub[f"cls{k}"] == i) & (cat == lev)
                ns.append(int(m.sum()))
                vals.append(sub.loc[m, "ai_use"].mean() * 100 if m.sum() >= 5 else np.nan)
                drows.append(dict(profile=labels[i], unfamiliar=lev, n=int(m.sum()),
                                  ai_use_pct=sub.loc[m, "ai_use"].mean() * 100 if m.sum() else np.nan))
            bars = ax4.bar(x + (ci - 1.5) * w4, [0 if np.isnan(v) else v for v in vals], w4, label=lev, color=col_)
            for b, nn, v in zip(bars, ns, vals):
                if not np.isnan(v):
                    ax4.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{nn}", ha="center", va="bottom", fontsize=5, color="#333333")
        ax4.set_xticks(x, xt, fontsize=6.5)
        ax4.set_ylim(0, 100)
        ax4.set_yticks([0, 20, 40, 60, 80, 100])
        ax4.set_ylabel("% using AI*")
        ax4.set_title("d  AI use by profile and unfamiliarity", loc="left", fontsize=9, fontweight="bold")
        ax4.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4,
                   title="number of surveyed applications not recognised", title_fontsize=6.5)
        pd.DataFrame(drows).round(2).to_csv(os.path.join(OUTDIR, f"H_ai_use_by_profile_x_unfamiliar_k{k}.csv"), index=False)
    else:
        fig.tight_layout(w_pad=2.5 if k == 4 else 2.0)
    fig.savefig(os.path.join(OUTDIR, fname + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, fname + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"  -> written {fname}.png/.pdf")


def make_composition_fig(k, fname):
    """Stacked bars: share of the six response options (and missing) among the 19 x n item cells per class."""
    labels = LABELS[k]
    prof = profile(sub[f"cls{k}"], labels)
    cols = [f"cells_{OPT_EN[o]}_pct" for o in OPTIONS] + ["cells_missing_pct"]
    names = ["daily", "weekly", "monthly", "rarely", "not available", "unclear what this is", "missing"]
    colors = ["#0C447C", "#185FA5", "#378ADD", "#9DC4EA", "#E0A458", "#B8336A", "#BBBBBB"]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.6 + 0.9 * k, 3.6))
    bottom = np.zeros(k)
    for c, nm, col in zip(cols, names, colors):
        vals = prof[c].values
        ax.bar(range(k), vals, 0.6, bottom=bottom, label=nm, color=col)
        for i, v in enumerate(vals):
            if v >= 6:
                ax.text(i, bottom[i] + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if col not in ("#9DC4EA", "#BBBBBB", "#E0A458") else "#222222")
        bottom += vals
    ax.set_xticks(range(k), [f"{tick(l)}\nn = {int(prof.loc[l, 'n'])}" for l in labels], fontsize=6.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"% of item responses ({N_ITEMS} items x respondents)")
    ax.legend(frameon=False, fontsize=6.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, fname + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"  -> written {fname}.png/.pdf")


if CODING == "primary":                               # three-level coding (19-item model)
    make_fig4(3, "fig_3level_k3", sector_panel=True)
    make_fig4(3, "fig_3level_k3_2panel")
    make_fig4(4, "fig_3level_k4")
else:                                                 # four-level coding; with --drop 12,15,16 --starts 100 these are Fig. 2 and Figure S8-1
    make_fig4(3, "fig2", sector_panel=True)
    make_fig4(4, "figS8_1", sector_panel=True)
    make_fig4(4, "figS8_1_2panel")
make_composition_fig(3, "fig_response_composition_k3")
make_composition_fig(4, "fig_response_composition_k4")

# ============================================================================= J  comparison with the primary coding
if CODING == "four":
    log("\n=== J  Comparison with the primary (three-level) coding ===")
    Xp = df.loc[keep, ITEM_COLS].astype(int)
    prim = {}
    for k in (3, 4):
        mp = fit(Xp, k)
        prim[k], _ = ordered_classes(mp, Xp, sub["digital_index"], k)
    PRIM_LABELS = LABELS_BY_CODING["primary"]
    assert sorted(prim[3].value_counts().tolist()) == [62, 113, 181]
    for k in (3, 4):
        for kp in (3, 4):
            ct = pd.crosstab(sub[f"cls{k}"].map(dict(enumerate(LABELS[k]))),
                             prim[kp].map(dict(enumerate(PRIM_LABELS[kp]))))
            ct = ct.loc[LABELS[k], PRIM_LABELS[kp]]
            ari = adjusted_rand_score(sub[f"cls{k}"], prim[kp])
            log(f"\nfour-level k = {k} (rows) x primary k = {kp} (columns); ARI = {ari:.3f}\n{ct.to_string()}")
            save(ct, f"J_crosstab_four_k{k}_x_primary_k{kp}.csv")
    # where do the members of the primary core-set class go under the four-level coding (k = 4)?
    core = prim[3] == 2
    dest = sub.loc[core, "cls4"].map(dict(enumerate(LABELS[4])))
    log("\nPrimary core-set class (n = %d) under the four-level coding, k = 4:" % int(core.sum()))
    log(dest.value_counts().to_string())
    comp = profile(sub["cls4"], LABELS[4])
    log("\nFour-level k = 4 classes: composition of the item responses (% of cells) and outcomes:\n"
        + comp[["n", "ai_use_pct", "provision_pct", "trained_pct", "hospital_pct",
                "cells_not_available_pct", "cells_unclear_pct", "cells_rarely_pct", "cells_daily_pct"]].round(1).to_string())
    # per-respondent counts of "not available" and "unclear" answers by class
    na_n = (raw_opts == "möchte ich, aber nicht verfügbar").sum(axis=1)
    un_n = (raw_opts == "unklar, was das ist").sum(axis=1)
    per = pd.DataFrame({"class": sub["cls4"].map(dict(enumerate(LABELS[4]))), "n_not_available": na_n, "n_unclear": un_n})
    summ = per.groupby("class").agg(n=("n_unclear", "size"),
                                    not_available_mean=("n_not_available", "mean"),
                                    not_available_median=("n_not_available", "median"),
                                    unclear_mean=("n_unclear", "mean"),
                                    unclear_median=("n_unclear", "median"),
                                    unclear_ge5_pct=("n_unclear", lambda v: (v >= 5).mean() * 100),
                                    unclear_0_pct=("n_unclear", lambda v: (v == 0).mean() * 100)).loc[LABELS[4]]
    log("\nPer-respondent counts of 'not available' and 'unclear' answers (of 19 items) by four-level class, k = 4:\n"
        + summ.round(2).to_string())
    save(summ.round(3), "J_per_respondent_counts_k4.csv")
    # regression: AI use on the two counts (continuous), adjusted for sector
    d = sub.loc[defined, ["ai_use", "hospital"]].copy()
    d["n_na"] = na_n[defined]
    d["n_un"] = un_n[defined]
    d["ai_use"] = d["ai_use"].astype(int)
    mr = smf.logit("ai_use ~ n_na + n_un + hospital", d).fit(disp=0)
    ors = np.exp(mr.params)
    ci = np.exp(mr.conf_int())
    log("\nLogistic regression of AI use on the number of 'not available' and 'unclear' answers (per item) and sector (n = %d):" % len(d))
    for v in ["n_na", "n_un", "hospital"]:
        log(f"   {v}: OR {ors[v]:.3f} (95% CI {ci.loc[v, 0]:.3f}-{ci.loc[v, 1]:.3f}), p = {mr.pvalues[v]:.3g}")
    pd.DataFrame({"OR": ors, "ci_low": ci[0], "ci_high": ci[1], "p": mr.pvalues}).round(4).to_csv(
        os.path.join(OUTDIR, "J_regression_ai_use_on_counts.csv"))

# ============================================================================= I  checksums
log("\n=== I  Checksums against the article / Supplementary Note 8 ===")
checks_3level19 = {
    "LCA n": (n, 356),
    "k=3 class sizes": (sel.loc[3, "class_sizes"], "181/113/62"),
    "k=4 class sizes": (sel.loc[4, "class_sizes"], "111/99/85/61"),
    "k=5 class sizes": (sel.loc[5, "class_sizes"], "92/81/80/56/47"),
    "BIC k=2..5 (rounded)": (tuple(int(round(v)) for v in sel.loc[2:5, "BIC"]), (11694, 11601, 11550, 11588)),
    "entropy k=2..5 (3 dp)": (tuple(round(v, 3) for v in sel.loc[2:5, "entropy"]), (0.929, 0.931, 0.915, 0.921)),
    "k=3 profile n (ordered by index)": (tuple(int(v) for v in profile(sub['cls3'], LABELS[3])['n']), (62, 113, 181)),
    "k=3 AI use % (ordered)": (tuple(round(v, 1) for v in profile(sub['cls3'], LABELS[3])['ai_use_pct']), (82.0, 51.9, 49.4)),
    "k=3 provision % (ordered)": (tuple(round(v, 1) for v in profile(sub['cls3'], LABELS[3])['provision_pct']), (59.7, 20.4, 23.8)),
    "k=3 trained % (ordered)": (tuple(round(v, 1) for v in profile(sub['cls3'], LABELS[3])['trained_pct']), (25.8, 12.4, 17.1)),
    "k=4 profile n (ordered by index)": (tuple(int(v) for v in profile(sub['cls4'], LABELS[4])['n']), (61, 111, 85, 99)),
    "k=4 AI use % (ordered)": (tuple(round(v, 1) for v in profile(sub['cls4'], LABELS[4])['ai_use_pct']), (82.0, 51.9, 42.9, 54.8)),
    "4x3 crosstab diagonal (Note 8)": (
        (int(ct34.iloc[0, 0]), int(ct34.iloc[1, 1]), int(ct34.iloc[2, 2]), int(ct34.iloc[3, 2])), (59, 108, 81, 96)),
    "100-start runs reproduce the reference maximum (k = 3, 4; three seeds)": (
        bool((conv["best_minus_reference"].abs().dropna() < 1e-3).all()
             and (conv["ARI_vs_reference_solution"].dropna() > 0.999).all()), True),
}
# Reproduction checks for the four-level coding (values pinned from the first run of 24.10.2026;
# there is no manuscript reference for this coding yet)
checks_4level19 = {
    "LCA n": (n, 356),
    "k=3 class sizes": (sel.loc[3, "class_sizes"], "166/115/75"),
    "k=4 class sizes": (sel.loc[4, "class_sizes"], "121/101/83/51"),
    "BIC k=2..5 (rounded)": (tuple(int(round(v)) for v in sel.loc[2:5, "BIC"]), (14482, 14344, 14348, 14468)),
    "k=3 AI use % (ordered)": (tuple(round(v, 1) for v in profile(sub['cls3'], LABELS[3])['ai_use_pct']), (61.0, 60.4, 36.9)),
    "100-start runs reproduce the reference maximum (k = 3, 4; three seeds)": (
        bool((conv["best_minus_reference"].abs().dropna() < 1e-3).all()
             and (conv["ARI_vs_reference_solution"].dropna() > 0.999).all()), True),
}
# Primary model: four-level coding, 16 items (without tele-rounds, digital
# pre-registration, triage tools), 100 starts, three classes. Values pinned from run lca_19items_4levels16.
checks_primary = {
    "LCA n": (n, 356),
    "k=3 class sizes": (sel.loc[3, "class_sizes"], "165/122/69"),
    "k=4 class sizes": (sel.loc[4, "class_sizes"], "119/89/87/61"),
    "BIC k=2..5 (rounded)": (tuple(int(round(v)) for v in sel.loc[2:5, "BIC"]), (12319, 12250, 12259, 12341)),
    "entropy k=2..5 (3 dp)": (tuple(round(v, 3) for v in sel.loc[2:5, "entropy"]), (0.883, 0.902, 0.911, 0.916)),
    "k=3 profile n (ordered by index)": (tuple(int(v) for v in profile(sub['cls3'], LABELS[3])['n']), (69, 122, 165)),
    "k=3 AI use % (ordered)": (tuple(round(v, 1) for v in profile(sub['cls3'], LABELS[3])['ai_use_pct']), (83.1, 54.9, 45.9)),
    "k=3 provision % (ordered)": (tuple(round(v, 1) for v in profile(sub['cls3'], LABELS[3])['provision_pct']), (56.5, 24.6, 20.6)),
    "k=3 trained % (ordered)": (tuple(round(v, 1) for v in profile(sub['cls3'], LABELS[3])['trained_pct']), (26.1, 15.6, 14.5)),
    "k=4 AI use % (ordered)": (tuple(round(v, 1) for v in profile(sub['cls4'], LABELS[4])['ai_use_pct']), (85.0, 55.9, 41.2, 50.0)),
    "100-start runs reproduce the reference maximum (k = 3, 4; three seeds)": (
        bool((conv["best_minus_reference"].abs().dropna() < 1e-3).all()
             and (conv["ARI_vs_reference_solution"].dropna() > 0.999).all()), True),
}
checks = checks_3level19 if CODING == "primary" else checks_4level19
if DROP:
    if CODING == "four" and sorted(DROP) == [12, 15, 16] and REF_STARTS == 100:
        checks = checks_primary
    else:   # no pinned values for other reduced item sets: reproduction checks only
        checks = {"LCA n": (n, 356),
                  "100-start runs reproduce the reference maximum (k = 3, 4; three seeds)": checks["100-start runs reproduce the reference maximum (k = 3, 4; three seeds)"]}
ok = True
for name, (obs, exp) in checks.items():
    good = obs == exp
    ok &= bool(good)
    log(f"{'OK ' if good else 'FAIL'} {name}: {obs}" + ("" if good else f"  (expected {exp})"))
assert ok, "CHECKSUM VIOLATED"
log(f"\nAll checksums OK (sections A-F, H). Runtime so far {time.time() - t0:.0f} s. Outputs in {OUTDIR}")

# ============================================================================= G  BLRT (optional; run last because it is slow)
if N_BLRT > 0:
    log(f"\n=== G  Parametric bootstrap LRT, {N_BLRT} replications, {BLRT_STARTS} starts per bootstrap fit ===")
    rng = np.random.RandomState(REF_SEED)
    blrt_rows = []
    for k in (2, 3, 4):
        null, alt = models[k], models[k + 1]
        real_stat = 2 * (alt.score(X) - null.score(X)) * n
        sampler = copy.deepcopy(null)
        sampler.set_params(random_state=REF_SEED)
        gen = []
        t1 = time.time()
        for b in range(N_BLRT):
            Xb, _, _ = sampler.sample(n)
            seed_b = int(rng.randint(1, 10**6))
            m_null = fit(Xb, k, n_init=BLRT_STARTS, seed=seed_b)
            m_alt = fit(Xb, k + 1, n_init=BLRT_STARTS, seed=seed_b)
            gen.append(2 * (m_alt.score(Xb) - m_null.score(Xb)) * n)
        gen = np.array(gen)
        p_b = (np.sum(gen >= real_stat) + 1) / (N_BLRT + 1)
        blrt_rows.append(dict(test=f"{k} vs {k + 1} classes", observed_2LLdiff=real_stat,
                              bootstrap_null_mean=gen.mean(), bootstrap_null_p95=np.percentile(gen, 95),
                              bootstrap_null_max=gen.max(), n_reps=N_BLRT, p_value=p_b,
                              seconds=time.time() - t1))
        log(f"  {k} vs {k + 1}: observed 2*dLL = {real_stat:.1f}; null mean {gen.mean():.1f}, "
            f"95th pct {np.percentile(gen, 95):.1f}, max {gen.max():.1f}; p = {p_b:.4f} "
            f"({time.time() - t1:.0f} s)")
        pd.Series(gen).to_csv(os.path.join(OUTDIR, f"G_blrt_null_{k}v{k + 1}.csv"), index=False)
    save(pd.DataFrame(blrt_rows).round(4), "G_blrt.csv", index=False)

log(f"Finished. Total runtime {time.time() - t0:.0f} s.")
LOG.close()
