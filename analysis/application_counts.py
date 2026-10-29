"""
application_counts.py  --  the LCA scripts: person-level measures accompanying the primary LCA ()

Item set and coding as in the primary model (16 items; tele-rounds, digital pre-registration and
triage tools excluded). All counts refer to these 16 items.

  1  Per-respondent counts: applications reported as "unclear what this is" (not recognised) and as
     "would like to, but not available"; mean use frequency among the remaining items
  2  AI use by number of applications not recognised (0 / 1-2 / 3-4 / 5+), overall, by sector and within
     the three use profiles of the primary model; trend tests
  3  Logistic regression of AI use: Table 2 covariates with the digital index replaced by its three components
     (number not available, number not recognised, mean use frequency among the rest); complete cases n = 322
  4  Employer provision x prior AI-specific training (direct cross-tabulation, full sample n = 362)
  5  Clinical decision support and speech recognition: "unclear" rates by sector and AI use by familiarity
  6  Correlates of the counts (self-rated professional digital competence, adoption group), descriptive only

Usage: python application_counts.py <analysis_dataset.csv> <outdir>
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from stepmix import StepMix

warnings.filterwarnings("ignore")
INFILE, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
LOG = open(os.path.join(OUTDIR, "application_counts_log.txt"), "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def save(df, name, **kw):
    df.to_csv(os.path.join(OUTDIR, name), **kw)


ITEM_COLS = [f"lca_item_{i:02d}" for i in range(1, 20)]
ITEM_LABELS = ["Practice/hospital IS", "Electronic patient record (ePA)", "e-medication list",
               "e-sick note (eAU)", "PACS / imaging", "Laboratory IS", "Clinical decision support",
               "DiGA", "Health apps", "Phone assistance", "Video consultation", "Tele-rounds",
               "Tele-/home monitoring", "Speech recognition", "Digital pre-registration",
               "Triage tools", "Medication safety (AMTS)", "Automated duty scheduling",
               "Knowledge management"]
DROP = ["Tele-rounds", "Digital pre-registration", "Triage tools"]
USED = [l for l in ITEM_LABELS if l not in DROP]
NA, UN = "möchte ich, aber nicht verfügbar", "unklar, was das ist"
FREQ = {"täglich": 4, "wöchentlich": 3, "monatlich": 2, "seltener": 1}
MAP4 = {NA: 0, UN: 1, "seltener": 2, "monatlich": 2, "wöchentlich": 3, "täglich": 3}
LAB3 = ["Wide range, routine use", "Wide range, occasional use", "Core set, routine use"]

df = pd.read_csv(INFILE)
rawall = df[df.columns[18:37]].copy()
rawall.columns = ITEM_LABELS
raw16 = rawall[USED]
valid19 = rawall.notna().sum(axis=1)
df["n_unclear16"] = (raw16 == UN).sum(axis=1)
df["n_not_available16"] = (raw16 == NA).sum(axis=1)
df["use_known16"] = raw16.apply(lambda c: c.map(FREQ)).mean(axis=1)
df["n_freq16"] = raw16.apply(lambda c: c.map(FREQ)).notna().sum(axis=1)
df.loc[valid19 < 15, ["n_unclear16", "n_not_available16", "use_known16"]] = np.nan
df["sector"] = np.where(df["ambulatory"] == 1, "ambulatory", np.where(df["ambulatory"] == 0, "hospital", "other"))
df["hospital"] = (df["ambulatory"] == 0).astype(int)
df["prov_yes"] = (df["provision"] == "yes").astype(int)

# primary classes (16 items, four levels, 100 starts, seed 42)
keep = df[ITEM_COLS].notna().all(axis=1)
X = raw16[keep].apply(lambda c: c.map(MAP4)).fillna(0).astype(int)
m3 = StepMix(n_components=3, measurement="categorical", n_init=100, random_state=42, verbose=0, progress_bar=False).fit(X)
r = pd.Series(m3.predict(X), index=X.index)
order = df.loc[keep, "digital_index"].groupby(r).mean().sort_values(ascending=False).index
df["profile"] = r.map({c: i for i, c in enumerate(order)}).map(dict(enumerate(LAB3)))
assert df["profile"].value_counts().tolist() == [165, 122, 69], df["profile"].value_counts()
lca = df[keep].copy()
defined = lca["ai_use"].notna()
log(f"LCA sample n = {len(lca)}; profiles {lca['profile'].value_counts().to_dict()}")

# ---------------------------------------------------------------- 1 counts
log("\n=== 1  Per-respondent counts over the 16 items (LCA sample) ===")
for v in ("n_unclear16", "n_not_available16", "use_known16"):
    s_ = lca[v]
    log(f"{v}: mean {s_.mean():.2f}, median {s_.median():.1f}, IQR {s_.quantile(.25):.1f}-{s_.quantile(.75):.1f}, max {s_.max():.1f}")
log("not recognised >= 1: %.1f%%; >= 3: %.1f%%; >= 5: %.1f%%" % ((lca.n_unclear16 >= 1).mean() * 100, (lca.n_unclear16 >= 3).mean() * 100, (lca.n_unclear16 >= 5).mean() * 100))
log("not available >= 1: %.1f%%; >= 5: %.1f%%; >= 8: %.1f%%" % ((lca.n_not_available16 >= 1).mean() * 100, (lca.n_not_available16 >= 5).mean() * 100, (lca.n_not_available16 >= 8).mean() * 100))
rho1, p1 = stats.spearmanr(lca.n_unclear16, lca.n_not_available16)
log(f"Spearman not recognised vs not available: rho = {rho1:.3f}, p = {p1:.2g}")
cnt = lca.groupby("profile").agg(n=("n_unclear16", "size"), unclear_mean=("n_unclear16", "mean"), unclear_median=("n_unclear16", "median"),
                                 unclear_ge5_pct=("n_unclear16", lambda v: (v >= 5).mean() * 100),
                                 not_avail_mean=("n_not_available16", "mean"), not_avail_median=("n_not_available16", "median"),
                                 use_known_mean=("use_known16", "mean")).loc[LAB3]
log(cnt.round(2).to_string())
save(cnt.round(3), "1_counts_by_profile.csv")

# ---------------------------------------------------------------- 2 gradient
log("\n=== 2  AI use by number of applications not recognised ===")
lca["un_cat"] = pd.cut(lca["n_unclear16"], [-1, 0, 2, 4, 99], labels=["0", "1-2", "3-4", "5+"])
dd = lca[defined].copy()
dd["ai_use"] = dd.ai_use.astype(int)
t = dd.groupby("un_cat", observed=True).ai_use.agg(["size", "mean"])
t["mean"] *= 100
log("overall:\n" + t.round(1).to_string())
save(t.round(2), "2_ai_use_by_unfamiliar_overall.csv")
pt = smf.logit("ai_use ~ n_unclear16", dd).fit(disp=0)
log(f"per-application OR (unadjusted): {np.exp(pt.params['n_unclear16']):.3f} ({np.exp(pt.conf_int().loc['n_unclear16', 0]):.3f}-{np.exp(pt.conf_int().loc['n_unclear16', 1]):.3f}), p = {pt.pvalues['n_unclear16']:.2g}")
pc = smf.logit("ai_use ~ cat_i", dd.assign(cat_i=dd["un_cat"].cat.codes)).fit(disp=0)
log(f"trend across the four categories (logit on category index): p = {pc.pvalues['cat_i']:.4f}")
for sec in ("hospital", "ambulatory"):
    ts = dd[dd.sector == sec].groupby("un_cat", observed=True).ai_use.agg(["size", "mean"])
    ts["mean"] *= 100
    log(f"{sec}:\n" + ts.round(1).to_string())
tp = dd.groupby(["profile", "un_cat"], observed=True).ai_use.agg(["size", "mean"])
tp["mean"] *= 100
log("by profile:\n" + tp.round(1).to_string())
save(tp.round(2), "2_ai_use_by_profile_x_unfamiliar.csv")
core = dd[dd.profile == "Core set, routine use"]
pcore = smf.logit("ai_use ~ n_unclear16 + hospital", core).fit(disp=0)
log(f"within the core-set profile (n = {len(core)}): OR per application not recognised {np.exp(pcore.params['n_unclear16']):.3f} (unrounded {np.exp(pcore.params['n_unclear16']):.6f}) "
    f"({np.exp(pcore.conf_int().loc['n_unclear16', 0]):.3f}-{np.exp(pcore.conf_int().loc['n_unclear16', 1]):.3f}), p = {pcore.pvalues['n_unclear16']:.3g}, adjusted for sector")
ppool = smf.logit("ai_use ~ n_unclear16 + n_not_available16 + hospital + C(profile)", dd).fit(disp=0)
log(f"all profiles, adjusted for profile and sector: not recognised OR {np.exp(ppool.params['n_unclear16']):.3f} "
    f"({np.exp(ppool.conf_int().loc['n_unclear16', 0]):.3f}-{np.exp(ppool.conf_int().loc['n_unclear16', 1]):.3f}), p = {ppool.pvalues['n_unclear16']:.3g}; "
    f"not available OR {np.exp(ppool.params['n_not_available16']):.3f} ({np.exp(ppool.conf_int().loc['n_not_available16', 0]):.3f}-{np.exp(ppool.conf_int().loc['n_not_available16', 1]):.3f}), p = {ppool.pvalues['n_not_available16']:.3g}")

# ---------------------------------------------------------------- 3 regression
log("\n=== 3  Table 2 model with the digital index decomposed (16-item counts), complete cases ===")
d = df.dropna(subset=["ai_use", "age4", "competence_prof", "ambulatory", "digital_index", "n_unclear16"]).copy()
d["ai_use"] = d.ai_use.astype(int)
d["ambulatory"] = d.ambulatory.astype(int)
d["z_index"] = (d["digital_index"] - d["digital_index"].mean()) / d["digital_index"].std(ddof=0)
cov = "C(age4, Treatment('<40')) + male + competence_prof + trained + ambulatory + C(provision, Treatment('no'))"
models = {"Table 2 main model (19-item index)": cov + " + z_index",
          "decomposed: not available + not recognised + use among the rest (16 items)": cov + " + n_not_available16 + n_unclear16 + use_known16",
          "decomposed without use among the rest": cov + " + n_not_available16 + n_unclear16"}
rows = []
for name, f in models.items():
    r_ = smf.logit("ai_use ~ " + f, d).fit(disp=0)
    ors, ci, pv = np.exp(r_.params), np.exp(r_.conf_int()), r_.pvalues
    log(f"\n{name}: n = {len(d)}, pseudo-R2 {r_.prsquared:.3f}, LL {r_.llf:.1f}")
    for v in r_.params.index:
        if v == "Intercept":
            continue
        log(f"   {v:48s} OR {ors[v]:6.3f} ({ci.loc[v, 0]:.3f}-{ci.loc[v, 1]:.3f}) p = {pv[v]:.3g}")
        rows.append(dict(model=name, term=v, OR=ors[v], ci_low=ci.loc[v, 0], ci_high=ci.loc[v, 1], p=pv[v], n=len(d)))
    if "decomposed" in name:
        r0 = smf.logit("ai_use ~ " + cov + " + " + " + ".join([t for t in f.split(" + ") if t.startswith(("n_not", "use_known"))]) if any(t.startswith(("n_not", "use_known")) for t in f.split(" + ")) else "ai_use ~ " + cov, d).fit(disp=0)
        lr = 2 * (r_.llf - r0.llf)
        log(f"   LR test for the number not recognised: chi2(1) = {lr:.2f}, p = {stats.chi2.sf(lr, 1):.4f}")
        d["z_un"] = (d.n_unclear16 - d.n_unclear16.mean()) / d.n_unclear16.std(ddof=0)
        rz = smf.logit("ai_use ~ " + f.replace("n_unclear16", "z_un"), d).fit(disp=0)
        log(f"   per-SD effect of the number not recognised: OR {np.exp(rz.params['z_un']):.3f} ({np.exp(rz.conf_int().loc['z_un', 0]):.3f}-{np.exp(rz.conf_int().loc['z_un', 1]):.3f}); SD = {d.n_unclear16.std(ddof=0):.2f}")
save(pd.DataFrame(rows).round(4), "3_regression_decomposed.csv", index=False)

# ---------------------------------------------------------------- 4 provision x training
log("\n=== 4  Employer provision x prior AI-specific training (full sample n = %d) ===" % len(df))
ct = pd.crosstab(df["provision"], df["trained"], margins=True)
log(ct.to_string())
for lvl in ("yes", "no", "dont_know"):
    m = df.provision == lvl
    log(f"provision = {lvl}: n = {int(m.sum())}, trained {df.loc[m, 'trained'].mean() * 100:.1f}% ({int(df.loc[m, 'trained'].sum())}), never trained {100 - df.loc[m, 'trained'].mean() * 100:.1f}%")
chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(df["provision"], df["trained"]), correction=False)
log(f"chi2({dof}) = {chi2:.1f}, p = {p:.2g}")
u = df[df.ai_use == 1]
for lvl in ("yes", "no"):
    m = u.provision == lvl
    log(f"AI users with provision = {lvl}: n = {int(m.sum())}, trained {u.loc[m, 'trained'].mean() * 100:.1f}%")
save(ct, "4_provision_x_trained.csv")

# ---------------------------------------------------------------- 5 CDSS and speech recognition
log("\n=== 5  Clinical decision support and speech recognition (LCA sample) ===")
for it in ("Clinical decision support", "Speech recognition"):
    col = rawall.loc[keep, it]
    log(f"{it}: 'unclear' {(col == UN).mean() * 100:.1f}% overall; ambulatory {(col[lca.sector == 'ambulatory'] == UN).mean() * 100:.1f}%, hospital {(col[lca.sector == 'hospital'] == UN).mean() * 100:.1f}%; "
        f"'not available' {(col == NA).mean() * 100:.1f}%; weekly-daily {col.isin(['täglich', 'wöchentlich']).mean() * 100:.1f}%")
    m = defined
    a = lca.loc[m & (col == UN), "ai_use"].mean() * 100
    b = lca.loc[m & (col != UN), "ai_use"].mean() * 100
    ctab = pd.crosstab(col[m] == UN, lca.loc[m, "ai_use"])
    pv = stats.chi2_contingency(ctab, correction=False)[1]
    log(f"   AI use {a:.1f}% if 'unclear' (n = {int((m & (col == UN)).sum())}) vs {b:.1f}% otherwise (n = {int((m & (col != UN)).sum())}); chi2 p = {pv:.3g}")

# ---------------------------------------------------------------- 6 correlates
log("\n=== 6  Correlates of the counts (descriptive) ===")
for v in ("n_unclear16", "n_not_available16"):
    rho, p = stats.spearmanr(lca[v], lca["competence_prof"])
    log(f"Spearman {v} vs self-rated professional digital competence: rho = {rho:.3f}, p = {p:.2g}")
seg = pd.crosstab(lca["un_cat"], lca["segment"].fillna("undefined"), normalize="index") * 100
seg["n"] = lca["un_cat"].value_counts().sort_index()
log("adoption group (%) by number not recognised:\n" + seg.round(1).to_string())
save(seg.round(2), "6_group_by_unfamiliar.csv")
age = pd.crosstab(lca["un_cat"], lca["age4"])
log(f"age x unfamiliar category chi2 p = {stats.chi2_contingency(age)[1]:.3f}")
log("\nDone.")
LOG.close()
