"""
lca_missing_sensitivity.py  --  sensitivity of the primary LCA to the coding of unanswered matrix cells
(revision, reviewer 3 point 6 / Supplementary Note 8).

In the primary model (lca.py --coding four --drop 12,15,16 --starts 100) unanswered cells of the
application matrix (Q13) are filled with level 0 ('not available'; lca.py line 182, fillna(0)).
This script
  1  counts the unanswered cells among the 16 items in the LCA sample (n = 356),
  2  re-fits the primary model (unanswered = 'not available') as reference,
  3  fits the same model with unanswered cells treated as missing (StepMix 'categorical_nan',
     full-information likelihood), k = 1..5, and compares the k = 3 partition with the reference,
  4  fits the reference coding on respondents without any unanswered cell (complete cases),
  5  lists the competing local maxima (k = 3) under both codings (150 single random starts).

Usage (from the repository root, like run_lca.sh):
  venv/bin/python -I analysis/lca_missing_sensitivity.py analysis_dataset.csv out/lca_missing
Output: out/lca_missing/lca_missing_report.txt and CSV tables.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score
from stepmix import StepMix

warnings.filterwarnings("ignore")

INFILE, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
REP = open(os.path.join(OUTDIR, "lca_missing_report.txt"), "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    REP.write(s + "\n")
    REP.flush()


STARTS, SEED = int(os.environ.get("LCA_STARTS", 100)), 42
ITEM_COLS = [f"lca_item_{i:02d}" for i in range(1, 20)]
ITEM_LABELS = ["Practice/hospital IS", "Electronic patient record (ePA)", "e-medication list",
               "e-sick note (eAU)", "PACS / imaging", "Laboratory IS", "Clinical decision support",
               "DiGA", "Health apps", "Phone assistance", "Video consultation", "Tele-rounds",
               "Tele-/home monitoring", "Speech recognition", "Digital pre-registration",
               "Triage tools", "Medication safety (AMTS)", "Automated duty scheduling",
               "Knowledge management"]
DROP = ["Tele-rounds", "Digital pre-registration", "Triage tools"]          # items 12, 15, 16
NA, UN = "möchte ich, aber nicht verfügbar", "unklar, was das ist"
MAP4 = {NA: 0, UN: 1, "seltener": 2, "monatlich": 2, "wöchentlich": 3, "täglich": 3}
LAB3 = ["Wide range, routine use", "Wide range, occasional use", "Core set, routine use"]

# ----------------------------------------------------------------------------- data (as in lca.py)
df = pd.read_csv(INFILE)
keep = df[ITEM_COLS].notna().all(axis=1)                 # >= 15 valid items, as in lca.py / prepare_data.py
sub = df[keep].copy()
raw = sub[df.columns[18:37]].copy()
raw.columns = ITEM_LABELS
raw16 = raw.drop(columns=DROP)
n, J = raw16.shape
log(f"LCA sample n = {n}, items = {J}, cells = {n * J}")

coded = raw16.apply(lambda c: c.map(MAP4))               # NaN where unanswered (or unknown text)
unknown = raw16.notna() & coded.isna()
assert not unknown.values.any(), "unexpected response text: " + str(pd.unique(raw16[unknown].stack()))

# ----------------------------------------------------------------------------- 1  unanswered cells
miss = coded.isna()
per_resp = miss.sum(axis=1)
log("\n=== 1  Unanswered cells (16 items) ===")
log(f"unanswered cells: {int(miss.values.sum())} of {n * J} ({miss.values.mean() * 100:.2f}%)")
log(f"respondents with >= 1 unanswered cell: {int((per_resp > 0).sum())} ({(per_resp > 0).mean() * 100:.1f}%)")
log("distribution of unanswered cells per respondent:\n" + per_resp.value_counts().sort_index().to_string())
log("unanswered cells per item:\n" + miss.sum().sort_values(ascending=False).to_string())
explicit_na = int((raw16 == NA).values.sum())
log(f"explicit 'not available' cells: {explicit_na}; after filling: {explicit_na + int(miss.values.sum())} "
    f"(+{int(miss.values.sum()) / max(explicit_na, 1) * 100:.1f}%)")
pd.DataFrame({"unanswered": miss.sum()}).to_csv(os.path.join(OUTDIR, "1_unanswered_per_item.csv"))

X_fill = coded.fillna(0).astype(int)                     # reference coding (as lca.py)
X_nan = coded.astype(float)                              # NaN kept


def fit(X, k, meas):
    return StepMix(n_components=k, measurement=meas, n_init=STARTS, random_state=SEED,
                   verbose=0, progress_bar=False).fit(X)


def ordered(m, X, idx):
    """Modal class, ordered by mean digital index (descending), as in lca.py."""
    r = pd.Series(m.predict(X), index=X.index)
    order = sub.loc[X.index, "digital_index"].groupby(r).mean().sort_values(ascending=False).index
    return r.map({c: i for i, c in enumerate(order)})


def profile(cl, X):
    s = sub.loc[X.index]
    rows = []
    for i in range(3):
        g = cl == i
        d = g & s["ai_use"].notna()
        rows.append(dict(cls=LAB3[i], n=int(g.sum()),
                         ai_use_pct=s.loc[d, "ai_use"].mean() * 100,
                         provision_pct=(s.loc[g, "provision"] == "yes").mean() * 100,
                         trained_pct=s.loc[g, "trained"].mean() * 100))
    return pd.DataFrame(rows).set_index("cls")


def selection(X, meas, tag):
    rows, models = [], {}
    for k in range(1, 6):
        m = fit(X, k, meas)
        models[k] = m
        pr = m.predict_proba(X)
        ent = 1 + (pr * np.log(np.clip(pr, 1e-12, None))).sum() / (len(X) * np.log(k)) if k > 1 else np.nan
        rows.append(dict(k=k, logLik=m.score(X) * len(X), BIC=m.bic(X), aBIC=-2 * m.score(X) * len(X) + m.n_parameters * np.log((len(X) + 2) / 24), AIC=m.aic(X),
                         entropy=ent, sizes="/".join(map(str, sorted(np.bincount(m.predict(X), minlength=k),
                                                                     reverse=True)))))
    t = pd.DataFrame(rows).set_index("k")
    log(f"\nModel selection ({tag}):\n" + t.round(2).to_string())
    log(f"BIC minimum at k = {t['BIC'].idxmin()}")
    t.to_csv(os.path.join(OUTDIR, f"selection_{tag}.csv"))
    return models, t


# ----------------------------------------------------------------------------- 2  reference
log("\n=== 2  Reference: unanswered = 'not available' (primary model) ===")
mods_ref, sel_ref = selection(X_fill, "categorical", "reference")
cl_ref = ordered(mods_ref[3], X_fill, X_fill.index)
p_ref = profile(cl_ref, X_fill)
log("k = 3 profiles:\n" + p_ref.round(1).to_string())

# ----------------------------------------------------------------------------- 3  missing as missing
log("\n=== 3  Unanswered cells treated as missing (categorical_nan, FIML) ===")
mods_nan, sel_nan = selection(X_nan, "categorical_nan", "missing_as_missing")
cl_nan = ordered(mods_nan[3], X_nan, X_nan.index)
p_nan = profile(cl_nan, X_nan)
log("k = 3 profiles:\n" + p_nan.round(1).to_string())
ari = adjusted_rand_score(cl_ref, cl_nan)
same = int((cl_ref == cl_nan).sum())
log(f"\nAgreement with reference (k = 3): ARI = {ari:.3f}; same class for {same} of {n} ({same / n * 100:.1f}%)")
ct = pd.crosstab(cl_ref.map(dict(enumerate(LAB3))), cl_nan.map(dict(enumerate(LAB3))),
                 rownames=["reference"], colnames=["missing as missing"])
log(ct.to_string())
ct.to_csv(os.path.join(OUTDIR, "3_crosstab_reference_vs_nan.csv"))
pr = mods_nan[3].predict_proba(X_nan)
log(f"mean posterior of assigned class: {pr.max(axis=1).mean():.3f}")
s = sub.loc[X_nan.index]
d = s["ai_use"].notna()
chi = stats.chi2_contingency(pd.crosstab(cl_nan[d], s.loc[d, "ai_use"]))
log(f"AI use across profiles (missing as missing): chi2({chi[2]}) = {chi[0]:.1f}, p = {chi[1]:.4f}")

# ----------------------------------------------------------------------------- 4  complete cases
log("\n=== 4  Respondents without unanswered cells (reference coding) ===")
cc = per_resp == 0
Xc = X_fill[cc]
m_cc = fit(Xc, 3, "categorical")
r = pd.Series(m_cc.predict(Xc), index=Xc.index)
order = sub.loc[Xc.index, "digital_index"].groupby(r).mean().sort_values(ascending=False).index
cl_cc = r.map({c: i for i, c in enumerate(order)})
log(f"n = {int(cc.sum())}; BIC k=3: {m_cc.bic(Xc):.1f}")
log("k = 3 profiles:\n" + profile(cl_cc, Xc).round(1).to_string())
ari_cc = adjusted_rand_score(cl_ref[cc], cl_cc)
log(f"ARI with reference partition (same respondents): {ari_cc:.3f}")


# ----------------------------------------------------------------------------- 5  competing local maxima (k = 3)
log("\n=== 5  Competing local maxima, k = 3 (150 single random starts per coding) ===")
for tag, X, meas in [("reference", X_fill, "categorical"), ("missing as missing", X_nan, "categorical_nan")]:
    rows = []
    for sd in range(150):
        m = StepMix(n_components=3, measurement=meas, n_init=1, random_state=sd,
                    verbose=0, progress_bar=False).fit(X)
        lab = ordered(m, X, X.index)
        g0 = lab == 0
        d0 = g0 & sub.loc[X.index, "ai_use"].notna()
        rows.append(dict(seed=sd, logLik=round(m.score(X) * len(X), 2), ARI_primary=round(adjusted_rand_score(cl_ref, lab), 2),
                         on_diagonal=int((lab == cl_ref).sum()),
                         sizes="/".join(str(int((lab == i).sum())) for i in range(3)),
                         routine_ai_use_pct=round(sub.loc[X.index][d0]["ai_use"].mean() * 100, 1),
                         routine_provision_pct=round((sub.loc[X.index][g0]["provision"] == "yes").mean() * 100, 1)))
    t = pd.DataFrame(rows).sort_values("logLik", ascending=False).drop_duplicates("logLik")
    best = t["logLik"].max()
    t = t[t["logLik"] >= best - 5]
    log(f"\n{tag}: distinct maxima within 5 LL units of the best ({len(t)}); sizes = routine/occasional/core set")
    log(t.to_string(index=False))
    t.to_csv(os.path.join(OUTDIR, f"5_local_maxima_{tag.replace(' ', '_')}.csv"), index=False)

# ----------------------------------------------------------------------------- summary
log("\n=== Summary for Supplementary Note 8 ===")
log(f"Unanswered cells: {int(miss.values.sum())} of {n * J} ({miss.values.mean() * 100:.1f}%), "
    f"in {int((per_resp > 0).sum())} respondents.")
log(f"Missing-as-missing: BIC-preferred k = {sel_nan['BIC'].idxmin()}; k = 3 ARI with primary = {ari:.2f} "
    f"({same} of {n} on the diagonal); class sizes {'/'.join(str(int(x)) for x in p_nan['n'])}; "
    f"AI use {'/'.join(f'{x:.1f}' for x in p_nan['ai_use_pct'])}%.")
log(f"Complete cases (n = {int(cc.sum())}): ARI with primary = {ari_cc:.2f}.")
REP.close()
