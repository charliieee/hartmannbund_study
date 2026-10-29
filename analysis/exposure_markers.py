"""
exposure_markers.py  --  analysis script for the npj Digital Medicine article: training preferences of physicians who report having used AI at work without formal
authorisation (Q32), and of physicians who use AI privately (Q21), within the covariate set of
Table 2.

Exposure 1  Q32 'Yes, I have used AI myself without formal authorisation before' (1) vs 'seen others' or
            'no case known' (0); 'no answer' excluded.  Q32 asks about ever having done so, not current practice.
Exposure 2  Q21 'I use AI applications privately (e.g. ChatGPT, AI image generators)' selected (1) vs not (0).

For each exposure and each of the 30 preference options tested in Supplementary Note 6:
  1  % with and without the exposure (respondents with defined use status), Fisher exact test, BH over 30
  2  Model S1: logit(option) ~ exposure + professional AI use (yes/no) + age (4) + sex + professional digital
     competence + sector  (age, sex, competence, sector, plus use status)
  3  Model S2: S1 + prior AI-specific training + employer provision (3) + the other exposure
     (i.e. unauthorised use adjusted for private use, and vice versa); covariate set of the Table 2
  4  Within users only (S1 covariates without use status)
Summary measures: 'any governance topic' (data protection, ethics and law, critical appraisal) and 'any
application topic' (diagnostic systems, large language models, practical implementation, treatment planning),
with the same models.  Overlap of the two exposures and of each with use status is tabulated.

Usage: python exposure_markers.py <analysis_dataset.csv> <outdir>
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
INFILE, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
LOG = open(os.path.join(OUTDIR, "exposure_markers_log.txt"), "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


df = pd.read_csv(INFILE)
COLS = df.columns
SELF = "Ja, habe selbst schon mal KI ohne formelle Genehmigung genutzt."
q32 = df[COLS[55]]
df["shadow"] = np.where(q32 == SELF, 1.0, np.where(q32.isin(["Ja, habe andere schon mal KI ohne formelle Genehmigung nutzen sehen.", "Nein, mir ist kein Fall bekannt"]), 0.0, np.nan))
df["private_ai"] = df[COLS[44]].fillna("").astype(str).str.contains("Ich nutze KI-Anwendungen privat", regex=False).astype(int)
STAGE = {"Nein, und aktuell nicht geplant": 0, "Nein, aber geplant/Interesse vorhanden": 1, "Ja, selten/experimentell (einmal pro Monat)": 2,
         "Ja, gelegentlich (einmal pro Woche)": 3, "Ja, regelmäßig (mehrfach pro Woche)": 4}
df["stage"] = df[COLS[42]].map(STAGE)

log("=== Variables ===")
log("Q32 (all 362):", q32.value_counts(dropna=False).to_dict())
log(f"Self-reported unauthorised use: {int((df.shadow == 1).sum())} of {int(df.shadow.notna().sum())} with a valid Q32 answer ({(df.shadow == 1).sum() / df.shadow.notna().sum() * 100:.1f}%)")
d = df[df.ai_use.notna() & df.shadow.notna()]
log(f"With defined use status and valid Q32: n = {len(d)}; self-reported unauthorised use {int(d.shadow.sum())}")
log("Unauthorised use by stage (%):", (d.groupby("stage").shadow.mean() * 100).round(1).to_dict(), "| n by stage:", d.groupby("stage").shadow.size().to_dict())
ct = pd.crosstab(d.shadow, d.ai_use)
log(f"Unauthorised use x professional AI use:\n{ct}")
log(f"   of the {int(d.shadow.sum())} with self-reported unauthorised use, {int(((d.shadow == 1) & (d.ai_use == 0)).sum())} report no current professional AI use")
ct2 = pd.crosstab(d.shadow, d.private_ai)
log(f"Unauthorised use x private AI use:\n{ct2}")
log(f"   private AI use among those with unauthorised use: {d[d.shadow == 1].private_ai.mean() * 100:.1f}%; among the others: {d[d.shadow == 0].private_ai.mean() * 100:.1f}%")
log(f"Private AI use: {int(df.private_ai.sum())} of 362 ({df.private_ai.mean() * 100:.1f}%); with defined use status {int(df[df.ai_use.notna()].private_ai.sum())} of {int(df.ai_use.notna().sum())}")

OPTIONS = [
    ("W", "Format", "Self-paced online courses", 48, "Online-Kurse"),
    ("W", "Format", "Live webinars", 48, "Live Online-Webinare"),
    ("W", "Format", "Nuggets (15-30 min)", 48, "Nuggets"),
    ("W", "Format", "Blended learning", 48, "Blended"),
    ("W", "Format", "On-site sessions", 48, "Präsenzveranstaltungen"),
    ("W", "Scope", "Short formats (1-2 h)", 49, "Kurzformate"),
    ("W", "Scope", "Half-day", 49, "Halbtägige"),
    ("W", "Scope", "Longitudinal", 49, "Längerfristige"),
    ("W", "Scope", "Full-day", 49, "Ganztägige"),
    ("W", "Orientation", "Application-oriented", 50, "Anwendungsorientiert"),
    ("C", "Depth", "Introductory", 50, "Einführend"),
    ("C", "Depth", "Specialised", 50, "Vertiefend"),
    ("C", "Depth", "Strategic", 50, "Strategisch"),
    ("C", "Topic", "Diagnostic systems", 51, "Diagnosesysteme"),
    ("C", "Topic", "Treatment planning", 51, "Therapieplanung"),
    ("C", "Topic", "Practical implementation", 51, "Praktische Implementierung"),
    ("C", "Topic", "Large language models", 51, "Sprachmodellen"),
    ("C", "Topic", "Data protection", 51, "Datenmanagement"),
    ("C", "Topic", "Diagnostic imaging", 51, "Bildgebung"),
    ("C", "Topic", "Critical appraisal", 51, "Kritische Berwertung"),
    ("C", "Topic", "Ethics and law", 51, "Ethische"),
    ("C", "Topic", "Patient communication", 51, "Patienteninteraktion"),
    ("P", "Provider", "Chambers of physicians", 52, "Ärztekammern"),
    ("P", "Provider", "Professional associations", 52, "Berufsverbände"),
    ("P", "Provider", "Employers", 52, "Arbeitgeber"),
    ("P", "Provider", "Specialty societies", 52, "Fachgesellschaften"),
    ("P", "Provider", "Health insurance assoc.", 52, "Kassenärztliche"),
    ("P", "Provider", "Universities", 52, "Universitäten"),
    ("P", "Provider", "Commercial AI vendors", 52, "Industrieanbieter"),
    ("P", "Provider", "Private providers", 52, "private Weiterbildungsinstitute"),
]
q28 = df[COLS[51]].fillna("")
df["gov_any"] = (q28.str.contains("Datenmanagement", regex=False) | q28.str.contains("Ethische", regex=False) | q28.str.contains("Kritische Berwertung", regex=False)).astype(int)
df["app_any"] = (q28.str.contains("Diagnosesysteme", regex=False) | q28.str.contains("Sprachmodellen", regex=False) | q28.str.contains("Praktische Implementierung", regex=False) | q28.str.contains("Therapieplanung", regex=False)).astype(int)
SUMMARY = [("S", "Summary", "Any governance topic", None, "gov_any"), ("S", "Summary", "Any application topic", None, "app_any")]

COV = "ai_use + C(age4, Treatment('<40')) + male + competence_prof + ambulatory"
COV_FULL = COV + " + trained + C(provision, Treatment('no'))"


def fit(formula, data):
    return smf.logit(formula, data=data).fit(disp=0, maxiter=200)


def orci(res, term):
    ci = np.exp(res.conf_int().loc[term])
    return float(np.exp(res.params[term])), float(ci[0]), float(ci[1]), float(res.pvalues[term])


def run(exposure, other, label):
    base = df[df.ai_use.notna() & df[exposure].notna()].copy()
    base["x"] = base[exposure].astype(int)
    adj = base.dropna(subset=["age4", "competence_prof", "ambulatory"]).copy()
    log(f"\n=== Exposure: {label} ===  descriptive n = {len(base)} (exposed {int(base.x.sum())}); adjusted n = {len(adj)} (exposed {int(adj.x.sum())}); users in adjusted sample {int(adj.ai_use.sum())}")
    rows = []
    for blk, dim, lab, col, key in OPTIONS + SUMMARY:
        if col is None:
            base["y"] = base[key]; adj["y"] = adj[key]
        else:
            base["y"] = base[COLS[col]].fillna("").str.contains(key, regex=False).astype(int)
            adj["y"] = base.loc[adj.index, "y"]
        k1, n1 = int(base.loc[base.x == 1, "y"].sum()), int((base.x == 1).sum())
        k0, n0 = int(base.loc[base.x == 0, "y"].sum()), int((base.x == 0).sum())
        _, pf = stats.fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])
        r = {"block": blk, "option": lab, "n_exposed": k1, "pct_exposed": 100 * k1 / n1, "n_unexposed": k0, "pct_unexposed": 100 * k0 / n0, "p_fisher": pf}
        m1 = fit("y ~ x + " + COV, adj)
        r["OR_S1"], r["ci_S1_low"], r["ci_S1_high"], r["p_S1"] = orci(m1, "x")
        m2 = fit(f"y ~ x + {other} + " + COV_FULL, adj)
        r["OR_S2"], r["ci_S2_low"], r["ci_S2_high"], r["p_S2"] = orci(m2, "x")
        r["OR_other_in_S2"] = float(np.exp(m2.params[other]))
        mu = fit("y ~ x + " + COV.replace("ai_use + ", ""), adj[adj.ai_use == 1])
        r["n_users"] = int(mu.nobs)
        r["OR_users"], r["ci_users_low"], r["ci_users_high"], r["p_users"] = orci(mu, "x")
        rows.append(r)
    res = pd.DataFrame(rows)
    tested = res.block != "S"
    res.loc[tested, "p_fisher_adj"] = multipletests(res.loc[tested, "p_fisher"], method="fdr_bh")[1]
    res.loc[tested, "p_S1_adj"] = multipletests(res.loc[tested, "p_S1"], method="fdr_bh")[1]
    res.loc[tested, "p_S2_adj"] = multipletests(res.loc[tested, "p_S2"], method="fdr_bh")[1]
    res.to_csv(os.path.join(OUTDIR, ("unauthorised_use" if exposure == "shadow" else exposure) + "_preferences.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 300)
    log(res[["option", "pct_exposed", "pct_unexposed", "p_fisher_adj", "OR_S1", "ci_S1_low", "ci_S1_high", "p_S1_adj", "OR_S2", "ci_S2_low", "ci_S2_high", "p_S2", "p_S2_adj", "OR_other_in_S2", "OR_users", "p_users"]].round(3).to_string(index=False))
    log(f"BH-significant (Fisher, 30 tests): {list(res.loc[tested & (res.p_fisher_adj < 0.05), 'option'])}")
    log(f"BH-significant (Model S1, 30 tests): {list(res.loc[tested & (res.p_S1_adj < 0.05), 'option'])}")
    log(f"BH-significant (Model S2, 30 tests): {list(res.loc[tested & (res.p_S2_adj < 0.05), 'option'])}")
    return res


res_sh = run("shadow", "private_ai", "self-reported unauthorised AI use (Q32)")
res_pr = run("private_ai", "shadow", "private AI use (Q21)")

# ----------------------------------------------------------------------------- joint view of the two exposures on the summary measures
log("\n=== Joint view: any governance topic by unauthorised use x private AI use (defined status, valid Q32) ===")
d = df[df.ai_use.notna() & df.shadow.notna()]
t = d.groupby(["shadow", "private_ai"]).agg(n=("gov_any", "size"), gov_any_pct=("gov_any", "mean"), app_any_pct=("app_any", "mean"), ai_use_pct=("ai_use", "mean"))
for c in ["gov_any_pct", "app_any_pct", "ai_use_pct"]:
    t[c] = (t[c] * 100).round(1)
log(t.to_string())
t.to_csv(os.path.join(OUTDIR, "governance_by_unauthorised_x_private.csv"))
log("\n=== Any governance topic by stage (defined status, n = 341) ===")
log((df[df.stage.notna()].groupby("stage").gov_any.mean() * 100).round(1).to_dict())

# ----------------------------------------------------------------------------- checksums
log("\n=== Checksums ===")
g = lambda res, opt, c: float(res.loc[res.option == opt, c].iloc[0])
checks = {
    "self-reported unauthorised use (valid Q32)": (int((df.shadow == 1).sum()), 96, 0),
    "valid Q32 answers": (int(df.shadow.notna().sum()), 334, 0),
    "gov_any shadow S1 OR < 1": (float(g(res_sh, "Any governance topic", "OR_S1") < 1), 1.0, 0),
    "ethics shadow S1 OR < 1": (float(g(res_sh, "Ethics and law", "OR_S1") < 1), 1.0, 0),
    "ethics private S1 OR < 1": (float(g(res_pr, "Ethics and law", "OR_S1") < 1), 1.0, 0),
    "vendors shadow S1 OR > 1": (float(g(res_sh, "Commercial AI vendors", "OR_S1") > 1), 1.0, 0),
}
bad = []
for k, (obs, exp, tol) in checks.items():
    good = abs(obs - exp) <= tol
    log(f"{'OK  ' if good else 'FAIL'} {k}: {obs} (expected {exp})")
    if not good:
        bad.append(k)
assert not bad, f"CHECKSUMS VIOLATED: {bad}"
log("\nAll checksums OK.")
LOG.close()
