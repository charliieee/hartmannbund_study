"""
early_career.py  --  analysis script for the npj Digital Medicine article: physicians early in their career.
.

The questionnaire did not record whether a respondent was in postgraduate specialty training (Weiterbildung).
Professional experience was recorded in bands counted from the third state examination (licensure); the band
0-4 years is used as a proxy for the training phase, because specialty training in Germany takes at least five
years.  The 15 pilot respondents (working group of junior physicians) are not part of the data set.

1  Overlap of experience with age (the two cannot both be adjusted for).
2  AI use, prior training, employer provision, private AI use and adoption group by experience group.
3  The 30 preference options: % with Wilson 95% CIs, risk difference (0-4 minus >= 5 years) with Newcombe 95% CI,
   Fisher exact tests with BH over 30; logistic regression adjusted for sex, professional digital competence,
   sector and professional AI use (not for age, which is nearly collinear with experience).
4  Sensitivity: the same comparison restricted to physicians under 40, where 0-4 and >= 5 years coexist.

Usage: python early_career.py <analysis_dataset.csv> <outdir>
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
LOG = open(os.path.join(OUTDIR, "early_career_log.txt"), "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def newcombe(k1, n1, k2, n2, z=1.959964):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return d, d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


df = pd.read_csv(INFILE)
COLS = df.columns
EXP = {"0-4 Jahre": 0, "5-9 Jahre": 1, "10-14 Jahre": 2, "15-19 Jahre": 3, "20-25 Jahre": 4, ">25 Jahre": 5}
exp_raw = df[COLS[14]].str.strip()
df["exp6"] = exp_raw.map(EXP)
# one respondent answered only the experience item for non-physician staff (> 25 years); counted under > 25 years as in Supplementary Note 2
other_item = df[COLS[16]].astype(str).str.strip().map(EXP)
df.loc[df.exp6.isna() & other_item.notna(), "exp6"] = other_item
df["exp04"] = np.where(df.exp6 == 0, 1.0, np.where(df.exp6.notna(), 0.0, np.nan))
AGE6 = {"20-29 Jahre": 0, "30-39 Jahre": 1, "40-49 Jahre": 2, "50-59 Jahre": 3, "60-69 Jahre": 4, "70 Jahre oder älter": 5}
df["age6"] = df[COLS[8]].str.strip().map(AGE6)
df["private_ai"] = df[COLS[44]].fillna("").astype(str).str.contains("Ich nutze KI-Anwendungen privat", regex=False).astype(int)

log("=== 1  Experience and age ===")
log("Experience bands (all 362):", exp_raw.value_counts(dropna=False).to_dict())
log(f"0-4 years: n = {int((df.exp04 == 1).sum())} ({(df.exp04 == 1).mean() * 100:.1f}% of 362); >= 5 years: n = {int((df.exp04 == 0).sum())}; no information: {int(df.exp04.isna().sum())}")
rho, p = stats.spearmanr(df.age6, df.exp6, nan_policy="omit")
log(f"Spearman correlation age band x experience band: rho = {rho:.2f} (p = {p:.2g})")
log("Age by experience group (counts):\n" + pd.crosstab(df[COLS[8]].str.strip(), df.exp04).to_string())
log(f"Share of the 0-4 group under 40: {(df.loc[df.exp04 == 1, 'age6'] <= 1).mean() * 100:.1f}%; share of the >= 5 group under 40: {(df.loc[df.exp04 == 0, 'age6'] <= 1).mean() * 100:.1f}%")
log("Work setting of the 0-4 group:", df.loc[df.exp04 == 1, COLS[11]].str.strip().value_counts().to_dict())

# ============================================================================= 2  adoption, training, access
log("\n=== 2  AI use, training, access by experience group ===")
rows = []
for lab, col, mask in [("Professional AI use (defined status)", "ai_use", df.ai_use.notna()), ("Prior AI-specific training", "trained", df.exp04.notna()),
                       ("Employer provides AI", "prov_yes", df.exp04.notna()), ("Private AI use", "private_ai", df.exp04.notna()),
                       ("Self-reported unauthorised use (valid Q32)", "shadow", None)]:
    if col == "prov_yes":
        df["prov_yes"] = (df.provision == "yes").astype(int)
    if col == "shadow":
        q32 = df[COLS[55]]
        df["shadow"] = np.where(q32.str.startswith("Ja, habe selbst", na=False), 1.0, np.where(q32.isin(["Ja, habe andere schon mal KI ohne formelle Genehmigung nutzen sehen.", "Nein, mir ist kein Fall bekannt"]), 0.0, np.nan))
        mask = df.shadow.notna()
    d = df[mask & df.exp04.notna()]
    k1, n1 = int(d.loc[d.exp04 == 1, col].sum()), int((d.exp04 == 1).sum())
    k0, n0 = int(d.loc[d.exp04 == 0, col].sum()), int((d.exp04 == 0).sum())
    rd, lo, hi = newcombe(k1, n1, k0, n0)
    _, pf = stats.fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])
    rows.append({"measure": lab, "n_0_4": n1, "pct_0_4": 100 * k1 / n1, "n_5plus": n0, "pct_5plus": 100 * k0 / n0, "rd_pp": 100 * rd, "rd_ci_low": 100 * lo, "rd_ci_high": 100 * hi, "p_fisher": pf})
t2 = pd.DataFrame(rows)
pd.set_option("display.width", 250)
log(t2.round(3).to_string(index=False))
t2.to_csv(os.path.join(OUTDIR, "adoption_by_career_stage.csv"), index=False, float_format="%.4f")
seg = pd.crosstab(df.exp04, df.segment, normalize="index") * 100
log("Adoption group by experience group (% of those with defined status):\n" + seg.round(1).to_string())
log("Group counts:\n" + pd.crosstab(df.exp04, df.segment).to_string())
# training among users
u = df[(df.ai_use == 1) & df.exp04.notna()]
log(f"Training among users: 0-4 years {u[u.exp04 == 1].trained.mean() * 100:.1f}% (n = {int((u.exp04 == 1).sum())}); >= 5 years {u[u.exp04 == 0].trained.mean() * 100:.1f}% (n = {int((u.exp04 == 0).sum())})")

# ============================================================================= 3  preferences
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
COV = "male + competence_prof + ambulatory + ai_use"


def fit(formula, data):
    return smf.logit(formula, data=data).fit(disp=0, maxiter=200)


def orci(res, term):
    ci = np.exp(res.conf_int().loc[term])
    return float(np.exp(res.params[term])), float(ci[0]), float(ci[1]), float(res.pvalues[term])


def compare(data, label, adjust=True):
    rows = []
    adj = data.dropna(subset=["competence_prof", "ambulatory", "ai_use"]).copy()
    for blk, dim, lab, col, key in OPTIONS:
        data["y"] = data[COLS[col]].fillna("").str.contains(key, regex=False).astype(int)
        k1, n1 = int(data.loc[data.exp04 == 1, "y"].sum()), int((data.exp04 == 1).sum())
        k0, n0 = int(data.loc[data.exp04 == 0, "y"].sum()), int((data.exp04 == 0).sum())
        rd, lo, hi = newcombe(k1, n1, k0, n0)
        _, pf = stats.fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])
        w1, w0 = wilson(k1, n1), wilson(k0, n0)
        r = {"block": blk, "option": lab, "n_0_4": k1, "pct_0_4": 100 * k1 / n1, "ci_0_4": f"{100 * w1[0]:.1f}-{100 * w1[1]:.1f}",
             "n_5plus": k0, "pct_5plus": 100 * k0 / n0, "ci_5plus": f"{100 * w0[0]:.1f}-{100 * w0[1]:.1f}",
             "rd_pp": 100 * rd, "rd_ci_low": 100 * lo, "rd_ci_high": 100 * hi, "p_fisher": pf}
        if adjust:
            adj["y"] = data.loc[adj.index, "y"]
            m = fit("y ~ exp04 + " + COV, adj)
            r["n_adj"] = int(m.nobs)
            r["OR_adj"], r["ci_adj_low"], r["ci_adj_high"], r["p_adj_model"] = orci(m, "exp04")
        rows.append(r)
    res = pd.DataFrame(rows)
    res["p_fisher_bh"] = multipletests(res["p_fisher"], method="fdr_bh")[1]
    if adjust:
        res["p_adj_model_bh"] = multipletests(res["p_adj_model"], method="fdr_bh")[1]
    log(f"\n=== 3  Preferences, {label}: 0-4 years (n = {int((data.exp04 == 1).sum())}) vs >= 5 years (n = {int((data.exp04 == 0).sum())}) ===")
    cols = ["option", "pct_0_4", "pct_5plus", "rd_pp", "rd_ci_low", "rd_ci_high", "p_fisher", "p_fisher_bh"] + (["OR_adj", "ci_adj_low", "ci_adj_high", "p_adj_model", "p_adj_model_bh"] if adjust else [])
    log(res[cols].round(3).to_string(index=False))
    log(f"BH-significant (Fisher): {list(res.loc[res.p_fisher_bh < 0.05, 'option'])}")
    if adjust:
        log(f"BH-significant (adjusted): {list(res.loc[res.p_adj_model_bh < 0.05, 'option'])}")
    log(f"largest absolute difference: {res.loc[res.rd_pp.abs().idxmax(), 'option']} ({res.rd_pp.abs().max():.1f} pp); options with |difference| >= 10 pp: {list(res.loc[res.rd_pp.abs() >= 10, 'option'])}")
    return res


d_all = df[df.exp04.notna()].copy()
res_all = compare(d_all, "all respondents (percentages of all 362 with experience information)")
res_all.to_csv(os.path.join(OUTDIR, "preferences_by_career_stage.csv"), index=False, float_format="%.4f")
d_young = df[df.exp04.notna() & (df.age6 <= 1)].copy()
res_young = compare(d_young, "physicians under 40 only")
res_young.to_csv(os.path.join(OUTDIR, "preferences_by_career_stage_under40.csv"), index=False, float_format="%.4f")

# ============================================================================= checksums
log("\n=== Checksums ===")
checks = {"0-4 years n": (int((df.exp04 == 1).sum()), 65), ">= 5 years n": (int((df.exp04 == 0).sum()), 295), "no information": (int(df.exp04.isna().sum()), 2),
          "rho age x experience >= 0.90": (int(rho >= 0.90), 1), "> 25 years n (incl. the one from the other item)": (int((df.exp6 == 5).sum()), 145)}
bad = [k for k, (o, e) in checks.items() if o != e]
for k, (o, e) in checks.items():
    log(f"{'OK  ' if o == e else 'FAIL'} {k}: {o} (expected {e})")
assert not bad, f"CHECKSUMS VIOLATED: {bad}"
log("\nAll checksums OK.")
LOG.close()
