"""
providers_by_sector.py  --  analysis script for the npj Digital Medicine article: preferred providers of AI-related CME by sector
(hospital versus ambulatory care), reported in one sentence of the RQ3 section and in Table 3.

The comparison of early-career physicians (early_career.py) showed that naming employers as providers was the one
apparent difference between physicians with up to four years since licensure and the rest, and that it disappeared once
sector was taken into account.  This script gives the sector pattern itself: the share of hospital respondents (inpatient
and outpatient, n = 152) and of ambulatory respondents (solo practice, group practice, MVZ, n = 195) naming each of the
eight provider options, with Wilson 95% confidence intervals, the risk difference with Newcombe 95% CI and a chi-square test
without continuity correction; p-values are Benjamini-Hochberg-corrected over the eight provider options.

Usage: python providers_by_sector.py <analysis_dataset.csv> <outdir>
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

INFILE, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
LOG = open(os.path.join(OUTDIR, "providers_by_sector_log.txt"), "w", encoding="utf-8")


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
setting = df[COLS[11]].fillna("")
hospital = setting.str.startswith("Krankenhaus")
ambulatory = setting.str.startswith("Praxis") | setting.str.startswith("MVZ")
H, A = df[hospital], df[ambulatory]
log(f"hospital (inpatient + outpatient) n = {len(H)}; ambulatory (solo, group practice, MVZ) n = {len(A)}; other or no information n = {len(df) - len(H) - len(A)}")
assert (len(H), len(A)) == (152, 195)

PROVIDERS = [
    ("Chambers of physicians", "Ärztekammern"),
    ("Professional associations", "Berufsverbände"),
    ("Employers", "Arbeitgeber"),
    ("Specialty societies", "Fachgesellschaften"),
    ("Health insurance assoc.", "Kassenärztliche Vereinigungen"),
    ("Universities", "Universitäten"),
    ("Commercial AI vendors", "Industrieanbieter"),
    ("Private providers", "Unabhängige private"),
]
q29 = df[COLS[52]].fillna("")
rows = []
for lab, key in PROVIDERS:
    kH = int(q29[H.index].str.contains(key, regex=False).sum())
    kA = int(q29[A.index].str.contains(key, regex=False).sum())
    rd, lo, hi = newcombe(kH, len(H), kA, len(A))
    chi2, p, _, _ = stats.chi2_contingency([[kH, len(H) - kH], [kA, len(A) - kA]], correction=False)
    wH, wA = wilson(kH, len(H)), wilson(kA, len(A))
    rows.append({"provider": lab, "n_hospital": kH, "pct_hospital": 100 * kH / len(H), "ci_hospital": f"{100 * wH[0]:.1f}-{100 * wH[1]:.1f}",
                 "n_ambulatory": kA, "pct_ambulatory": 100 * kA / len(A), "ci_ambulatory": f"{100 * wA[0]:.1f}-{100 * wA[1]:.1f}",
                 "rd_pp": 100 * rd, "rd_ci_low": 100 * lo, "rd_ci_high": 100 * hi, "chi2": chi2, "p": p})
res = pd.DataFrame(rows)
res["p_adj"] = multipletests(res["p"], method="fdr_bh")[1]
res.to_csv(os.path.join(OUTDIR, "providers_by_sector.csv"), index=False, float_format="%.4f")
pd.set_option("display.width", 250)
log(res.round(3).to_string(index=False))

# among employers' nominations: are they the first-named provider in hospitals?
emp = res.set_index("provider")
log(f"\nEmployers named by {emp.loc['Employers', 'pct_hospital']:.1f}% of hospital and {emp.loc['Employers', 'pct_ambulatory']:.1f}% of ambulatory respondents "
    f"(difference {emp.loc['Employers', 'rd_pp']:+.1f} pp, 95% CI {emp.loc['Employers', 'rd_ci_low']:.1f} to {emp.loc['Employers', 'rd_ci_high']:.1f}; p = {emp.loc['Employers', 'p']:.2e})")
log(f"Chambers: {emp.loc['Chambers of physicians', 'pct_hospital']:.1f}% vs {emp.loc['Chambers of physicians', 'pct_ambulatory']:.1f}%; "
    f"health insurance associations: {emp.loc['Health insurance assoc.', 'pct_hospital']:.1f}% vs {emp.loc['Health insurance assoc.', 'pct_ambulatory']:.1f}%")

log("\n=== Checksums ===")
checks = {"employers hospital %": (round(emp.loc["Employers", "pct_hospital"], 1), 82.2), "employers ambulatory %": (round(emp.loc["Employers", "pct_ambulatory"], 1), 24.1),
          "health insurance assoc. hospital %": (round(emp.loc["Health insurance assoc.", "pct_hospital"], 1), 14.5), "health insurance assoc. ambulatory %": (round(emp.loc["Health insurance assoc.", "pct_ambulatory"], 1), 50.8),
          "chambers hospital %": (round(emp.loc["Chambers of physicians", "pct_hospital"], 1), 61.8), "chambers ambulatory %": (round(emp.loc["Chambers of physicians", "pct_ambulatory"], 1), 68.7)}
bad = [k for k, (o, e) in checks.items() if o != e]
for k, (o, e) in checks.items():
    log(f"{'OK  ' if o == e else 'FAIL'} {k}: {o} (expected {e})")
assert not bad, f"CHECKSUMS VIOLATED: {bad}"
log("\nAll checksums OK.")
LOG.close()
