"""
no_intention_group.py  --  analysis script for the npj Digital Medicine article: who the
63 physicians without current intention to use AI are, and whether the threshold findings depend on composition.

1  Composition: no intention (63) vs interested non-users (86) vs users (192): age, sex, sector, experience,
   self-rated competence, private AI use, employer provision, prior training, number of surveyed applications not
   recognised (16 items), digital application-use profile.
2  Threshold (stage 0 vs 1-4) for the governance and application options with the full covariate set of the
   Table 2 (age, sex, competence, sector, training, private AI use, provision), and additionally
   interested (1) vs no intention (0) among non-users with the same set.
Usage: python no_intention_group.py <analysis_dataset.csv> <outdir>
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
INFILE, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
LOG = open(os.path.join(OUTDIR, "no_intention_group_log.txt"), "w", encoding="utf-8")


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


df = pd.read_csv(INFILE)
COLS = df.columns
STAGE = {"Nein, und aktuell nicht geplant": 0, "Nein, aber geplant/Interesse vorhanden": 1, "Ja, selten/experimentell (einmal pro Monat)": 2,
         "Ja, gelegentlich (einmal pro Woche)": 3, "Ja, regelmäßig (mehrfach pro Woche)": 4}
df["stage"] = df[COLS[42]].map(STAGE)
df["private_ai"] = df[COLS[44]].fillna("").astype(str).str.contains("Ich nutze KI-Anwendungen privat", regex=False).astype(int)
df["prov_yes"] = (df.provision == "yes").astype(int)
df["age6"] = df[COLS[8]].str.strip().map({"20-29 Jahre": 0, "30-39 Jahre": 1, "40-49 Jahre": 2, "50-59 Jahre": 3, "60-69 Jahre": 4, "70 Jahre oder älter": 5})
df["age50"] = (df.age6 >= 3).astype(float).where(df.age6.notna())
df["exp25"] = (df[COLS[14]].str.strip() == ">25 Jahre").astype(int)
df["sector"] = np.where(df.ambulatory == 1, "ambulatory", np.where(df.ambulatory == 0, "hospital", "other"))
# unfamiliarity count, 16 items, as in application_counts.py
ITEM_LABELS = ["Practice/hospital IS", "Electronic patient record (ePA)", "e-medication list", "e-sick note (eAU)", "PACS / imaging", "Laboratory IS",
               "Clinical decision support", "DiGA", "Health apps", "Phone assistance", "Video consultation", "Tele-rounds", "Tele-/home monitoring",
               "Speech recognition", "Digital pre-registration", "Triage tools", "Medication safety (AMTS)", "Automated duty scheduling", "Knowledge management"]
raw = df[df.columns[18:37]].copy(); raw.columns = ITEM_LABELS
raw16 = raw[[l for l in ITEM_LABELS if l not in ["Tele-rounds", "Digital pre-registration", "Triage tools"]]]
df["n_unclear16"] = (raw16 == "unklar, was das ist").sum(axis=1).where(raw.notna().sum(axis=1) >= 15)
q28 = df[COLS[51]].fillna(""); q29 = df[COLS[52]].fillna(""); q27 = df[COLS[50]].fillna(""); q16 = df[COLS[39]].fillna("")
OPT = {"Ethics and law": q28.str.contains("Ethische", regex=False), "Critical appraisal": q28.str.contains("Kritische Berwertung", regex=False),
       "Data protection": q28.str.contains("Datenmanagement", regex=False), "Diagnostic systems": q28.str.contains("Diagnosesysteme", regex=False),
       "Large language models": q28.str.contains("Sprachmodellen", regex=False), "Commercial AI vendors": q29.str.contains("Industrieanbieter", regex=False),
       "Chambers of physicians": q29.str.contains("Ärztekammern", regex=False), "Introductory": q27.str.contains("Einführend", regex=False),
       "Specialised": q27.str.contains("Vertiefend", regex=False), "Application-oriented": q27.str.contains("Anwendungsorientiert", regex=False)}
for k, v in OPT.items():
    df[k.replace(" ", "_")] = v.astype(int)
df["gov_any"] = ((df.Ethics_and_law + df.Critical_appraisal + df.Data_protection) > 0).astype(int)
df["b_dataprot"] = q16.str.contains("Sicherheits-/Datenschutzbedenken", regex=False).astype(int)
df["b_nobenefit"] = (q16.str.contains("Keine wahrgenommene Nützlichkeit", regex=False) | q16.str.contains("kein bedarf", regex=False)).astype(int)
df["b_knowledge"] = q16.str.contains("Fehlendes Wissen", regex=False).astype(int)
df["b_notavail"] = q16.str.contains("Nicht verfügbar", regex=False).astype(int)

N, I, U = df[df.segment == "no_intention"], df[df.segment == "interested"], df[df.segment == "user"]

# ============================================================================= 1  composition
log("=== 1  Composition: no intention (n = %d) | interested (n = %d) | users (n = %d) ===" % (len(N), len(I), len(U)))
rows = []
for lab, fn in [("age >= 50 years, %", lambda d: d.age50.mean() * 100), ("age 60+, %", lambda d: (d.age6 >= 4).mean() * 100), ("age < 40, %", lambda d: (d.age6 <= 1).mean() * 100),
                ("male, %", lambda d: d.male.mean() * 100), ("hospital, %", lambda d: (d.sector == "hospital").mean() * 100), ("ambulatory, %", lambda d: (d.sector == "ambulatory").mean() * 100),
                ("other sector, %", lambda d: (d.sector == "other").mean() * 100), ("> 25 years experience, %", lambda d: d.exp25.mean() * 100),
                ("self-rated competence, mean", lambda d: d.competence_prof.mean()), ("private AI use, %", lambda d: d.private_ai.mean() * 100),
                ("employer provides AI, %", lambda d: d.prov_yes.mean() * 100), ("prior AI training, %", lambda d: d.trained.mean() * 100),
                ("applications not recognised, mean (16 items)", lambda d: d.n_unclear16.mean()), ("not recognised >= 3, %", lambda d: (d.n_unclear16 >= 3).mean() * 100)]:
    rows.append({"measure": lab, "no_intention": fn(N), "interested": fn(I), "users": fn(U)})
t1 = pd.DataFrame(rows)
pd.set_option("display.width", 250)
log(t1.round(1).to_string(index=False))
t1.to_csv(os.path.join(OUTDIR, "1_composition.csv"), index=False, float_format="%.2f")
for v, lab in [("age50", "age >= 50"), ("private_ai", "private AI use"), ("exp25", "> 25 years"), ("prov_yes", "provision")]:
    tab = [[int(N[v].sum()), int(len(N) - N[v].sum())], [int(I[v].sum()), int(len(I) - I[v].sum())]]
    log(f"   {lab}: no intention vs interested, Fisher p = {stats.fisher_exact(tab)[1]:.3f}")
log(f"   competence, no intention vs interested: Mann-Whitney p = {stats.mannwhitneyu(N.competence_prof.dropna(), I.competence_prof.dropna()).pvalue:.3f}")

# ============================================================================= 2  threshold with the full covariate set
log("\n=== 2  Threshold with the full covariate set of the Table 2 ===")
d = df[df.stage.notna()].dropna(subset=["age4", "competence_prof", "ambulatory"]).copy()
d["stage0"] = (d.stage == 0).astype(int); d["interested"] = (d.stage == 1).astype(int)
COV_A = "C(age4, Treatment('<40')) + male + competence_prof + ambulatory"
COV_F = COV_A + " + trained + private_ai + C(provision, Treatment('no'))"


def fit(f, data):
    return smf.logit(f, data=data).fit(disp=0, maxiter=200)


def orci(m, t):
    ci = np.exp(m.conf_int().loc[t]); return f"{np.exp(m.params[t]):.2f} ({ci[0]:.2f}-{ci[1]:.2f}), p = {m.pvalues[t]:.3f}"


log(f"n = {len(d)}; stage 0: {int(d.stage0.sum())}")
log(f"{'option':24s} {'stage 0 vs 1-4, Model A':38s} {'stage 0 vs 1-4, full set':38s} {'interested vs none, non-users, full set':40s}")
for k in ["Ethics_and_law", "Critical_appraisal", "Data_protection", "gov_any", "Diagnostic_systems", "Large_language_models", "Commercial_AI_vendors", "Specialised", "Introductory"]:
    d["y"] = d[k]
    mA = fit("y ~ stage0 + " + COV_A, d); mF = fit("y ~ stage0 + " + COV_F, d)
    nu = d[d.stage <= 1]
    mI = fit("y ~ interested + " + COV_F, nu)
    log(f"{k:24s} {orci(mA, 'stage0'):38s} {orci(mF, 'stage0'):38s} {orci(mI, 'interested'):40s}")
# does private AI use explain the threshold?  threshold among private users and among non-private users separately
log("\nThreshold within strata of private AI use (ethics and law; any governance topic), unadjusted %:")
for v in ["Ethics_and_law", "gov_any", "Diagnostic_systems", "Commercial_AI_vendors"]:
    for pa in [0, 1]:
        s = d[d.private_ai == pa]
        a, b = s[s.stage0 == 1][v], s[s.stage0 == 0][v]
        log(f"   {v:22s} private AI use = {pa}: stage 0 {a.mean() * 100:.1f}% (n = {len(a)}) vs stages 1-4 {b.mean() * 100:.1f}% (n = {len(b)})")

log("\n=== Checksums ===")
checks = {"no intention n": (len(N), 63), "interested n": (len(I), 86), "users n": (len(U), 192), "ethics named in no-intention group": (int(N.Ethics_and_law.sum()), 19),
          "critical appraisal in no-intention group": (int(N.Critical_appraisal.sum()), 20)}
bad = [k for k, (o, e) in checks.items() if o != e]
for k, (o, e) in checks.items():
    log(f"{'OK  ' if o == e else 'FAIL'} {k}: {o} (expected {e})")
assert not bad
log("\nAll checksums OK.")
LOG.close()
