"""
content_measures.py  --  composite content measures and exposure characteristics by adoption group,
reported in the RQ4 section, Table 4 and Fig. 5 and tabulated in Supplementary Note 10 (Table S10-2b).

Per group (no current intention / interested non-users / non-routine use / routine use): share naming any governance topic
(data protection, ethics and law, critical appraisal), any clinical application topic (diagnostic systems, treatment planning,
diagnostic imaging), no topic; private AI use, active information-seeking and passive engagement (item 21); number of
coded areas of AI use among users (item 20; median and mean of five).  Fisher's exact tests for the two adjacent contrasts
that the text reports (interested vs no intention; non-routine vs interested).

Usage: python content_measures.py <analysis_dataset.csv> <outdir>
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

INFILE, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
LOG = open(os.path.join(OUTDIR, "content_measures_log.txt"), "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


df = pd.read_csv(INFILE)
COLS = df.columns
G4 = {"Nein, und aktuell nicht geplant": 0, "Nein, aber geplant/Interesse vorhanden": 1, "Ja, selten/experimentell (einmal pro Monat)": 2,
      "Ja, gelegentlich (einmal pro Woche)": 2, "Ja, regelmäßig (mehrfach pro Woche)": 3}
GLAB = ["No current intention", "Interested non-users", "Non-routine use", "Routine use"]
df["g"] = df[COLS[42]].map(G4)
d = df[df.g.notna()].copy()
assert tuple(d.g.value_counts().sort_index().astype(int)) == (63, 86, 85, 107)

q28 = d[COLS[51]].fillna("")
q21 = d[COLS[44]].fillna("")
q20 = d[COLS[43]].fillna("")
d["gov_any"] = (q28.str.contains("Datenmanagement", regex=False) | q28.str.contains("Ethische", regex=False) | q28.str.contains("Kritische Berwertung", regex=False)).astype(int)
d["clin_any"] = (q28.str.contains("Diagnosesysteme", regex=False) | q28.str.contains("Therapieplanung", regex=False) | q28.str.contains("Bildgebung", regex=False)).astype(int)
TOPICS = ["Bildgebung", "Diagnosesysteme", "Therapieplanung", "Sprachmodellen", "Datenmanagement", "Ethische", "Kritische Berwertung", "Praktische Implementierung", "medizinischen Forschung", "Patienteninteraktion"]
d["n_topics"] = sum(q28.str.contains(t, regex=False).astype(int) for t in TOPICS)
d["no_topic"] = (d.n_topics == 0).astype(int)
d["private"] = q21.str.contains("Ich nutze KI-Anwendungen privat", regex=False).astype(int)
d["active"] = q21.str.contains("Ich informiere mich aktiv", regex=False).astype(int)
d["passive"] = q21.str.contains("Eher gar nicht", regex=False).astype(int)
AREA = ["Administrative Aufgaben", "Diagnostik und Befund", "Behandlungsplanung", "Forschung (z.B.", "Patientenkommunikation"]
d["n_areas"] = sum(q20.str.contains(a, regex=False).astype(int) for a in AREA)

rows = []
for g in range(4):
    s = d[d.g == g]
    users = s[s.g >= 2]
    rows.append({"group": GLAB[g], "n": len(s),
                 "any_governance_topic_pct": 100 * s.gov_any.mean(), "any_clinical_topic_pct": 100 * s.clin_any.mean(), "no_topic_pct": 100 * s.no_topic.mean(),
                 "private_ai_use_pct": 100 * s.private.mean(), "informs_actively_pct": 100 * s.active.mean(), "passive_engagement_pct": 100 * s.passive.mean(),
                 "areas_of_use_median": float(users.n_areas.median()) if len(users) else np.nan, "areas_of_use_mean": users.n_areas.mean() if len(users) else np.nan})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUTDIR, "content_measures.csv"), index=False, float_format="%.4f")
pd.set_option("display.width", 250)
log(res.round(2).to_string(index=False))


def fisher(var, ga, gb):
    a, b = d[d.g == ga], d[d.g == gb]
    ka, kb = int(a[var].sum()), int(b[var].sum())
    _, p = stats.fisher_exact([[ka, len(a) - ka], [kb, len(b) - kb]])
    return ka, kb, p


tests = []
for var, lab in [("gov_any", "any governance topic"), ("clin_any", "any clinical application topic"), ("no_topic", "no topic named"), ("private", "private AI use"), ("passive", "passive engagement"), ("active", "informs actively")]:
    for ga, gb, name in [(1, 0, "interested vs no intention"), (2, 1, "non-routine vs interested"), (3, 2, "routine vs non-routine")]:
        ka, kb, p = fisher(var, ga, gb)
        tests.append({"measure": lab, "contrast": name, "pct_a": 100 * ka / (d.g == ga).sum(), "pct_b": 100 * kb / (d.g == gb).sum(), "p_fisher": p})
        log(f"{lab:32s} {name:28s} {100 * ka / (d.g == ga).sum():5.1f} vs {100 * kb / (d.g == gb).sum():5.1f}  Fisher p = {p:.4f}")
pd.DataFrame(tests).to_csv(os.path.join(OUTDIR, "content_measures_tests.csv"), index=False, float_format="%.4f")
u = d[d.g >= 2]
_, p_areas = stats.mannwhitneyu(u[u.g == 3].n_areas, u[u.g == 2].n_areas)
log(f"areas of use, routine vs non-routine users: median {u[u.g == 3].n_areas.median():.0f} vs {u[u.g == 2].n_areas.median():.0f}, Mann-Whitney p = {p_areas:.4f}")

log("\n=== Checksums ===")
T = pd.DataFrame(tests).set_index(["measure", "contrast"])
r = res.set_index("group")
checks = {"gov any no intention": (round(r.loc["No current intention", "any_governance_topic_pct"], 1), 60.3), "gov any interested": (round(r.loc["Interested non-users", "any_governance_topic_pct"], 1), 34.9),
          "clin any no intention": (round(r.loc["No current intention", "any_clinical_topic_pct"], 1), 49.2), "clin any interested": (round(r.loc["Interested non-users", "any_clinical_topic_pct"], 1), 77.9),
          "gov p int vs noint < 0.005": (float(T.loc[("any governance topic", "interested vs no intention"), "p_fisher"]) < 0.005, True),
          "clin p int vs noint < 0.001": (float(T.loc[("any clinical application topic", "interested vs no intention"), "p_fisher"]) < 0.001, True),
          "private interested": (round(r.loc["Interested non-users", "private_ai_use_pct"], 1), 52.3), "private non-routine": (round(r.loc["Non-routine use", "private_ai_use_pct"], 1), 72.9),
          "private p nonroutine vs int": (round(float(T.loc[("private AI use", "non-routine vs interested"), "p_fisher"]), 3), 0.007),
          "passive interested": (round(r.loc["Interested non-users", "passive_engagement_pct"], 1), 19.8), "passive non-routine": (round(r.loc["Non-routine use", "passive_engagement_pct"], 1), 7.1),
          "active routine": (round(r.loc["Routine use", "informs_actively_pct"], 1), 65.4), "areas median non-routine": (r.loc["Non-routine use", "areas_of_use_median"], 1.0), "areas median routine": (r.loc["Routine use", "areas_of_use_median"], 2.0),
          "no topic no intention": (round(r.loc["No current intention", "no_topic_pct"], 1), 14.3)}
bad = [k for k, (o, e) in checks.items() if o != e]
for k, (o, e) in checks.items():
    log(f"{'OK  ' if o == e else 'FAIL'} {k}: {o} (expected {e})")
assert not bad, f"CHECKSUMS VIOLATED: {bad}"
log("\nAll checksums OK.")
LOG.close()
