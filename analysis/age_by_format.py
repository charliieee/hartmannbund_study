"""
age_by_format.py  --  preferred formats by age group (physicians under 40 versus 50 years or older),
as reported in the RQ3 section and Supplementary Note 5 (Table S5-1). Reproduces the figures of results_R4.py and adds the
on-site row quoted in the text.

Usage: python age_by_format.py <analysis_dataset.csv> <outdir>
"""
import os
import sys

import pandas as pd

INFILE, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
df = pd.read_csv(INFILE)
COLS = df.columns
age = df[COLS[8]]
young = age.isin(["20-29 Jahre", "30-39 Jahre"])
older = age.isin(["50-59 Jahre", "60-69 Jahre", "70 Jahre oder älter"])
q25 = df[COLS[48]].fillna("")
rows = []
for label, key in [("Self-paced online course", "Online-Kurse"), ("Live webinars (interactive)", "Live Online-Webinare"), ("Information nuggets (15-30 min)", "Nuggets"),
                   ("Blended learning", "Blended Learning"), ("On-site sessions / workshops", "Präsenzveranstaltungen")]:
    ky, ko = int(q25[young].str.contains(key, regex=False).sum()), int(q25[older].str.contains(key, regex=False).sum())
    rows.append({"format": label, "under_40": f"{ky}/{int(young.sum())} = {100 * ky / young.sum():.1f}%", "50_or_older": f"{ko}/{int(older.sum())} = {100 * ko / older.sum():.1f}%",
                 "pct_under_40": 100 * ky / young.sum(), "pct_50_or_older": 100 * ko / older.sum()})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUTDIR, "age_by_format.csv"), index=False, float_format="%.4f")
print(res.to_string(index=False))
r = res.set_index("format")
checks = {"self-paced <40": (round(r.loc["Self-paced online course", "pct_under_40"], 1), 77.9), "self-paced 50+": (round(r.loc["Self-paced online course", "pct_50_or_older"], 1), 65.5),
          "nuggets <40": (round(r.loc["Information nuggets (15-30 min)", "pct_under_40"], 1), 45.0), "nuggets 50+": (round(r.loc["Information nuggets (15-30 min)", "pct_50_or_older"], 1), 29.2),
          "on-site <40": (round(r.loc["On-site sessions / workshops", "pct_under_40"], 1), 35.9), "on-site 50+": (round(r.loc["On-site sessions / workshops", "pct_50_or_older"], 1), 36.3),
          "n under 40": (int(young.sum()), 131), "n 50+": (int(older.sum()), 171)}
bad = [k for k, (o, e) in checks.items() if o != e]
for k, (o, e) in checks.items():
    print(f"{'OK  ' if o == e else 'FAIL'} {k}: {o} (expected {e})")
assert not bad, f"CHECKSUMS VIOLATED: {bad}"
print("\nAll checksums OK.")
