"""
item_screen.py  --  applicability screen of the 19 application items (Supplementary Note 8,
Table S8-1) recomputed at full precision from the data, so that the table is rounded in one step (a CSV rounded to two decimals
would give 41.8% instead of 41.9% for the 'unclear' rate of clinical decision support, 149 of 356). Sample: respondents with at least 15 valid items of 19 (n = 356); sectors as in Table S8-1
(ambulatory = practices and MVZ; hospital = inpatient and outpatient).

Usage: python item_screen.py <analysis_dataset.csv> <outdir>
"""
import os
import sys

import pandas as pd

INFILE, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
df = pd.read_csv(INFILE)
COLS = df.columns
ITEMS = [(i + 1, COLS[18 + i]) for i in range(19)]
LABELS = {1: "Practice/hospital IS", 2: "Electronic patient record (ePA)", 3: "e-medication list", 4: "e-sick note (eAU)", 5: "PACS / imaging", 6: "Laboratory IS",
          7: "Clinical decision support", 8: "DiGA", 9: "Health apps", 10: "Phone assistance", 11: "Video consultation", 12: "Tele-rounds", 13: "Tele-/home monitoring",
          14: "Speech recognition", 15: "Digital pre-registration", 16: "Triage tools", 17: "Medication safety (AMTS)", 18: "Automated duty scheduling", 19: "Knowledge management"}
raw = df[[c for _, c in ITEMS]].copy()
raw.columns = [n for n, _ in ITEMS]
valid = raw.notna().sum(axis=1)
lca = raw[valid >= 15]
assert len(lca) == 356
setting = df.loc[lca.index, COLS[11]].fillna("")
amb = setting.str.startswith("Praxis") | setting.str.startswith("MVZ")
hos = setting.str.startswith("Krankenhaus")
USE = ["täglich", "wöchentlich", "monatlich", "seltener"]
rows = []
for n, _ in ITEMS:
    col = lca[n].fillna("")
    rows.append({"no": n, "item": LABELS[n],
                 "wk_daily_all": 100 * col.isin(["täglich", "wöchentlich"]).mean(), "wk_daily_amb": 100 * col[amb].isin(["täglich", "wöchentlich"]).mean(),
                 "wk_daily_hos": 100 * col[hos].isin(["täglich", "wöchentlich"]).mean(), "any_use_all": 100 * col.isin(USE).mean(),
                 "not_avail_all": 100 * (col == "möchte ich, aber nicht verfügbar").mean(), "unclear_all": 100 * (col == "unklar, was das ist").mean(),
                 "n_unclear": int((col == "unklar, was das ist").sum())})
res = pd.DataFrame(rows).sort_values("wk_daily_all")
res.to_csv(os.path.join(OUTDIR, "item_screen.csv"), index=False, float_format="%.6f")
pd.set_option("display.width", 250)
print(res.round(2).to_string(index=False))
r = res.set_index("item")
checks = {"n ambulatory": (int(amb.sum()), 191), "n hospital": (int(hos.sum()), 150), "CDSS unclear n": (int(r.loc["Clinical decision support", "n_unclear"]), 149),
          "CDSS unclear %": (round(r.loc["Clinical decision support", "unclear_all"], 1), 41.9), "triage unclear %": (round(r.loc["Triage tools", "unclear_all"], 1), 39.9),
          "triage wk-daily amb": (round(r.loc["Triage tools", "wk_daily_amb"], 1), 1.6), "triage wk-daily hos": (round(r.loc["Triage tools", "wk_daily_hos"], 1), 2.7),
          "pre-registration amb": (round(r.loc["Digital pre-registration", "wk_daily_amb"], 1), 2.6), "pre-registration hos": (round(r.loc["Digital pre-registration", "wk_daily_hos"], 1), 25.3),
          "tele-rounds amb": (round(r.loc["Tele-rounds", "wk_daily_amb"], 1), 5.8), "tele-rounds hos": (round(r.loc["Tele-rounds", "wk_daily_hos"], 1), 4.0),
          "CDSS wk-daily all": (round(r.loc["Clinical decision support", "wk_daily_all"], 1), 11.8)}
bad = [k for k, (o, e) in checks.items() if o != e]
for k, (o, e) in checks.items():
    print(f"{'OK  ' if o == e else 'FAIL'} {k}: {o} (expected {e})")
assert not bad, f"CHECKSUMS VIOLATED: {bad}"
print("\nAll checksums OK.")
