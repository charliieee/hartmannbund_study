"""
Data preparation: DataCleaned.csv (n = 362) -> analysis variables.

Encodes all decisions from the analysis sessions (July 2026).
Cleaning of the raw data (405) -> DataCleaned (362) is NOT part of
this script (see cleaning documentation / STROBE flow).

Built-in checksums at the end: the script aborts if a mandatory
key figure is not reproduced.
"""

import numpy as np
import pandas as pd

INFILE = "DataCleaned.csv"
OUTFILE = "analysis_dataset.csv"


# =============================================================================
#  LOAD
# =============================================================================
# The export is cp1252-encoded and semicolon-separated (column names are
# the full German question texts).
df = pd.read_csv(INFILE, encoding="cp1252", sep=";")

# Strip rule (mandatory): the export contains non-breaking spaces (\xa0),
# e.g. "Krankenhaus (ambulant)\xa0". Without stripping, comparisons fail
# silently (e.g. df[C[11]].isin(HOSPITAL)).
df = df.apply(
    lambda col: col.str.replace("\xa0", " ").str.strip()
    if col.dtype == "object" else col
)

C = df.columns  # column access by index (column names are question texts)


# =============================================================================
#  AI USE & SEGMENTS
# =============================================================================
# Decision: show "Weiss nicht" (15) and "Keine Antwort" (6) descriptively,
# but exclude them from all analyses with a defined usage status (n = 341).
use_raw = df[C[42]]
df["ai_use"] = np.where(
    use_raw.str.startswith("Ja", na=False), 1,
    np.where(use_raw.str.startswith("Nein", na=False), 0, np.nan),
)
df["segment"] = np.select(
    [
        use_raw.str.startswith("Ja", na=False),
        use_raw == "Nein, aber geplant/Interesse vorhanden",
        use_raw == "Nein, und aktuell nicht geplant",
    ],
    ["user", "interested", "no_intention"],
    default=None,
)


# =============================================================================
#  PREDICTORS
# =============================================================================
# Age: 4 categories (reference "<40"), as in Table 1/2.
df["age4"] = df[C[8]].map({
    "20-29 Jahre": "<40", "30-39 Jahre": "<40",
    "40-49 Jahre": "40-49", "50-59 Jahre": "50-59",
    "60-69 Jahre": "60+", "70 Jahre oder älter": "60+",
})
df["male"] = (df[C[10]] == "männlich").astype(int)
df["competence_prof"] = df[C[40]].astype(float)          # 1-5
df["trained"] = (df[C[45]] == "Ja").astype(int)

# Sector: binary (reference hospital); public-health/EMS/no-answer = NaN.
AMBULANT = [
    "Praxis (Einzelpraxis)",
    "Praxis (Gemeinschaftspraxis/Berufsausübungsgemeinschaft)",
    "MVZ (Medizinisches Versorgungszentrum)",
]
HOSPITAL = ["Krankenhaus (stationär)", "Krankenhaus (ambulant)"]
df["ambulatory"] = np.where(
    df[C[11]].isin(AMBULANT), 1,
    np.where(df[C[11]].isin(HOSPITAL), 0, np.nan),
)
df["solo_practice"] = (df[C[11]] == "Praxis (Einzelpraxis)").astype(int)

# Employer provision: 3 categories (reference "no").
df["provision"] = np.select(
    [df[C[37]] == "Ja", df[C[37]] == "Nein"],
    ["yes", "no"], default="dont_know",
)


# =============================================================================
#  DIGITAL INDEX
# =============================================================================
# 19 items (columns 18-36), frequency 0-4; "not available"/"unclear" = 0.
# >= 15 valid items required, remaining missings = 0; z-standardisation is
# done later in the respective analysis sample (not here).
FREQ = {
    "täglich": 4, "wöchentlich": 3, "monatlich": 2, "seltener": 1,
    "möchte ich, aber nicht verfügbar": 0, "unklar, was das ist": 0,
}
items = df[C[18:37]].apply(lambda col: col.map(FREQ))
valid = items.notna().sum(axis=1)
df["digital_index"] = items.fillna(0).mean(axis=1).where(valid >= 15)

# Sector-neutral index (sensitivity c): 8 items with |Cohen's d| < 0.25
# between sectors; >= 6 valid items.
NEUTRAL = [18, 20, 23, 24, 29, 33, 34, 35]  # PVS/KIS, eML, LIS, CDSS,
# tele-visit, triage, AMTS, duty scheduling
n_items = df[[C[i] for i in NEUTRAL]].apply(lambda col: col.map(FREQ))
df["digital_index_neutral"] = (
    n_items.fillna(0).mean(axis=1).where(n_items.notna().sum(axis=1) >= 6)
)

# LCA input: 19 items on 3 levels (0 / rare-monthly / weekly-daily),
# only respondents with >= 15 valid items (n = 356), missings = 0.
lca_ok = valid >= 15
for j, i in enumerate(range(18, 37)):
    df[f"lca_item_{j+1:02d}"] = pd.cut(
        items[C[i]].fillna(0), [-1, 0.5, 2.5, 4], labels=[0, 1, 2]
    ).astype(int).where(lca_ok)

# NOTE: The free-text recodings for barriers, training formats and areas of
# use are applied in results_R2.py (the authoritative mappings live there);
# they are intentionally not duplicated here.


# =============================================================================
#  CHECKSUMS
# =============================================================================
# Each entry is (observed, expected); the script aborts on any mismatch.
checks = {
    "n total": (len(df), 362),
    "defined usage status": (df["ai_use"].notna().sum(), 341),
    "users": ((df["ai_use"] == 1).sum(), 192),
    "definitive non-users": ((df["ai_use"] == 0).sum(), 149),
    "segments user/interested/no_intention": (
        tuple(df["segment"].value_counts()[["user", "interested", "no_intention"]]),
        (192, 86, 63),
    ),
    "trained": (df["trained"].sum(), 62),
    "provision yes": ((df["provision"] == "yes").sum(), 105),
    "digital index present": (df["digital_index"].notna().sum(), 356),
    "regression complete case": (
        df[["ai_use", "age4", "competence_prof", "ambulatory", "digital_index"]]
        .dropna().shape[0], 322,
    ),
}
for name, (obs, exp) in checks.items():
    assert tuple(np.atleast_1d(obs)) == tuple(np.atleast_1d(exp)), \
        f"CHECKSUM VIOLATED: {name}: {obs} != {exp}"
    print(f"OK  {name}: {obs}")


# =============================================================================
#  WRITE
# =============================================================================
df.to_csv(OUTFILE, index=False)
print(f"\nWritten: {OUTFILE} ({len(df)} rows)")
