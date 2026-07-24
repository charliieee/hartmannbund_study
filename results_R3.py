"""
Results R3 (RQ2): multivariable logistic regression of AI use.

Input:  analysis_dataset.csv (from prepare_data.py)
Output: Table 2 (console + table2.csv), block LR tests,
        sensitivity analyses a-d (supplement table, console)

Specification (as decided):
  Outcome    AI use yes/no (defined status only)
  Person     age (4 cat., ref. <40), sex (ref. female),
             digital competence (1-5, linear), training (ref. none)
  Structure  sector (ambulatory, ref. hospital),
             provision (3 cat., ref. no), digital index (per SD)
  Complete case n = 322; chi2/LR tests; McFadden pseudo-R2.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# =============================================================================
#  CONFIG
# =============================================================================
df = pd.read_csv("analysis_dataset.csv")

PERSON = ["age_40_49", "age_50_59", "age_60plus", "male", "competence_prof", "trained"]
STRUCT = ["ambulatory", "prov_yes", "prov_dontknow", "index_z"]
LABELS = {
    "age_40_49": "Age 40-49 (ref. <40)", "age_50_59": "Age 50-59",
    "age_60plus": "Age >=60", "male": "Male sex (ref. female)",
    "competence_prof": "Digital competence (per point)",
    "trained": "Prior AI training (ref. none)",
    "ambulatory": "Ambulatory sector (ref. hospital)",
    "prov_yes": "Employer provides AI: yes (ref. no)",
    "prov_dontknow": "Employer provides AI: don't know",
    "index_z": "Digital maturity index (per SD)",
}


# =============================================================================
#  HELPERS
# =============================================================================
def build(base, index_col="digital_index", exclude_solo=False, prov_binary=False):
    """Build the analysis DataFrame for one model variant (complete case)."""
    d = base.copy()
    if exclude_solo:
        d = d[d["solo_practice"] == 0]
    d = pd.DataFrame({
        "use": d["ai_use"],
        "age_40_49": (d["age4"] == "40-49").astype(float),
        "age_50_59": (d["age4"] == "50-59").astype(float),
        "age_60plus": (d["age4"] == "60+").astype(float),
        "male": d["male"].astype(float),
        "competence_prof": d["competence_prof"],
        "trained": d["trained"].astype(float),
        "ambulatory": d["ambulatory"],
        "prov_yes": (d["provision"] == "yes").astype(float),
        "prov_dontknow": (d["provision"] == "dont_know").astype(float),
        "index_raw": d[index_col],
    }).dropna()
    if prov_binary:
        d = d.drop(columns="prov_dontknow")          # dont_know -> reference "no"
    d["index_z"] = (d["index_raw"] - d["index_raw"].mean()) / d["index_raw"].std()
    return d.drop(columns="index_raw")


def fit(d):
    X = sm.add_constant(d.drop(columns="use"))
    return sm.Logit(d["use"], X).fit(disp=0)


def lr_test(d, full, block):
    """Likelihood-ratio test: full vs. predictor set reduced by 'block'."""
    red = fit(d.drop(columns=[b for b in block if b in d.columns]))
    lr = 2 * (full.llf - red.llf)
    k = len([b for b in block if b in d.columns])
    return lr, k, stats.chi2.sf(lr, k)


def or_table(m):
    ci = np.exp(m.conf_int())
    return pd.DataFrame({
        "OR": np.exp(m.params), "CI_low": ci[0], "CI_high": ci[1], "p": m.pvalues,
    }).drop("const").rename(index=LABELS)


# =============================================================================
#  MAIN MODEL
# =============================================================================
d0 = build(df)
m0 = fit(d0)
print(f"Main model: n = {len(d0)}, users = {int(d0['use'].sum())}, "
      f"McFadden R2 = {m0.prsquared:.3f}\n")
t2 = or_table(m0)
print(t2.round(3).to_string())
for name, block in [("Person block", PERSON), ("Structure block", STRUCT),
                    ("Age block", PERSON[:3])]:
    lr, k, p = lr_test(d0, m0, block)
    print(f"{name}: LR chi2({k}) = {lr:.1f}, p = {p:.4g}")
t2.round(4).to_csv("table2.csv")
print("\nWritten: table2.csv")


# =============================================================================
#  SENSITIVITY ANALYSES a-d
# =============================================================================
print("\nSensitivity analyses (OR [95% CI], p):")
variants = [
    ("a) provision binary",        dict(prov_binary=True)),
    ("b) excluding solo practices", dict(exclude_solo=True)),
    ("c) sector-neutral index",    dict(index_col="digital_index_neutral")),
]
for name, kw in variants:
    d = build(df, **kw)
    m = fit(d)
    t = or_table(m)
    row = lambda lab: (f"{t.loc[lab,'OR']:.2f} [{t.loc[lab,'CI_low']:.2f}-"
                       f"{t.loc[lab,'CI_high']:.2f}], p={t.loc[lab,'p']:.3f}")
    print(f"  {name} (n={len(d)}): provision {row(LABELS['prov_yes'])} | "
          f"index {row(LABELS['index_z'])} | competence {row(LABELS['competence_prof'])}")

# d) index z-standardised within sector (rescale within sector; z-of-z is
#    equivalent to z-of-raw-index, being affine-invariant per group).
d4 = build(df)
d4["index_z"] = d4.groupby("ambulatory")["index_z"].transform(
    lambda x: (x - x.mean()) / x.std())
m4 = fit(d4)
t4 = or_table(m4)
print(f"  d) index z-std. within sector (n={len(d4)}): "
      f"provision {t4.loc[LABELS['prov_yes'],'OR']:.2f}, "
      f"index {t4.loc[LABELS['index_z'],'OR']:.2f} "
      f"(p={t4.loc[LABELS['index_z'],'p']:.3f})")


# =============================================================================
#  CHECKSUMS
# =============================================================================
assert len(d0) == 322, f"n main model {len(d0)} != 322"
assert abs(t2.loc[LABELS["prov_yes"], "OR"] - 11.39) < 0.05, "provision OR deviates"
assert abs(t2.loc[LABELS["index_z"], "OR"] - 1.47) < 0.02, "index OR deviates"
assert abs(m0.prsquared - 0.218) < 0.005, "pseudo-R2 deviates"
print("\nAll checksums OK.")
