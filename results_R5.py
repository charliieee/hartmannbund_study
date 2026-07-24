"""
Results R5 (RQ4a): Segment comparisons of training preferences and Figure 3.

Input:  analysis_dataset.csv (from prepare_data.py)
Output: console output, suppl_segment_tests.csv (full test table),
        fig3.png / fig3.pdf (dot plot by taxonomy block)

Method (as prespecified in the manuscript):
  - Segments: users (192), interested non-users (86), no current intention (63)
  - One 3x2 chi-square test per preference option (person-based counting)
  - Cell rule: options with any expected cell count < 5 are not tested
  - Benjamini-Hochberg FDR correction across ALL tested options (one family)
  - Effect sizes: Cramer's V
  - Taxonomy (interpretation only, NOT the FDR family):
      W = delivery-related, C = content-related, P = provider-related
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# =============================================================================
# STEP 1: LOAD DATA AND DEFINE SEGMENTS
# =============================================================================
df = pd.read_csv("analysis_dataset.csv")
C = df.columns
SEGMENTS = ["user", "interested", "no_intention"]
seg = df["segment"]
sub = seg.isin(SEGMENTS)
n_seg = seg.value_counts()
print("Segments:", n_seg[SEGMENTS].to_dict())

# =============================================================================
# STEP 2: DEFINE ALL PREFERENCE OPTIONS
# -----------------------------------------------------------------------------
# (block, dimension, English label, column index, unique substring)
# =============================================================================
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
    ("W", "Scope", "Multi-day", 49, "Mehrtägige"),
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
    ("C", "Topic", "AI in medical research", 51, "medizinischen Forschung"),
    ("P", "Provider", "Chambers of physicians", 52, "Ärztekammern"),
    ("P", "Provider", "Professional associations", 52, "Berufsverbände"),
    ("P", "Provider", "Employers", 52, "Arbeitgeber"),
    ("P", "Provider", "Specialty societies", 52, "Fachgesellschaften"),
    ("P", "Provider", "Health insurance assoc.", 52, "Kassenärztliche"),
    ("P", "Provider", "Universities", 52, "Universitäten"),
    ("P", "Provider", "Commercial AI vendors", 52, "Industrieanbieter"),
    ("P", "Provider", "Private providers", 52, "private Weiterbildungsinstitute"),
]

# =============================================================================
# STEP 3: CHI-SQUARE TESTS, CELL RULE, BH-FDR, CRAMER'S V
# =============================================================================
rows = []
for blk, dim, lab, col, key in OPTIONS:
    y = df[C[col]].fillna("").str.contains(key, regex=False)
    ct = pd.crosstab(seg[sub], y[sub])
    chi2, p, dof, expected = stats.chi2_contingency(ct, correction=False)
    rows.append({
        "block": blk, "dimension": dim, "option": lab,
        "pct_user": y[seg == "user"].mean() * 100,
        "pct_interested": y[seg == "interested"].mean() * 100,
        "pct_no_intention": y[seg == "no_intention"].mean() * 100,
        "cramers_v": np.sqrt(chi2 / ct.values.sum()),
        "chi2": chi2, "p": p, "min_expected": expected.min(),
    })
res = pd.DataFrame(rows)
tested = res["min_expected"] >= 5                    # cell rule
res.loc[tested, "p_adj"] = multipletests(res.loc[tested, "p"],
                                         method="fdr_bh")[1]
res["significant"] = res["p_adj"] < 0.05
res.loc[~tested, "significant"] = False

res.round(4).to_csv("suppl_segment_tests.csv", index=False)
print(f"\nTested: {int(tested.sum())} of {len(res)} options "
      f"(excluded by cell rule: {list(res.loc[~tested, 'option'])})")
print(f"Significant after BH-FDR: {int(res['significant'].sum())}")
print(res[res["significant"]][["option", "pct_user", "pct_interested",
      "pct_no_intention", "cramers_v", "p_adj"]].round(3).to_string(index=False))
print("\nWritten: suppl_segment_tests.csv")

# =============================================================================
# STEP 4: FIGURE 3 (DOT PLOT, GROUPED BY TAXONOMY BLOCK)
# =============================================================================
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"user": "#1b4f8a", "interested": "#1baf7a", "no_intention": "#eb6834"}
MRK = {"user": "o", "interested": "s", "no_intention": "D"}
BLOCK_TITLES = {"W": "Delivery-related",
                "C": "Content-related", "P": "Provider-related"}

plot = res[tested].copy()
fig, ax = plt.subplots(figsize=(7.09, 8.6))
ypos, ylabels, yticks = 0, [], []
for blk in ["W", "C", "P"]:
    ax.text(-2, ypos, BLOCK_TITLES[blk], fontsize=8.5, fontweight="bold",
            ha="right", va="center")
    ypos -= 1
    for _, r in plot[plot["block"] == blk].iterrows():
        star = " *" if r["significant"] else ""
        yticks.append(ypos)
        ylabels.append(r["option"] + star)
        for s, xcol in [("user", "pct_user"), ("interested", "pct_interested"),
                        ("no_intention", "pct_no_intention")]:
            ax.plot(r[xcol], ypos, MRK[s], color=COL[s], markersize=5.5,
                    markeredgewidth=0)
        ypos -= 1
    ypos -= 0.6                                       # gap between blocks

ax.set_yticks(yticks, ylabels)
for lab, tick in zip(ylabels, ax.get_yticklabels()):
    if lab.endswith("*"):
        tick.set_fontweight("bold")
ax.set_xlim(0, 100)
ax.set_xlabel("% of segment naming the option")
handles = [plt.Line2D([], [], marker=MRK[s], color=COL[s], linestyle="",
                      markersize=6, label=lab)
           for s, lab in [("user", "Users (n = 192)"),
                          ("interested", "Interested non-users (n = 86)"),
                          ("no_intention", "No current intention (n = 63)")]]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.5)
ax.set_title("* significant after Benjamini-Hochberg correction",
             loc="right", fontsize=7.5, style="italic")

fig.tight_layout()
fig.savefig("fig3.png", dpi=300, bbox_inches="tight")
fig.savefig("fig3.pdf", bbox_inches="tight")
print("Written: fig3.png, fig3.pdf")

# =============================================================================
# STEP 5: CHECKSUMS
# =============================================================================
assert int(tested.sum()) == 30, "expected 30 tested options"
assert int(res["significant"].sum()) == 7, "expected 7 significant options"
spec = res.loc[res["option"] == "Specialised"].iloc[0]
assert spec["p_adj"] < 0.001, "Specialised should be the top finding"
assert set(res.loc[~tested, "option"]) == {"Multi-day", "AI in medical research"}
print("All checksums OK.")
