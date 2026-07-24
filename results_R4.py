"""
Results R4 (RQ3): Training preferences and Figure 2 (panels a-e).

Input:  analysis_dataset.csv (from prepare_data.py)
Output: console output of all R4 numbers, fig2.png / fig2.pdf

Panels: a format | b scope | c level | d topics | e providers
All percentages based on the full analysis sample (n = 362).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
df = pd.read_csv("analysis_dataset.csv")
C = df.columns
n = len(df)
BLUE = "#2a78d6"

# =============================================================================
# STEP 2: DEFINE OPTIONS PER DIMENSION
# -----------------------------------------------------------------------------
# Each entry: (English label, unique substring of the questionnaire option).
# Counting is person-based via str.contains; free-text answers (n = 1 each)
# are not plotted and documented in the Supplement.
# =============================================================================
DIMS = {
    "a  Preferred format": (48, [
        ("Self-paced online courses", "Online-Kurse"),
        ("Live webinars (interactive)", "Live Online-Webinare"),
        ("Information nuggets (15-30 min)", "Nuggets"),
        ("Blended learning", "Blended"),
        ("On-site sessions / workshops", "Präsenzveranstaltungen"),
    ]),
    "b  Preferred scope": (49, [
        ("Short formats (1-2 h)", "Kurzformate"),
        ("Half-day sessions", "Halbtägige"),
        ("Longitudinal (weeks/months)", "Längerfristige"),
        ("Full-day sessions", "Ganztägige"),
        ("Multi-day courses", "Mehrtägige"),
    ]),
    "c  Preferred level": (50, [
        ("Application-oriented", "Anwendungsorientiert"),
        ("Introductory / basics", "Einführend"),
        ("Specialised / advanced", "Vertiefend"),
        ("Strategic / management", "Strategisch"),
    ]),
    "d  Relevant topics": (51, [
        ("AI-based diagnostic systems", "Diagnosesysteme"),
        ("Treatment planning / personalised medicine", "Therapieplanung"),
        ("Practical implementation in daily work", "Praktische Implementierung"),
        ("Large language models (LLMs)", "Sprachmodellen"),
        ("Data management / data protection", "Datenmanagement"),
        ("Diagnostic imaging", "Bildgebung"),
        ("Critical appraisal of AI / studies", "Kritische Berwertung"),
        ("Ethical and legal frameworks", "Ethische"),
        ("Patient interaction / communication", "Patienteninteraktion"),
        ("AI in medical research", "medizinischen Forschung"),
    ]),
    "e  Preferred providers": (52, [
        ("Chambers of physicians", "Ärztekammern"),
        ("Professional associations", "Berufsverbände"),
        ("Employers", "Arbeitgeber"),
        ("Medical specialty societies", "Fachgesellschaften"),
        ("Statutory health insurance assoc.", "Kassenärztliche"),
        ("Universities", "Universitäten"),
        ("Commercial AI vendors", "Industrieanbieter"),
        ("Private education providers", "private Weiterbildungsinstitute"),
    ]),
}

# =============================================================================
# STEP 3: COUNT (PERSON-BASED) AND PRINT
# =============================================================================
counts = {}
for title, (col, options) in DIMS.items():
    counts[title] = [(lab, int(df[C[col]].fillna("").str.contains(key, regex=False).sum()))
                     for lab, key in options]
    print(title)
    for lab, cnt in counts[title]:
        print(f"  {cnt:3d} ({cnt/n*100:4.1f} %)  {lab}")

# =============================================================================
# STEP 4: FIGURE 2 (GRID: a+b / c+d / e FULL WIDTH)
# =============================================================================
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})


def hbar(ax, data, title, denom=n):
    """Horizontal bar panel with count and percentage labels."""
    labels = [lab for lab, _ in data]
    vals = [cnt for _, cnt in data]
    pct = [v / denom * 100 for v in vals]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, pct, height=0.62, color=BLUE)
    ax.set_yticks(y, labels)
    for yi, (v, p) in zip(y, zip(vals, pct)):
        ax.text(p + 1, yi, f"{v} ({p:.1f}%)", va="center", fontsize=7)
    ax.set_xlim(0, max(pct) * 1.35)
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    ax.set_xlabel("% of physicians (n = 362)")


fig = plt.figure(figsize=(7.09, 9.2))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.9, 1.3])
panel_axes = {
    "a  Preferred format": fig.add_subplot(gs[0, 0]),
    "b  Preferred scope": fig.add_subplot(gs[0, 1]),
    "c  Preferred level": fig.add_subplot(gs[1, 0]),
    "d  Relevant topics": fig.add_subplot(gs[1, 1]),
    "e  Preferred providers": fig.add_subplot(gs[2, :]),
}
for title, ax in panel_axes.items():
    hbar(ax, counts[title], title)

fig.tight_layout(w_pad=2.0, h_pad=1.8)
fig.savefig("fig2.png", dpi=300, bbox_inches="tight")
fig.savefig("fig2.pdf", bbox_inches="tight")
print("\nWritten: fig2.png, fig2.pdf")

# =============================================================================
# STEP 5: CHECKSUMS
# -----------------------------------------------------------------------------
# Verified against the manuscript numbers (R4 draft, 2026-07-19). The script
# aborts if any count deviates: your data or coding differ from the paper.
# =============================================================================
EXPECTED = {"Self-paced online courses": 263, "Short formats (1-2 h)": 288,
            "Application-oriented": 331, "AI-based diagnostic systems": 164,
            "AI in medical research": 10, "Chambers of physicians": 238}
flat = {lab: cnt for data in counts.values() for lab, cnt in data}
for lab, exp in EXPECTED.items():
    assert flat[lab] == exp, f"CHECKSUM FAILED: {lab}: {flat[lab]} != {exp}"
print("All checksums OK.")

# =============================================================================
# STEP 6: AGE-STRATIFIED FORMAT PREFERENCES (manuscript sentence)
# -----------------------------------------------------------------------------
# Younger physicians favour online / short formats more strongly. Compare
# "under 40" vs. "50 and older" for the format items (column 48). Percentages
# are within age group; respondents with missing age are excluded.
# =============================================================================
AGE_GROUPS = {"<40": df["age4"] == "<40",
              ">=50": df["age4"].isin(["50-59", "60+"])}
AGE_ITEMS = [("Self-paced online courses", "Online-Kurse"),
             ("Information nuggets (15-30 min)", "Nuggets")]

print("\nAge-stratified format preferences (% within age group):")
age_pct = {}
for lab, key in AGE_ITEMS:
    line = f"  {lab}:"
    for gname, gmask in AGE_GROUPS.items():
        hit = int(df.loc[gmask, C[48]].fillna("").str.contains(key, regex=False).sum())
        denom = int(gmask.sum())
        age_pct[(lab, gname)] = hit / denom * 100
        line += f"  {gname} {hit}/{denom} = {hit/denom*100:.1f}%"
    print(line)

# Checksums against the manuscript numbers.
AGE_EXPECTED = {("Self-paced online courses", "<40"): 77.9,
                ("Self-paced online courses", ">=50"): 65.5,
                ("Information nuggets (15-30 min)", "<40"): 45.0,
                ("Information nuggets (15-30 min)", ">=50"): 29.2}
for kkey, exp in AGE_EXPECTED.items():
    assert abs(age_pct[kkey] - exp) < 0.05, \
        f"CHECKSUM FAILED: {kkey}: {age_pct[kkey]:.1f} != {exp}"
print("Age-stratified checksums OK.")
