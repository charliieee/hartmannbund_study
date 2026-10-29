"""
fig5.py  --  tiered structure for AI-related CME derived from the four adoption groups (numbers from the pipeline; foundation with a governance core,
layers as offers rather than assignments, Tier 2 titled descriptively).  All numbers are computed here and checked.

Q30 values refer to the coding after the authors' review (nonuser_reasons.py with
q30_author_review.csv); layout loosened (wider boxes, shorter reason lines) after the text-density check.

Usage: python fig5.py <analysis_dataset.csv> <q30_interested_coded.csv> <q30_no_intention_coded.csv> <outdir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

INFILE, Q30I, Q30N, OUTDIR = sys.argv[1:5]
os.makedirs(OUTDIR, exist_ok=True)
df = pd.read_csv(INFILE)
COLS = df.columns
G4 = {"Nein, und aktuell nicht geplant": 0, "Nein, aber geplant/Interesse vorhanden": 1, "Ja, selten/experimentell (einmal pro Monat)": 2, "Ja, gelegentlich (einmal pro Woche)": 2, "Ja, regelmäßig (mehrfach pro Woche)": 3}
df["g"] = df[COLS[42]].map(G4)
d = df[df.g.notna()]
n = d.g.value_counts().sort_index().astype(int)


def pct(col, key, g):
    s = d[d.g == g][COLS[col]].fillna("")
    return 100 * s.str.contains(key, regex=False).mean()


def rng(col, key):
    v = [pct(col, key, g) for g in range(4)]
    return min(v), max(v)


AREA = {"ADMIN": "Administrative Aufgaben", "DIAG": "Diagnostik und Befund", "TREAT": "Behandlungsplanung", "RES": "Forschung (z.B.", "PAT": "Patientenkommunikation"}


def breadth(cell):
    return sum(k in str(cell) for k in AREA.values())


q30i = pd.read_csv(Q30I); q30n = pd.read_csv(Q30N)
orient_I = 100 * q30i.codes.fillna("").str.contains("ORIENTATION").mean()
benefit_N = 100 * q30n.codes.fillna("").str.contains("BENEFIT").mean()
relrej_N = 100 * q30n.codes.fillna("").str.contains("RELIABILITY|REJECTION").mean()
q28 = d[COLS[51]].fillna("")
gov_any = (q28.str.contains("Datenmanagement", regex=False) | q28.str.contains("Ethische", regex=False) | q28.str.contains("Kritische Berwertung", regex=False))
clin_any = (q28.str.contains("Diagnosesysteme", regex=False) | q28.str.contains("Therapieplanung", regex=False) | q28.str.contains("Bildgebung", regex=False))
V = {
    "adv": [pct(50, "Vertiefend", g) for g in range(4)], "strat": [pct(50, "Strategisch", g) for g in range(4)], "appl": [pct(50, "Anwendungsorientiert", g) for g in range(4)],
    "intro": rng(50, "Einführend"), "appl_r": rng(50, "Anwendungsorientiert"),
    "diag": [pct(51, "Diagnosesysteme", g) for g in range(4)], "treat": [pct(51, "Therapieplanung", g) for g in range(4)], "impl": [pct(51, "Praktische Implementierung", g) for g in range(4)],
    "llm": [pct(51, "Sprachmodellen", g) for g in range(4)], "imaging": [pct(51, "Bildgebung", g) for g in range(4)], "crit": [pct(51, "Kritische Berwertung", g) for g in range(4)],
    "ethics": [pct(51, "Ethische", g) for g in range(4)], "dp": [pct(51, "Datenmanagement", g) for g in range(4)],
    "gov_any": [100 * gov_any[d.g == g].mean() for g in range(4)], "clin_any": [100 * clin_any[d.g == g].mean() for g in range(4)],
    "vendor": [pct(52, "Industrieanbieter", g) for g in range(4)], "private": [pct(44, "Ich nutze KI-Anwendungen privat", g) for g in range(4)],
    "active": [pct(44, "Ich informiere mich aktiv", g) for g in range(4)],
    "q16_na": pct(39, "Nicht verfügbar", 1), "q16_know": pct(39, "Fehlendes Wissen", 1),
    # 'no perceived benefit' includes the free-text entry 'kein bedarf' mapped to this category (as in Fig. 1d / nonuser_reasons.py)
    "q16_benefit": 100 * d[d.g == 0][COLS[39]].fillna("").apply(lambda c: ("Keine wahrgenommene" in c) or any(t.strip() == "kein bedarf" for t in c.split(";"))).mean(),
    "admin3": pct(43, "Administrative Aufgaben", 3), "pat3": pct(43, "Patientenkommunikation", 3),
    "breadth_med": [d[d.g == g][COLS[43]].apply(breadth).median() for g in (2, 3)],
    # share of all 362 respondents (as reported in the RQ3 section), not only of the 341 with defined use status
    "ethics_all": 100 * df[COLS[51]].fillna("").str.contains("Ethische", regex=False).mean(), "dp_all": 100 * df[COLS[51]].fillna("").str.contains("Datenmanagement", regex=False).mean(),
}
# checks against adoption_groups.py / nonuser_reasons.py outputs
assert tuple(n) == (63, 86, 85, 107)
assert abs(V["adv"][3] - 60.7) < 0.1 and abs(V["adv"][2] - 36.5) < 0.1 and abs(V["adv"][1] - 23.3) < 0.1 and abs(V["adv"][0] - 17.5) < 0.1
assert abs(V["ethics"][0] - 30.2) < 0.1 and abs(V["vendor"][0] - 4.8) < 0.1 and abs(V["gov_any"][0] - 60.3) < 0.1 and abs(V["clin_any"][1] - 77.9) < 0.1
assert abs(V["q16_benefit"] - 30.2) < 0.1 and abs(V["ethics_all"] - 16.6) < 0.1 and abs(V["dp_all"] - 27.9) < 0.1
assert abs(V["private"][2] - 72.9) < 0.1 and abs(orient_I - 14.0) < 0.1 and abs(benefit_N - 25.4) < 0.1 and abs(relrej_N - 22.2) < 0.1
print({k: (np.round(v, 1).tolist() if isinstance(v, list) else (np.round(v, 1) if not isinstance(v, tuple) else tuple(np.round(v, 1)))) for k, v in V.items()})

# ----------------------------------------------------------------------------- figure
plt.rcParams.update({"font.family": "sans-serif", "font.size": 7})
fig = plt.figure(figsize=(7.09, 5.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
L, R = 23, 79          # layer x-extent
layers = [  # (y0, y1, colour, text colour)
    ("T3", 80, 97, "#2a78d6", "white"), ("T2", 61, 78, "#8dbaed", "#0b0b0b"), ("T1", 38, 59, None, None), ("F", 24, 36, "#eeedea", "#0b0b0b"),
]


def box(x0, x1, y0, y1, color):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2", facecolor=color, edgecolor=color, linewidth=0.6))


def txt(x, y, s, size=6.2, weight="normal", color="#0b0b0b", ha="left", va="top"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, linespacing=1.35)


box(L, R, 80, 97, "#2a78d6")
txt(L + 1.5, 95.5, "Tier 3  Specialised and strategic deepening", 7, "bold", "white")
txt(L + 1.5, 91.5, f"Routine use (n = {n[3]})", 6.3, color="white")
txt(L + 1.5, 88, f"Advanced {V['adv'][3]:.0f}% · strategic {V['strat'][3]:.0f}% · application-oriented {V['appl'][3]:.0f}% · introductory still {pct(50, 'Einführend', 3):.0f}%\n"
    f"Diagnostic systems {V['diag'][3]:.0f}% · language models {V['llm'][3]:.0f}% · imaging {V['imaging'][3]:.0f}% · informs actively about AI {V['active'][3]:.0f}%\n"
    f"Areas of AI use: median {V['breadth_med'][1]:.0f} of 5 (documentation {V['admin3']:.0f}%, patient communication {V['pat3']:.0f}%)", 5.8, color="white")

box(L, R, 61, 78, "#8dbaed")
txt(L + 1.5, 76.5, "Tier 2  Application-oriented deepening", 7, "bold")
txt(L + 1.5, 72.5, f"Non-routine use (n = {n[2]})", 6.3)
txt(L + 1.5, 69, f"Application-oriented {V['appl'][2]:.0f}% · advanced {V['adv'][2]:.0f}% · strategic {V['strat'][2]:.0f}%\n"
    f"Treatment planning {V['treat'][2]:.0f}% · language models {V['llm'][2]:.0f}% · diagnostic systems {V['diag'][2]:.0f}%\n"
    f"Areas of AI use: median {V['breadth_med'][0]:.0f} of 5 · private AI use {V['private'][2]:.0f}% · informs actively {V['active'][2]:.0f}%", 5.8)

M = (L + R) / 2
box(L, M - 0.8, 35, 59, "#d5e5f8"); box(M + 0.8, R, 35, 59, "#f8d8c7")
txt(L + 1.5, 57.5, "Tier 1a  Entry through application", 7, "bold")
txt(L + 1.5, 53.8, f"Interested non-users (n = {n[1]})", 6.3)
txt(L + 1.5, 50.5, f"Diagnostic systems {V['diag'][1]:.0f}% · implementation {V['impl'][1]:.0f}%\nTreatment planning {V['treat'][1]:.0f}% · language models {V['llm'][1]:.0f}%\n"
    f"Any governance topic {V['gov_any'][1]:.0f}% (lowest group)\nReasons (Q16): not available {V['q16_na']:.0f}%,\nlack of knowledge {V['q16_know']:.0f}%\nFree text: no orientation where to start {orient_I:.0f}%", 5.5)
txt(M + 2.3, 57.5, "Tier 1b  Entry through appraisal", 7, "bold")
txt(M + 2.3, 53.8, f"Non-users without intention (n = {n[0]})", 6.3)
txt(M + 2.3, 50.5, f"Critical appraisal {V['crit'][0]:.0f}% · ethics and law {V['ethics'][0]:.0f}%\nData protection {V['dp'][0]:.0f}% · any governance topic {V['gov_any'][0]:.0f}%\n"
    f"Clinical topics {V['clin_any'][0]:.0f}% (others {min(V['clin_any'][1:]):.0f}–{max(V['clin_any'][1:]):.0f}%) · vendors {V['vendor'][0]:.0f}%\n"
    f"Reasons (Q16): no perceived benefit {V['q16_benefit']:.0f}%\nFree text: no benefit {benefit_N:.0f}%,\nreliability doubts or rejection {relrej_N:.0f}%", 5.5)

box(L, R, 20, 33, "#eeedea")
txt(L + 1.5, 31.5, "Foundation  ·  offered to all groups", 7, "bold")
txt(L + 1.5, 28, f"Introductory and application-oriented level, named at similar rates in every group\n(introductory {V['intro'][0]:.0f}–{V['intro'][1]:.0f}%, application-oriented {V['appl_r'][0]:.0f}–{V['appl_r'][1]:.0f}%; no group difference after correction)\n"
    f"Governance core (ethics and law, data protection): placed here on normative grounds, not by\npreference (named among the top three topics by {V['ethics_all']:.0f}% and {V['dp_all']:.0f}% of respondents)", 5.5)

# boundaries (left margin)
txt(L - 1.5, 80.5, "Boundary by depth", 6, "bold", "#52514e", ha="right")
txt(L - 1.5, 77.3, f"advanced {V['adv'][2]:.0f}% → {V['adv'][3]:.0f}%\n(BH-adjusted p = 0.007)", 5.5, color="#8a8985", ha="right")
txt(L - 1.5, 61.5, "Boundary by status", 6, "bold", "#52514e", ha="right")
txt(L - 1.5, 58.3, f"first professional use;\n1a and Tier 2 did not differ\non any option;\nprivate AI use {V['private'][1]:.0f}% → {V['private'][2]:.0f}%", 5.5, color="#8a8985", ha="right")
ax.plot([L - 1, L - 1], [79.5, 79.5], color="#52514e")
txt(M, 17.8, f"Boundary by content: governance versus clinical application (any governance topic {V['gov_any'][1]:.0f}% vs {V['gov_any'][0]:.0f}%; any clinical topic {V['clin_any'][1]:.0f}% vs {V['clin_any'][0]:.0f}%); same preferred level", 5.5, color="#8a8985", ha="center")
# axes arrows
ax.annotate("", xy=(R, 14.3), xytext=(L, 14.3), arrowprops=dict(arrowstyle="<->", color="#52514e", linewidth=0.7))
txt(M, 12.8, "Content follows adoption stance", 6, color="#52514e", ha="center")
ax.annotate("", xy=(3.2, 97), xytext=(3.2, 20), arrowprops=dict(arrowstyle="->", color="#52514e", linewidth=0.7))
ax.text(1.3, 60, "Level follows adoption stage", fontsize=6, color="#52514e", rotation=90, ha="center", va="center")
# right-hand bars: advanced training wanted
bx0, bx1 = 85, 96
txt(bx0, 99, "Advanced training wanted", 6.3, "bold")
txt(bx0, 96, "% of group (Q27); rises across\nadoption groups", 5.5, color="#52514e")
for lab, g, y in [("3", 3, 88), ("2", 2, 69), ("1a", 1, 53), ("1b", 0, 45)]:
    v = V["adv"][g]
    ax.add_patch(FancyBboxPatch((bx0, y - 1.3), bx1 - bx0, 2.6, boxstyle="square,pad=0", facecolor="white", edgecolor="#c9c8c3", linewidth=0.5))
    ax.add_patch(FancyBboxPatch((bx0, y - 1.3), (bx1 - bx0) * v / 100, 2.6, boxstyle="square,pad=0", facecolor={3: "#2a78d6", 2: "#8dbaed", 1: "#d5e5f8", 0: "#f8d8c7"}[g], edgecolor="none"))
    txt(bx0 - 0.8, y + 0.9, lab, 6, color="#52514e", ha="right"); txt(bx1 + 0.8, y + 0.9, f"{v:.0f}%", 6)
for x_, l_ in [(bx0, "0"), ((bx0 + bx1) / 2, "50"), (bx1, "100")]:
    ax.plot([x_, x_], [37.5, 38.3], color="#c9c8c3", linewidth=0.5); txt(x_, 37, l_, 5.5, color="#8a8985", ha="center")
ax.plot([bx0, bx1], [38.3, 38.3], color="#c9c8c3", linewidth=0.5)
txt(bx0 - 2.5, 32, "Layers are offers, not\nassignments; the bars show\nwhere demand for depth sits.\nLayer size carries no quantity;\ngroup sizes are given as n.", 5.3, color="#8a8985")
for ext in ["png", "pdf", "svg"]:
    fig.savefig(os.path.join(OUTDIR, f"fig5.{ext}"), dpi=300, bbox_inches="tight")
print("written")
