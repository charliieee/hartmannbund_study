"""
adoption_groups.py  --  analysis script for the npj Digital Medicine article: training preferences across four adoption groups.
One consistent grouping of Q19:

  0  no current intention            (n = 63)   'Nein, und aktuell nicht geplant'
  1  interested non-user             (n = 86)   'Nein, aber geplant/Interesse vorhanden'
  2  non-routine use                 (n = 85)   'Ja, selten/experimentell (einmal pro Monat)' + 'Ja, gelegentlich (einmal pro Woche)'
  3  routine use                     (n = 107)  'Ja, regelmäßig (mehrfach pro Woche)'
  'Ich weiß es nicht / Bin mir unsicher' (15) and 'Keine Antwort' (6) are described but not compared (n = 341).

A  Omnibus: one 4 x 2 chi-square test per option (no continuity correction), cell rule expected >= 5, BH over the
   tested options as one family, Cramer's V = sqrt(chi2 / n).  Wilson 95% CIs per group.
B  Pairwise: for the omnibus-significant options, six Fisher exact tests (all pairs), BH over this family; risk
   differences with Newcombe 95% CIs.
C  Equivalence bounds (delivery options): per contrast the largest absolute difference not excluded at 90%.
D  Trend over the four ordered groups: Model A (age 4, sex, competence, sector; n = 328) and Model F (+ training,
   private AI use, provision); non-linearity (3 df categorical vs linear, 2 df); decomposition into the three links
   (interested vs none among non-users; users vs non-users; routine vs non-routine among users), threshold 0 vs 1-3,
   breadth of use within users adjusted for routine use.  BH over 30 per family.  Classification rule:
   gradient = BH-significant trend AND no non-linearity (p >= 0.05) AND routine vs non-routine p < 0.05;
   threshold = BH-significant stage-0 contrast AND not gradient (a within-user difference, if any, is reported alongside);
   else invariant.
E  Composition of Q27 (levels) and Q28 (topics) by group.
F  Figure 3 with four groups and Wilson CIs; figure of the option shapes across the four groups.

Usage: python adoption_groups.py <analysis_dataset.csv> <outdir>
"""
import os
import sys
import warnings
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
INFILE, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
LOG = open(os.path.join(OUTDIR, "adoption_groups_log.txt"), "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def save(d, name, **kw):
    d.to_csv(os.path.join(OUTDIR, name), **kw)


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


def pf(p):
    return "" if pd.isna(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


# ----------------------------------------------------------------------------- data
df = pd.read_csv(INFILE)
COLS = df.columns
G4 = {"Nein, und aktuell nicht geplant": 0, "Nein, aber geplant/Interesse vorhanden": 1, "Ja, selten/experimentell (einmal pro Monat)": 2,
      "Ja, gelegentlich (einmal pro Woche)": 2, "Ja, regelmäßig (mehrfach pro Woche)": 3}
GLAB = ["No current intention", "Interested non-users", "Non-routine use", "Routine use"]
GSHORT = ["N", "I", "R-", "R+"]
df["g4"] = df[COLS[42]].map(G4)
df["g4_cat"] = df["g4"].astype("category")
df["private_ai"] = df[COLS[44]].fillna("").astype(str).str.contains("Ich nutze KI-Anwendungen privat", regex=False).astype(int)
n_g = df.g4.value_counts().sort_index().astype(int)
log("=== Groups ===")
log({GLAB[i]: int(n_g[i]) for i in range(4)}, "| n =", int(df.g4.notna().sum()), "| not compared:", df[COLS[42]].isin(["Ich weiß es nicht/Bin mir unsicher", "Keine Antwort"]).sum())
OPTIONS = [
    ("W", "Format", "Self-paced online courses", 48, "Online-Kurse"), ("W", "Format", "Live webinars", 48, "Live Online-Webinare"),
    ("W", "Format", "Nuggets (15-30 min)", 48, "Nuggets"), ("W", "Format", "Blended learning", 48, "Blended"), ("W", "Format", "On-site sessions", 48, "Präsenzveranstaltungen"),
    ("W", "Scope", "Short formats (1-2 h)", 49, "Kurzformate"), ("W", "Scope", "Half-day", 49, "Halbtägige"), ("W", "Scope", "Longitudinal", 49, "Längerfristige"),
    ("W", "Scope", "Full-day", 49, "Ganztägige"), ("W", "Scope", "Multi-day", 49, "Mehrtägige"), ("W", "Orientation", "Application-oriented", 50, "Anwendungsorientiert"),
    ("C", "Depth", "Introductory", 50, "Einführend"), ("C", "Depth", "Specialised", 50, "Vertiefend"), ("C", "Depth", "Strategic", 50, "Strategisch"),
    ("C", "Topic", "Diagnostic systems", 51, "Diagnosesysteme"), ("C", "Topic", "Treatment planning", 51, "Therapieplanung"),
    ("C", "Topic", "Practical implementation", 51, "Praktische Implementierung"), ("C", "Topic", "Large language models", 51, "Sprachmodellen"),
    ("C", "Topic", "Data protection", 51, "Datenmanagement"), ("C", "Topic", "Diagnostic imaging", 51, "Bildgebung"), ("C", "Topic", "Critical appraisal", 51, "Kritische Berwertung"),
    ("C", "Topic", "Ethics and law", 51, "Ethische"), ("C", "Topic", "Patient communication", 51, "Patienteninteraktion"), ("C", "Topic", "AI in medical research", 51, "medizinischen Forschung"),
    ("P", "Provider", "Chambers of physicians", 52, "Ärztekammern"), ("P", "Provider", "Professional associations", 52, "Berufsverbände"), ("P", "Provider", "Employers", 52, "Arbeitgeber"),
    ("P", "Provider", "Specialty societies", 52, "Fachgesellschaften"), ("P", "Provider", "Health insurance assoc.", 52, "Kassenärztliche"), ("P", "Provider", "Universities", 52, "Universitäten"),
    ("P", "Provider", "Commercial AI vendors", 52, "Industrieanbieter"), ("P", "Provider", "Private providers", 52, "private Weiterbildungsinstitute"),
]
ind = pd.DataFrame({lab: df[COLS[col]].fillna("").str.contains(key, regex=False).astype(int) for _, _, lab, col, key in OPTIONS})
D = df[df.g4.notna()].copy()

# ============================================================================= A  omnibus
log("\n=== A  Omnibus over four groups (chi-square, BH over tested options) ===")
rows, counts = [], {}
for blk, dim, lab, col, key in OPTIONS:
    y = ind.loc[D.index, lab]
    ct = pd.crosstab(D.g4, y).reindex(index=range(4)).fillna(0)
    chi2, p, dof, exp = stats.chi2_contingency(ct.values, correction=False)
    k = {g: int(y[D.g4 == g].sum()) for g in range(4)}
    counts[lab] = k
    r = {"block": blk, "dimension": dim, "option": lab}
    for g in range(4):
        lo, hi = wilson(k[g], n_g[g])
        r[f"n_{GSHORT[g]}"], r[f"pct_{GSHORT[g]}"], r[f"lo_{GSHORT[g]}"], r[f"hi_{GSHORT[g]}"] = k[g], 100 * k[g] / n_g[g], 100 * lo, 100 * hi
    r.update({"chi2": chi2, "df": dof, "cramers_v": np.sqrt(chi2 / ct.values.sum()), "p": p, "min_expected": exp.min()})
    rows.append(r)
res = pd.DataFrame(rows)
tested = res.min_expected >= 5
res["tested"] = tested
res.loc[tested, "p_adj"] = multipletests(res.loc[tested, "p"], method="fdr_bh")[1]
res["significant"] = (res.p_adj < 0.05) & tested
pd.set_option("display.width", 300)
log(res[["block", "option", "pct_N", "pct_I", "pct_R-", "pct_R+", "cramers_v", "chi2", "p", "p_adj", "significant"]].round(3).to_string(index=False))
log(f"tested {int(tested.sum())} of 32 (excluded: {list(res.loc[~tested, 'option'])}); BH-significant: {list(res.loc[res.significant, 'option'])}")
log(f"borderline (0.05 <= BH p < 0.10): {[(o, round(p, 3)) for o, p in zip(res.loc[tested & (res.p_adj >= 0.05) & (res.p_adj < 0.10), 'option'], res.loc[tested & (res.p_adj >= 0.05) & (res.p_adj < 0.10), 'p_adj'])]}")
save(res, "A_omnibus.csv", index=False, float_format="%.4f")

# ============================================================================= B  pairwise
log("\n=== B  Pairwise contrasts for omnibus-significant options (Fisher exact; BH over the family; Newcombe 95% CIs) ===")
pairs = list(combinations(range(4), 2))
pw = []
for _, r in res[res.significant].iterrows():
    k = counts[r.option]
    for a, b in pairs:
        _, p = stats.fisher_exact([[k[a], n_g[a] - k[a]], [k[b], n_g[b] - k[b]]])
        d, lo, hi = newcombe(k[a], n_g[a], k[b], n_g[b])
        pw.append({"option": r.option, "contrast": f"{GSHORT[a]} vs {GSHORT[b]}", "pct_a": 100 * k[a] / n_g[a], "pct_b": 100 * k[b] / n_g[b], "rd_pp": 100 * d, "lo": 100 * lo, "hi": 100 * hi, "p_fisher": p})
pw = pd.DataFrame(pw)
pw["p_fisher_adj"] = multipletests(pw.p_fisher, method="fdr_bh")[1]
pw["sig"] = pw.p_fisher_adj < 0.05
log(f"family size {len(pw)}")
log(pw.round(3).to_string(index=False))
save(pw, "B_pairwise.csv", index=False, float_format="%.4f")
for opt in res.loc[res.significant, "option"]:
    s = pw[pw.option == opt]
    log(f"  {opt}: " + "; ".join(f"{c} {d:+.1f} pp{'*' if sg else ''}" for c, d, sg in zip(s.contrast, s.rd_pp, s.sig)))

# ============================================================================= C  equivalence bounds
log("\n=== C  Delivery options: differences with 95% CIs and equivalence bounds (largest |difference| not excluded at 90%) ===")
rd = []
for _, r in res[tested].iterrows():
    k = counts[r.option]
    for a, b in pairs:
        d, lo95, hi95 = newcombe(k[a], n_g[a], k[b], n_g[b])
        _, lo90, hi90 = newcombe(k[a], n_g[a], k[b], n_g[b], z=1.644854)
        rd.append({"block": r.block, "option": r.option, "contrast": f"{GSHORT[a]} vs {GSHORT[b]}", "rd_pp": 100 * d, "lo95": 100 * lo95, "hi95": 100 * hi95, "equiv_bound_pp": 100 * max(abs(lo90), abs(hi90))})
rd = pd.DataFrame(rd)
save(rd, "C_risk_differences.csv", index=False, float_format="%.2f")
w = rd[rd.block == "W"]
eq = w.pivot(index="option", columns="contrast", values="equiv_bound_pp").round(1)
eq["max"] = eq.max(axis=1)
log(eq.to_string())
log(f"delivery contrasts whose 95% CI excludes zero: {[(r.option, r.contrast, round(r.rd_pp, 1)) for _, r in w.iterrows() if r.lo95 > 0 or r.hi95 < 0]}")
log(f"equivalence bounds: single contrasts {w.equiv_bound_pp.min():.1f}-{w.equiv_bound_pp.max():.1f} pp; per option (max over six contrasts) {eq['max'].min():.1f}-{eq['max'].max():.1f} pp")

# ============================================================================= D  trend and decomposition
log("\n=== D  Trend over the four ordered groups, non-linearity, decomposition ===")
COV_A = "C(age4, Treatment('<40')) + male + competence_prof + ambulatory"
COV_F = COV_A + " + trained + private_ai + C(provision, Treatment('no'))"
A = D.dropna(subset=["age4", "competence_prof", "ambulatory"]).copy()
A["g0"] = (A.g4 == 0).astype(int); A["interested"] = (A.g4 == 1).astype(int); A["user"] = (A.g4 >= 2).astype(int); A["routine"] = (A.g4 == 3).astype(int)
AREA_MAP = {"Administrative Aufgaben (z.B. Dokumentation wie Arztbrief schreiben, Abrechnung, etc.)": ["ADMIN"], "Diagnostik und Befund (z.B. Bildanalyse, Mustererkennung, EKG-Auswertung etc.)": ["DIAG"],
            "Behandlungsplanung und Therapieentscheidungen (z.B. Mediaktionsmanagement, personalisierte Therapieempfehlungen)": ["TREAT"], "Forschung (z.B. Datenanalyse, Literaturrecherche, Simulationen)": ["RES"],
            "Patientenkommunikation und Service (z.B. Chatbots, Terminvereinbarung, automatisierte Anamnese-Tools)": ["PAT"], "Recherche": ["RES"], "Recherche, Patienteninformation": ["RES", "PAT"],
            "Übersetzungen für fremdsprachliche Patienten": ["PAT"], "Spracherkennung und Literaturrecherche": ["RES", "ADMIN"], "Behandlungsdokumentation": ["ADMIN"], "OpenEvidence": ["RES"],
            "Diktiergerät": ["ADMIN"], "Marketing": ["ADMIN"], "Dokumentation / spracherkennung": ["ADMIN"], "Unterstützung bei seltenen Diagnosen zur Unterstützung bei Diagnose- und Therapiefindung": ["DIAG"],
            "Bei der Strahlentherapie selbst": ["DIAG"], "zusätzliche Auswertung der Lunge": ["DIAG"]}


def breadth(cell):
    out = set()
    for tok in str(cell).split(";"):
        tok = tok.strip()
        if tok and tok != "nan":
            out.update(AREA_MAP.get(tok, []))
    return len(out)


A.loc[A.user == 1, "breadth"] = A.loc[A.user == 1, COLS[43]].apply(breadth)
log(f"analysis sample n = {len(A)}: {A.g4.value_counts().sort_index().astype(int).to_dict()}")


def fit(f, d):
    return smf.logit(f, data=d).fit(disp=0, maxiter=200)


def orci(m, t):
    ci = np.exp(m.conf_int().loc[t]); return float(np.exp(m.params[t])), float(ci[0]), float(ci[1]), float(m.pvalues[t])


rows = []
for blk, dim, lab, col, key in OPTIONS:
    if not bool(res.loc[res.option == lab, "tested"].iloc[0]):
        continue
    A["y"] = ind.loc[A.index, lab]
    r = {"block": blk, "option": lab}
    mA = fit("y ~ g4 + " + COV_A, A); r["OR_A"], r["loA"], r["hiA"], r["p_A"] = orci(mA, "g4")
    mC = fit("y ~ g4_cat + " + COV_A, A); lr = 2 * (mC.llf - mA.llf); r["p_nonlin"] = stats.chi2.sf(lr, 2)
    mF = fit("y ~ g4 + " + COV_F, A); r["OR_F"], r["loF"], r["hiF"], r["p_F"] = orci(mF, "g4")
    r["OR_priv"], _, _, r["p_priv"] = orci(mF, "private_ai")
    mT = fit("y ~ g0 + " + COV_A, A); r["OR_thr"], r["lo_thr"], r["hi_thr"], r["p_thr"] = orci(mT, "g0")
    mTF = fit("y ~ g0 + " + COV_F, A); r["OR_thr_F"], r["lo_thr_F"], r["hi_thr_F"], r["p_thr_F"] = orci(mTF, "g0")
    mI = fit("y ~ interested + " + COV_A, A[A.g4 <= 1]); r["OR_int"], r["lo_int"], r["hi_int"], r["p_int"] = orci(mI, "interested")
    mU = fit("y ~ user + " + COV_A, A); r["OR_user"], r["lo_user"], r["hi_user"], r["p_user"] = orci(mU, "user")
    mR = fit("y ~ routine + " + COV_A, A[A.user == 1]); r["OR_routine"], r["lo_routine"], r["hi_routine"], r["p_routine"] = orci(mR, "routine")
    mB = fit("y ~ breadth + routine + " + COV_A, A[A.user == 1]); r["OR_breadth"], r["lo_breadth"], r["hi_breadth"], r["p_breadth"] = orci(mB, "breadth")
    rows.append(r)
T = pd.DataFrame(rows)
for c in ["p_A", "p_F", "p_thr", "p_thr_F"]:
    T[c + "_adj"] = multipletests(T[c], method="fdr_bh")[1]
gradient = (T.p_A_adj < 0.05) & (T.p_nonlin >= 0.05) & (T.p_routine < 0.05)
threshold = (T.p_thr_adj < 0.05) & ~gradient
T["pattern"] = np.select([gradient, threshold & (T.p_routine < 0.05), threshold], ["gradient", "threshold + routine", "threshold"], default="invariant")
log(T[["block", "option", "OR_A", "loA", "hiA", "p_A_adj", "p_nonlin", "OR_F", "p_F_adj", "OR_thr", "lo_thr", "hi_thr", "p_thr_adj", "OR_thr_F", "p_thr_F_adj", "pattern"]].round(3).to_string(index=False))
log("\nlinks (Model A covariates): interested vs none (non-users) | users vs non-users | routine vs non-routine (users) | per area of use (users, routine held constant)")
log(T[["option", "OR_int", "p_int", "OR_user", "p_user", "OR_routine", "lo_routine", "hi_routine", "p_routine", "OR_breadth", "lo_breadth", "hi_breadth", "p_breadth", "OR_priv", "p_priv"]].round(3).to_string(index=False))
for pat in ["gradient", "threshold", "threshold + routine", "invariant"]:
    log(f"{pat}: {list(T.loc[T.pattern == pat, 'option'])}")
save(T, "D_trend.csv", index=False, float_format="%.4f")

# ============================================================================= E  composition
log("\n=== E  Composition of Q27 and Q28 by group ===")
D["n_levels"] = ind.loc[D.index, ["Introductory", "Application-oriented", "Specialised", "Strategic"]].sum(axis=1)
D["intro_only_depth"] = ((ind.loc[D.index, "Introductory"] == 1) & (ind.loc[D.index, "Specialised"] == 0) & (ind.loc[D.index, "Strategic"] == 0)).astype(int)
D["intro_and_spec"] = ((ind.loc[D.index, "Introductory"] == 1) & (ind.loc[D.index, "Specialised"] == 1)).astype(int)
D["spec_no_intro"] = ((ind.loc[D.index, "Specialised"] == 1) & (ind.loc[D.index, "Introductory"] == 0)).astype(int)
TOP = [o[2] for o in OPTIONS if o[1] == "Topic"]
D["n_topics"] = ind.loc[D.index, TOP].sum(axis=1)
D["at_cap"] = (D.n_topics == 3).astype(int)
D["gov_n"] = ind.loc[D.index, ["Data protection", "Ethics and law", "Critical appraisal"]].sum(axis=1)
D["gov_any"] = (D.gov_n > 0).astype(int)
D["gov_share"] = D.gov_n / D.n_topics.replace(0, np.nan)
comp = D.groupby("g4").agg(n=("g4", "size"), n_levels=("n_levels", "mean"), intro_only_depth_pct=("intro_only_depth", "mean"), intro_and_spec_pct=("intro_and_spec", "mean"),
                           spec_no_intro_pct=("spec_no_intro", "mean"), n_topics=("n_topics", "mean"), at_cap_pct=("at_cap", "mean"), gov_any_pct=("gov_any", "mean"), gov_share=("gov_share", "mean"))
for c in comp.columns:
    if c.endswith("_pct"):
        comp[c] *= 100
comp.index = GLAB
log(comp.round(2).to_string())
for v in ["n_levels", "n_topics", "gov_share"]:
    log(f"  Kruskal-Wallis {v}: p = {stats.kruskal(*[D.loc[D.g4 == g, v].dropna() for g in range(4)]).pvalue:.4f}")
save(comp.round(3), "E_composition.csv")

# ============================================================================= F  figures
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
COL = ["#eb6834", "#1baf7a", "#8ab4d8", "#1b4f8a"]       # no intention, interested, non-routine, routine (light/dark blue for the two user groups)
MRK = ["D", "s", "o", "o"]
OFF = [-0.3, -0.1, 0.1, 0.3]
BLOCK_TITLES = {"W": "Delivery-related", "C": "Content-related", "P": "Provider-related"}
plot = res[tested]
fig, ax = plt.subplots(figsize=(7.09, 9.6))
ypos, ylabels, yticks = 0, [], []
for blk in ["W", "C", "P"]:
    ax.text(-2, ypos, BLOCK_TITLES[blk], fontsize=8.5, fontweight="bold", ha="right", va="center")
    ypos -= 1
    for _, r in plot[plot.block == blk].iterrows():
        # display label: the questionnaire option 'Vertiefend/Spezialisiert' is labelled 'Advanced / specialised' in the figure
        yticks.append(ypos); ylabels.append(("Advanced / specialised" if r.option == "Specialised" else r.option) + (" *" if r.significant else ""))
        if r.significant:
            ax.axhspan(ypos - 0.5, ypos + 0.5, color="#f3f3f1", zorder=0, linewidth=0)
        for g in range(4):
            yy = ypos + OFF[g]
            ax.plot([r[f"lo_{GSHORT[g]}"], r[f"hi_{GSHORT[g]}"]], [yy, yy], color=COL[g], linewidth=0.9, zorder=2)
            ax.plot(r[f"pct_{GSHORT[g]}"], yy, MRK[g], color=COL[g], markersize=3.8, markeredgewidth=0, zorder=3)
        ypos -= 1
    ypos -= 0.6
ax.set_yticks(yticks, ylabels)
for lab, tick in zip(ylabels, ax.get_yticklabels()):
    if lab.endswith("*"):
        tick.set_fontweight("bold")
ax.set_xlim(0, 100); ax.set_ylim(ypos + 0.4, 0.6)
ax.set_xlabel("% of group naming the option (Wilson 95% CI)")
ax.grid(axis="x", color="#e5e5e5", linewidth=0.5, zorder=0)
handles = [plt.Line2D([], [], marker=MRK[g], color=COL[g], linestyle="-", linewidth=0.9, markersize=5, label=f"{GLAB[g]} (n = {n_g[g]})") for g in range(4)]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.5)
ax.set_title("* significant after Benjamini-Hochberg correction across the 30 tested options", loc="right", fontsize=7.5, style="italic")
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig4.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(OUTDIR, "fig4.pdf"), bbox_inches="tight")

pick = {"a  Rising with use": ["Specialised", "Strategic"], "b  Distinct without intention": ["Ethics and law", "Commercial AI vendors", "Large language models", "Diagnostic systems", "Critical appraisal"],
        "c  No pattern": ["Introductory", "Application-oriented", "Short formats (1-2 h)", "Chambers of physicians", "Data protection"]}
PAL = ["#1b4f8a", "#eb6834", "#1baf7a", "#6a51a3", "#b15928"]; MK = ["o", "s", "D", "^", "v"]
fig, axes = plt.subplots(1, 3, figsize=(7.09, 4.0), sharey=True)
x = np.arange(4)
for ax, (title, labs) in zip(axes, pick.items()):
    for j, lab in enumerate(labs):
        r = res[res.option == lab].iloc[0]
        pct = np.array([r[f"pct_{GSHORT[g]}"] for g in range(4)]); lo = np.array([r[f"lo_{GSHORT[g]}"] for g in range(4)]); hi = np.array([r[f"hi_{GSHORT[g]}"] for g in range(4)])
        off = (j - (len(labs) - 1) / 2) * 0.07
        ax.errorbar(x + off, pct, yerr=[pct - lo, hi - pct], fmt=MK[j] + "-", color=PAL[j], markersize=4, linewidth=1.2, elinewidth=0.7, capsize=0, markeredgewidth=0, label=lab)
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
    ax.set_xticks(x, ["no intention", "interested", "non-routine", "routine"], fontsize=6.5, rotation=35, ha="right")
    ax.set_ylim(0, 100); ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
    ax.legend(frameon=False, fontsize=6.3, loc="upper left", bbox_to_anchor=(-0.02, -0.22), ncol=1, handlelength=1.6, labelspacing=0.3)
axes[0].set_ylabel("% naming the option (Wilson 95% CI)")
fig.text(0.5, 0.005, f"Adoption group (n = {n_g[0]} / {n_g[1]} / {n_g[2]} / {n_g[3]}); non-routine = about once a month or once a week, routine = several times a week", ha="center", fontsize=6.5)
fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.40, wspace=0.12)
fig.savefig(os.path.join(OUTDIR, "fig4_shapes.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(OUTDIR, "fig4_shapes.pdf"), bbox_inches="tight")

# ============================================================================= Note tables (markdown)
md = [f"| Option | {GLAB[0]} (n = {n_g[0]}) | {GLAB[1]} (n = {n_g[1]}) | {GLAB[2]} (n = {n_g[2]}) | {GLAB[3]} (n = {n_g[3]}) | χ²(3) | Cramér's V | p | BH-adjusted p |", "|---|---|---|---|---|---|---|---|---|"]
for _, r in res.iterrows():
    cells = [f"{r[f'pct_{s}']:.1f} ({r[f'lo_{s}']:.1f}-{r[f'hi_{s}']:.1f})" for s in GSHORT]
    md.append(f"| {r.option}{' *' if r.significant else ''} | " + " | ".join(cells) + f" | {r.chi2:.2f} | {r.cramers_v:.2f} | {pf(r.p)} | {pf(r.p_adj) if r.tested else 'not tested'} |")
open(os.path.join(OUTDIR, "note_table_A_omnibus.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
md = ["| Option | OR per group, Model A (95% CI) | BH p | Non-linearity p | OR per group, Model F (95% CI) | No intention vs rest, OR (95% CI), BH p | Interested vs no intention (non-users), OR (95% CI) | Routine vs non-routine (users), OR (95% CI) | Per area of use (users), OR (95% CI) | Pattern |", "|---|---|---|---|---|---|---|---|---|---|"]
for _, r in T.iterrows():
    md.append(f"| {r.option} | {r.OR_A:.2f} ({r.loA:.2f}-{r.hiA:.2f}) | {pf(r.p_A_adj)} | {r.p_nonlin:.3f} | {r.OR_F:.2f} ({r.loF:.2f}-{r.hiF:.2f}) | {r.OR_thr:.2f} ({r.lo_thr:.2f}-{r.hi_thr:.2f}), {pf(r.p_thr_adj)} | "
              f"{r.OR_int:.2f} ({r.lo_int:.2f}-{r.hi_int:.2f}) | {r.OR_routine:.2f} ({r.lo_routine:.2f}-{r.hi_routine:.2f}) | {r.OR_breadth:.2f} ({r.lo_breadth:.2f}-{r.hi_breadth:.2f}) | {r.pattern} |")
open(os.path.join(OUTDIR, "note_table_D_trend.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

# ============================================================================= checksums
log("\n=== Checksums ===")
g = lambda opt, c: float(res.loc[res.option == opt, c].iloc[0])
checks = {"group sizes": (tuple(int(n_g[i]) for i in range(4)), (63, 86, 85, 107), 0), "tested": (int(tested.sum()), 30, 0),
          "significant": (int(res.significant.sum()), 5, 0), "Specialised pct R+": (g("Specialised", "pct_R+"), 60.7, 0.1), "Specialised pct R-": (g("Specialised", "pct_R-"), 36.5, 0.1),
          "Ethics pct N": (g("Ethics and law", "pct_N"), 30.2, 0.1), "Vendors pct N": (g("Commercial AI vendors", "pct_N"), 4.8, 0.1),
          "Diagnostic systems BH p": (g("Diagnostic systems", "p_adj"), 0.062, 0.001), "Critical appraisal BH p": (g("Critical appraisal", "p_adj"), 0.062, 0.001),
          "Model A n": (len(A), 328, 0)}
bad = []
for k, (obs, exp, tol) in checks.items():
    good = (obs == exp) if isinstance(obs, tuple) else abs(obs - exp) <= tol
    log(f"{'OK  ' if good else 'FAIL'} {k}: {obs if isinstance(obs, (tuple, int)) else round(obs, 4)} (expected {exp})")
    if not good:
        bad.append(k)
assert not bad, f"CHECKSUMS VIOLATED: {bad}"
log("\nAll checksums OK.")
LOG.close()
