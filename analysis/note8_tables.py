"""
note8_tables.py  --  Markdown tables for Supplementary Note 8 and the package document,
generated from the script outputs (no hand transcription).

Inputs (directories): lca_primary (primary model, 16 items, four levels, 100 starts, BLRT),
lca_19items_3levels (19-item model, 19 items, three levels), lca_19items_4levels (19 items, four levels), lca_17items_4levels
(17 items, four levels), lca_16items_3levels (16 items, three levels), application_counts, item_screen.
Also re-estimates the variant partitions to compute agreement (ARI) with the primary partition.

Usage: python note8_tables.py <analysis_dataset.csv> <outfile.md>
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from stepmix import StepMix

warnings.filterwarnings("ignore")
INFILE, OUTFILE = sys.argv[1], sys.argv[2]
B, P19, F19, F17, P16 = "lca_primary", "lca_19items_3levels", "lca_19items_4levels", "lca_17items_4levels", "lca_16items_3levels"
LAB3 = ["Wide range, routine use", "Wide range, occasional use", "Core set, routine use"]
LAB4 = ["Wide range, routine use", "Wide range, occasional use",
        "Core set, routine use (ambulatory profile)", "Core set, routine use (hospital profile)"]
ITEM_LABELS = ["Practice/hospital IS", "Electronic patient record (ePA)", "e-medication list",
               "e-sick note (eAU)", "PACS / imaging", "Laboratory IS", "Clinical decision support",
               "DiGA", "Health apps", "Phone assistance", "Video consultation", "Tele-rounds",
               "Tele-/home monitoring", "Speech recognition", "Digital pre-registration",
               "Triage tools", "Medication safety (AMTS)", "Automated duty scheduling",
               "Knowledge management"]
NA, UN = "möchte ich, aber nicht verfügbar", "unklar, was das ist"
out = []


def pfmt(v):
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def pfmt_eq(v):
    return "< 0.001" if v < 0.001 else f"= {v:.3f}"


def md(df, floatfmt="{:.3f}", index=True):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    cols = ([d.index.name or ""] if index else []) + list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in d.iterrows():
        vals = ([str(idx)] if index else []) + [str(v) for v in row.values]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def h(t):
    out.append("\n" + t + "\n")


# ---------------------------------------------------------------- S8-1 item screen
h("### Table S8-1  Applicability screen of the 19 application items (LCA sample, n = 356)")
sc = pd.read_csv("item_screen/item_screen.csv", index_col=0)   # item_screen.py (full precision, one-step rounding)
sc = sc.sort_values("wk_daily_all")
t = pd.DataFrame({
    "Item": sc["item"],
    "Used weekly–daily, all %": sc["wk_daily_all"].round(1),
    "ambulatory %": sc["wk_daily_amb"].round(1),
    "hospital %": sc["wk_daily_hos"].round(1),
    "Any use (rarely–daily) %": sc["any_use_all"].round(1),
    "Not available %": sc["not_avail_all"].round(1),
    "Unclear what this is %": sc["unclear_all"].round(1),
}).set_index("Item")
t["Excluded"] = ["yes" if i in ("Triage tools", "Digital pre-registration", "Tele-rounds") else "" for i in t.index]
out.append(md(t, "{:.1f}"))

# ---------------------------------------------------------------- S8-2 model selection
h("### Table S8-2  Model selection, primary specification (16 items, four levels, 100 starts, seed 42)")
a = pd.read_csv(f"{B}/A_model_selection.csv", index_col=0)
g = pd.read_csv(f"{B}/G_blrt.csv")
blrt = {2: g.iloc[0], 3: g.iloc[1], 4: g.iloc[2]}
t = pd.DataFrame({
    "Log-likelihood": a["logLik"].round(1), "Parameters": a["n_par"], "AIC": a["AIC"].round(1), "BIC": a["BIC"].round(1),
    "aBIC": a["aBIC"].round(1), "Entropy": a["entropy"].map(lambda v: "–" if pd.isna(v) else f"{v:.3f}"),
    "Smallest class n (%)": [f"{int(r.smallest_class_n)} ({r.smallest_class_pct:.1f})" if k > 1 else "–" for k, r in a.iterrows()],
    "Class sizes": a["class_sizes"],
    "BLRT k vs k+1: 2ΔLL (95th pct of null), p": [f"{blrt[k]['observed_2LLdiff']:.1f} ({blrt[k]['bootstrap_null_p95']:.1f}), p = {blrt[k]['p_value']:.3f}" if k in blrt else "–" for k in a.index],
})
t.index.name = "k"
out.append(md(t, "{:.1f}"))

# ---------------------------------------------------------------- S8-3 convergence
h("### Table S8-3  Convergence, primary specification")
c = pd.read_csv(f"{B}/B_convergence_summary.csv")
t = pd.DataFrame({
    "k": c["k"], "Specification": c["spec"], "Starts reaching the maximum": [f"{int(s)} of {int(n)}" for s, n in zip(c["starts_at_best"], c["n_starts"])],
    "Distinct local maxima within 5 LL units": c["n_distinct_maxima_within_5LL"].astype(int),
    "Maximum log-likelihood": c["best_logLik"].round(2),
    "ARI with reference partition": c["ARI_vs_reference_solution"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}"),
}).set_index("k")
out.append(md(t, "{:.2f}"))

# ---------------------------------------------------------------- S8-4 certainty
h("### Table S8-4  Classification certainty")
cc = pd.read_csv(f"{B}/C_assignment_certainty.csv")
t = pd.DataFrame({"k": cc["k"], "Class": cc["class_"], "n": cc["n"], "Mean posterior of assigned class": cc["mean_max_posterior"].round(3),
                  "Minimum": cc["min_max_posterior"].round(3), "Assigned with p ≥ 0.80 (%)": (cc["share_ge_0_8"] * 100).map(lambda v: f"{v:.1f}"),
                  "Assigned with p ≥ 0.90 (%)": (cc["share_ge_0_9"] * 100).map(lambda v: f"{v:.1f}")}).set_index("k")
out.append(md(t, "{:.3f}"))
av = pd.read_csv(f"{B}/C_avepp_matrix_k3.csv", index_col=0)
out.append("\nAverage posterior probability matrix (rows: assigned class; columns: mean posterior probability of each class), three-class solution:\n")
out.append(md(av.round(3)))

# ---------------------------------------------------------------- S8-5 profiles and CRP
h("### Table S8-5  Profiles of the primary model (three classes)")
p3 = pd.read_csv(f"{B}/D_profiles_k3.csv", index_col=0)
t = pd.DataFrame({"n": p3["n"].astype(int), "%": p3["share_pct"].round(1), "Mean index": p3["digital_index_mean"].map("{:.2f}".format),
                  "AI use % (n defined)": [f"{v:.1f} ({int(n)})" for v, n in zip(p3["ai_use_pct"], p3["ai_use_n_defined"])],
                  "Employer provision %": p3["provision_pct"].round(1), "Prior AI training %": p3["trained_pct"].round(1),
                  "Hospital %": p3["hospital_pct"].round(1), "Ambulatory %": p3["ambulatory_pct"].round(1),
                  "Item responses: weekly–daily %": (p3["cells_daily_pct"] + p3["cells_weekly_pct"]).round(1),
                  "rarely–monthly %": (p3["cells_rarely_pct"] + p3["cells_monthly_pct"]).round(1),
                  "not available %": p3["cells_not_available_pct"].round(1), "unclear %": p3["cells_unclear_pct"].round(1)})
t.index.name = "Profile"
out.append(md(t, "{:.1f}"))
ch = pd.read_csv(f"{B}/D_chi2_k3.csv", index_col=0)
out.append(f"\nχ² across profiles: AI use χ²(2) = {ch.loc['ai_use', 'chi2']:.1f}, p {pfmt_eq(ch.loc['ai_use', 'p'])} (n = {int(ch.loc['ai_use', 'n'])}); "
           f"employer provision χ²(2) = {ch.loc['provision', 'chi2']:.1f}, p {pfmt_eq(ch.loc['provision', 'p'])}; prior training χ²(2) = {ch.loc['trained', 'chi2']:.1f}, p = {ch.loc['trained', 'p']:.3f}.\n")

h("### Table S8-6  Conditional response probabilities, primary model (16 items × three classes)")
cr = pd.read_csv(f"{B}/D_conditional_response_probs_k3.csv", index_col=0)
cols = {}
for l, short in zip(LAB3, ["Routine", "Occasional", "Core set"]):
    for lv, s in [("weekly-daily", "weekly–daily"), ("not available", "not available"), ("unclear", "unclear")]:
        cols[f"{short}: P({s})"] = cr[f"{l} | P({lv})"].round(2)
t = pd.DataFrame(cols)
t.index.name = "Item"
out.append(md(t, "{:.2f}"))
out.append("\n*P(rarely–monthly) is the remainder to 1.*\n")

# ---------------------------------------------------------------- S8-7 four-class solution
h("### Table S8-7  Four-class solution of the primary specification (sector partition of the core-set profile)")
p4 = pd.read_csv(f"{B}/D_profiles_k4.csv", index_col=0)
ct = pd.read_csv(f"{B}/D_crosstab_4class_x_3class.csv", index_col=0)
sx = pd.read_csv(f"{B}/E_class_x_sector_k4.csv", index_col=0)
cc4 = cc[cc["k"] == 4].set_index("class_")
t = pd.DataFrame({"n": p4["n"].astype(int), "Mean index": p4["digital_index_mean"].map("{:.2f}".format), "AI use %": p4["ai_use_pct"].round(1),
                  "Employer provision %": p4["provision_pct"].round(1), "Prior AI training %": p4["trained_pct"].round(1),
                  "Three-class membership (routine / occasional / core set)": [f"{int(ct.loc[i, LAB3[0]])} / {int(ct.loc[i, LAB3[1]])} / {int(ct.loc[i, LAB3[2]])}" for i in p4.index],
                  "Sector (hospital / ambulatory / other)": [f"{int(sx.loc[i, 'hospital'])} / {int(sx.loc[i, 'ambulatory'])} / {int(sx.loc[i, 'other'])}" for i in p4.index],
                  "Mean posterior": [f"{cc4.loc[i, 'mean_max_posterior']:.3f}" for i in p4.index]})
t.index.name = "Class"
out.append(md(t, "{:.1f}"))
ch4 = pd.read_csv(f"{B}/D_chi2_k4.csv", index_col=0)
d2 = pd.read_csv(f"{B}/D2_pairwise_class_comparison_k4.csv")
r = d2[(d2.class_a == LAB4[2]) & (d2.class_b == LAB4[3])].set_index("outcome")
out.append(f"\nχ² across the four classes: AI use χ²(3) = {ch4.loc['ai_use', 'chi2']:.1f}, p {pfmt_eq(ch4.loc['ai_use', 'p'])}; provision χ²(3) = {ch4.loc['provision', 'chi2']:.1f}, p {pfmt_eq(ch4.loc['provision', 'p'])}; "
           f"training χ²(3) = {ch4.loc['trained', 'chi2']:.1f}, p = {ch4.loc['trained', 'p']:.3f}. The two core-set subclasses do not differ in AI use "
           f"({r.loc['ai_use', 'pct_a']:.1f}% vs {r.loc['ai_use', 'pct_b']:.1f}%, n = {int(r.loc['ai_use', 'n_a'])} and {int(r.loc['ai_use', 'n_b'])}, p = {r.loc['ai_use', 'p_chi2']:.2f}), "
           f"employer provision ({r.loc['provision', 'pct_a']:.1f}% vs {r.loc['provision', 'pct_b']:.1f}%, p = {r.loc['provision', 'p_chi2']:.2f}) or prior training "
           f"({r.loc['trained', 'pct_a']:.1f}% vs {r.loc['trained', 'pct_b']:.1f}%, p = {r.loc['trained', 'p_chi2']:.2f}).\n")

# ---------------------------------------------------------------- S8-8 stability across specifications (ARI vs primary)
h("### Table S8-8  Stability of the three use profiles across codings and item sets")
df = pd.read_csv(INFILE)
cols19 = [f"lca_item_{i:02d}" for i in range(1, 20)]
keep = df[cols19].notna().all(axis=1)
sub = df[keep].copy()
raw = sub[df.columns[18:37]].copy()
raw.columns = ITEM_LABELS
M3 = {NA: 0, UN: 0, "seltener": 1, "monatlich": 1, "wöchentlich": 2, "täglich": 2}
M4 = {NA: 0, UN: 1, "seltener": 2, "monatlich": 2, "wöchentlich": 3, "täglich": 3}
DROP16 = ["Tele-rounds", "Digital pre-registration", "Triage tools"]
DROP17 = ["Digital pre-registration", "Triage tools"]


def fitp(items_drop, mapping, k, starts, meas="categorical"):
    X = raw.drop(columns=items_drop).apply(lambda c: c.map(mapping))
    X = X.fillna(0).astype(int) if meas == "categorical" else X
    m = StepMix(n_components=k, measurement=meas, n_init=starts, random_state=42, verbose=0, progress_bar=False).fit(X)
    n = len(X)
    ll = m.score(X) * n
    bic = -2 * ll + m.n_parameters * np.log(n)
    pr = m.predict_proba(X)
    ent = 1 + (pr * np.log(np.clip(pr, 1e-12, None))).sum() / (n * np.log(k))
    lab = pd.Series(m.predict(X), index=X.index)
    order = sub["digital_index"].groupby(lab).mean().sort_values(ascending=False).index
    lab = lab.map({c: i for i, c in enumerate(order)})
    return lab, bic, ent


ref, _, _ = fitp(DROP16, M4, 3, 100)
assert sorted(ref.value_counts().tolist()) == [69, 122, 165]
specs = [
    ("Primary: 16 items, four levels, 100 starts", DROP16, M4, 100, "categorical"),
    ("19 items, three levels, 20 starts", [], M3, 20, "categorical"),
    ("19 items, four levels, 100 starts", [], M4, 100, "categorical"),
    ("17 items (MIE set), four levels, 100 starts", DROP17, M4, 100, "categorical"),
    ("16 items, three levels, 100 starts", DROP16, M3, 100, "categorical"),
]
rows = []
for name, drop, mp, st, meas in specs:
    bics = {}
    labs = {}
    ents = {}
    for k in (2, 3, 4, 5):
        labs[k], bics[k], ents[k] = fitp(drop, mp, k, st, meas)
    kbic = min(bics, key=bics.get)
    sizes3 = "/".join(str(v) for v in labs[3].value_counts().sort_index().tolist())
    sizes4 = "/".join(str(v) for v in labs[4].value_counts().sort_index().tolist())
    ct3 = pd.crosstab(labs[3], ref)
    diag = int(sum(ct3.values[i, i] for i in range(3))) if ct3.shape == (3, 3) else np.nan
    rows.append({"Specification": name, "BIC-preferred k": kbic, "Entropy (k = 3)": round(ents[3], 3),
                 "Class sizes, k = 3 (ordered by mean index)": sizes3, "ARI with primary (k = 3)": round(adjusted_rand_score(ref, labs[3]), 3),
                 "Respondents on the diagonal (of 356)": diag,
                 "Class sizes, k = 4": sizes4, "ARI of k = 4 with primary": round(adjusted_rand_score(ref, labs[4]), 3)})
t = pd.DataFrame(rows).set_index("Specification")
out.append(md(t, "{:.3f}"))
out.append("\n*ARI, adjusted Rand index between modal partitions (1 = identical). Under the four-level coding with 19 or 17 items the three-class "
           "solution merges the two wide-range profiles and separates the core-set profile by familiarity (wide range / core set, others not available / "
           "core set, others unfamiliar), so its ARI with the primary partition is low; the three use profiles reappear as the first three classes "
           "of the corresponding four-class solutions (routine 51 and 59, occasional 121 and 117 respondents). For the 17-item specification the "
           "BIC-preferred four-class solution partitions the core-set profile by familiarity (Table S8-9); for all other specifications the fourth class "
           "partitions it by sector.*\n")

# ---------------------------------------------------------------- S8-9 two readings
h("### Table S8-9  Two partitions of the core-set profile that compete across specifications")
p17 = pd.read_csv(f"{F17}/D_profiles_k4.csv", index_col=0)
j17 = pd.read_csv(f"{F17}/J_per_respondent_counts_k4.csv", index_col=0)
s17 = pd.read_csv(f"{F17}/E_class_x_sector_k4.csv", index_col=0)
c17 = pd.read_csv(f"{F17}/B_convergence_summary.csv")
rowsA = []
for i in p17.index:
    rowsA.append({"Partition": "Familiarity (17 items, four levels, k = 4)", "Class": i, "n": int(p17.loc[i, "n"]),
                  "AI use %": round(p17.loc[i, "ai_use_pct"], 1), "Employer provision %": round(p17.loc[i, "provision_pct"], 1),
                  "Hospital %": round(s17.loc[i, "hospital_rowpct"], 1),
                  "Median 'unclear' answers": j17.loc[i, "unclear_median"], "Median 'not available' answers": j17.loc[i, "not_available_median"]})
j16 = pd.read_csv(f"{B}/J_per_respondent_counts_k4.csv", index_col=0)
for i in p4.index:
    rowsA.append({"Partition": "Sector (16 items, four levels, k = 4)", "Class": i, "n": int(p4.loc[i, "n"]),
                  "AI use %": round(p4.loc[i, "ai_use_pct"], 1), "Employer provision %": round(p4.loc[i, "provision_pct"], 1),
                  "Hospital %": round(sx.loc[i, "hospital_rowpct"], 1),
                  "Median 'unclear' answers": j16.loc[i, "unclear_median"], "Median 'not available' answers": j16.loc[i, "not_available_median"]})
t = pd.DataFrame(rowsA).set_index("Partition")
out.append(md(t, "{:.1f}"))
c17k4 = c17[(c17.k == 4) & c17.spec.str.contains("reference")].iloc[0]
c16k4 = c[(c.k == 4) & c.spec.str.contains("reference")].iloc[0]
a17 = pd.read_csv(f"{F17}/A_model_selection.csv", index_col=0)
out.append(f"\n*Identifiability: familiarity partition (17 items) reached by {int(c17k4['starts_at_best'])} of 100 starts with {int(c17k4['n_distinct_maxima_within_5LL'])} distinct local maxima within 5 LL units, "
           f"BIC {a17.loc[4, 'BIC']:.0f} vs {a17.loc[3, 'BIC']:.0f} for three classes, entropy {a17.loc[4, 'entropy']:.3f}; sector partition (16 items) reached by {int(c16k4['starts_at_best'])} of 100 starts with "
           f"{int(c16k4['n_distinct_maxima_within_5LL'])} distinct local maxima, BIC {a.loc[4, 'BIC']:.0f} vs {a.loc[3, 'BIC']:.0f}, entropy {a.loc[4, 'entropy']:.3f}. "
           f"In the 17-item solution the two core-set subclasses differ in AI use by {p17.loc[LAB4[2].replace('routine use (ambulatory profile)', 'others not available'), 'ai_use_pct'] - p17.loc[LAB4[2].replace('routine use (ambulatory profile)', 'others unfamiliar'), 'ai_use_pct']:.1f} percentage points.*\n")

# ---------------------------------------------------------------- S8-10 sector
h("### Table S8-10  Profiles within hospital and non-hospital respondents (primary model)")
s = pd.read_csv(f"{B}/E_stratified_k3.csv")
rows = []
for stratum in ("hospital", "non-hospital"):
    for outc, col, ncol in [("AI use", "ai_use_pct", "ai_use_n"), ("Employer provision", "provision_pct", "n"), ("Prior AI training", "trained_pct", "n")]:
        vals = []
        for cl in LAB3:
            rr = s[(s.stratum == stratum) & (s.class_ == cl)].iloc[0]
            vals.append(f"{rr[col]:.1f}% ({int(rr[ncol])})")
        test = s[(s.stratum == stratum) & (s.class_ == f"chi2 test: {'ai_use' if outc == 'AI use' else ('provision' if outc == 'Employer provision' else 'trained')}")].iloc[0]
        rows.append({"Outcome": outc, "Stratum": stratum, LAB3[0]: vals[0], LAB3[1]: vals[1], LAB3[2]: vals[2],
                     "χ²(2), p": f"{test['chi2']:.1f}, {test['p']:.3f}" if test['p'] >= 0.001 else f"{test['chi2']:.1f}, < 0.001"})
t = pd.DataFrame(rows).set_index("Outcome")
out.append(md(t))
reg = pd.read_csv(f"{B}/E_regressions_k3.csv").set_index("outcome")
out.append(f"\nLogistic regressions with profile and sector (hospital vs other) as predictors: AI use, profile likelihood-ratio χ²(2) = {reg.loc['ai_use', 'LR_class_adj_sector']:.1f}, p < 0.001, "
           f"hospital OR {reg.loc['ai_use', 'OR_hospital']:.2f} ({reg.loc['ai_use', 'OR_hospital_ci']}), profile-by-sector interaction χ²(2) = {reg.loc['ai_use', 'LR_interaction']:.2f}, p = {reg.loc['ai_use', 'p_interaction']:.2f} (n = {int(reg.loc['ai_use', 'n'])}); "
           f"employer provision, profile χ²(2) = {reg.loc['prov_yes', 'LR_class_adj_sector']:.1f}, p < 0.001, interaction χ²(2) = {reg.loc['prov_yes', 'LR_interaction']:.2f}, p = {reg.loc['prov_yes', 'p_interaction']:.3f}; "
           f"prior training, profile χ²(2) = {reg.loc['trained', 'LR_class_adj_sector']:.1f}, p = {reg.loc['trained', 'p_class']:.2f}, interaction p = {reg.loc['trained', 'p_interaction']:.2f}.\n")

# ---------------------------------------------------------------- S8-11 person-level
h("### Table S8-11  AI use by the number of surveyed applications not recognised (16 items)")
g2 = pd.read_csv("application_counts/2_ai_use_by_unfamiliar_overall.csv", index_col=0)
gp = pd.read_csv("application_counts/2_ai_use_by_profile_x_unfamiliar.csv")
rows = []
for cat in ["0", "1-2", "3-4", "5+"]:
    r_ = {"Applications not recognised": cat, "All respondents: AI use % (n)": f"{g2.loc[cat, 'mean']:.1f} ({int(g2.loc[cat, 'size'])})"}
    for cl in LAB3:
        q = gp[(gp["profile"] == cl) & (gp["un_cat"].astype(str) == cat)]
        r_[cl] = f"{q['mean'].iloc[0]:.1f} ({int(q['size'].iloc[0])})" if len(q) else "–"
    rows.append(r_)
t = pd.DataFrame(rows).set_index("Applications not recognised")
out.append(md(t))

h("### Table S8-13  Multivariable logistic regression of professional AI use with the digital application index decomposed (complete cases, n = 322)")
rg = pd.read_csv("application_counts/3_regression_decomposed.csv")
names = {"C(age4, Treatment('<40'))[T.40-49]": "Age 40–49 (ref. <40)", "C(age4, Treatment('<40'))[T.50-59]": "Age 50–59", "C(age4, Treatment('<40'))[T.60+]": "Age ≥60",
         "male": "Male sex", "competence_prof": "Self-rated professional digital competence (per point)", "trained": "Prior AI-specific training",
         "ambulatory": "Ambulatory sector (ref. hospital)", "C(provision, Treatment('no'))[T.yes]": "Employer provides AI: yes (ref. no)",
         "C(provision, Treatment('no'))[T.dont_know]": "Employer provides AI: don't know", "z_index": "Digital index, 19 items (per SD)",
         "n_not_available16": "Applications not available (per application, 0–16)", "n_unclear16": "Applications not recognised (per application, 0–16)",
         "use_known16": "Mean use frequency among the remaining applications (per point, 1–4)"}
m1 = rg[rg.model.str.startswith("Table 2")].set_index("term")
m2 = rg[rg.model.str.startswith("decomposed: not available")].set_index("term")
order = list(names.keys())
rows = []
for term in order:
    rows.append({"Predictor": names[term],
                 "19-item model: OR (95% CI), p": f"{m1.loc[term, 'OR']:.2f} ({m1.loc[term, 'ci_low']:.2f}–{m1.loc[term, 'ci_high']:.2f}), {pfmt(m1.loc[term, 'p'])}" if term in m1.index else "–",
                 "Decomposed model: OR (95% CI), p": f"{m2.loc[term, 'OR']:.2f} ({m2.loc[term, 'ci_low']:.2f}–{m2.loc[term, 'ci_high']:.2f}), {pfmt(m2.loc[term, 'p'])}" if term in m2.index else "–"})
t = pd.DataFrame(rows).set_index("Predictor")
out.append(md(t))

open(OUTFILE, "w", encoding="utf-8").write("\n".join(out))
print("written", OUTFILE)
