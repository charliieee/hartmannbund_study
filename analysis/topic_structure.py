"""
topic_structure.py  --  is there a structure of topic groups (Q28), and do the four adoption groups
differ on it?

1  Topic block (10 items, max. three selections).  Principal component analysis on the Pearson correlations of the 0/1
   indicators and on tetrachoric correlations (less affected by the selection cap);
   eigenvalues, loadings of the first three components.  Hierarchical clustering of the topics (average linkage on
   1 - tetrachoric r) as a second way to read topic groups.
2  Person scores on the first two Pearson components (standardised indicators x loadings); mean score per adoption
   group with 95% CIs, Kruskal-Wallis, and OLS adjusted for age, sex, competence, sector.
3  Topic-group counts per person from the clustering (number of topics selected in each group), by adoption group.
4  All 32 options together: PCA (Pearson), first three components, group means; checks whether the first component is
   the number of selections (response style).

Usage: python topic_structure.py <analysis_dataset.csv> <outdir>
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats, optimize
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

warnings.filterwarnings("ignore")
INFILE, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
LOG = open(os.path.join(OUTDIR, "topic_structure_log.txt"), "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


df = pd.read_csv(INFILE)
COLS = df.columns
G4 = {"Nein, und aktuell nicht geplant": 0, "Nein, aber geplant/Interesse vorhanden": 1, "Ja, selten/experimentell (einmal pro Monat)": 2, "Ja, gelegentlich (einmal pro Woche)": 2, "Ja, regelmäßig (mehrfach pro Woche)": 3}
GLAB = ["no intention", "interested", "non-routine use", "routine use"]
df["g4"] = df[COLS[42]].map(G4)
OPTIONS = [
    ("W", "Self-paced online courses", 48, "Online-Kurse"), ("W", "Live webinars", 48, "Live Online-Webinare"), ("W", "Nuggets (15-30 min)", 48, "Nuggets"),
    ("W", "Blended learning", 48, "Blended"), ("W", "On-site sessions", 48, "Präsenzveranstaltungen"), ("W", "Short formats (1-2 h)", 49, "Kurzformate"),
    ("W", "Half-day", 49, "Halbtägige"), ("W", "Longitudinal", 49, "Längerfristige"), ("W", "Full-day", 49, "Ganztägige"), ("W", "Multi-day", 49, "Mehrtägige"),
    ("W", "Application-oriented", 50, "Anwendungsorientiert"), ("C", "Introductory", 50, "Einführend"), ("C", "Specialised", 50, "Vertiefend"), ("C", "Strategic", 50, "Strategisch"),
    ("T", "Diagnostic systems", 51, "Diagnosesysteme"), ("T", "Treatment planning", 51, "Therapieplanung"), ("T", "Practical implementation", 51, "Praktische Implementierung"),
    ("T", "Large language models", 51, "Sprachmodellen"), ("T", "Data protection", 51, "Datenmanagement"), ("T", "Diagnostic imaging", 51, "Bildgebung"),
    ("T", "Critical appraisal", 51, "Kritische Berwertung"), ("T", "Ethics and law", 51, "Ethische"), ("T", "Patient communication", 51, "Patienteninteraktion"),
    ("T", "AI in medical research", 51, "medizinischen Forschung"), ("P", "Chambers of physicians", 52, "Ärztekammern"), ("P", "Professional associations", 52, "Berufsverbände"),
    ("P", "Employers", 52, "Arbeitgeber"), ("P", "Specialty societies", 52, "Fachgesellschaften"), ("P", "Health insurance assoc.", 52, "Kassenärztliche"),
    ("P", "Universities", 52, "Universitäten"), ("P", "Commercial AI vendors", 52, "Industrieanbieter"), ("P", "Private providers", 52, "private Weiterbildungsinstitute"),
]
ind = pd.DataFrame({lab: df[COLS[col]].fillna("").str.contains(key, regex=False).astype(int) for _, lab, col, key in OPTIONS})
TOPICS = [o[1] for o in OPTIONS if o[0] == "T"]
D = df[df.g4.notna()].copy()


def tetrachoric(x, y):
    a = ((x == 1) & (y == 1)).sum() + 0.5; b = ((x == 1) & (y == 0)).sum() + 0.5; c = ((x == 0) & (y == 1)).sum() + 0.5; d = ((x == 0) & (y == 0)).sum() + 0.5
    n = a + b + c + d
    px, py = (a + b) / n, (a + c) / n
    tx, ty = stats.norm.ppf(px), stats.norm.ppf(py)

    def nll(rho):
        p11 = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([tx, ty])
        ps = np.clip([p11, px - p11, py - p11, 1 - px - py + p11], 1e-12, None)
        return -(a * np.log(ps[0]) + b * np.log(ps[1]) + c * np.log(ps[2]) + d * np.log(ps[3]))
    return optimize.minimize_scalar(nll, bounds=(-0.98, 0.98), method="bounded").x


def pca(R, labels, k=3):
    w, v = np.linalg.eigh(R)
    idx = np.argsort(w)[::-1]; w, v = w[idx], v[:, idx]
    load = pd.DataFrame(v[:, :k] * np.sqrt(w[:k]), index=labels, columns=[f"PC{i + 1}" for i in range(k)])
    # sign convention: largest absolute loading positive
    for c in load.columns:
        if load[c].iloc[np.argmax(np.abs(load[c].values))] < 0:
            load[c] *= -1; v[:, int(c[2:]) - 1] *= -1
    return load, w, v


# ============================================================================= 1  topic block
log("=== 1  Topic block (10 items; n = %d with defined group, of whom %d selected at least one topic) ===" % (len(D), int((ind.loc[D.index, TOPICS].sum(axis=1) > 0).sum())))
X = ind.loc[D.index, TOPICS]
log("selection rates (%):", (X.mean() * 100).round(1).to_dict())
Rp = pd.DataFrame(np.corrcoef(X.values.T), index=TOPICS, columns=TOPICS)
Rt = pd.DataFrame(np.eye(10), index=TOPICS, columns=TOPICS)
for i, a_ in enumerate(TOPICS):
    for j, b_ in enumerate(TOPICS):
        if j > i:
            Rt.iloc[i, j] = Rt.iloc[j, i] = tetrachoric(X[a_].values, X[b_].values)
Rt.round(3).to_csv(os.path.join(OUTDIR, "1_tetrachoric_topics.csv")); Rp.round(3).to_csv(os.path.join(OUTDIR, "1_pearson_topics.csv"))
log("\nPearson correlations (0/1 indicators), lower triangle:\n" + Rp.where(np.tril(np.ones(Rp.shape), -1).astype(bool)).round(2).fillna("").to_string())
log("\nTetrachoric correlations, lower triangle:\n" + Rt.where(np.tril(np.ones(Rt.shape), -1).astype(bool)).round(2).fillna("").to_string())
loadP, wP, vP = pca(Rp.values, TOPICS)
loadT, wT, vT = pca(Rt.values, TOPICS)
log(f"\nPearson PCA eigenvalues: {np.round(wP, 2)}  (variance explained PC1-3: {np.round(wP[:3] / wP.sum() * 100, 1)} %)")
log(f"Tetrachoric PCA eigenvalues: {np.round(wT, 2)}  (PC1-3: {np.round(wT[:3] / wT.sum() * 100, 1)} %)")
log("\nLoadings, Pearson PCA:\n" + loadP.round(2).sort_values("PC1").to_string())
log("\nLoadings, tetrachoric PCA:\n" + loadT.round(2).sort_values("PC1").to_string())
loadP.round(4).to_csv(os.path.join(OUTDIR, "1_loadings_pearson.csv")); loadT.round(4).to_csv(os.path.join(OUTDIR, "1_loadings_tetrachoric.csv"))
CHK = {}
for c in ["PC1", "PC2", "PC3"]:
    log(f"agreement Pearson vs tetrachoric {c}: Spearman rho = {stats.spearmanr(loadP[c], loadT[c])[0]:.2f}")
    CHK[f"loading agreement {c}"] = float(stats.spearmanr(loadP[c], loadT[c])[0])
# parallel analysis: eigenvalues of random 0/1 data with the same marginals
rng = np.random.default_rng(42)
sims = []
for _ in range(200):
    Xr = np.column_stack([rng.permutation(X[c].values) for c in TOPICS])
    sims.append(np.sort(np.linalg.eigvalsh(np.corrcoef(Xr.T)))[::-1])
sims = np.array(sims)
CHK["components above chance"] = int((wP > np.percentile(sims, 95, axis=0)).sum())
log(f"parallel analysis (200 permutations): 95th percentile of random eigenvalues {np.round(np.percentile(sims, 95, axis=0)[:4], 2)}; observed {np.round(wP[:4], 2)}; components above chance: {int((wP > np.percentile(sims, 95, axis=0)).sum())}")
# hierarchical clustering of topics
Z = linkage(1 - Rt.values[np.triu_indices(10, 1)], method="average")
for k in [2, 3, 4]:
    cl = fcluster(Z, k, criterion="maxclust")
    groups = {i: [t for t, c in zip(TOPICS, cl) if c == i] for i in sorted(set(cl))}
    log(f"hierarchical clustering (average linkage, 1 - tetrachoric r), k = {k}: {groups}")
fig, ax = plt.subplots(figsize=(6, 3.2))
dendrogram(Z, labels=TOPICS, ax=ax, leaf_rotation=35, leaf_font_size=7, color_threshold=0)
ax.set_ylabel("1 - tetrachoric r"); fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig_topic_dendrogram.png"), dpi=200)

# ============================================================================= 2  person scores by group
log("\n=== 2  Person scores on the first two Pearson components, by adoption group ===")
Xs = (X - X.mean()) / X.std(ddof=1)
S = pd.DataFrame(Xs.values @ vP[:, :2], index=X.index, columns=["PC1", "PC2"])
S["g4"] = D.g4
S["n_topics"] = X.sum(axis=1)
S = S.join(df[["age4", "male", "competence_prof", "ambulatory"]])
log(f"correlation of PC1 score with number of topics selected: r = {np.corrcoef(S.PC1, S.n_topics)[0, 1]:.2f}; PC2: {np.corrcoef(S.PC2, S.n_topics)[0, 1]:.2f}")
for pc in ["PC1", "PC2"]:
    t = S.groupby("g4")[pc].agg(["size", "mean", "sem"])
    t["ci_low"] = t["mean"] - 1.96 * t["sem"]; t["ci_high"] = t["mean"] + 1.96 * t["sem"]
    t.index = GLAB
    kw = stats.kruskal(*[S.loc[S.g4 == g, pc] for g in range(4)])
    log(f"\n{pc}: mean (95% CI) by group; Kruskal-Wallis p = {kw.pvalue:.4f}\n" + t[["size", "mean", "ci_low", "ci_high"]].round(3).to_string())
    m = smf.ols(f"{pc} ~ C(g4, Treatment(0)) + C(age4, Treatment('<40')) + male + competence_prof + ambulatory", S.dropna()).fit()
    red = smf.ols(f"{pc} ~ C(age4, Treatment('<40')) + male + competence_prof + ambulatory", S.dropna()).fit()
    log(f"  adjusted (OLS, n = {int(m.nobs)}): interested vs no intention {m.params['C(g4, Treatment(0))[T.1.0]']:+.3f} (p = {m.pvalues['C(g4, Treatment(0))[T.1.0]']:.3f}); "
        f"non-routine {m.params['C(g4, Treatment(0))[T.2.0]']:+.3f} (p = {m.pvalues['C(g4, Treatment(0))[T.2.0]']:.3f}); routine {m.params['C(g4, Treatment(0))[T.3.0]']:+.3f} (p = {m.pvalues['C(g4, Treatment(0))[T.3.0]']:.3f}); "
        f"group block F-test p = {m.compare_f_test(red)[1]:.4f}")
    CHK[f"{pc} Kruskal-Wallis p"] = float(kw.pvalue)
    CHK[f"{pc} adjusted group F-test p"] = float(m.compare_f_test(red)[1])
    for k, t_ in [("interested", "T.1.0"), ("non-routine", "T.2.0"), ("routine", "T.3.0")]:
        CHK[f"{pc} {k} vs no intention, adjusted difference"] = float(m.params[f"C(g4, Treatment(0))[{t_}]"])
        CHK[f"{pc} {k} vs no intention, adjusted p"] = float(m.pvalues[f"C(g4, Treatment(0))[{t_}]"])
    for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        p = stats.mannwhitneyu(S.loc[S.g4 == a, pc], S.loc[S.g4 == b, pc]).pvalue
        log(f"    {GLAB[a]} vs {GLAB[b]}: Mann-Whitney p = {p:.3f}")
        CHK[f"{pc} {GLAB[a]} vs {GLAB[b]} p"] = float(p)
S.round(4).to_csv(os.path.join(OUTDIR, "2_person_scores_topics.csv"))
fig, ax = plt.subplots(figsize=(5.2, 4.4))
cols = ["#eb6834", "#1baf7a", "#8ab4d8", "#1b4f8a"]
for g in range(4):
    s = S[S.g4 == g]
    ax.plot(s.PC1 + np.random.default_rng(g).normal(0, 0.03, len(s)), s.PC2 + np.random.default_rng(g + 10).normal(0, 0.03, len(s)), "o", color=cols[g], markersize=3, alpha=0.45, markeredgewidth=0, label=f"{GLAB[g]} (n = {len(s)})")
    ax.plot(s.PC1.mean(), s.PC2.mean(), "X", color=cols[g], markersize=11, markeredgecolor="white", markeredgewidth=1)
for t, rr in loadP.iterrows():
    ax.annotate(t, (rr.PC1 * 3, rr.PC2 * 3), fontsize=6.5, color="#333333", ha="center")
    ax.plot([0, rr.PC1 * 3], [0, rr.PC2 * 3], color="#999999", linewidth=0.5)
ax.axhline(0, color="#dddddd", linewidth=0.6); ax.axvline(0, color="#dddddd", linewidth=0.6)
ax.set_xlabel(f"PC1 ({wP[0] / wP.sum() * 100:.0f}%)"); ax.set_ylabel(f"PC2 ({wP[1] / wP.sum() * 100:.0f}%)")
ax.legend(frameon=False, fontsize=6.5, loc="lower left"); ax.set_title("Topic block: respondents (dots), group means (X), loadings (lines)", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "fig_topic_scores.png"), dpi=200)

# ============================================================================= 3  topic-group counts
log("\n=== 3  Topic groups from the clustering (k = 3): number selected per person, by adoption group ===")
cl = fcluster(Z, 3, criterion="maxclust")
groups = {i: [t for t, c in zip(TOPICS, cl) if c == i] for i in sorted(set(cl))}
rows = []
for i, items in groups.items():
    cnt = X[items].sum(axis=1)
    r = {"topic group": " | ".join(items), "k_items": len(items)}
    for g in range(4):
        r[GLAB[g] + " mean"] = cnt[D.g4 == g].mean()
        r[GLAB[g] + " any %"] = (cnt[D.g4 == g] > 0).mean() * 100
    r["KW p"] = stats.kruskal(*[cnt[D.g4 == g] for g in range(4)]).pvalue
    rows.append(r)
t3 = pd.DataFrame(rows)
pd.set_option("display.width", 300)
log(t3.round(3).to_string(index=False))
t3.to_csv(os.path.join(OUTDIR, "3_topic_group_counts.csv"), index=False, float_format="%.4f")

# ============================================================================= 4  all 32 options
log("\n=== 4  All 32 options together: PCA (Pearson) ===")
Xall = ind.loc[D.index]
Rall = np.corrcoef(Xall.values.T)
loadA, wA, vA = pca(Rall, list(Xall.columns), k=3)
log(f"eigenvalues: {np.round(wA[:6], 2)}; variance explained PC1-3: {np.round(wA[:3] / wA.sum() * 100, 1)} %")
for c in ["PC1", "PC2", "PC3"]:
    l = loadA[c].sort_values()
    log(f"{c}: lowest {[(i, round(v, 2)) for i, v in l.head(5).items()]}; highest {[(i, round(v, 2)) for i, v in l.tail(5).items()]}")
Xas = (Xall - Xall.mean()) / Xall.std(ddof=1)
SA = pd.DataFrame(Xas.values @ vA[:, :3], index=Xall.index, columns=["PC1", "PC2", "PC3"])
SA["g4"] = D.g4; SA["n_sel"] = Xall.sum(axis=1)
log(f"correlation PC1 with total number of selections: r = {np.corrcoef(SA.PC1, SA.n_sel)[0, 1]:.2f}; PC2: {np.corrcoef(SA.PC2, SA.n_sel)[0, 1]:.2f}; PC3: {np.corrcoef(SA.PC3, SA.n_sel)[0, 1]:.2f}")
for pc in ["PC1", "PC2", "PC3"]:
    t = SA.groupby("g4")[pc].mean(); t.index = GLAB
    log(f"{pc} mean by group: {t.round(3).to_dict()}; Kruskal-Wallis p = {stats.kruskal(*[SA.loc[SA.g4 == g, pc] for g in range(4)]).pvalue:.4f}")
log(f"total selections by group: {SA.groupby('g4').n_sel.mean().round(2).to_dict()}; Kruskal-Wallis p = {stats.kruskal(*[SA.loc[SA.g4 == g, 'n_sel'] for g in range(4)]).pvalue:.4f}")
# ============================================================================= checksums against Supplementary Note 10 (topic structure)
log("\n=== Checksums ===")
checks = {"components above chance": (CHK["components above chance"], 2, 0),
          "loading agreement PC1": (CHK["loading agreement PC1"], 0.99, 0.005), "loading agreement PC2": (CHK["loading agreement PC2"], 0.99, 0.005),
          "PC1 adjusted difference, interested vs no intention": (CHK["PC1 interested vs no intention, adjusted difference"], 1.01, 0.005),
          "PC1 adjusted difference, non-routine vs no intention": (CHK["PC1 non-routine vs no intention, adjusted difference"], 0.77, 0.005),
          "PC1 adjusted difference, routine vs no intention": (CHK["PC1 routine vs no intention, adjusted difference"], 0.72, 0.005),
          "PC1 adjusted p, routine vs no intention (largest of the three, <= 0.001)": (CHK["PC1 routine vs no intention, adjusted p"], 0.001, 0.001),
          "PC1 interested vs non-routine p (smallest among the other three groups)": (CHK["PC1 interested vs non-routine use p"], 0.089, 0.0005),
          "PC1 non-routine vs routine p (largest among the other three groups)": (CHK["PC1 non-routine use vs routine use p"], 0.785, 0.0005),
          "PC2 Kruskal-Wallis p": (CHK["PC2 Kruskal-Wallis p"], 0.026, 0.0005), "PC2 adjusted group F-test p": (CHK["PC2 adjusted group F-test p"], 0.072, 0.0005)}
bad = []
for k, (obs, exp, tol) in checks.items():
    good = abs(obs - exp) <= tol
    log(f"{'OK  ' if good else 'FAIL'} {k}: {obs:.4f} (expected {exp})")
    if not good:
        bad.append(k)
assert not bad, f"CHECKSUMS VIOLATED: {bad}"
log("\nAll checksums OK.")
LOG.close()
