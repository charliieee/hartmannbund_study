"""
Results R6 (RQ4b): Latent class analysis of the digital work environment,
class profiles, and Figure 4.

Input:  analysis_dataset.csv (from prepare_data.py; lca_item_01..19 prebuilt)
Output: console output (BIC table, class profiles, chi-square tests),
        fig4.png / fig4.pdf (item heatmap + class characteristics)

Model (as prespecified): StepMix categorical LCA, 19 items on 3 levels,
20 random starts, n = 356 (>= 15 valid items, missings coded 0).
Primary model: 3 classes; the BIC-preferred 4-class solution is reported
as a sensitivity analysis (see Results / Supplement).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from stepmix import StepMix

# =============================================================================
# STEP 1: LOAD DATA AND BUILD LCA INPUT
# =============================================================================
df = pd.read_csv("analysis_dataset.csv")
ITEM_COLS = [f"lca_item_{i:02d}" for i in range(1, 20)]
keep = df[ITEM_COLS].notna().all(axis=1)
X = df.loc[keep, ITEM_COLS].astype(int)
sub = df[keep]
print(f"LCA sample: n = {len(X)}")

# =============================================================================
# STEP 2: MODEL SELECTION (BIC, ENTROPY) FOR k = 2..5
# =============================================================================
models = {}
print(f"\n{'k':>3} {'BIC':>9} {'entropy':>8}  class sizes")
for k in range(2, 6):
    m = StepMix(n_components=k, measurement="categorical", n_init=20,
                random_state=42, verbose=0, progress_bar=False)
    m.fit(X)
    ll = m.score(X) * len(X)
    n_par = (k - 1) + k * 19 * 2                     # weights + item params
    bic = -2 * ll + n_par * np.log(len(X))
    pr = m.predict_proba(X)
    entropy = 1 + (pr * np.log(np.clip(pr, 1e-12, None))).sum() / (len(X) * np.log(k))
    sizes = sorted(np.bincount(m.predict(X), minlength=k), reverse=True)
    models[k] = m
    print(f"{k:>3} {bic:>9.0f} {entropy:>8.3f}  {sizes}")

# =============================================================================
# STEP 3: PRIMARY MODEL (k = 3), CLASSES ORDERED BY DIGITAL INDEX
# =============================================================================
cl = pd.Series(models[3].predict(X), index=X.index)
order = sub["digital_index"].groupby(cl).mean().sort_values(ascending=False).index
NAMES = ["Frontrunner", "Intermediate", "Basic"]
cl_named = cl.map({c: NAMES[i] for i, c in enumerate(order)})

defined = sub["ai_use"].notna()
profile = pd.DataFrame({
    "n": cl_named.value_counts(),
    "share_%": (cl_named.value_counts() / len(X) * 100).round(1),
    "digital_index_M": sub["digital_index"].groupby(cl_named).mean().round(2),
    "AI_use_%": (sub.loc[defined, "ai_use"].groupby(cl_named[defined]).mean() * 100).round(1),
    "provision_%": ((sub["provision"] == "yes").groupby(cl_named).mean() * 100).round(1),
    "trained_%": (sub["trained"].groupby(cl_named).mean() * 100).round(1),
}).loc[NAMES]
print("\nClass profiles (3-class primary model):")
print(profile.to_string())

# =============================================================================
# STEP 4: CHI-SQUARE TESTS OF CLASS MEMBERSHIP
# -----------------------------------------------------------------------------
# AI use is evaluated among participants with a defined use status only.
# =============================================================================
for name, var, mask in [
        ("AI use x class", sub["ai_use"], defined),
        ("Employer provision x class", sub["provision"] == "yes", pd.Series(True, index=sub.index)),
        ("Prior training x class", sub["trained"], pd.Series(True, index=sub.index))]:
    ct = pd.crosstab(cl_named[mask], var[mask])
    chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
    print(f"  {name}: chi2({dof}) = {chi2:.1f}, p = {p:.2g} (n = {int(mask.sum())})")

# =============================================================================
# STEP 5: FIGURE 4 (a ITEM HEATMAP | b CLASS CHARACTERISTICS)
# =============================================================================
ITEM_LABELS = ["Practice/hospital IS", "Electronic patient record (ePA)",
               "e-medication list", "e-sick note (eAU)", "PACS / imaging",
               "Laboratory IS", "Clinical decision support", "DiGA",
               "Health apps", "Phone assistance", "Video consultation",
               "Tele-rounds", "Tele-/home monitoring", "Speech recognition",
               "Digital pre-registration", "Triage tools",
               "Medication safety (AMTS)", "Automated duty scheduling",
               "Knowledge management"]
# Mean coded frequency (0-4) per item and class, from the raw 0-4 items.
RAW_ITEM_COLS = df.columns[18:37]
FREQ = {"täglich": 4, "wöchentlich": 3, "monatlich": 2, "seltener": 1,
        "möchte ich, aber nicht verfügbar": 0, "unklar, was das ist": 0}
raw_items = sub[RAW_ITEM_COLS].apply(lambda col: col.map(FREQ)).fillna(0)
heat = np.array([[raw_items.loc[cl_named == g, c].mean() for g in NAMES]
                 for c in RAW_ITEM_COLS])

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.09, 5.6),
                               gridspec_kw={"width_ratios": [1.15, 1]})

im = ax1.imshow(heat, cmap="Blues", vmin=0, vmax=4, aspect="auto")
ax1.set_xticks(range(3), NAMES)
ax1.set_yticks(range(19), ITEM_LABELS)
for i in range(19):
    for j in range(3):
        ax1.text(j, i, f"{heat[i, j]:.1f}", ha="center", va="center",
                 fontsize=6.5, color="white" if heat[i, j] >= 2 else "#0C447C")
ax1.text(-0.62, 1.02, "a  Mean use frequency of 19 digital applications (0-4)",
         transform=ax1.transAxes, ha="left", va="bottom",
         fontsize=9, fontweight="bold")
ax1.spines[["top", "right", "left", "bottom"]].set_visible(False)

x = np.arange(3)
width = 0.26
for i, (lab, col_) in enumerate([("AI use*", "AI_use_%"),
                                 ("Employer provision", "provision_%"),
                                 ("Prior AI training", "trained_%")]):
    ax2.bar(x + (i - 1) * width, profile[col_], width,
            label=lab, color=["#185FA5", "#378ADD", "#B5D4F4"][i])
ax2.set_xticks(x, [f"{g}\nn = {profile.loc[g, 'n']}" for g in NAMES],
               fontsize=7)
ax2.set_ylim(0, 100)
ax2.set_ylabel("%")
ax2.set_title("b  Class characteristics", loc="left", fontsize=9,
              fontweight="bold")
ax2.legend(frameon=False, fontsize=7.5)

fig.tight_layout(w_pad=2.0)
fig.savefig("fig4.png", dpi=300, bbox_inches="tight")
fig.savefig("fig4.pdf", bbox_inches="tight")
print("\nWritten: fig4.png, fig4.pdf")

# =============================================================================
# STEP 6: CHECKSUMS
# =============================================================================
assert len(X) == 356, f"LCA n {len(X)} != 356"
assert sorted(profile["n"]) == [62, 113, 181], f"class sizes {list(profile['n'])}"
assert profile.loc["Frontrunner", "AI_use_%"] > 80
assert profile.loc["Basic", "AI_use_%"] > 45      # the key substantive finding
print("All checksums OK.")
