"""
regression.py  --  analysis script for the npj Digital Medicine article: the multivariable
logistic regression of professional AI use (Table 2) and Supplementary Note 3.

Decisions (26.10.2026, Perret/Claude):
  Table 2   AI use on age (four categories), sex, self-rated professional digital competence, prior AI-specific
            training, private AI use (Q21) | sector (ambulatory vs hospital), employer provision (yes / no /
            don't know). Complete cases n = 328. The Q13 application items are not in the model.
  Note 3    Table A: robustness of the model (variants of coding and sample; the 16-item counts of
            application_counts.py added). Table B: the 19-item index examined (decomposition, 'unclear' as
            missing, availability index, item sets without AI-based applications, item level).
  Comment 3 Employed physicians only (hospital, MVZ), practice-based only, provision x setting interaction,
            AI use by provision within setting, training among users by route into AI use.

Usage: python regression.py <analysis_dataset.csv> <outdir>
Checksums at the end; the script aborts on any violation.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
INFILE, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
LOG = open(os.path.join(OUTDIR, "regression_log.txt"), "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def save(df_, name, **kw):
    df_.to_csv(os.path.join(OUTDIR, name), **kw)


# ----------------------------------------------------------------------------- constants
ITEM_LABELS = ["Practice/hospital IS", "Electronic patient record (ePA)", "e-medication list", "e-sick note (eAU)",
               "PACS / imaging", "Laboratory IS", "Clinical decision support", "DiGA", "Health apps", "Phone assistance",
               "Video consultation", "Tele-rounds", "Tele-/home monitoring", "Speech recognition", "Digital pre-registration",
               "Triage tools", "Medication safety (AMTS)", "Automated duty scheduling", "Knowledge management"]
DROP16 = ["Tele-rounds", "Digital pre-registration", "Triage tools"]            # the LCA scripts applicability screen
AI2 = ["Clinical decision support", "Speech recognition"]                         # AI-based applications
AI4 = AI2 + ["Phone assistance", "Automated duty scheduling"]
SET16 = [l for l in ITEM_LABELS if l not in DROP16]
SET14 = [l for l in SET16 if l not in AI2]
SET12 = [l for l in SET16 if l not in AI4]
NA, UN = "möchte ich, aber nicht verfügbar", "unklar, was das ist"
FREQ = {"täglich": 4, "wöchentlich": 3, "monatlich": 2, "seltener": 1}
FREQ0 = dict(FREQ, **{NA: 0, UN: 0})
FREQ_UNMISS = dict(FREQ, **{NA: 0})
HOSPITAL = ["Krankenhaus (stationär)", "Krankenhaus (ambulant)"]
MVZ = ["MVZ (Medizinisches Versorgungszentrum)"]
PRACTICE = ["Praxis (Einzelpraxis)", "Praxis (Gemeinschaftspraxis/Berufsausübungsgemeinschaft)"]
AGE4, AGE5 = "C(age4, Treatment('<40'))", "C(age5, Treatment('30-39'))"
PROV3, SEC3 = "C(provision, Treatment('no'))", "C(setting3, Treatment('hospital'))"
ENG3 = "C(engage, Treatment('none/passive'))"
PY, PDK = f"{PROV3}[T.yes]", f"{PROV3}[T.dont_know]"
PERSON = [AGE4, "male", "competence_prof", "trained", "private_ai"]
ACCESS = ["ambulatory", PROV3]
MAIN = " + ".join(PERSON + ACCESS)
PRETTY = {f"{AGE4}[T.40-49]": "Age 40-49 years (ref. <40)", f"{AGE4}[T.50-59]": "Age 50-59 years", f"{AGE4}[T.60+]": "Age >=60 years",
          f"{AGE5}[T.20-29]": "Age 20-29 (ref. 30-39)", f"{AGE5}[T.40-49]": "Age 40-49", f"{AGE5}[T.50-59]": "Age 50-59", f"{AGE5}[T.60+]": "Age >=60",
          "male": "Male sex (ref. female)", "competence_prof": "Self-rated professional digital competence (per point, 1-5)",
          "trained": "Prior AI-specific training (ref. none)", "private_ai": "Private AI use (ref. none)",
          "ambulatory": "Ambulatory sector (ref. hospital)", "mvz": "MVZ (ref. hospital)",
          f"{SEC3}[T.MVZ]": "MVZ (ref. hospital)", f"{SEC3}[T.practice]": "Practice (ref. hospital)",
          PY: "Employer provides AI applications: yes (ref. no)", PDK: "Employer provides AI applications: don't know",
          "prov_yes": "Employer provides AI applications: yes (ref. no, incl. don't know)",
          f"{ENG3}[T.informs only]": "Private engagement with AI: informs only (ref. hardly at all)", f"{ENG3}[T.private use]": "Private engagement with AI: uses AI privately",
          "z_score19": "Composite index, 19 items (per SD)", "z_unmiss19s": "Index with 'unclear' as missing, at least 15 recognised items (per SD)", "z_score16": "Composite score, 16 items (per SD)",
          "z_score14": "Composite score, 14 items without AI-based applications (per SD)", "z_score12": "Composite score, 12 items (per SD)",
          "z_unmiss19": "Index with 'unclear' as missing, 19 items (per SD)", "z_avail19": "Availability index, 19 items (per SD)",
          "z_neutral": "Sector-neutral index, 8 items (per SD)",
          "na16": "Applications not available (per application, 0-16)", "un16": "Applications not recognised (per application, 0-16)",
          "fr16": "Mean use frequency among the remaining applications (per point, 1-4)",
          "na14": "Applications not available (per application, 0-14)", "un14": "Applications not recognised (per application, 0-14)",
          "fr14": "Mean use frequency among the remaining applications, 14 items (per point, 1-4)"}

# ============================================================================= variables
df = pd.read_csv(INFILE)
raw19 = df[df.columns[18:37]].copy()
raw19.columns = ITEM_LABELS
ok = raw19.notna().sum(axis=1) >= 15
df["score19"] = raw19.apply(lambda c: c.map(FREQ0)).fillna(0).mean(axis=1).where(ok)
assert np.allclose(df.score19.fillna(-9), df.digital_index.fillna(-9)), "19-item index not reproduced"
for k, items in [(16, SET16), (14, SET14), (12, SET12)]:
    raw = raw19[items]
    df[f"score{k}"] = raw.apply(lambda c: c.map(FREQ0)).fillna(0).mean(axis=1).where(ok)
    df[f"na{k}"] = (raw == NA).sum(axis=1).where(ok)
    df[f"un{k}"] = (raw == UN).sum(axis=1).where(ok)
    df[f"fr{k}"] = raw.apply(lambda c: c.map(FREQ)).mean(axis=1).where(ok)
m19 = raw19.apply(lambda c: c.map(FREQ_UNMISS))
df["unmiss19"] = m19.mean(axis=1).where(ok)                                    # 'unclear' as missing, sample rule of the 19-item index
df["unmiss19s"] = m19.mean(axis=1).where(m19.notna().sum(axis=1) >= 15)        # 'unclear' as missing, at least 15 recognised items required
df["avail19"] = (1 - (raw19 == NA).sum(axis=1) / raw19.notna().sum(axis=1)).where(ok)
df["setting"] = df[df.columns[11]].str.strip()
df["setting3"] = np.select([df.setting.isin(HOSPITAL), df.setting.isin(MVZ), df.setting.isin(PRACTICE)], ["hospital", "MVZ", "practice"], default="other")
df["employee"] = df.setting.isin(HOSPITAL + MVZ)
df["practice_based"] = df.setting.isin(PRACTICE)
df["prov_yes"] = (df.provision == "yes").astype(int)
q21 = df[df.columns[44]].fillna("").astype(str)
df["private_ai"] = q21.str.contains("Ich nutze KI-Anwendungen privat", regex=False).astype(int)
informs = (q21.str.contains("Ich informiere mich aktiv", regex=False) | q21.str.contains("Ich lese gelegentlich", regex=False) | q21.str.contains("Über soziale Medien", regex=False))
df["engage"] = np.select([df.private_ai == 1, informs], ["private use", "informs only"], default="none/passive")
df["age5"] = df[df.columns[8]].str.strip().map({"20-29 Jahre": "20-29", "30-39 Jahre": "30-39", "40-49 Jahre": "40-49", "50-59 Jahre": "50-59", "60-69 Jahre": "60+", "70 Jahre oder älter": "60+"})

log("=== Variables ===")
log(f"n = {len(df)}; defined AI-use status {int(df.ai_use.notna().sum())}; private AI use {int(df.private_ai.sum())} ({df.private_ai.mean() * 100:.1f}%)")
log(f"engagement (Q21): {df.engage.value_counts().to_dict()}")


# ----------------------------------------------------------------------------- machinery
def prep(cols=(), subset=None):
    need = ["ai_use", "age4", "age5", "male", "competence_prof", "trained", "ambulatory", "provision", "private_ai"] + list(cols)
    d = df if subset is None else df[subset]
    d = d.dropna(subset=need).copy()
    d["ai_use"] = d.ai_use.astype(int)
    d["ambulatory"] = d.ambulatory.astype(int)
    d["mvz"] = (d.setting3 == "MVZ").astype(int)
    for c in cols:
        if c.startswith(("score", "unmiss", "avail", "digital_index_neutral")):
            z = "z_" + ("neutral" if c == "digital_index_neutral" else c)
            d[z] = (d[c] - d[c].mean()) / d[c].std(ddof=1)
    return d


def fit(d, rhs):
    return smf.logit("ai_use ~ " + rhs, d).fit(disp=0, maxiter=500)


def lr(d, rhs, drop):
    full = fit(d, rhs)
    red = fit(d, " + ".join([t for t in rhs.split(" + ") if t not in drop]))
    k = len(full.params) - len(red.params)
    chi2 = 2 * (full.llf - red.llf)
    return chi2, k, stats.chi2.sf(chi2, k)


def fmt(res, t, dp=2):
    if t not in res.params:
        return ""
    o, c, p = np.exp(res.params[t]), np.exp(res.conf_int().loc[t]), res.pvalues[t]
    ps = "< 0.001" if p < 0.001 else f"{p:.3f}"
    return f"{o:.{dp}f} ({c[0]:.{dp}f}-{c[1]:.{dp}f}), p = {ps}"


def ortable(res, model):
    rows = []
    for t in res.params.index[1:]:
        ci = np.exp(res.conf_int().loc[t])
        rows.append(dict(model=model, term=PRETTY.get(t, t), OR=np.exp(res.params[t]), ci_low=ci[0], ci_high=ci[1], p=res.pvalues[t], n=int(res.nobs)))
    return rows


# ============================================================================= 1  Table 2
log("\n=== 1  Table 2: main model ===")
d0 = prep()
m0 = fit(d0, MAIN)
n0, users0 = int(m0.nobs), int(d0.ai_use.sum())
npar = len(m0.params) - 1
log(f"n = {n0} (users {users0}, non-users {n0 - users0}) of {int(df.ai_use.notna().sum())} with defined use status; "
    f"excluded: {int(df.ai_use.notna().sum()) - n0} not in the ambulatory or hospital sector; parameters {npar}; events per parameter {(n0 - users0) / npar:.1f}")
log(f"McFadden pseudo-R2 = {m0.prsquared:.3f}; LL = {m0.llf:.2f}; AIC = {m0.aic:.1f}")
for t in m0.params.index[1:]:
    log(f"   {PRETTY.get(t, t):62s} {fmt(m0, t, 3)}")
bl = {}
for lab, terms in [("person block (age, sex, competence, training, private AI use)", PERSON),
                   ("person block without private AI use (age, sex, competence, training)", PERSON[:-1]),
                   ("access block (sector, provision)", ACCESS), ("age", [AGE4]), ("provision", [PROV3]), ("sector", ["ambulatory"]), ("private AI use", ["private_ai"])]:
    bl[lab] = lr(d0, MAIN, terms)
    log(f"   LR test, {lab}: chi2({bl[lab][1]}) = {bl[lab][0]:.1f}, p = {bl[lab][2]:.3g}")
t2 = pd.DataFrame(ortable(m0, "Table 2"))
t2.to_csv(os.path.join(OUTDIR, "table2.csv"), index=False)
pd.DataFrame([dict(test=k, chi2=v[0], df=v[1], p=v[2]) for k, v in bl.items()]).round(4).to_csv(os.path.join(OUTDIR, "table2_block_tests.csv"), index=False)
# Hosmer-Lemeshow style check (deciles) and discrimination
pred = m0.predict()
auc = stats.mannwhitneyu(pred[d0.ai_use == 1], pred[d0.ai_use == 0]).statistic / (users0 * (n0 - users0))
log(f"   discrimination: AUC = {auc:.3f}")
# descriptives for the Results paragraph
dd = df[df.ai_use.notna()]
log("\nDescriptives for the Results paragraph:")
log(f"   provision: yes {int((df.provision == 'yes').sum())} ({(df.provision == 'yes').mean() * 100:.1f}%), no {int((df.provision == 'no').sum())} ({(df.provision == 'no').mean() * 100:.1f}%), "
    f"don't know {int((df.provision == 'dont_know').sum())} ({(df.provision == 'dont_know').mean() * 100:.1f}%)")
log(f"   users without provision: {int(((df.ai_use == 1) & (df.provision == 'no')).sum())} of {int((df.ai_use == 1).sum())} ({((df.ai_use == 1) & (df.provision == 'no')).sum() / (df.ai_use == 1).sum() * 100:.1f}%)")
for lvl in ["yes", "no", "dont_know"]:
    s = dd[dd.provision == lvl]
    log(f"   AI use with provision = {lvl}: {s.ai_use.mean() * 100:.1f}% (n = {len(s)})")
log(f"   private AI use: {int(df.private_ai.sum())} of {len(df)} ({df.private_ai.mean() * 100:.1f}%); professional use among private users {dd[dd.private_ai == 1].ai_use.mean() * 100:.1f}% (n = {int((dd.private_ai == 1).sum())}), "
    f"among others {dd[dd.private_ai == 0].ai_use.mean() * 100:.1f}% (n = {int((dd.private_ai == 0).sum())})")
ct = pd.crosstab(d0.private_ai, d0.ai_use)
phi = (ct.loc[0, 0] * ct.loc[1, 1] - ct.loc[0, 1] * ct.loc[1, 0]) / np.sqrt(ct.sum(axis=1).prod() * ct.sum(axis=0).prod())
log(f"   private x professional use (n = {n0}): phi = {phi:.2f}; {int(ct.loc[0, 1])} professional users without private use ({ct.loc[0, 1] / ct[1].sum() * 100:.1f}% of users); "
    f"{int(ct.loc[1, 0])} private users without professional use ({ct.loc[1, 0] / ct.loc[1].sum() * 100:.1f}% of private users)")
u = df[df.ai_use == 1]
r1, r2, r3 = u[u.provision == "yes"], u[(u.provision != "yes") & (u.private_ai == 1)], u[(u.provision != "yes") & (u.private_ai == 0)]
log(f"   training among users by route: provision {r1.trained.mean() * 100:.1f}% (n = {len(r1)}); no provision, private use {r2.trained.mean() * 100:.1f}% (n = {len(r2)}); "
    f"no provision, no private use {r3.trained.mean() * 100:.1f}% (n = {len(r3)})")

# ============================================================================= 2  Note 3, Table A: robustness of the main model
log("\n=== 2  Note 3, Table A: robustness of the main model ===")
A = []


def rowA(name, res, extra_terms=(), note=""):
    r = dict(analysis=name, n=int(res.nobs), parameters=len(res.params) - 1, provision_yes=fmt(res, PY) or fmt(res, "prov_yes"),
             private_ai=fmt(res, "private_ai"), competence=fmt(res, "competence_prof"), trained=fmt(res, "trained"),
             extra=" | ".join(f"{PRETTY.get(t, t)}: {fmt(res, t)}" for t in extra_terms), note=note)
    A.append(r)
    log(f"  {name} (n = {r['n']}, {r['parameters']} par.): provision {r['provision_yes']}; private {r['private_ai']}; competence {r['competence']}" + (f"; {r['extra']}" if r["extra"] else ""))
    return res


rowA("A1 Main model (Table 2)", m0)
rowA("A2 Without private AI use", fit(d0, " + ".join(PERSON[:-1] + ACCESS)))
rowA("A3 Employer provision binary (don't know -> no)", fit(d0, MAIN.replace(PROV3, "prov_yes")))
rowA("A4 Excluding solo practices", fit(prep(subset=(df.solo_practice == 0)), MAIN))
dS = prep()
mS = rowA("A5 Sector in three categories (hospital / MVZ / practice)", fit(dS, MAIN.replace("ambulatory", SEC3)), [f"{SEC3}[T.MVZ]", f"{SEC3}[T.practice]"])
mA5 = rowA("A6 Age in five categories (20-29 separate)", fit(d0, MAIN.replace(AGE4, AGE5)), [f"{AGE5}[T.20-29]", f"{AGE5}[T.40-49]"])
rowA("A7 Private engagement with AI in three categories instead of private use", fit(d0, MAIN.replace("private_ai", ENG3)), [f"{ENG3}[T.informs only]", f"{ENG3}[T.private use]"])
dE = prep(subset=df.employee)
mE = rowA("A8 Employed physicians only (hospital inpatient and outpatient, MVZ)", fit(dE, MAIN.replace("ambulatory", "mvz")), ["mvz"])
dP = prep(subset=df.practice_based)
mP = rowA("A9 Practice-based physicians only (solo and group practices)", fit(dP, MAIN.replace(" + ambulatory", "")))
dH = prep(subset=df.setting.isin(HOSPITAL))
mH = rowA("A10 Hospital physicians only", fit(dH, MAIN.replace(" + ambulatory", "")))
d16 = prep(["na16", "un16", "fr16"])
m16 = rowA("A11 Main model plus the three application counts (16 items)", fit(d16, MAIN + " + na16 + un16 + fr16"), ["na16", "un16", "fr16"])
lrA11 = lr(d16, MAIN + " + na16 + un16 + fr16", ["na16", "un16", "fr16"])
log(f"      LR test for the three counts: chi2({lrA11[1]}) = {lrA11[0]:.1f}, p = {lrA11[2]:.3f}")
d16s = prep(["score16"])
m16s = rowA("A12 Main model plus the 16-item composite score (per SD)", fit(d16s, MAIN + " + z_score16"), ["z_score16"])
d14 = prep(["na14", "un14", "fr14"])
m14 = rowA("A13 Main model plus the counts on 14 items without AI-based applications", fit(d14, MAIN + " + na14 + un14 + fr14"), ["na14", "un14", "fr14"])
lrA13 = lr(d14, MAIN + " + na14 + un14 + fr14", ["na14", "un14", "fr14"])
log(f"      LR test for the three counts (14 items): chi2({lrA13[1]}) = {lrA13[0]:.1f}, p = {lrA13[2]:.3f}")
pd.DataFrame(A).to_csv(os.path.join(OUTDIR, "note3_tableA.csv"), index=False)
# interaction provision x practice-based in the main model
dI = prep()
dI["practice"] = (dI.setting3 == "practice").astype(int)
dI["yes_x_practice"] = ((dI.provision == "yes") & (dI.practice == 1)).astype(int)
dI["dk_x_practice"] = ((dI.provision == "dont_know") & (dI.practice == 1)).astype(int)
baseI = MAIN.replace("ambulatory", SEC3)
mI0, mI1 = fit(dI, baseI), fit(dI, baseI + " + yes_x_practice + dk_x_practice")
chiI = 2 * (mI1.llf - mI0.llf)
log(f"  Interaction provision x practice-based (main model, n = {int(mI1.nobs)}): LR chi2(2) = {chiI:.2f}, p = {stats.chi2.sf(chiI, 2):.3f}; "
    f"provision OR {np.exp(mI1.params[PY]):.2f} in hospitals/MVZ vs {np.exp(mI1.params[PY] + mI1.params['yes_x_practice']):.2f} in practices")

# ============================================================================= 3  Note 3, Table B: the 19-item index examined
log("\n=== 3  Note 3, Table B: the 19-item index examined (covariate set without private AI use) ===")
SUB = " + ".join(PERSON[:-1] + ACCESS)
B = []


def rowB(name, res, terms, note=""):
    r = dict(analysis=name, n=int(res.nobs), provision_yes=fmt(res, PY), competence=fmt(res, "competence_prof"),
             index_terms=" | ".join(f"{PRETTY.get(t, t)}: {fmt(res, t)}" for t in terms), note=note)
    B.append(r)
    log(f"  {name} (n = {r['n']}): {r['index_terms']}; provision {r['provision_yes']}")
    return res


dB = prep(["score19"])
mB1 = rowB("B1 Index on 19 items, per SD", fit(dB, SUB + " + z_score19"), ["z_score19"])
dB2 = prep(["na16", "un16", "fr16"])
mB2 = rowB("B2 Index replaced by its components (16 items): not available, not recognised, use frequency among the rest", fit(dB2, SUB + " + na16 + un16 + fr16"), ["na16", "un16", "fr16"])
dB3 = prep(["unmiss19"])
mB3 = rowB("B3 Index with 'unclear what this is' treated as missing (19 items; denominator = answered, recognised items)", fit(dB3, SUB + " + z_unmiss19"), ["z_unmiss19"])
dB3s = prep(["unmiss19s"])
mB3s = rowB("B3b As B3, with at least 15 recognised items required", fit(dB3s, SUB + " + z_unmiss19s"), ["z_unmiss19s"])
dB4 = prep(["avail19"])
mB4 = rowB("B4 Availability index: share of answered applications not reported as unavailable (19 items), per SD", fit(dB4, SUB + " + z_avail19"), ["z_avail19"])
dB5 = prep(["score16"])
mB5 = rowB("B5 Index on the 16 items of the latent class analysis, per SD", fit(dB5, SUB + " + z_score16"), ["z_score16"])
dB6 = prep(["score14"])
mB6 = rowB("B6 Index on 14 items without AI-based applications (clinical decision support, speech recognition), per SD", fit(dB6, SUB + " + z_score14"), ["z_score14"])
dB7 = prep(["score12"])
mB7 = rowB("B7 Index on 12 items (additionally without phone assistance, automated duty scheduling), per SD", fit(dB7, SUB + " + z_score12"), ["z_score12"])
dB8 = prep(["na14", "un14", "fr14"])
mB8 = rowB("B8 Components on the 14 items without AI-based applications", fit(dB8, SUB + " + na14 + un14 + fr14"), ["na14", "un14", "fr14"])
dB9 = prep(["digital_index_neutral"])
mB9 = rowB("B9 Sector-neutral index (8 items common to both sectors)", fit(dB9, SUB + " + z_neutral"), ["z_neutral"])
dB10 = prep(["na16", "un16", "fr16"])
mB10 = rowB("B10 Components (16 items) with private AI use added", fit(dB10, MAIN + " + na16 + un16 + fr16"), ["na16", "un16", "fr16"])
pd.DataFrame(B).to_csv(os.path.join(OUTDIR, "note3_tableB.csv"), index=False)
# item level: weekly-daily use of each item in the covariate set without private AI use
log("\n  Item level (covariate set without private AI use + weekly-daily use of the item, its 'unclear' and 'not available' indicators):")
dL = prep()
L = []
for it in SET16:
    dL["wd"] = raw19.loc[dL.index, it].isin(["täglich", "wöchentlich"]).astype(int)
    dL["un"] = (raw19.loc[dL.index, it] == UN).astype(int)
    dL["na"] = (raw19.loc[dL.index, it] == NA).astype(int)
    m = fit(dL, SUB + " + wd + un + na")
    L.append(dict(item=it, n_weekly_daily=int(dL.wd.sum()), OR_weekly_daily=fmt(m, "wd"), n_unclear=int(dL.un.sum()), OR_unclear=fmt(m, "un"), n_not_available=int(dL.na.sum()), OR_not_available=fmt(m, "na")))
    log(f"   {it:34s} weekly-daily (n = {int(dL.wd.sum()):3d}) {fmt(m, 'wd')}")
pd.DataFrame(L).to_csv(os.path.join(OUTDIR, "note3_item_level.csv"), index=False)

# ============================================================================= 4  Comment 3: provision x setting descriptives
log("\n=== 4  Provision x setting (comment 3) ===")
ct = pd.crosstab(df.setting3, df.provision).reindex(columns=["yes", "no", "dont_know"])
ct["n"] = ct.sum(axis=1)
for c in ["yes", "no", "dont_know"]:
    ct[c + "_pct"] = (ct[c] / ct.n * 100).round(1)
log(ct.to_string())
save(ct, "provision_by_setting3.csv")
t = dd.groupby(["setting3", "provision"]).ai_use.agg(["size", "mean"])
t["mean"] = (t["mean"] * 100).round(1)
log("AI use by provision within setting:\n" + t.to_string())
save(t, "ai_use_by_provision_x_setting3.csv")
tu = u.groupby("setting3").provision.value_counts().unstack().fillna(0).astype(int)
tu["users"] = tu.sum(axis=1)
tu["withprovision_form_pct"] = (tu["no"] / tu.users * 100).round(1)
log("Users by provision within setting:\n" + tu.to_string())

# ============================================================================= 5  checksums
log("\n=== 5  Checksums ===")
checks = {
    "Table 2 n": (n0, 328, 0), "Table 2 users": (users0, 185, 0), "Table 2 parameters": (npar, 10, 0),
    "provision OR": (float(np.exp(m0.params[PY])), 14.39, 0.01), "private AI use OR": (float(np.exp(m0.params["private_ai"])), 4.11, 0.01),
    "competence OR": (float(np.exp(m0.params["competence_prof"])), 1.34, 0.005), "training OR": (float(np.exp(m0.params["trained"])), 0.71, 0.005),
    "sector OR": (float(np.exp(m0.params["ambulatory"])), 1.09, 0.005), "pseudo-R2": (float(m0.prsquared), 0.264, 0.001),
    "person block chi2": (bl["person block (age, sex, competence, training, private AI use)"][0], 42.2, 0.1),
    "access block chi2": (bl["access block (sector, provision)"][0], 64.0, 0.1),
    "A2 provision OR (no private use)": (float(np.exp(fit(d0, " + ".join(PERSON[:-1] + ACCESS)).params[PY])), 13.47, 0.01),
    "A8 employees n": (int(mE.nobs), 176, 0), "A8 employees provision OR": (float(np.exp(mE.params[PY])), 16.93, 0.01),
    "A9 practice n": (int(mP.nobs), 152, 0), "A9 practice provision OR": (float(np.exp(mP.params[PY])), 15.44, 0.01),
    "A10 hospital n": (int(mH.nobs), 144, 0),
    "interaction provision x practice chi2": (chiI, 1.17, 0.01),
    "A11 not recognised OR": (float(np.exp(m16.params["un16"])), 0.87, 0.005),
    "B1 provision OR": (float(np.exp(mB1.params[PY])), 11.38, 0.005), "B1 19-item index OR": (float(np.exp(mB1.params["z_score19"])), 1.475, 0.001),
    "B1 n": (int(mB1.nobs), 322, 0),
    "B2 not available OR": (float(np.exp(mB2.params["na16"])), 0.965, 0.001), "B2 not recognised OR": (float(np.exp(mB2.params["un16"])), 0.843, 0.001),
    "B2 use frequency OR": (float(np.exp(mB2.params["fr16"])), 1.269, 0.001), "B2 provision OR": (float(np.exp(mB2.params[PY])), 13.03, 0.005),
    "B3 index OR": (float(np.exp(mB3.params["z_unmiss19"])), 1.18, 0.005), "B3b n": (int(mB3s.nobs), 261, 0), "B3b index OR": (float(np.exp(mB3s.params["z_unmiss19s"])), 1.25, 0.005), "B4 availability OR": (float(np.exp(mB4.params["z_avail19"])), 0.98, 0.005),
    "B5 16-item index OR": (float(np.exp(mB5.params["z_score16"])), 1.39, 0.005), "B6 14-item index OR": (float(np.exp(mB6.params["z_score14"])), 1.26, 0.005),
    "B7 12-item index OR": (float(np.exp(mB7.params["z_score12"])), 1.15, 0.005), "B9 sector-neutral OR": (float(np.exp(mB9.params["z_neutral"])), 1.45, 0.005),
}
bad = []
for k, (obs, exp, tol) in checks.items():
    good = abs(obs - exp) <= tol
    log(f"{'OK  ' if good else 'FAIL'} {k}: {obs:.4f} (expected {exp})" if isinstance(obs, float) else f"{'OK  ' if good else 'FAIL'} {k}: {obs} (expected {exp})")
    if not good:
        bad.append(k)
assert not bad, f"CHECKSUMS VIOLATED: {bad}"
log("\nAll checksums OK.")
LOG.close()
