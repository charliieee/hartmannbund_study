"""
provision_form.py  --  analysis script for the npj Digital Medicine article: what the respondents' AI use
consists of, as far as the questionnaire allows.

The questionnaire did not ask which tools were used.  Three items constrain the interpretation:
  Q15  form of employer-provided AI (multiple response; asked where the employer provides AI)
  Q20  areas of use (multiple response; users)
  Q21  private AI use ('I use AI applications privately (e.g. ChatGPT, AI image generators)')

1  Q15 among the 105 respondents with employer provision; among the 92 users with provision.
2  Q20 areas by form of provision (standalone tools vs integrated into medical software) among users with provision.
3  Users without employer provision (n = 84): private AI use, areas of use, self-reported unauthorised use.
4  Areas of use by private AI use among all users.
5  Free-text answers in Q20 and Q15 that name a tool.

Usage: python provision_form.py <analysis_dataset.csv> <outdir>
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

INFILE, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
LOG = open(os.path.join(OUTDIR, "provision_form_log.txt"), "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


df = pd.read_csv(INFILE)
COLS = df.columns
q15 = df[COLS[38]].fillna("")
q20 = df[COLS[43]].fillna("")
df["private_ai"] = df[COLS[44]].fillna("").astype(str).str.contains("Ich nutze KI-Anwendungen privat", regex=False).astype(int)
q32 = df[COLS[55]]
df["shadow"] = np.where(q32.str.startswith("Ja, habe selbst", na=False), 1.0, np.where(q32.isin(["Ja, habe andere schon mal KI ohne formelle Genehmigung nutzen sehen.", "Nein, mir ist kein Fall bekannt"]), 0.0, np.nan))

FORMS = [("Standalone tools or programs (e.g. AI models or chatbots, speech recognition)", "Als eigenständige Tools"),
         ("Integrated into existing medical software (e.g. diagnostic systems, decision support)", "In bestehender medizinischer Software"),
         ("In trial / pilot phase (external software)", "In Erprobung"),
         ("In-house developed AI models", "Inhouse"),
         ("No AI tools available", "Keine KI-Tools"), ("Don't know", "Weiß nicht")]
for lab, key in FORMS:
    df["f_" + key] = q15.str.contains(key, regex=False).astype(int)
PREDEF15 = [k for _, k in FORMS]
df["f_other"] = q15.apply(lambda c: int(any(t.strip() and not any(k in t for k in PREDEF15) for t in str(c).split(";"))))
AREAS = [("Administration", "Administrative Aufgaben"), ("Diagnostics", "Diagnostik und Befund"), ("Treatment planning", "Behandlungsplanung"),
         ("Research", "Forschung"), ("Patient communication", "Patientenkommunikation")]
# free-text areas recoded as in results_R2.py / Supplementary Note 4
FREE20 = {"Recherche": ["Research"], "Recherche, Patienteninformation": ["Research", "Patient communication"], "RECHERCHE": ["Research"],
          "Übersetzungen für fremdsprachliche Patienten": ["Patient communication"], "Spracherkennung und Literaturrecherche": ["Research", "Administration"],
          "Behandlungsdokumentation": ["Administration"], "OpenEvidence": ["Research"], "Diktiergerät": ["Administration"], "Marketing": ["Administration"],
          "Dokumentation / spracherkennung": ["Administration"], "Unterstützung bei seltenen Diagnosen zur Unterstützung bei Diagnose- und Therapiefindung": ["Diagnostics"],
          "Bei der Strahlentherapie selbst": ["Diagnostics"], "zusätzliche Auswertung der Lunge": ["Diagnostics"]}


def areas(cell):
    out = set()
    for t in str(cell).split(";"):
        t = t.strip()
        for lab, key in AREAS:
            if key in t:
                out.add(lab)
        out.update(FREE20.get(t, []))
    return out


df["areas"] = q20.apply(areas)
for lab, _ in AREAS:
    df["a_" + lab] = df.areas.apply(lambda s: int(lab in s))
df["n_areas"] = df.areas.apply(len)

# ============================================================================= 1  Q15
log("=== 1  Form of employer-provided AI (Q15) ===")
P = df[df.provision == "yes"]
UP = df[(df.provision == "yes") & (df.ai_use == 1)]
log(f"respondents with employer provision: {len(P)}; of whom users {len(UP)}, non-users {int((P.ai_use == 0).sum())}, undefined {int(P.ai_use.isna().sum())}")
rows = []
for lab, key in FORMS + [("Other (free text)", "other")]:
    rows.append({"form": lab, "n_provision": int(P["f_" + key].sum()), "pct_provision": 100 * P["f_" + key].mean(), "n_users_with_provision": int(UP["f_" + key].sum()), "pct_users_with_provision": 100 * UP["f_" + key].mean()})
t1 = pd.DataFrame(rows)
pd.set_option("display.width", 250)
log(t1.round(1).to_string(index=False))
t1.to_csv(os.path.join(OUTDIR, "provision_forms.csv"), index=False, float_format="%.2f")
log(f"Q15 entries per respondent among provision = yes: {q15[P.index].apply(lambda c: len([t for t in str(c).split(';') if t.strip()])).value_counts().sort_index().to_dict()} (the item allowed multiple answers; the export shows at most one)")
log(f"Q15 answered (non-empty) among provision = yes: {int((q15[P.index] != '').sum())}; among provision = no: {int((q15[df.provision == 'no'] != '').sum())}; among don't know: {int((q15[df.provision == 'dont_know'] != '').sum())}")
both = int(((UP["f_Als eigenständige Tools"] == 1) & (UP["f_In bestehender medizinischer Software"] == 1)).sum())
log(f"users with provision: standalone only {int(((UP['f_Als eigenständige Tools'] == 1) & (UP['f_In bestehender medizinischer Software'] == 0)).sum())}, integrated only "
    f"{int(((UP['f_Als eigenständige Tools'] == 0) & (UP['f_In bestehender medizinischer Software'] == 1)).sum())}, both {both}, neither {int(((UP['f_Als eigenständige Tools'] == 0) & (UP['f_In bestehender medizinischer Software'] == 0)).sum())}")
log("Free-text forms (all respondents):", [t.strip() for c in q15 for t in str(c).split(";") if t.strip() and not any(k in t for k in PREDEF15)])

# ============================================================================= 2  Q20 by form among users with provision
log("\n=== 2  Areas of use by form of provision (users with provision, n = %d) ===" % len(UP))
UP = UP.copy()
UP["form3"] = np.select([(UP["f_Als eigenständige Tools"] == 1) & (UP["f_In bestehender medizinischer Software"] == 0),
                         (UP["f_Als eigenständige Tools"] == 0) & (UP["f_In bestehender medizinischer Software"] == 1),
                         (UP["f_Als eigenständige Tools"] == 1) & (UP["f_In bestehender medizinischer Software"] == 1)],
                        ["standalone only", "integrated only", "both"], default="other/pilot/in-house only")
rows = []
for form, g in UP.groupby("form3"):
    r = {"form": form, "n": len(g), "mean_areas": g.n_areas.mean(), "private_ai_pct": 100 * g.private_ai.mean()}
    for lab, _ in AREAS:
        r[lab + " %"] = 100 * g["a_" + lab].mean()
    rows.append(r)
t2 = pd.DataFrame(rows)
log(t2.round(1).to_string(index=False))
t2.to_csv(os.path.join(OUTDIR, "areas_by_provision_form.csv"), index=False, float_format="%.2f")
sa, it = UP[UP["f_Als eigenständige Tools"] == 1], UP[UP["f_Als eigenständige Tools"] == 0]
for lab, _ in AREAS:
    k1, n1, k0, n0 = int(sa["a_" + lab].sum()), len(sa), int(it["a_" + lab].sum()), len(it)
    _, p = stats.fisher_exact([[k1, n1 - k1], [k0, n0 - k0]])
    log(f"   {lab:22s} standalone named (n = {n1}): {100 * k1 / n1:.1f}%  vs not named (n = {n0}): {100 * k0 / n0:.1f}%  Fisher p = {p:.3f}")

# ============================================================================= 3  users without provision
log("\n=== 3  Users without employer provision ===")
U = df[df.ai_use == 1].copy()
for lvl in ["yes", "no", "dont_know"]:
    g = U[U.provision == lvl]
    log(f"provision = {lvl:9s}: n = {len(g):3d}; private AI use {100 * g.private_ai.mean():.1f}%; unauthorised use (valid Q32) {100 * g.shadow.mean():.1f}% (n = {int(g.shadow.notna().sum())}); "
        f"mean areas {g.n_areas.mean():.2f}; " + "; ".join(f"{lab} {100 * g['a_' + lab].mean():.1f}%" for lab, _ in AREAS))
UN = U[U.provision == "no"]
log(f"users without provision: {len(UN)} of {len(U)} ({100 * len(UN) / len(U):.1f}%); with private AI use {int(UN.private_ai.sum())} ({100 * UN.private_ai.mean():.1f}%); "
    f"neither provision nor private use: {int((UN.private_ai == 0).sum())}")
log(f"Q15 'no AI tools available' or 'don't know' ticked by users without provision (after the correction rule): {int((UN['f_Keine KI-Tools'] + UN['f_Weiß nicht']).sum())}")
# ============================================================================= 4  areas by private AI use among all users
log("\n=== 4  Areas of use by private AI use (all users, n = %d) ===" % len(U))
for lab, _ in AREAS:
    a, b = U[U.private_ai == 1], U[U.private_ai == 0]
    _, p = stats.fisher_exact([[int(a["a_" + lab].sum()), len(a) - int(a["a_" + lab].sum())], [int(b["a_" + lab].sum()), len(b) - int(b["a_" + lab].sum())]])
    log(f"   {lab:22s} private users (n = {len(a)}): {100 * a['a_' + lab].mean():.1f}%  others (n = {len(b)}): {100 * b['a_' + lab].mean():.1f}%  p = {p:.3f}")
log(f"users naming administration only (no other area): {int(((U.a_Administration == 1) & (U.n_areas == 1)).sum())}; users naming any decision-proximal area (diagnostics or treatment planning): "
    f"{int(((U.a_Diagnostics == 1) | (U['a_Treatment planning'] == 1)).sum())} ({100 * ((U.a_Diagnostics == 1) | (U['a_Treatment planning'] == 1)).mean():.1f}%)")

# ============================================================================= 5  free text naming tools
log("\n=== 5  Free-text entries naming a tool or product (Q20, Q15, Q16) ===")
tools = []
for col in [43, 38, 39]:
    for idx, c in df[COLS[col]].fillna("").items():
        for t in str(c).split(";"):
            t = t.strip()
            predefined = t.startswith(("Patientenkommunikation", "Als eigenständige Tools", "In bestehender", "Administrative", "Diagnostik", "Behandlungsplanung", "Forschung", "keine Antwort"))
            if not predefined and any(w in t.lower() for w in ["chat", "gpt", "openevidence", "amboss", "acrobat", "office", "copilot", "privat", "account"]):
                tools.append((COLS[col][:30], idx, t))
for row in tools:
    log("  ", row)

log("\n=== Checksums ===")
checks = {"provision yes": (len(P), 105), "users with provision": (len(UP), 92), "users without provision": (len(UN), 84), "users without provision, private use": (int(UN.private_ai.sum()), 64),
          "users": (len(U), 192), "administration among users": (int(U.a_Administration.sum()), 100), "diagnostics among users": (int(U.a_Diagnostics.sum()), 81),
          "treatment planning among users": (int(U["a_Treatment planning"].sum()), 65), "research among users (person-based; 'RECHERCHE' mapped to Research as intended by the Note 4 mapping; the repository's results_R2.py gives 59 because its key 'Recherche' does not match the entry 'RECHERCHE'; the original Fig. 1b states 61)": (int(U.a_Research.sum()), 60), "patient communication among users": (int(U["a_Patient communication"].sum()), 41)}
bad = [k for k, (o, e) in checks.items() if o != e]
for k, (o, e) in checks.items():
    log(f"{'OK  ' if o == e else 'FAIL'} {k}: {o} (expected {e})")
assert not bad, f"CHECKSUMS VIOLATED: {bad}"
log("\nAll checksums OK.")
LOG.close()
