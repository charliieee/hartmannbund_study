"""
unsure_group.py  --  analysis script for the npj Digital Medicine article: description of the 15 respondents who answered
'I don't know / am unsure' to the item on professional AI use , reported in one sentence of the
RQ1 section and in Supplementary Note 10.  Reference values are the whole sample (n = 362) and the 16 application items
retained in the latent class analysis (unrecognised = 'unklar, was das ist').

Usage: python unsure_group.py <analysis_dataset.csv> <outdir>
"""
import os
import sys

import pandas as pd

INFILE, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
LOG = open(os.path.join(OUTDIR, "unsure_group_log.txt"), "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


df = pd.read_csv(INFILE)
COLS = df.columns
unsure = df[df[COLS[42]] == "Ich weiß es nicht/Bin mir unsicher"]
no_answer = df[df[COLS[42]].isna() | (df[COLS[42]] == "Keine Antwort")]
log(f"unsure n = {len(unsure)}; no answer n = {len(no_answer)}; undefined status in total {int(df.ai_use.isna().sum())}")
assert len(unsure) == 15 and int(df.ai_use.isna().sum()) == 21

ITEMS16 = [COLS[18 + i] for i in range(19) if (i + 1) not in (12, 15, 16)]
assert len(ITEMS16) == 16
unrec = df[ITEMS16].apply(lambda col: col.fillna("").str.contains("unklar", regex=False)).sum(axis=1)
q21 = df[COLS[44]].fillna("")
q27 = df[COLS[50]].fillna("")
q29 = df[COLS[52]].fillna("")
rows = []


def add(measure, u, s):
    rows.append({"measure": measure, "unsure (n = 15)": u, "all respondents (n = 362)": s})
    log(f"{measure:60s} unsure: {u}   sample: {s}")


add("self-rated professional digital competence, mean (1-5)", round(unsure.competence_prof.mean(), 2), round(df.competence_prof.mean(), 2))
add("surveyed applications not recognised, mean of 16", round(unrec[unsure.index].mean(), 2), round(unrec.mean(), 2))
add("three or more applications not recognised, n", int((unrec[unsure.index] >= 3).sum()), int((unrec >= 3).sum()))
add("employer provision: 'don't know', n", int((unsure.provision == "dont_know").sum()), int((df.provision == "dont_know").sum()))
add("employer provision: yes, n", int((unsure.provision == "yes").sum()), int((df.provision == "yes").sum()))
add("private AI use, n", int(q21[unsure.index].str.contains("Ich nutze KI-Anwendungen privat", regex=False).sum()), int(q21.str.contains("Ich nutze KI-Anwendungen privat", regex=False).sum()))
add("prior AI-specific training, n", int(unsure.trained.sum()), int(df.trained.sum()))
add("introductory level named, n", int(q27[unsure.index].str.contains("Einführend", regex=False).sum()), int(q27.str.contains("Einführend", regex=False).sum()))
add("chambers of physicians named as provider, n", int(q29[unsure.index].str.contains("Ärztekammern", regex=False).sum()), int(q29.str.contains("Ärztekammern", regex=False).sum()))
add("commercial AI vendors named as provider, n", int(q29[unsure.index].str.contains("Industrieanbieter", regex=False).sum()), int(q29.str.contains("Industrieanbieter", regex=False).sum()))
add("hospital-based, n", int(unsure[COLS[11]].fillna("").str.startswith("Krankenhaus").sum()), int(df[COLS[11]].fillna("").str.startswith("Krankenhaus").sum()))
add("age 50 years or older, n", int(unsure[COLS[8]].isin(["50-59 Jahre", "60-69 Jahre", "70 Jahre oder älter"]).sum()), int(df[COLS[8]].isin(["50-59 Jahre", "60-69 Jahre", "70 Jahre oder älter"]).sum()))
pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "unsure_group.csv"), index=False)

log("\nFree-text areas of use given by unsure respondents (Q20):")
for idx, t in unsure[COLS[43]].fillna("").items():
    if t.strip():
        log(f"  [{idx}] {t}")

log("\n=== Checksums ===")
checks = {"competence unsure": (round(unsure.competence_prof.mean(), 1), 2.5), "unrecognised unsure": (round(unrec[unsure.index].mean(), 1), 3.1),
          "unrecognised sample": (round(unrec.mean(), 1), 1.6), "provision don't know": (int((unsure.provision == "dont_know").sum()), 9),
          "private use": (int(q21[unsure.index].str.contains("Ich nutze KI-Anwendungen privat", regex=False).sum()), 7),
          "introductory": (int(q27[unsure.index].str.contains("Einführend", regex=False).sum()), 7),
          "chambers": (int(q29[unsure.index].str.contains("Ärztekammern", regex=False).sum()), 13),
          "vendors": (int(q29[unsure.index].str.contains("Industrieanbieter", regex=False).sum()), 0)}
bad = [k for k, (o, e) in checks.items() if o != e]
for k, (o, e) in checks.items():
    log(f"{'OK  ' if o == e else 'FAIL'} {k}: {o} (expected {e})")
assert not bad, f"CHECKSUMS VIOLATED: {bad}"
log("\nAll checksums OK.")
LOG.close()
