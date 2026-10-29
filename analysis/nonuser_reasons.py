"""
nonuser_reasons.py  --  analysis script for the npj Digital Medicine article: what the two groups of
non-users give as reasons.

1  Q16 (closed barrier item, up to three options; answered by all respondents, analysed for definite
   non-users) separately for interested non-users (n = 86) and non-users without current intention (n = 63):
   % with Wilson 95% CIs, risk difference (no intention minus interested) with Newcombe 95% CI, Fisher exact test,
   BH over the six predefined options.
2  Q30 (free text: 'What reasons keep you from pursuing further training on AI in medicine?') for the 63
   respondents without current intention, coded against categories defined before coding (two passes; the
   second pass is documented as a list of changes).  Several categories per answer are allowed.  Q30 asks
   about training, not about use; the categories therefore cover both.
3  Q31 (free text: further comments) for the same 63, sighted and listed.

4  Optional override layer: a CSV of coding changes made by the authors on reading every
   answer (file, id, codes_before, codes_after, author, date, note).  It is applied after the rule-based coding and
   before any count; the checksums below refer to the coding with the override applied.

Usage: python nonuser_reasons.py <analysis_dataset.csv> <outdir> [<q30_author_review.csv>]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

INFILE, OUTDIR = sys.argv[1], sys.argv[2]
OVERRIDE = sys.argv[3] if len(sys.argv) > 3 else None
os.makedirs(OUTDIR, exist_ok=True)
LOG = open(os.path.join(OUTDIR, "nonuser_reasons_log.txt"), "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


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


df = pd.read_csv(INFILE)
COLS = df.columns
q16 = df[COLS[39]].fillna("")
BARRIERS = [
    ("Not available", "Nicht verfügbar"),
    ("Lack of knowledge / training", "Fehlendes Wissen"),
    ("Data protection / security concerns", "Sicherheits-/Datenschutzbedenken"),
    ("Insufficient system integration", "Integrationsmöglichkeiten"),
    ("No perceived benefit", "Keine wahrgenommene Nützlichkeit"),
    ("Lack of time / capacity", "fehlende Kapazitäten"),
]
# free-text mappings to the predefined categories, identical to results_R2.py / Supplementary Note 4
FREE_MAP = {
    "kein bedarf": "No perceived benefit",
    "Als sporadisch in der Notfallpraxis tätiger Arzt bisher nicht erforderlich für mich, wird auch nicht bereitgestellt": "Not available",
    "Die Firmen, die Programme für die Blutbanklogistik entwickeln, haben KI noch nicht in deren vorhandenen Programme implementiert.": "Not available",
}
# ============================================================================= 1  Q16 by non-user group
log("=== 1  Q16 barriers: interested non-users vs no current intention ===")
I, N = df[df.segment == "interested"], df[df.segment == "no_intention"]
log(f"n interested = {len(I)}, n no intention = {len(N)}; all 149 definite non-users answered Q16: {int((q16[df.ai_use == 0] != '').sum())}")


def has(cell, key, lab):
    toks = [t.strip() for t in str(cell).split(";")]
    if any(key in t for t in toks):
        return 1
    return int(any(FREE_MAP.get(t) == lab for t in toks))


rows = []
for lab, key in BARRIERS:
    kI = int(sum(has(c, key, lab) for c in q16[I.index]))
    kN = int(sum(has(c, key, lab) for c in q16[N.index]))
    kAll = kI + kN
    rd, lo, hi = newcombe(kN, len(N), kI, len(I))
    _, pf = stats.fisher_exact([[kN, len(N) - kN], [kI, len(I) - kI]])
    wI, wN = wilson(kI, len(I)), wilson(kN, len(N))
    rows.append({"barrier": lab, "n_interested": kI, "pct_interested": 100 * kI / len(I), "ci_interested": f"{100 * wI[0]:.1f}-{100 * wI[1]:.1f}",
                 "n_no_intention": kN, "pct_no_intention": 100 * kN / len(N), "ci_no_intention": f"{100 * wN[0]:.1f}-{100 * wN[1]:.1f}",
                 "n_all_nonusers": kAll, "pct_all_nonusers": 100 * kAll / 149,
                 "rd_pp": 100 * rd, "rd_ci_low": 100 * lo, "rd_ci_high": 100 * hi, "p_fisher": pf})
res = pd.DataFrame(rows)
res["p_fisher_adj"] = multipletests(res["p_fisher"], method="fdr_bh")[1]
res.to_csv(os.path.join(OUTDIR, "q16_by_nonuser_group.csv"), index=False, float_format="%.4f")
pd.set_option("display.width", 250)
log(res.round(3).to_string(index=False))
nb_I = pd.Series([sum(has(c, key, lab) for lab, key in BARRIERS) for c in q16[I.index]])
nb_N = pd.Series([sum(has(c, key, lab) for lab, key in BARRIERS) for c in q16[N.index]])
log(f"Number of barriers named (0-3): interested mean {nb_I.mean():.2f}, no intention mean {nb_N.mean():.2f}; "
    f"none named: interested {int((nb_I == 0).sum())}, no intention {int((nb_N == 0).sum())}")

# ============================================================================= 2  Q30 free text, no-intention group
log("\n=== 2  Q30 free text, respondents without current intention (n = 63): coding ===")
CATS = {
    "TIME": "Lack of time or capacity",
    "ACCESS": "No access: no offers, no systems or support at work, costs, infrastructure",
    "BENEFIT": "No perceived benefit, need or relevance for own work",
    "RELIABILITY": "Doubts about the maturity, error rate or controllability of AI; unwillingness to rely on its output",
    "REJECTION": "Principled rejection of AI in medicine; scepticism about the 'hype'; preference for established routine",
    "DATA_LEGAL": "Data protection, legal, ethical or environmental concerns",
    "CAREER": "Career stage: approaching retirement, other training priorities",
    "PROVIDERS": "Difficulty judging the seriousness of providers and offers",
    "ORIENTATION": "Lack of basic knowledge or orientation: not knowing where or how to start, which training is adequate, entry hurdle",
    "NONE": "No reason given, not codable, or 'nothing keeps me / I already train'",
}
# first pass (index in analysis_dataset.csv -> codes)
PASS1 = {
    14: ["NONE"], 20: ["BENEFIT", "ACCESS"], 24: ["TIME", "ACCESS"], 26: ["RELIABILITY"], 28: ["TIME", "BENEFIT"], 41: ["TIME"], 48: ["TIME"],
    51: ["ACCESS", "TIME"], 53: ["BENEFIT"], 63: ["TIME"], 69: ["NONE"], 78: ["TIME", "RELIABILITY"], 98: ["BENEFIT"], 100: ["TIME"],
    112: ["ACCESS"], 117: ["ACCESS", "TIME"], 120: ["TIME"], 135: ["TIME", "BENEFIT"], 136: ["BENEFIT"], 143: ["TIME"], 144: ["TIME"],
    145: ["TIME", "BENEFIT"], 159: ["ACCESS"], 164: ["ACCESS"], 167: ["REJECTION"], 174: ["ACCESS"], 176: ["RELIABILITY"],
    180: ["BENEFIT", "REJECTION"], 181: ["BENEFIT", "RELIABILITY"], 190: ["PROVIDERS", "BENEFIT"], 191: ["ACCESS", "DATA_LEGAL"],
    193: ["BENEFIT", "RELIABILITY"], 198: ["TIME"], 206: ["REJECTION"], 210: ["TIME", "ACCESS"], 216: ["ACCESS"], 221: ["NONE"],
    227: ["BENEFIT"], 238: ["DATA_LEGAL", "RELIABILITY"], 249: ["REJECTION", "DATA_LEGAL", "RELIABILITY"], 256: ["ACCESS"],
    267: ["REJECTION", "RELIABILITY"], 273: ["TIME"], 277: ["PROVIDERS"], 283: ["TIME"], 287: ["BENEFIT", "ACCESS"], 288: ["ACCESS"],
    290: ["ACCESS"], 294: ["TIME"], 297: ["TIME"], 309: ["ACCESS"], 311: ["RELIABILITY"], 316: ["NONE"], 320: ["TIME"], 329: ["TIME"],
    337: ["CAREER"], 340: ["NONE"], 345: ["CAREER"], 351: ["REJECTION"], 354: ["NONE"], 355: ["BENEFIT"], 360: ["CAREER"], 361: ["ACCESS", "TIME"],
}
# second pass: re-reading of every answer against the category definitions; changes relative to the first pass
PASS2_CHANGES = {
    145: (["TIME", "BENEFIT", "REJECTION"], "'keine Lust mehr auf Innovation' is a general stance towards innovation, not only a benefit judgement"),
}
PASS2_NOTES = {
    26: "'Skepsis gegenüber KI-vermittelten Inhalten' is ambiguous (AI-generated content in general, or AI-delivered training content); coded as RELIABILITY",
    227: "'Abwehr von Patienten' (patients' resistance) has no category of its own; coded under BENEFIT (relevance for own work)",
    311: "control concern about machine autonomy; coded RELIABILITY, could also be read as REJECTION",
    340: "wording unclear ('systems not so old that AI could be spoken of'); not codable",
}
codes = {k: list(v) for k, v in PASS1.items()}
for k, (new, why) in PASS2_CHANGES.items():
    log(f"second pass, change for [{k}]: {PASS1[k]} -> {new} ({why})")
    codes[k] = new
for k, note in PASS2_NOTES.items():
    log(f"second pass, note for [{k}]: {note}")
assert set(codes) == set(N.index), "coding does not cover exactly the 63 no-intention respondents"


def apply_override(code_dict, fname, group_label):
    """Apply the authors' review of the coding (q30_author_review.csv) to one group's code dictionary."""
    if OVERRIDE is None:
        return 0
    ov = pd.read_csv(OVERRIDE)
    ov = ov[ov["file"] == fname]
    n_changed = 0
    for _, r in ov.iterrows():
        idx = int(r["id"])
        before = [c.strip() for c in str(r["codes_before"]).split(";")]
        after = [c.strip() for c in str(r["codes_after"]).split(";")]
        assert idx in code_dict, f"override id {idx} not in {group_label}"
        assert sorted(code_dict[idx]) == sorted(before), f"override for [{idx}] expects {before}, found {code_dict[idx]}"
        assert all(c in CATS for c in after), f"unknown code in override for [{idx}]: {after}"
        log(f"author review ({r['author']}, {r['date']}), change for [{idx}] ({group_label}): {before} -> {after} ({r['note'] if pd.notna(r['note']) else ''})")
        code_dict[idx] = after
        n_changed += 1
    return n_changed


n_ov_N = apply_override(codes, "q30_no_intention_coded.csv", "no intention")
q30 = df[COLS[53]]
rows = []
for idx in N.index:
    rows.append({"id": idx, "text": q30[idx], "codes": "; ".join(codes[idx])})
pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "q30_no_intention_coded.csv"), index=False)
counts = {c: sum(c in v for v in codes.values()) for c in CATS}
log("\nQ30 categories (n = 63; several per answer possible):")
tab = []
for c, n in sorted(counts.items(), key=lambda x: -x[1]):
    lo, hi = wilson(n, 63)
    tab.append({"category": c, "definition": CATS[c], "n": n, "pct": 100 * n / 63, "ci95": f"{100 * lo:.1f}-{100 * hi:.1f}"})
    log(f"  {c:12s} {n:2d} ({100 * n / 63:4.1f}%, 95% CI {100 * lo:.1f}-{100 * hi:.1f})  {CATS[c]}")
pd.DataFrame(tab).to_csv(os.path.join(OUTDIR, "q30_no_intention_categories.csv"), index=False, float_format="%.1f")
n_codes = pd.Series([len(v) for v in codes.values()])
log(f"codes per answer: mean {n_codes.mean():.2f}; answers with a substantive reason (not NONE): {63 - counts['NONE']}")
rel_or_rej = sum(("RELIABILITY" in v) or ("REJECTION" in v) for v in codes.values())
log(f"answers naming reliability doubts or principled rejection (either): {rel_or_rej} ({100 * rel_or_rej / 63:.1f}%)")
only_time_access = sum(set(v) <= {"TIME", "ACCESS"} and v != ["NONE"] for v in codes.values())
log(f"answers naming only time and/or access: {only_time_access} ({100 * only_time_access / 63:.1f}%)")
log("'trust' / 'Vertrauen' used literally in any Q30 answer of the group: " + str(int(q30[N.index].fillna("").str.lower().str.contains("vertrau").sum())))

# ----------------------------------------------------------------------------- Q30 for the 86 interested non-users (same scheme; second coding)
PASS1_I = {
    3: ["NONE"], 4: ["TIME"], 8: ["ACCESS"], 11: ["ACCESS"], 19: ["BENEFIT"], 29: ["NONE"], 37: ["ACCESS", "TIME"], 46: ["ACCESS"], 50: ["ACCESS"],
    58: ["ACCESS"], 59: ["TIME"], 67: ["TIME", "CAREER", "ORIENTATION"], 72: ["TIME", "ACCESS"], 73: ["TIME"], 80: ["TIME"], 81: ["NONE"], 84: ["TIME"],
    85: ["ACCESS"], 97: ["NONE"], 101: ["TIME"], 106: ["ACCESS"], 107: ["ACCESS"], 111: ["NONE"], 115: ["ACCESS"], 116: ["ACCESS"],
    122: ["ORIENTATION", "BENEFIT"], 125: ["TIME"], 127: ["ORIENTATION", "PROVIDERS"], 129: ["ORIENTATION"], 131: ["ORIENTATION"],
    132: ["TIME", "DATA_LEGAL"], 134: ["TIME"], 140: ["TIME"], 148: ["TIME"], 150: ["ACCESS"], 162: ["TIME"], 165: ["ACCESS"], 168: ["ACCESS"],
    173: ["TIME", "ACCESS"], 175: ["ORIENTATION"], 183: ["TIME", "CAREER"], 187: ["TIME"], 188: ["ACCESS"], 189: ["TIME"], 192: ["NONE"], 194: ["TIME"],
    196: ["TIME"], 204: ["ACCESS", "RELIABILITY"], 207: ["ACCESS"], 212: ["TIME"], 213: ["NONE"], 228: ["ACCESS"], 230: ["TIME"], 231: ["NONE"],
    232: ["ORIENTATION"], 241: ["NONE"], 246: ["TIME"], 248: ["ACCESS"], 250: ["ORIENTATION"], 251: ["TIME", "ACCESS"], 253: ["ACCESS"], 261: ["TIME"],
    265: ["TIME", "ORIENTATION"], 269: ["NONE"], 275: ["CAREER", "BENEFIT"], 276: ["TIME"], 279: ["TIME"], 280: ["TIME"], 282: ["TIME", "ORIENTATION"],
    291: ["ORIENTATION"], 292: ["TIME", "ACCESS"], 299: ["ACCESS"], 303: ["TIME"], 305: ["TIME", "ACCESS"], 306: ["NONE"], 321: ["ACCESS", "DATA_LEGAL"],
    323: ["ACCESS"], 328: ["TIME"], 331: ["ACCESS"], 333: ["REJECTION"], 335: ["ACCESS"], 343: ["ORIENTATION"], 344: ["NONE"], 347: ["ACCESS"],
    356: ["TIME", "RELIABILITY"], 358: ["NONE"],
}
NOTES_I = {19: "'Patientenaufklaerung zu aufwendig' (effort of informing patients): coded BENEFIT (effort vs benefit)", 132: "'geringe Akzeptanz in der Allgemeinheit' has no category; TIME and DATA_LEGAL coded",
           165: "waits for the tool that will be used: ACCESS", 232: "'Angst vor dem Unbekannten': ORIENTATION (entry hurdle), could be read as RELIABILITY",
           269: "'Arbeitserleichterung' and 306 'Zeitersparnis' name a benefit, not a reason against: NONE", 333: "'Angst vor Verlust der eigenen Meinungsbildung': REJECTION (principled)",
           356: "'kein Vertrauen auf Sicherheit' is the only literal use of 'trust' among the 149 non-users: RELIABILITY"}
assert set(PASS1_I) == set(I.index), "coding does not cover exactly the 86 interested non-users"
for k, note in NOTES_I.items():
    log(f"interested, note for [{k}]: {note}")
n_ov_I = apply_override(PASS1_I, "q30_interested_coded.csv", "interested")
log(f"author-review override applied: {n_ov_N} changes (no intention), {n_ov_I} changes (interested); file: {OVERRIDE}")
counts_I = {c: sum(c in v for v in PASS1_I.values()) for c in CATS}
log("\nQ30 categories, interested non-users (n = 86) vs no intention (n = 63):")
rows_c = []
for c in CATS:
    kI, kN = counts_I[c], counts[c]
    rd_, lo_, hi_ = newcombe(kN, 63, kI, 86)
    _, pf_ = stats.fisher_exact([[kN, 63 - kN], [kI, 86 - kI]])
    rows_c.append({"category": c, "n_interested": kI, "pct_interested": 100 * kI / 86, "n_no_intention": kN, "pct_no_intention": 100 * kN / 63, "rd_pp": 100 * rd_, "rd_ci_low": 100 * lo_, "rd_ci_high": 100 * hi_, "p_fisher": pf_})
    log(f"  {c:12s} interested {kI:2d} ({100 * kI / 86:4.1f}%)  no intention {kN:2d} ({100 * kN / 63:4.1f}%)  difference {100 * rd_:+.1f} pp ({100 * lo_:.1f} to {100 * hi_:.1f}), Fisher p = {pf_:.3f}")
pd.DataFrame(rows_c).to_csv(os.path.join(OUTDIR, "q30_categories_by_nonuser_group.csv"), index=False, float_format="%.4f")
pd.DataFrame([{"id": idx, "text": q30[idx], "codes": "; ".join(PASS1_I[idx])} for idx in I.index]).to_csv(os.path.join(OUTDIR, "q30_interested_coded.csv"), index=False)
rr_I = sum(("RELIABILITY" in v) or ("REJECTION" in v) for v in PASS1_I.values())
log(f"reliability doubts or rejection: interested {rr_I} ({100 * rr_I / 86:.1f}%) vs no intention {rel_or_rej} ({100 * rel_or_rej / 63:.1f}%)")
EVAL = ("BENEFIT", "RELIABILITY", "REJECTION")
ev_I = sum(any(c in v for c in EVAL) for v in PASS1_I.values())
ev_N = sum(any(c in v for c in EVAL) for v in codes.values())
rd_e, lo_e, hi_e = newcombe(ev_N, 63, ev_I, 86)
_, p_e = stats.fisher_exact([[ev_N, 63 - ev_N], [ev_I, 86 - ev_I]])
log(f"any evaluative reason (BENEFIT, RELIABILITY or REJECTION): interested {ev_I} ({100 * ev_I / 86:.1f}%) vs no intention {ev_N} ({100 * ev_N / 63:.1f}%); "
    f"difference {100 * rd_e:+.1f} pp ({100 * lo_e:.1f} to {100 * hi_e:.1f}), Fisher p = {p_e:.4f}")
pd.DataFrame([{"measure": "any evaluative reason (BENEFIT|RELIABILITY|REJECTION)", "n_interested": ev_I, "pct_interested": 100 * ev_I / 86,
               "n_no_intention": ev_N, "pct_no_intention": 100 * ev_N / 63, "rd_pp": 100 * rd_e, "rd_ci_low": 100 * lo_e, "rd_ci_high": 100 * hi_e, "p_fisher": p_e},
              {"measure": "reliability doubts or rejection (RELIABILITY|REJECTION)", "n_interested": rr_I, "pct_interested": 100 * rr_I / 86,
               "n_no_intention": rel_or_rej, "pct_no_intention": 100 * rel_or_rej / 63, "rd_pp": np.nan, "rd_ci_low": np.nan, "rd_ci_high": np.nan, "p_fisher": np.nan}
              ]).to_csv(os.path.join(OUTDIR, "q30_composites_by_nonuser_group.csv"), index=False, float_format="%.4f")

# ============================================================================= 3  Q31
log("\n=== 3  Q31 further comments, no-intention group (non-empty, not a placeholder) ===")
q31 = df[COLS[54]]
placeholders = {"-", "nein", "keine", ".", "/", "--", "x", "n", "?", "nein.", "keine.", "nichts"}
n31 = 0
for idx in N.index:
    t = str(q31[idx]).strip()
    if t and t.lower() not in placeholders and t != "nan":
        n31 += 1
        log(f"[{idx}] {t}")
log(f"substantive Q31 comments in the group: {n31} of 63")

# ============================================================================= checksums
log("\n=== Checksums ===")
if OVERRIDE is None:   # rule-based coding
    EXP = {"TIME": 23, "ACCESS": 17, "BENEFIT": 14, "RELIABILITY": 9, "REJECTION": 7, "NONE": 6, "I_BENEFIT": 3, "I_RELIABILITY": 2}
else:                  # with the authors' review applied
    EXP = {"TIME": 23, "ACCESS": 18, "BENEFIT": 16, "RELIABILITY": 9, "REJECTION": 8, "NONE": 5, "I_BENEFIT": 4, "I_RELIABILITY": 2}
checks = {"n interested": (len(I), 86), "n no intention": (len(N), 63), "Q30 coded": (len(codes), 63),
          "TIME count": (counts["TIME"], EXP["TIME"]), "ACCESS count": (counts["ACCESS"], EXP["ACCESS"]), "BENEFIT count": (counts["BENEFIT"], EXP["BENEFIT"]),
          "RELIABILITY count": (counts["RELIABILITY"], EXP["RELIABILITY"]), "REJECTION count": (counts["REJECTION"], EXP["REJECTION"]), "NONE count": (counts["NONE"], EXP["NONE"]),
          "interested TIME": (counts_I["TIME"], 36), "interested ACCESS": (counts_I["ACCESS"], 31), "interested ORIENTATION": (counts_I["ORIENTATION"], 12), "interested NONE": (counts_I["NONE"], 13),
          "interested BENEFIT": (counts_I["BENEFIT"], EXP["I_BENEFIT"]), "interested RELIABILITY": (counts_I["RELIABILITY"], EXP["I_RELIABILITY"]),
          "not available, all non-users": (int(res.loc[res.barrier == "Not available", "n_all_nonusers"].iloc[0]), 91),
          "lack of knowledge, all non-users": (int(res.loc[res.barrier == "Lack of knowledge / training", "n_all_nonusers"].iloc[0]), 63),
          "data protection, all non-users": (int(res.loc[res.barrier == "Data protection / security concerns", "n_all_nonusers"].iloc[0]), 52)}
bad = [k for k, (o, e) in checks.items() if o != e]
for k, (o, e) in checks.items():
    log(f"{'OK  ' if o == e else 'FAIL'} {k}: {o} (expected {e})")
assert not bad, f"CHECKSUMS VIOLATED: {bad}"
log("\nAll checksums OK.")
LOG.close()
