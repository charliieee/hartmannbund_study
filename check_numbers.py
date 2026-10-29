"""
check_numbers.py  --  consistency check of the manuscript, Supplementary Information and response letter
against the script outputs in out/.

Part A  Named checks: a curated list of (claim in text, value computed from a CSV/log in out/). Every check asserts that the
        string appears in the given text file and that the recomputed value rounds to it.
Part B  Coverage scan: every numeric token in the main text (Abstract, Results, Discussion, Methods, legends, Tables 2-4) is
        looked up in the pool of all numbers that occur in out/**/*.csv, out/**/*.txt and out/**/*.md (rounded to the
        precision used in the text).  Tokens without a match are listed for manual review (they are usually dates, counts
        that appear only in logs, reference numbers, or arithmetic such as '53%').

Usage: python check_numbers.py <out_dir> <manuscript.md> <supplement.md> <response.md>
"""
import glob
import os
import re
import sys

import pandas as pd

OUT, MS, SUPP, RESP = sys.argv[1:5]
ms = open(MS, encoding="utf8").read()
supp = open(SUPP, encoding="utf8").read()
resp = open(RESP, encoding="utf8").read()
ALL = ms + "\n" + supp + "\n" + resp
fails = []
n_ok = 0


def csv(path, **kw):
    return pd.read_csv(os.path.join(OUT, path), **kw)


def check(label, text_value, computed, where=ALL, places=1):
    """text_value: string as it should appear in the text; computed: number from the outputs."""
    global n_ok
    if isinstance(computed, (int,)) or places == 0:
        ok_val = str(int(round(float(computed)))) == str(text_value).replace("%", "").replace("+", "").replace("−", "").strip()
    else:
        ok_val = f"{float(computed):.{places}f}" == str(text_value).replace("%", "").replace("+", "").replace("−", "").strip()
    present = str(text_value) in where
    if ok_val and present:
        n_ok += 1
    else:
        fails.append(f"{label}: text '{text_value}' | computed {computed} | value match {ok_val} | present in text {present}")


# ---------------------------------------------------------------- A  named checks
A4 = csv("adoption_groups/A_omnibus.csv").set_index("option")
for opt, col, val in [("Specialised", "pct_N", "17.5"), ("Specialised", "pct_I", "23.3"), ("Specialised", "pct_R-", "36.5"), ("Specialised", "pct_R+", "60.7"),
                      ("Strategic", "pct_N", "17.5"), ("Strategic", "pct_I", "26.7"), ("Strategic", "pct_R-", "29.4"), ("Strategic", "pct_R+", "45.8"),
                      ("Large language models", "pct_N", "17.5"), ("Large language models", "pct_I", "36.0"), ("Large language models", "pct_R-", "43.5"), ("Large language models", "pct_R+", "40.2"),
                      ("Ethics and law", "pct_N", "30.2"), ("Ethics and law", "pct_I", "10.5"), ("Ethics and law", "pct_R-", "12.9"), ("Ethics and law", "pct_R+", "15.0"),
                      ("Commercial AI vendors", "pct_N", "4.8"), ("Commercial AI vendors", "pct_I", "22.1"), ("Commercial AI vendors", "pct_R-", "15.3"), ("Commercial AI vendors", "pct_R+", "29.9"),
                      ("Diagnostic systems", "pct_N", "28.6"), ("Diagnostic systems", "pct_I", "52.3"), ("Diagnostic systems", "pct_R-", "43.5"), ("Diagnostic systems", "pct_R+", "51.4"),
                      ("Critical appraisal", "pct_N", "31.7"), ("Critical appraisal", "pct_I", "10.5"), ("Critical appraisal", "pct_R-", "20.0"), ("Critical appraisal", "pct_R+", "18.7")]:
    check(f"omnibus {opt} {col}", val, A4.loc[opt, col], ms)
check("V advanced", "0.36", A4.loc["Specialised", "cramers_v"], ms, 2)
check("V strategic", "0.22", A4.loc["Strategic", "cramers_v"], ms, 2)
check("V introductory", "0.03", A4.loc["Introductory", "cramers_v"], ms, 2)
check("BH strategic", "0.007", A4.loc["Strategic", "p_adj"], ms, 3)
check("BH LLM", "0.045", A4.loc["Large language models", "p_adj"], ms, 3)
check("BH ethics", "0.045", A4.loc["Ethics and law", "p_adj"], ms, 3)
check("BH vendors", "0.007", A4.loc["Commercial AI vendors", "p_adj"], ms, 3)
check("BH diagnostic systems", "0.062", A4.loc["Diagnostic systems", "p_adj"], ms, 3)
check("BH critical appraisal", "0.062", A4.loc["Critical appraisal", "p_adj"], ms, 3)
check("n tested", "30", int(A4.tested.sum()), ms, 0)
assert int(A4.significant.sum()) == 5 and "Five options differed" in ms
# ranges of delivery options
for opt, lo, hi in [("Short formats (1-2 h)", "73", "86"), ("Self-paced online courses", "63", "81"), ("Application-oriented", "87", "96"), ("Introductory", "51", "55")]:
    vals = [A4.loc[opt, c] for c in ["pct_N", "pct_I", "pct_R-", "pct_R+"]]
    check(f"range {opt} low", lo, min(vals), ms, 0)
    check(f"range {opt} high", hi, max(vals), ms, 0)

PW = csv("adoption_groups/B_pairwise.csv")
r = PW[(PW.option == "Specialised") & (PW.contrast == "R- vs R+")].iloc[0]
check("advanced R- vs R+ rd", "+24", -r.rd_pp, ms, 0)
check("advanced R- vs R+ lo", "10", -r.hi, ms, 0)
check("advanced R- vs R+ hi", "37", -r.lo, ms, 0)
check("advanced R- vs R+ p", "0.007", r.p_fisher_adj, ms, 3)
r = PW[(PW.option == "Ethics and law") & (PW.contrast == "N vs R+")].iloc[0]
check("ethics N vs R+ rd", "+15", r.rd_pp, ms, 0)
check("ethics N vs R+ p", "0.056", r.p_fisher_adj, ms, 3)
r = PW[(PW.option == "Commercial AI vendors") & (PW.contrast == "N vs R-")].iloc[0]
check("vendors N vs R- rd", "11", abs(r.rd_pp), ms, 0)
check("vendors N vs R- p", "0.10", r.p_fisher_adj, ms, 2)

RD = csv("adoption_groups/C_risk_differences.csv")
W = RD[RD.block == "W"]
check("equiv bound min", "13", W.groupby("option").equiv_bound_pp.max().min(), ms, 0)
check("equiv bound max", "36", W.groupby("option").equiv_bound_pp.max().max(), ms, 0)
for opt, val in [("Self-paced online courses", "18"), ("Live webinars", "24"), ("Short formats (1-2 h)", "13")]:
    rr = W[(W.option == opt) & (W.contrast == "N vs R+")].iloc[0]
    check(f"N vs R+ {opt}", val, abs(rr.rd_pp), ms, 0)
    assert rr.hi95 < 0, f"{opt} CI should exclude zero"

D = csv("adoption_groups/D_trend.csv").set_index("option")
check("advanced OR per group", "1.9", D.loc["Specialised", "OR_A"], ms, 1)
check("advanced OR lo", "1.5", D.loc["Specialised", "loA"], ms, 1)
check("advanced OR hi", "2.4", D.loc["Specialised", "hiA"], ms, 1)
check("advanced routine OR", "2.5", D.loc["Specialised", "OR_routine"], ms, 1)
check("advanced routine lo", "1.3", D.loc["Specialised", "lo_routine"], ms, 1)
check("advanced routine hi", "4.8", D.loc["Specialised", "hi_routine"], ms, 1)
check("advanced breadth OR", "1.7", D.loc["Specialised", "OR_breadth"], ms, 1)
check("advanced breadth lo", "1.2", D.loc["Specialised", "lo_breadth"], ms, 1)
check("advanced breadth hi", "2.5", D.loc["Specialised", "hi_breadth"], ms, 1)
check("LLM threshold OR", "0.28", D.loc["Large language models", "OR_thr"], ms, 2)
check("ethics threshold OR", "3.61", D.loc["Ethics and law", "OR_thr"], ms, 2)
check("vendors threshold OR", "0.16", D.loc["Commercial AI vendors", "OR_thr"], ms, 2)
check("vendors routine OR", "2.3", D.loc["Commercial AI vendors", "OR_routine"], ms, 1)
check("vendors routine lo", "1.0", D.loc["Commercial AI vendors", "lo_routine"], ms, 1)
check("vendors routine hi", "4.9", D.loc["Commercial AI vendors", "hi_routine"], ms, 1)
check("diagnostic systems threshold OR", "0.40", D.loc["Diagnostic systems", "OR_thr"], ms, 2)
check("diagnostic systems threshold lo", "0.22", D.loc["Diagnostic systems", "lo_thr"], ms, 2)
check("diagnostic systems threshold hi", "0.74", D.loc["Diagnostic systems", "hi_thr"], ms, 2)
CG = csv("content_measures/content_measures.csv").set_index("group")
check("clin any no intention", "49.2", CG.loc["No current intention", "any_clinical_topic_pct"], ms, 1)
check("clin any interested", "77.9", CG.loc["Interested non-users", "any_clinical_topic_pct"], ms, 1)
check("private non-routine", "72.9", CG.loc["Non-routine use", "private_ai_use_pct"], ms, 1)
check("passive interested", "19.8", CG.loc["Interested non-users", "passive_engagement_pct"], ms, 1)
check("passive non-routine", "7.1", CG.loc["Non-routine use", "passive_engagement_pct"], ms, 1)
check("active routine", "65", CG.loc["Routine use", "informs_actively_pct"], ms, 0)
CT = csv("content_measures/content_measures_tests.csv").set_index(["measure", "contrast"])
check("private p non-routine vs interested", "0.007", CT.loc[("private AI use", "non-routine vs interested"), "p_fisher"], ms, 3)
check("gov any p", "0.003", CT.loc[("any governance topic", "interested vs no intention"), "p_fisher"], ms, 3)
assert CT.loc[("any clinical application topic", "interested vs no intention"), "p_fisher"] < 0.001
PE2 = csv("early_career/preferences_by_career_stage.csv").set_index("option")
check("early universities", "46", PE2.loc["Universities", "pct_0_4"], resp, 0)
check("later universities", "24", PE2.loc["Universities", "pct_5plus"], resp, 0)
check("early associations", "40", PE2.loc["Professional associations", "pct_0_4"], resp, 0)
check("later associations", "60", PE2.loc["Professional associations", "pct_5plus"], resp, 0)
check("early KV", "20", PE2.loc["Health insurance assoc.", "pct_0_4"], resp, 0)
check("later KV", "38", PE2.loc["Health insurance assoc.", "pct_5plus"], resp, 0)
SX = csv("item_screen/item_screen.csv").set_index("item")
check("CDSS unclear", "41.9", SX.loc["Clinical decision support", "unclear_all"], supp, 1)
AF = csv("age_by_format/age_by_format.csv").set_index("format")
check("on-site under 40", "35.9", AF.loc["On-site sessions / workshops", "pct_under_40"], ms, 1)
check("on-site 50+", "36.3", AF.loc["On-site sessions / workshops", "pct_50_or_older"], ms, 1)

E = csv("adoption_groups/E_composition.csv", index_col=0)
check("intro only N", "38", E.loc["No current intention", "intro_only_depth_pct"], ms, 0)
check("intro only R+", "20", E.loc["Routine use", "intro_only_depth_pct"], ms, 0)
check("spec no intro N", "10", E.loc["No current intention", "spec_no_intro_pct"], ms, 0)
check("spec no intro R+", "31", E.loc["Routine use", "spec_no_intro_pct"], ms, 0)
check("gov any N", "60.3", E.loc["No current intention", "gov_any_pct"], ms, 1)
check("gov any I", "34.9", E.loc["Interested non-users", "gov_any_pct"], ms, 1)
check("topics N", "2.3", E.loc["No current intention", "n_topics"], ms, 1)

Q16 = csv("nonuser_reasons/q16_by_nonuser_group.csv").set_index("barrier")
check("Q16 benefit N", "30.2", Q16.loc["No perceived benefit", "pct_no_intention"], ms, 1)
check("Q16 benefit I", "14.0", Q16.loc["No perceived benefit", "pct_interested"], ms, 1)
check("Q16 benefit rd", "16", Q16.loc["No perceived benefit", "rd_pp"], ms, 0)
check("Q16 knowledge N", "33.3", Q16.loc["Lack of knowledge / training", "pct_no_intention"], ms, 1)
check("Q16 knowledge I", "48.8", Q16.loc["Lack of knowledge / training", "pct_interested"], ms, 1)
Q30 = csv("nonuser_reasons/q30_categories_by_nonuser_group.csv").set_index("category")
check("Q30 benefit N", "25.4", Q30.loc["BENEFIT", "pct_no_intention"], ms, 1)
check("Q30 benefit I", "4.7", Q30.loc["BENEFIT", "pct_interested"], ms, 1)
check("Q30 reliability N", "14.3", Q30.loc["RELIABILITY", "pct_no_intention"], ms, 1)
check("Q30 reliability I", "2.3", Q30.loc["RELIABILITY", "pct_interested"], ms, 1)
check("Q30 rejection N", "12.7", Q30.loc["REJECTION", "pct_no_intention"], ms, 1)
check("Q30 rejection I", "1.2", Q30.loc["REJECTION", "pct_interested"], ms, 1)
check("Q30 orientation I", "14.0", Q30.loc["ORIENTATION", "pct_interested"], ms, 1)
CM = csv("nonuser_reasons/q30_composites_by_nonuser_group.csv").iloc[0]
check("evaluative N", "39.7", CM.pct_no_intention, ms, 1)
check("evaluative I", "7.0", CM.pct_interested, ms, 1)
check("evaluative rd", "33", CM.rd_pp, ms, 0)
check("evaluative lo", "19", CM.rd_ci_low, ms, 0)
check("evaluative hi", "46", CM.rd_ci_high, ms, 0)

C1 = csv("no_intention_group/1_composition.csv").set_index("measure")
check("private N", "34.9", C1.loc["private AI use, %", "no_intention"], ms, 1)
check("private I", "52.3", C1.loc["private AI use, %", "interested"], ms, 1)
check("provision N", "4.8", C1.loc["employer provides AI, %", "no_intention"], ms, 1)
check("unrec3 N", "34.9", C1.loc["not recognised >= 3, %", "no_intention"], ms, 1)
check("unrec3 I", "23.3", C1.loc["not recognised >= 3, %", "interested"], ms, 1)

T2 = csv("regression/table2.csv").set_index("term")
check("OR provision", "14.39", T2.loc["Employer provides AI applications: yes (ref. no)", "OR"], ms, 2)
check("OR provision lo", "6.53", T2.loc["Employer provides AI applications: yes (ref. no)", "ci_low"], ms, 2)
check("OR provision hi", "31.72", T2.loc["Employer provides AI applications: yes (ref. no)", "ci_high"], ms, 2)
check("OR private", "4.11", T2.loc["Private AI use (ref. none)", "OR"], ms, 2)
check("OR private lo", "2.34", T2.loc["Private AI use (ref. none)", "ci_low"], ms, 2)
check("OR private hi", "7.21", T2.loc["Private AI use (ref. none)", "ci_high"], ms, 2)
check("OR competence", "1.34", T2.loc["Self-rated professional digital competence (per point, 1-5)", "OR"], ms, 2)
check("OR competence lo", "1.01", T2.loc["Self-rated professional digital competence (per point, 1-5)", "ci_low"], ms, 2)
check("OR competence hi", "1.77", T2.loc["Self-rated professional digital competence (per point, 1-5)", "ci_high"], ms, 2)
check("OR training", "0.71", T2.loc["Prior AI-specific training (ref. none)", "OR"], ms, 2)
check("OR training lo", "0.32", T2.loc["Prior AI-specific training (ref. none)", "ci_low"], ms, 2)
check("OR training hi", "1.61", T2.loc["Prior AI-specific training (ref. none)", "ci_high"], ms, 2)
check("OR sector", "1.09", T2.loc["Ambulatory sector (ref. hospital)", "OR"], ms, 2)
check("OR male", "1.24", T2.loc["Male sex (ref. female)", "OR"], ms, 2)
check("OR age40", "2.19", T2.loc["Age 40-49 years (ref. <40)", "OR"], ms, 2)
check("OR age50", "0.71", T2.loc["Age 50-59 years", "OR"], ms, 2)
check("OR age50 hi", "1.60", T2.loc["Age 50-59 years", "ci_high"], ms, 2)
check("OR age60", "0.95", T2.loc["Age >=60 years", "OR"], ms, 2)
check("OR dontknow", "0.99", T2.loc["Employer provides AI applications: don't know", "OR"], ms, 2)
BT = csv("regression/table2_block_tests.csv").set_index("test")
check("block person", "42.2", BT.loc["person block (age, sex, competence, training, private AI use)", "chi2"], ms, 1)
check("block access", "64.0", BT.loc["access block (sector, provision)", "chi2"], ms, 1)
check("block private", "26.2", BT.loc["private AI use", "chi2"], ms, 1)
check("block person w/o private", "12.2", BT.loc["person block without private AI use (age, sex, competence, training)", "chi2"], ms, 1)
check("block person w/o private p", "0.057", BT.loc["person block without private AI use (age, sex, competence, training)", "p"], ms, 3)
check("age block p", "0.094", BT.loc["age", "p"], ms, 3)
log2 = open(os.path.join(OUT, "regression/regression_log.txt"), encoding="utf8").read()
check("pseudo R2", "0.264", float(re.search(r"McFadden pseudo-R2 = ([0-9.]+)", log2).group(1)), ms, 3)
check("AUC", "0.83", float(re.search(r"AUC = ([0-9.]+)", log2).group(1)), ms, 2)
NA = csv("regression/note3_tableA.csv")
NA["key"] = NA.analysis.str.extract(r"^(A\d+)")[0]
NA = NA.set_index("key")
check("A8 employed OR", "16.9", float(NA.loc["A8", "provision_yes"].split(" ")[0]), ms, 1)
check("A9 practice OR", "15.4", float(NA.loc["A9", "provision_yes"].split(" ")[0]), ms, 1)
assert int(NA.loc["A8", "n"]) == 176 and int(NA.loc["A9", "n"]) == 152
check("A2 provision without private", "13.5", float(NA.loc["A2", "provision_yes"].split(" ")[0]), resp, 1)
assert "0.87 (0.74-1.01)" in str(NA.loc["A11", "extra"]), "A11 not recognised OR 0.87 not found in note3_tableA"
assert "OR 0.87, 95% CI 0.74–1.01" in ms and "OR 0.87 per application, 95% CI 0.74–1.01" in supp

PT = csv("application_counts/4_provision_x_trained.csv", index_col=0)
check("provision yes trained", "36.2", 100 * PT.loc["yes", "1"] / PT.loc["yes", "All"], ms, 1)
check("provision no trained", "11.1", 100 * PT.loc["no", "1"] / PT.loc["no", "All"], ms, 1)
UO = csv("application_counts/2_ai_use_by_unfamiliar_overall.csv").set_index("un_cat")
check("AI use unrec 0", "66.0", UO.loc["0", "mean"], ms, 1)
check("AI use unrec 1-2", "51.3", UO.loc["1-2", "mean"], ms, 1)
check("AI use unrec 3-4", "50.0", UO.loc["3-4", "mean"], ms, 1)
check("AI use unrec 5+", "37.9", UO.loc["5+", "mean"], ms, 1)
logp = open(os.path.join(OUT, "application_counts/application_counts_log.txt"), encoding="utf8").read()
check("within core OR", "0.85", float(re.search(r"OR per application not recognised [0-9.]+ \(unrounded ([0-9.]+)\)", logp).group(1)), ms, 2)
check("trend p", "0.002", float(re.search(r"trend across the four categories \(logit on category index\): p = ([0-9.]+)", logp).group(1)), ms, 3)

PS = csv("providers_by_sector/providers_by_sector.csv").set_index("provider")
check("employers hospital", "82.2", PS.loc["Employers", "pct_hospital"], ms, 1)
check("employers ambulatory", "24.1", PS.loc["Employers", "pct_ambulatory"], ms, 1)
check("KV hospital", "14.5", PS.loc["Health insurance assoc.", "pct_hospital"], ms, 1)
check("KV ambulatory", "50.8", PS.loc["Health insurance assoc.", "pct_ambulatory"], ms, 1)
check("chambers hospital", "61.8", PS.loc["Chambers of physicians", "pct_hospital"], ms, 1)
check("chambers ambulatory", "68.7", PS.loc["Chambers of physicians", "pct_ambulatory"], ms, 1)

US = csv("unsure_group/unsure_group.csv").set_index("measure")
check("unsure competence", "2.5", US.loc["self-rated professional digital competence, mean (1-5)", "unsure (n = 15)"], ms, 1)
check("unsure unrec", "3.1", US.loc["surveyed applications not recognised, mean of 16", "unsure (n = 15)"], ms, 1)
check("sample unrec", "1.6", US.loc["surveyed applications not recognised, mean of 16", "all respondents (n = 362)"], ms, 1)

AD = csv("early_career/adoption_by_career_stage.csv").set_index("measure")
check("early use", "60", AD.loc["Professional AI use (defined status)", "pct_0_4"], ms, 0)
check("later use", "56", AD.loc["Professional AI use (defined status)", "pct_5plus"], ms, 0)
check("early training", "14", AD.loc["Prior AI-specific training", "pct_0_4"], ms, 0)
check("later training", "18", AD.loc["Prior AI-specific training", "pct_5plus"], ms, 0)
check("early private", "71", AD.loc["Private AI use", "pct_0_4"], ms, 0)
check("later private", "60", AD.loc["Private AI use", "pct_5plus"], ms, 0)
check("early unauthorised", "50", AD.loc["Self-reported unauthorised use (valid Q32)", "pct_0_4"], ms, 0)
check("later unauthorised", "24", AD.loc["Self-reported unauthorised use (valid Q32)", "pct_5plus"], ms, 0)
PE = csv("early_career/preferences_by_career_stage.csv").set_index("option")
check("early employers", "82", PE.loc["Employers", "pct_0_4"], ms, 0)
check("later employers", "43", PE.loc["Employers", "pct_5plus"], ms, 0)

logs = open(os.path.join(OUT, "exposure_markers/exposure_markers_log.txt"), encoding="utf8").read()
gov_lines = [l for l in logs.split("\n") if l.strip().startswith("Any governance topic")]
shadow_line, private_line = gov_lines[0].split(), gov_lines[1].split()   # first block: unauthorised use; second: private AI use
check("gov any unauthorised", "34", float(shadow_line[3]), ms, 0)
check("gov any not unauthorised", "50", float(shadow_line[4]), ms, 0)
check("gov any unauth OR S1", "0.53", float(shadow_line[6]), supp, 2)
check("gov any private", "41", float(private_line[3]), ms, 0)
check("gov any not private", "55", float(private_line[4]), ms, 0)
check("gov any private OR S1", "0.50", float(private_line[6]), ms, 2)

QF = csv("provision_form/provision_forms.csv").set_index("form")
check("integrated", "46", QF.loc["Integrated into existing medical software (e.g. diagnostic systems, decision support)", "pct_provision"], ms, 0)
check("standalone", "33", QF.loc["Standalone tools or programs (e.g. AI models or chatbots, speech recognition)", "pct_provision"], ms, 0)
logprov = open(os.path.join(OUT, "provision_form/provision_form_log.txt"), encoding="utf8").read()
check("users without provision private", "76.2", float(re.search(r"with private AI use 64 \(([0-9.]+)%\)", logprov).group(1)), ms, 1)
check("decision-proximal", "57.3", float(re.search(r"any decision-proximal area \(diagnostics or treatment planning\): 110 \(([0-9.]+)%\)", logprov).group(1)), ms, 1)

# LCA (the LCA scripts outputs, lca_primary and application_counts)
D3 = csv("lca_primary/D_profiles_k3.csv").set_index("cls3")
for cls, n, ai, prov, tr, hosp in [("Wide range, routine use", 69, "83.1", "56.5", "26.1", "26.1"), ("Wide range, occasional use", 122, "54.9", "24.6", "15.6", "42.6"), ("Core set, routine use", 165, "45.9", "20.6", "14.5", "48.5")]:
    assert int(D3.loc[cls, "n"]) == n
    check(f"LCA {cls} AI use", ai, D3.loc[cls, "ai_use_pct"], ms, 1)
    check(f"LCA {cls} provision", prov, D3.loc[cls, "provision_pct"], ms, 1)
    check(f"LCA {cls} trained", tr, round(D3.loc[cls, "trained_pct"] / 100 * n) / n * 100, ms, 1)
    check(f"LCA {cls} hospital", hosp, D3.loc[cls, "hospital_pct"], ms, 1)
check("core set not available cells", "38.1", D3.loc["Core set, routine use", "cells_not_available_pct"], ms, 1)
check("core set unclear cells", "13.9", D3.loc["Core set, routine use", "cells_unclear_pct"], ms, 1)
check("routine weekly-daily cells", "62", D3.loc["Wide range, routine use", "cells_daily_pct"] + D3.loc["Wide range, routine use", "cells_weekly_pct"], ms, 0)
check("occasional rarely-monthly cells", "56", D3.loc["Wide range, occasional use", "cells_monthly_pct"] + D3.loc["Wide range, occasional use", "cells_rarely_pct"], ms, 0)
MS8 = csv("lca_primary/A_model_selection.csv").set_index("k")
check("BIC k3", "12250", MS8.loc[3, "BIC"], ms, 0)
check("BIC k4", "12259", MS8.loc[4, "BIC"], ms, 0)
check("BIC k2", "12319", MS8.loc[2, "BIC"], ms, 0)
check("entropy k3", "0.902", MS8.loc[3, "entropy"], ms, 3)
CA = csv("lca_primary/C_assignment_certainty.csv")
allrow = CA[(CA.k == 3) & (CA.class_ == "all")].iloc[0]
check("mean posterior", "0.960", allrow.mean_max_posterior, ms, 3)
check("share >= 0.8", "92.1", 100 * allrow.share_ge_0_8, ms, 1)
check("share < 0.8 (limitations)", "7.9", 100 * (1 - allrow.share_ge_0_8), ms, 1)
CH = csv("lca_primary/D_chi2_k3.csv", index_col=0)
check("chi2 provision", "32.2", CH.loc["provision", "chi2"], ms, 1)
check("chi2 ai use", "26.0", CH.loc["ai_use", "chi2"], ms, 1)
check("p trained", "0.087", CH.loc["trained", "p"], ms, 3)
ER = csv("lca_primary/E_regressions_k3.csv").set_index("outcome")
check("LR class adj sector", "28.8", ER.loc["ai_use", "LR_class_adj_sector"], ms, 1)
check("interaction LR", "0.32", ER.loc["ai_use", "LR_interaction"], ms, 2)
check("interaction p", "0.85", ER.loc["ai_use", "p_interaction"], ms, 2)
CV = csv("lca_primary/B_convergence_summary.csv")
k3 = CV[(CV.k == 3)].starts_at_best.astype(int); k4 = CV[(CV.k == 4)].starts_at_best.astype(int)
assert k3.min() >= 20 and k3.max() <= 22 and k4.min() >= 1 and k4.max() <= 3, "convergence counts changed"
ST = csv("lca_primary/E_stratified_k3.csv")
h = ST[(ST.k == 3) & (ST.stratum == "hospital") & (ST.class_ == "Core set, routine use")].iloc[0]
nh = ST[(ST.k == 3) & (ST.stratum != "hospital") & (ST.class_ == "Core set, routine use")].iloc[0]
check("core set hospital AI use", "48.1", h.ai_use_pct, ms, 1)
check("core set non-hospital AI use", "43.8", nh.ai_use_pct, ms, 1)
assert int(h.ai_use_n) == 77 and int(nh.ai_use_n) == 80
PU = csv("application_counts/2_ai_use_by_profile_x_unfamiliar.csv")
cs = PU[PU.profile == "Core set, routine use"].set_index("un_cat")
check("core set unrec 0", "58.3", cs.loc["0", "mean"], ms, 1)
check("core set unrec 5+", "29.2", cs.loc["5+", "mean"], ms, 1)
CB = csv("application_counts/1_counts_by_profile.csv").set_index("profile")
n5 = sum(round(CB.loc[p, "unclear_ge5_pct"] / 100 * CB.loc[p, "n"]) for p in CB.index)
n5_core = round(CB.loc["Core set, routine use", "unclear_ge5_pct"] / 100 * 165)
assert n5 == 34 and n5_core == 27 and "27 of these 34" in ms
check("share 5+ unrecognised", "9.6", 100 * n5 / 356, ms, 1)

print(f"\nPart A: {n_ok} named checks OK, {len(fails)} failed")
for f in fails:
    print("  FAIL", f)

# ---------------------------------------------------------------- B  coverage scan
pool = set()
for f in glob.glob(os.path.join(OUT, "**", "*"), recursive=True):
    if f.endswith((".csv", ".txt", ".md")):
        try:
            txt = open(f, encoding="utf8", errors="ignore").read()
        except Exception:
            continue
        for tok in re.findall(r"-?\d+\.\d+|\d+", txt):
            try:
                v = float(tok)
            except ValueError:
                continue
            for d in (0, 1, 2, 3):
                pool.add(round(v, d))
                pool.add(round(abs(v), d))
# numbers from the original descriptives that live in the repository-root scripts (Fig. 1, Fig. 3, Table 1, Note 2) are
# whitelisted here from the manuscript and from fig1.py output
whitelist = set()
for f in ["fig1/fig1_log.txt"]:
    pass
body = ms.split("# References")[0]
tokens = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])", body)
missing = []
for tok in sorted(set(tokens), key=lambda x: float(x)):
    v = float(tok)
    d = len(tok.split(".")[1]) if "." in tok else 0
    if round(v, d) in pool or round(abs(v), d) in pool:
        continue
    missing.append(tok)
print(f"\nPart B: {len(set(tokens))} distinct numeric tokens in the main text; {len(missing)} without a match in out/ (review manually):")
print("  " + ", ".join(missing))
if fails:
    sys.exit(1)
