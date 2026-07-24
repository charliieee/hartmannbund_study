"""
Results R2 (RQ1): key figures + Fig. 1 (panels a-d).

Input:  analysis_dataset.csv (from prepare_data.py)
Output: console printout of all R2 figures, fig1.png / fig1.pdf
        (portrait) and fig1a..fig1d .png/.pdf (individual panels)

Panels: a Usage (n=362) | b Areas of use (users, n=192)
        c Training: rate + formats (n=62) | d Barriers (non-users, n=149)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("analysis_dataset.csv")
C = df.columns
n = len(df)

BLUE = ["#1b4f8a", "#3d7ab5", "#8ab4d8"]      # user shades
GRAY = ["#8a8a8a", "#b0b0b0", "#d4d4d4"]      # non-users / unclear


def multi_counts(series, mapping=None):
    """Count multiple-response answers; optionally recode free text via mapping.

    mapping: dict free-text -> list of target categories ([] = uncodable).
    Counted per person as a set (no double counting after recoding).
    """
    def to_set(cell):
        out = set()
        for tok in str(cell).split(";"):
            tok = tok.strip()
            if not tok or tok == "nan":
                continue
            out.update(mapping.get(tok, [tok]) if mapping else [tok])
        return out
    sets = series.fillna("").apply(to_set)
    return pd.Series([x for s in sets for x in s]).value_counts()


# =============================================================================
#  a) USAGE
# =============================================================================
use_order = [
    ("Ja, regelmäßig (mehrfach pro Woche)", "Regularly (several times/week)", BLUE[0]),
    ("Ja, gelegentlich (einmal pro Woche)", "Occasionally (about once/week)", BLUE[1]),
    ("Ja, selten/experimentell (einmal pro Monat)", "Rarely / experimentally", BLUE[2]),
    ("Nein, aber geplant/Interesse vorhanden", "Interested non-user", GRAY[0]),
    ("Nein, und aktuell nicht geplant", "No current intention", GRAY[1]),
]
vc = df[C[42]].value_counts()
a_vals = [vc[k] for k, _, _ in use_order] + [int(df["ai_use"].isna().sum())]
a_labs = [lab for _, lab, _ in use_order] + ["Uncertain / no answer"]
a_cols = [c for _, _, c in use_order] + [GRAY[2]]
n_users = int(df["ai_use"].eq(1).sum())
print(f"a) users {n_users}/{n} = {n_users/n*100:.1f}%  |", dict(zip(a_labs, a_vals)))


# =============================================================================
#  b) AREAS OF USE (users)
# =============================================================================
# Area-of-use categories (names used as dict values and plot labels).
RESEARCH = "Research"
ADMIN = "Administration"
DIAGNOSTIC = "Diagnostics"
TREATMENT = "Treatment planning"
PATIENT = "Patient communication"
AREA_MAP = {
    "Administrative Aufgaben (z.B. Dokumentation wie Arztbrief schreiben, Abrechnung, etc.)": [ADMIN],
    "Diagnostik und Befund (z.B. Bildanalyse, Mustererkennung, EKG-Auswertung etc.)": [DIAGNOSTIC],
    "Behandlungsplanung und Therapieentscheidungen (z.B. Mediaktionsmanagement, personalisierte Therapieempfehlungen)": [TREATMENT],
    "Forschung (z.B. Datenanalyse, Literaturrecherche, Simulationen)": [RESEARCH],
    "Patientenkommunikation und Service (z.B. Chatbots, Terminvereinbarung, automatisierte Anamnese-Tools)": [PATIENT],
    "Recherche": [RESEARCH], "Recherche, Patienteninformation": [RESEARCH, PATIENT],
    "Übersetzungen für fremdsprachliche Patienten": [PATIENT],
    "Spracherkennung und Literaturrecherche": [RESEARCH, ADMIN],
    "Behandlungsdokumentation": [ADMIN], "OpenEvidence": [RESEARCH],
    "Diktiergerät": [ADMIN], "Marketing": [ADMIN],
    "Dokumentation / spracherkennung": [ADMIN],
    "Unterstützung bei seltenen Diagnosen zur Unterstützung bei Diagnose- und Therapiefindung": [DIAGNOSTIC],
    "Bei der Strahlentherapie selbst": [DIAGNOSTIC],
    "zusätzliche Auswertung der Lunge": [DIAGNOSTIC],
    "Erstellen von Prüfungsfragen": [], "Ich benutze meine privaten Account": [],
    "Nur wenig Nützliches bei uns vorhanden, für Dienstplan etc. kein Geld zur Verfügung gestellt": [],
}
users = df["ai_use"] == 1
b = multi_counts(df.loc[users, C[43]], AREA_MAP)
b_order = [ADMIN, DIAGNOSTIC, TREATMENT, RESEARCH, PATIENT]
print("b)", {k: int(b[k]) for k in b_order}, f"of n={n_users}")


# =============================================================================
#  c) TRAINING: rate + formats
# =============================================================================
n_tr = int(df["trained"].sum())
FORMAT_MAP = {
    "Kurzvortrag/Informationsveranstaltung (<2 Stunden)": ["Short lecture (<2 h)"],
    "Online-Modul/E-learning": ["E-learning module"],
    "Workshop (wenige Stunden bis 1 Tag)": ["Workshop"],
    "Integrierte in anderen Fortbildungen": ["Integrated in other CME"],
    "Mehrtägiger Kurs/Zertifikatslehrgang": ["Certificate / multi-day"],
    "Seminar des Hartmannbund. Ich habe nix verstanden und nach 15 min abgebrochen": ["Short lecture (<2 h)"],
    "Keine Antwort": [],
}
cformats = multi_counts(df.loc[df["trained"] == 1, C[46]], FORMAT_MAP)
useful = df.loc[df["trained"] == 1, C[47]].astype(float)
print(f"c) trained {n_tr}/{n} = {n_tr/n*100:.1f}% |", dict(cformats),
      f"| usefulness median {useful.median():.0f} "
      f"[IQR {useful.quantile(.25):.0f}-{useful.quantile(.75):.0f}], "
      f"top-2 {(useful >= 4).mean()*100:.1f}%")


# =============================================================================
#  d) BARRIERS (non-users, n = 149)
# =============================================================================
BARRIER_MAP = {
    "Nicht verfügbar": ["Not available"],
    "Fehlendes Wissen/Schulungen": ["Lack of knowledge / training"],
    "Sicherheits-/Datenschutzbedenken": ["Data protection / security concerns"],
    "Mangelende Integrationsmöglichkeiten in bestehenden Systemen": ["Insufficient system integration"],
    "Keine wahrgenommene Nützlichkeit/Zeitgewinn": ["No perceived benefit"],
    "fehlende Kapazitäten / Zeitmangel für Implementierung": ["Lack of time / capacity"],
    "kein bedarf": ["No perceived benefit"],
    "Als sporadisch in der Notfallpraxis tätiger Arzt bisher nicht erforderlich für mich, wird auch nicht bereitgestellt": ["Not available"],
    "Die Firmen, die Programme für die Blutbanklogistik entwickeln, haben KI noch nicht in deren vorhandenen Programme implementiert.": ["Not available"],
    "Kosten": ["Costs"], "Kh hat kein Geld": ["Costs"], "Testphase": ["Other"],
}
nonuser = df["ai_use"] == 0
n_nu = int(nonuser.sum())
d = multi_counts(df.loc[nonuser, C[39]], BARRIER_MAP)
d_order = ["Not available", "Lack of knowledge / training",
           "Data protection / security concerns", "Insufficient system integration",
           "No perceived benefit", "Lack of time / capacity"]
print("d)", {k: int(d[k]) for k in d_order}, f"of n={n_nu}")


# =============================================================================
#  FIGURE
# =============================================================================
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})


def hbar(ax, labels, counts, denom, colors, title, xlabel, annot=None):
    y = np.arange(len(labels))[::-1]
    pct = [c / denom * 100 for c in counts]
    ax.barh(y, pct, height=0.62, color=colors)
    ax.set_yticks(y, labels)
    for yi, (c, p) in zip(y, zip(counts, pct)):
        ax.text(p + 1, yi, f"{c} ({p:.1f}%)", va="center", fontsize=7)
    ax.set_xlim(0, max(pct) * 1.32)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    if annot:
        ax.text(0.98, 0.06, annot, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, style="italic")


c_order = ["Short lecture (<2 h)", "E-learning module", "Workshop",
           "Integrated in other CME", "Certificate / multi-day"]

# One panel = one entry (list order = top-to-bottom stacking a-d in portrait).
panels = [
    dict(key="a", labels=a_labs, counts=a_vals, denom=n, colors=a_cols,
         title=f"a  AI use in professional routine (n = {n})",
         xlabel="% of respondents", annot=f"Users: {n_users} (53.0%)"),
    dict(key="b", labels=b_order, counts=[int(b[k]) for k in b_order],
         denom=n_users, colors=BLUE[1],
         title=f"b  Areas of use (users, n = {n_users}; multiple response)",
         xlabel="% of users"),
    dict(key="c", labels=c_order, counts=[int(cformats[k]) for k in c_order],
         denom=n_tr, colors=BLUE[1],
         title=f"c  Formats of prior AI training (trained, n = {n_tr})",
         xlabel="% of trained physicians",
         annot=f"Ever trained: {n_tr}/{n} (17.1%)"),
    dict(key="d", labels=d_order, counts=[int(d[k]) for k in d_order],
         denom=n_nu, colors=GRAY[0],
         title=f"d  Barriers to use (non-users, n = {n_nu}; multiple response)",
         xlabel="% of non-users"),
]

# --- (1) Combined portrait figure: 4 panels stacked vertically ------------
fig, axes = plt.subplots(4, 1, figsize=(7.09, 10.5))
for ax, p in zip(axes, panels):
    hbar(ax, p["labels"], p["counts"], p["denom"], p["colors"],
         p["title"], p["xlabel"], p.get("annot"))
fig.tight_layout(h_pad=2.4)
fig.subplots_adjust(left=0.30)   # room for long y-axis labels on the left
fig.savefig("fig1.png", dpi=300, bbox_inches="tight")
fig.savefig("fig1.pdf", bbox_inches="tight")

# --- (2) Each panel additionally as a separate file (fig1a .. fig1d) ------
for p in panels:
    f, ax = plt.subplots(figsize=(7.09, 2.6))
    hbar(ax, p["labels"], p["counts"], p["denom"], p["colors"],
         p["title"], p["xlabel"], p.get("annot"))
    f.tight_layout()
    f.subplots_adjust(left=0.30)
    f.savefig(f"fig1{p['key']}.png", dpi=300, bbox_inches="tight")
    f.savefig(f"fig1{p['key']}.pdf", bbox_inches="tight")
    plt.close(f)

print("\nWritten: fig1.png/.pdf (portrait) "
      "+ fig1a..fig1d .png/.pdf (individual panels)")
