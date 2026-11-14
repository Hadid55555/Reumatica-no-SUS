#!/usr/bin/env python3
"""Figuras (PNG 300 dpi) e tabelas (markdown), em inglês, do manuscrito v2 (período 2003–2025, definição ampliada).
Lê só resultados congelados: resultados_fase6, fase8, fase9, sim_longo, fase3/5 (sensibilidades 2008–2025).
Saídas: manuscrito/figuras_v2/Figure1–4.png, FigureS1–S4.png, manuscrito/tabelas_v2.md, numeros_v2.json.
Uso: python valvar/manuscrito/figuras_tabelas_v2.py"""
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
V = os.path.dirname(AQUI)
FIG = os.path.join(AQUI, "figuras_v2")
os.makedirs(FIG, exist_ok=True)
R = {k: os.path.join(V, f"resultados_{k}") for k in ("fase6", "fase8", "fase9", "fase10", "sim_longo")}
SURF, TXT, TXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
REG_EN = {"Norte": "North", "Nordeste": "Northeast", "Centro-Oeste": "Centre-West", "Sudeste": "Southeast", "Sul": "South",
          "Brasil": "Brazil"}
REG_COR = {"North": S[3], "Northeast": S[1], "Centre-West": S[2], "Southeast": S[0], "South": S[4]}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": TXT2,
                     "xtick.color": TXT2, "ytick.color": TXT2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})
NUM = {}


def rd(nome, k):
    return pd.read_csv(os.path.join(R[k], nome + ".csv"))


def tit(ax, t):
    ax.set_title(t, loc="left", fontsize=9, color=TXT, fontweight="bold")


def pand(ax):
    ax.axvspan(2019.5, 2021.5, color=GRID, alpha=0.7, lw=0)


def ci(r, lo, hi, d=2):
    f = lambda v: f"{v:.{d}f}".replace(".", "·").replace("-", "−")  # noqa: E731
    sep = " to " if min(lo, hi) < 0 else "–"
    return f"{f(r)} ({f(lo)}{sep}{f(hi)})"


def n(x):
    return f"{int(round(x)):,}"


def p(x, d=1):
    return f"{x:.{d}f}".replace(".", "·")


# ------------------------------------------------------------------ figures
def figura1():
    s = rd("1_serie_2003_2025", "fase6")
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.7))
    linhas = [("Reumática ampliada", "Rheumatic (expanded)", S[0], "-"), ("Reumática estrita", "Rheumatic (strict, I05–I09)", S[0], ":"),
              ("Não reumática", "Non-rheumatic (I34–I37)", S[1], "-"), ("Aórtica não reumática (I35)", "Non-rheumatic aortic (I35)", S[1], ":")]
    for ax, col, t, yl in [(axs[0], "n", "A  Annual number of valve surgeries", "Operations per year"),
                           (axs[1], "taxa_padronizada", "B  Age-standardised rate", "Per 100,000 population")]:
        for d, lab, cor, ls in linhas:
            x = s[s.definicao == d]
            ax.plot(x.ano, x[col], color=cor, lw=2 if ls == "-" else 1.4, ls=ls, label=lab)
            if col != "n" and ls == "-":
                ax.fill_between(x.ano, x.ic95_inf, x.ic95_sup, color=cor, alpha=0.15, lw=0)
        pand(ax)
        ax.axvline(2007.5, color=TXT2, lw=0.7, ls="--")
        ax.set_ylim(0, None)
        ax.set_xticks(range(2003, 2026, 4))
        ax.set_ylabel(yl)
        tit(ax, t)
    axs[0].text(2007.7, axs[0].get_ylim()[1] * 0.97, "SIGTAP", fontsize=7, color=TXT2, va="top")
    axs[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "Figure1.png"), dpi=300, bbox_inches="tight")


def figura2():
    r = rd("3a_razoes_20_anos_2003_05_x_2023_25", "fase6")
    m = rd("3_razao_idade_mortalidade_2022_24_x_2003_05", "sim_longo")
    co = rd("D3_razao_taxas_coorte_modelo_idade_coorte", "fase9")
    fig, axs = plt.subplots(1, 3, figsize=(11.2, 3.6))
    ax = axs[0]
    fx = [f for f in r.faixa.unique() if "a" in f and "anos" in f]
    for d, lab, cor in [("Reumática ampliada", "Rheumatic", S[0]), ("Aórtica não reumática (I35)", "Non-rheumatic aortic (I35)", S[1])]:
        x = r[(r.definicao == d) & r.faixa.isin(fx)]
        idade = [int(f.split()[0]) + 2 for f in x.faixa]
        ax.plot(idade, x.razao, color=cor, lw=2, marker="o", ms=3, label=lab)
        ax.fill_between(idade, x.ic95_inf, x.ic95_sup, color=cor, alpha=0.15, lw=0)
    a4 = rd("A4_razoes_5anos_2017_19", "fase10")
    for d, cor in [("Reumática ampliada", S[0]), ("Aórtica não reumática (I35)", S[1])]:
        x = a4[a4.definicao == d]
        ax.plot([int(f.split()[0]) + 2 for f in x.faixa], x.razao, color=cor, lw=1.2, ls="--")
    ax.plot([], [], color=TXT2, lw=1.2, ls="--", label="End 2017–19 (pre-pandemic)")
    ax.axhline(1, color=TXT2, lw=0.8)
    ax.set_yscale("log"); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.25, 0.5, 1, 2])
    ax.set_yticklabels(["0·25", "0·5", "1", "2"])
    ax.set_xlabel("Age at surgery, years")
    ax.set_ylabel("Rate ratio vs 2003–05 (solid: 2023–25)")
    ax.legend(fontsize=7, loc="upper left")
    tit(ax, "A  Valve surgery (SIH/SUS)")
    ax = axs[1]
    x = m[~m.faixa.isin(["5-19", "20-39", "40-59", "60+"])]
    idade = [int(f.split("-")[0].rstrip("+")) + (5 if "-" in f and int(f.split("-")[1]) - int(f.split("-")[0]) > 5 else 2) for f in x.faixa]
    ax.plot(idade, x.razao, color=S[2], lw=2, marker="o", ms=3, label="Deaths, underlying cause I00–I09")
    ax.fill_between(idade, x.ic95_inf, x.ic95_sup, color=S[2], alpha=0.15, lw=0)
    ax.axhline(1, color=TXT2, lw=0.8)
    ax.set_yscale("log"); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.25, 0.5, 1, 2])
    ax.set_yticklabels(["0·25", "0·5", "1", "2"])
    ax.set_xlabel("Age at death, years")
    ax.set_ylabel("Rate ratio, 2022–24 vs 2003–05")
    ax.legend(fontsize=7, loc="upper left")
    tit(ax, "B  Rheumatic heart disease mortality (SIM)")
    ax = axs[2]
    co = co[co.coorte_nascimento.between(1930, 2000)]
    ax.plot(co.coorte_nascimento, co.rr_vs_1950, color=S[0], lw=2, marker="o", ms=3)
    ax.axhline(1, color=TXT2, lw=0.8)
    ax.set_yscale("log"); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.1, 0.25, 0.5, 1])
    ax.set_yticklabels(["0·1", "0·25", "0·5", "1"])
    ax.set_xlabel("Birth cohort")
    ax.set_ylabel("Rate ratio vs 1950 cohort")
    tit(ax, "C  Rheumatic surgery by birth cohort")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "Figure2.png"), dpi=300, bbox_inches="tight")


def figura3():
    s = rd("B1_serie_regional_2003_2025", "fase8")
    a = rd("A1_razoes_2023_25_x_2003_05_regiao_sexo", "fase8")
    tm = rd("C1_cirurgia_x_obito_razoes_idade_regiao", "fase8")
    fig, axs = plt.subplots(1, 3, figsize=(11.6, 3.7), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axs[0]
    for reg in ["North", "Northeast", "Centre-West", "Southeast", "South"]:
        x = s[s.regiao.map(REG_EN) == reg]
        ax.plot(x.ano, x.taxa_padronizada_15mais, color=REG_COR[reg], lw=1.8, label=reg)
    pand(ax)
    ax.set_ylim(0, None)
    ax.set_xticks(range(2003, 2026, 4))
    ax.set_ylabel("Per 100,000 (age ≥15 years)")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    tit(ax, "A  Age-standardised surgery rate")
    regs = ["North", "Northeast", "Centre-West", "Southeast", "South"]
    for ax, df, est, gr, t, yl in [
            (axs[1], a[a.definicao == "Reumática ampliada"], "estrato", ["15-39", "40-54", "55+"], "B  Surgery rate ratio by age", "2023–25 vs 2003–05"),
            (axs[2], tm[tm.medida == "obito"], "regiao", ["15-39", "40-59", "60+"], "C  Mortality rate ratio by age", "2022–24 vs 2003–05")]:
        for i, reg in enumerate(regs):
            for j, g in enumerate(gr):
                if est == "estrato":
                    row = df[(df.estrato == "Região " + {v: k for k, v in REG_EN.items()}[reg]) & (df.faixa == g)].iloc[0]
                else:
                    row = df[(df.regiao == {v: k for k, v in REG_EN.items()}[reg]) & (df.grupo == g)].iloc[0]
                xx = j + (i - 2) * 0.14
                ax.plot([xx, xx], [row.ic95_inf, row.ic95_sup], color=REG_COR[reg], lw=1.1)
                ax.plot(xx, row.razao, "o", color=REG_COR[reg], ms=4, label=reg if j == 0 else None)
        ax.axhline(1, color=TXT2, lw=0.8)
        ax.set_yscale("log"); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks([0.25, 0.5, 1, 2, 4])
        ax.set_yticklabels(["0·25", "0·5", "1", "2", "4"])
        ax.set_xticks(range(3))
        ax.set_xticklabels([g.replace("-", "–") + " y" for g in gr])
        ax.set_ylabel("Rate ratio, " + yl)
        ax.grid(axis="x", visible=False)
        tit(ax, t)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "Figure3.png"), dpi=300, bbox_inches="tight")


def figura4():
    d = rd("D_das_gupta_regiao_sexo", "fase8")
    d = d[d.comparacao.str.startswith("2003 × 2024")]
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axs[0]
    b = d[d.estrato == "Brasil"].iloc[0]
    passos = [("2003", b.n_inicio, None), ("Population\nsize", b.efeito_tamanho_pop, S[2]), ("Population\nageing", b.efeito_estrutura_etaria, S[3]),
              ("Age-specific\nrates", b.efeito_taxa_especifica, S[0]), ("2024", b.n_fim, None)]
    acum = 0
    for i, (lab, v, cor) in enumerate(passos):
        if cor is None:
            ax.bar(i, v, color=TXT2, width=0.6)
            acum = v
            ax.text(i, v + 120, n(v), ha="center", fontsize=7.5, color=TXT)
        else:
            ax.bar(i, v, bottom=acum, color=cor, width=0.6)
            ax.text(i, max(acum, acum + v) + 120, f"{'+' if v > 0 else '−'}{n(abs(v))}", ha="center", fontsize=7.5, color=TXT)
            acum += v
    ax.set_ylim(0, 11800)
    ax.set_xticks(range(5))
    ax.set_xticklabels([q[0] for q in passos], fontsize=7.5)
    ax.set_ylabel("Rheumatic valve surgeries per year")
    ax.grid(axis="x", visible=False)
    tit(ax, "A  Brazil, 2003 to 2024")
    ax = axs[1]
    est = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul", "Homens", "Mulheres"]
    lab = ["North", "Northeast", "Centre-West", "Southeast", "South", "Men", "Women"]
    x = d.set_index("estrato").loc[est]
    w = 0.26
    for k, (col, l, cor) in enumerate([("efeito_tamanho_pop", "Population size", S[2]), ("efeito_estrutura_etaria", "Population ageing", S[3]),
                                       ("efeito_taxa_especifica", "Age-specific rates", S[0])]):
        ax.bar(np.arange(len(est)) + (k - 1) * w, x[col], width=w, color=cor, label=l)
    ax.plot(np.arange(len(est)), x.diferenca, "D", color=TXT, ms=4, label="Net change")
    ax.axhline(0, color=TXT2, lw=0.8)
    ax.axvline(4.5, color=GRID, lw=1)
    ax.set_xticks(range(len(est)))
    ax.set_xticklabels(lab, fontsize=7.5)
    ax.set_ylabel("Change in annual surgeries")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="lower left")
    tit(ax, "B  By region of residence and sex")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "Figure4.png"), dpi=300, bbox_inches="tight")


def suplementares():
    c = rd("A1_composicao_anual_por_cid", "fase9")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    base = np.zeros(len(c))
    for col, lab, cor in [("I05", "I05 mitral", S[0]), ("I06", "I06 aortic", S[2]), ("I07", "I07 tricuspid", S[3]),
                          ("I08", "I08 multiple", S[4]), ("I09", "I09 other", TXT2), ("Q22.5", "Q22.5 (adults)", S[1]), ("I38", "I38", "#9c4a1d")]:
        ax.bar(c.ano, c[col], bottom=base, color=cor, width=0.8, label=lab)
        base += c[col].to_numpy()
    ax.set_ylabel("Operations per year")
    ax.legend(fontsize=6.5, ncol=4, loc="lower left")
    ax.grid(axis="x", visible=False)
    tit(ax, "Principal diagnosis of rheumatic (expanded) valve surgery")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FigureS1.png"), dpi=300, bbox_inches="tight")
    o = rd("B2_observado_esperado_2020_2025", "fase9")
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for g, cor in zip(["15-39", "40-54", "55+"], [S[0], S[2], S[1]]):
        x = o[o.faixa == g]
        ax.plot(x.ano, x.obs_div_esp, color=cor, lw=2, marker="o", ms=3, label=g.replace("-", "–") + " years")
    ax.axhline(1, color=TXT2, lw=0.8)
    ax.set_ylabel("Observed / expected (2010–19 trend)")
    ax.legend(fontsize=7)
    tit(ax, "Rheumatic surgery after the COVID-19 pandemic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FigureS2.png"), dpi=300, bbox_inches="tight")
    u = rd("C1_uf_taxas_razoes_renda", "fase9")
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for _, r in u.iterrows():
        cor = REG_COR[REG_EN[r.regiao]]
        ax.plot(r.renda_pc_2012, r.razao_taxas, "o", color=cor, ms=4 + 4 * np.sqrt(r.n_2022_24 / u.n_2022_24.max()))
        ax.annotate(r.uf, (r.renda_pc_2012, r.razao_taxas), xytext=(3, 2), textcoords="offset points", fontsize=6, color=TXT2)
    ax.axhline(1, color=TXT2, lw=0.8)
    ax.set_yscale("log"); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Household income per capita, 2012 (BRL/month)")
    ax.set_ylabel("Rate ratio, 2022–24 vs 2003–05")
    tit(ax, "Change in age-standardised rate by state")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "FigureS3.png"), dpi=300, bbox_inches="tight")


# ------------------------------------------------------------------ tables
def tabelas():
    out = []
    t = rd("E_tabela1_periodos", "fase9").set_index("periodo")
    cols = ["2003–05", "2013–15", "2023–25"]
    lin = [("Operations, n (per year)", lambda r: f"{n(r.n)} ({n(r.n_por_ano)})"),
           ("Age, years, median (IQR)", lambda r: f"{r.idade_mediana:.0f} ({r.idade_p25:.0f}–{r.idade_p75:.0f})"),
           ("Women, %", lambda r: p(r.pct_mulheres)), ("Aged <15 years, %", lambda r: p(r.pct_menor15)),
           ("Aged 15–39 years, %", lambda r: p(r.pct_15_39)), ("Aged ≥55 years, %", lambda r: p(r.pct_55mais)),
           ("Born before 1975, %", lambda r: p(r.pct_nascidos_antes_1975))]
    lin += [(f"Region of residence: {REG_EN[g]}, %", (lambda g: lambda r: p(r[f"pct_{g}"]))(g)) for g in ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]]
    lin += [("Principal diagnosis I05 (mitral), %", lambda r: p(r.pct_I05)), ("I06 (aortic), %", lambda r: p(r.pct_I06)),
            ("I07 (tricuspid), %", lambda r: p(r.pct_I07)), ("I08 (multiple valves), %", lambda r: p(r.pct_I08)),
            ("Q22.5 or I38 (adults), %", lambda r: p(r.pct_Q225_I38)), ("In-hospital death, %", lambda r: p(r.mort_hosp_pct))]
    out.append("### Table 1\n\n| Characteristic | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for nome, f in lin:
        out.append(f"| {nome} | " + " | ".join(f(t.loc[c]) for c in cols) + " |")
    c1 = rd("C1_perfil_cirurgico_periodo_idade_regiao", "fase10")
    c1 = c1[c1.estrato == "Total"].set_index("periodo")
    for nome, col in [("Multivalve surgery, %†", "pct_multivalvar"), ("Valve surgery with CABG, %†", "pct_valvar_mais_rvm"),
                      ("Annuloplasty ring billed, %†", "pct_anel"), ("Multiple-surgery record, %†", "pct_registro_multiplo")]:
        out.append(f"| {nome} | {p(c1.loc['2008–10', col])}* | {p(c1.loc['2013–15', col])} | {p(c1.loc['2023–25', col])} |")
    sp = t.loc["2008–10 (SIH-SP)"]
    out.append(f"| Mechanical among replacements, %† | {p(sp.pct_mecanica_entre_trocas)}* | {p(t.loc['2013–15'].pct_mecanica_entre_trocas)} | {p(t.loc['2023–25'].pct_mecanica_entre_trocas)} |")
    out.append(f"| I05 operations without prosthesis, %† | {p(sp.pct_sem_protese_mitral_I05)}* | {p(t.loc['2013–15'].pct_sem_protese_mitral_I05)} | {p(t.loc['2023–25'].pct_sem_protese_mitral_I05)} |")
    # Table 2
    r = rd("3a_razoes_20_anos_2003_05_x_2023_25", "fase6").set_index(["definicao", "faixa"])
    rr = rd("3b_reum_vs_I35_20_anos", "fase6").set_index("faixa")
    es = rd("A2_razoes_estrita_x_ampliada", "fase9").set_index(["definicao", "faixa"])
    pp = rd("B3_razao_geracional_pre_pandemia_2017_19", "fase9").set_index("faixa")
    a8 = rd("A1_razoes_2023_25_x_2003_05_regiao_sexo", "fase8")
    a8 = a8[a8.definicao == "Reumática ampliada"].set_index(["estrato", "faixa"])
    tm = rd("C1_cirurgia_x_obito_razoes_idade_regiao", "fase8").set_index(["regiao", "medida", "grupo"])
    g3 = ["15-39", "40-54", "55+"]
    a1 = rd("A1_razoes_pre_pandemia_x_recente", "fase10").set_index(["fim", "definicao", "faixa"])
    tm = rd("C1_cirurgia_x_obito_razoes_idade_regiao", "fase8").set_index(["regiao", "medida", "grupo"])
    t17 = rd("A2_cirurgia_obito_2017_19_x_2003_05", "fase10").set_index(["regiao", "medida", "grupo"])
    a8 = rd("A1_razoes_2023_25_x_2003_05_regiao_sexo", "fase8")
    a8 = a8[a8.definicao == "Reumática ampliada"].set_index(["estrato", "faixa"])
    es = rd("A2_razoes_estrita_x_ampliada", "fase9").set_index(["definicao", "faixa"])
    g3 = ["15-39", "40-54", "55+"]
    out.append("\n### Table 2\n\n| Comparison | 15–39 years | 40–54 years | ≥55 years |\n|---|---|---|---|")
    def row(lbl, get, gs=g3):
        out.append(f"| {lbl} | " + " | ".join(get(g) for g in gs) + " |")
    def a1r(fim, d):
        return lambda g: ci(*a1.loc[(fim, d, g), ["razao", "ic95_inf", "ic95_sup"]])
    for fim, cab in [("2023–25", "**2023–2025 vs 2003–2005 (most recent)**"), ("2017–19", "**2017–2019 vs 2003–2005 (pre-pandemic)**")]:
        out.append(f"| {cab} | | | |")
        row("Rheumatic, expanded", a1r(fim, "Reumática ampliada"))
        if fim == "2023–25":
            row("Rheumatic, strict (I05–I09)", lambda g: ci(*es.loc[("Estrita (I05–I09)", g), ["razao", "ic95_inf", "ic95_sup"]]))
        row("Non-rheumatic aortic (I35)", a1r(fim, "Aórtica não reumática (I35)"))
        row("All non-rheumatic (I34–I37)", a1r(fim, "Não reumática (I34–I37)"))
        row("Ratio of rate ratios, rheumatic/I35", a1r(fim, "Razão das razões reum ÷ Aórtica não reumática (I35)"))
        if fim == "2023–25":
            row("Rheumatic, men", lambda g: ci(*a8.loc[("Homens", g), ["razao", "ic95_inf", "ic95_sup"]]))
            row("Rheumatic, women", lambda g: ci(*a8.loc[("Mulheres", g), ["razao", "ic95_inf", "ic95_sup"]]))
    out.append("| **Surgery and mortality, age groups of the mortality tabulation‡** | **15–39 years** | **40–59 years** | **≥60 years** |")
    g6 = ["15-39", "40-59", "60+"]
    row("Rheumatic surgery, 2022–24 vs 2003–05", lambda g: ci(*tm.loc[("Brasil", "cirurgia", g), ["razao", "ic95_inf", "ic95_sup"]]), g6)
    row("Rheumatic heart disease deaths, 2022–24 vs 2003–05", lambda g: ci(*tm.loc[("Brasil", "obito", g), ["razao", "ic95_inf", "ic95_sup"]]), g6)
    row("Rheumatic surgery, 2017–19 vs 2003–05", lambda g: ci(*t17.loc[("Brasil", "cirurgia", g), ["razao", "ic95_inf", "ic95_sup"]]), g6)
    row("Rheumatic heart disease deaths, 2017–19 vs 2003–05", lambda g: ci(*t17.loc[("Brasil", "obito", g), ["razao", "ic95_inf", "ic95_sup"]]), g6)
    # Table 3
    j = rd("2a_joinpoint_2003_2024_segmentos", "fase6")
    aa = rd("2b_joinpoint_2003_2024_AAPC", "fase6").set_index(["definicao", "faixa"])
    out.append("\n### Table 3\n\n| Series | Segment | APC, % (95% CI) | AAPC 2003–2024, % (95% CI) |\n|---|---|---|---|")
    for (dfn, fx), g in j.groupby(["definicao", "faixa"], sort=False):
        lbl = ("Rheumatic" if dfn.startswith("Reum") else "Non-rheumatic aortic (I35)") + f", age {fx.replace('-', '–').replace('≥15', '≥15')}"
        for k, (_, sg) in enumerate(g.iterrows()):
            a_ = aa.loc[(dfn, fx)]
            out.append(f"| {lbl if k == 0 else ''} | {int(sg.inicio)}–{int(sg.fim)} | {ci(sg.APC, sg.ic95_inf, sg.ic95_sup, 1)} | "
                       f"{ci(a_.AAPC_2003_2024, a_.ic95_inf, a_.ic95_sup, 1) if k == 0 else ''} |")
    d = rd("D_das_gupta_regiao_sexo", "fase8")
    d = d[d.comparacao.str.startswith("2003 × 2024")]
    out.append("\n| Das Gupta decomposition, 2003 to 2024 | 2003 | 2024 | Net change | Population size | Population ageing | Age-specific rates |\n|---|---|---|---|---|---|---|")
    lab = {"Brasil": "Brazil", "Homens": "Men", "Mulheres": "Women", **REG_EN}
    for _, x in d.iterrows():
        sg = lambda v: f"{v:+,.0f}".replace("-", "−")  # noqa: E731
        out.append(f"| {lab[x.estrato]} | {n(x.n_inicio)} | {n(x.n_fim)} | {sg(x.diferenca)} | {sg(x.efeito_tamanho_pop)} | "
                   f"{sg(x.efeito_estrutura_etaria)} | {sg(x.efeito_taxa_especifica)} |")
    open(os.path.join(AQUI, "tabelas_v2.md"), "w").write("\n".join(out) + "\n")


def main():
    figura1()
    figura2()
    figura3()
    figura4()
    suplementares()
    tabelas()
    print("ok")


if __name__ == "__main__":
    main()
