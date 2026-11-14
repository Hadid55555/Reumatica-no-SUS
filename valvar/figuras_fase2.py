#!/usr/bin/env python3
"""Figuras do artigo (PNG 300 dpi) a partir de valvar/resultados_fase2. Uso: python valvar/figuras_fase2.py"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(AQUI, "resultados_fase2")
OUT = os.path.join(AQUI, "figuras")
os.makedirs(OUT, exist_ok=True)

SURF, TXT, TXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
REUM, NREUM, TERC = "#2a78d6", "#eb6834", "#1baf7a"   # categóricas 1–3 (validadas)
RAMPA = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal azul (validada)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": TXT2,
                     "xtick.color": TXT2, "ytick.color": TXT2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})


def rotulo(ax, x, y, txt, cor, dy=0):
    ax.annotate(txt, (x, y), xytext=(4, dy), textcoords="offset points", va="center", fontsize=8, color=TXT)
    ax.plot([x], [y], "o", ms=4, color=cor)


def covid(ax):
    ax.set_xticks(range(2008, 2026, 2))
    ax.axvspan(2019.5, 2021.5, color=GRID, alpha=0.5, lw=0)
    ax.text(2020.5, ax.get_ylim()[1] * 0.97, "pandemia", ha="center", va="top", fontsize=7, color=TXT2)


def fig1():
    s = pd.read_csv(os.path.join(R, "01_serie_nacional.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col, tit in [(axs[0], "n", "A  Cirurgias valvares no SUS (n por ano)"),
                         (axs[1], "taxa_padronizada", "B  Taxa padronizada por idade (/100 mil hab.)")]:
        for d, cor, lab in [("Reumática", REUM, "Reumática (I05–I09)"), ("Não reumática", NREUM, "Não reumática (I34–I37)")]:
            x = s[s.definicao == d]
            ax.plot(x.ano, x[col], color=cor, lw=2, label=lab)
            rotulo(ax, 2025, x[col].iloc[-1], f"{x[col].iloc[-1]:,.0f}".replace(",", ".") if col == "n"
                   else f"{x[col].iloc[-1]:.2f}".replace(".", ","), cor, dy=(6 if d == "Não reumática" else -6))
        if col == "taxa_padronizada":
            x = s[s.definicao == "Reumática"]
            ax.fill_between(x.ano, x.ic95_inf, x.ic95_sup, color=REUM, alpha=0.15, lw=0)
        ax.set_ylim(0, None)
        ax.set_xlim(2007.5, 2027)
        ax.set_title(tit, loc="left", fontsize=9.5, color=TXT)
        covid(ax)
    axs[0].legend(loc="lower left", fontsize=8)
    fig.text(0.01, -0.02, "Inclui cirurgias múltiplas/sequenciais com ato valvar no SIH-SP. Padrão: população brasileira 2010.",
             fontsize=7, color=TXT2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_serie_reumatica_nao_reumatica.png"), dpi=300, bbox_inches="tight")


def fig2():
    fa = pd.read_csv(os.path.join(R, "05_febre_reumatica_aguda_5a19.csv"))
    s = pd.read_csv(os.path.join(R, "01_serie_nacional.csv"))
    s = s[s.definicao == "Reumática"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    f = fa[fa.ano >= 2008]
    i1 = 100 * f.taxa_100mil / f.taxa_100mil.iloc[0]
    i2 = 100 * s.n.to_numpy() / s.n.iloc[0]
    ax.plot(f.ano, i1, color=TERC, lw=2, label="Internações por febre reumática aguda, 5–19 anos (taxa)")
    ax.plot(s.ano, i2, color=REUM, lw=2, label="Cirurgias valvares reumáticas (n)")
    rotulo(ax, 2025, i1.iloc[-1], f"{i1.iloc[-1]:.0f}", TERC)
    rotulo(ax, 2025, i2[-1], f"{i2[-1]:.0f}", REUM)
    ax.axhline(100, color=TXT2, lw=0.8)
    ax.set_ylim(0, 130)
    ax.set_xlim(2007.5, 2026.5)
    ax.set_ylabel("Índice (2008 = 100)")
    ax.set_title("Febre reumática aguda cai; cirurgia da sequela não", loc="left", fontsize=9.5, color=TXT)
    covid(ax)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_febre_aguda_vs_cirurgia.png"), dpi=300, bbox_inches="tight")


def fig3():
    a = pd.read_csv(os.path.join(R, "03a_taxa_idade_periodo_reumatica.csv"))
    ordem = ["0 a 4 anos", "5 a 9 anos", "10 a 14 anos", "15 a 19 anos", "20 a 24 anos", "25 a 29 anos", "30 a 34 anos",
             "35 a 39 anos", "40 a 44 anos", "45 a 49 anos", "50 a 54 anos", "55 a 59 anos", "60 a 64 anos",
             "65 a 69 anos", "70 a 74 anos", "75 a 79 anos", "80 anos ou mais"]
    a["i"] = a.faixa.map({f: i for i, f in enumerate(ordem)})
    a = a[a.i >= 3].sort_values("i")
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for per, cor in zip(["2008-10", "2011-16", "2017-19", "2023-25"], RAMPA):
        x = a[a.periodo == per]
        ax.plot(x.i, x.taxa_100mil, color=cor, lw=2, marker="o", ms=3, label=per.replace("-", "–20"))
    ax.set_xticks(range(3, 17))
    ax.set_xticklabels([f.split(" ")[0] + ("+" if "mais" in f else "") for f in ordem[3:]], fontsize=7.5)
    ax.set_xlabel("Idade (início da faixa de 5 anos)")
    ax.set_ylabel("Cirurgias reumáticas /100 mil hab.")
    ax.set_title("Taxa por idade: queda nos jovens, estabilidade acima de 50 anos", loc="left", fontsize=9.5, color=TXT)
    ax.legend(title="Período", fontsize=8, title_fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_taxa_idade_periodo.png"), dpi=300, bbox_inches="tight")


def fig4():
    m = pd.read_csv(os.path.join(R, "07a_mitral_reumatica_estrategia.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axs[0]
    ax.plot(m.ano, m.pct_sem_protese_entre_cirurgias, color=REUM, lw=2, label="Toda cirurgia mitral reumática")
    ax.plot(m.ano, m.pct_plastica_valva_unica, color=NREUM, lw=2, label="Só valva única (código plástica × prótese)")
    rotulo(ax, 2025, m.pct_sem_protese_entre_cirurgias.iloc[-1], f"{m.pct_sem_protese_entre_cirurgias.iloc[-1]:.0f}%", REUM)
    rotulo(ax, 2025, m.pct_plastica_valva_unica.iloc[-1], f"{m.pct_plastica_valva_unica.iloc[-1]:.0f}%", NREUM)
    ax.set_ylim(0, 22)
    ax.set_xlim(2007.5, 2027)
    ax.set_title("A  Cirurgia sem prótese (plastia), % — CID I05", loc="left", fontsize=9.5, color=TXT)
    ax.legend(loc="lower left", fontsize=7.5)
    ax = axs[1]
    ax.plot(m.ano, m["Percutânea"], color=TERC, lw=2)
    rotulo(ax, 2025, m["Percutânea"].iloc[-1], f"{m['Percutânea'].iloc[-1]:.0f}", TERC)
    ax.set_ylim(0, None)
    ax.set_xlim(2007.5, 2027)
    ax.set_title("B  Valvuloplastia mitral percutânea (n) — CID I05", loc="left", fontsize=9.5, color=TXT)
    for a in axs:
        a.set_xticks(range(2008, 2026, 2))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_mitral_reumatica.png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print(sorted(os.listdir(OUT)))
