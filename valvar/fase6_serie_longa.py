#!/usr/bin/env python3
"""Fase 6 — série longa 2003–2025 (definição ampliada), juntando a extração da Fase 1 (2003–2007, tabelas SIH 32/48)
à base da Fase 2 (2008–2025, SIGTAP + cirurgias múltiplas recuperadas no SIH-SP).

Por que começa em 2003: até 2002 ~25–37% das cirurgias valvares tinham CID genérico (I38, Z95, I50…); a regra de
compatibilidade CID × procedimento entrou ao longo de 2002 (ano de transição). De 2004 a 2012 o Q22.5 foi usado como
coringa → a definição ampliada (I05–I09 + Q22.5/I38 em adultos) é a principal nesta série. Cirurgias múltiplas antes
de 2008 (código 31000002; 33–60/ano com CID valvar) ficam de fora (sem SIH-SP para confirmar o ato valvar).

Saídas: valvar/resultados_fase6/*.csv, valvar/fase6_serie_longa.xlsx, valvar/figuras/fig8_serie_longa_2003_2025.png.
Uso: python valvar/fase6_serie_longa.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import fase2_valvar as F  # noqa: E402
import fase4_revisao2 as J  # noqa: E402
from fase1_valvar import ABERTA, carregar as carregar_fase1  # noqa: E402

OUT = os.path.join(AQUI, "resultados_fase6")
os.makedirs(OUT, exist_ok=True)
T, RES = {}, {}
GRUPOS = {"15-39": range(3, 8), "40-54": range(8, 11), "55+": range(11, F.NF)}


def salvar(df, nome):
    T[nome] = df
    df.to_csv(os.path.join(OUT, nome + ".csv"), index=False)


def montar():
    if os.path.exists(os.path.join(os.path.dirname(AQUI), "saida_valvar", "parciais")):
        d, _ = carregar_fase1()
        a = d[d.tipo.isin(ABERTA) & d.ano.between(2003, 2007)].copy()
        a["sexo"] = np.where(a.SEXO.eq("1"), "M", np.where(a.SEXO.isin(["2", "3"]), "F", ""))
        a["etiologia"] = np.select([a.grupo.eq("Reumática"), a.grupo.eq("Não reumática")], ["Reumática", "Não reumática"], "Outra")
        a["reum_ampliada"] = a.grupo.eq("Reumática") | ((a.DIAG_PRINC.eq("Q225") | a.cid3.eq("I38")) & (a.idade >= 18))
    else:  # sem os parciais locais (nova sessão): recorte versionado
        a = pd.read_csv(os.path.join(AQUI, "base_valvar_2003_2007.csv.gz"), dtype={"N_AIH": str, "CNES": str, "DIAG_PRINC": str, "PROC_REA": str})
    a["fonte"] = "2003–2007 (tabelas 32/48)"
    p = os.path.join(os.path.dirname(AQUI), "saida_valvar", "base_valvar_2008_2025.csv.gz")
    if not os.path.exists(p):
        p = os.path.join(AQUI, "base_valvar_2008_2025.csv.gz")
    b = pd.read_csv(p, dtype={"N_AIH": str, "CNES": str, "DIAG_PRINC": str, "PROC_REA": str})
    b = b[b.cirurgia].copy()
    b["fonte"] = "2008–2025 (SIGTAP + SP)"
    cols = ["N_AIH", "ano", "idade", "sexo", "regiao_res", "etiologia", "reum_ampliada", "DIAG_PRINC", "fonte"]
    c = pd.concat([a[cols], b[cols]], ignore_index=True)
    c["fx"] = F.faixa5(c.idade).to_numpy()
    c["reum"] = c.etiologia.eq("Reumática")
    c["i35"] = c.DIAG_PRINC.str[:3].eq("I35")
    return c


def serie(c, pop):
    pa = pop.groupby(["ano", "faixa"])["pop"].sum()
    padrao = pa.loc[2010]
    lin = []
    for nome, m in [("Reumática ampliada", c.reum_ampliada), ("Reumática estrita", c.reum),
                    ("Não reumática", c.etiologia.eq("Não reumática")), ("Aórtica não reumática (I35)", c.i35)]:
        x = c[m & (c.fx >= 0)]
        k = x.groupby(["ano", "fx"]).size()
        for a in range(2003, 2026):
            t, v = J.padronizada(k.get(a, pd.Series(dtype=float)), pa.loc[a], padrao, list(range(F.NF)))
            g = x[x.ano == a]
            lin.append({"definicao": nome, "ano": a, "n": len(g), "taxa_padronizada": t,
                        "ic95_inf": t - 1.96 * v ** .5, "ic95_sup": t + 1.96 * v ** .5, "var": v,
                        "idade_mediana": g.idade.median(), "pct_15_39": 100 * g.idade.between(15, 39.999).mean(),
                        "pct_60mais": 100 * (g.idade >= 60).mean(),
                        "pct_nascidos_antes_1975": 100 * ((g.ano - g.idade) < 1975).mean()})
    s = pd.DataFrame(lin)
    salvar(s.drop(columns="var"), "1_serie_2003_2025")
    return s


def joinpoints(c, pop, s):
    pa = pop.groupby(["ano", "faixa"])["pop"].sum()
    padrao = pa.loc[2010]
    lin, aap = [], []
    series = [("Reumática ampliada", "≥15", c.reum_ampliada, list(range(3, F.NF))),
              ("Reumática ampliada", "15-39", c.reum_ampliada, list(range(3, 8))),
              ("Reumática ampliada", "40+", c.reum_ampliada, list(range(8, F.NF))),
              ("Aórtica não reumática (I35)", "≥15", c.i35, list(range(3, F.NF)))]
    for nome, fxn, m, fxs in series:
        k = c[m & (c.fx >= 0)].groupby(["ano", "fx"]).size()
        rows = []
        for a in range(2003, 2025):
            if a in (2020, 2021):
                continue
            t, v = J.padronizada(k.get(a, pd.Series(dtype=float)), pa.loc[a], padrao, fxs)
            rows.append((a, t, v))
        an, tx, vr = zip(*rows)
        jps, segs, aa = J.joinpoint(an, tx, vr, max_jp=3)
        for sg in segs:
            lin.append({"definicao": nome, "faixa": fxn, "joinpoints": ",".join(str(int(j)) for j in jps), **sg})
        aap.append({"definicao": nome, "faixa": fxn, "AAPC_2003_2024": aa[0], "ic95_inf": aa[1], "ic95_sup": aa[2]})
    salvar(pd.DataFrame(lin), "2a_joinpoint_2003_2024_segmentos")
    salvar(pd.DataFrame(aap), "2b_joinpoint_2003_2024_AAPC")


def razoes(c, pop):
    pa = pop.groupby(["ano", "faixa"])["pop"].sum()
    A1, A2 = [2003, 2004, 2005], [2023, 2024, 2025]
    lin = []
    for nome, m in [("Reumática ampliada", c.reum_ampliada), ("Aórtica não reumática (I35)", c.i35),
                    ("Não reumática", c.etiologia.eq("Não reumática"))]:
        x = c[m & (c.fx >= 0)]
        k1, k2 = x[x.ano.isin(A1)].groupby("fx").size(), x[x.ano.isin(A2)].groupby("fx").size()
        for fx_nome, fxs in [(F.FAIXAS[f], [f]) for f in range(3, F.NF)] + list(GRUPOS.items()):
            n1, n2 = sum(k1.get(f, 0) for f in fxs), sum(k2.get(f, 0) for f in fxs)
            p1 = sum(pa.loc[(a, f)] for a in A1 for f in fxs)
            p2 = sum(pa.loc[(a, f)] for a in A2 for f in fxs)
            rr = (n2 / p2) / (n1 / p1)
            se = np.sqrt(1 / n1 + 1 / n2)
            lin.append({"definicao": nome, "faixa": fx_nome, "n_2003_05": n1, "n_2023_25": n2, "razao": rr,
                        "ic95_inf": rr * np.exp(-1.96 * se), "ic95_sup": rr * np.exp(1.96 * se)})
    t = pd.DataFrame(lin)
    a = t[t.definicao == "Reumática ampliada"].set_index("faixa")
    b = t[t.definicao == "Aórtica não reumática (I35)"].set_index("faixa")
    rr = []
    for f in a.index:
        se = np.sqrt(1 / a.loc[f, "n_2003_05"] + 1 / a.loc[f, "n_2023_25"] + 1 / b.loc[f, "n_2003_05"] + 1 / b.loc[f, "n_2023_25"])
        r = a.loc[f, "razao"] / b.loc[f, "razao"]
        rr.append({"faixa": f, "razao_reum": a.loc[f, "razao"], "razao_I35": b.loc[f, "razao"], "razao_das_razoes": r,
                   "ic95_inf": r * np.exp(-1.96 * se), "ic95_sup": r * np.exp(1.96 * se)})
    salvar(t, "3a_razoes_20_anos_2003_05_x_2023_25")
    salvar(pd.DataFrame(rr), "3b_reum_vs_I35_20_anos")


def decomposicao(c, pop):
    pa = pop.groupby(["ano", "faixa"])["pop"].sum()
    k = c[c.reum_ampliada & (c.fx >= 0)].groupby(["ano", "fx"]).size()
    f = {}
    for a in (2003, 2024):
        p = pa.loc[a].reindex(range(F.NF)).to_numpy()
        n = np.array([k.get((a, i), 0) for i in range(F.NF)])
        f[a] = (p.sum(), p / p.sum(), n / p)
    (P1, s1, r1), (P2, s2, r2) = f[2003], f[2024]
    N = lambda P, s, r: P * (s * r).sum()  # noqa: E731
    efP = ((N(P2, s1, r1) - N(P1, s1, r1)) + (N(P2, s2, r2) - N(P1, s2, r2))) / 3 + ((N(P2, s1, r2) - N(P1, s1, r2)) + (N(P2, s2, r1) - N(P1, s2, r1))) / 6
    efS = ((N(P1, s2, r1) - N(P1, s1, r1)) + (N(P2, s2, r2) - N(P2, s1, r2))) / 3 + ((N(P1, s2, r2) - N(P1, s1, r2)) + (N(P2, s2, r1) - N(P2, s1, r1))) / 6
    efR = ((N(P1, s1, r2) - N(P1, s1, r1)) + (N(P2, s2, r2) - N(P2, s2, r1))) / 3 + ((N(P1, s2, r2) - N(P1, s2, r1)) + (N(P2, s1, r2) - N(P2, s1, r1))) / 6
    salvar(pd.DataFrame([{"anos": "2003–2024", "definicao": "Reumática ampliada", "n_inicio": N(P1, s1, r1),
                          "n_fim": N(P2, s2, r2), "diferenca": N(P2, s2, r2) - N(P1, s1, r1), "efeito_tamanho_pop": efP,
                          "efeito_estrutura_etaria": efS, "efeito_taxa_especifica": efR,
                          "conferencia": efP + efS + efR - (N(P2, s2, r2) - N(P1, s1, r1))}]), "4_das_gupta_2003_2024")


def figura(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, col, tit in [(axs[0], "n", "A  Cirurgias por ano (SUS)"), (axs[1], "taxa_padronizada", "B  Taxa padronizada por idade /100 mil")]:
        for d, cor, ls in [("Reumática ampliada", "#2a78d6", "-"), ("Reumática estrita", "#2a78d6", ":"), ("Não reumática", "#eb6834", "-")]:
            x = s[s.definicao == d]
            ax.plot(x.ano, x[col], color=cor, lw=2, ls=ls, label=d)
        ax.axvline(2007.5, color="#52514e", lw=0.8, ls="--")
        ax.text(2007.6, ax.get_ylim()[1] * 0.98 if col == "n" else 4.6, "SIGTAP", fontsize=7, color="#52514e", va="top")
        ax.axvspan(2019.5, 2021.5, color="#e4e3df", alpha=0.6, lw=0)
        ax.set_ylim(0, None)
        ax.set_xticks(range(2003, 2026, 2))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(tit, loc="left", fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(color="#e4e3df", lw=0.6)
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(AQUI, "figuras", "fig8_serie_longa_2003_2025.png"), dpi=300, bbox_inches="tight")


def main():
    pop = F.populacao()
    c = montar()
    RES["n_2003_2007_ampliada"] = int(c[c.reum_ampliada & c.ano.between(2003, 2007)].shape[0])
    RES["n_2003_2025_ampliada"] = int(c.reum_ampliada.sum())
    s = serie(c, pop)
    joinpoints(c, pop, s)
    razoes(c, pop)
    decomposicao(c, pop)
    figura(s)
    with pd.ExcelWriter(os.path.join(AQUI, "fase6_serie_longa.xlsx")) as w:
        for k, v in T.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    json.dump(RES, open(os.path.join(OUT, "resumo.json"), "w"), indent=1)
    print(json.dumps(RES, indent=1))


if __name__ == "__main__":
    main()
