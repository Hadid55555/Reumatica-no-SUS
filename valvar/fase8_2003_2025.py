#!/usr/bin/env python3
"""Fase 8 — análises do período principal 2003–2025 (definição ampliada), para o manuscrito.

A. Regiões e sexo: razão de taxas por idade 2023–25 ÷ 2003–05 (15–39, 40–54, 55+) por região de residência e por sexo,
   com controle I35; testes de interação período × idade × região (e × sexo), Poisson com escala de Pearson.
B. Séries anuais 2003–2025 por região: taxa padronizada por idade (padrão Brasil 2010), idade mediana, % 15–39;
   variação anual média 2003–2024 (Poisson, sem 2020–2021).
C. Triangulação cirurgia × mortalidade (SIM/TabNet, causa básica I00–I09): razões 2022–24 ÷ 2003–05 por idade e região
   (reaproveita fase7.triangulacao), mais séries anuais padronizadas 2003–2024 por região e tendência nas duas medidas.
D. Decomposição de Das Gupta 2003 × 2024 (tamanho × estrutura etária × taxa específica): Brasil, regiões, sexo;
   sensibilidade com triênios (2003–05 × 2022–24, médias anuais).
E. Conferências independentes: soma das regiões + ignorada = Brasil; recontagem direta dos totais da série;
   Das Gupta fecha (soma dos efeitos = diferença observada).

Só dados versionados: base_valvar_2003_2007.csv.gz, base_valvar_2008_2025.csv.gz, sim/tabnet_*.csv, artigo/dados_ref.
Saídas: valvar/resultados_fase8/*.csv, resumo.json, valvar/fase8_2003_2025.xlsx. Uso: python valvar/fase8_2003_2025.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import fase2_valvar as F  # noqa: E402
import fase4_revisao2 as J  # noqa: E402
import fase6_serie_longa as L  # noqa: E402
import fase7_extras as P  # noqa: E402

OUT = os.path.join(AQUI, "resultados_fase8")
os.makedirs(OUT, exist_ok=True)
T, RES = {}, {}
REGS5 = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
REGS = ["Brasil"] + REGS5
GRUPOS = {"15-39": list(range(3, 8)), "40-54": list(range(8, 11)), "55+": list(range(11, F.NF))}
FX = list(range(3, F.NF))  # ≥15 anos
A1, A2 = [2003, 2004, 2005], [2023, 2024, 2025]


def salvar(df, nome):
    T[nome] = df
    df.to_csv(os.path.join(OUT, nome + ".csv"), index=False)


def razao(n1, p1, n2, p2):
    rr = (n2 / p2) / (n1 / p1)
    se = np.sqrt(1 / n1 + 1 / n2)
    return rr, rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se)


def denominador(pop, reg=None, sexo=None):
    p = pop
    if reg and reg != "Brasil":
        p = p[p.regiao == reg]
    if sexo:
        p = p[p.sexo == sexo]
    return p.groupby(["ano", "faixa"])["pop"].sum()


# ------------------------------------------------------------------ A
def regiao_sexo(c, pop):
    lin = []
    estratos = [("Brasil", None, None)] + [(f"Região {r}", r, None) for r in REGS5] + [("Homens", None, "M"), ("Mulheres", None, "F")]
    for defn, m in [("Reumática ampliada", c.reum_ampliada), ("Aórtica não reumática (I35)", c.i35)]:
        for rot, reg, sx in estratos:
            x = c[m & (c.fx >= 0)]
            if reg:
                x = x[x.regiao_res == reg]
            if sx:
                x = x[x.sexo == sx]
            pa = denominador(pop, reg, sx)
            k1, k2 = x[x.ano.isin(A1)].groupby("fx").size(), x[x.ano.isin(A2)].groupby("fx").size()
            for g, fxs in GRUPOS.items():
                n1, n2 = sum(k1.get(f, 0) for f in fxs), sum(k2.get(f, 0) for f in fxs)
                p1 = sum(pa.loc[(a, f)] for a in A1 for f in fxs)
                p2 = sum(pa.loc[(a, f)] for a in A2 for f in fxs)
                rr, lo, hi = razao(n1, p1, n2, p2)
                lin.append({"definicao": defn, "estrato": rot, "faixa": g, "n_2003_05": n1, "n_2023_25": n2,
                            "taxa_2003_05": 1e5 * n1 / p1, "taxa_2023_25": 1e5 * n2 / p2,
                            "razao": rr, "ic95_inf": lo, "ic95_sup": hi})
    t = pd.DataFrame(lin)
    # razão das razões (reumática ÷ I35) por estrato e faixa
    a = t[t.definicao == "Reumática ampliada"].set_index(["estrato", "faixa"])
    b = t[t.definicao == "Aórtica não reumática (I35)"].set_index(["estrato", "faixa"])
    rr = []
    for ix in a.index:
        se = np.sqrt(sum(1 / v for v in (a.loc[ix, "n_2003_05"], a.loc[ix, "n_2023_25"], b.loc[ix, "n_2003_05"], b.loc[ix, "n_2023_25"])))
        r = a.loc[ix, "razao"] / b.loc[ix, "razao"]
        rr.append({"estrato": ix[0], "faixa": ix[1], "razao_reum": a.loc[ix, "razao"], "razao_I35": b.loc[ix, "razao"],
                   "razao_das_razoes": r, "ic95_inf": r * np.exp(-1.96 * se), "ic95_sup": r * np.exp(1.96 * se)})
    salvar(t, "A1_razoes_2023_25_x_2003_05_regiao_sexo")
    salvar(pd.DataFrame(rr), "A2_razao_das_razoes_reum_x_I35")
    # gradiente jovens ÷ 55+ em cada estrato
    g = []
    for rot in a.index.get_level_values(0).unique():
        y, o = a.loc[(rot, "15-39")], a.loc[(rot, "55+")]
        r = y.razao / o.razao
        se = np.sqrt(1 / y.n_2003_05 + 1 / y.n_2023_25 + 1 / o.n_2003_05 + 1 / o.n_2023_25)
        g.append({"estrato": rot, "gradiente_15_39_vs_55mais": r, "ic95_inf": r * np.exp(-1.96 * se), "ic95_sup": r * np.exp(1.96 * se)})
    salvar(pd.DataFrame(g), "A3_gradiente_jovens_vs_55mais")
    # interação período × grupo etário × região (e × sexo)
    cir = c[c.reum_ampliada & (c.fx >= 0)]
    for dim, col, cats in [("regiao", "regiao_res", REGS5), ("sexo", "sexo", ["M", "F"])]:
        pp = pop.groupby(["ano", "faixa", dim])["pop"].sum()
        k = cir.groupby(["ano", "fx", col]).size()
        rows = [{"cat": cat, "per": per, "fx": f, "n": sum(k.get((a_, f, cat), 0) for a_ in anos),
                 "pop": sum(pp.loc[(a_, f, cat)] for a_ in anos)}
                for cat in cats for per, anos in [("ini", A1), ("fim", A2)] for f in FX]
        d = pd.DataFrame(rows)
        d["grupo"] = pd.cut(d.fx, [2, 7, 10, 20], labels=list(GRUPOS)).astype(str)
        m0 = smf.glm("n ~ C(fx)*cat + per*cat + per*C(grupo)", data=d, family=sm.families.Poisson(), offset=np.log(d["pop"])).fit(scale="X2")
        m1 = smf.glm("n ~ C(fx)*cat + per*cat*C(grupo)", data=d, family=sm.families.Poisson(), offset=np.log(d["pop"])).fit(scale="X2")
        dfn = m0.df_resid - m1.df_resid
        fv = (m0.deviance - m1.deviance) / dfn / m1.scale
        RES[f"A_interacao_{dim}_x_periodo_x_idade"] = {"F": round(fv, 2), "gl": [int(dfn), int(m1.df_resid)],
                                                       "p": round(float(stats.f.sf(fv, dfn, m1.df_resid)), 3)}


# ------------------------------------------------------------------ B
def series_regionais(c, pop):
    padrao = pop.groupby(["ano", "faixa"])["pop"].sum().loc[2010]
    lin, tend = [], []
    x0 = c[c.reum_ampliada & (c.fx >= 0)]
    for reg in REGS:
        x = x0 if reg == "Brasil" else x0[x0.regiao_res == reg]
        pa = denominador(pop, reg)
        k = x.groupby(["ano", "fx"]).size()
        for a in range(2003, 2026):
            t, v = J.padronizada(k.get(a, pd.Series(dtype=float)), pa.loc[a], padrao, FX)
            h = x[x.ano == a]
            lin.append({"regiao": reg, "ano": a, "n": len(h), "taxa_padronizada_15mais": t,
                        "ic95_inf": t - 1.96 * v ** .5, "ic95_sup": t + 1.96 * v ** .5, "idade_mediana": h.idade.median(),
                        "pct_15_39": 100 * h.idade.between(15, 39.999).mean(), "pct_55mais": 100 * (h.idade >= 55).mean()})
        for gr, fxs in [("≥15", FX)] + list(GRUPOS.items()):
            d = pd.DataFrame([{"ano": a, "fx": f, "n": k.get((a, f), 0), "pop": pa.loc[(a, f)]}
                              for a in range(2003, 2025) if a not in (2020, 2021) for f in fxs])
            m = smf.glm("n ~ C(fx) + ano", data=d, family=sm.families.Poisson(), offset=np.log(d["pop"])).fit(scale="X2")
            ci = m.conf_int().loc["ano"]
            tend.append({"regiao": reg, "faixa": gr, "variacao_anual_pct_2003_2024": 100 * (np.exp(m.params["ano"]) - 1),
                         "ic95_inf": 100 * (np.exp(ci[0]) - 1), "ic95_sup": 100 * (np.exp(ci[1]) - 1)})
    s = pd.DataFrame(lin)
    salvar(s, "B1_serie_regional_2003_2025")
    salvar(pd.DataFrame(tend), "B2_variacao_anual_regional_2003_2024")
    return s


# ------------------------------------------------------------------ C
def triangulacao(c, pop):
    P.salvar = salvar  # grava as tabelas da fase 7 nesta pasta (mesmo cálculo: 2022–24 ÷ 2003–05)
    P.RES = RES
    P.triangulacao(pop)
    T["C1_cirurgia_x_obito_razoes_idade_regiao"] = T.pop("1a_cirurgia_x_obito_razoes_por_idade_regiao")
    T["C2_gradiente_cirurgia_x_obito"] = T.pop("1b_gradiente_geracional_cirurgia_x_obito")
    for velho, novo in [("1a_cirurgia_x_obito_razoes_por_idade_regiao", "C1_cirurgia_x_obito_razoes_idade_regiao"),
                        ("1b_gradiente_geracional_cirurgia_x_obito", "C2_gradiente_cirurgia_x_obito")]:
        os.replace(os.path.join(OUT, velho + ".csv"), os.path.join(OUT, novo + ".csv"))
    RES["C_interacao_medida_x_periodo_x_idade_Brasil"] = RES.pop("triangulacao_interacao_medida_periodo_idade")
    # séries anuais padronizadas (faixas TabNet; padrão Brasil 2010), 2003–2024, cirurgia × óbito
    pg = pop.assign(g=pop.faixa.map(P.POPF))
    gs = sorted(set(P.POPF.values()), key=lambda s: int(s.split("-")[0].rstrip("+")))
    padrao = pg[pg.ano == 2010].groupby("g")["pop"].sum().reindex(gs)
    w = padrao / padrao.sum()
    mo = pd.read_csv(os.path.join(AQUI, "sim", "tabnet_obitos_reumaticos_2000_2024.csv"))
    mo = mo[(mo.causa == "I00–I09") & mo.faixa_tabnet.isin(P.MAPA)].assign(g=lambda x: x.faixa_tabnet.map(P.MAPA))
    ci_ = c[c.reum_ampliada & (c.fx >= 0)].assign(g=lambda x: x.fx.map(P.POPF))
    lin, tend = [], []
    for reg in REGS:
        den = (pg if reg == "Brasil" else pg[pg.regiao == reg]).groupby(["ano", "g"])["pop"].sum()
        for med, k in [("cirurgia", (ci_ if reg == "Brasil" else ci_[ci_.regiao_res == reg]).groupby(["ano", "g"]).size()),
                       ("obito", (mo if reg == "Brasil" else mo[mo.regiao == reg]).groupby(["ano", "g"]).obitos.sum())]:
            for a in range(2003, 2025):
                r = np.array([k.get((a, g), 0) / den.loc[(a, g)] for g in gs])
                lin.append({"regiao": reg, "medida": med, "ano": a, "n": int(sum(k.get((a, g), 0) for g in gs)),
                            "taxa_padronizada": 1e5 * (r * w.to_numpy()).sum()})
            for rot, anos in [("2003–2024", range(2003, 2025)), ("2013–2024", range(2013, 2025))]:
                d = pd.DataFrame([{"ano": a, "g": g, "n": k.get((a, g), 0), "pop": den.loc[(a, g)]}
                                  for a in anos if a not in (2020, 2021) for g in gs if g not in ("0-4",)])
                m = smf.glm("n ~ C(g) + ano", data=d, family=sm.families.Poisson(), offset=np.log(d["pop"])).fit(scale="X2")
                ci = m.conf_int().loc["ano"]
                tend.append({"regiao": reg, "medida": med, "periodo": rot, "variacao_anual_pct": 100 * (np.exp(m.params["ano"]) - 1),
                             "ic95_inf": 100 * (np.exp(ci[0]) - 1), "ic95_sup": 100 * (np.exp(ci[1]) - 1)})
    salvar(pd.DataFrame(lin), "C3_series_cirurgia_obito_regiao_2003_2024")
    salvar(pd.DataFrame(tend), "C4_tendencia_cirurgia_obito_regiao")


# ------------------------------------------------------------------ D
def dg(k1, k2, den1, den2):
    """Das Gupta para N = P × Σ s_i r_i; k = contagens por faixa, den = população por faixa (arrays)."""
    f = {}
    for i, (k, den) in enumerate([(k1, den1), (k2, den2)]):
        f[i] = (den.sum(), den / den.sum(), k / den)
    (P1, s1, r1), (P2, s2, r2) = f[0], f[1]
    N = lambda P, s, r: P * (s * r).sum()  # noqa: E731
    efP = ((N(P2, s1, r1) - N(P1, s1, r1)) + (N(P2, s2, r2) - N(P1, s2, r2))) / 3 \
        + ((N(P2, s1, r2) - N(P1, s1, r2)) + (N(P2, s2, r1) - N(P1, s2, r1))) / 6
    efS = ((N(P1, s2, r1) - N(P1, s1, r1)) + (N(P2, s2, r2) - N(P2, s1, r2))) / 3 \
        + ((N(P1, s2, r2) - N(P1, s1, r2)) + (N(P2, s2, r1) - N(P2, s1, r1))) / 6
    efR = ((N(P1, s1, r2) - N(P1, s1, r1)) + (N(P2, s2, r2) - N(P2, s2, r1))) / 3 \
        + ((N(P1, s2, r2) - N(P1, s2, r1)) + (N(P2, s1, r2) - N(P2, s1, r1))) / 6
    n1, n2 = N(P1, s1, r1), N(P2, s2, r2)
    return {"n_inicio": n1, "n_fim": n2, "diferenca": n2 - n1, "efeito_tamanho_pop": efP,
            "efeito_estrutura_etaria": efS, "efeito_taxa_especifica": efR, "conferencia_soma": efP + efS + efR - (n2 - n1)}


def das_gupta(c, pop):
    x0 = c[c.reum_ampliada & (c.fx >= 0)]
    lin = []
    estratos = [("Brasil", None, None)] + [(r, r, None) for r in REGS5] + [("Homens", None, "M"), ("Mulheres", None, "F")]
    for rot_per, a1, a2 in [("2003 × 2024 (principal)", [2003], [2024]), ("2003–05 × 2022–24 (média anual)", A1[:3], [2022, 2023, 2024])]:
        for rot, reg, sx in estratos:
            x = x0
            if reg:
                x = x[x.regiao_res == reg]
            if sx:
                x = x[x.sexo == sx]
            pa = denominador(pop, reg, sx)
            k = x.groupby(["ano", "fx"]).size()
            cont = lambda anos: np.array([sum(k.get((a, f), 0) for a in anos) for f in range(F.NF)]) / len(anos)  # noqa: E731
            dens = lambda anos: np.array([sum(pa.loc[(a, f)] for a in anos) for f in range(F.NF)]) / len(anos)  # noqa: E731
            lin.append({"comparacao": rot_per, "estrato": rot, **dg(cont(a1), cont(a2), dens(a1), dens(a2))})
    t = pd.DataFrame(lin)
    for col in ("efeito_tamanho_pop", "efeito_estrutura_etaria", "efeito_taxa_especifica"):
        t["pct_" + col.replace("efeito_", "")] = 100 * t[col] / t.n_inicio
    salvar(t, "D_das_gupta_regiao_sexo")
    RES["D_das_gupta_fecha"] = bool((t.conferencia_soma.abs() < 1e-6).all())


# ------------------------------------------------------------------ E
def conferencias(c, s):
    x = c[c.reum_ampliada]
    por_ano = x.groupby("ano").size()
    reg = x.groupby(["ano", x.regiao_res.fillna("Ignorada")]).size().unstack(fill_value=0)
    serie6 = pd.read_csv(os.path.join(AQUI, "resultados_fase6", "1_serie_2003_2025.csv"))
    serie6 = serie6[serie6.definicao == "Reumática ampliada"].set_index("ano").n
    # série regional (B1) conta só idade válida; soma das 5 regiões + residência ignorada = Brasil (B1)
    xv = x[x.fx >= 0]
    ign_v = xv[xv.regiao_res.isna()].groupby("ano").size().reindex(por_ano.index, fill_value=0)
    br = s[s.regiao == "Brasil"].set_index("ano").n
    regs = s[s.regiao != "Brasil"].groupby("ano").n.sum()
    RES["E_conferencias"] = {"regioes_mais_ignorada_igual_brasil": bool((reg.sum(axis=1) == por_ano).all()),
                             "residencia_ignorada_total": int(x.regiao_res.isna().sum()),
                             "totais_iguais_fase6": bool((serie6.reindex(por_ano.index) == por_ano).all()),
                             "B1_regioes_mais_ignorada_igual_brasil": bool((regs + ign_v == br).all()),
                             "sem_idade_valida": int((x.fx < 0).sum()),
                             "n_2003": int(por_ano[2003]), "n_2024": int(por_ano[2024]), "n_2003_2025": int(por_ano.sum())}


def main():
    pop = F.populacao()
    c = L.montar()
    regiao_sexo(c, pop)
    print("ok: A", flush=True)
    s = series_regionais(c, pop)
    print("ok: B", flush=True)
    triangulacao(c, pop)
    print("ok: C", flush=True)
    das_gupta(c, pop)
    print("ok: D", flush=True)
    conferencias(c, s)
    with pd.ExcelWriter(os.path.join(AQUI, "fase8_2003_2025.xlsx")) as w:
        for k, v in T.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    json.dump(RES, open(os.path.join(OUT, "resumo.json"), "w"), indent=1, default=str)
    print(json.dumps(RES, indent=1, default=str))


if __name__ == "__main__":
    main()
