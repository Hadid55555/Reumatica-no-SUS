#!/usr/bin/env python3
"""Fase 4 — segunda revisão: joinpoint, sequência idade–período–coorte, oferta efetiva por UF e SIM (2016–2024).

A. Joinpoint (log-linear, 0–2 pontos, seleção por BIC; mínimo de 3 anos por segmento) da taxa padronizada por idade:
   Brasil, regiões, 15–39 e 40+; série 2008–2024 sem 2020–2021 e série 2008–2019. APC por segmento e AAPC.
B. Modelos de Clayton–Schifflers (Poisson): idade, idade + drift, idade–período, idade–coorte, idade–período–coorte;
   faixas de 5 anos (15–79) × períodos 2010–14, 2015–19, 2020–24; teste de desvio.
C. Oferta efetiva (SIH): hospitais com cirurgia valvar por UF de internação e ano (≥1 e ≥20/ano) por milhão;
   % de residentes operados na própria região; associação ecológica entre mudança de oferta e de taxa reumática por UF.
D. SIM 2016–2024 (se extraído por valvar/sim/extrair_sim.py): óbitos por doença reumática (causa básica I00–I09 e
   I05–I08; menção em qualquer linha), taxas padronizadas por região e comparação com a cirurgia.

Saídas: valvar/resultados_fase4/*.csv, valvar/fase4_revisao2.xlsx. Uso: python valvar/fase4_revisao2.py
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
import fase2_valvar as F  # noqa: E402

OUT = os.path.join(AQUI, "resultados_fase4")
os.makedirs(OUT, exist_ok=True)
T, RES = {}, {}
COD_UF = {int(k): v for k, v in F.UF_IBGE.items()}


def salvar(df, nome):
    T[nome] = df
    df.to_csv(os.path.join(OUT, nome + ".csv"), index=False)


def carregar():
    p = os.path.join(RAIZ, "saida_valvar", "base_valvar_2008_2025.csv.gz")
    if not os.path.exists(p):
        p = os.path.join(AQUI, "base_valvar_2008_2025.csv.gz")
    c = pd.read_csv(p, dtype={"N_AIH": str, "CNES": str, "DIAG_PRINC": str, "PROC_REA": str})
    c = c[c.cirurgia].copy()
    c["reum"] = c.etiologia.eq("Reumática")
    c["fx"] = F.faixa5(c.idade).to_numpy()
    return c


def padronizada(k, den, padrao, fxs):
    """Taxa padronizada (/100 mil) e variância, faixas fxs; k e den indexados por faixa."""
    w = padrao.loc[fxs] / padrao.loc[fxs].sum()
    r = np.array([k.get(f, 0) / den.loc[f] for f in fxs])
    v = np.array([k.get(f, 0) / den.loc[f] ** 2 for f in fxs])
    return 1e5 * (r * w.to_numpy()).sum(), 1e10 * (v * w.to_numpy() ** 2).sum()


def series(c, pop):
    pb = pop.groupby(["ano", "faixa"])["pop"].sum()
    pr = pop.groupby(["regiao", "ano", "faixa"])["pop"].sum()
    padrao = pb.loc[2010]
    faixas = {"todas (15+)": list(range(3, F.NF)), "15-39": list(range(3, 8)), "40+": list(range(8, F.NF))}
    linhas = []
    for dn, col in [("estrita", "reum"), ("ampliada", "reum_ampliada")]:
        x = c[c[col] & (c.fx >= 0)]
        for reg in ["Brasil", "Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
            xr = x if reg == "Brasil" else x[x.regiao_res == reg]
            k = xr.groupby(["ano", "fx"]).size()
            for nome, fxs in faixas.items():
                if dn == "ampliada" and nome != "todas (15+)":
                    continue
                for a in range(2008, 2026):
                    den = pb.loc[a] if reg == "Brasil" else pr.loc[(reg, a)]
                    t, v = padronizada(k.get(a, pd.Series(dtype=float)), den, padrao, fxs)
                    linhas.append({"definicao": dn, "regiao": reg, "faixa": nome, "ano": a, "taxa": t, "var": v})
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ A. joinpoint
def ajusta_jp(anos, y, w, jps):
    X = [np.ones_like(anos, float), anos - anos.min()]
    for j in jps:
        X.append(np.clip(anos - j, 0, None))
    X = np.column_stack(X)
    m = sm.WLS(y, X, weights=w).fit()
    return m


def joinpoint(anos, taxa, var, max_jp=2, min_seg=3):
    anos, taxa, var = np.asarray(anos, float), np.asarray(taxa, float), np.asarray(var, float)
    y = np.log(taxa)
    w = 1 / (var / taxa ** 2)  # variância do log pelo método delta
    melhor = None
    cand = anos[min_seg - 1:-(min_seg - 1)] if len(anos) > 2 * min_seg else []
    for k in range(0, max_jp + 1):
        for jps in itertools.combinations(cand, k):
            pts = [anos.min()] + list(jps) + [anos.max()]
            if any(b - a < min_seg - 1 for a, b in zip(pts, pts[1:])):
                continue
            m = ajusta_jp(anos, y, w, jps)
            n = len(y)
            rss = (m.resid ** 2 * w).sum()
            bic = n * np.log(rss / n) + (2 + 2 * k) * np.log(n)  # parâmetros: 2 + inclinação e local por ponto
            if melhor is None or bic < melhor[0]:
                melhor = (bic, jps, m)
    bic, jps, m = melhor
    b, cov = m.params, m.cov_params()
    pts = [anos.min()] + list(jps) + [anos.max()]
    segs = []
    for i in range(len(pts) - 1):
        L = np.zeros(len(b))
        L[1:2 + i] = 1
        s, se = L @ b, np.sqrt(L @ cov @ L)
        segs.append({"inicio": int(pts[i]), "fim": int(pts[i + 1]), "APC": 100 * (np.exp(s) - 1),
                     "ic95_inf": 100 * (np.exp(s - 1.96 * se) - 1), "ic95_sup": 100 * (np.exp(s + 1.96 * se) - 1)})
    # AAPC: média das inclinações ponderada pela duração (Clegg et al. 2009)
    L = np.zeros(len(b))
    tot = anos.max() - anos.min()
    for i in range(len(pts) - 1):
        L[1:2 + i] += (pts[i + 1] - pts[i]) / tot
    s, se = L @ b, np.sqrt(L @ cov @ L)
    return jps, segs, (100 * (np.exp(s) - 1), 100 * (np.exp(s - 1.96 * se) - 1), 100 * (np.exp(s + 1.96 * se) - 1))


def joinpoints(s):
    linhas, aapc = [], []
    for (dn, reg, fx), g in s.groupby(["definicao", "regiao", "faixa"]):
        for rot, filt in [("2008–2024 sem 2020–21", g.ano.between(2008, 2024) & ~g.ano.isin([2020, 2021])),
                          ("2008–2019", g.ano.between(2008, 2019))]:
            h = g[filt]
            jps, segs, a = joinpoint(h.ano, h.taxa, h["var"])
            for sg in segs:
                linhas.append({"definicao": dn, "regiao": reg, "faixa": fx, "serie": rot, "joinpoints": ",".join(str(int(j)) for j in jps),
                               **sg})
            aapc.append({"definicao": dn, "regiao": reg, "faixa": fx, "serie": rot, "joinpoints": ",".join(str(int(j)) for j in jps),
                         "AAPC": a[0], "ic95_inf": a[1], "ic95_sup": a[2]})
    salvar(pd.DataFrame(linhas), "A1_joinpoint_segmentos")
    salvar(pd.DataFrame(aapc), "A2_joinpoint_AAPC")
    salvar(s, "A0_series_padronizadas")


# ------------------------------------------------------------------ B. APC
def apc(c, pop):
    pb = pop.groupby(["ano", "faixa"])["pop"].sum()
    x = c[c.reum & c.fx.between(3, 15)]
    per = {a: p for p, anos in {2010: range(2010, 2015), 2015: range(2015, 2020), 2020: range(2020, 2025)}.items() for a in anos}
    x = x[x.ano.isin(per)].assign(per=lambda d: d.ano.map(per))
    k = x.groupby(["per", "fx"]).size()
    rows = []
    for p, anos in {2010: range(2010, 2015), 2015: range(2015, 2020), 2020: range(2020, 2025)}.items():
        for f in range(3, 16):
            rows.append({"per": p, "fx": f, "n": k.get((p, f), 0), "pop": sum(pb.loc[(a, f)] for a in anos)})
    g = pd.DataFrame(rows)
    g["coorte"] = g.per - 5 * g.fx  # ano de nascimento (início da faixa), passos de 5 anos
    g["t"] = (g.per - 2015) / 5
    g["lp"] = np.log(g["pop"])
    mods = {"Idade": "n ~ C(fx)", "Idade + drift": "n ~ C(fx) + t", "Idade–período": "n ~ C(fx) + C(per)",
            "Idade–coorte": "n ~ C(fx) + C(coorte)", "Idade–período–coorte": "n ~ C(fx) + C(per) + C(coorte)"}
    fit = {k: smf.glm(f, data=g, family=sm.families.Poisson(), offset=g.lp).fit() for k, f in mods.items()}
    lin = []
    for k, m in fit.items():
        lin.append({"modelo": k, "desvio": m.deviance, "gl": m.df_resid, "AIC": m.aic})
    t = pd.DataFrame(lin)
    disp = fit["Idade–período–coorte"].deviance / max(fit["Idade–período–coorte"].df_resid, 1)
    comp = []
    for a, b in [("Idade", "Idade + drift"), ("Idade + drift", "Idade–coorte"), ("Idade + drift", "Idade–período"),
                 ("Idade–período", "Idade–período–coorte"), ("Idade–coorte", "Idade–período–coorte")]:
        dd, dg = fit[a].deviance - fit[b].deviance, fit[a].df_resid - fit[b].df_resid
        comp.append({"de": a, "para": b, "delta_desvio": dd, "delta_gl": dg, "p_qui2": stats.chi2.sf(dd, dg),
                     "F_superdispersao": (dd / dg) / disp if dg else np.nan,
                     "p_F": stats.f.sf((dd / dg) / disp, dg, fit["Idade–período–coorte"].df_resid) if dg else np.nan})
    salvar(t, "B1_apc_modelos")
    salvar(pd.DataFrame(comp), "B2_apc_comparacoes")
    nd = fit["Idade + drift"]
    ci = nd.conf_int().loc["t"]
    RES["net_drift_pct_por_ano"] = [round(100 * (np.exp(v / 5) - 1), 2) for v in (nd.params["t"], ci[0], ci[1])]


# ------------------------------------------------------------------ C. oferta
def oferta(c, pop):
    pu = pop.groupby(["uf", "ano"])["pop"].sum()
    h = c.groupby(["uf_int", "ano", "CNES"]).size().rename("n").reset_index()
    o = h.groupby(["uf_int", "ano"]).agg(hosp_1mais=("n", "size"), hosp_20mais=("n", lambda v: (v >= 20).sum())).reset_index()
    o["pop"] = [pu.get((u, a), np.nan) for u, a in zip(o.uf_int, o.ano)]
    o["hosp20_por_milhao"] = 1e6 * o.hosp_20mais / o["pop"]
    o["regiao"] = o.uf_int.map(F.UF_REG)
    salvar(o, "C1_oferta_uf_ano")
    r = o.groupby(["regiao", "ano"])[["hosp_1mais", "hosp_20mais", "pop"]].sum().reset_index()
    r["hosp20_por_milhao"] = 1e6 * r.hosp_20mais / r["pop"]
    x = c[c.reum & c.regiao_res.notna()]
    x = x.assign(mesma_regiao=(x.regiao_res == x.uf_int.map(F.UF_REG)))
    m = x.groupby(["regiao_res", "ano"]).mesma_regiao.mean().mul(100).rename("pct_operado_na_propria_regiao").reset_index()
    salvar(r.merge(m, left_on=["regiao", "ano"], right_on=["regiao_res", "ano"], how="left").drop(columns="regiao_res"),
           "C2_oferta_regiao_ano")
    # ecológico: mudança 2013–19 → 2022–24 por UF de residência
    pb = pop.groupby(["uf", "ano", "faixa"])["pop"].sum()
    padrao = pop.groupby(["ano", "faixa"])["pop"].sum().loc[2010]
    xr = c[c.reum & (c.fx >= 3)]
    lin = []
    for uf in sorted(COD_UF.values()):
        k = xr[xr.uf_res == uf].groupby(["ano", "fx"]).size()

        def taxa(anos):
            vals = []
            for a in anos:
                t, _ = padronizada(k.get(a, pd.Series(dtype=float)), pb.loc[(uf, a)], padrao, list(range(3, F.NF)))
                vals.append(t)
            return np.mean(vals)
        oo = o[o.uf_int == uf].set_index("ano")
        lin.append({"uf": uf, "regiao": F.UF_REG[uf], "taxa_2013_19": taxa(range(2013, 2020)),
                    "taxa_2022_24": taxa(range(2022, 2025)),
                    "hosp20_pm_2013_19": oo.hosp20_por_milhao.reindex(range(2013, 2020)).fillna(0).mean(),
                    "hosp20_pm_2022_24": oo.hosp20_por_milhao.reindex(range(2022, 2025)).fillna(0).mean()})
    e = pd.DataFrame(lin)
    e["log_razao_taxa"] = np.log(e.taxa_2022_24 / e.taxa_2013_19)
    e["delta_oferta"] = e.hosp20_pm_2022_24 - e.hosp20_pm_2013_19
    salvar(e, "C3_ecologico_uf")
    ok = e.replace([np.inf, -np.inf], np.nan).dropna()
    rho, p = stats.spearmanr(ok.delta_oferta, ok.log_razao_taxa)
    rho2, p2 = stats.spearmanr(ok.hosp20_pm_2022_24, ok.taxa_2022_24)
    RES["spearman_mudanca_oferta_x_mudanca_taxa"] = [round(rho, 3), round(p, 4), len(ok)]
    RES["spearman_oferta_x_taxa_2022_24"] = [round(rho2, 3), round(p2, 4), len(ok)]


# ------------------------------------------------------------------ D. SIM
def sim(c, pop):
    p = os.path.join(AQUI, "sim", "sim_reumatica_2016_2024.csv.gz")
    if not os.path.exists(p):
        RES["sim"] = "não extraído"
        return
    d = pd.read_csv(p, dtype=str)
    d["ano"] = d.DTOBITO.str[-4:].astype(int)
    d = d[d.ano.between(2016, 2024)]
    i = pd.to_numeric(d.IDADE, errors="coerce")
    d["idade"] = np.where(i // 100 == 4, i % 100, np.where(i // 100 == 5, 100 + i % 100, np.where(i < 400, 0, np.nan)))
    d["fx"] = F.faixa5(d.idade).to_numpy()
    d["uf"] = d.CODMUNRES.str[:2].map(F.UF_IBGE)
    d["regiao"] = d.uf.map(F.UF_REG)
    cb = d.CAUSABAS.fillna("").str[:3]
    d["cb_reum"] = cb.str.match(r"I0[0-9]")
    d["cb_valvar"] = cb.isin(["I05", "I06", "I07", "I08"])
    d["mencao"] = True
    pb = pop.groupby(["ano", "faixa"])["pop"].sum()
    pr = pop.groupby(["regiao", "ano", "faixa"])["pop"].sum()
    padrao = pb.loc[2010]
    lin = []
    for col in ["cb_reum", "cb_valvar", "mencao"]:
        x = d[d[col] & (d.fx >= 0)]
        for reg in ["Brasil", "Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
            xr = x if reg == "Brasil" else x[x.regiao == reg]
            k = xr.groupby(["ano", "fx"]).size()
            for a in range(2016, 2025):
                den = pb.loc[a] if reg == "Brasil" else pr.loc[(reg, a)]
                t, v = padronizada(k.get(a, pd.Series(dtype=float)), den, padrao, list(range(F.NF)))
                lin.append({"definicao": col, "regiao": reg, "ano": a, "obitos": int((xr.ano == a).sum()),
                            "taxa_padronizada": t, "idade_mediana": xr[xr.ano == a].idade.median()})
    s = pd.DataFrame(lin)
    salvar(s, "D1_sim_mortalidade_reumatica")
    tot = pd.read_csv(os.path.join(AQUI, "sim", "conferencia_totais.csv"))
    salvar(tot, "D0_sim_totais_anuais")
    # tendência 2016–2024 (sem 2020–21) da mortalidade e da cirurgia, por região
    cir = T["A0_series_padronizadas"]
    cir = cir[(cir.faixa == "todas (15+)") & (cir.definicao == "estrita")]
    comp = []
    for reg in ["Brasil", "Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
        for rot, df, col in [("óbito I00–I09 (causa básica)", s[(s.definicao == "cb_reum") & (s.regiao == reg)], "taxa_padronizada"),
                             ("cirurgia reumática", cir[cir.regiao == reg], "taxa")]:
            h = df[df.ano.between(2016, 2024) & ~df.ano.isin([2020, 2021])]
            b, se = np.polyfit(h.ano, np.log(h[col]), 1, cov=True)
            comp.append({"regiao": reg, "medida": rot, "variacao_anual_pct_2016_24": 100 * (np.exp(b[0]) - 1),
                         "ic95_inf": 100 * (np.exp(b[0] - 1.96 * se[0, 0] ** .5) - 1),
                         "ic95_sup": 100 * (np.exp(b[0] + 1.96 * se[0, 0] ** .5) - 1)})
    salvar(pd.DataFrame(comp), "D2_sim_x_cirurgia_tendencia")


def main():
    pop = F.populacao()
    c = carregar()
    s = series(c, pop)
    joinpoints(s)
    apc(c, pop)
    oferta(c, pop)
    sim(c, pop)
    with pd.ExcelWriter(os.path.join(AQUI, "fase4_revisao2.xlsx")) as w:
        for k, v in T.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    json.dump(RES, open(os.path.join(OUT, "resumo.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(RES, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
