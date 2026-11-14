#!/usr/bin/env python3
"""Fase 9 — análises pedidas pelas revisões externas (5/nov/2026), período 2003–2025, definição ampliada.

A. Composição da definição: contribuição anual de cada CID (I05–I09, Q22.5, I38) e razões geracionais na definição
   estrita (I05–I09) ao lado da ampliada.
B. Pandemia: (i) tendência 2003–2019 × 2003–2024 sem 2020–21; (ii) observado ÷ esperado em 2020–2025 pela tendência
   2010–2019 por faixa etária; (iii) razão geracional com fim pré-pandemia (2017–19 ÷ 2003–05).
C. UF: taxa padronizada (≥15 anos) 2003–05 e 2022–24 por UF de residência; razão; correlação de Spearman com a renda
   domiciliar per capita (PNAD Contínua, SIDRA 7531, 2012). Associação ecológica.
D. Idade-período-coorte (Clayton–Schifflers): idades 15–79 (13 faixas de 5 anos) × períodos 2005–09, 2010–14, 2015–19,
   2020–24; Poisson com escala de Pearson; tabela de deviance (idade, idade+drift, idade+coorte, idade+período, IPC);
   razões de taxas por coorte no modelo idade-coorte (drift atribuído à coorte; não identificável).
E. Tabela 1 por períodos (2003–05, 2013–15, 2023–25).

Saídas: valvar/resultados_fase9/*.csv, resumo.json, valvar/fase9_suplemento.xlsx.
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

OUT = os.path.join(AQUI, "resultados_fase9")
os.makedirs(OUT, exist_ok=True)
T, RES = {}, {}
GRUPOS = {"15-39": list(range(3, 8)), "40-54": list(range(8, 11)), "55+": list(range(11, F.NF))}
FX = list(range(3, F.NF))
UF_SIGLA = {11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO", 21: "MA", 22: "PI", 23: "CE", 24: "RN",
            25: "PB", 26: "PE", 27: "AL", 28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP", 41: "PR", 42: "SC",
            43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF"}


def salvar(df, nome):
    T[nome] = df
    df.to_csv(os.path.join(OUT, nome + ".csv"), index=False)


def poisson(d, f):
    return smf.glm(f, data=d, family=sm.families.Poisson(), offset=np.log(d["pop"])).fit(scale="X2")


def pct(m, v="ano"):
    ci = m.conf_int().loc[v]
    return [100 * (np.exp(x) - 1) for x in (m.params[v], ci[0], ci[1])]


def razao_geracional(x, pa, a1, a2):
    k1, k2 = x[x.ano.isin(a1)].groupby("fx").size(), x[x.ano.isin(a2)].groupby("fx").size()
    out = {}
    for g, fxs in GRUPOS.items():
        n1, n2 = sum(k1.get(f, 0) for f in fxs), sum(k2.get(f, 0) for f in fxs)
        p1 = sum(pa.loc[(a, f)] for a in a1 for f in fxs)
        p2 = sum(pa.loc[(a, f)] for a in a2 for f in fxs)
        rr, se = (n2 / p2) / (n1 / p1), np.sqrt(1 / n1 + 1 / n2)
        out[g] = (n1, n2, rr, rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se))
    return out


# ------------------------------------------------------------------ A
def composicao(c, pop):
    x = c[c.reum_ampliada].copy()
    x["cid"] = np.where(x.DIAG_PRINC.str[:4].eq("Q225"), "Q22.5", x.DIAG_PRINC.str[:3])
    t = x.groupby(["ano", "cid"]).size().unstack(fill_value=0)
    t["total_ampliada"] = t.sum(axis=1)
    t["pct_Q22.5_I38"] = 100 * (t.get("Q22.5", 0) + t.get("I38", 0)) / t.total_ampliada
    salvar(t.reset_index(), "A1_composicao_anual_por_cid")
    pa = pop.groupby(["ano", "faixa"])["pop"].sum()
    lin = []
    for nome, m in [("Ampliada (I05–I09 + Q22.5/I38 adultos)", c.reum_ampliada), ("Estrita (I05–I09)", c.reum)]:
        r = razao_geracional(c[m & (c.fx >= 0)], pa, [2003, 2004, 2005], [2023, 2024, 2025])
        lin += [{"definicao": nome, "faixa": g, "n_2003_05": v[0], "n_2023_25": v[1], "razao": v[2], "ic95_inf": v[3],
                 "ic95_sup": v[4]} for g, v in r.items()]
    salvar(pd.DataFrame(lin), "A2_razoes_estrita_x_ampliada")


# ------------------------------------------------------------------ B
def pandemia(c, pop):
    pa = pop.groupby(["ano", "faixa"])["pop"].sum()
    x = c[c.reum_ampliada & (c.fx >= 0)]
    k = x.groupby(["ano", "fx"]).size()
    lin, oe = [], []
    for g, fxs in [("≥15", FX)] + list(GRUPOS.items()):
        for rot, anos in [("2003–2019", range(2003, 2020)), ("2003–2024 sem 2020–21", [a for a in range(2003, 2025) if a not in (2020, 2021)]),
                          ("2003–2025 todos os anos", range(2003, 2026))]:
            d = pd.DataFrame([{"ano": a, "fx": f, "n": k.get((a, f), 0), "pop": pa.loc[(a, f)]} for a in anos for f in fxs])
            v = pct(poisson(d, "n ~ C(fx) + ano"))
            lin.append({"faixa": g, "serie": rot, "variacao_anual_pct": v[0], "ic95_inf": v[1], "ic95_sup": v[2]})
        d = pd.DataFrame([{"ano": a, "fx": f, "n": k.get((a, f), 0), "pop": pa.loc[(a, f)]} for a in range(2010, 2020) for f in fxs])
        m = poisson(d, "n ~ C(fx) + ano")
        for a in range(2020, 2026):
            nd = pd.DataFrame([{"ano": a, "fx": f, "pop": pa.loc[(a, f)]} for f in fxs])
            esp = float((m.predict(nd, offset=np.log(nd["pop"]))).sum())
            obs = int(sum(k.get((a, f), 0) for f in fxs))
            oe.append({"faixa": g, "ano": a, "observado": obs, "esperado_tendencia_2010_19": esp, "obs_div_esp": obs / esp})
    salvar(pd.DataFrame(lin), "B1_tendencia_com_e_sem_pandemia")
    salvar(pd.DataFrame(oe), "B2_observado_esperado_2020_2025")
    r = razao_geracional(x, pa, [2003, 2004, 2005], [2017, 2018, 2019])
    salvar(pd.DataFrame([{"faixa": g, "n_2003_05": v[0], "n_2017_19": v[1], "razao": v[2], "ic95_inf": v[3], "ic95_sup": v[4]}
                         for g, v in r.items()]), "B3_razao_geracional_pre_pandemia_2017_19")


# ------------------------------------------------------------------ C
def uf(c, pop):
    renda = json.load(open(os.path.join(AQUI, "ref", "sidra7531_renda_domiciliar_pc_uf_2012_2025.json")))[1:]
    renda = {UF_SIGLA[int(r["D1C"])]: float(r["V"]) for r in renda
             if r["D3N"] == "2012" and r["D2C"] == "10824" and r["D4N"] == "Total"}
    padrao = pop.groupby(["ano", "faixa"])["pop"].sum().loc[2010]
    x = c[c.reum_ampliada & (c.fx >= 0)]
    lin = []
    for u in sorted(pop.uf.unique()):
        pu = pop[pop.uf == u].groupby(["ano", "faixa"])["pop"].sum()
        xu = x[x.uf_res == u]
        row = {"uf": u, "regiao": F.UF_REG[u], "renda_pc_2012": renda.get(u)}
        for rot, anos in [("2003_05", [2003, 2004, 2005]), ("2022_24", [2022, 2023, 2024])]:
            kk = xu[xu.ano.isin(anos)].groupby("fx").size()
            den = sum(pu.loc[a] for a in anos)
            t, v = J.padronizada(kk, den, padrao, FX)
            row.update({f"n_{rot}": int(kk.sum()), f"taxa_{rot}": t * len(anos) / len(anos), f"var_{rot}": v})
        # taxa média anual: den somado em 3 anos → dividir por 1 já dá taxa por pessoa-ano
        row["razao_taxas"] = row["taxa_2022_24"] / row["taxa_2003_05"]
        se = np.sqrt(row["var_2003_05"] / row["taxa_2003_05"] ** 2 + row["var_2022_24"] / row["taxa_2022_24"] ** 2)
        row["ic95_inf"], row["ic95_sup"] = row["razao_taxas"] * np.exp(-1.96 * se), row["razao_taxas"] * np.exp(1.96 * se)
        r = razao_geracional(xu, pu, [2003, 2004, 2005], [2022, 2023, 2024])
        row["razao_15_39"], row["razao_55mais"] = r["15-39"][2], r["55+"][2]
        lin.append(row)
    t = pd.DataFrame(lin).drop(columns=["var_2003_05", "var_2022_24"])
    salvar(t, "C1_uf_taxas_razoes_renda")
    ok = t.dropna(subset=["renda_pc_2012"])
    RES["C_spearman_renda_x_razao"] = [round(v, 3) for v in stats.spearmanr(ok.renda_pc_2012, ok.razao_taxas)]
    RES["C_spearman_renda_x_taxa_2022_24"] = [round(v, 3) for v in stats.spearmanr(ok.renda_pc_2012, ok.taxa_2022_24)]
    RES["C_spearman_renda_x_taxa_2003_05"] = [round(v, 3) for v in stats.spearmanr(ok.renda_pc_2012, ok.taxa_2003_05)]
    RES["C_n_uf_com_renda"] = int(len(ok))


def capacidade(c, pop):
    """Volume anual de toda cirurgia valvar realizada na UF (local de internação) e hospitais com ≥20/ano, 2003–05 × 2022–24;
    correlação (Spearman) da mudança de volume com a razão de taxas reumáticas dos residentes (C1). Ecológica."""
    a = pd.read_csv(os.path.join(AQUI, "base_valvar_2003_2007.csv.gz"), dtype=str)
    b = pd.read_csv(os.path.join(AQUI, "base_valvar_2008_2025.csv.gz"), dtype=str)
    x = pd.concat([a, b[b.cirurgia == "True"]])
    x["ano"] = x.ano.astype(int)
    lin = {}
    for rot, anos in [("2003_05", [2003, 2004, 2005]), ("2022_24", [2022, 2023, 2024])]:
        g = x[x.ano.isin(anos)]
        h = g.groupby(["uf_int", "CNES"]).size() / 3
        lin[f"hosp20_{rot}"] = h[h >= 20].groupby(level=0).size()
        lin[f"valvares_ano_{rot}"] = g.groupby("uf_int").size() / 3
    t = pd.DataFrame(lin).fillna(0)
    t["razao_volume"] = t.valvares_ano_2022_24 / t.valvares_ano_2003_05.replace(0, np.nan)
    u = T["C1_uf_taxas_razoes_renda"].set_index("uf")
    t["razao_taxas_reum_residentes"] = u.razao_taxas
    t = t.rename_axis("uf").reset_index()
    salvar(t, "C2_uf_capacidade_cirurgica")
    ok = t.dropna(subset=["razao_volume", "razao_taxas_reum_residentes"])
    ok = ok[np.isfinite(ok.razao_volume)]
    RES["C_spearman_volume_valvar_uf_x_razao_reum"] = [round(v, 3) for v in stats.spearmanr(ok.razao_volume, ok.razao_taxas_reum_residentes)] + [int(len(ok))]


# ------------------------------------------------------------------ D
def ipc(c, pop):
    PER = {2005: range(2005, 2010), 2010: range(2010, 2015), 2015: range(2015, 2020), 2020: range(2020, 2025)}
    fxs = list(range(3, 16))  # 15–19 … 75–79
    pa = pop.groupby(["ano", "faixa"])["pop"].sum()
    x = c[c.reum_ampliada & c.fx.isin(fxs)]
    k = x.groupby(["ano", "fx"]).size()
    d = pd.DataFrame([{"periodo": p, "fx": f, "idade": 5 * f, "n": sum(k.get((a, f), 0) for a in anos),
                       "pop": sum(pa.loc[(a, f)] for a in anos)} for p, anos in PER.items() for f in fxs])
    d["coorte"] = d.periodo - d.idade  # ano de nascimento (centro aproximado: período início − idade início)
    d["drift"] = (d.periodo - 2005) / 5
    mods = {"Idade": "n ~ C(fx)", "Idade + drift": "n ~ C(fx) + drift", "Idade + período": "n ~ C(fx) + C(periodo)",
            "Idade + coorte": "n ~ C(fx) + C(coorte)", "Idade + período + coorte": "n ~ C(fx) + C(periodo) + C(coorte)"}
    fit = {k_: smf.glm(f, data=d, family=sm.families.Poisson(), offset=np.log(d["pop"])).fit() for k_, f in mods.items()}
    phi = fit["Idade + período + coorte"].pearson_chi2 / fit["Idade + período + coorte"].df_resid
    tab = [{"modelo": k_, "deviance": m.deviance, "gl_resid": int(m.df_resid)} for k_, m in fit.items()]
    salvar(pd.DataFrame(tab), "D1_ipc_deviance")
    comp = []
    for a, b, rot in [("Idade + drift", "Idade + coorte", "curvatura de coorte (dado drift)"),
                      ("Idade + drift", "Idade + período", "curvatura de período (dado drift)"),
                      ("Idade + coorte", "Idade + período + coorte", "período, dado idade + coorte"),
                      ("Idade + período", "Idade + período + coorte", "coorte, dado idade + período"),
                      ("Idade", "Idade + drift", "drift (tendência linear)")]:
        dd, gl = fit[a].deviance - fit[b].deviance, int(fit[a].df_resid - fit[b].df_resid)
        fv = dd / gl / phi
        comp.append({"teste": rot, "delta_deviance": dd, "gl": gl, "F": fv, "p": float(stats.f.sf(fv, gl, fit["Idade + período + coorte"].df_resid))})
    salvar(pd.DataFrame(comp), "D2_ipc_testes")
    RES["D_dispersao_phi"] = round(float(phi), 2)
    m = smf.glm(mods["Idade + drift"], data=d, family=sm.families.Poisson(), offset=np.log(d["pop"])).fit(scale="X2")
    RES["D_net_drift_pct_ano"] = [round(v, 2) for v in [100 * (np.exp(m.params["drift"] / 5) - 1)] +
                                  [100 * (np.exp(q / 5) - 1) for q in m.conf_int().loc["drift"]]]
    mc = fit["Idade + coorte"]
    ref = 1950
    co = sorted(d.coorte.unique())
    rr = []
    base = mc.params.get(f"C(coorte)[T.{ref}]", 0.0)
    for cc in co:
        b = mc.params.get(f"C(coorte)[T.{cc}]", 0.0)
        rr.append({"coorte_nascimento": int(cc), "rr_vs_1950": float(np.exp(b - base)),
                   "n": int(d[d.coorte == cc].n.sum())})
    salvar(pd.DataFrame(rr), "D3_razao_taxas_coorte_modelo_idade_coorte")


# ------------------------------------------------------------------ E
def tabela1(c):
    b = pd.read_csv(os.path.join(AQUI, "base_valvar_2008_2025.csv.gz"), dtype={"N_AIH": str, "DIAG_PRINC": str})
    b = b[b.cirurgia & b.reum_ampliada]
    a = pd.read_csv(os.path.join(AQUI, "base_valvar_2003_2007.csv.gz"), dtype={"N_AIH": str, "DIAG_PRINC": str})
    a = a[a.reum_ampliada]
    x = pd.concat([a.assign(sp=False), b.assign(sp=True)], ignore_index=True)
    x["cid3"] = x.DIAG_PRINC.str[:3]
    per = {"2003–05": [2003, 2004, 2005], "2013–15": [2013, 2014, 2015], "2023–25": [2023, 2024, 2025], "2008–10 (SIH-SP)": [2008, 2009, 2010]}
    lin = []
    for p, anos in per.items():
        g = x[x.ano.isin(anos)]
        r = {"periodo": p, "n": len(g), "n_por_ano": len(g) / 3, "idade_mediana": g.idade.median(), "idade_p25": g.idade.quantile(.25),
             "idade_p75": g.idade.quantile(.75), "pct_mulheres": 100 * g.sexo.eq("F").mean(),
             "pct_15_39": 100 * g.idade.between(15, 39.999).mean(), "pct_55mais": 100 * (g.idade >= 55).mean(),
             "pct_menor15": 100 * (g.idade < 15).mean(),
             "pct_nascidos_antes_1975": 100 * ((g.ano - g.idade) < 1975).mean(),
             "mort_hosp_pct": 100 * g.obito.astype(float).mean()}
        for reg in ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
            r[f"pct_{reg}"] = 100 * g.regiao_res.eq(reg).mean()
        for cid in ["I05", "I06", "I07", "I08", "I09"]:
            r[f"pct_{cid}"] = 100 * g.cid3.eq(cid).mean()
        r["pct_Q225_I38"] = 100 * (g.DIAG_PRINC.str[:4].eq("Q225") | g.cid3.eq("I38")).mean()
        if g.sp.all():
            r["pct_mecanica_entre_trocas"] = 100 * g.mec.astype(float).gt(0).sum() / (g.mec.astype(float).gt(0) | g.bio.astype(float).gt(0)).sum()
            r["pct_sem_protese_mitral_I05"] = 100 * (g[g.cid3.eq("I05")].n_protese.astype(float).eq(0)).mean()
        lin.append(r)
    salvar(pd.DataFrame(lin), "E_tabela1_periodos")


def main():
    pop = F.populacao()
    c = L.montar()
    c["uf_res"] = pd.concat([pd.read_csv(os.path.join(AQUI, "base_valvar_2003_2007.csv.gz"), dtype=str, usecols=["N_AIH", "uf_res"]),
                             pd.read_csv(os.path.join(AQUI, "base_valvar_2008_2025.csv.gz"), dtype=str, usecols=["N_AIH", "uf_res"])]
                            ).drop_duplicates("N_AIH").set_index("N_AIH").uf_res.reindex(c.N_AIH).to_numpy()
    for nome, fun in [("A", composicao), ("B", pandemia), ("C", uf), ("C2", capacidade), ("D", ipc)]:
        fun(c, pop)
        print("ok:", nome, flush=True)
    tabela1(c)
    with pd.ExcelWriter(os.path.join(AQUI, "fase9_suplemento.xlsx")) as w:
        for k, v in T.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    json.dump(RES, open(os.path.join(OUT, "resumo.json"), "w"), indent=1, default=str)
    print(json.dumps(RES, indent=1, default=str))


if __name__ == "__main__":
    main()
