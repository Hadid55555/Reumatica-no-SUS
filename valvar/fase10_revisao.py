#!/usr/bin/env python3
"""Fase 10 — respostas à revisão simulada (Lancet RHA, 5/nov/2026). Definição ampliada salvo indicação.

A. Comparação pré-pandemia como co-principal: razões 2017–19 ÷ 2003–05 (cirurgia reumática, I35, I34–I37, razão das
   razões) e mortalidade 2017–19 ÷ 2003–05 nas faixas do TabNet, Brasil e regiões; gradientes.
B. Codificação e padrão: definição estrita só com 2003 (antes do Q22.5 coringa) ÷ 2023–25 e ÷ 2017–19; taxas padronizadas
   com população-padrão 2022.
C. Perfil cirúrgico (SIH-SP, 2008–2025) por período, faixa etária e região: multivalvar, anel, valvar + RVM, plastia sem
   prótese, prótese mecânica (e em mulheres de 15–44 anos), CID de prótese prévia (reoperação, limite inferior).
D. Valvuloplastia mitral percutânea 2008–2025 por faixa etária e região (CID I05 e todas).
E. Mortalidade hospitalar 2003–2025: bruta e ajustada (logística; idade em spline, sexo, região, ano; 2008–2025 também
   multivalvar, RVM, urgência, registro de cirurgia múltipla), EP agrupado por hospital.
F. Acesso: % operados fora da UF e fora da região de residência; volume anual médio por hospital (cirurgia valvar total e
   reumática), por região e período.

Saídas: valvar/resultados_fase10/*.csv, resumo.json, valvar/fase10_revisao.xlsx.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import fase2_valvar as F  # noqa: E402
import fase4_revisao2 as J  # noqa: E402
import fase6_serie_longa as L  # noqa: E402
import fase7_extras as P  # noqa: E402

OUT = os.path.join(AQUI, "resultados_fase10")
os.makedirs(OUT, exist_ok=True)
T, RES = {}, {}
GRUPOS = {"15-39": list(range(3, 8)), "40-54": list(range(8, 11)), "55+": list(range(11, F.NF))}
A0, PRE, FIM = [2003, 2004, 2005], [2017, 2018, 2019], [2023, 2024, 2025]
REGS = ["Brasil", "Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
PER = {"2008–10": [2008, 2009, 2010], "2013–15": [2013, 2014, 2015], "2017–19": PRE, "2023–25": FIM}


def salvar(df, nome):
    T[nome] = df
    df.to_csv(os.path.join(OUT, nome + ".csv"), index=False)


def rr(n1, p1, n2, p2):
    r, se = (n2 / p2) / (n1 / p1), np.sqrt(1 / n1 + 1 / n2)
    return r, r * np.exp(-1.96 * se), r * np.exp(1.96 * se), se


def razoes(x, pa, a1, a2, grupos):
    k1, k2 = x[x.ano.isin(a1)].groupby("fx").size(), x[x.ano.isin(a2)].groupby("fx").size()
    out = {}
    for g, fxs in grupos.items():
        n1, n2 = sum(k1.get(f, 0) for f in fxs), sum(k2.get(f, 0) for f in fxs)
        p1 = sum(pa.loc[(a, f)] for a in a1 for f in fxs)
        p2 = sum(pa.loc[(a, f)] for a in a2 for f in fxs)
        out[g] = (n1, n2) + rr(n1, p1, n2, p2)
    return out


# ------------------------------------------------------------------ A
def pre_pandemia(c, pop):
    pa = pop.groupby(["ano", "faixa"])["pop"].sum()
    lin = []
    defs = [("Reumática ampliada", c.reum_ampliada), ("Aórtica não reumática (I35)", c.i35), ("Não reumática (I34–I37)", c.etiologia.eq("Não reumática"))]
    for a2, rot in [(PRE, "2017–19"), (FIM, "2023–25")]:
        rs = {}
        for nome, m in defs:
            rs[nome] = razoes(c[m & (c.fx >= 0)], pa, A0, a2, GRUPOS)
            for g, v in rs[nome].items():
                lin.append({"fim": rot, "definicao": nome, "faixa": g, "n_inicio": v[0], "n_fim": v[1], "razao": v[2], "ic95_inf": v[3], "ic95_sup": v[4]})
        for comp in ["Aórtica não reumática (I35)", "Não reumática (I34–I37)"]:
            for g in GRUPOS:
                a, b = rs["Reumática ampliada"][g], rs[comp][g]
                r, se = a[2] / b[2], np.sqrt(a[5] ** 2 + b[5] ** 2)
                lin.append({"fim": rot, "definicao": f"Razão das razões reum ÷ {comp}", "faixa": g, "razao": r,
                            "ic95_inf": r * np.exp(-1.96 * se), "ic95_sup": r * np.exp(1.96 * se)})
    salvar(pd.DataFrame(lin), "A1_razoes_pre_pandemia_x_recente")
    # razões por faixa de 5 anos (figura 2), fim 2017–19
    fx5 = {F.FAIXAS[f]: [f] for f in range(3, F.NF)}
    lin = []
    for nome, m in defs[:2]:
        for g, v in razoes(c[m & (c.fx >= 0)], pa, A0, PRE, fx5).items():
            lin.append({"definicao": nome, "faixa": g, "n_2003_05": v[0], "n_2017_19": v[1], "razao": v[2], "ic95_inf": v[3], "ic95_sup": v[4]})
    salvar(pd.DataFrame(lin), "A4_razoes_5anos_2017_19")
    # mortalidade e cirurgia nas faixas do TabNet, Brasil e regiões, fim 2017–19
    popg = pop.assign(g=pop.faixa.map(P.POPF))
    m = pd.read_csv(os.path.join(AQUI, "sim", "tabnet_obitos_reumaticos_2000_2024.csv"))
    m = m[(m.causa == "I00–I09") & m.faixa_tabnet.isin(P.MAPA)].assign(g=lambda x: x.faixa_tabnet.map(P.MAPA))
    cc = c[c.reum_ampliada & (c.fx >= 0)].assign(g=lambda x: x.fx.map(P.POPF))
    lin, gr = [], []
    for reg in REGS:
        pr = (popg if reg == "Brasil" else popg[popg.regiao == reg]).groupby(["ano", "g"])["pop"].sum()
        for med, k in [("cirurgia", (cc if reg == "Brasil" else cc[cc.regiao_res == reg]).groupby(["ano", "g"]).size()),
                       ("obito", (m if reg == "Brasil" else m[m.regiao == reg]).groupby(["ano", "g"]).obitos.sum())]:
            res = {}
            for gn, gs in P.G3.items():
                n1, n2 = sum(k.get((a, g), 0) for a in A0 for g in gs), sum(k.get((a, g), 0) for a in PRE for g in gs)
                p1, p2 = sum(pr.loc[(a, g)] for a in A0 for g in gs), sum(pr.loc[(a, g)] for a in PRE for g in gs)
                res[gn] = rr(n1, p1, n2, p2)
                lin.append({"regiao": reg, "medida": med, "grupo": gn, "n_2003_05": n1, "n_2017_19": n2, "razao": res[gn][0],
                            "ic95_inf": res[gn][1], "ic95_sup": res[gn][2]})
            y, o = res["15-39"], res["60+"]
            r, se = y[0] / o[0], np.sqrt(y[3] ** 2 + o[3] ** 2)
            gr.append({"regiao": reg, "medida": med, "gradiente_jovens_vs_60mais": r, "ic95_inf": r * np.exp(-1.96 * se), "ic95_sup": r * np.exp(1.96 * se)})
    salvar(pd.DataFrame(lin), "A2_cirurgia_obito_2017_19_x_2003_05")
    salvar(pd.DataFrame(gr), "A3_gradiente_cirurgia_obito_pre_pandemia")


# ------------------------------------------------------------------ B
def codificacao(c, pop):
    pa = pop.groupby(["ano", "faixa"])["pop"].sum()
    lin = []
    for rot, m in [("Estrita (I05–I09)", c.reum), ("Ampliada", c.reum_ampliada)]:
        for a2, r2 in [(PRE, "2017–19"), (FIM, "2023–25")]:
            for g, v in razoes(c[m & (c.fx >= 0)], pa, [2003], a2, GRUPOS).items():
                lin.append({"definicao": rot, "inicio": "2003 (antes do Q22.5 coringa)", "fim": r2, "faixa": g, "n_inicio": v[0],
                            "n_fim": v[1], "razao": v[2], "ic95_inf": v[3], "ic95_sup": v[4]})
    salvar(pd.DataFrame(lin), "B1_razoes_inicio_2003_sem_coringa")
    lin = []
    for padrao_ano in (2010, 2022):
        padrao = pa.loc[padrao_ano]
        for a in (2003, 2019, 2024, 2025):
            k = c[c.reum_ampliada & (c.fx >= 0) & (c.ano == a)].groupby("fx").size()
            t, v = J.padronizada(k, pa.loc[a], padrao, list(range(F.NF)))
            lin.append({"populacao_padrao": padrao_ano, "ano": a, "taxa_padronizada": t})
    t = pd.DataFrame(lin)
    t["queda_pct_vs_2003"] = t.groupby("populacao_padrao").taxa_padronizada.transform(lambda s: 100 * (s / s.iloc[0] - 1))
    salvar(t, "B2_padrao_alternativo_2022")


# ------------------------------------------------------------------ C
def perfil_sp(pop):
    b = pd.read_csv(os.path.join(AQUI, "base_valvar_2008_2025.csv.gz"), dtype={"PROC_REA": str, "DIAG_PRINC": str, "atos_valva": str})
    x = b[b.cirurgia & b.reum_ampliada].copy()
    av = x.atos_valva.fillna("")
    x["multivalvar"] = x.PROC_REA.eq("0406010820") | av.str.contains("0406010820") | av.str.contains(r"\+") | (x.n_protese >= 2)
    x["rvm"] = x.tipo.eq("Valvar + RVM") | av.str.contains("0406011206|0406010811")
    x["anel"] = x.anel > 0
    x["sem_protese"] = x.n_protese.eq(0)
    x["troca"] = (x.mec > 0) | (x.bio > 0)
    x["mecanica"] = x.mec > 0
    x["registro_multiplo"] = x.tipo.eq("Múltipla com ato valvar")
    x["gr"] = pd.cut(x.idade, [-1, 14.999, 39.999, 54.999, 200], labels=["<15", "15-39", "40-54", "55+"]).astype(str)
    lin = []
    for per, anos in PER.items():
        g0 = x[x.ano.isin(anos)]
        for est, g in [("Total", g0)] + [(k, g0[g0.gr == k]) for k in ["15-39", "40-54", "55+"]] + \
                      [(r, g0[g0.regiao_res == r]) for r in REGS[1:]] + [("Mulheres 15–44", g0[g0.sexo.eq("F") & g0.idade.between(15, 44.999)])]:
            tr = g[g.troca]
            lin.append({"periodo": per, "estrato": est, "n": len(g), "pct_multivalvar": 100 * g.multivalvar.mean(),
                        "pct_valvar_mais_rvm": 100 * g.rvm.mean(), "pct_anel": 100 * g.anel.mean(),
                        "pct_sem_protese": 100 * g.sem_protese.mean(),
                        "pct_sem_protese_I05": 100 * g[g.DIAG_PRINC.str[:3].eq("I05")].sem_protese.mean(),
                        "pct_mecanica_entre_trocas": 100 * tr.mecanica.mean() if len(tr) else np.nan,
                        "pct_registro_multiplo": 100 * g.registro_multiplo.mean(), "mort_hosp_pct": 100 * g.obito.mean()})
    salvar(pd.DataFrame(lin), "C1_perfil_cirurgico_periodo_idade_regiao")
    t = pd.read_csv(os.path.join(AQUI, "resultados_fase7", "4a_cirurgias_com_codigo_de_protese_previa.csv"))
    RES["C_reoperacao_codigo_protese_previa_pct_2023_2025_todas_valvares"] = round(float(t[t.ano.between(2023, 2025)].com_codigo_protese_previa.sum()
                                                                                         / t[t.ano.between(2023, 2025)].cirurgias.sum() * 100), 1)
    return x


# ------------------------------------------------------------------ D
def valvuloplastia(pop):
    b = pd.read_csv(os.path.join(AQUI, "base_valvar_2008_2025.csv.gz"), dtype={"DIAG_PRINC": str})
    v = b[b.tipo.eq("Valvuloplastia mitral percutânea")].copy()
    v["gr"] = pd.cut(v.idade, [-1, 14.999, 39.999, 54.999, 200], labels=["<15", "15-39", "40-54", "55+"]).astype(str)
    s = v.groupby("ano").agg(total=("N_AIH", "size"), cid_I05=("DIAG_PRINC", lambda d: d.str[:3].eq("I05").sum()),
                             idade_mediana=("idade", "median"), pct_mulheres=("sexo", lambda d: 100 * d.eq("F").mean())).reset_index()
    for g in ["<15", "15-39", "40-54", "55+"]:
        s[f"n_{g}"] = s.ano.map(v[v.gr == g].groupby("ano").size()).fillna(0).astype(int)
    for r in REGS[1:]:
        s[f"n_{r}"] = s.ano.map(v[v.regiao_res == r].groupby("ano").size()).fillna(0).astype(int)
    salvar(s, "D1_valvuloplastia_mitral_percutanea_ano")
    # razão valvuloplastia ÷ cirurgia mitral reumática (I05) por faixa, 2008–10 × 2023–25
    c = b[b.cirurgia & b.DIAG_PRINC.str[:3].eq("I05")].copy()
    c["gr"] = pd.cut(c.idade, [-1, 14.999, 39.999, 54.999, 200], labels=["<15", "15-39", "40-54", "55+"]).astype(str)
    lin = []
    for per, anos in [("2008–10", [2008, 2009, 2010]), ("2023–25", FIM)]:
        for g in ["15-39", "40-54", "55+"]:
            nv = int(((v.ano.isin(anos)) & (v.gr == g)).sum())
            nc = int(((c.ano.isin(anos)) & (c.gr == g)).sum())
            lin.append({"periodo": per, "faixa": g, "valvuloplastias": nv, "cirurgias_I05": nc, "pct_valvuloplastia": 100 * nv / (nv + nc)})
    salvar(pd.DataFrame(lin), "D2_valvuloplastia_x_cirurgia_mitral")


# ------------------------------------------------------------------ E
def mortalidade_hosp(x08):
    a = pd.read_csv(os.path.join(AQUI, "base_valvar_2003_2007.csv.gz"), dtype={"CNES": str})
    a = a[a.reum_ampliada]
    b = pd.read_csv(os.path.join(AQUI, "base_valvar_2008_2025.csv.gz"), dtype={"CNES": str})
    b = b[b.cirurgia & b.reum_ampliada]
    cols = ["ano", "idade", "sexo", "regiao_res", "obito", "CNES"]
    d = pd.concat([a[cols], b[cols]], ignore_index=True)
    d = d[d.sexo.isin(["M", "F"]) & d.idade.between(0, 100)].copy()
    d["CNES"] = d.CNES.fillna("NA")
    d["t5"] = (d.ano - 2003) / 5
    lin = []
    for rot, f_ in [("bruta", "obito ~ t5"), ("idade+sexo+região", "obito ~ t5 + bs(idade, df=4) + C(sexo) + C(regiao_res)")]:
        m = smf.glm(f_, data=d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.CNES)[0]})
        ci = m.conf_int().loc["t5"]
        lin.append({"periodo": "2003–2025", "ajuste": rot, "OR_por_5_anos": np.exp(m.params["t5"]), "ic95_inf": np.exp(ci[0]), "ic95_sup": np.exp(ci[1]), "n": len(d)})
    y = x08[x08.sexo.isin(["M", "F"])].copy()
    y["t5"] = (y.ano - 2008) / 5
    y["CNES"] = y.CNES.fillna("NA").astype(str)
    f_ = "obito ~ t5 + bs(idade, df=4) + C(sexo) + C(regiao_res) + multivalvar + rvm + urgencia + registro_multiplo"
    m = smf.glm(f_, data=y.astype({"multivalvar": int, "rvm": int, "registro_multiplo": int}), family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(y.CNES)[0]})
    ci = m.conf_int().loc["t5"]
    lin.append({"periodo": "2008–2025", "ajuste": "idade+sexo+região+multivalvar+RVM+urgência+registro múltiplo", "OR_por_5_anos": np.exp(m.params["t5"]),
                "ic95_inf": np.exp(ci[0]), "ic95_sup": np.exp(ci[1]), "n": len(y)})
    salvar(pd.DataFrame(lin), "E1_mortalidade_hospitalar_tendencia")
    salvar(d.groupby("ano").obito.agg(["size", "mean"]).rename(columns={"size": "n", "mean": "mort"}).reset_index().assign(mort=lambda t: 100 * t.mort),
           "E2_mortalidade_hospitalar_bruta_ano")


# ------------------------------------------------------------------ F
def acesso():
    a = pd.read_csv(os.path.join(AQUI, "base_valvar_2003_2007.csv.gz"), dtype={"CNES": str})
    b = pd.read_csv(os.path.join(AQUI, "base_valvar_2008_2025.csv.gz"), dtype={"CNES": str})
    b = b[b.cirurgia]
    cols = ["ano", "uf_res", "uf_int", "regiao_res", "CNES", "reum_ampliada"]
    d = pd.concat([a[cols], b[cols]], ignore_index=True)
    d["regiao_int"] = d.uf_int.map(F.UF_REG)
    lin, vol = [], []
    for per, anos in {"2003–05": A0, **PER}.items():
        g = d[d.ano.isin(anos)]
        r = g[g.reum_ampliada]
        for reg in REGS:
            rr_ = r if reg == "Brasil" else r[r.regiao_res == reg]
            lin.append({"periodo": per, "regiao_residencia": reg, "n_reum": len(rr_), "pct_fora_da_uf": 100 * (rr_.uf_int != rr_.uf_res).mean(),
                        "pct_fora_da_regiao": 100 * (rr_.regiao_int != rr_.regiao_res).mean()})
            gg = g if reg == "Brasil" else g[g.regiao_int == reg]
            h = gg.groupby("CNES").size() / len(anos)
            hr = gg[gg.reum_ampliada].groupby("CNES").size() / len(anos)
            vol.append({"periodo": per, "regiao_hospital": reg, "hospitais": int((h > 0).sum()), "hospitais_20mais_ano": int((h >= 20).sum()),
                        "volume_valvar_mediano_por_hospital": float(h.median()), "pct_cirurgias_em_hospitais_100mais": 100 * h[h >= 100].sum() / h.sum(),
                        "volume_reum_mediano_por_hospital": float(hr.median()) if len(hr) else np.nan})
    salvar(pd.DataFrame(lin), "F1_operados_fora_da_residencia")
    salvar(pd.DataFrame(vol), "F2_volume_por_hospital_regiao")


def main():
    pop = F.populacao()
    c = L.montar()
    pre_pandemia(c, pop)
    print("ok: A", flush=True)
    codificacao(c, pop)
    print("ok: B", flush=True)
    x08 = perfil_sp(pop)
    print("ok: C", flush=True)
    valvuloplastia(pop)
    print("ok: D", flush=True)
    mortalidade_hosp(x08)
    print("ok: E", flush=True)
    acesso()
    print("ok: F", flush=True)
    with pd.ExcelWriter(os.path.join(AQUI, "fase10_revisao.xlsx")) as w:
        for k, v in T.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    json.dump(RES, open(os.path.join(OUT, "resumo.json"), "w"), indent=1, default=str)
    print(json.dumps(RES, indent=1, default=str))


if __name__ == "__main__":
    main()
