#!/usr/bin/env python3
"""Fase 7 — quatro análises adicionais.

1. Triangulação cirurgia × mortalidade por geração e região: razão por idade 2022–24 ÷ 2003–05 nas duas medidas
   (cirurgia reumática ampliada, SIH; óbito com causa básica I00–I09, SIM/TabNet), faixas 15–39, 40–59, 60+, Brasil e
   regiões; modelo de Poisson (escala de Pearson) com interação medida × período × idade.
2. "Carga não operada": internações com CID principal I05–I09 sem cirurgia valvar nem valvuloplastia, 2008–2025 —
   número, taxa padronizada, idade, região, óbito hospitalar.
3. Mulheres em idade fértil (15–49) e morte materna: óbitos de mulheres 15–49 com causa básica I00–I09 (TabNet,
   2000–2024) e óbitos maternos (causa básica capítulo O) com doença reumática em qualquer linha (microdados SIM 2016–2024).
4. Disfunção de prótese: cirurgias valvares e TAVI com CID principal T82 ou Z95.2/Z95.4 (presença de prótese valvar),
   2008–2025, como indicador indireto de reoperação (o CID passou a ser aceito em ~2015).

Saídas: valvar/resultados_fase7/*.csv e valvar/fase7_extras.xlsx. Uso: python valvar/fase7_extras.py
"""
import glob
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
import fase2_valvar as F  # noqa: E402
import fase4_revisao2 as J  # noqa: E402
import fase6_serie_longa as L  # noqa: E402

OUT = os.path.join(AQUI, "resultados_fase7")
os.makedirs(OUT, exist_ok=True)
T, RES = {}, {}
REGS = ["Brasil", "Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
G3 = {"15-39": ["15-19", "20-29", "30-39"], "40-59": ["40-49", "50-59"], "60+": ["60-69", "70-79", "80+"]}
POPF = {0: "0-4", 1: "5-9", 2: "10-14", 3: "15-19", 4: "20-29", 5: "20-29", 6: "30-39", 7: "30-39", 8: "40-49", 9: "40-49",
        10: "50-59", 11: "50-59", 12: "60-69", 13: "60-69", 14: "70-79", 15: "70-79", 16: "80+"}
MAPA = {"Menor 1 ano": "0-4", "1 a 4 anos": "0-4", "5 a 9 anos": "5-9", "10 a 14 anos": "10-14", "15 a 19 anos": "15-19",
        "20 a 29 anos": "20-29", "30 a 39 anos": "30-39", "40 a 49 anos": "40-49", "50 a 59 anos": "50-59",
        "60 a 69 anos": "60-69", "70 a 79 anos": "70-79", "80 anos e mais": "80+"}
A1, A2 = (2003, 2004, 2005), (2022, 2023, 2024)


def salvar(df, nome):
    T[nome] = df
    df.to_csv(os.path.join(OUT, nome + ".csv"), index=False)


# ------------------------------------------------------------------ 1
def triangulacao(pop):
    pop = pop.assign(g=pop.faixa.map(POPF))
    pr = pop.groupby(["regiao", "ano", "g"])["pop"].sum()
    pb = pop.groupby(["ano", "g"])["pop"].sum()
    m = pd.read_csv(os.path.join(AQUI, "sim", "tabnet_obitos_reumaticos_2000_2024.csv"))
    m = m[(m.causa == "I00–I09") & m.faixa_tabnet.isin(MAPA)].assign(g=lambda x: x.faixa_tabnet.map(MAPA))
    c = L.montar()
    c = c[c.reum_ampliada & (c.fx >= 0)].assign(g=lambda x: x.fx.map(POPF))
    rows = []
    for reg in REGS:
        km = (m if reg == "Brasil" else m[m.regiao == reg]).groupby(["ano", "g"]).obitos.sum()
        kc = (c if reg == "Brasil" else c[c.regiao_res == reg]).groupby(["ano", "g"]).size()
        for med, k in [("cirurgia", kc), ("obito", km)]:
            for per, anos in [("ini", A1), ("fim", A2)]:
                for gr, gs in G3.items():
                    n = sum(k.get((a, g), 0) for a in anos for g in gs)
                    p = sum((pb.loc[(a, g)] if reg == "Brasil" else pr.loc[(reg, a, g)]) for a in anos for g in gs)
                    rows.append({"regiao": reg, "medida": med, "per": per, "grupo": gr, "n": n, "pop": p})
    g = pd.DataFrame(rows)
    lin = []
    for (reg, med, gr), h in g.groupby(["regiao", "medida", "grupo"]):
        i, f = h[h.per == "ini"].iloc[0], h[h.per == "fim"].iloc[0]
        rr = (f.n / f["pop"]) / (i.n / i["pop"])
        se = np.sqrt(1 / i.n + 1 / f.n)
        lin.append({"regiao": reg, "medida": med, "grupo": gr, "n_2003_05": i.n, "n_2022_24": f.n, "razao": rr,
                    "ic95_inf": rr * np.exp(-1.96 * se), "ic95_sup": rr * np.exp(1.96 * se)})
    salvar(pd.DataFrame(lin), "1a_cirurgia_x_obito_razoes_por_idade_regiao")
    b = g[g.regiao == "Brasil"]
    m0 = smf.glm("n ~ C(grupo)*medida + per*medida + per*C(grupo)", data=b, family=sm.families.Poisson(), offset=np.log(b["pop"])).fit(scale="X2")
    m1 = smf.glm("n ~ C(grupo)*medida + per*medida*C(grupo)", data=b, family=sm.families.Poisson(), offset=np.log(b["pop"])).fit(scale="X2")
    dfn = m0.df_resid - m1.df_resid
    RES["triangulacao_interacao_medida_periodo_idade"] = ("teste não informativo (modelo saturado)" if m1.df_resid == 0 else
                                                         float(stats.f.sf((m0.deviance - m1.deviance) / dfn / m1.scale, dfn, m1.df_resid)))
    # gradiente: razão jovens ÷ 60+ em cada medida (quanto mais < 1, mais geracional)
    t = pd.DataFrame(lin)
    gr = []
    for (reg, med), h in t.groupby(["regiao", "medida"]):
        h = h.set_index("grupo")
        y, o = h.loc["15-39"], h.loc["60+"]
        r = y.razao / o.razao
        se = np.sqrt(1 / y.n_2003_05 + 1 / y.n_2022_24 + 1 / o.n_2003_05 + 1 / o.n_2022_24)
        gr.append({"regiao": reg, "medida": med, "gradiente_jovens_vs_60mais": r, "ic95_inf": r * np.exp(-1.96 * se),
                   "ic95_sup": r * np.exp(1.96 * se)})
    salvar(pd.DataFrame(gr), "1b_gradiente_geracional_cirurgia_x_obito")


# ------------------------------------------------------------------ 2
def nao_operada(pop):
    from sih_valvar import PROCS_VALVA
    fs = sorted(glob.glob(os.path.join(RAIZ, "saida_valvar", "parciais", "RD*.csv")))
    cols = ["N_AIH", "IDENT", "PROC_REA", "DIAG_PRINC", "DT_INTER", "IDADE", "COD_IDADE", "SEXO", "MUNIC_RES", "MORTE",
            "DIAS_PERM", "CAR_INT"]
    d = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, usecols=cols) for f in fs])
    d = d.apply(lambda s: s.str.strip())
    d = d[d.IDENT != "5"].drop_duplicates("N_AIH")
    d["ano"] = pd.to_numeric(d.DT_INTER.str[:4], errors="coerce")
    d = d[d.ano.between(2008, 2025) & d.DIAG_PRINC.str[:3].isin(["I05", "I06", "I07", "I08", "I09"])]
    cir = d.PROC_REA.isin(PROCS_VALVA) | d.PROC_REA.isin(["0415010012", "0415020034", "0415010020", "0415020018"])
    x = d[~cir].copy()
    i = pd.to_numeric(x.IDADE, errors="coerce")
    x["idade"] = np.where(x.COD_IDADE.eq("4"), i, np.where(x.COD_IDADE.eq("5"), 100 + i, np.where(x.COD_IDADE.isin(["0", "1", "2", "3"]), 0, np.nan)))
    x["fx"] = F.faixa5(x.idade).to_numpy()
    x["regiao"] = x.MUNIC_RES.str[:1].map(F.REGIAO)
    x["obito"] = x.MORTE.eq("1")
    x["grupo_proc"] = x.PROC_REA.str[:6].map({"030306": "tratamento clínico cardiovascular", "030301": "tratamento clínico outros"}).fillna("outros")
    pa = pop.groupby(["ano", "faixa"])["pop"].sum()
    pr = pop.groupby(["regiao", "ano", "faixa"])["pop"].sum()
    padrao = pa.loc[2010]
    lin = []
    for reg in REGS:
        xr = x if reg == "Brasil" else x[x.regiao == reg]
        k = xr[xr.fx >= 0].groupby(["ano", "fx"]).size()
        for a in range(2008, 2026):
            den = pa.loc[a] if reg == "Brasil" else pr.loc[(reg, a)]
            t, v = J.padronizada(k.get(a, pd.Series(dtype=float)), den, padrao, list(range(F.NF)))
            h = xr[xr.ano == a]
            lin.append({"regiao": reg, "ano": a, "internacoes": len(h), "taxa_padronizada": t, "idade_mediana": h.idade.median(),
                        "pct_60mais": 100 * (h.idade >= 60).mean(), "obitos_hosp": int(h.obito.sum()),
                        "mort_hosp_pct": 100 * h.obito.mean(), "pct_urgencia": 100 * h.CAR_INT.isin(["02", "03", "04", "05", "06"]).mean()})
    salvar(pd.DataFrame(lin), "2a_internacoes_reumaticas_nao_cirurgicas")
    top = x.groupby("PROC_REA").size().sort_values(ascending=False).head(10)
    nm = pd.read_csv(os.path.join(AQUI, "ref", "sigtap_nomes_microdatasus2019.csv"), dtype=str).set_index("cod").nome
    salvar(pd.DataFrame({"PROC_REA": top.index, "n": top.values, "nome": [nm.get(k, "?") for k in top.index]}), "2b_procedimentos_nao_cirurgicos_top")


# ------------------------------------------------------------------ 3
def tabnet(ano, ids, sexo):
    url = "http://tabnet.datasus.gov.br/cgi/tabcgi.exe?sim/cnv/obt10uf.def"
    todas = ["SRegião", "SUnidade_da_Federação", "SCapítulo_CID-10", "SGrupo_CID-10", "SCausa_-_CID-BR-10",
             "SCausa_mal_definidas", "SFaixa_Etária", "SFaixa_Etária_OPS", "SFaixa_Etária_det", "SFx.Etária_Menor_1A",
             "SCor/raça", "SEscolaridade", "SEstado_civil", "SLocal_ocorrência"]
    campos = [("Linha", "Região"), ("Coluna", "Faixa_Etária"), ("Incremento", "Óbitos_p/Residênc"),
              ("Arquivos", f"obtuf{ano % 100:02d}.dbf"), ("SSexo", sexo)] + [(s, "TODAS_AS_CATEGORIAS__") for s in todas]
    campos += [("SCategoria_CID-10", i) for i in ids] + [("formato", "prn"), ("mostre", "Mostra")]
    body = urllib.parse.urlencode(campos, encoding="latin-1").encode()
    for t in range(4):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=180).read().decode("latin-1")
            pre = html.unescape(re.search(r"<PRE>(.*?)</PRE>", r, re.S | re.I).group(1))
            break
        except Exception:
            time.sleep(2 ** (t + 1))
    linhas = [l.split(";") for l in pre.strip().split("\n") if l.startswith('"')]
    cab = [c.strip('"') for c in linhas[0]]
    out = []
    for l in linhas[1:]:
        nome = l[0].strip('"')
        if nome == "Total":
            continue
        for c, v in zip(cab[1:-1], l[1:-1]):
            out.append({"ano": ano, "regiao": F.REGIAO[nome.split()[0]], "faixa_tabnet": c, "obitos": 0 if v.strip() in ("-", "") else int(v)})
    return out


def mulheres(pop):
    p = os.path.join(AQUI, "sim", "tabnet_obitos_reumaticos_mulheres_2000_2024.csv")
    if not os.path.exists(p):
        lin = []
        for ano in range(2000, 2025):
            lin += tabnet(ano, ["635", "636", "637", "638", "639", "640", "641", "642"], "2")
            time.sleep(0.5)
        pd.DataFrame(lin).to_csv(p, index=False)
    d = pd.read_csv(p)
    d = d[d.faixa_tabnet.isin(["15 a 19 anos", "20 a 29 anos", "30 a 39 anos", "40 a 49 anos"])]
    pf = pop[(pop.sexo == "F") & pop.faixa.between(3, 9)]
    pb, pr = pf.groupby("ano")["pop"].sum(), pf.groupby(["regiao", "ano"])["pop"].sum()
    lin = []
    for reg in REGS:
        x = d if reg == "Brasil" else d[d.regiao == reg]
        k = x.groupby("ano").obitos.sum()
        for a in range(2000, 2025):
            den = pb.loc[a] if reg == "Brasil" else pr.loc[(reg, a)]
            lin.append({"regiao": reg, "ano": a, "obitos_mulheres_15_49": int(k.get(a, 0)), "taxa_100mil": 1e5 * k.get(a, 0) / den})
    s = pd.DataFrame(lin)
    salvar(s, "3a_obitos_reumaticos_mulheres_15_49")
    b = s[s.regiao == "Brasil"]
    g = b[b.ano.between(2003, 2024)].assign(t=lambda x: x.ano - 2003, pop=lambda x: 1e5 * x.obitos_mulheres_15_49 / x.taxa_100mil)
    m = smf.glm("obitos_mulheres_15_49 ~ t", data=g, family=sm.families.Poisson(), offset=np.log(g["pop"])).fit(scale="X2")
    ci = m.conf_int().loc["t"]
    RES["mulheres_15_49_variacao_anual_pct_2003_2024"] = [round(100 * (np.exp(v) - 1), 1) for v in (m.params["t"], ci[0], ci[1])]
    # morte materna com doença reumática em qualquer linha (microdados 2016–2024)
    mm = pd.read_csv(os.path.join(AQUI, "sim", "sim_reumatica_2016_2024.csv.gz"), dtype=str)
    mm["ano"] = mm.DTOBITO.str[-4:].astype(int)
    mat = mm[mm.CAUSABAS.fillna("").str.startswith("O")]
    salvar(mat.groupby("ano").agg(obitos_maternos_com_doenca_reumatica=("CAUSABAS", "size"),
                                  causa_O994=("CAUSABAS", lambda v: (v.str[:4] == "O994").sum())).reset_index(),
           "3b_obitos_maternos_com_doenca_reumatica")
    RES["obitos_maternos_reumaticos_2016_2024"] = int(len(mat))


# ------------------------------------------------------------------ 4
def protese(pop):
    fs = sorted(glob.glob(os.path.join(RAIZ, "saida_valvar", "parciais", "RD*.csv")))
    cols = ["N_AIH", "IDENT", "PROC_REA", "DIAG_PRINC", "DT_INTER", "IDADE", "COD_IDADE"] + [f"DIAGSEC{i}" for i in range(1, 10)]
    d = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, usecols=cols) for f in fs])
    d = d.apply(lambda s: s.str.strip())
    d = d[d.IDENT != "5"].drop_duplicates("N_AIH")
    d["ano"] = pd.to_numeric(d.DT_INTER.str[:4], errors="coerce")
    p = os.path.join(RAIZ, "saida_valvar", "base_valvar_2008_2025.csv.gz")
    b = pd.read_csv(p if os.path.exists(p) else os.path.join(AQUI, "base_valvar_2008_2025.csv.gz"),
                    dtype={"N_AIH": str}, usecols=["N_AIH", "cirurgia", "tipo", "bio", "mec", "idade"])
    b = b[b.cirurgia | b.tipo.eq("TAVI")]
    x = b.merge(d, on="N_AIH", how="left")
    princ = x.DIAG_PRINC.str[:3].eq("T82") | x.DIAG_PRINC.str[:4].isin(["Z952", "Z954"])
    sec = x[[f"DIAGSEC{i}" for i in range(1, 10)]].apply(lambda c: c.str[:3].eq("T82") | c.str[:4].isin(["Z952", "Z954"])).any(axis=1)
    x["protese_previa"] = princ | sec
    t = x[x.ano.between(2008, 2025)].groupby(["ano"]).agg(cirurgias=("N_AIH", "size"),
                                                           com_codigo_protese_previa=("protese_previa", "sum"),
                                                           T82_principal=("DIAG_PRINC", lambda v: v.str[:3].eq("T82").sum()),
                                                           tavi=("tipo", lambda v: v.eq("TAVI").sum())).reset_index()
    t["pct"] = 100 * t.com_codigo_protese_previa / t.cirurgias
    tv = x[x.tipo.eq("TAVI")]
    t["tavi_valve_in_valve_codigo"] = t.ano.map(tv[tv.protese_previa].groupby("ano").size()).fillna(0).astype(int)
    salvar(t, "4a_cirurgias_com_codigo_de_protese_previa")
    y = x[x.protese_previa & x.ano.between(2015, 2025) & x.cirurgia]
    RES["protese_previa_2015_2025_idade_mediana"] = float(y.idade.median()) if len(y) else None


def main():
    pop = F.populacao()
    for nome, fun in [("triangulacao", triangulacao), ("nao_operada", nao_operada), ("mulheres", mulheres), ("protese", protese)]:
        fun(pop)
        print("ok:", nome, flush=True)
    with pd.ExcelWriter(os.path.join(AQUI, "fase7_extras.xlsx")) as w:
        for k, v in T.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    json.dump(RES, open(os.path.join(OUT, "resumo.json"), "w"), indent=1, default=str)
    print(json.dumps(RES, indent=1, default=str))


if __name__ == "__main__":
    main()
