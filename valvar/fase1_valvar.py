#!/usr/bin/env python3
"""Fase 1 — cirurgia valvar por grupo de CID no SIH/SUS, 2000–2025 (ano da internação).

Lê os parciais de valvar/sih_valvar.py (saida_valvar/parciais) e gera tabelas em valvar/resultados_fase1/ e
valvar/fase1_valvar.xlsx. Não altera os dados; uma AIH por N_AIH (IDENT 5 = longa permanência descartada).

Uso: python valvar/fase1_valvar.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
from sih_valvar import PROCS_VALVA  # noqa: E402

PARC = os.path.join(RAIZ, "saida_valvar", "parciais")
REF = os.path.join(RAIZ, "artigo", "dados_ref")
OUT = os.path.join(AQUI, "resultados_fase1")
os.makedirs(OUT, exist_ok=True)
ANOS = range(2000, 2026)

# Tipo de procedimento (equivalência entre tabelas: piloto_codigos.py)
TIPO = {}
for c in ("0406010692", "48010138", "32020015"):
    TIPO[c] = "Prótese (troca) única"
for c in ("0406010803", "48010120", "32026013"):
    TIPO[c] = "Plástica valvar"
for c in ("0406010820", "48010146", "32041012", "32019017"):
    TIPO[c] = "Plástica e/ou troca múltipla"
for c in ("0406011206", "48010154", "32042019", "0406010811", "48011185"):
    TIPO[c] = "Valvar + RVM"
for c in ("0406010340", "0406010358", "0406010552", "0406010846"):
    TIPO[c] = "Outras valvares abertas"
for c in ("0406030120", "48030120"):
    TIPO[c] = "Valvuloplastia mitral percutânea"
for c in ("0406030111", "0406030138", "0406030146", "48030040", "48030058", "32024010"):
    TIPO[c] = "Outras valvuloplastias percutâneas"
TIPO["0406030162"] = "TAVI"
ABERTA = ["Prótese (troca) única", "Plástica valvar", "Plástica e/ou troca múltipla", "Valvar + RVM",
          "Outras valvares abertas"]
# Cirurgias múltiplas/sequenciais: o procedimento valvar fica só no SIH-SP (análise de sensibilidade)
MULTIPLAS = {"31000002", "0415010012", "0415020034", "0415010020", "0415020018", "0415020026"}

UF_IBGE = {"11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO", "21": "MA", "22": "PI",
           "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE", "29": "BA", "31": "MG", "32": "ES",
           "33": "RJ", "35": "SP", "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF"}
REGIAO = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}


def grupo_cid(c):
    c3 = c.str[:3]
    g = pd.Series("Outro", index=c.index)
    g[c3.isin(["I05", "I06", "I07", "I08", "I09"])] = "Reumática"
    g[c3.isin(["I00", "I01", "I02"])] = "Febre reumática aguda"
    g[c3.isin(["I34", "I35", "I36", "I37"])] = "Não reumática"
    g[c3.isin(["I38", "I39"])] = "Inespecífica (I38/I39)"
    g[c3.eq("I33")] = "Endocardite"
    g[c3.isin(["Q22", "Q23"])] = "Congênita"
    g[c3.isin(["I20", "I21", "I22", "I23", "I24", "I25"])] = "Isquêmica"
    return g


def valva_cid(c):
    c3 = c.str[:3]
    m = {"I05": "Mitral", "I34": "Mitral", "I06": "Aórtica", "I35": "Aórtica", "I07": "Tricúspide", "I36": "Tricúspide",
         "I08": "Múltipla", "I37": "Pulmonar", "Q22": "Pulmonar/tricúspide congênita", "Q23": "Aórtica/mitral congênita"}
    return c3.map(m).fillna("Não especificada")


def carregar():
    fs = sorted(glob.glob(os.path.join(PARC, "RD*.csv")))
    d = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in fs], ignore_index=True)
    for c in d.columns:
        d[c] = d[c].str.strip()
    n0 = len(d)
    d = d[d.IDENT != "5"]
    n5 = n0 - len(d)
    dup = d.duplicated("N_AIH", keep="first")
    d = d[~dup].copy()
    d["dt"] = pd.to_datetime(d.DT_INTER, format="%Y%m%d", errors="coerce")
    d["ano"] = d.dt.dt.year
    d = d[d.ano.between(2000, 2025)].copy()
    d["ano"] = d.ano.astype(int)
    d["tipo"] = d.PROC_REA.map(TIPO)
    d["multipla"] = d.PROC_REA.isin(MULTIPLAS)
    d["grupo"] = grupo_cid(d.DIAG_PRINC)
    d["valva"] = valva_cid(d.DIAG_PRINC)
    d["cid3"] = d.DIAG_PRINC.str[:3]
    idade = pd.to_numeric(d.IDADE, errors="coerce")
    d["idade"] = np.where(d.COD_IDADE.eq("4"), idade, np.where(d.COD_IDADE.eq("5"), idade + 100,
                          np.where(d.COD_IDADE.isin(["0", "1", "2", "3"]), 0, np.nan)))
    d["fem"] = d.SEXO.isin(["2", "3"]).astype(float)
    d.loc[~d.SEXO.isin(["1", "2", "3"]), "fem"] = np.nan
    d["obito"] = pd.to_numeric(d.MORTE, errors="coerce")
    d["regiao_res"] = d.MUNIC_RES.str[:1].map(REGIAO)
    d["uf_res"] = d.MUNIC_RES.str[:2].map(UF_IBGE)
    d["regiao_int"] = d.MUNIC_MOV.str[:1].map(REGIAO)
    sec = [f"DIAGSEC{i}" for i in range(1, 10)] + ["DIAG_SECUN"]
    s = d[sec].apply(lambda col: col.str[:3])
    d["sec_reum"] = s.isin(["I05", "I06", "I07", "I08", "I09"]).any(axis=1)
    d["sec_algum"] = (s.ne("") & s.ne("000")).any(axis=1)
    info = {"registros_lidos": n0, "ident5": n5, "duplicados_N_AIH": int(dup.sum()), "no_recorte_2000_2025": len(d)}
    return d, info


def populacao():
    """2000: Censo; 2001–2006: estimativas IBGE (6579); 2007: interpolada; 2008–2025: projeção 2018 corrigida
    pelas estimativas 2024/2025 (mesma regra de artigo/analise_cec.py)."""
    COD = {int(k): v for k, v in UF_IBGE.items()}
    est = {(COD[int(r["D1C"])], int(r["D3C"])): float(r["V"]) for r in json.load(open(os.path.join(REF, "pop_6579.json")))[1:]
           if r["V"] not in ("-", "...") and int(r["D1C"]) in COD}
    cen = {(COD[int(r["D1C"])], int(r["D3C"])): float(r["V"]) for r in json.load(open(os.path.join(REF, "censo_1286.json")))[1:]
           if r["V"] not in ("-", "...") and int(r["D1C"]) in COD}
    p = pd.read_csv(os.path.join(REF, "populacao_uf_ano_faixa.csv"))
    tot = p.groupby(["uf", "ano"])["pop"].sum()
    linhas = []
    for uf in COD.values():
        r24, r25 = est[(uf, 2024)] / tot[(uf, 2024)], est[(uf, 2025)] / tot[(uf, 2025)]
        ratio = {2022: 1 + (r24 - 1) / 3, 2023: 1 + 2 * (r24 - 1) / 3, 2024: r24, 2025: r25}
        for a in range(2008, 2026):
            linhas.append((uf, a, tot[(uf, a)] * ratio.get(a, 1.0)))
        linhas.append((uf, 2000, cen[(uf, 2000)]))
        for a in range(2001, 2007):
            linhas.append((uf, a, est[(uf, a)]))
        linhas.append((uf, 2007, (est[(uf, 2006)] + tot[(uf, 2008)]) / 2))
    q = pd.DataFrame(linhas, columns=["uf", "ano", "pop"])
    reg = {v: REGIAO[k[0]] for k, v in UF_IBGE.items()}
    q["regiao"] = q.uf.map(reg)
    return q


def resumo(g):
    return pd.Series({"n": len(g), "idade_media": g.idade.mean(), "idade_mediana": g.idade.median(),
                      "idade_p25": g.idade.quantile(.25), "idade_p75": g.idade.quantile(.75),
                      "pct_fem": 100 * g.fem.mean(), "obitos": g.obito.sum(), "mort_pct": 100 * g.obito.mean()})


def main():
    d, info = carregar()
    pop = populacao()
    popbr = pop.groupby("ano")["pop"].sum()
    T = {}
    ab = d[d.tipo.isin(ABERTA)]
    info["cirurgia_aberta_total"] = len(ab)

    # 1. Contagem por ano × grupo de CID (cirurgia aberta)
    t = ab.pivot_table(index="ano", columns="grupo", values="N_AIH", aggfunc="size", fill_value=0)
    t["Total"] = t.sum(axis=1)
    for g in ["Reumática", "Não reumática", "Endocardite", "Total"]:
        t[f"taxa_100mil_{g}"] = 1e5 * t[g] / popbr.reindex(t.index)
    t["pct_reumatica"] = 100 * t["Reumática"] / t["Total"]
    t["pct_inespecifica"] = 100 * t.get("Inespecífica (I38/I39)", 0) / t["Total"]
    t["pct_outro"] = 100 * t.get("Outro", 0) / t["Total"]
    t["populacao"] = popbr.reindex(t.index)
    T["1_ano_grupo"] = t.reset_index()

    # 2. Idade, sexo e mortalidade por ano × grupo
    gr = ab[ab.grupo.isin(["Reumática", "Não reumática", "Endocardite"])]
    T["2_ano_grupo_idade_sexo"] = gr.groupby(["grupo", "ano"]).apply(resumo, include_groups=False).reset_index()

    # 3. CID de 3 caracteres por ano
    c = ab.pivot_table(index="cid3", columns="ano", values="N_AIH", aggfunc="size", fill_value=0)
    c = c.loc[c.sum(axis=1).sort_values(ascending=False).index]
    T["3_cid3_ano"] = c.reset_index()

    # 4. Tipo de procedimento × ano (todas as AIH com código valvar)
    T["4_tipo_ano"] = d[d.tipo.notna()].pivot_table(index="ano", columns="tipo", values="N_AIH", aggfunc="size",
                                                    fill_value=0).reset_index()
    # 4b. por código e tabela
    T["4b_codigo_ano"] = d[d.tipo.notna()].pivot_table(index=["PROC_REA"], columns="ano", values="N_AIH",
                                                       aggfunc="size", fill_value=0).reset_index()
    T["4b_codigo_ano"].insert(1, "nome", T["4b_codigo_ano"].PROC_REA.map(PROCS_VALVA))

    # 5. Mitral reumática (I05): plástica × prótese × percutânea; idem I34
    mit = d[d.cid3.isin(["I05", "I34"]) & d.tipo.isin(["Prótese (troca) única", "Plástica valvar",
                                                        "Valvuloplastia mitral percutânea"])]
    m = mit.pivot_table(index=["cid3", "ano"], columns="tipo", values="N_AIH", aggfunc="size", fill_value=0)
    m["pct_plastica_entre_cirurgicas"] = 100 * m["Plástica valvar"] / (m["Plástica valvar"] + m["Prótese (troca) única"])
    T["5_mitral_plastica_troca"] = m.reset_index()

    # 6. Região de residência × ano: reumática (n, taxa) e % reumática
    r = ab.pivot_table(index=["regiao_res", "ano"], columns="grupo", values="N_AIH", aggfunc="size", fill_value=0)
    r["Total"] = r.sum(axis=1)
    r["pct_reumatica"] = 100 * r["Reumática"] / r["Total"]
    pr = pop.groupby(["regiao", "ano"])["pop"].sum()
    r["taxa_reum_100mil"] = 1e5 * r["Reumática"] / pr.reindex(r.index).to_numpy()
    r["taxa_naoreum_100mil"] = 1e5 * r["Não reumática"] / pr.reindex(r.index).to_numpy()
    T["6_regiao_res_ano"] = r.reset_index()

    # 7. Coerência: idade por CID; CID secundário reumático; mês a mês nas trocas de tabela
    T["7a_idade_por_cid3"] = ab.groupby("cid3").apply(resumo, include_groups=False).sort_values("n", ascending=False).head(25).reset_index()
    sec = ab[ab.ano >= 2008].groupby(["ano", "grupo"]).agg(n=("N_AIH", "size"), com_sec=("sec_algum", "mean"),
                                                           sec_reum=("sec_reum", "mean")).reset_index()
    T["7b_cid_secundario"] = sec
    mm = ab.assign(mes=ab.dt.dt.to_period("M").astype(str))
    mm = mm[mm.ano.isin([2003, 2004, 2005, 2007, 2008])]
    T["7c_mensal_trocas_tabela"] = mm.pivot_table(index="mes", columns="grupo", values="N_AIH", aggfunc="size",
                                                  fill_value=0).reset_index()
    # 7d. faixa etária × grupo × período
    ab2 = ab.assign(faixa=pd.cut(ab.idade, [-1, 14, 29, 44, 59, 74, 200], labels=["0-14", "15-29", "30-44", "45-59", "60-74", "75+"]),
                    periodo=pd.cut(ab.ano, [1999, 2004, 2009, 2014, 2019, 2025], labels=["2000-04", "2005-09", "2010-14", "2015-19", "2020-25"]))
    T["7d_faixa_periodo_grupo"] = ab2.pivot_table(index=["grupo", "periodo"], columns="faixa", values="N_AIH",
                                                  aggfunc="size", fill_value=0, observed=False).reset_index()

    # 8. Sensibilidade: cirurgias múltiplas/sequenciais com CID valvar (procedimento valvar invisível no RD)
    mu = d[d.multipla & d.grupo.isin(["Reumática", "Não reumática", "Endocardite", "Inespecífica (I38/I39)", "Congênita"])]
    T["8_multiplas_cid_valvar"] = mu.pivot_table(index="ano", columns="grupo", values="N_AIH", aggfunc="size",
                                                 fill_value=0).reset_index()
    # 9. Febre reumática aguda e doença reumática crônica: todas as internações (qualquer procedimento)
    T["9_internacoes_I00_I09"] = d[d.grupo.isin(["Febre reumática aguda", "Reumática"])].pivot_table(
        index="ano", columns="grupo", values="N_AIH", aggfunc="size", fill_value=0).reset_index()
    # 9b. Febre reumática aguda (I00–I02) em 5–19 anos: a partir de 2008 a maioria das AIH I00–I02 é de adultos
    #     (I01 em tratamento clínico; idade mediana 14–18 → 43–52 anos), então a faixa 5–19 é o indicador limpo
    fra = d[d.grupo.eq("Febre reumática aguda")]
    T["9b_febre_reumatica_5a19"] = fra[fra.idade.between(5, 19)].pivot_table(
        index="ano", columns="cid3", values="N_AIH", aggfunc="size", fill_value=0).assign(
        total=lambda x: x.sum(axis=1)).reset_index()
    T["9c_febre_reumatica_idade"] = fra.groupby("ano").agg(n=("N_AIH", "size"), idade_mediana=("idade", "median"),
                                                          pct_5a19=("idade", lambda v: 100 * v.between(5, 19).mean())).reset_index()
    # 10. Valva (pelo CID) × grupo, cirurgia aberta, por período
    T["10_valva_grupo"] = ab.pivot_table(index=["grupo", "valva"], columns=pd.cut(ab.ano, [1999, 2007, 2015, 2025],
                                         labels=["2000-07", "2008-15", "2016-25"]), values="N_AIH", aggfunc="size",
                                         fill_value=0, observed=False).reset_index()
    # 11. Série combinada: cirurgia aberta + cirurgias múltiplas/sequenciais (0415010012, 0415020034) com o mesmo CID
    #     (piloto SIH-SP: 89% dessas AIH têm ato valvar; piloto_sp_multiplas.py)
    mu2 = d[d.PROC_REA.isin(["0415010012", "0415020034", "31000002"])]
    comb = pd.concat([ab.assign(fonte="aberta"), mu2.assign(fonte="multipla")])
    k = comb.groupby(["ano", "grupo", "fonte"]).size()
    out = pd.DataFrame(index=pd.Index(sorted(comb.ano.unique()), name="ano"))
    for g in ["Reumática", "Não reumática"]:
        for f in ["aberta", "multipla"]:
            out[f"{g}_{f}"] = [int(k.get((a, g, f), 0)) for a in out.index]
        out[f"{g}_soma"] = out[f"{g}_aberta"] + out[f"{g}_multipla"]
        out[f"{g}_taxa_100mil"] = 1e5 * out[f"{g}_soma"] / popbr.reindex(out.index)
    out["I34_soma"] = comb[comb.cid3.eq("I34")].groupby("ano").size().reindex(out.index, fill_value=0)
    out["pct_reum_soma"] = 100 * out["Reumática_soma"] / (out["Reumática_soma"] + out["Não reumática_soma"])
    T["11_combinada_aberta_multiplas"] = out.reset_index()
    T["11b_idade_combinada"] = comb[comb.grupo.isin(["Reumática", "Não reumática"])].groupby(["grupo", "ano"]).apply(
        resumo, include_groups=False).reset_index()

    T["0_info"] = pd.DataFrame([info]).T.reset_index().rename(columns={"index": "item", 0: "valor"})

    for k, v in T.items():
        v.to_csv(os.path.join(OUT, f"{k}.csv"), index=False)
    with pd.ExcelWriter(os.path.join(AQUI, "fase1_valvar.xlsx")) as w:
        for k, v in T.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    print(json.dumps(info, indent=1))
    print(T["1_ano_grupo"][["ano", "Reumática", "Não reumática", "Endocardite", "Total", "pct_reumatica",
                            "taxa_100mil_Reumática", "taxa_100mil_Não reumática"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
