#!/usr/bin/env python3
"""Fase 2 — cirurgia valvar reumática no SUS, 2008–2025: base analítica (RD + SIH-SP) e análises do artigo.

Entradas: saida_valvar/parciais (RD, sih_valvar.py), saida_valvar/sp_parciais (SP, sih_sp_valvar.py),
artigo/dados_ref (população IBGE). Saídas: valvar/resultados_fase2/*.csv, valvar/fase2_valvar.xlsx,
valvar/resultados_fase2/resumo.json e saida_valvar/base_valvar_2008_2025.csv.gz (uma linha por AIH; CNS do
cirurgião pseudonimizado por hash com sal local, nunca gravado em claro).

Definições:
- Ano = ano da internação (DT_INTER). Uma AIH por N_AIH; IDENT 5 descartada.
- Cirurgia valvar = código valvar aberto (0406010692, 0406010803, 0406010820, 0406011206, 0406010811, 0406010340,
  0406010358, 0406010552, 0406010846) OU cirurgia múltipla/sequencial (0415010012, 0415020034) com CID valvar e ato
  valvar ou prótese/anel no SIH-SP.
- Etiologia pelo CID principal: reumática estrita I05–I09; não reumática I34–I37; sensibilidades: reumática ampliada
  (+ Q22.5 e I38 em ≥ 18 anos) e limite superior (+ I34 em < 45 anos).
- Prótese e anel pelo material (OPM) no SIH-SP: 0702040541/0702040550 biológica, 0702040568/0702040576 mecânica,
  0702040029 anel de anuloplastia.

Uso: python valvar/fase2_valvar.py
"""
import glob
import hashlib
import json
import os
import secrets
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
from fase1_valvar import REGIAO, UF_IBGE, grupo_cid  # noqa: E402

SAIDA = os.path.join(RAIZ, "saida_valvar")
REF = os.path.join(RAIZ, "artigo", "dados_ref")
OUT = os.path.join(AQUI, "resultados_fase2")
os.makedirs(OUT, exist_ok=True)
RES = {}
T = {}

ABERTA = {"0406010692": "Prótese única", "0406010803": "Plástica", "0406010820": "Plástica e/ou troca múltipla",
          "0406011206": "Valvar + RVM", "0406010811": "Valvar + RVM", "0406010340": "Outras abertas",
          "0406010358": "Outras abertas", "0406010552": "Outras abertas", "0406010846": "Outras abertas"}
MULTIPLAS = {"0415010012", "0415020034"}
ATOS_VALVA = set(ABERTA) - {"0406010846"}
OPM = {"0702040541": "bio", "0702040550": "bio", "0702040568": "mec", "0702040576": "mec", "0702040029": "anel",
       "0702040193": "cec", "0702040223": "kit_vmp"}
COD_UF = {int(k): v for k, v in UF_IBGE.items()}
UF_REG = {v: REGIAO[k[0]] for k, v in UF_IBGE.items()}
FAIXAS = ["0 a 4 anos", "5 a 9 anos", "10 a 14 anos", "15 a 19 anos", "20 a 24 anos", "25 a 29 anos", "30 a 34 anos",
          "35 a 39 anos", "40 a 44 anos", "45 a 49 anos", "50 a 54 anos", "55 a 59 anos", "60 a 64 anos",
          "65 a 69 anos", "70 a 74 anos", "75 a 79 anos", "80 anos ou mais"]
NF = len(FAIXAS)  # 17 faixas; 80+ agregado (a projeção não abre 80–84/85+ antes de 2010)


def salvar(df, nome):
    T[nome] = df
    df.to_csv(os.path.join(OUT, nome + ".csv"), index=False)


def faixa5(idade):
    i = np.clip(np.floor(np.asarray(idade, float) / 5), 0, NF - 1)
    return pd.Series(np.where(np.isnan(i), -1, i).astype(int), index=getattr(idade, "index", None))


# ------------------------------------------------------------------ população
def populacao():
    """UF × sexo × faixa de 5 anos × ano, projeção IBGE 2018 (tabela 7358) corrigida pelas estimativas 2024/2025
    (razão por UF aplicada a todas as faixas; 2022–2023 interpolados; mesma regra de artigo/analise_cec.py)."""
    p = pd.DataFrame([{"uf": COD_UF[int(r["D1C"])], "sexo": "M" if r["D4C"] == "4" else "F",
                       "faixa": FAIXAS.index(r["D5N"]), "ano": int(r["D6N"]), "pop": float(r["V"])}
                      for r in json.load(open(os.path.join(REF, "proj7358_sexo_idade_2000_2025.json")))])
    est = {(COD_UF[int(r["D1C"])], int(r["D3C"])): float(r["V"])
           for r in json.load(open(os.path.join(REF, "pop_6579.json")))[1:]
           if r["V"] not in ("-", "...") and int(r["D1C"]) in COD_UF}
    tot = p.groupby(["uf", "ano"])["pop"].sum()
    f = {}
    for uf in COD_UF.values():
        r24, r25 = est[(uf, 2024)] / tot[(uf, 2024)], est[(uf, 2025)] / tot[(uf, 2025)]
        f.update({(uf, 2022): 1 + (r24 - 1) / 3, (uf, 2023): 1 + 2 * (r24 - 1) / 3, (uf, 2024): r24, (uf, 2025): r25})
    p["pop"] *= [f.get((u, a), 1.0) for u, a in zip(p.uf, p.ano)]
    p["regiao"] = p.uf.map(UF_REG)
    return p


# ------------------------------------------------------------------ base
def carregar_rd():
    fs = sorted(glob.glob(os.path.join(SAIDA, "parciais", "RD*.csv")))
    d = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in fs], ignore_index=True)
    for c in d.columns:
        d[c] = d[c].str.strip()
    d = d[d.IDENT != "5"].drop_duplicates("N_AIH")
    d["ano"] = pd.to_numeric(d.DT_INTER.str[:4], errors="coerce")
    d = d[d.ano.between(2008, 2025)].copy()
    d["ano"] = d.ano.astype(int)
    idade = pd.to_numeric(d.IDADE, errors="coerce")
    d["idade"] = np.where(d.COD_IDADE.eq("4"), idade, np.where(d.COD_IDADE.eq("5"), idade + 100,
                          np.where(d.COD_IDADE.isin(["0", "1", "2", "3"]), 0, np.nan)))
    d["sexo"] = np.where(d.SEXO.eq("1"), "M", np.where(d.SEXO.isin(["2", "3"]), "F", ""))
    d["obito"] = pd.to_numeric(d.MORTE, errors="coerce").fillna(0).astype(int)
    d["grupo"] = grupo_cid(d.DIAG_PRINC)
    d["cid3"] = d.DIAG_PRINC.str[:3]
    d["uf_res"] = d.MUNIC_RES.str[:2].map(UF_IBGE)
    d["regiao_res"] = d.MUNIC_RES.str[:1].map(REGIAO)
    d["uf_int"] = d.MUNIC_MOV.str[:2].map(UF_IBGE)
    d["regiao_int"] = d.MUNIC_MOV.str[:1].map(REGIAO)
    d["urgencia"] = d.CAR_INT.isin(["02", "03", "04", "05", "06"]).astype(int)
    d["dias"] = pd.to_numeric(d.DIAS_PERM, errors="coerce")
    d["uti"] = pd.to_numeric(d.UTI_MES_TO, errors="coerce")
    d["val"] = pd.to_numeric(d.VAL_TOT, errors="coerce")
    return d


def carregar_sp(alvo):
    fs = sorted(glob.glob(os.path.join(SAIDA, "sp_parciais", "SP*.csv")))
    cols = ["SP_NAIH", "SP_ATOPROF", "SP_QTD_ATO", "SP_PF_CBO", "SP_PF_DOC"]
    s = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, usecols=cols) for f in fs], ignore_index=True)
    s = s[s.SP_NAIH.isin(alvo)]
    s["q"] = pd.to_numeric(s.SP_QTD_ATO, errors="coerce").fillna(1)
    o = s[s.SP_ATOPROF.isin(OPM)].assign(t=lambda x: x.SP_ATOPROF.map(OPM)).pivot_table(
        index="SP_NAIH", columns="t", values="q", aggfunc="sum", fill_value=0)
    atos = s[s.SP_ATOPROF.isin(ATOS_VALVA)].groupby("SP_NAIH").SP_ATOPROF.apply(lambda v: "+".join(sorted(set(v))))
    # cirurgião: CBO 2252* (médico cirurgião) no ato valvar/cardíaco principal; o mais frequente na AIH
    cir = s[s.SP_PF_CBO.str.startswith("2252") & s.SP_ATOPROF.str.startswith("0406") & s.SP_PF_DOC.ne("")]
    cir = cir.groupby("SP_NAIH").SP_PF_DOC.agg(lambda v: v.value_counts().index[0])
    sal = secrets.token_hex(16)
    cir = cir.map(lambda x: hashlib.sha256((sal + x).encode()).hexdigest()[:12])
    sp = pd.DataFrame(index=pd.Index(sorted(set(s.SP_NAIH)), name="N_AIH"))
    for c in ["bio", "mec", "anel", "cec", "kit_vmp"]:
        sp[c] = o[c].reindex(sp.index).fillna(0) if c in o else 0
    sp["atos_valva"] = atos.reindex(sp.index).fillna("")
    sp["cirurgiao"] = cir.reindex(sp.index).fillna("")
    return sp


def montar():
    d = carregar_rd()
    cid_valva = d.DIAG_PRINC.str.match(r"^(I0[5-9]|I3[3-9]|Q2[23])")
    cand = d[d.PROC_REA.isin(ABERTA) | (d.PROC_REA.isin(MULTIPLAS) & cid_valva)
             | d.PROC_REA.isin(["0406030120", "0406030162"])].copy()
    sp = carregar_sp(set(cand.N_AIH))
    cand = cand.join(sp, on="N_AIH")
    cand["no_sp"] = cand.bio.notna()
    for c in ["bio", "mec", "anel", "cec", "kit_vmp"]:
        cand[c] = cand[c].fillna(0)
    cand["atos_valva"] = cand.atos_valva.fillna("")
    cand["cirurgiao"] = cand.cirurgiao.fillna("")
    cand["n_protese"] = cand.bio + cand.mec
    mult_ok = cand.PROC_REA.isin(MULTIPLAS) & ((cand.atos_valva != "") | (cand.n_protese > 0) | (cand.anel > 0))
    cand["tipo"] = cand.PROC_REA.map(ABERTA)
    cand.loc[mult_ok, "tipo"] = "Múltipla com ato valvar"
    cand.loc[cand.PROC_REA.eq("0406030120"), "tipo"] = "Valvuloplastia mitral percutânea"
    cand.loc[cand.PROC_REA.eq("0406030162"), "tipo"] = "TAVI"
    cand["cirurgia"] = cand.PROC_REA.isin(ABERTA) | mult_ok
    RES["multiplas_cid_valvar"] = int((cand.PROC_REA.isin(MULTIPLAS)).sum())
    RES["multiplas_com_ato_valvar"] = int(mult_ok.sum())
    RES["multiplas_no_sp"] = int((cand.PROC_REA.isin(MULTIPLAS) & cand.no_sp).sum())
    RES["aih_candidatas"] = len(cand)
    RES["aih_candidatas_no_sp_pct"] = round(100 * cand.no_sp.mean(), 2)
    c = cand
    c["reum"] = c.grupo.eq("Reumática")
    c["naoreum"] = c.grupo.eq("Não reumática")
    adulto = c.idade >= 18
    c["reum_ampliada"] = c.reum | ((c.DIAG_PRINC.eq("Q225") | c.cid3.eq("I38")) & adulto)
    c["reum_limite"] = c.reum_ampliada | (c.cid3.eq("I34") & (c.idade < 45))
    c["etiologia"] = np.select([c.reum, c.naoreum], ["Reumática", "Não reumática"], "Outra")
    c["sem_protese"] = (c.n_protese == 0).astype(int)
    c["mecanica"] = (c.mec > 0).astype(int)
    c["fx"] = faixa5(c.idade).to_numpy()
    c["coorte"] = c.ano - c.idade
    c["fora_uf"] = (c.uf_res != c.uf_int).astype(int)
    return c


# ------------------------------------------------------------------ análises
def taxas(c, pop):
    """Contagens, taxa bruta e padronizada por idade (padrão: Brasil 2010, faixas de 5 anos)."""
    cir = c[c.cirurgia]
    pa = pop.groupby(["ano", "faixa"])["pop"].sum()
    padrao = pa.loc[2010] / pa.loc[2010].sum()
    linhas = []
    defs = {"Reumática": cir.reum, "Reumática ampliada": cir.reum_ampliada, "Reumática limite superior": cir.reum_limite,
            "Não reumática": cir.naoreum, "Todas valvares": pd.Series(True, index=cir.index)}
    for nome, m in defs.items():
        x = cir[m & (cir.fx >= 0)]
        k = x.groupby(["ano", "fx"]).size()
        for a in range(2008, 2026):
            ka = k.get(a, pd.Series(dtype=float)).reindex(range(NF), fill_value=0)
            r = ka.to_numpy() / pa.loc[a].to_numpy()
            var = (ka.to_numpy() / pa.loc[a].to_numpy() ** 2 * padrao.to_numpy() ** 2).sum()
            std = (r * padrao.to_numpy()).sum()
            linhas.append({"definicao": nome, "ano": a, "n": int(m[cir.ano == a].sum()),
                           "taxa_bruta": 1e5 * m[cir.ano == a].sum() / pa.loc[a].sum(),
                           "taxa_padronizada": 1e5 * std, "ic95_inf": 1e5 * (std - 1.96 * var ** .5),
                           "ic95_sup": 1e5 * (std + 1.96 * var ** .5),
                           "idade_mediana": x[x.ano == a].idade.median(),
                           "multiplas": int((m & cir.tipo.eq("Múltipla com ato valvar"))[cir.ano == a].sum())})
    salvar(pd.DataFrame(linhas), "01_serie_nacional")
    # cirurgia × aberta só (sem múltiplas) para mostrar o viés
    return padrao


def tendencia(c, pop, nome_col, rotulo, anos=(2008, 2025), excluir_covid=False):
    """Binomial negativa: contagem por ano × faixa × sexo ~ faixa + sexo + ano + pandemia, offset log(pop)."""
    cir = c[c.cirurgia & c[nome_col] & (c.fx >= 0) & c.sexo.isin(["M", "F"])]
    k = cir.groupby(["ano", "fx", "sexo"]).size().rename("n")
    pp = pop.groupby(["ano", "faixa", "sexo"])["pop"].sum()
    g = pd.DataFrame({"pop": pp}).join(k.rename_axis(["ano", "faixa", "sexo"])).fillna(0).reset_index()
    g = g[g.ano.between(*anos) & (g.faixa >= 3)]
    if excluir_covid:
        g = g[~g.ano.isin([2020, 2021])]
    g["t"] = g.ano - 2008
    g["covid"] = g.ano.isin([2020, 2021]).astype(int)
    f = "n ~ C(faixa) + C(sexo) + t" + ("" if excluir_covid else " + covid")
    m = smf.glm(f, data=g, family=sm.families.NegativeBinomial(alpha=0.05), offset=np.log(g["pop"])).fit()
    b, se = m.params["t"], m.bse["t"]
    return {"modelo": rotulo, "anos": f"{anos[0]}–{anos[1]}" + (" sem 2020–21" if excluir_covid else ""),
            "variacao_anual_pct": 100 * (np.exp(b) - 1), "ic95_inf": 100 * (np.exp(b - 1.96 * se) - 1),
            "ic95_sup": 100 * (np.exp(b + 1.96 * se) - 1), "p": m.pvalues["t"],
            "irr_pandemia": float(np.exp(m.params.get("covid", np.nan)))}


def tendencias(c, pop):
    linhas = []
    for col, rot in [("reum", "Reumática"), ("reum_ampliada", "Reumática ampliada"),
                     ("reum_limite", "Reumática limite superior"), ("naoreum", "Não reumática")]:
        linhas.append(tendencia(c, pop, col, rot))
        linhas.append(tendencia(c, pop, col, rot, excluir_covid=True))
        linhas.append(tendencia(c, pop, col, rot, anos=(2013, 2025)))
    # sem as cirurgias múltiplas (o que se concluiria só com o código valvar)
    c2 = c.copy()
    c2["cirurgia"] = c2.PROC_REA.isin(ABERTA)
    linhas.append(tendencia(c2, pop, "reum", "Reumática SEM cirurgias múltiplas (viés)"))
    salvar(pd.DataFrame(linhas), "02_tendencia_binomial_negativa")
    # por faixa etária, binomial negativa com alfa estimado (statsmodels discreto)
    pp = pop.groupby(["ano", "faixa", "sexo"])["pop"].sum()
    lin = []
    for col, rot in [("reum", "Reumática"), ("naoreum", "Não reumática")]:
        x = c[c.cirurgia & c[col] & (c.fx >= 0) & c.sexo.isin(["M", "F"])]
        k = x.groupby(["ano", "fx", "sexo"]).size().rename("n")
        g = pd.DataFrame({"pop": pp}).join(k.rename_axis(["ano", "faixa", "sexo"])).fillna(0).reset_index()
        g = g[g.ano.between(2008, 2025) & (g.faixa >= 3)].copy()
        g["t"], g["covid"], g["lp"] = g.ano - 2008, g.ano.isin([2020, 2021]).astype(int), np.log(g["pop"])
        g["grupo"] = pd.cut(g.faixa, [2, 7, 11, 20], labels=["15-39", "40-59", "60+"])
        for gr, h in [("todas (alfa estimado)", g)] + list(g.groupby("grupo", observed=True)):
            m = smf.negativebinomial("n ~ C(faixa) + C(sexo) + t + covid", data=h, offset=h.lp).fit(disp=0, maxiter=300)
            ci = m.conf_int().loc["t"]
            lin.append({"etiologia": rot, "faixa": gr, "alfa": m.params["alpha"],
                        "variacao_anual_pct": 100 * (np.exp(m.params["t"]) - 1),
                        "ic95_inf": 100 * (np.exp(ci[0]) - 1), "ic95_sup": 100 * (np.exp(ci[1]) - 1)})
    salvar(pd.DataFrame(lin), "02b_tendencia_por_faixa_etaria")


def idade_coorte(c, pop):
    cir = c[c.cirurgia & c.reum]
    per = pd.cut(cir.ano, [2007, 2010, 2016, 2019, 2022, 2025], labels=["2008-10", "2011-16", "2017-19", "2020-22", "2023-25"])
    pa = pop.groupby(["ano", "faixa"])["pop"].sum()
    linhas = []
    for p, g in cir.groupby(per, observed=True):
        anos = sorted(g.ano.unique())
        den = sum(pa.loc[a] for a in anos)
        k = g[g.fx >= 0].groupby("fx").size().reindex(range(NF), fill_value=0)
        for fx in range(NF):
            linhas.append({"periodo": p, "faixa": FAIXAS[fx], "n": int(k[fx]), "taxa_100mil": 1e5 * k[fx] / den[fx]})
    salvar(pd.DataFrame(linhas), "03a_taxa_idade_periodo_reumatica")
    co = cir.assign(dec=(np.floor(cir.coorte / 10) * 10)).pivot_table(index="ano", columns="dec", values="N_AIH",
                                                                       aggfunc="size", fill_value=0)
    co.columns = [f"nasc_{int(x)}s" for x in co.columns]
    salvar(co.reset_index(), "03b_coorte_nascimento_reumatica")
    resumo = cir.groupby("ano").agg(n=("N_AIH", "size"), idade_mediana=("idade", "median"),
                                    p25=("idade", lambda v: v.quantile(.25)), p75=("idade", lambda v: v.quantile(.75)),
                                    pct_menor40=("idade", lambda v: 100 * (v < 40).mean()),
                                    pct_60mais=("idade", lambda v: 100 * (v >= 60).mean()),
                                    pct_fem=("sexo", lambda v: 100 * v.eq("F").mean()),
                                    nasc_mediano=("coorte", "median")).reset_index()
    salvar(resumo, "03c_perfil_reumatica_ano")


def projecao(c, pop):
    """Demanda 2026–2035: taxas específicas por idade de 2023–2025 × população projetada (A: taxas constantes;
    B: taxas caindo no ritmo anual estimado pelo modelo 2008–2025)."""
    pr = pd.DataFrame([{"faixa": FAIXAS.index(r["D5N"]) if r["D5N"] in FAIXAS else NF - 1, "ano": int(r["D6N"]),
                        "pop": float(r["V"])}
                       for r in json.load(open(os.path.join(REF, "proj7358_brasil_idade_2026_2040.json")))])
    pr = pr.groupby(["ano", "faixa"], as_index=False)["pop"].sum()
    pa = pop.groupby(["ano", "faixa"])["pop"].sum()
    bruto25 = sum(float(r["V"]) for r in json.load(open(os.path.join(REF, "proj7358_sexo_idade_2000_2025.json")))
                  if r["D6N"] == "2025")
    fator = pa.loc[2025].sum() / bruto25  # nível da estimativa 2025 aplicado à projeção futura
    cir = c[c.cirurgia & c.reum & (c.fx >= 0)]
    k = cir[cir.ano.between(2023, 2025)].groupby("fx").size().reindex(range(NF), fill_value=0)
    den = sum(pa.loc[a] for a in (2023, 2024, 2025))
    taxa = (k / den.to_numpy()).to_numpy()
    tend = T["02_tendencia_binomial_negativa"]
    v = tend[(tend.modelo == "Reumática") & (tend.anos == "2008–2025")].variacao_anual_pct.iloc[0] / 100
    linhas = []
    for a in range(2026, 2036):
        pa_ = fator * pr[pr.ano == a].set_index("faixa")["pop"].reindex(range(NF)).to_numpy()
        linhas.append({"ano": a, "cenario_A_taxas_constantes": float((taxa * pa_).sum()),
                       "cenario_B_tendencia": float((taxa * pa_ * (1 + v) ** (a - 2024)).sum())})
    salvar(pd.DataFrame(linhas), "04_projecao_demanda_reumatica")
    RES["projecao_fator_populacao"] = round(fator, 4)


def febre_aguda(pop):
    fs = sorted(glob.glob(os.path.join(SAIDA, "parciais", "RD*.csv")))
    d = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False,
                               usecols=["N_AIH", "IDENT", "DIAG_PRINC", "DT_INTER", "IDADE", "COD_IDADE", "MUNIC_RES"])
                   for f in fs])
    d = d[d.IDENT.str.strip() != "5"].drop_duplicates("N_AIH")
    d["ano"] = pd.to_numeric(d.DT_INTER.str[:4], errors="coerce")
    idade = pd.to_numeric(d.IDADE, errors="coerce").where(d.COD_IDADE.str.strip() == "4")
    d = d[d.DIAG_PRINC.str[:3].isin(["I00", "I01", "I02"]) & idade.between(5, 19) & d.ano.between(2000, 2025)]
    d["regiao"] = d.MUNIC_RES.str[:1].map(REGIAO)
    p519 = pop[pop.faixa.isin([1, 2, 3])].groupby("ano")["pop"].sum()
    k = d.groupby("ano").size()
    t = pd.DataFrame({"n_5a19": k, "pop_5a19": p519}).reset_index()
    t["taxa_100mil"] = 1e5 * t.n_5a19 / t.pop_5a19
    r = d.pivot_table(index="ano", columns="regiao", values="N_AIH", aggfunc="size", fill_value=0)
    salvar(t.join(r, on="ano"), "05_febre_reumatica_aguda_5a19")
    g = t.copy()
    g["t"] = g.ano - 2000
    m = smf.glm("n_5a19 ~ t", data=g, family=sm.families.NegativeBinomial(alpha=0.05),
                offset=np.log(g.pop_5a19)).fit()
    b, se = m.params["t"], m.bse["t"]
    RES["febre_aguda_variacao_anual_pct"] = [round(100 * (np.exp(x) - 1), 1) for x in (b, b - 1.96 * se, b + 1.96 * se)]


def regioes(c, pop):
    cir = c[c.cirurgia & c.reum & (c.fx >= 0) & c.regiao_res.notna()]
    pa = pop.groupby(["regiao", "ano", "faixa"])["pop"].sum()
    pb = pop.groupby(["ano", "faixa"])["pop"].sum()
    padrao = pb.loc[2010] / pb.loc[2010].sum()
    linhas = []
    for (rg, a), g in cir.groupby(["regiao_res", "ano"]):
        k = g.groupby("fx").size().reindex(range(NF), fill_value=0).to_numpy()
        linhas.append({"regiao": rg, "ano": a, "n": len(g), "taxa_padronizada": 1e5 * (k / pa.loc[(rg, a)].to_numpy()
                                                                                        * padrao.to_numpy()).sum(),
                       "pct_operado_fora_uf": 100 * g.fora_uf.mean(), "idade_mediana": g.idade.median()})
    salvar(pd.DataFrame(linhas), "06_regiao_residencia")
    h = c[c.cirurgia & c.reum].groupby("ano").agg(hospitais=("CNES", "nunique"),
                                                  cirurgioes=("cirurgiao", lambda v: v[v != ""].nunique())).reset_index()
    salvar(h, "06b_hospitais_cirurgioes_ano")


def mitral(c):
    """Doença mitral reumática (I05): estratégia sem prótese × com prótese; percutânea; mecânica × biológica."""
    x = c[c.cid3.eq("I05") & (c.cirurgia | c.tipo.eq("Valvuloplastia mitral percutânea"))]
    x = x.assign(estrategia=np.select(
        [x.tipo.eq("Valvuloplastia mitral percutânea"), x.cirurgia & (x.n_protese == 0), x.cirurgia & (x.n_protese == 1),
         x.cirurgia & (x.n_protese >= 2)],
        ["Percutânea", "Cirurgia sem prótese (plastia)", "Cirurgia com 1 prótese", "Cirurgia com ≥2 próteses"], "Outra"))
    t = x.pivot_table(index="ano", columns="estrategia", values="N_AIH", aggfunc="size", fill_value=0)
    cirs = t[["Cirurgia sem prótese (plastia)", "Cirurgia com 1 prótese", "Cirurgia com ≥2 próteses"]].sum(axis=1)
    t["pct_sem_protese_entre_cirurgias"] = 100 * t["Cirurgia sem prótese (plastia)"] / cirs
    # valva única: código 0406010803 × 0406010692
    u = x[x.PROC_REA.isin(["0406010803", "0406010692"])]
    t["pct_plastica_valva_unica"] = 100 * u.groupby("ano").apply(lambda g: g.PROC_REA.eq("0406010803").mean(),
                                                                include_groups=False)
    t["pct_anel_nas_sem_protese"] = 100 * x[x.estrategia.eq("Cirurgia sem prótese (plastia)")].groupby("ano").anel.apply(
        lambda v: (v > 0).mean())
    salvar(t.reset_index(), "07a_mitral_reumatica_estrategia")
    # mecânica × biológica entre trocas reumáticas (qualquer valva) por idade
    tr = c[c.cirurgia & c.reum & (c.n_protese > 0)]
    tr = tr.assign(fx3=pd.cut(tr.idade, [-1, 39, 59, 200], labels=["<40", "40-59", "60+"]))
    mb = tr.groupby(["ano", "fx3"], observed=True).agg(n=("N_AIH", "size"), pct_mecanica=("mecanica", "mean")).reset_index()
    mb["pct_mecanica"] *= 100
    salvar(mb, "07b_mecanica_biologica_reumatica")
    # variação entre hospitais (I05, cirurgia, 2016–2025, hospitais com ≥30)
    h = x[x.cirurgia & x.ano.between(2016, 2025)].groupby("CNES").agg(n=("N_AIH", "size"), sem_protese=("sem_protese", "mean"))
    h = h[h.n >= 30]
    salvar(h.describe(percentiles=[.1, .25, .5, .75, .9]).reset_index(), "07c_hospitais_plastia_mitral_reum")
    RES["mitral_hospitais_ge30"] = int(len(h))
    RES["mitral_hospitais_zero_plastia"] = int((h.sem_protese == 0).sum())


def mortalidade(c):
    """Mortalidade hospitalar: logística com EP robusto por hospital."""
    x = c[c.cirurgia & c.etiologia.isin(["Reumática", "Não reumática"]) & c.idade.between(18, 100)
          & c.sexo.isin(["M", "F"])].copy()
    x["reum"] = (x.etiologia == "Reumática").astype(int)
    x["t"] = x.ano - 2008
    x["multi"] = ((x.n_protese >= 2) | x.tipo.isin(["Plástica e/ou troca múltipla"])).astype(int)
    x["rvm"] = x.tipo.eq("Valvar + RVM").astype(int)
    x["mult"] = x.tipo.eq("Múltipla com ato valvar").astype(int)
    x["regiao_int"] = x.regiao_int.fillna("Ignorado")
    f = ("obito ~ reum + bs(idade, df=4) + C(sexo) + urgencia + multi + rvm + mult + t + C(regiao_int)")
    m = smf.glm(f, data=x, family=sm.families.Binomial()).fit(cov_type="cluster",
                                                              cov_kwds={"groups": pd.factorize(x.CNES)[0]})
    linhas = []
    for v in ["reum", "urgencia", "multi", "rvm", "mult", "t"]:
        b, se = m.params[v], m.bse[v]
        linhas.append({"variavel": v, "OR": np.exp(b), "ic95_inf": np.exp(b - 1.96 * se), "ic95_sup": np.exp(b + 1.96 * se),
                       "p": m.pvalues[v]})
    salvar(pd.DataFrame(linhas), "08a_mortalidade_logistica")
    br = x.groupby(["ano", "etiologia"]).agg(n=("obito", "size"), mort_pct=("obito", "mean")).reset_index()
    br["mort_pct"] *= 100
    salvar(br, "08b_mortalidade_bruta_ano")
    RES["mortalidade_n_modelo"] = int(len(x))


def cobertura(c):
    rd = [json.loads(l) for l in open(os.path.join(SAIDA, "parciais", "_cobertura.jsonl"))]
    sp = [json.loads(l) for l in open(os.path.join(SAIDA, "sp_parciais", "_cobertura.jsonl"))]
    rdu = {r["arquivo"]: r for r in rd}
    spu = {r["arquivo"]: r for r in sp}
    RES["rd_arquivos"] = {"lidos": sum(r["status"] == "lido" for r in rdu.values()),
                          "falhou": sorted(k for k, r in rdu.items() if r["status"] != "lido")}
    RES["sp_arquivos"] = {"lidos": sum(r["status"] == "lido" for r in spu.values()),
                          "falhou": sorted(k for k, r in spu.items() if r["status"] != "lido")}
    ok = [r for r in spu.values() if r["status"] == "lido"]
    RES["sp_aih_alvo"] = sum(r["aih_alvo"] for r in ok)
    RES["sp_aih_encontradas"] = sum(r["aih_encontradas"] for r in ok)
    cir = c[c.cirurgia]
    t = cir.groupby(["ano", "tipo"]).size().unstack(fill_value=0)
    salvar(t.reset_index(), "09a_tipo_ano")
    q = cir.groupby("PROC_REA").agg(n=("N_AIH", "size"), no_sp=("no_sp", "mean"),
                                   n_protese_media=("n_protese", "mean"), sem_protese=("sem_protese", "mean"),
                                   com_anel=("anel", lambda v: (v > 0).mean()), com_cec=("cec", lambda v: (v > 0).mean()),
                                   com_cirurgiao=("cirurgiao", lambda v: (v != "").mean())).reset_index()
    salvar(q, "09b_qualidade_sp_por_codigo")


def main():
    pop = populacao()
    c = montar()
    taxas(c, pop)
    tendencias(c, pop)
    idade_coorte(c, pop)
    projecao(c, pop)
    febre_aguda(pop)
    regioes(c, pop)
    mitral(c)
    mortalidade(c)
    cobertura(c)
    cols = ["N_AIH", "ano", "PROC_REA", "tipo", "cirurgia", "DIAG_PRINC", "etiologia", "reum_ampliada", "reum_limite",
            "idade", "sexo", "RACA_COR", "uf_res", "regiao_res", "uf_int", "CNES", "urgencia", "obito", "dias", "uti",
            "val", "n_protese", "bio", "mec", "anel", "cec", "atos_valva", "cirurgiao", "no_sp"]
    c[cols].to_csv(os.path.join(SAIDA, "base_valvar_2008_2025.csv.gz"), index=False)
    with pd.ExcelWriter(os.path.join(AQUI, "fase2_valvar.xlsx")) as w:
        pd.DataFrame([{"item": k, "valor": json.dumps(v, ensure_ascii=False)} for k, v in RES.items()]).to_excel(
            w, sheet_name="00_resumo", index=False)
        for k, v in T.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    json.dump(RES, open(os.path.join(OUT, "resumo.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(RES, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
