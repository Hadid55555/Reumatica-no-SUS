#!/usr/bin/env python3
"""Fase 5 — análises pedidas na revisão final, antes de congelar e escrever.

Estimando primário: razão de taxas específicas por idade entre 2023–2025 e 2008–2010 (15 anos de calendário, logo
gerações nascidas com 15 anos de diferença na mesma idade), por faixa de 5 anos; contraste-chave = 15–39 × 55+.
(Idade, período e coorte são colineares: a razão é ao mesmo tempo "15 anos de período" e "15 anos de coorte"; o padrão
geracional aparece como dependência da razão em relação à idade.)

1. Controle não reumático: mesma razão por idade para a não reumática; razão das razões (reum ÷ não reum) com IC95%.
2. Sensibilidade: estrita, ampliada, sem cirurgias múltiplas, janela 2013–2015 × 2023–2025, e série longa 2003–2005 ×
   2023–2025 com a definição ampliada (2003–2005 da extração da Fase 1, códigos abertos das tabelas 32/48).
3. Região e sexo: razões para 15–39, 40–54 e 55+; teste de interação (Poisson com escala de Pearson).
4. Degrau 2013–2017: séries anuais de cirurgia reumática, não reumática, CRVM (SUS), hospitais ativos, valor real
   mediano por AIH valvar (IPCA dez/2025), % cirurgias múltiplas, distribuição por código e % troca com prótese faturada.
5. Conferência: taxa padronizada com padrão alternativo (população 2020) e recontagem dos numeradores das razões.

Saídas: valvar/resultados_fase5/*.csv, valvar/fase5_congelamento.xlsx, valvar/figuras/fig7_*.png.
Uso: python valvar/fase5_congelamento.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)
import fase2_valvar as F  # noqa: E402

OUT = os.path.join(AQUI, "resultados_fase5")
os.makedirs(OUT, exist_ok=True)
T, RES = {}, {}
FX = list(range(3, F.NF))  # 15 anos ou mais
GRUPOS = {"15-39": range(3, 8), "40-54": range(8, 11), "55+": range(11, F.NF)}


def salvar(df, nome):
    T[nome] = df
    df.to_csv(os.path.join(OUT, nome + ".csv"), index=False)


def carregar():
    p = os.path.join(RAIZ, "saida_valvar", "base_valvar_2008_2025.csv.gz")
    if not os.path.exists(p):
        p = os.path.join(AQUI, "base_valvar_2008_2025.csv.gz")
    c = pd.read_csv(p, dtype={"N_AIH": str, "CNES": str, "DIAG_PRINC": str, "PROC_REA": str})
    c["fx"] = F.faixa5(c.idade).to_numpy()
    c["multipla"] = c.tipo.eq("Múltipla com ato valvar")
    return c


def antigos_2003_2005():
    """Cirurgias valvares abertas 2003–2005 (ano da internação) pela definição ampliada, da extração da Fase 1."""
    from fase1_valvar import ABERTA, carregar as c1
    d, _ = c1()
    d = d[d.tipo.isin(ABERTA) & d.ano.between(2003, 2005)].copy()
    d["reum_ampliada"] = d.grupo.eq("Reumática") | ((d.DIAG_PRINC.eq("Q225") | d.cid3.eq("I38")) & (d.idade >= 18))
    d["naoreum"] = d.grupo.eq("Não reumática")
    d["fx"] = F.faixa5(d.idade).to_numpy()
    d["sexo"] = np.where(d.SEXO.eq("1"), "M", np.where(d.SEXO.isin(["2", "3"]), "F", ""))
    return d


def razao(n1, p1, n2, p2):
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan
    rr = (n2 / p2) / (n1 / p1)
    se = np.sqrt(1 / n1 + 1 / n2)
    return rr, rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se)


def razoes(x1, x2, pop, a1, a2, rotulo, filtro_pop=None):
    """Razões por faixa de 5 anos e por grupo etário entre os anos a2 e a1 (listas)."""
    pp = pop if filtro_pop is None else pop[filtro_pop(pop)]
    pa = pp.groupby(["ano", "faixa"])["pop"].sum()
    k1 = x1[x1.ano.isin(a1)].groupby("fx").size()
    k2 = x2[x2.ano.isin(a2)].groupby("fx").size()
    lin = []
    for nome, fxs in [(F.FAIXAS[f], [f]) for f in FX] + list(GRUPOS.items()):
        n1, n2 = sum(k1.get(f, 0) for f in fxs), sum(k2.get(f, 0) for f in fxs)
        p1 = sum(pa.loc[(a, f)] for a in a1 for f in fxs)
        p2 = sum(pa.loc[(a, f)] for a in a2 for f in fxs)
        rr, lo, hi = razao(n1, p1, n2, p2)
        lin.append({"analise": rotulo, "faixa": nome, "n_inicio": n1, "n_fim": n2, "taxa_inicio": 1e5 * n1 / p1 * len(a1) / len(a1),
                    "taxa_fim": 1e5 * n2 / p2, "razao": rr, "ic95_inf": lo, "ic95_sup": hi})
    return lin


A1, A2 = [2008, 2009, 2010], [2023, 2024, 2025]


# ------------------------------------------------------------------ 1 e 2
def geracional(c, pop):
    cir = c[c.cirurgia & (c.fx >= 0)]
    R = cir[cir.etiologia.eq("Reumática")]
    N = cir[cir.etiologia.eq("Não reumática")]
    lin = []
    lin += razoes(R, R, pop, A1, A2, "Reumática estrita (principal)")
    lin += razoes(N, N, pop, A1, A2, "Não reumática (controle)")
    c3 = cir.DIAG_PRINC.str[:3]
    for cid, nome in [("I35", "Controle I35 (aórtica não reumática)"), ("I34", "I34 (mitral não reumática)"),
                      ("I05", "I05 (mitral reumática)"), ("I06", "I06 (aórtica reumática)")]:
        x = cir[c3.eq(cid)]
        lin += razoes(x, x, pop, A1, A2, nome)
    # mudança de compatibilidade em 2015: CIDs inespecíficos que passam a aparecer (teto do vazamento da reumática)
    o = c[c.cirurgia].assign(c3=c.DIAG_PRINC.str[:3])
    vz = o.groupby("ano").apply(lambda g: pd.Series({"reumatica": (g.etiologia == "Reumática").sum(),
                                                     "I50_I42_I33_T82": g.c3.isin(["I50", "I42", "I33", "T82"]).sum()}),
                                include_groups=False).reset_index()
    salvar(vz, "4c_cids_liberados_2015")
    Ra = cir[cir.reum_ampliada]
    lin += razoes(Ra, Ra, pop, A1, A2, "Reumática ampliada")
    Rs = R[~R.multipla]
    lin += razoes(Rs, Rs, pop, A1, A2, "Reumática sem cirurgias múltiplas")
    lin += razoes(R, R, pop, [2013, 2014, 2015], A2, "Reumática estrita, 2013–15 × 2023–25")
    old = antigos_2003_2005()
    o = old[old.reum_ampliada & (old.fx >= 0)]
    lin += razoes(o, Ra, pop, [2003, 2004, 2005], A2, "Reumática ampliada, 2003–05 × 2023–25 (20 anos)")
    on = old[old.naoreum & (old.fx >= 0)]
    lin += razoes(on, N, pop, [2003, 2004, 2005], A2, "Não reumática, 2003–05 × 2023–25 (20 anos)")
    t = pd.DataFrame(lin)
    salvar(t, "1_razoes_geracionais_e_sensibilidade")
    # razão das razões (reumática ÷ não reumática)
    a = t[t.analise == "Reumática estrita (principal)"].set_index("faixa")
    b = t[t.analise == "Controle I35 (aórtica não reumática)"].set_index("faixa")
    rr = []
    for f in a.index:
        se = np.sqrt(sum(1 / a.loc[f, k] for k in ("n_inicio", "n_fim")) + sum(1 / b.loc[f, k] for k in ("n_inicio", "n_fim")))
        r = a.loc[f, "razao"] / b.loc[f, "razao"]
        rr.append({"faixa": f, "razao_reum": a.loc[f, "razao"], "razao_I35": b.loc[f, "razao"], "razao_das_razoes": r,
                   "ic95_inf": r * np.exp(-1.96 * se), "ic95_sup": r * np.exp(1.96 * se)})
    salvar(pd.DataFrame(rr), "1b_reum_vs_I35_razao_das_razoes")
    # interação etiologia × período × grupo etário (Poisson, escala de Pearson)
    pa = pop.groupby(["ano", "faixa"])["pop"].sum()
    rows = []
    for et, x in [("reum", R), ("naoreum", N)]:
        k = x.groupby(["ano", "fx"]).size()
        for per, anos in [("ini", A1), ("fim", A2)]:
            for f in FX:
                rows.append({"et": et, "per": per, "fx": f, "n": sum(k.get((a, f), 0) for a in anos),
                             "pop": sum(pa.loc[(a, f)] for a in anos)})
    g = pd.DataFrame(rows)
    g["grupo"] = pd.cut(g.fx, [2, 7, 10, 20], labels=["15-39", "40-54", "55+"]).astype(str)
    m0 = smf.glm("n ~ C(fx)*et + per*et + per*C(grupo)", data=g, family=sm.families.Poisson(), offset=np.log(g["pop"])).fit(scale="X2")
    m1 = smf.glm("n ~ C(fx)*et + per*et*C(grupo)", data=g, family=sm.families.Poisson(), offset=np.log(g["pop"])).fit(scale="X2")
    from scipy import stats
    dd = (m0.deviance - m1.deviance) / (m0.df_resid - m1.df_resid) / m1.scale
    RES["interacao_etiologia_periodo_idade_F_p"] = [round(dd, 2), float(stats.f.sf(dd, m0.df_resid - m1.df_resid, m1.df_resid))]


# ------------------------------------------------------------------ 3
def regiao_sexo(c, pop):
    cir = c[c.cirurgia & (c.fx >= 0) & c.etiologia.eq("Reumática")]
    lin = []
    for reg in ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
        x = cir[cir.regiao_res == reg]
        for l in razoes(x, x, pop, A1, A2, f"Região {reg}", lambda p, r=reg: p.regiao == r):
            if l["faixa"] in GRUPOS:
                lin.append(l)
    for sx, nome in [("M", "Homens"), ("F", "Mulheres")]:
        x = cir[cir.sexo == sx]
        for l in razoes(x, x, pop, A1, A2, nome, lambda p, s=sx: p.sexo == s):
            if l["faixa"] in GRUPOS:
                lin.append(l)
    salvar(pd.DataFrame(lin), "3_razoes_regiao_sexo")
    # teste de interação período × grupo etário × região (e × sexo)
    for dim, col, popcol in [("regiao", "regiao_res", "regiao"), ("sexo", "sexo", "sexo")]:
        pp = pop.groupby(["ano", "faixa", popcol])["pop"].sum()
        rows = []
        cats = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"] if dim == "regiao" else ["M", "F"]
        k = cir.groupby(["ano", "fx", col]).size()
        for cat in cats:
            for per, anos in [("ini", A1), ("fim", A2)]:
                for f in FX:
                    rows.append({"cat": cat, "per": per, "fx": f, "n": sum(k.get((a, f, cat), 0) for a in anos),
                                 "pop": sum(pp.loc[(a, f, cat)] for a in anos)})
        g = pd.DataFrame(rows)
        g["grupo"] = pd.cut(g.fx, [2, 7, 10, 20], labels=["15-39", "40-54", "55+"]).astype(str)
        m0 = smf.glm("n ~ C(fx)*cat + per*cat + per*C(grupo)", data=g, family=sm.families.Poisson(), offset=np.log(g["pop"])).fit(scale="X2")
        m1 = smf.glm("n ~ C(fx)*cat + per*cat*C(grupo)", data=g, family=sm.families.Poisson(), offset=np.log(g["pop"])).fit(scale="X2")
        from scipy import stats
        dd = (m0.deviance - m1.deviance) / (m0.df_resid - m1.df_resid) / m1.scale
        RES[f"interacao_{dim}_periodo_idade_F_p"] = [round(dd, 2), float(stats.f.sf(dd, m0.df_resid - m1.df_resid, m1.df_resid))]


# ------------------------------------------------------------------ 4
def degrau(c):
    ipca = pd.read_csv(os.path.join(RAIZ, "artigo", "dados_ref", "ipca_fator_dez2025.csv")).set_index("ano").fator_dez2025
    cir = c[c.cirurgia].copy()
    cir["val_real"] = cir.val * cir.ano.map(ipca)
    s = cir.groupby("ano").agg(valvar_total=("N_AIH", "size"),
                               reumatica=("etiologia", lambda v: (v == "Reumática").sum()),
                               nao_reumatica=("etiologia", lambda v: (v == "Não reumática").sum()),
                               hospitais_ativos=("CNES", "nunique"),
                               valor_real_mediano=("val_real", "median"),
                               pct_multiplas=("multipla", "mean")).reset_index()
    s["pct_multiplas"] *= 100
    s["reumatica_sem_multiplas"] = cir[~cir.multipla & cir.etiologia.eq("Reumática")].groupby("ano").size().reindex(s.ano).to_numpy()
    tro = cir[cir.PROC_REA.astype(str).str.zfill(10).eq("0406010692")]
    s["pct_troca_com_protese_faturada"] = (100 * tro.groupby("ano").n_protese.apply(lambda v: (v > 0).mean())).reindex(s.ano).to_numpy()
    # CRVM no SUS (consolidado do projeto irmão, total por ano da internação)
    try:
        cr = pd.read_excel(os.path.join(RAIZ, "saida_crvm", "crvm_2008_2025_consolidado.xlsx"), sheet_name="Por_ano", header=None)
        cr = cr[pd.to_numeric(cr[0], errors="coerce").between(2008, 2025)][[0, 19]]
        s = s.merge(cr.rename(columns={0: "ano", 19: "crvm_sus"}).astype({"ano": int}), on="ano", how="left")
    except Exception as e:  # noqa: BLE001
        RES["crvm_erro"] = str(e)
    base = s.set_index("ano").loc[2013]
    for col in ["reumatica", "nao_reumatica", "valvar_total", "crvm_sus", "hospitais_ativos", "valor_real_mediano"]:
        if col in s:
            s[f"indice_{col}_2013_100"] = 100 * s[col] / base[col]
    salvar(s, "4a_degrau_2013_2017_series")
    cod = cir.assign(cod=np.where(cir.multipla, "múltipla (0415)", cir.PROC_REA.astype(str).str.zfill(10)))
    cod = cod[cod.etiologia.eq("Reumática")].pivot_table(index="ano", columns="cod", values="N_AIH", aggfunc="size", fill_value=0)
    salvar(cod.reset_index(), "4b_reumatica_por_codigo_ano")
    x = s.set_index("ano")
    RES["degrau_2013_2017_variacao_pct"] = {k: round(100 * (x.loc[2017, k] / x.loc[2013, k] - 1), 1)
                                            for k in ["reumatica", "nao_reumatica", "valvar_total", "crvm_sus",
                                                      "hospitais_ativos", "valor_real_mediano"] if k in x}


# ------------------------------------------------------------------ 5
def conferencia(c, pop):
    pa = pop.groupby(["ano", "faixa"])["pop"].sum()
    lin = []
    for padrao_ano in (2010, 2020):
        w = (pa.loc[padrao_ano] / pa.loc[padrao_ano].sum()).to_numpy()
        for et in ["Reumática", "Não reumática"]:
            k = c[c.cirurgia & c.etiologia.eq(et) & (c.fx >= 0)].groupby(["ano", "fx"]).size()
            for a in (2008, 2013, 2017, 2024, 2025):
                r = np.array([k.get((a, f), 0) for f in range(F.NF)]) / pa.loc[a].to_numpy()
                lin.append({"padrao": padrao_ano, "etiologia": et, "ano": a, "taxa_padronizada": 1e5 * (r * w).sum()})
    t = pd.DataFrame(lin).pivot_table(index=["etiologia", "ano"], columns="padrao", values="taxa_padronizada").reset_index()
    salvar(t, "5_conferencia_padrao_alternativo")
    # recontagem direta do numerador principal (15–39, 2008–10 e 2023–25)
    cir = c[c.cirurgia & c.etiologia.eq("Reumática")]
    RES["recontagem_15_39"] = {"2008_10": int(((cir.idade >= 15) & (cir.idade < 40) & cir.ano.isin(A1)).sum()),
                               "2023_25": int(((cir.idade >= 15) & (cir.idade < 40) & cir.ano.isin(A2)).sum())}


def figura():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = T["1_razoes_geracionais_e_sensibilidade"]
    t = t[~t.faixa.isin(GRUPOS)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for an, cor, lab in [("Reumática estrita (principal)", "#2a78d6", "Reumática (I05–I09)"),
                         ("Controle I35 (aórtica não reumática)", "#eb6834", "Aórtica não reumática (I35)")]:
        x = t[t.analise == an].reset_index(drop=True)
        xs = np.arange(len(x))
        ax.fill_between(xs, x.ic95_inf, x.ic95_sup, color=cor, alpha=0.15, lw=0)
        ax.plot(xs, x.razao, color=cor, lw=2, marker="o", ms=4, label=lab)
        ax.annotate(lab, (xs[-1], x.razao.iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7.5, va="center")
    ax.axhline(1, color="#52514e", lw=0.8)
    ax.set_xticks(range(len(FX)))
    ax.set_xticklabels([F.FAIXAS[f].split(" ")[0] + ("+" if "mais" in F.FAIXAS[f] else "") for f in FX], fontsize=7.5)
    ax.set_xlabel("Idade na cirurgia (início da faixa de 5 anos)")
    ax.set_ylabel("Razão de taxas 2023–25 ÷ 2008–10")
    ax.set_title("Mesma idade, 15 anos depois: a reumática cai mais nos jovens e não cresce nos idosos", loc="left", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e4e3df", lw=0.6)
    ax.set_xlim(-0.5, len(FX) + 2.5)
    fig.tight_layout()
    fig.savefig(os.path.join(AQUI, "figuras", "fig7_geracional_reum_vs_naoreum.png"), dpi=300, bbox_inches="tight")


def main():
    pop = F.populacao()
    c = carregar()
    geracional(c, pop)
    regiao_sexo(c, pop)
    degrau(c)
    conferencia(c, pop)
    figura()
    with pd.ExcelWriter(os.path.join(AQUI, "fase5_congelamento.xlsx")) as w:
        for k, v in T.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    json.dump(RES, open(os.path.join(OUT, "resumo.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(RES, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
