#!/usr/bin/env python3
"""Fase 3 — análises pedidas pela segunda opinião metodológica (PROMPT_SEGUNDA_OPINIAO_valvar.md).

A. Teste decisivo: taxa por faixa etária (15–39, 40–59, 60+) por região de residência, nas três definições de
   reumática, com e sem cirurgias múltiplas; 2013–2019 × 2022–2024 (razão de taxas) e tendência 2013–2019.
B. Decomposição de Das Gupta (tamanho × estrutura etária × taxa específica), 2008 × 2024, Brasil e regiões.
C. Tendência segmentada: 2008–2019 e 2022–2024, com e sem 2025; conciliação modelo × ponta a ponta.
D. Estabilidade do perfil de I05 / I06 / I34 / I35 (idade, % mulheres), 2013–2025.
E. Transferências (motivo de saída) e AIH de continuidade.
F. Auditoria procedimento × prótese × anel (OPM) por código e ano.
G. Plastia mitral reumática ajustada por idade (o envelhecimento explica a queda?).
H. Mapa de Lexis (idade × ano) — figura.

Entradas: base de fase2 (saida_valvar/base_valvar_2008_2025.csv.gz ou valvar/), RD parciais, população de fase2.
Saídas: valvar/resultados_fase3/*.csv, valvar/fase3_revisao.xlsx, valvar/figuras/fig5_lexis_reumatica.png.
Uso: python valvar/fase3_revisao.py
"""
import glob
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

OUT = os.path.join(AQUI, "resultados_fase3")
os.makedirs(OUT, exist_ok=True)
T, RES = {}, {}
GRUPOS = {"15-39": range(3, 8), "40-59": range(8, 12), "60+": range(12, F.NF)}
DEFS = {"estrita": "reum", "ampliada": "reum_ampliada", "limite": "reum_limite"}


def salvar(df, nome):
    T[nome] = df
    df.to_csv(os.path.join(OUT, nome + ".csv"), index=False)


def carregar():
    p = os.path.join(RAIZ, "saida_valvar", "base_valvar_2008_2025.csv.gz")
    if not os.path.exists(p):
        p = os.path.join(AQUI, "base_valvar_2008_2025.csv.gz")
    c = pd.read_csv(p, dtype={"N_AIH": str, "CNES": str, "DIAG_PRINC": str, "PROC_REA": str, "RACA_COR": str})
    c = c[c.cirurgia].copy()
    c["reum"] = c.etiologia.eq("Reumática")
    c["fx"] = F.faixa5(c.idade).to_numpy()
    c["multipla"] = c.tipo.eq("Múltipla com ato valvar")
    c["cid3"] = c.DIAG_PRINC.str[:3]
    # motivo de saída e IDENT do RD
    fs = sorted(glob.glob(os.path.join(RAIZ, "saida_valvar", "parciais", "RD*.csv")))
    rd = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, usecols=["N_AIH", "IDENT", "COBRANCA"]) for f in fs])
    rd = rd.apply(lambda s: s.str.strip()).drop_duplicates("N_AIH").set_index("N_AIH")
    c = c.join(rd, on="N_AIH")
    return c


def contar(x, por):
    return x[x.fx >= 0].groupby(por).size()


# ------------------------------------------------------------------ A
def teste_decisivo(c, pop):
    pr = pop.groupby(["regiao", "ano", "faixa"])["pop"].sum()
    pb = pop.groupby(["ano", "faixa"])["pop"].sum()
    linhas, slopes = [], []
    for dn, col in DEFS.items():
        for mult in ("com", "sem"):
            x = c[c[col] & (True if mult == "com" else ~c.multipla)]
            for reg in ["Brasil", "Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
                xr = x if reg == "Brasil" else x[x.regiao_res == reg]
                den = pb if reg == "Brasil" else pr.loc[reg]
                k = contar(xr, ["ano", "fx"])
                for g, fxs in GRUPOS.items():
                    def taxa(anos):
                        n = sum(k.get((a, f), 0) for a in anos for f in fxs)
                        p = sum(den.loc[(a, f)] for a in anos for f in fxs)
                        return n, p
                    n1, p1 = taxa(range(2013, 2020))
                    n2, p2 = taxa(range(2022, 2025))
                    rr = (n2 / p2) / (n1 / p1) if n1 and n2 else np.nan
                    se = np.sqrt(1 / n1 + 1 / n2) if n1 and n2 else np.nan
                    linhas.append({"definicao": dn, "multiplas": mult, "regiao": reg, "faixa": g,
                                   "n_2013_19": n1, "taxa_2013_19": 1e5 * n1 / p1 / 7 * 7,
                                   "n_2022_24": n2, "taxa_2022_24": 1e5 * n2 / p2,
                                   "razao_taxas": rr, "ic95_inf": rr * np.exp(-1.96 * se), "ic95_sup": rr * np.exp(1.96 * se)})
                    # tendência anual 2013–2019 (Poisson com escala de Pearson)
                    rows = [{"ano": a, "fx": f, "n": k.get((a, f), 0), "pop": den.loc[(a, f)]}
                            for a in range(2013, 2020) for f in fxs]
                    g_ = pd.DataFrame(rows)
                    if g_.n.sum() >= 30:
                        m = smf.glm("n ~ C(fx) + ano", data=g_, family=sm.families.Poisson(),
                                    offset=np.log(g_["pop"])).fit(scale="X2")
                        ci = m.conf_int().loc["ano"]
                        slopes.append({"definicao": dn, "multiplas": mult, "regiao": reg, "faixa": g,
                                       "variacao_anual_2013_19_pct": 100 * (np.exp(m.params["ano"]) - 1),
                                       "ic95_inf": 100 * (np.exp(ci[0]) - 1), "ic95_sup": 100 * (np.exp(ci[1]) - 1)})
    a = pd.DataFrame(linhas)
    a["taxa_2013_19"] = a["taxa_2013_19"]  # taxa média anual no período (n/p acumulados)
    salvar(a, "A1_razao_taxas_regiao_faixa_definicao")
    salvar(pd.DataFrame(slopes), "A2_tendencia_2013_2019_regiao_faixa")
    j = a[(a.faixa == "15-39")]
    RES["jovens_caem_em_todas_regioes_definicoes"] = bool((j[j.regiao != "Brasil"].ic95_sup < 1).all())
    RES["jovens_celulas_sem_queda"] = j[(j.regiao != "Brasil") & ~(j.ic95_sup < 1)][
        ["definicao", "multiplas", "regiao", "razao_taxas", "ic95_inf", "ic95_sup"]].round(3).to_dict("records")


# ------------------------------------------------------------------ B
def das_gupta(c, pop, anos=(2008, 2024)):
    """Decomposição de Das Gupta para N = P × Σ s_i × r_i (P = população, s = estrutura etária, r = taxa específica)."""
    pr = pop.groupby(["regiao", "ano", "faixa"])["pop"].sum()
    pb = pop.groupby(["ano", "faixa"])["pop"].sum()
    linhas = []
    for dn, col in [("estrita", "reum"), ("não reumática", None)]:
        x = c[c[col]] if col else c[c.etiologia.eq("Não reumática")]
        for reg in ["Brasil", "Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
            xr = x if reg == "Brasil" else x[x.regiao_res == reg]
            den = pb if reg == "Brasil" else pr.loc[reg]
            k = contar(xr, ["ano", "fx"])
            f = {}
            for a in anos:
                pa = den.loc[a].reindex(range(F.NF)).to_numpy()
                ka = np.array([k.get((a, i), 0) for i in range(F.NF)])
                f[a] = (pa.sum(), pa / pa.sum(), ka / pa)
            (P1, s1, r1), (P2, s2, r2) = f[anos[0]], f[anos[1]]
            N = lambda P, s, r: P * (s * r).sum()  # noqa: E731
            efP = ((N(P2, s1, r1) - N(P1, s1, r1)) + (N(P2, s2, r2) - N(P1, s2, r2))) / 3 \
                + ((N(P2, s1, r2) - N(P1, s1, r2)) + (N(P2, s2, r1) - N(P1, s2, r1))) / 6
            efS = ((N(P1, s2, r1) - N(P1, s1, r1)) + (N(P2, s2, r2) - N(P2, s1, r2))) / 3 \
                + ((N(P1, s2, r2) - N(P1, s1, r2)) + (N(P2, s2, r1) - N(P2, s1, r1))) / 6
            efR = ((N(P1, s1, r2) - N(P1, s1, r1)) + (N(P2, s2, r2) - N(P2, s2, r1))) / 3 \
                + ((N(P1, s2, r2) - N(P1, s2, r1)) + (N(P2, s1, r2) - N(P2, s1, r1))) / 6
            n1, n2 = N(P1, s1, r1), N(P2, s2, r2)
            linhas.append({"etiologia": dn, "regiao": reg, "n_inicio": n1, "n_fim": n2, "diferenca": n2 - n1,
                           "efeito_tamanho_pop": efP, "efeito_estrutura_etaria": efS, "efeito_taxa_especifica": efR,
                           "conferencia_soma": efP + efS + efR - (n2 - n1), "anos": f"{anos[0]}–{anos[1]}"})
    salvar(pd.DataFrame(linhas), "B_decomposicao_das_gupta_2008_2024")


# ------------------------------------------------------------------ C
def segmentada(c, pop):
    pp = pop.groupby(["ano", "faixa", "sexo"])["pop"].sum()
    linhas = []
    for dn, col in list(DEFS.items()) + [("não reumática", None)]:
        x = c[(c[col] if col else c.etiologia.eq("Não reumática")) & c.sexo.isin(["M", "F"])]
        k = contar(x, ["ano", "fx", "sexo"]).rename("n")
        g = pd.DataFrame({"pop": pp}).join(k.rename_axis(["ano", "faixa", "sexo"])).fillna(0).reset_index()
        g = g[g.ano.between(2008, 2025) & (g.faixa >= 3)].copy()
        g["lp"] = np.log(g["pop"])
        for fim in (2024, 2025):
            h = g[(g.ano <= fim) & ~g.ano.isin([2020, 2021])].copy()
            h["pos"] = (h.ano >= 2022).astype(int)
            h["t_pre"] = np.where(h.pos == 0, h.ano - 2008, 0)
            h["t_pos"] = np.where(h.pos == 1, h.ano - 2022, 0)
            m = smf.negativebinomial("n ~ C(faixa) + C(sexo) + t_pre + pos + t_pos", data=h, offset=h.lp).fit(
                disp=0, maxiter=300)
            for v, rot in [("t_pre", "2008–2019"), ("t_pos", f"2022–{fim}")]:
                ci = m.conf_int().loc[v]
                linhas.append({"definicao": dn, "segmento": rot, "variacao_anual_pct": 100 * (np.exp(m.params[v]) - 1),
                               "ic95_inf": 100 * (np.exp(ci[0]) - 1), "ic95_sup": 100 * (np.exp(ci[1]) - 1),
                               "modelo_ate": fim})
    salvar(pd.DataFrame(linhas), "C1_tendencia_segmentada")
    s = pd.read_csv(os.path.join(F.OUT, "01_serie_nacional.csv"))
    t2 = pd.read_csv(os.path.join(F.OUT, "02_tendencia_binomial_negativa.csv"))
    conc = []
    for d, rot in [("Reumática", "Reumática"), ("Não reumática", "Não reumática")]:
        x = s[s.definicao == d].set_index("ano")
        for a0, a1 in [(2008, 2024), (2008, 2025), (2013, 2024)]:
            conc.append({"definicao": d, "anos": f"{a0}–{a1}",
                         "ponta_a_ponta_bruta_n_pct_ano": 100 * ((x.n[a1] / x.n[a0]) ** (1 / (a1 - a0)) - 1),
                         "ponta_a_ponta_padronizada_pct_ano": 100 * ((x.taxa_padronizada[a1] / x.taxa_padronizada[a0])
                                                                     ** (1 / (a1 - a0)) - 1)})
        mm = t2[(t2.modelo == rot) & (t2.anos == "2008–2025")].iloc[0]
        conc.append({"definicao": d, "anos": "2008–2025 (modelo BN: idade, sexo, pandemia; todos os anos pesam)",
                     "modelo_pct_ano": mm.variacao_anual_pct})
    salvar(pd.DataFrame(conc), "C2_conciliacao_modelo_ponta_a_ponta")


# ------------------------------------------------------------------ D, E, F, G
def perfil_cid(c):
    x = c[c.cid3.isin(["I05", "I06", "I08", "I34", "I35"]) & c.ano.between(2008, 2025)]
    t = x.groupby(["cid3", "ano"]).agg(n=("N_AIH", "size"), idade_mediana=("idade", "median"),
                                       pct_fem=("sexo", lambda v: 100 * v.eq("F").mean())).reset_index()
    salvar(t, "D_perfil_idade_sexo_por_cid")


def transferencias(c):
    # SIH: motivo de saída 31/32 = transferência; IDENT 3 = AIH de continuação
    c = c.assign(transf=c.COBRANCA.isin(["31", "32"]), contin=c.IDENT.eq("3"))
    t = c[c.reum].groupby("ano").agg(n=("N_AIH", "size"), pct_transferencia_saida=("transf", "mean"),
                                     pct_continuacao=("contin", "mean")).reset_index()
    t[["pct_transferencia_saida", "pct_continuacao"]] *= 100
    t["n_sem_transf_e_contin"] = c[c.reum & ~c.transf & ~c.contin].groupby("ano").size().reindex(t.ano).to_numpy()
    salvar(t, "E_transferencias_continuacao_reumatica")
    RES["motivo_saida_codigos_top"] = c.COBRANCA.value_counts().head(8).to_dict()


def auditoria_opm(c):
    c = c.assign(com_protese=c.n_protese > 0, com_anel=c.anel > 0, sem_cec=c.cec == 0)
    t = c.groupby(["PROC_REA", "ano"]).agg(n=("N_AIH", "size"), pct_com_protese=("com_protese", "mean"),
                                           pct_2mais_proteses=("n_protese", lambda v: (v >= 2).mean()),
                                           pct_com_anel=("com_anel", "mean"), pct_sem_cec=("sem_cec", "mean")).reset_index()
    for k in ["pct_com_protese", "pct_2mais_proteses", "pct_com_anel", "pct_sem_cec"]:
        t[k] *= 100
    salvar(t, "F1_auditoria_opm_codigo_ano")
    r = c[c.PROC_REA.isin([406010692, 406010803, 406010820, "0406010692", "0406010803", "0406010820"])]
    disc = r.groupby(r.PROC_REA.astype(str).str.zfill(10)).agg(
        n=("N_AIH", "size"),
        troca_sem_protese=("n_protese", lambda v: (v == 0).mean()),
        plastica_com_protese=("n_protese", lambda v: (v > 0).mean())).reset_index()
    salvar(disc, "F2_discordancia_codigo_material")
    # prótese sem tipo: todo OPM de prótese é tipado (bio/mec); "sem tipo" = código de troca sem nenhum OPM de prótese
    tro = c[c.PROC_REA.astype(str).str.zfill(10).eq("0406010692")]
    salvar(tro.groupby("ano").agg(n=("N_AIH", "size"), pct_sem_opm_protese=("n_protese", lambda v: 100 * (v == 0).mean()))
           .reset_index(), "F3_troca_sem_material_por_ano")


def plastia_ajustada(c):
    x = c[c.cid3.eq("I05") & c.idade.between(10, 90) & c.sexo.isin(["M", "F"])].copy()
    x["sem_protese"] = (x.n_protese == 0).astype(int)
    x["t"] = x.ano - 2008
    x["regiao_res"] = x.regiao_res.fillna("Ignorada")
    lin = []
    for rot, f in [("bruta", "sem_protese ~ t"), ("idade+sexo", "sem_protese ~ t + bs(idade, df=4) + C(sexo)"),
                   ("idade+sexo+região+multivalvar", "sem_protese ~ t + bs(idade, df=4) + C(sexo) + C(regiao_res)")]:
        m = smf.glm(f, data=x, family=sm.families.Binomial()).fit(cov_type="cluster",
                                                                  cov_kwds={"groups": pd.factorize(x.CNES)[0]})
        ci = m.conf_int().loc["t"]
        lin.append({"modelo": rot, "OR_por_ano": np.exp(m.params["t"]), "ic95_inf": np.exp(ci[0]),
                    "ic95_sup": np.exp(ci[1]), "n": len(x)})
    salvar(pd.DataFrame(lin), "G_plastia_mitral_reum_ajustada_idade")


# ------------------------------------------------------------------ H
def lexis(c, pop):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pb = pop.groupby(["ano", "faixa"])["pop"].sum()
    k = contar(c[c.reum], ["ano", "fx"])
    anos = list(range(2008, 2026))
    fxs = list(range(3, F.NF))
    M = np.array([[1e5 * k.get((a, f), 0) / pb.loc[(a, f)] for a in anos] for f in fxs])
    salvar(pd.DataFrame(M, index=[F.FAIXAS[f] for f in fxs], columns=anos).reset_index().rename(columns={"index": "faixa"}),
           "H_lexis_taxa_idade_ano")
    rampa = matplotlib.colors.LinearSegmentedColormap.from_list(
        "azul", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=rampa, extent=[2007.5, 2025.5, 2.5, F.NF - 0.5])
    for coorte in (1950, 1960, 1970, 1980):  # diagonais de coorte (idade = ano − nascimento)
        xs = np.array([2008, 2025])
        ax.plot(xs, (xs - coorte - 2) / 5, color="#fcfcfb", lw=0.8, ls="--")
        y = (2025 - coorte - 2) / 5
        if 2.5 < y < F.NF - 0.5:
            ax.text(2025.6, y, f"nasc. {coorte}", fontsize=7, va="center", color="#52514e")
    ax.set_yticks(fxs)
    ax.set_yticklabels([F.FAIXAS[f].split(" ")[0] + ("+" if "mais" in F.FAIXAS[f] else "") for f in fxs], fontsize=7.5)
    ax.set_xticks(range(2008, 2026, 2))
    ax.set_ylim(2.5, F.NF - 0.5)
    ax.set_xlabel("Ano da internação")
    ax.set_ylabel("Idade (início da faixa)")
    ax.set_title("Cirurgia valvar reumática no SUS: taxa por idade e ano (/100 mil)", loc="left", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, pad=0.12)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("/100 mil hab.", fontsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(AQUI, "figuras", "fig5_lexis_reumatica.png"), dpi=300, bbox_inches="tight")


def main():
    pop = F.populacao()
    c = carregar()
    teste_decisivo(c, pop)
    das_gupta(c, pop)
    segmentada(c, pop)
    perfil_cid(c)
    transferencias(c)
    auditoria_opm(c)
    plastia_ajustada(c)
    lexis(c, pop)
    with pd.ExcelWriter(os.path.join(AQUI, "fase3_revisao.xlsx")) as w:
        for k, v in T.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    json.dump(RES, open(os.path.join(OUT, "resumo.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(RES, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
