#!/usr/bin/env python3
"""Fase 10 (complemento) — cirurgia × óbito por UF de residência (revisor 1, crítica 3).

Taxas padronizadas por idade (faixas do TabNet a partir de 5 anos; padrão Brasil 2010) de cirurgia reumática ampliada
(SIH) e de óbito com causa básica I00–I09 (SIM/TabNet por UF), 2003–05 × 2022–24 e 2003–05 × 2017–19; razões e correlação
de Spearman entre as razões das duas medidas, com sensibilidades (sem RO, artefato de codificação; só UF com ≥30 óbitos
em cada período). Relaciona também a razão do óbito com a mudança do volume de cirurgia valvar realizada na UF.
Ecológico e exploratório. Saídas: resultados_fase10/G1_uf_cirurgia_obito.csv, G2_uf_correlacoes.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import fase2_valvar as F  # noqa: E402
import fase6_serie_longa as L  # noqa: E402
import fase7_extras as P  # noqa: E402

OUT = os.path.join(AQUI, "resultados_fase10")
A0, PRE, FIM = [2003, 2004, 2005], [2017, 2018, 2019], [2022, 2023, 2024]


def main():
    pop = F.populacao().assign(g=lambda x: x.faixa.map(P.POPF))
    gs = [g for g in ["5-9", "10-14", "15-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]]
    padrao = pop[pop.ano == 2010].groupby("g")["pop"].sum().reindex(gs)
    w = padrao / padrao.sum()
    m = pd.read_csv(os.path.join(AQUI, "sim", "tabnet_obitos_reumaticos_uf.csv"))
    m = m[m.faixa_tabnet.isin(P.MAPA)].assign(g=lambda x: x.faixa_tabnet.map(P.MAPA))
    c = L.montar()
    ufres = pd.concat([pd.read_csv(os.path.join(AQUI, f), dtype=str, usecols=["N_AIH", "uf_res"])
                       for f in ("base_valvar_2003_2007.csv.gz", "base_valvar_2008_2025.csv.gz")]).drop_duplicates("N_AIH").set_index("N_AIH").uf_res
    c["uf_res"] = ufres.reindex(c.N_AIH).to_numpy()
    c = c[c.reum_ampliada & (c.fx >= 0)].assign(g=lambda x: x.fx.map(P.POPF))
    ks = {"cirurgia": c.groupby(["uf_res", "ano", "g"]).size(), "obito": m.groupby(["uf", "ano", "g"]).obitos.sum()}
    den = pop.groupby(["uf", "ano", "g"])["pop"].sum()
    lin = []
    for uf in sorted(pop.uf.unique()):
        row = {"uf": uf, "regiao": F.UF_REG[uf]}
        for med, k in ks.items():
            for rot, anos in [("2003_05", A0), ("2017_19", PRE), ("2022_24", FIM)]:
                n = np.array([sum(k.get((uf, a, g), 0) for a in anos) for g in gs], float)
                p = np.array([sum(den.loc[(uf, a, g)] for a in anos) for g in gs])
                row[f"{med}_n_{rot}"] = int(n.sum())
                row[f"{med}_taxa_{rot}"] = 1e5 * float((n / p * w.to_numpy()).sum())
                row[f"{med}_var_{rot}"] = 1e10 * float((n / p ** 2 * w.to_numpy() ** 2).sum())
            for fim in ("2017_19", "2022_24"):
                t0, t1 = row[f"{med}_taxa_2003_05"], row[f"{med}_taxa_{fim}"]
                rr = t1 / t0 if t0 > 0 else np.nan
                se = np.sqrt(row[f"{med}_var_2003_05"] / t0 ** 2 + row[f"{med}_var_{fim}"] / t1 ** 2) if t0 > 0 and t1 > 0 else np.nan
                row[f"{med}_razao_{fim}"], row[f"{med}_ic95_inf_{fim}"], row[f"{med}_ic95_sup_{fim}"] = rr, rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se)
        lin.append(row)
    t = pd.DataFrame(lin)
    cap = pd.read_csv(os.path.join(OUT, "..", "resultados_fase9", "C2_uf_capacidade_cirurgica.csv")).set_index("uf").razao_volume
    t["razao_volume_valvar_na_uf"] = t.uf.map(cap)
    t.drop(columns=[c_ for c_ in t.columns if "_var_" in c_]).to_csv(os.path.join(OUT, "G1_uf_cirurgia_obito.csv"), index=False)
    cor = []
    for fim in ("2022_24", "2017_19"):
        for rot, msk in [("todas as UF", t.uf.notna()), ("sem RO", t.uf != "RO"),
                         ("UF com ≥30 óbitos em cada período, sem RO", (t.uf != "RO") & (t.obito_n_2003_05 >= 30) & (t[f"obito_n_{fim}"] >= 30))]:
            x = t[msk].dropna(subset=[f"cirurgia_razao_{fim}", f"obito_razao_{fim}"])
            r, p = stats.spearmanr(x[f"cirurgia_razao_{fim}"], x[f"obito_razao_{fim}"])
            cor.append({"fim": fim, "amostra": rot, "n_uf": len(x), "comparacao": "razão cirurgia × razão óbito", "spearman": r, "p": p})
            y = x[np.isfinite(x.razao_volume_valvar_na_uf)]
            r, p = stats.spearmanr(y.razao_volume_valvar_na_uf, y[f"obito_razao_{fim}"])
            cor.append({"fim": fim, "amostra": rot, "n_uf": len(y), "comparacao": "volume valvar na UF × razão óbito", "spearman": r, "p": p})
    g = pd.DataFrame(cor)
    g.to_csv(os.path.join(OUT, "G2_uf_correlacoes.csv"), index=False)
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
