#!/usr/bin/env python3
"""Fase 10 (complemento) — sensibilidade da mortalidade reumática à melhora da certificação de óbito.

Fração de óbitos com causa mal definida (capítulo XVIII, R00–R99) por região e ano (TabNet; sim/tabnet_mal_definidas_regiao.csv).
Redistribuição proporcional: óbitos I00–I09 corrigidos = observados ÷ (1 − fração mal definida da região no ano).
Razões de taxas 2022–24 e 2017–19 ÷ 2003–05 por idade (faixas TabNet) e região, observadas e corrigidas.
Saída: resultados_fase10/G3_mortalidade_corrigida_mal_definidas.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import fase2_valvar as F  # noqa: E402
import fase7_extras as P  # noqa: E402

A0, PRE, FIM = [2003, 2004, 2005], [2017, 2018, 2019], [2022, 2023, 2024]
md = pd.read_csv(os.path.join(AQUI, "sim", "tabnet_mal_definidas_regiao.csv"))
md["regiao"] = md.regiao.str.replace("Região ", "")
fat = md.set_index(["regiao", "ano"]).pct_mal_definidos.div(100).rsub(1).rdiv(1)  # 1/(1−f)
m = pd.read_csv(os.path.join(AQUI, "sim", "tabnet_obitos_reumaticos_2000_2024.csv"))
m = m[(m.causa == "I00–I09") & m.faixa_tabnet.isin(P.MAPA) & m.ano.isin(A0 + PRE + FIM)].assign(g=lambda x: x.faixa_tabnet.map(P.MAPA))
m["obitos_corr"] = m.obitos * [fat.loc[(r, a)] for r, a in zip(m.regiao, m.ano)]
pop = F.populacao().assign(g=lambda x: x.faixa.map(P.POPF))
lin = []
for reg in ["Brasil", "Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
    x = m if reg == "Brasil" else m[m.regiao == reg]
    pr = (pop if reg == "Brasil" else pop[pop.regiao == reg]).groupby(["ano", "g"])["pop"].sum()
    for fim, anos in [("2022–24", FIM), ("2017–19", PRE)]:
        for gr, gs in P.G3.items():
            p1 = sum(pr.loc[(a, g)] for a in A0 for g in gs)
            p2 = sum(pr.loc[(a, g)] for a in anos for g in gs)
            q = {}
            for col in ("obitos", "obitos_corr"):
                n1 = x[x.ano.isin(A0) & x.g.isin(gs)][col].sum()
                n2 = x[x.ano.isin(anos) & x.g.isin(gs)][col].sum()
                q[col] = (n2 / p2) / (n1 / p1)
            n1 = x[x.ano.isin(A0) & x.g.isin(gs)].obitos.sum()
            n2 = x[x.ano.isin(anos) & x.g.isin(gs)].obitos.sum()
            se = np.sqrt(1 / n1 + 1 / n2)
            lin.append({"regiao": reg, "fim": fim, "grupo": gr, "razao_observada": q["obitos"], "razao_corrigida": q["obitos_corr"],
                        "ic95_inf_corrigida": q["obitos_corr"] * np.exp(-1.96 * se), "ic95_sup_corrigida": q["obitos_corr"] * np.exp(1.96 * se)})
t = pd.DataFrame(lin)
t.to_csv(os.path.join(AQUI, "resultados_fase10", "G3_mortalidade_corrigida_mal_definidas.csv"), index=False)
print(t.round(2).to_string())
