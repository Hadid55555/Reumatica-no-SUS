#!/usr/bin/env python3
"""Mortalidade por doença reumática (SIM oficial via TabNet), 2000–2024: taxa padronizada por idade (padrão Brasil 2010),
joinpoint (sem 2020–21), razão por idade 2022–24 ÷ 2003–05 e por região. Faixas do TabNet: 0–4, 5–9, 10–14, 15–19,
20–29, …, 80+. Saídas: valvar/resultados_sim_longo/*.csv. Uso: python valvar/sim/mortalidade_longa.py"""
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
V = os.path.dirname(AQUI)
sys.path.insert(0, V)
import fase2_valvar as F  # noqa: E402
import fase4_revisao2 as J  # noqa: E402

OUT = os.path.join(V, "resultados_sim_longo")
os.makedirs(OUT, exist_ok=True)
MAPA = {"Menor 1 ano": "0-4", "1 a 4 anos": "0-4", "5 a 9 anos": "5-9", "10 a 14 anos": "10-14", "15 a 19 anos": "15-19",
        "20 a 29 anos": "20-29", "30 a 39 anos": "30-39", "40 a 49 anos": "40-49", "50 a 59 anos": "50-59",
        "60 a 69 anos": "60-69", "70 a 79 anos": "70-79", "80 anos e mais": "80+"}
ORD = ["0-4", "5-9", "10-14", "15-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
POPF = {0: "0-4", 1: "5-9", 2: "10-14", 3: "15-19", 4: "20-29", 5: "20-29", 6: "30-39", 7: "30-39", 8: "40-49", 9: "40-49",
        10: "50-59", 11: "50-59", 12: "60-69", 13: "60-69", 14: "70-79", 15: "70-79", 16: "80+"}

d = pd.read_csv(os.path.join(AQUI, "tabnet_obitos_reumaticos_2000_2024.csv"))
d = d[d.faixa_tabnet.isin(MAPA)].assign(g=lambda x: x.faixa_tabnet.map(MAPA))
d = d.groupby(["causa", "ano", "regiao", "g"]).obitos.sum().reset_index()
pop = F.populacao().assign(g=lambda x: x.faixa.map(POPF))
pr = pop.groupby(["regiao", "ano", "g"])["pop"].sum()
pb = pop.groupby(["ano", "g"])["pop"].sum()
padrao = pb.loc[2010].reindex(ORD)
lin = []
for causa in d.causa.unique():
    for reg in ["Brasil", "Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
        x = d[d.causa == causa] if reg == "Brasil" else d[(d.causa == causa) & (d.regiao == reg)]
        k = x.groupby(["ano", "g"]).obitos.sum()
        for a in range(2000, 2025):
            den = (pb.loc[a] if reg == "Brasil" else pr.loc[(reg, a)]).reindex(ORD)
            ka = k.get(a, pd.Series(dtype=float))
            t, v = J.padronizada(ka, den, padrao, ORD)
            lin.append({"causa": causa, "regiao": reg, "ano": a, "obitos": int(ka.sum()), "taxa_padronizada": t, "var": v})
s = pd.DataFrame(lin)
s.drop(columns="var").to_csv(os.path.join(OUT, "1_mortalidade_padronizada_2000_2024.csv"), index=False)
jp = []
for (causa, reg), g in s.groupby(["causa", "regiao"]):
    h = g[g.ano.between(2003, 2024) & ~g.ano.isin([2020, 2021])]
    jps, segs, aa = J.joinpoint(h.ano, h.taxa_padronizada, h["var"], max_jp=2)
    jp.append({"causa": causa, "regiao": reg, "joinpoints": ",".join(str(int(j)) for j in jps),
               "segmentos": "; ".join(f"{x['inicio']}–{x['fim']}: {x['APC']:+.1f}%" for x in segs),
               "AAPC_2003_2024": aa[0], "ic95_inf": aa[1], "ic95_sup": aa[2]})
pd.DataFrame(jp).to_csv(os.path.join(OUT, "2_joinpoint_mortalidade_2003_2024.csv"), index=False)
# razão por idade: 2022–24 ÷ 2003–05 (mesma idade, ~19 anos de diferença)
rz = []
x = d[d.causa == "I00–I09"]
k = x.groupby(["ano", "g"]).obitos.sum()
grupos = {"5-19": ["5-9", "10-14", "15-19"], "20-39": ["20-29", "30-39"], "40-59": ["40-49", "50-59"], "60+": ["60-69", "70-79", "80+"]}
for nome, gs in [(g, [g]) for g in ORD[1:]] + list(grupos.items()):
    n1 = sum(k.get((a, g), 0) for a in (2003, 2004, 2005) for g in gs)
    n2 = sum(k.get((a, g), 0) for a in (2022, 2023, 2024) for g in gs)
    p1 = sum(pb.loc[(a, g)] for a in (2003, 2004, 2005) for g in gs)
    p2 = sum(pb.loc[(a, g)] for a in (2022, 2023, 2024) for g in gs)
    r = (n2 / p2) / (n1 / p1)
    se = np.sqrt(1 / n1 + 1 / n2)
    rz.append({"faixa": nome, "obitos_2003_05": n1, "obitos_2022_24": n2, "razao": r, "ic95_inf": r * np.exp(-1.96 * se),
               "ic95_sup": r * np.exp(1.96 * se)})
pd.DataFrame(rz).to_csv(os.path.join(OUT, "3_razao_idade_mortalidade_2022_24_x_2003_05.csv"), index=False)
print(s[(s.causa == "I00–I09") & (s.regiao == "Brasil")][["ano", "obitos", "taxa_padronizada"]].round(2).T.to_string())
print(pd.DataFrame(jp).round(2).to_string())
print(pd.DataFrame(rz).round(2).to_string())
