#!/usr/bin/env python3
"""Sensibilidade: excluir hospital-anos com "CID padrão" (CID principal de 3 caracteres mais frequente em ≥70% ou ≥80%
das cirurgias valvares do hospital no ano, com ≥20 AIH) e recalcular a razão geracional (2023–25 ÷ 2008–10).
Sugestão vinda da sessão de prótese valvar (mesmo repositório). Saída: valvar/resultados_fase5/6_sens_cid_padrao.csv"""
import os
import sys

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import fase2_valvar as F  # noqa: E402
import fase5_congelamento as G  # noqa: E402

pop = F.populacao()
c = G.carregar()
cir = c[c.cirurgia].assign(c3=lambda x: x.DIAG_PRINC.str[:3])
h = cir.groupby(["CNES", "ano"]).agg(n=("N_AIH", "size"), top=("c3", lambda v: v.value_counts(normalize=True).iloc[0])).reset_index()
lin = []
for lim in (None, 0.8, 0.7):
    ruins = h[(h.n >= 20) & (h.top >= lim)] if lim else h.iloc[0:0]
    x = cir.merge(ruins[["CNES", "ano"]].assign(fora=1), on=["CNES", "ano"], how="left")
    x = x[x.fora.isna() & (x.fx >= 0)]
    for nome, d in [("Reumática", x[x.etiologia.eq("Reumática")]), ("I35", x[x.c3.eq("I35")])]:
        for r in G.razoes(d, d, pop, G.A1, G.A2, nome):
            if r["faixa"] in G.GRUPOS:
                lin.append({"limiar_cid_padrao": lim or "sem exclusão", "hospital_anos_excluidos": len(ruins),
                            "aih_excluidas": int(ruins.n.sum()), **r})
t = pd.DataFrame(lin)
t.to_csv(os.path.join(AQUI, "resultados_fase5", "6_sens_cid_padrao.csv"), index=False)
print(t[["limiar_cid_padrao", "analise", "faixa", "razao", "ic95_inf", "ic95_sup"]].round(2).to_string())
