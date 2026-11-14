#!/usr/bin/env python3
"""As mesmas gerações seguem sendo operadas? Taxa de cirurgia valvar reumática por coorte de nascimento × idade.

Taxa por célula (faixa etária de 5 anos × coorte de nascimento de 5 anos) = cirurgias / população, somando os anos
2008–2025 sem 2020–2021; coorte = ano da internação − idade (centro da faixa). Também: participação de cada coorte
nas cirurgias por ano. Saídas: valvar/resultados_coortes/*.csv e valvar/figuras/fig6_coortes_nascimento.png.
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import fase2_valvar as F  # noqa: E402

OUT = os.path.join(AQUI, "resultados_coortes")
os.makedirs(OUT, exist_ok=True)
p = os.path.join(os.path.dirname(AQUI), "saida_valvar", "base_valvar_2008_2025.csv.gz")
if not os.path.exists(p):
    p = os.path.join(AQUI, "base_valvar_2008_2025.csv.gz")
c = pd.read_csv(p, dtype={"N_AIH": str, "CNES": str, "DIAG_PRINC": str, "PROC_REA": str})
c = c[c.cirurgia & c.etiologia.eq("Reumática") & ~c.ano.isin([2020, 2021])].copy()
c["fx"] = F.faixa5(c.idade).to_numpy()
c = c[c.fx.between(3, 15)]
pop = F.populacao()
pb = pop[pop.faixa.between(3, 15) & pop.ano.between(2008, 2025) & ~pop.ano.isin([2020, 2021])].groupby(["ano", "faixa"])["pop"].sum().reset_index()


def coorte(ano, fx):
    return 5 * np.floor((ano - (5 * fx + 2)) / 5)  # coorte de 5 anos pelo centro da faixa


c["coorte"] = coorte(c.ano, c.fx)
pb["coorte"] = coorte(pb.ano, pb.faixa)
k = c.groupby(["fx", "coorte"]).size().rename("n")
d = pb.groupby(["faixa", "coorte"])["pop"].sum().to_frame().join(k.rename_axis(["faixa", "coorte"])).fillna(0).reset_index()
d = d[d["pop"] > 0]
d["taxa"] = 1e5 * d.n / d["pop"]
d["idade"] = d.faixa.map(lambda f: F.FAIXAS[f])
d.to_csv(os.path.join(OUT, "taxa_coorte_idade.csv"), index=False)

# razão entre coortes na mesma idade: cada coorte ÷ coorte de 15 anos antes
w = d.pivot(index="faixa", columns="coorte", values="taxa")
lin = []
for f in w.index:
    for co in w.columns:
        if co - 15 in w.columns and pd.notna(w.loc[f, co]) and pd.notna(w.loc[f, co - 15]):
            lin.append({"idade": F.FAIXAS[f], "coorte": int(co), "coorte_15_anos_antes": int(co - 15),
                        "razao": w.loc[f, co] / w.loc[f, co - 15]})
pd.DataFrame(lin).to_csv(os.path.join(OUT, "razao_mesma_idade_15_anos.csv"), index=False)

# participação por coorte (década) nas cirurgias de cada ano
c2 = c.assign(decada=(10 * np.floor((c.ano - c.idade) / 10)).astype(int))
sh = c2.pivot_table(index="ano", columns="decada", values="N_AIH", aggfunc="size", fill_value=0)
sh = (100 * sh.div(sh.sum(axis=1), axis=0)).round(1)
sh.to_csv(os.path.join(OUT, "participacao_decada_nascimento.csv"))
nasc75 = c.assign(antes=(c.ano - c.idade) < 1975).groupby("ano").antes.mean().mul(100).round(1)
nasc75.to_csv(os.path.join(OUT, "pct_nascidos_antes_1975.csv"))

# figura: em cada idade, taxa da geração nascida 15 anos depois ÷ taxa da geração anterior
r = pd.DataFrame(lin)
fig, ax = plt.subplots(figsize=(6.8, 3.8))
y = np.arange(len(r))[::-1]
ax.axvline(1, color="#52514e", lw=0.8)
ax.hlines(y, r.razao, 1, color="#86b6ef", lw=2)
ax.plot(r.razao, y, "o", color="#2a78d6", ms=7)
for yi, (_, row) in zip(y, r.iterrows()):
    ax.annotate(f"{row.razao:.2f}".replace(".", ","), (row.razao, yi), xytext=(-8, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7.5, color="#0b0b0b")
ax.set_yticks(y)
ax.set_yticklabels([f"{i.split(' ')[0]}–{i.split(' ')[2]} anos: nasc. {c2}–{c2 + 4} ÷ {c1}–{c1 + 4}"
                    for i, c2, c1 in zip(r.idade, r.coorte, r.coorte_15_anos_antes)], fontsize=7.5)
ax.set_xlim(0, 1.1)
ax.set_xlabel("Razão de taxas (geração mais nova ÷ geração 15 anos mais velha, na mesma idade)")
ax.set_title("Gerações mais novas operam menos; as nascidas até ~1970 operam igual", loc="left", fontsize=9.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="x", color="#e4e3df", lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(AQUI, "figuras", "fig6_coortes_nascimento.png"), dpi=300, bbox_inches="tight")
print(w.round(2).to_string())
print(pd.DataFrame(lin).round(2).to_string())
print(sh.loc[[2008, 2013, 2019, 2025]].to_string())
print(nasc75.to_dict())
