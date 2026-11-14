#!/usr/bin/env python3
"""Mulheres em idade fértil (15–44 anos) com troca valvar reumática: biológica × mecânica, 2008–2025, comparadas com
homens de 15–44 anos e com mulheres ≥45; por período e região; modelo logístico (EP robusto por hospital).
Tipo de prótese pelo material faturado no SIH-SP (OPM). Saídas: valvar/resultados_fase7/5*_protese_*.csv"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "resultados_fase7")
p = os.path.join(os.path.dirname(AQUI), "saida_valvar", "base_valvar_2008_2025.csv.gz")
c = pd.read_csv(p if os.path.exists(p) else os.path.join(AQUI, "base_valvar_2008_2025.csv.gz"),
                dtype={"N_AIH": str, "CNES": str, "DIAG_PRINC": str, "PROC_REA": str})
c = c[c.cirurgia & c.etiologia.eq("Reumática") & (c.n_protese > 0) & c.sexo.isin(["M", "F"]) & c.idade.between(15, 99)].copy()
c["mecanica"] = (c.mec > 0).astype(int)
c["grupo"] = np.select([(c.sexo == "F") & (c.idade < 45), (c.sexo == "M") & (c.idade < 45), (c.sexo == "F")],
                       ["Mulheres 15–44", "Homens 15–44", "Mulheres ≥45"], "Homens ≥45")
c["periodo"] = pd.cut(c.ano, [2007, 2012, 2017, 2021, 2025], labels=["2008–12", "2013–17", "2018–21", "2022–25"])
t = c.groupby(["grupo", "periodo"], observed=True).agg(n=("N_AIH", "size"), pct_mecanica=("mecanica", "mean")).reset_index()
t["pct_mecanica"] *= 100
t.to_csv(os.path.join(OUT, "5a_protese_por_sexo_idade_periodo.csv"), index=False)
w = c[c.grupo == "Mulheres 15–44"]
r = w.groupby(["regiao_res", "periodo"], observed=True).agg(n=("N_AIH", "size"), pct_mecanica=("mecanica", "mean")).reset_index()
r["pct_mecanica"] *= 100
r.to_csv(os.path.join(OUT, "5b_protese_mulheres_15_44_regiao.csv"), index=False)
a = w.groupby("ano").agg(n=("N_AIH", "size"), pct_mecanica=("mecanica", "mean")).reset_index()
a["pct_mecanica"] *= 100
a.to_csv(os.path.join(OUT, "5c_protese_mulheres_15_44_ano.csv"), index=False)
x = c[c.idade < 45].copy()
x["mulher"] = (x.sexo == "F").astype(int)
x["regiao_res"] = x.regiao_res.fillna("Ignorada")
m = smf.glm("mecanica ~ mulher + bs(idade, df=3) + C(periodo) + C(regiao_res) + n_protese", data=x,
            family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.CNES)[0]})
ci = m.conf_int().loc["mulher"]
print(t.round(1).to_string())
print(r.pivot(index="regiao_res", columns="periodo", values="pct_mecanica").round(1).to_string())
print("OR mecânica, mulher × homem (15–44, ajustado):", round(np.exp(m.params["mulher"]), 2), np.exp(ci).round(2).tolist())
print("n mulheres 15–44 com troca reumática:", len(w))
