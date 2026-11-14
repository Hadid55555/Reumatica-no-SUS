#!/usr/bin/env python3
"""Conferência independente da Fase 1: recontagem direta dos parciais crus com o módulo csv (sem pandas e sem o
código de fase1_valvar.py). Cirurgia valvar aberta por ano da internação × grupo de CID; compara com
resultados_fase1/1_ano_grupo.csv."""
import csv
import glob
import os
from collections import Counter

AQUI = os.path.dirname(os.path.abspath(__file__))
ABERTA = {"0406010692", "48010138", "32020015", "0406010803", "48010120", "32026013", "0406010820", "48010146",
          "32041012", "32019017", "0406011206", "48010154", "32042019", "0406010811", "48011185", "0406010340",
          "0406010358", "0406010552", "0406010846"}
GR = {**{k: "Reumática" for k in ("I05", "I06", "I07", "I08", "I09")}, **{k: "Não reumática" for k in ("I34", "I35", "I36", "I37")},
      "I33": "Endocardite"}
cont, vistos = Counter(), set()
for f in sorted(glob.glob(os.path.join(os.path.dirname(AQUI), "saida_valvar", "parciais", "RD*.csv"))):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if r["IDENT"].strip() == "5" or r["N_AIH"] in vistos:
            continue
        vistos.add(r["N_AIH"])
        ano = r["DT_INTER"].strip()[:4]
        if r["PROC_REA"].strip() in ABERTA and "2000" <= ano <= "2025":
            cont[(int(ano), GR.get(r["DIAG_PRINC"].strip()[:3], "Outros"))] += 1
            cont[(int(ano), "Total")] += 1
ref = {}
for r in csv.DictReader(open(os.path.join(AQUI, "resultados_fase1", "1_ano_grupo.csv"))):
    for g in ("Reumática", "Não reumática", "Endocardite", "Total"):
        ref[(int(r["ano"]), g)] = int(float(r.get(g) or 0))
dif = {k: (cont.get(k, 0), v) for k, v in ref.items() if cont.get(k, 0) != v}
print("células comparadas:", len(ref), "| divergências:", len(dif), dif if dif else "")
print("total cirurgia aberta 2000–2025:", sum(v for (a, g), v in cont.items() if g == "Total"))
