#!/usr/bin/env python3
"""Conferência independente da Fase 2 (módulo csv, sem pandas e sem fase2_valvar.py): cirurgias valvares com CID
reumático (I05–I09) por ano da internação, 2008–2025 = AIH com código valvar aberto + AIH de cirurgia múltipla com
ato valvar ou prótese/anel no SIH-SP. Compara com resultados_fase2/01_serie_nacional.csv."""
import csv
import glob
import gzip
import os
from collections import Counter

AQUI = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(os.path.dirname(AQUI), "saida_valvar")
ABERTA = {"0406010692", "0406010803", "0406010820", "0406011206", "0406010811", "0406010340", "0406010358",
          "0406010552", "0406010846"}
MULT = {"0415010012", "0415020034"}
VALVA_SP = (ABERTA - {"0406010846"}) | {"0702040541", "0702040550", "0702040568", "0702040576", "0702040029"}
REUM = ("I05", "I06", "I07", "I08", "I09")

com_valva = set()
for f in glob.glob(os.path.join(SAIDA, "sp_parciais", "SP*.csv")):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if r["SP_ATOPROF"].strip() in VALVA_SP:
            com_valva.add(r["SP_NAIH"].strip())
n, vistos = Counter(), set()
for f in sorted(glob.glob(os.path.join(SAIDA, "parciais", "RD*.csv"))):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        aih = r["N_AIH"].strip()
        if r["IDENT"].strip() == "5" or aih in vistos:
            continue
        vistos.add(aih)
        ano, proc, cid = r["DT_INTER"].strip()[:4], r["PROC_REA"].strip(), r["DIAG_PRINC"].strip()[:3]
        if "2008" <= ano <= "2025" and cid in REUM and (proc in ABERTA or (proc in MULT and aih in com_valva)):
            n[int(ano)] += 1
ref = {int(r["ano"]): int(r["n"]) for r in csv.DictReader(open(os.path.join(AQUI, "resultados_fase2", "01_serie_nacional.csv")))
       if r["definicao"] == "Reumática"}
dif = {a: (n[a], ref[a]) for a in ref if n[a] != ref[a]}
print("anos comparados:", len(ref), "| divergências:", dif if dif else 0)
print("total reumática 2008–2025:", sum(n.values()))
