#!/usr/bin/env python3
"""Envolvimento/intervenção tricúspide nas cirurgias valvares reumáticas, 2008–2025.
Marcadores: (A) ato de correção tricúspide no SIH-SP (0406010340, 0406011400); (B) CID principal com tricúspide
(I07.x reumática tricúspide; I08.1 mitral+tricúspide; I08.2 aórtica+tricúspide; I08.3 mitral+aórtica+tricúspide);
(C) código multivalvar (0406010820) ou cirurgia múltipla com CID de B. "Tricúspide provável" = A ou C.
Saída: valvar/resultados_fase7/6_tricuspide.csv"""
import glob
import os

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
p = os.path.join(RAIZ, "saida_valvar", "base_valvar_2008_2025.csv.gz")
c = pd.read_csv(p if os.path.exists(p) else os.path.join(AQUI, "base_valvar_2008_2025.csv.gz"),
                dtype={"N_AIH": str, "CNES": str, "DIAG_PRINC": str, "PROC_REA": str})
c = c[c.cirurgia & c.etiologia.eq("Reumática")].copy()
s = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, usecols=["SP_NAIH", "SP_ATOPROF"])
               for f in glob.glob(os.path.join(RAIZ, "saida_valvar", "sp_parciais", "SP*.csv"))])
ato = set(s[s.SP_ATOPROF.isin(["0406010340", "0406011400"])].SP_NAIH)
c["A_ato_sp"] = c.N_AIH.isin(ato)
c["B_cid_tric"] = c.DIAG_PRINC.str[:3].eq("I07") | c.DIAG_PRINC.str[:4].isin(["I081", "I082", "I083"])
multi = c.PROC_REA.astype(str).str.zfill(10).eq("0406010820") | c.tipo.eq("Múltipla com ato valvar")
c["C_multi_cid_tric"] = multi & c.B_cid_tric
c["tric_provavel"] = c.A_ato_sp | c.C_multi_cid_tric
c["mitral"] = c.DIAG_PRINC.str[:3].isin(["I05", "I08"])
t = c.groupby("ano").agg(n=("N_AIH", "size"), A_ato_sp=("A_ato_sp", "mean"), B_cid_tric=("B_cid_tric", "mean"),
                         tric_provavel=("tric_provavel", "mean"), pct_multiplas=("tipo", lambda v: v.eq("Múltipla com ato valvar").mean())).reset_index()
for k in ["A_ato_sp", "B_cid_tric", "tric_provavel", "pct_multiplas"]:
    t[k] *= 100
t.to_csv(os.path.join(AQUI, "resultados_fase7", "6_tricuspide.csv"), index=False)
print(t.round(1).to_string())
x = c[c.ano >= 2018]
print("2018–2025, tricúspide provável por idade:", x.groupby(pd.cut(x.idade, [0, 39, 59, 120], labels=["<40", "40–59", "60+"]), observed=True).tric_provavel.mean().mul(100).round(1).to_dict())
print("por região:", x.groupby("regiao_res").tric_provavel.mean().mul(100).round(1).to_dict())
print("mortalidade hospitalar 2018–25, com × sem tricúspide provável:", x.groupby("tric_provavel").obito.mean().mul(100).round(1).to_dict())
print("CID I08 por 4º dígito (2018–25):", x[x.DIAG_PRINC.str[:3].eq("I08")].DIAG_PRINC.str[:4].value_counts().to_dict())
