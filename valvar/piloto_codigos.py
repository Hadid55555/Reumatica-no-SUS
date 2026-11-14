#!/usr/bin/env python3
"""Piloto: quais PROC_REA aparecem com CID valvar (I05-I09, I33-I39, Q22-Q23) nas tabelas antigas e no SIGTAP.

Lê arquivos inteiros (SHA-256 conferido) de SP, MG, PE, RS em competências de 2002, 2003, 2006, 2007 e 2009.
Uso: python valvar/piloto_codigos.py <pasta_temporaria>
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, pyarrow.parquet as pq
import sih_crvm as S

DL = sys.argv[1]
mpath = os.path.join(os.path.dirname(S.__file__), "saida_crvm", "manifest_sih_rd.json")
if not os.path.exists(mpath):
    S.baixar(f"{S.ESPELHO}/manifest.json", mpath)
man = json.load(open(mpath))["partitions"]
tarefas = [(a, m, uf) for a in (2002, 2003, 2006, 2007, 2009) for m in (3, 9) for uf in ("SP", "MG", "PE", "RS")]
CID = r"^(I0[0-9]|I3[3-9]|Q2[23])"

def um(t):
    a, m, uf = t
    k = f"{a}-{m:02d}-{uf}"
    out = []
    for of in man[k]["output_files"]:
        dest = os.path.join(DL, k + os.path.basename(of["path"]))
        S.baixar(f"{S.ESPELHO}/{of['path'][len('sih/rd/'):]}", dest)
        assert S.sha256(dest) == of["sha256"]
        d = pq.ParquetFile(dest).read(columns=["PROC_REA", "DIAG_PRINC", "MORTE", "IDADE", "COD_IDADE", "VAL_TOT"]).to_pandas()
        os.remove(dest)
        out.append(d[d.DIAG_PRINC.str.strip().str.match(CID)])
    d = pd.concat(out); d["ano"] = a
    return d

with ThreadPoolExecutor(8) as ex:
    d = pd.concat(ex.map(um, tarefas))
d["cid3"] = d.DIAG_PRINC.str[:3]
d["VAL_TOT"] = pd.to_numeric(d.VAL_TOT, errors="coerce")
d.to_csv(os.path.join(DL, "piloto_valvar.csv"), index=False)
for a, g in d.groupby("ano"):
    t = g.groupby("PROC_REA").agg(n=("cid3", "size"), val_med=("VAL_TOT", "median"),
                                  cids=("cid3", lambda s: ",".join(f"{k}:{v}" for k, v in s.value_counts().head(5).items())))
    t = t[t.val_med > 1500].sort_values("n", ascending=False)
    print(f"\n== {a} (PROC com valor mediano > R$1.500, i.e. cirúrgicos) ==")
    print(t.head(25).to_string())
