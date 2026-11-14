#!/usr/bin/env python3
"""Piloto SIH-SP: as AIH de cirurgias múltiplas/sequenciais (0415010012, 0415020034) com CID valvar (I05–I09,
I34–I36) têm ato valvar nos serviços profissionais? MG e SP, competências mar/jun/set de 2024 (SHA-256 conferido).
Uso: python valvar/piloto_sp_multiplas.py <pasta_temporaria>"""
import glob, json, os, sys
import pandas as pd, pyarrow.parquet as pq
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import sih_crvm as S  # noqa: E402

DL = sys.argv[1]
mp = os.path.join(DL, "manifest_sp.json")
if not os.path.exists(mp):
    S.baixar("https://data.sidneybissoli.com/sih/sp/manifest.json", mp)
m = json.load(open(mp))["partitions"]
rd = pd.concat(pd.read_csv(f, dtype=str, keep_default_na=False)
               for f in glob.glob(os.path.join(RAIZ, "saida_valvar", "parciais", "RD[MS][GP]24*.csv")))
alvo = rd[rd.PROC_REA.isin(["0415010012", "0415020034"]) & rd.DIAG_PRINC.str.match(r"I0[5-9]|I3[4-6]")]
res = []
for uf in ("MG", "SP"):
    for mes in ("03", "06", "09"):
        of = m[f"2024-{mes}-{uf}"]["output_files"][0]
        dest = os.path.join(DL, "sp.parquet")
        S.baixar(f"https://data.sidneybissoli.com/{of['path']}", dest)
        assert S.sha256(dest) == of["sha256"]
        t = pq.ParquetFile(dest).read(columns=["SP_NAIH", "SP_ATOPROF", "SP_PF_CBO"]).to_pandas()
        res.append(t[t.SP_NAIH.isin(alvo.N_AIH)])
        os.remove(dest)
s = pd.concat(res)
VALVA = {"0406010692", "0406010803", "0406010820", "0406010811", "0406011206", "0406010340"}
print("AIH alvo (RD, MG+SP 2024):", len(alvo), "| encontradas no SP (3 competências):", s.SP_NAIH.nunique(),
      "| com ato valvar:", s[s.SP_ATOPROF.isin(VALVA)].SP_NAIH.nunique())
print(s[s.SP_ATOPROF.str.startswith("0406")].groupby("SP_ATOPROF").SP_NAIH.nunique().sort_values(ascending=False).head(15))
