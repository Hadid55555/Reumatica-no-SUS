#!/usr/bin/env python3
"""SIM 2016–2024: óbitos com doença reumática (I00–I09) como causa básica ou em qualquer linha da DO.

Fonte: compilação pública do SIM (Hugging Face jgchaicoski/datasus_sim, SIM_CONSOLIDADO_DECODED.parquet, CC-BY-SA-4.0;
13.220.321 DO de 2016–2024, 27 UFs). Não é fonte oficial e não tem SHA-256 por arquivo; o FTP do DATASUS não é
alcançável deste ambiente. Totais anuais conferidos com os publicados pelo DATASUS (ver conferencia_totais.csv).
Lê só as colunas necessárias, por grupo de linhas. Uso: python valvar/sim/extrair_sim.py
"""
import os
import re

import fsspec
import pandas as pd
import pyarrow.parquet as pq

AQUI = os.path.dirname(os.path.abspath(__file__))
URL = "https://huggingface.co/datasets/jgchaicoski/datasus_sim/resolve/main/SIM_CONSOLIDADO_DECODED.parquet"
COLS = ["ANO_ARQUIVO", "DTOBITO", "CAUSABAS", "LINHAA", "LINHAB", "LINHAC", "LINHAD", "LINHAII", "IDADE", "SEXO",
        "RACACOR", "CODMUNRES", "UF_ORIGEM"]
RE = re.compile(r"I0[0-9]")
p = pq.ParquetFile(fsspec.open(URL, block_size=8 << 20).open())
partes, tot = [], []
for i in range(p.metadata.num_row_groups):
    d = p.read_row_group(i, columns=COLS).to_pandas()
    tot.append(d.groupby("ANO_ARQUIVO").size())
    linhas = d[["LINHAA", "LINHAB", "LINHAC", "LINHAD", "LINHAII"]].fillna("").astype(str).agg(" ".join, axis=1)
    m = d.CAUSABAS.fillna("").str.match(r"^I0[0-9]") | linhas.str.contains(RE)
    partes.append(d[m])
    if i % 20 == 0:
        print(i, "/", p.metadata.num_row_groups, flush=True)
r = pd.concat(partes)
r.to_csv(os.path.join(AQUI, "sim_reumatica_2016_2024.csv.gz"), index=False)
pd.concat(tot).groupby(level=0).sum().rename("obitos_total").to_csv(os.path.join(AQUI, "conferencia_totais.csv"))
print("DO lidas:", int(pd.concat(tot).sum()), "| com I00–I09:", len(r))
