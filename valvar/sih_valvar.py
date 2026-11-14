#!/usr/bin/env python3
"""Cirurgia valvar e doença valvar/reumática no SIH/SUS (RD), competências jan/2000 a jun/2026.

Reaproveita sih_crvm.py (download do espelho healthbr-data com SHA-256, parciais por arquivo, cobertura,
retomável). Só troca o filtro e as colunas. Um registro entra se:
  (a) PROC_REA está em PROCS_VALVA (códigos valvares do SIGTAP e das tabelas SIH antigas 32/48), ou
  (b) DIAG_PRINC é I00–I09 (febre reumática aguda e doença reumática crônica), I33–I39 (endocardite e
      valvopatias não reumáticas) ou Q22–Q23 (valvopatias congênitas), com qualquer procedimento.
Assim dá para medir tanto a cirurgia quanto códigos de procedimento não previstos (ex.: cirurgias múltiplas).

Uso: python valvar/sih_valvar.py [--workers 8]
"""
import argparse
import os
import re
import sys
from types import SimpleNamespace

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
import pandas as pd  # noqa: E402
import sih_crvm as S  # noqa: E402

# Equivalência empírica (piloto_codigos.py: mesmo perfil de CID e valor; troca de tabela jul/2004 e jan/2008).
PROCS_VALVA = {
    # SIGTAP (2008+)
    "0406010692": "Implante de prótese valvar",
    "0406010803": "Plástica valvar",
    "0406010811": "Plástica valvar c/ revascularização miocárdica",
    "0406010820": "Plástica valvar e/ou troca valvar múltipla",
    "0406011206": "Troca valvar c/ revascularização miocárdica",
    "0406010340": "Correção de insuficiência da válvula tricúspide",
    "0406010358": "Correção de insuficiência mitral congênita",
    "0406010552": "Implante c/ troca de posição de valvas (Ross)",
    "0406010846": "Reconstrução da raiz da aorta c/ tubo valvado",
    "0406030120": "Valvuloplastia mitral percutânea",
    "0406030111": "Valvuloplastia aórtica percutânea",
    "0406030138": "Valvuloplastia pulmonar percutânea",
    "0406030146": "Valvuloplastia tricúspide percutânea",
    "0406030162": "Implante percutâneo de valva aórtica (TAVI; nome a confirmar)",
    # Tabela SIH "48" (jul/2004–dez/2007)
    "48010138": "Implante de prótese valvar (tabela 48)",
    "48010120": "Plástica valvar (tabela 48)",
    "48010146": "Plástica valvar e/ou troca valvar múltipla (tabela 48)",
    "48010154": "Troca valvar c/ revascularização miocárdica (tabela 48)",
    "48011185": "Plástica valvar c/ revascularização miocárdica (tabela 48)",
    "48030120": "Valvuloplastia mitral percutânea (tabela 48)",
    "48030040": "Valvuloplastia pulmonar percutânea (tabela 48; provável)",
    "48030058": "Valvuloplastia aórtica percutânea (tabela 48; provável)",
    # Tabela SIH "32" (até jun/2004)
    "32020015": "Implante de prótese valvar (tabela 32; provável)",
    "32041012": "Plástica e/ou troca valvar múltipla (tabela 32; provável)",
    "32019017": "Valvar, código antigo até 2002 (tabela 32; a confirmar)",
    "32026013": "Valvar, código antigo até 2002 (tabela 32; a confirmar)",
    "32042019": "Troca valvar c/ revascularização miocárdica (tabela 32)",
    "32024010": "Valvotomia/valvuloplastia (tabela 32; a confirmar)",
}
CID_RE = r"^(I0[0-9]|I3[3-9]|Q2[23])"
COLUNAS = ["ANO_CMPT", "MES_CMPT", "N_AIH", "IDENT", "PROC_REA", "PROC_SOLIC", "MORTE", "DIAS_PERM", "IDADE",
           "COD_IDADE", "SEXO", "RACA_COR", "DT_INTER", "DT_SAIDA", "DIAG_PRINC", "DIAG_SECUN"] + \
          [f"DIAGSEC{i}" for i in range(1, 10)] + \
          ["CID_MORTE", "CAR_INT", "CNES", "CGC_HOSP", "MUNIC_MOV", "MUNIC_RES", "COBRANCA", "VAL_TOT", "UTI_MES_TO"]


def ler_espelho(chave, info, tmpdir):
    import pyarrow.parquet as pq

    tabelas, total = [], 0
    for of in info["output_files"]:
        destino = os.path.join(tmpdir, chave + "_" + os.path.basename(of["path"]))
        S.baixar(f"{S.ESPELHO}/{of['path'][len('sih/rd/'):]}", destino)
        try:
            if S.sha256(destino) != of["sha256"]:
                raise ValueError(f"SHA-256 divergente em {of['path']}")
            arq = pq.ParquetFile(destino)
            presentes = [c for c in COLUNAS if c in arq.schema_arrow.names]
            d = arq.read(columns=presentes).to_pandas(ignore_metadata=True)
            total += len(d)
            proc = d["PROC_REA"].astype(str).str.strip()
            cid = d["DIAG_PRINC"].astype(str).str.strip()
            tabelas.append(d[proc.isin(PROCS_VALVA) | cid.str.match(CID_RE)])
        finally:
            os.remove(destino)
    df = pd.concat(tabelas, ignore_index=True) if tabelas else pd.DataFrame()
    return df, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    S.COLUNAS = COLUNAS
    S.ler_espelho = ler_espelho
    saida = os.path.join(RAIZ, "saida_valvar")
    os.makedirs(saida, exist_ok=True)
    man = os.path.join(RAIZ, "saida_crvm", "manifest_sih_rd.json")
    if os.path.exists(man) and not os.path.exists(os.path.join(saida, "manifest_sih_rd.json")):
        os.symlink(man, os.path.join(saida, "manifest_sih_rd.json"))
    args = SimpleNamespace(ini=2000, fim=2025, cmpt_fim=202606, saida=saida, fonte="espelho", workers=a.workers)
    tarefas = S.extrair(args)
    cob = S.cobertura(args, tarefas)
    cob.to_csv(os.path.join(saida, "cobertura_arquivos.csv"), index=False)
    print("Verificação cobertura:", cob.status.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
