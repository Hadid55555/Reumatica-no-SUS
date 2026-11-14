#!/usr/bin/env python3
"""Fase 2 — SIH-SP (serviços profissionais) das AIH valvares identificadas no RD, competências jan/2008–jun/2026.

Alvo: AIH do RD (saida_valvar/parciais) com código valvar (aberto, percutâneo, TAVI) ou cirurgia múltipla/sequencial
(0415010012, 0415020034) com CID principal valvar (I05–I09, I33–I39, Q22–Q23). Para cada arquivo RD<UF><AA><MM> com
alvo, lê SP<UF><AA><MM> do espelho healthbr-data (SHA-256 conferido) e guarda as linhas cujo SP_NAIH está no alvo.
Retomável: um CSV por arquivo em saida_valvar/sp_parciais/ + _cobertura.jsonl.

Uso: python valvar/sih_sp_valvar.py [--workers 8]
"""
import argparse
import glob
import json
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(0, AQUI)
import sih_crvm as S  # noqa: E402
from sih_valvar import PROCS_VALVA  # noqa: E402

SAIDA = os.path.join(RAIZ, "saida_valvar")
PASTA = os.path.join(SAIDA, "sp_parciais")
ESPELHO_SP = "https://data.sidneybissoli.com/sih/sp"
MULTIPLAS = {"0415010012", "0415020034"}
CID_RE = r"^(I0[0-9]|I3[3-9]|Q2[23])"
COLS = ["SP_UF", "SP_AA", "SP_MM", "SP_CNES", "SP_NAIH", "SP_PROCREA", "SP_DTINTER", "SP_DTSAIDA", "SP_ATOPROF",
        "SP_TP_ATO", "SP_QTD_ATO", "SP_PTSP", "SP_VALATO", "SP_PF_CBO", "SP_PF_DOC", "SP_PJ_DOC", "SP_CIDPRI",
        "SP_CIDSEC", "SP_QT_PROC", "SP_COMPLEX", "SP_FINANC"]
lock = threading.Lock()


def alvos():
    out = {}
    for f in sorted(glob.glob(os.path.join(SAIDA, "parciais", "RD*.csv"))):
        nome = os.path.basename(f)[:-4]
        if int(nome[4:6]) < 8 or int(nome[4:6]) > 26:
            continue
        d = pd.read_csv(f, dtype=str, keep_default_na=False, usecols=["N_AIH", "PROC_REA", "DIAG_PRINC"])
        proc, cid = d.PROC_REA.str.strip(), d.DIAG_PRINC.str.strip()
        sel = d[(proc.isin(PROCS_VALVA) & ~proc.str[:2].isin(["32", "48"])) | (proc.isin(MULTIPLAS) & cid.str.match(CID_RE))]
        if len(sel):
            out[nome] = set(sel.N_AIH.str.strip())
    return out


def processar(nome, naih, man, tmpdir):
    uf, aa, mm = nome[2:4], int(nome[4:6]), int(nome[6:8])
    chave = f"{2000 + aa}-{mm:02d}-{uf}"
    try:
        info = man.get(chave)
        if info is None:
            raise FileNotFoundError(f"partição SP {chave} ausente no espelho")
        import pyarrow.parquet as pq
        partes, total = [], 0
        for of in info["output_files"]:
            dest = os.path.join(tmpdir, chave + os.path.basename(of["path"]))
            S.baixar(f"{ESPELHO_SP}/{of['path'][len('sih/sp/'):]}", dest)
            try:
                if S.sha256(dest) != of["sha256"]:
                    raise ValueError(f"SHA-256 divergente em {of['path']}")
                f = pq.ParquetFile(dest)
                t = f.read(columns=[c for c in COLS if c in f.schema_arrow.names]).to_pandas(ignore_metadata=True)
                total += len(t)
                partes.append(t[t.SP_NAIH.astype(str).str.strip().isin(naih)])
            finally:
                os.remove(dest)
        df = pd.concat(partes, ignore_index=True)
        arq = os.path.join(PASTA, "SP" + nome[2:] + ".csv")
        df.to_csv(arq + ".tmp", index=False)
        os.replace(arq + ".tmp", arq)
        reg = {"arquivo": nome, "status": "lido", "linhas_sp": total, "linhas_alvo": len(df), "aih_alvo": len(naih),
               "aih_encontradas": int(df.SP_NAIH.astype(str).str.strip().nunique())}
    except Exception as e:
        reg = {"arquivo": nome, "status": "falhou", "erro": f"{type(e).__name__}: {e}", "aih_alvo": len(naih)}
    with lock:
        with open(os.path.join(PASTA, "_cobertura.jsonl"), "a") as fh:
            fh.write(json.dumps(reg, ensure_ascii=False) + "\n")
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(PASTA, exist_ok=True)
    mp = os.path.join(SAIDA, "manifest_sih_sp.json")
    if not os.path.exists(mp):
        S.baixar(f"{ESPELHO_SP}/manifest.json", mp)
    man = json.load(open(mp))["partitions"]
    alvo = alvos()
    feitos = {f[:-4] for f in os.listdir(PASTA) if f.startswith("SP") and f.endswith(".csv")}
    pend = [k for k in alvo if "SP" + k[2:] not in feitos]
    print(f"{len(alvo)} arquivos com alvo ({sum(map(len, alvo.values()))} AIH); {len(pend)} pendentes", flush=True)
    tmp = tempfile.mkdtemp(prefix="sp_dl_", dir=SAIDA)
    ok = fal = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(processar, k, alvo[k], man, tmp) for k in pend]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            if r["status"] == "lido":
                ok += 1
            else:
                fal += 1
                print(f"  FALHOU {r['arquivo']}: {r['erro']}", flush=True)
            if i % 100 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} (ok={ok}, falhas={fal})", flush=True)
    os.rmdir(tmp)
    print("Verificação SP: fim", flush=True)


if __name__ == "__main__":
    main()
