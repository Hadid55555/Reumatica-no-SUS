#!/usr/bin/env python3
"""SIM oficial via TabNet/DATASUS: óbitos por residência com causa básica I00–I09 (e I05–I09), por ano × região ×
faixa etária, 2000–2024. Fonte: http://tabnet.datasus.gov.br/cgi/tabcgi.exe?sim/cnv/obt10uf.def (arquivos obtuf{AA}.dbf).
Saída: valvar/sim/tabnet_obitos_reumaticos_2000_2024.csv. Uso: python valvar/sim/tabnet_sim.py"""
import os
import html
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
URL = "http://tabnet.datasus.gov.br/cgi/tabcgi.exe?sim/cnv/obt10uf.def"
CATS = {"I00–I09": ["635", "636", "637", "638", "639", "640", "641", "642"], "I05–I09": ["638", "639", "640", "641", "642"]}
TODAS = ["SRegião", "SUnidade_da_Federação", "SCapítulo_CID-10", "SGrupo_CID-10", "SCausa_-_CID-BR-10",
         "SCausa_mal_definidas", "SFaixa_Etária", "SFaixa_Etária_OPS", "SFaixa_Etária_det", "SFx.Etária_Menor_1A", "SSexo",
         "SCor/raça", "SEscolaridade", "SEstado_civil", "SLocal_ocorrência"]
REG = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}


def consulta(ano, ids):
    campos = [("Linha", "Região"), ("Coluna", "Faixa_Etária"), ("Incremento", "Óbitos_p/Residênc"),
              ("Arquivos", f"obtuf{ano % 100:02d}.dbf")] + [(s, "TODAS_AS_CATEGORIAS__") for s in TODAS]
    campos += [("SCategoria_CID-10", i) for i in ids] + [("formato", "prn"), ("mostre", "Mostra")]
    body = urllib.parse.urlencode(campos, encoding="latin-1").encode()
    for t in range(4):
        try:
            r = urllib.request.urlopen(urllib.request.Request(URL, data=body), timeout=180).read().decode("latin-1")
            pre = html.unescape(re.search(r"<PRE>(.*?)</PRE>", r, re.S | re.I).group(1))
            break
        except Exception:
            time.sleep(2 ** (t + 1))
    linhas = [l.split(";") for l in pre.strip().split("\n") if l.startswith('"')]
    cab = [c.strip('"') for c in linhas[0]]
    out = []
    for l in linhas[1:]:
        nome = l[0].strip('"')
        if nome == "Total":
            continue
        reg = REG[nome.split()[0]]
        for c, v in zip(cab[1:-1], l[1:-1]):
            out.append({"ano": ano, "regiao": reg, "faixa_tabnet": c, "obitos": 0 if v.strip() in ("-", "") else int(v)})
    return out


lin = []
for cat, ids in CATS.items():
    for ano in range(2000, 2025):
        for r in consulta(ano, ids):
            lin.append({"causa": cat, **r})
        time.sleep(0.5)
d = pd.DataFrame(lin)
d.to_csv(os.path.join(AQUI, "tabnet_obitos_reumaticos_2000_2024.csv"), index=False)
print(d.groupby(["causa", "ano"]).obitos.sum().unstack(0).to_string())
