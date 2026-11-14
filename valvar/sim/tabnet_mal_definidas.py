#!/usr/bin/env python3
"""SIM/TabNet: óbitos por residência com causa mal definida (capítulo XVIII, R00–R99) e total, por região e ano
(2003–2005, 2017–2019, 2022–2024). Saída: valvar/sim/tabnet_mal_definidas_regiao.csv."""
import html
import os
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
URL = "http://tabnet.datasus.gov.br/cgi/tabcgi.exe?sim/cnv/obt10uf.def"
TODAS = ["SRegião", "SUnidade_da_Federação", "SCapítulo_CID-10", "SGrupo_CID-10", "SCategoria_CID-10", "SCausa_-_CID-BR-10",
         "SCausa_mal_definidas", "SFaixa_Etária", "SFaixa_Etária_OPS", "SFaixa_Etária_det", "SFx.Etária_Menor_1A", "SSexo",
         "SCor/raça", "SEscolaridade", "SEstado_civil", "SLocal_ocorrência"]
REG = {"1": "Norte", "2": "Nordeste", "3": "Sudeste", "4": "Sul", "5": "Centro-Oeste"}
lin = []
for ano in [2003, 2004, 2005, 2017, 2018, 2019, 2022, 2023, 2024]:
    campos = [("Linha", "Região"), ("Coluna", "Capítulo_CID-10"), ("Incremento", "Óbitos_p/Residênc"),
              ("Arquivos", f"obtuf{ano % 100:02d}.dbf")] + [(s, "TODAS_AS_CATEGORIAS__") for s in TODAS] + [("formato", "prn"), ("mostre", "Mostra")]
    body = urllib.parse.urlencode(campos, encoding="latin-1").encode()
    for t in range(5):
        try:
            r = urllib.request.urlopen(urllib.request.Request(URL, data=body), timeout=180).read().decode("latin-1")
            pre = html.unescape(re.search(r"<PRE>(.*?)</PRE>", r, re.S | re.I).group(1))
            break
        except Exception:
            time.sleep(2 ** (t + 1))
    L = [l.split(";") for l in pre.strip().split("\n") if l.startswith('"')]
    cab = [c.strip('"') for c in L[0]]
    ix = [i for i, c in enumerate(cab) if "XVIII" in c][0]
    for l in L[1:]:
        nome = l[0].strip('"')
        if nome == "Total":
            continue
        md = 0 if l[ix].strip() in ("-", "") else int(l[ix])
        lin.append({"ano": ano, "regiao": REG[nome.split()[0]], "obitos_mal_definidos": md, "obitos_total": int(l[-1]),
                    "pct_mal_definidos": 100 * md / int(l[-1])})
    time.sleep(0.5)
d = pd.DataFrame(lin)
d.to_csv(os.path.join(AQUI, "tabnet_mal_definidas_regiao.csv"), index=False)
print(d.pivot(index="regiao", columns="ano", values="pct_mal_definidos").round(1).to_string())
