#!/usr/bin/env python3
"""SIM oficial via TabNet/DATASUS: óbitos por residência com causa básica I00–I09 por ano × UF × faixa etária, nos anos
das comparações do artigo (2003–2005, 2017–2019, 2022–2024). Mesma consulta de tabnet_sim.py, com linha = UF.
Conferência: soma das UF = arquivo por região (tabnet_obitos_reumaticos_2000_2024.csv).
Saída: valvar/sim/tabnet_obitos_reumaticos_uf.csv. Uso: python valvar/sim/tabnet_sim_uf.py"""
import html
import os
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
URL = "http://tabnet.datasus.gov.br/cgi/tabcgi.exe?sim/cnv/obt10uf.def"
IDS = ["635", "636", "637", "638", "639", "640", "641", "642"]  # I00–I09
TODAS = ["SRegião", "SUnidade_da_Federação", "SCapítulo_CID-10", "SGrupo_CID-10", "SCausa_-_CID-BR-10",
         "SCausa_mal_definidas", "SFaixa_Etária", "SFaixa_Etária_OPS", "SFaixa_Etária_det", "SFx.Etária_Menor_1A", "SSexo",
         "SCor/raça", "SEscolaridade", "SEstado_civil", "SLocal_ocorrência"]
UF = {"11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
      "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
      "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF"}
ANOS = [2003, 2004, 2005, 2017, 2018, 2019, 2022, 2023, 2024]


def consulta(ano):
    campos = [("Linha", "Unidade_da_Federação"), ("Coluna", "Faixa_Etária"), ("Incremento", "Óbitos_p/Residênc"),
              ("Arquivos", f"obtuf{ano % 100:02d}.dbf")] + [(s, "TODAS_AS_CATEGORIAS__") for s in TODAS]
    campos += [("SCategoria_CID-10", i) for i in IDS] + [("formato", "prn"), ("mostre", "Mostra")]
    body = urllib.parse.urlencode(campos, encoding="latin-1").encode()
    for t in range(5):
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
        for c, v in zip(cab[1:-1], l[1:-1]):
            out.append({"ano": ano, "uf": UF[nome.split()[0]], "faixa_tabnet": c, "obitos": 0 if v.strip() in ("-", "") else int(v)})
    return out


lin = []
for ano in ANOS:
    lin += consulta(ano)
    time.sleep(0.5)
d = pd.DataFrame(lin)
d.to_csv(os.path.join(AQUI, "tabnet_obitos_reumaticos_uf.csv"), index=False)
r = pd.read_csv(os.path.join(AQUI, "tabnet_obitos_reumaticos_2000_2024.csv"))
r = r[(r.causa == "I00–I09") & r.ano.isin(ANOS)].groupby("ano").obitos.sum()
u = d.groupby("ano").obitos.sum()
print(pd.DataFrame({"uf_soma": u, "regiao_soma": r, "igual": u == r}).to_string())
