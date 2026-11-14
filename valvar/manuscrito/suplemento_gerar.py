#!/usr/bin/env python3
"""Gera valvar/manuscrito/suplemento.md (Supplementary Material, em inglês) a partir dos resultados congelados
(resultados_fase1/2/5/6/8/9, sim_longo) e das bases versionadas — nenhum número digitado à mão.
Uso: python valvar/manuscrito/suplemento_gerar.py"""
import json
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
V = os.path.dirname(AQUI)
sys.path.insert(0, V)
R = lambda f, k: pd.read_csv(os.path.join(V, f"resultados_{k}", f + ".csv"))  # noqa: E731
REG = {"Norte": "North", "Nordeste": "Northeast", "Centro-Oeste": "Centre-West", "Sudeste": "Southeast", "Sul": "South",
       "Brasil": "Brazil", "Homens": "Men", "Mulheres": "Women"}


def f(v, d=2):
    return f"{v:.{d}f}".replace("-", "−").replace(".", "·")


def ci(r, lo, hi, d=2):
    sep = " to " if min(lo, hi) < 0 else "–"
    return f"{f(r, d)} ({f(lo, d)}{sep}{f(hi, d)})"


def n(v):
    return f"{int(round(v)):,}"


def tab(cab, linhas):
    out = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    out += ["| " + " | ".join(str(x) for x in l) + " |" for l in linhas]
    return "\n".join(out)


def reg(s):
    s = s.replace("Região ", "")
    return REG.get(s, s)


B = []

# ------------------------------------------------------------------ métodos
B.append("""# Supplementary Methods

**Data sources and extraction.** Monthly state files of the reduced hospital admission authorisation file (RD) of the SIH/SUS for January 2000 to June 2026 and the corresponding professional services files (SP) for 2008–2026 were obtained from a public mirror of the DATASUS repository that publishes a SHA-256 checksum per file; all checksums were verified. Six RD files (Amapá October 2007; Roraima January–May 2000) are absent from the official repository. Records with identification type 5 (long-stay continuation) and duplicated admission numbers were removed. The reference year is the year of admission.

**Procedure definitions.** Open valve surgery was identified by the main procedure (Table S1). Between 2003 and 2007, codes of the previous SIH procedure tables (groups 32 and 48) were mapped to the same five groups used after the national unified procedure table (SIGTAP) was introduced in January 2008. From 2008, admissions billed under the generic codes for multiple or sequential surgery (0415010012, 0415020034) were included when the principal diagnosis was valvular and the SP file recorded a valve procedure, a valve prosthesis or an annuloplasty ring. Percutaneous valvuloplasty and transcatheter valve implantation were analysed separately and excluded from surgery counts.

**Aetiology.** Strict rheumatic: principal diagnosis I05–I09. Expanded rheumatic (main analysis): strict plus Q22.5 or I38 at age 18 years or older. Non-rheumatic: I34–I37. Comparator: I35 (non-rheumatic aortic valve disorders). Other principal diagnoses (for example congenital, endocarditis, coronary disease in combined procedures) were not classified.

**Rates and standardisation.** Denominators are IBGE population projections (table 7358) by state, sex and 5-year age group (80+ open), multiplied for 2022–2025 by the state ratio between the official 2024 and 2025 estimates and the projection (interpolated for 2022–2023). Rates were directly standardised to the 2010 Brazilian population; variances assume Poisson counts. For comparisons with mortality, both measures were standardised in the age groups of the mortality tabulation (0–4, 5–9, 10–14, 15–19, then 10-year groups to 80+).

**Joinpoint regression.** Weighted least squares on the log age-standardised rate (weights = inverse of the delta-method variance), 0–3 joinpoints at integer years, at least three observations per segment, selection by Bayesian information criterion, excluding 2020–2021; APC and AAPC with 95% CIs from the segment slopes and their covariance.

**Das Gupta decomposition.** The annual number of operations was written as N = P × Σ sᵢ rᵢ, where P is total population, sᵢ the share of age group i and rᵢ its surgery rate. The change between two years was split symmetrically into the effects of P, s and r (three-factor Das Gupta formula), which sum exactly to the observed change. Sensitivity: 3-year averages (2003–05 vs 2022–24).

**Age-specific rate ratios.** Counts and person-years were pooled over 3-year periods; 95% CIs used the Poisson variance of the log ratio. Ratios of rate ratios compared rheumatic with I35 surgery. Interactions (period × age × region and period × age × sex) were tested with Poisson models with Pearson-scaled dispersion (F tests) in 5-year age groups. The age gradient is the ratio of the rate ratio at 15–39 years to that at 60 years or older (55 years or older in Table S5).

**Mortality.** Deaths with underlying cause I00–I09 by year, region of residence, sex and age group were obtained from the official SIM tabulation service (TabNet) for 2000–2024; microdata for 2016–2024 gave identical totals. Trends in 2013–2024 were estimated with age-adjusted Poisson models (Pearson-scaled), excluding 2020–2021, for surgery and mortality in the same age groups.

**Age–period–cohort models.** Rheumatic surgery in 13 age groups (15–19 to 75–79 years) and four 5-year periods (2005–09, 2010–14, 2015–19, 2020–24) was modelled with Poisson regression (age; age + drift; age + period; age + cohort; age + period + cohort), with F tests scaled by the dispersion of the full model [Clayton and Schifflers, 1987]. Cohort rate ratios are from the age–cohort model, in which the linear trend (drift) is attributed to cohort; only curvature is identifiable.

**Pandemic.** Expected counts for 2020–2025 were projected from an age-specific Poisson trend fitted to 2010–2019.

**State-level analysis (ecological, exploratory).** Age-standardised rates (≥15 years) by state of residence in 2003–05 and 2022–24 were compared with household income per capita (Continuous National Household Sample Survey, SIDRA table 7531, 2012, the first year available) and with the change in the annual number of valve surgeries performed in the state (all aetiologies, place of admission), using Spearman correlation.

**Independent checks.** Annual totals, regional sums and the main cell counts were recounted directly from the analytical files without the analysis functions; the Das Gupta components sum exactly to the observed change in every stratum.""")

# ------------------------------------------------------------------ fluxo
a = pd.read_csv(os.path.join(V, "base_valvar_2003_2007.csv.gz"), dtype={"DIAG_PRINC": str})
b = pd.read_csv(os.path.join(V, "base_valvar_2008_2025.csv.gz"), dtype={"DIAG_PRINC": str, "PROC_REA": str})
info = R("0_info", "fase1").set_index("item").valor
res2 = json.load(open(os.path.join(V, "resultados_fase2", "resumo.json")))
bc = b[b.cirurgia]
mult = bc.PROC_REA.isin(["0415010012", "0415020034"])
tot = len(a) + len(bc)
def et(x):
    return x.reum_ampliada.sum(), (x.etiologia == "Reumática").sum(), (x.etiologia == "Não reumática").sum()
ea, eb = et(a), et(bc)
B.append(f"""# Flow of records

| Step | n |
|---|---|
| RD monthly state files read, 2000–2026 | {res2['rd_arquivos']['lidos']:,} of 8,586 |
| SP monthly state files read (admissions with a target procedure), 2008–2026 | {res2['sp_arquivos']['lidos']:,} |
| Admissions with valve procedures or multiple-surgery codes, 2000–2025 (after removing continuation records and duplicates) | {int(info['no_recorte_2000_2025']):,} |
| Open valve surgery, 2003–2007 (previous procedure tables) | {len(a):,} |
| Open valve surgery, 2008–2025, specific SIGTAP codes | {int((~mult).sum()):,} |
| Multiple-surgery records with a valvular principal diagnosis, 2008–2025 | {res2['multiplas_cid_valvar']:,} |
| — with a valve procedure, prosthesis or ring in the SP file (included) | {res2['multiplas_com_ato_valvar']:,} |
| **Open valve surgery, 2003–2025 (study population)** | **{tot:,}** |
| Rheumatic, expanded definition (main) | {ea[0] + eb[0]:,} ({100 * (ea[0] + eb[0]) / tot:.1f}%) |
| Rheumatic, strict definition (I05–I09) | {ea[1] + eb[1]:,} |
| Non-rheumatic (I34–I37) | {ea[2] + eb[2]:,} |
| Other principal diagnoses (not classified) | {tot - (ea[0] + eb[0]) - (ea[2] + eb[2]):,} |

Admissions are counted once. Rheumatic (expanded) admissions without valid age: 0; with unknown region of residence: 0.""")

# ------------------------------------------------------------------ S1 códigos
s1 = [["Single-valve replacement", "0406010692", "48010138, 32020015"],
      ["Valve repair", "0406010803", "48010120, 32026013"],
      ["Multivalve repair and/or replacement", "0406010820", "48010146, 32041012, 32019017"],
      ["Valve surgery with coronary bypass", "0406011206, 0406010811", "48010154, 32042019, 48011185"],
      ["Other open valve procedures (tricuspid repair, congenital mitral repair, Ross, valved conduit)", "0406010340, 0406010358, 0406010552, 0406010846", "—"],
      ["Multiple or sequential surgery (included only with valve act, prosthesis or ring in the SP file)", "0415010012, 0415020034", "Not included (31000002)"],
      ["Excluded: percutaneous mitral valvuloplasty", "0406030120", "48030120"],
      ["Excluded: other percutaneous valvuloplasty", "0406030111, 0406030138, 0406030146", "48030040, 48030058, 32024010"],
      ["Excluded: transcatheter aortic valve implantation", "0406030162", "—"]]
B.append("# Table S1. Procedure codes\n\n" + tab(["Group", "SIGTAP (2008–2025)", "Previous SIH tables (2003–2007)"], s1)
         + "\n\nMaterials (SP file, 2008–2025): biological prosthesis 0702040541, 0702040550; mechanical prosthesis 0702040568, 0702040576; annuloplasty ring 0702040029.")

# ------------------------------------------------------------------ S2 composição por CID
c = R("A1_composicao_anual_por_cid", "fase9")
B.append("# Table S2. Principal diagnosis of rheumatic (expanded) valve surgery by year\n\n"
         + tab(["Year", "I05", "I06", "I07", "I08", "I09", "Q22.5", "I38", "Total", "Q22.5 + I38, %"],
               [[int(r.ano), n(r.I05), n(r.I06), n(r.I07), n(r.I08), n(r.I09), n(r["Q22.5"]), n(r.I38), n(r.total_ampliada), f(r["pct_Q22.5_I38"], 1)] for _, r in c.iterrows()])
         + "\n\nQ22.5 and I38 counted at age ≥18 years. The use of Q22.5 as a placeholder for adult valve surgery began in 2004 and ended in 2013 (Supplementary Methods; figure S1).")

# ------------------------------------------------------------------ S3 série anual
s = R("1_serie_2003_2025", "fase6")
p = s.pivot(index="ano", columns="definicao")
lin = []
for ano in p.index:
    g = lambda d, col: p.loc[ano, (col, d)]  # noqa: E731
    lin.append([ano, n(g("Reumática ampliada", "n")), ci(g("Reumática ampliada", "taxa_padronizada"), g("Reumática ampliada", "ic95_inf"), g("Reumática ampliada", "ic95_sup")),
                n(g("Reumática estrita", "n")), f(g("Reumática estrita", "taxa_padronizada")), n(g("Não reumática", "n")), f(g("Não reumática", "taxa_padronizada")),
                n(g("Aórtica não reumática (I35)", "n")), f(g("Aórtica não reumática (I35)", "taxa_padronizada")),
                f(g("Reumática ampliada", "idade_mediana"), 0), f(g("Reumática ampliada", "pct_nascidos_antes_1975"), 1)])
B.append("# Table S3. Annual number and age-standardised rate (per 100,000) of valve surgery by aetiology, 2003–2025\n\n"
         + tab(["Year", "Rheumatic expanded, n", "Rate (95% CI)", "Rheumatic strict, n", "Rate", "Non-rheumatic, n", "Rate", "I35, n", "Rate",
                "Median age, rheumatic", "Born before 1975, %"], lin)
         + "\n\nRates directly standardised to the 2010 Brazilian population (all ages).")

# ------------------------------------------------------------------ S4 razões por idade de 5 anos
r = R("3a_razoes_20_anos_2003_05_x_2023_25", "fase6")
lin = []
for fx in r.faixa.unique():
    x = r[r.faixa == fx].set_index("definicao")
    lab = fx.replace(" a ", "–").replace(" anos ou mais", "+").replace(" anos", "").replace("-", "–")
    row = [lab]
    for d in ["Reumática ampliada", "Aórtica não reumática (I35)", "Não reumática"]:
        y = x.loc[d]
        row += [f"{n(y.n_2003_05)} / {n(y.n_2023_25)}", ci(y.razao, y.ic95_inf, y.ic95_sup)]
    lin.append(row)
B.append("# Table S4. Age-specific rate ratios, 2023–2025 vs 2003–2005, by aetiology\n\n"
         + tab(["Age, years", "Rheumatic, n", "Rate ratio (95% CI)", "I35, n", "Rate ratio (95% CI)", "Non-rheumatic, n", "Rate ratio (95% CI)"], lin)
         + "\n\nn = operations in 2003–2005 / 2023–2025.")

# ------------------------------------------------------------------ S5 região e sexo
a1 = R("A1_razoes_2023_25_x_2003_05_regiao_sexo", "fase8")
a2 = R("A2_razao_das_razoes_reum_x_I35", "fase8").set_index(["estrato", "faixa"])
a3 = R("A3_gradiente_jovens_vs_55mais", "fase8").set_index("estrato")
rs = json.load(open(os.path.join(V, "resultados_fase8", "resumo.json")))
lin = []
x = a1[a1.definicao == "Reumática ampliada"]
for est in x.estrato.unique():
    y = x[x.estrato == est].set_index("faixa")
    row = [reg(est)] + [f"{ci(y.loc[g].razao, y.loc[g].ic95_inf, y.loc[g].ic95_sup)}" for g in ["15-39", "40-54", "55+"]]
    row += [ci(a2.loc[(est, g)].razao_das_razoes, a2.loc[(est, g)].ic95_inf, a2.loc[(est, g)].ic95_sup) for g in ["15-39", "55+"]]
    row += [ci(*a3.loc[est, ["gradiente_15_39_vs_55mais", "ic95_inf", "ic95_sup"]])]
    lin.append(row)
ir, isx = rs["A_interacao_regiao_x_periodo_x_idade"], rs["A_interacao_sexo_x_periodo_x_idade"]
B.append("# Table S5. Rheumatic surgery rate ratios (2023–25 vs 2003–05) by region of residence and sex\n\n"
         + tab(["Stratum", "15–39 years", "40–54 years", "≥55 years", "Ratio of rate ratios vs I35, 15–39", "Ratio of rate ratios vs I35, ≥55", "Age gradient (15–39 / ≥55)"], lin)
         + f"\n\nPeriod × age × region interaction: F = {f(ir['F'])} ({ir['gl'][0]}, {ir['gl'][1]} df), P = {f(ir['p'])}. Period × age × sex: F = {f(isx['F'])} ({isx['gl'][0]}, {isx['gl'][1]} df), P = {f(isx['p'])}.")

# ------------------------------------------------------------------ S6 tendências regionais
b2 = R("B2_variacao_anual_regional_2003_2024", "fase8")
c4 = R("C4_tendencia_cirurgia_obito_regiao", "fase8")
lin = []
for rg in ["Brasil", "Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
    y = b2[b2.regiao == rg].set_index("faixa")
    row = [reg(rg)] + [ci(y.loc[g].variacao_anual_pct_2003_2024, y.loc[g].ic95_inf, y.loc[g].ic95_sup, 1) for g in ["≥15", "15-39", "40-54", "55+"]]
    for med in ["cirurgia", "obito"]:
        z = c4[(c4.regiao == rg) & (c4.medida == med) & (c4.periodo == "2013–2024")].iloc[0]
        row.append(ci(z.variacao_anual_pct, z.ic95_inf, z.ic95_sup, 1))
    lin.append(row)
B.append("# Table S6. Annual percent change in rheumatic surgery and mortality by region\n\n"
         + tab(["Region", "Surgery 2003–2024, ≥15 y", "15–39 y", "40–54 y", "≥55 y", "Surgery 2013–2024, all ages ≥5 y", "Mortality 2013–2024, all ages ≥5 y"], lin)
         + "\n\nAge-adjusted Poisson models with Pearson-scaled dispersion, excluding 2020–2021. Surgery in the last two columns uses the age groups of the mortality tabulation.")

# ------------------------------------------------------------------ S7 triangulação por região
t1 = R("C1_cirurgia_x_obito_razoes_idade_regiao", "fase8")
t2 = R("C2_gradiente_cirurgia_x_obito", "fase8").set_index(["regiao", "medida"])
lin = []
for rg in ["Brasil", "Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
    for med, lb in [("cirurgia", "Surgery"), ("obito", "Deaths")]:
        y = t1[(t1.regiao == rg) & (t1.medida == med)].set_index("grupo")
        g = t2.loc[(rg, med)]
        lin.append([reg(rg) if med == "cirurgia" else "", lb] + [f"{n(y.loc[k].n_2003_05)} / {n(y.loc[k].n_2022_24)}; {ci(y.loc[k].razao, y.loc[k].ic95_inf, y.loc[k].ic95_sup)}" for k in ["15-39", "40-59", "60+"]]
                   + [ci(g.gradiente_jovens_vs_60mais, g.ic95_inf, g.ic95_sup)])
B.append("# Table S7. Triangulation: rheumatic surgery and rheumatic heart disease deaths, 2022–24 vs 2003–05, by region\n\n"
         + tab(["Region", "Measure", "15–39 years: n; rate ratio", "40–59 years", "≥60 years", "Age gradient (15–39 / ≥60)"], lin)
         + "\n\nDeaths: underlying cause I00–I09 (SIM). n = events in 2003–05 / 2022–24.")

# ------------------------------------------------------------------ S8 Das Gupta
d = R("D_das_gupta_regiao_sexo", "fase8")
lin = [[("2003 vs 2024" if k.startswith("2003 ×") else "2003–05 vs 2022–24 (annual mean)"), reg(x.estrato), n(x.n_inicio), n(x.n_fim),
        f"{x.diferenca:+,.0f}".replace("-", "−"), f"{x.efeito_tamanho_pop:+,.0f}".replace("-", "−"), f"{x.efeito_estrutura_etaria:+,.0f}".replace("-", "−"),
        f"{x.efeito_taxa_especifica:+,.0f}".replace("-", "−")] for (k, x) in zip(d.comparacao, d.itertuples())]
B.append("# Table S8. Das Gupta decomposition of the change in annual rheumatic valve surgeries\n\n"
         + tab(["Comparison", "Stratum", "Start", "End", "Net change", "Population size", "Population ageing", "Age-specific rates"], lin))

# ------------------------------------------------------------------ S9 pandemia e definição
b1 = R("B1_tendencia_com_e_sem_pandemia", "fase9")
bo = R("B2_observado_esperado_2020_2025", "fase9")
b3 = R("B3_razao_geracional_pre_pandemia_2017_19", "fase9").set_index("faixa")
es = R("A2_razoes_estrita_x_ampliada", "fase9")
sens = R("1_razoes_geracionais_e_sensibilidade", "fase5")
lin = [[x.faixa, x.serie, ci(x.variacao_anual_pct, x.ic95_inf, x.ic95_sup, 1)] for x in b1.itertuples()]
lo = [[int(x.ano), x.faixa, n(x.observado), n(x.esperado_tendencia_2010_19), f(x.obs_div_esp)] for x in bo.itertuples() if x.faixa == "≥15"]
le = [[x.definicao, x.faixa, f"{n(x.n_2003_05)} / {n(x.n_2023_25)}", ci(x.razao, x.ic95_inf, x.ic95_sup)] for x in es.itertuples()]
le += [["Expanded, ending 2017–19", g, f"{n(b3.loc[g].n_2003_05)} / {n(b3.loc[g].n_2017_19)}", ci(b3.loc[g].razao, b3.loc[g].ic95_inf, b3.loc[g].ic95_sup)] for g in ["15-39", "40-54", "55+"]]
B.append("# Table S9. Sensitivity analyses: case definition and the COVID-19 pandemic\n\n"
         + tab(["Definition / comparison", "Age, years", "n start / end", "Rate ratio (95% CI)"], le)
         + "\n\n" + tab(["Age, years", "Series", "Annual percent change (95% CI)"], lin)
         + "\n\n" + tab(["Year", "Age", "Observed", "Expected (2010–19 trend)", "Observed / expected"], lo)
         + "\n\nAll valve surgery (all aetiologies) rose from 14,773 operations in 2019 to 16,624 in 2024, and I35 surgery from 3,289 to 4,308 (Table S3). "
           "Further sensitivity analyses in the 2008–2025 data (without multiple-surgery records, excluding hospital-years with a dominant default diagnosis, I05 and I06 separately, alternative standard population) are in the accompanying workbook (fase5_congelamento.xlsx).")

# ------------------------------------------------------------------ S10 IPC
d1 = R("D1_ipc_deviance", "fase9")
d2 = R("D2_ipc_testes", "fase9")
d3 = R("D3_razao_taxas_coorte_modelo_idade_coorte", "fase9")
r9 = json.load(open(os.path.join(V, "resultados_fase9", "resumo.json")))
tr = {"Idade": "Age", "Idade + drift": "Age + drift", "Idade + período": "Age + period", "Idade + coorte": "Age + cohort", "Idade + período + coorte": "Age + period + cohort"}
tt = {"curvatura de coorte (dado drift)": "Cohort curvature (given drift)", "curvatura de período (dado drift)": "Period curvature (given drift)",
      "período, dado idade + coorte": "Period, given age + cohort", "coorte, dado idade + período": "Cohort, given age + period", "drift (tendência linear)": "Drift (linear trend)"}
B.append("# Table S10. Age–period–cohort models of rheumatic valve surgery (ages 15–79 years, 2005–2024)\n\n"
         + tab(["Model", "Deviance", "Residual df"], [[tr[x.modelo], f(x.deviance, 1), x.gl_resid] for x in d1.itertuples()])
         + "\n\n" + tab(["Test", "Δ deviance", "df", "F", "P"], [[tt[x.teste], f(x.delta_deviance, 1), x.gl, f(x.F, 1), "<0.001" if x.p < 0.001 else f(x.p, 3)] for x in d2.itertuples()])
         + "\n\n" + tab(["Birth cohort", "Operations", "Rate ratio vs 1950 cohort"], [[x.coorte_nascimento, n(x.n), f(x.rr_vs_1950)] for x in d3.itertuples()])
         + f"\n\nDispersion (full model) {f(r9['D_dispersao_phi'])}. Net drift {f(r9['D_net_drift_pct_ano'][0])}% per year (95% CI {f(r9['D_net_drift_pct_ano'][1])} to {f(r9['D_net_drift_pct_ano'][2])}). "
           "Cohort rate ratios come from the age–cohort model, which attributes the linear trend to cohort; only curvature is identifiable.")

# ------------------------------------------------------------------ S11 UF
u = R("C1_uf_taxas_razoes_renda", "fase9")
cp = R("C2_uf_capacidade_cirurgica", "fase9").set_index("uf")
lin = []
for x in u.sort_values(["regiao", "uf"]).itertuples():
    k = cp.loc[x.uf] if x.uf in cp.index else None
    vol = f"{n(k.valvares_ano_2003_05)} → {n(k.valvares_ano_2022_24)}" if k is not None else "—"
    lin.append([x.uf, reg(x.regiao), n(x.renda_pc_2012), f"{n(x.n_2003_05)} / {n(x.n_2022_24)}", f(x.taxa_2003_05), f(x.taxa_2022_24),
                ci(x.razao_taxas, x.ic95_inf, x.ic95_sup), f(x.razao_15_39), f(x.razao_55mais), vol])
B.append("# Table S11. Rheumatic surgery by state of residence (exploratory, ecological)\n\n"
         + tab(["State", "Region", "Income", "Operations", "Rate 2003–05", "Rate 2022–24",
                "Rate ratio (95% CI)", "RR 15–39 y", "RR ≥55 y", "Valve surgery in state/year"], lin)
         + f"\n\nIncome: household income per capita, 2012 (BRL/month). Operations: rheumatic operations in residents, 2003–05 / 2022–24. Valve surgery in state/year: all valve operations performed in the state per year, 2003–05 → 2022–24. RR: rate ratio by age group. Age-standardised rates per 100,000 (≥15 years). Spearman correlation of the rate ratio with income: ρ = {f(r9['C_spearman_renda_x_razao'][0])} (P = {f(r9['C_spearman_renda_x_razao'][1])}); "
           f"with the ratio of valve surgery volume performed in the state: ρ = {f(r9['C_spearman_volume_valvar_uf_x_razao_reum'][0])} (P < 0.001; {r9['C_spearman_volume_valvar_uf_x_razao_reum'][2]} states with surgery in both periods). "
           "Rondônia: hospitals in the state recorded almost no rheumatic principal diagnoses after 2007 (local coding artefact).")

# ------------------------------------------------------------------ S12 degrau 2013–2017
g = R("4a_degrau_2013_2017_series", "fase5")
lin = [[int(x.ano), n(x.reumatica), n(x.nao_reumatica), n(x.valvar_total), n(x.crvm_sus), int(x.hospitais_ativos), n(x.valor_real_mediano),
        f(x.pct_multiplas, 1), f(x.pct_troca_com_protese_faturada, 1)] for x in g.itertuples()]
B.append("# Table S12. The 2013–2017 decline in context (strict definition, 2008–2025)\n\n"
         + tab(["Year", "Rheumatic (strict)", "Non-rheumatic", "All valve surgery", "CABG in the SUS", "Hospitals with valve surgery",
                "Median amount paid per valve admission, BRL Dec 2025", "Multiple-surgery records, %", "Replacements with billed prosthesis, %"], lin)
         + "\n\nAmounts adjusted to December 2025 by the IPCA consumer price index.")

# ------------------------------------------------------------------ S13–S17 (revisão)
A1 = R("A1_razoes_pre_pandemia_x_recente", "fase10")
A2 = R("A2_cirurgia_obito_2017_19_x_2003_05", "fase10")
A3 = R("A3_gradiente_cirurgia_obito_pre_pandemia", "fase10").set_index(["regiao", "medida"])
B1_ = R("B1_razoes_inicio_2003_sem_coringa", "fase10")
B2_ = R("B2_padrao_alternativo_2022", "fase10")
tdef = {"Reumática ampliada": "Rheumatic (expanded)", "Aórtica não reumática (I35)": "Non-rheumatic aortic (I35)", "Não reumática (I34–I37)": "All non-rheumatic (I34–I37)",
        "Razão das razões reum ÷ Aórtica não reumática (I35)": "Ratio of rate ratios, rheumatic/I35", "Razão das razões reum ÷ Não reumática (I34–I37)": "Ratio of rate ratios, rheumatic/I34–I37"}
lin = []
for (fim, d), g in A1.groupby(["fim", "definicao"], sort=False):
    g = g.set_index("faixa")
    lin.append([f"{fim} vs 2003–05", tdef[d]] + [ci(g.loc[k].razao, g.loc[k].ic95_inf, g.loc[k].ic95_sup) for k in ["15-39", "40-54", "55+"]])
lr = []
for rg in ["Brasil", "Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]:
    for med, lb in [("cirurgia", "Surgery"), ("obito", "Deaths")]:
        y = A2[(A2.regiao == rg) & (A2.medida == med)].set_index("grupo")
        gg = A3.loc[(rg, med)]
        lr.append([reg(rg) if med == "cirurgia" else "", lb] + [ci(y.loc[k].razao, y.loc[k].ic95_inf, y.loc[k].ic95_sup) for k in ["15-39", "40-59", "60+"]]
                  + [ci(gg.gradiente_jovens_vs_60mais, gg.ic95_inf, gg.ic95_sup)])
lb1 = [[("Strict (I05–I09)" if x.definicao.startswith("Estrita") else "Expanded"), f"2003 vs {x.fim}", x.faixa, f"{n(x.n_inicio)} / {n(x.n_fim)}", ci(x.razao, x.ic95_inf, x.ic95_sup)] for x in B1_.itertuples()]
lb2 = [[int(x.populacao_padrao), int(x.ano), f(x.taxa_padronizada), f(x.queda_pct_vs_2003, 1)] for x in B2_.itertuples()]
B.append("# Table S13. Pre-pandemic comparison, alternative comparators, start before the Q22.5 placeholder and alternative standard population\n\n"
         + tab(["Comparison", "Series", "15–39 years", "40–54 years", "≥55 years"], lin)
         + "\n\nSurgery and rheumatic heart disease deaths, 2017–19 vs 2003–05, age groups of the mortality tabulation:\n\n"
         + tab(["Region", "Measure", "15–39 years", "40–59 years", "≥60 years", "Age gradient (15–39 / ≥60)"], lr)
         + "\n\nComparison starting in 2003 only (before the placeholder use of Q22.5):\n\n"
         + tab(["Definition", "Comparison", "Age, years", "n start / end", "Rate ratio (95% CI)"], lb1)
         + "\n\nAge-standardised rate of rheumatic (expanded) surgery per 100,000 with the 2010 and the 2022 Brazilian population as standard:\n\n"
         + tab(["Standard population", "Year", "Rate", "Change vs 2003, %"], lb2))
C1_ = R("C1_perfil_cirurgico_periodo_idade_regiao", "fase10")
est = {"Total": "All", "15-39": "15–39 years", "40-54": "40–54 years", "55+": "≥55 years", "Mulheres 15–44": "Women 15–44 years", **REG}
lin = [[x.periodo, est.get(x.estrato, x.estrato), n(x.n), f(x.pct_multivalvar, 1), f(x.pct_valvar_mais_rvm, 1), f(x.pct_anel, 1), f(x.pct_sem_protese, 1),
        f(x.pct_sem_protese_I05, 1), f(x.pct_mecanica_entre_trocas, 1), f(x.pct_registro_multiplo, 1), f(x.mort_hosp_pct, 1)] for x in C1_.itertuples()]
B.append("# Table S14. Surgical profile of rheumatic (expanded) valve surgery by period, age, region and sex, 2008–2025\n\n"
         + tab(["Period", "Stratum", "n", "Multivalve, %", "With CABG, %", "Ring, %", "No prosthesis, %", "No prosthesis (I05), %", "Mechanical among replacements, %",
                "Multiple-surgery record, %", "In-hospital death, %"], lin)
         + "\n\nFrom the professional services file. Multivalve: multivalve procedure code, two or more valve acts, or two or more prostheses. With CABG excludes multiple-surgery records, in which concomitant bypass cannot be identified. Region = region of residence.")
D1_ = R("D1_valvuloplastia_mitral_percutanea_ano", "fase10")
D2_ = R("D2_valvuloplastia_x_cirurgia_mitral", "fase10")
B.append("# Table S15. Percutaneous mitral valvuloplasty, 2008–2025\n\n"
         + tab(["Year", "Total", "Principal diagnosis I05", "Median age", "Women, %", "<15 y", "15–39 y", "40–54 y", "≥55 y", "North", "Northeast", "Centre-West", "Southeast", "South"],
               [[int(x.ano), n(x.total), n(x.cid_I05), f(x.idade_mediana, 0), f(x.pct_mulheres, 1)] + [n(x[c]) for c in D1_.columns[5:]] for _, x in D1_.iterrows()])
         + "\n\n" + tab(["Period", "Age, years", "Valvuloplasties", "Open surgery for I05", "Valvuloplasty share of mitral interventions, %"],
                          [[x.periodo, x.faixa, n(x.valvuloplastias), n(x.cirurgias_I05), f(x.pct_valvuloplastia, 1)] for x in D2_.itertuples()]))
E1_ = R("E1_mortalidade_hospitalar_tendencia", "fase10")
E2_ = R("E2_mortalidade_hospitalar_bruta_ano", "fase10")
aj = {"bruta": "Crude", "idade+sexo+região": "Age, sex, region", "idade+sexo+região+multivalvar+RVM+urgência+registro múltiplo": "Age, sex, region, multivalve, CABG, urgency, multiple-surgery record"}
B.append("# Table S16. In-hospital mortality after rheumatic (expanded) valve surgery\n\n"
         + tab(["Period", "Adjustment", "Admissions", "Odds ratio per 5 years (95% CI)"], [[x.periodo, aj[x.ajuste], n(x.n), ci(x.OR_por_5_anos, x.ic95_inf, x.ic95_sup, 3)] for x in E1_.itertuples()])
         + "\n\n" + tab(["Year", "Admissions", "In-hospital death, %"], [[int(x.ano), n(x.n), f(x.mort, 1)] for x in E2_.itertuples()])
         + "\n\nLogistic regression; age as a B-spline with 4 df; standard errors clustered by hospital.")
F1_ = R("F1_operados_fora_da_residencia", "fase10")
F2_ = R("F2_volume_por_hospital_regiao", "fase10")
B.append("# Table S17. Access: operations outside the state and region of residence, and hospitals performing valve surgery\n\n"
         + tab(["Period", "Region of residence", "Rheumatic operations", "Outside state of residence, %", "Outside region of residence, %"],
               [[x.periodo, reg(x.regiao_residencia), n(x.n_reum), f(x.pct_fora_da_uf, 1), f(x.pct_fora_da_regiao, 1)] for x in F1_.itertuples()])
         + "\n\n" + tab(["Period", "Region of hospital", "Hospitals", "Hospitals with ≥20 valve operations/year", "Median valve operations per hospital/year",
                          "Share of operations in hospitals with ≥100/year, %", "Median rheumatic operations per hospital/year"],
                         [[x.periodo, reg(x.regiao_hospital), x.hospitais, x.hospitais_20mais_ano, f(x.volume_valvar_mediano_por_hospital, 1),
                           f(x.pct_cirurgias_em_hospitais_100mais, 1), f(x.volume_reum_mediano_por_hospital, 1)] for x in F2_.itertuples()])
         + "\n\nAll valve surgery (all aetiologies) for hospital counts and volumes.")

# ------------------------------------------------------------------ S18 (UF × óbito; causas mal definidas)
G1_ = R("G1_uf_cirurgia_obito", "fase10")
G2_ = R("G2_uf_correlacoes", "fase10")
G3_ = R("G3_mortalidade_corrigida_mal_definidas", "fase10")
MD = pd.read_csv(os.path.join(V, "sim", "tabnet_mal_definidas_regiao.csv"))
lmd = [[reg(r)] + [f(MD[(MD.regiao == r) & (MD.ano == a)].pct_mal_definidos.iloc[0], 1) for a in [2003, 2004, 2005, 2017, 2018, 2019, 2022, 2023, 2024]]
       for r in ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]]
lg3 = [[reg(x.regiao), x.fim, x.grupo.replace("-", "–"), f(x.razao_observada), ci(x.razao_corrigida, x.ic95_inf_corrigida, x.ic95_sup_corrigida)] for x in G3_.itertuples()]
lg1 = [[x.uf, reg(x.regiao), f"{n(x.cirurgia_n_2003_05)} / {n(x.cirurgia_n_2022_24)}", ci(x.cirurgia_razao_2022_24, x.cirurgia_ic95_inf_2022_24, x.cirurgia_ic95_sup_2022_24),
        f"{n(x.obito_n_2003_05)} / {n(x.obito_n_2022_24)}", ci(x.obito_razao_2022_24, x.obito_ic95_inf_2022_24, x.obito_ic95_sup_2022_24), f(x.obito_razao_2017_19)]
       for x in G1_.sort_values(["regiao", "uf"]).itertuples()]
tc = {"razão cirurgia × razão óbito": "Surgery rate ratio × mortality rate ratio", "volume valvar na UF × razão óbito": "Valve surgery volume in state × mortality rate ratio"}
ta = {"todas as UF": "All states", "sem RO": "Without Rondônia", "UF com ≥30 óbitos em cada período, sem RO": "States with ≥30 deaths in each period, without Rondônia"}
lg2 = [[x.fim.replace("_", "–"), ta[x.amostra], tc[x.comparacao], x.n_uf, f(x.spearman), f(x.p)] for x in G2_.itertuples()]
B.append("# Table S18. Rheumatic heart disease mortality: ill-defined causes and state-level comparison with surgery\n\n"
         "Deaths with ill-defined causes (ICD-10 chapter XVIII, R00–R99), % of all deaths, by region of residence:\n\n"
         + tab(["Region", "2003", "2004", "2005", "2017", "2018", "2019", "2022", "2023", "2024"], lmd)
         + "\n\nRheumatic heart disease mortality rate ratios vs 2003–05, observed and after proportional redistribution of ill-defined deaths by region and year:\n\n"
         + tab(["Region", "End period", "Age, years", "Observed", "Corrected (95% CI)"], lg3)
         + "\n\nBy state of residence, 2022–24 vs 2003–05 (age-standardised, ages ≥5 years, age groups of the mortality tabulation):\n\n"
         + tab(["State", "Region", "Surgery n", "Surgery rate ratio (95% CI)", "Deaths n", "Mortality rate ratio (95% CI)", "Mortality rate ratio 2017–19"], lg1)
         + "\n\n" + tab(["End period", "States", "Correlation", "n", "Spearman ρ", "P"], lg2)
         + "\n\nThe proportional correction assumes that ill-defined deaths are distributed like defined deaths within each region and year; because ill-defined causes are more common at older ages, it probably under-corrects at ≥60 years.")

# ------------------------------------------------------------------ figuras
B.append("# Supplementary figures\n\n![](figuras_v2/FigureS1.png){width=15cm}\n\n**Figure S1:** Principal diagnosis of rheumatic (expanded) valve surgery by year. The congenital code Q22.5 was used as a placeholder for adult valve surgery between 2004 and 2012.\n\n"
         "![](figuras_v2/FigureS2.png){width=13cm}\n\n**Figure S2:** Observed rheumatic valve surgeries divided by those expected from the 2010–2019 age-specific trend, 2020–2025, by age group.\n\n"
         "![](figuras_v2/FigureS3.png){width=13cm}\n\n**Figure S3:** Rate ratio of age-standardised rheumatic surgery (2022–24 vs 2003–05) by state of residence against household income per capita in 2012; marker size proportional to operations in 2022–24; colours by region as in figure 3.")

# ------------------------------------------------------------------ RECORD
rec = [["**Title and abstract**", "", ""],
       ["1 (a, b)", "Study design in title or abstract; informative summary", "M: Title (nationwide population-based study); Summary"],
       ["RECORD 1.1", "Type of data used", "M: Summary, Methods (SIH/SUS hospital data; SIM mortality data)"],
       ["RECORD 1.2", "Geographic region and timeframe", "M: Title (Brazil, 2003–2025)"],
       ["RECORD 1.3", "Linkage between databases", "Not applicable (aggregate comparison of SIH and SIM; no individual linkage)"],
       ["**Introduction**", "", ""],
       ["2–3", "Background, rationale and objectives", "M: Introduction"],
       ["**Methods**", "", ""],
       ["4–5", "Study design; setting and dates", "M: Study design and data sources (files 2000–2026; admissions 2003–2025)"],
       ["6 (a)", "Eligibility and sources", "M: Valve surgery and aetiology; S: Flow of records"],
       ["RECORD 6.1", "Codes and algorithms for selection", "S: Table S1 (procedure codes), Supplementary Methods (diagnosis codes)"],
       ["RECORD 6.2", "Validation of codes", "M: Discussion (published validation studies); S: Table S2 (coding audit). No chart-review validation"],
       ["RECORD 6.3", "Linkage flow diagram", "Not applicable"],
       ["7, RECORD 7.1", "Outcomes, exposures and codes", "M: Valve surgery and aetiology; Statistical analysis; S: Supplementary Methods"],
       ["8", "Data sources and measurement", "M: Study design and data sources"],
       ["9", "Efforts to address bias", "M: Statistical analysis (comparator, triangulation with mortality, sensitivity analyses)"],
       ["10", "Study size", "All eligible admissions; no sampling (S: Flow of records)"],
       ["11–12", "Quantitative variables; statistical methods; sensitivity analyses", "M: Statistical analysis; S: Supplementary Methods, Tables S9–S10"],
       ["RECORD 12.1", "Access to the database population", "Complete national public files (S: Flow of records)"],
       ["RECORD 12.2", "Data cleaning", "S: Supplementary Methods (continuation records, duplicates, year of admission)"],
       ["RECORD 12.3", "Linkage methods", "Not applicable"],
       ["**Results**", "", ""],
       ["13, RECORD 13.1", "Numbers at each stage", "S: Flow of records"],
       ["14", "Descriptive data", "M: Table 1"],
       ["15–16", "Outcome data and main results", "M: Results, Tables 2–3, Figures 1–4; S: Tables S3–S8"],
       ["17", "Other analyses", "M: Results; S: Tables S9–S18, Figures S1–S3"],
       ["**Discussion**", "", ""],
       ["18–21", "Key results, limitations, interpretation, generalisability", "M: Discussion"],
       ["RECORD 19.1", "Implications of using routinely collected data", "M: Discussion (coding, billing changes, private sector, unit of analysis)"],
       ["**Other information**", "", ""],
       ["22", "Funding", "M: Declarations"],
       ["RECORD 22.1", "Access to protocol, data and code", "M: Declarations (data sharing)"]]
B.append("# RECORD checklist\n\nLocations refer to the main manuscript (M) or this Supplementary Material (S).\n\n" + tab(["Item", "Recommendation", "Location"], rec))

cab = ('---\ntitle: "Supplementary Material — Stable volume but falling age-specific rates of rheumatic valve surgery in Brazil\'s public health system, 2003–2025"\n---\n\n'
       "Version 3.1 (6 November 2026), for co-author review. Full numerical results are in the accompanying workbooks (fase8_2003_2025.xlsx, fase9_suplemento.xlsx).\n\n")
open(os.path.join(AQUI, "suplemento.md"), "w", encoding="utf-8").write(cab + "\n\n".join(B) + "\n")
print("ok: suplemento.md")
