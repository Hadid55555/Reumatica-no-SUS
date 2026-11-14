# Rheumatic valve surgery in Brazil's public health system, 2003–2025

Code and data for the study *"Stable volume but falling age-specific rates of rheumatic valve surgery in Brazil's public
health system, 2003–2025: a nationwide study of cohort and regional patterns"* (Hadid L, Salek L, Passos FS, Caldonazo T;
manuscript in preparation).

The repository reproduces every table and figure of the manuscript and its supplementary material from public, anonymized
data:

- **SIH/SUS** (DATASUS): reduced hospital admission files (RD), 2000–2026, and professional services files (SP), 2008–2026.
- **SIM** (DATASUS, TabNet): deaths with rheumatic heart disease (ICD-10 I00–I09) as underlying cause, 2000–2024.
- **IBGE**: population projections (table 7358) and official estimates; household income per capita (PNAD Contínua,
  SIDRA table 7531).

## Quick start (no download needed)

```bash
pip install -r requirements.txt
python valvar/fase6_serie_longa.py      # national series, joinpoint, rate ratios, Das Gupta (2003–2025)
python valvar/fase8_2003_2025.py        # regions, sex, triangulation with mortality, Das Gupta by region and sex
python valvar/fase9_suplemento.py       # coding audit, pandemic, states, age–period–cohort, Table 1
python valvar/fase10_revisao.py         # pre-pandemic comparison, surgical profile, valvuloplasty, in-hospital mortality, access
python valvar/fase10_uf_obito.py        # surgery vs mortality by state
python valvar/fase10_mal_definidas.py   # mortality corrected for ill-defined causes
python valvar/manuscrito/figuras_tabelas_v2.py   # Figures 1–4, S1–S3 and Tables 1–3
python valvar/manuscrito/suplemento_gerar.py     # Supplementary Tables S1–S18 (markdown)
```

Run the commands from the repository root. Results are written to `valvar/resultados_*/` (CSV, already included) and
figures to `valvar/manuscrito/figuras_v2/`. Re-running the scripts above reproduces the included results exactly.

## Contents

| Path | Content |
|---|---|
| `valvar/base_valvar_2003_2007.csv.gz` | Open valve surgery admissions, 2003–2007 (previous SIH procedure tables), one row per admission |
| `valvar/base_valvar_2008_2025.csv.gz` | Valve procedures, 2008–2025 (SIGTAP), one row per admission, with prosthesis and ring data from the SP file |
| `valvar/sim/` | SIM/TabNet tabulations (deaths I00–I09 by year, region, state, sex and age; ill-defined causes) and the scripts that download them |
| `valvar/ref/` | Procedure names (SIGTAP) and household income by state (SIDRA 7531) |
| `artigo/dados_ref/` | IBGE population projections and estimates; IPCA price index |
| `valvar/resultados_*/` | Aggregated results of each analysis phase (CSV) |
| `valvar/*.py` | Extraction and analysis scripts (phases 1–10) |
| `valvar/manuscrito/` | Scripts that build the manuscript figures and tables and the supplementary tables |
| `sih_crvm.py` | Resumable downloader for SIH/SUS RD files, used by the extraction scripts |

### Main variables of the analytical datasets

`N_AIH` (admission code, pseudonymized), `ano` (year of admission), `PROC_REA` (main procedure), `tipo` (procedure group),
`cirurgia` (open valve surgery), `DIAG_PRINC` (principal diagnosis, ICD-10), `etiologia` (rheumatic I05–I09, non-rheumatic
I34–I37, other), `reum_ampliada` (expanded rheumatic definition: I05–I09 plus Q22.5 and I38 at age ≥18 years), `idade`
(age, years), `sexo`, `uf_res`/`regiao_res` (state/region of residence), `uf_int` (state of the hospital), `CNES` (hospital,
pseudonymized), `urgencia`, `obito` (in-hospital death), `dias` (length of stay), `val` (amount paid), and, from 2008,
`n_protese`, `bio`, `mec` (prostheses billed), `anel` (annuloplasty ring), `atos_valva` (valve acts in the SP file).

## Privacy

All source data are public and anonymized. In the analytical datasets, admission numbers and hospital identifiers (CNES)
were replaced by meaningless codes, and a pseudonymized surgeon identifier was removed; the analyses use these fields only
as grouping keys, so results are unchanged. Death-certificate microdata (SIM 2016–2024), used only for an exploratory
analysis of maternal deaths, are not redistributed; they can be downloaded from DATASUS.

## Full re-extraction from the source files

The analytical datasets can be rebuilt from the raw DATASUS files with `valvar/sih_valvar.py` (RD) and
`valvar/sih_sp_valvar.py` (SP), followed by `valvar/fase1_valvar.py` and `valvar/fase2_valvar.py`. Downloads come from the
public healthbr-data mirror of DATASUS, which publishes a SHA-256 checksum per file (verified on download); the official
FTP is also supported. Re-extraction takes several hours and writes partial files to `saida_valvar/` (not included).
Phases 1–5 and parts of phase 7 read these partial files and were run before the analytical datasets were frozen; their
results are included in `valvar/resultados_fase1` … `resultados_fase7`.

## Licence

Code: MIT (see `LICENSE`). Derived datasets and results: CC BY 4.0. Source data: DATASUS and IBGE (public domain, Brazilian
Access to Information Law).

## Ethics

The study used only public, anonymized data. Under Brazilian National Health Council Resolution No. 510/2016 (Article 1,
sole paragraph, items II and III), it did not require ethics committee approval or informed consent.

## Citation

See `CITATION.cff`. A DOI will be assigned by Zenodo on the first release.
