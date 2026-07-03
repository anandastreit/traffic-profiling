# PARAFAC — Traffic Profiles COVID

Modelo PARAFAC de **5 componentes (nComp=5)** treinado no período de referência
pré-COVID `2019-08-19_2019-09-22` para classificar traffic profiles de usuários ISP
ao longo da pandemia (2018–2022).

**Dados em:** `D:\land\land_sueste\sueste_old\st1\matlab\parafac_coronavirus\`

---

## Conteúdo desta pasta

```
matlab/
  README.md                  ← este arquivo
  data/
    loadings_all.csv         ← todos os loadings A, B, C extraídos (referência + semanas)
  scripts/
    extract_loadings.py      ← script principal de extração
    verify_model5.py         ← verifica se Model5 do .mat bate com os CSVs canônicos
    check_ncomp.py           ← inspeciona quais nComp estão em cada .mat
```

### data/loadings_all.csv

| Coluna | Descrição |
|--------|-----------|
| `source` | Período do arquivo (ex: `2019-08-19_2019-09-22`) |
| `type` | `reference` (modelo final de referência) ou `weekly` (modelos semanais) |
| `week` | `reference` ou número da semana no período (1, 2, 3...) |
| `week_date` | `reference` ou datas reais da semana (ex: `2019-08-19_2019-08-25`) |
| `mode` | `A` (usuários) / `B` (tempo) / `C` (métrica) |
| `row` | Índice da linha: usuário (A), minuto do dia 0–1439 (B), 0=down/1=up (C) |
| `comp_1`..`comp_5` | Valores dos 5 componentes PARAFAC |

**Resumo de linhas:**
- `reference` A: 58.048 linhas (usuários × 5 componentes)
- `reference` B: 1.440 linhas (minutos do dia × 5 componentes)
- `reference` C: 2 linhas (down/up × 5 componentes)
- `weekly` A: 2.660.901 linhas (todos os períodos, todas as semanas)
- `weekly` B: 177.120 linhas
- `weekly` C: 246 linhas
- **Total: 2.897.757 linhas**

---

## Modelo final de referência

### CSVs canônicos (fonte primária)

```
D:\land\land_sueste\sueste_old\st1\matlab\parafac_coronavirus\
  tables\models\train\2019-08-19_2019-09-22\
    model5_mode1_trafficDownUp_train_noWeights_filter_NaNs.csv  ← A  (58048 × 5)
    model5_mode2_trafficDownUp_train_noWeights.csv              ← B  (1440  × 5)
    model5_mode3_trafficDownUp_train_noWeights.csv              ← C  (2     × 5)
```

- **A** — loading por usuário, passou por `fac2let()` no MATLAB (escalonamento de variância para modo 1). Versão correta para treino do classifier.
- **B** — padrão temporal (minutos do dia). **Fixado** na classificação de novos períodos.
- **C** — row 0 = traffic_down, row 1 = traffic_up. **Fixado** na classificação de novos períodos.

### .mat de origem

```
D:\land\land_sueste\sueste_old\st1\matlab\parafac_coronavirus\
  models_1-08_trafficDownUp_2019-08-19_2019-09-22_noWeights.mat
```

- Único dos três arquivos `models_1-08_*` que contém `Model5` no topo
- B e C dos CSVs conferem com o .mat (verificado numericamente: max_diff < 5e-6)
- A difere por escalonamento de `fac2let()` — os CSVs são a versão correta
- Os arquivos `_filter_NaNs.mat` e `_filter_NaNs_v2.mat` só têm até nComp4

---

## Classifier

```
D:\land\land_sueste\sueste\gigalink_data_analysis\ananda_notebooks\parafac_coronavirus\
  run_classification\
    classify_loadings.py
    input\2019-08-19_2019-09-22\model5_cluster{N}\
      decision_tree_max_depth6_min_samples_leaf5.pkl
```

**Pipeline completo:**
1. A de referência → fit `MinMaxScaler` → treina `DecisionTreeClassifier(max_depth=6)`
2. Novo período → MATLAB `run_offline_singleUsers` re-fita apenas A (B e C fixos)
3. A novo → mesma escala → predict → cluster (A, B, C, D, E...)

---

## Modelos semanais

```
D:\land\land_sueste\sueste_old\st1\matlab\parafac_coronavirus\
  weekModel5_trafficDownUp_{DATES}_seed456_v2.mat   (32 arquivos, 2018-08-20 a 2022-01-30)
```

- Versão final: sufixo `_seed456_v2` (seed=456, NCOMP=5 fixo em `model_weeks.m`)
- Versões `_seed456` e sem sufixo são intermediárias — não usar
- Estrutura interna:
  ```
  Model.week1.model  →  {A (n_users×5), B (1440×5), C (2×5)}
  Model.week1.date   →  'YYYY-MM-DD_YYYY-MM-DD'
  ```

---

## Como re-rodar a extração

```bash
# Dependências (instalar uma vez)
py -m pip install scipy h5py pandas numpy

# Rodar
cd D:\land\land_sueste\sueste_old\st1\matlab\parafac_coronavirus
py extract_loadings.py
# Saída: loadings_all.csv (~427 MB) na mesma pasta
# Copiar para cá: data\loadings_all.csv
```

> Usar `py` (não `python`) — Python 3.14 instalado em `C:\Users\nanda\AppData\Local\Python\`

---

## Mapa completo dos arquivos .mat

| Arquivo | Conteúdo | Uso |
|---------|----------|-----|
| `models_1-08_..._noWeights.mat` | Model3–Model7 + Dataset | **Origem do modelo final (Model5)** |
| `models_1-08_..._filter_NaNs.mat` | Model4 + Dataset(nComp4) | Comparação — sem Model5 |
| `models_1-08_..._filter_NaNs_v2.mat` | Dataset(nComp3, nComp4) | Comparação — sem Model5 |
| `weekModels_1e-08_..._filter_NaNs.mat` | Week1–5 × Model2–7 | Tucker congruence validation |
| `weekModel5_..._seed456_v2.mat` (×32) | Model.week1–N × {A,B,C} | **Modelos deployados** |
| `jackknife_Week4.mat` | 25 GB | Análise de estabilidade jackknife |
