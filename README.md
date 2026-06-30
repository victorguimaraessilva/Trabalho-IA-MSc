# PGC308A — Estimativa de Conformidade a SLAs com Machine Learning

Trabalho prático da disciplina **PGC308A** (PPGCO/FACOM-UFU, 2026/1).

**Autores:** Victor Guimarães Silva e Raquel de Fátima Alves  
**Orientador:** Prof. Rafael Pasquini — FACOM/UFU

---

## Contexto

O SLA é definido como: serviço **conforme** quando a taxa de frames no cliente (`DispFrames`) for **≥ 18 fps**; caso contrário, **viola**. O objetivo é aprender uma função que estime a qualidade percebida a partir de 9 métricas do servidor Linux — sem medir diretamente no terminal do usuário.

---

## Estrutura

```
.
├── X.csv                           # Métricas do servidor (9 variáveis, 3600 obs.)
├── Y.csv                           # Taxa de frames no cliente (DispFrames, fps)
├── src/
│   ├── config.py                   # Constantes: SLA=18 fps, split 70/30, seed=42
│   ├── data.py                     # Carrega e une X + Y
│   ├── metrics.py                  # Funções NMAE e ERR
│   ├── splits.py                   # Divisão treino/teste e subconjuntos
│   ├── labels.py                   # Nomes das colunas em PT-BR
│   └── gnettrack.py                # Parser dos traces 5G (bônus)
├── notebooks/
│   ├── 01_task_i_exploracao.ipynb          # Task I — Exploração
│   ├── 02_task_ii_regressao.ipynb          # Task II — Regressão linear (NMAE)
│   ├── 03_task_iii_classificacao.ipynb     # Task III — Regressão logística (ERR)
│   └── 04_bonus_atividade_a_sinal_5g.ipynb # Bônus — Qualidade de sinal 5G
├── data/g-nettrack-pro/            # Traces de campo 5G (São Paulo)
├── figures/                        # Gráficos gerados pelos notebooks
└── results/                        # Tabelas de resultados em CSV
```

---

## Como executar

**1. Instalar dependências**

```bash
pip install -r requirements.txt
```

**2. Abrir os notebooks**

```bash
jupyter notebook notebooks/
```

Execute na ordem: `01` → `02` → `03` → `04`.  
Figuras são salvas em `figures/` e tabelas em `results/`.

---

## Dados

### Variáveis de entrada — servidor (X.csv)

| Variável | Significado | Unidade |
|----------|-------------|---------|
| `all_..idle` | CPU ociosa | % |
| `X..memused` | Memória RAM usada | % |
| `proc.s` | Taxa de criação de processos | proc/s |
| `cswch.s` | Trocas de contexto | trocas/s |
| `file.nr` | File handles abertos | contagem |
| `sum_intr.s` | Taxa de interrupções | int/s |
| `ldavg.1` | Média de carga (1 min) | adimensional |
| `tcpsck` | Sockets TCP em uso | contagem |
| `pgfree.s` | Liberação de páginas | páginas/s |

### Variável alvo — cliente (Y.csv)

| Variável | Significado | SLA |
|----------|-------------|-----|
| `DispFrames` | Taxa de frames no cliente | ≥ 18 fps → conforme |

**Protocolo:** 70% treino (2.520 obs.) / 30% teste (1.080 obs.), `random_state=42`.

---

## Resultados

### Task II — Regressão Linear (NMAE < 15% = acurado)

| Modelo | Treino | NMAE (%) |
|--------|:------:|:--------:|
| M₁ | 50 | 11,52 |
| M₂ | 500 | 10,58 |
| M₃ | 1000 | 10,49 |
| M₄ | 1500 | 10,38 |
| **M** | **2520** | **10,39** |

### Task III — Regressão Logística (ERR < 15% = acurado)

| Classificador | Treino | ERR (%) |
|:-------------:|:------:|:-------:|
| C₁ | 50 | 15,37 ❌ |
| C₂ | 500 | 12,96 |
| C₃ | 1000 | 12,87 |
| C₄ | 1500 | 11,39 |
| **C₅** | **2520** | **11,11** |

### Bônus — Predição de qualidade de sinal 5G

Atividade A do [5G Datasets Challenge (UNICAMP/INTRIG)](https://github.com/intrig-unicamp/hackathon5G) — regressão linear sobre traces coletados em campo em São Paulo.

| Métrica | Valor |
|---------|-------|
| MAE | 3,20 dB |
| RMSE | 3,80 dB |
| R² | 0,29 |

---

## Dependências

```
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.13
scikit-learn >= 1.3
jupyter >= 1.0
```
