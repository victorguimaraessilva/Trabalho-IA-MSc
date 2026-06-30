# PGC308A — Estimativa de Conformidade a SLAs com Machine Learning

Trabalho prático da disciplina **PGC308A** (PPGCO/FACOM-UFU, 2026/1).  
Tema: estimar conformidade a SLAs em um serviço de **vídeo sob demanda (VoD)**
usando aprendizado de máquina supervisionado, com extensão para dados de rede **5G** (atividade bônus).

**Autores:** Victor Guimarães Silva e Raquel de Fátima Alves  
**Orientador:** Prof. Rafael Pasquini — FACOM/UFU

---

## Sumário

- [Contexto](#contexto)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Dados](#dados)
- [Resultados Principais](#resultados-principais)
- [Como Reproduzir](#como-reproduzir)
- [Atividade Bônus — 5G](#atividade-bônus--5g)
- [Relatório](#relatório)
- [Dependências](#dependências)

---

## Contexto

O SLA é definido como: o serviço **conforma** quando a taxa de frames no cliente
(`DispFrames`) for **≥ 18 fps**; caso contrário, **viola**. O objetivo é aprender
uma função **M: X → Ŷ** que estime a qualidade percebida a partir de 9 métricas
do servidor Linux — sem precisar medir diretamente no terminal do usuário.

---

## Estrutura do Repositório

```
.
├── X.csv, Y.csv                    # Dados originais (servidor + cliente, 3600 obs.)
├── src/
│   ├── config.py                   # Constantes: SLA=18 fps, split 70/30, seed=42
│   ├── data.py                     # Carrega e une X + Y
│   ├── metrics.py                  # NMAE, ERR, rótulos SLA
│   ├── splits.py                   # Divisão treino/teste
│   ├── labels.py                   # Nomes das colunas em PT-BR
│   ├── gnettrack.py                # Parser dos traces g-nettrack-pro (bônus)
│   ├── report_config.py            # Metadados do relatório (autores, título)
│   └── report_data.py              # Pipeline de coleta de métricas para relatório
├── notebooks/
│   ├── 01_task_i_exploracao.ipynb          # Task I: Exploração dos dados
│   ├── 02_task_ii_regressao.ipynb          # Task II: Regressão linear (NMAE)
│   ├── 03_task_iii_classificacao.ipynb     # Task III: Regressão logística (ERR)
│   └── 04_bonus_atividade_a_sinal_5g.ipynb # Bônus: Qualidade de sinal 5G
├── data/
│   └── g-nettrack-pro/             # Traces de campo 5G (São Paulo — UNICAMP/INTRIG)
├── figures/                        # Gráficos gerados pelos notebooks
├── results/                        # Tabelas de resultados em CSV
└── relatorio/
    ├── relatorio.tex               # Fonte LaTeX do relatório final
    ├── relatorio.pdf               # Relatório compilado (PDF entregável)
    └── figs/                       # Figuras incluídas no relatório
```

---

## Dados

### Variáveis do servidor (X)

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

### Variável alvo (Y)

| Variável | Significado | SLA |
|----------|-------------|-----|
| `DispFrames` | Taxa de frames no cliente | ≥ 18 fps → conforme |

---

## Resultados Principais

### Task II — Regressão Linear

| Modelo | Treino (obs.) | NMAE (%) | Acurado (< 15%)? |
|--------|:---:|:---:|:---:|
| M₁ | 50 | 11,52 | ✅ |
| M₂ | 500 | 10,58 | ✅ |
| M₃ | 1000 | 10,49 | ✅ |
| M₄ | 1500 | 10,38 | ✅ |
| **M (completo)** | **2520** | **10,39** | ✅ |

### Task III — Regressão Logística

| Classificador | Treino (obs.) | ERR (%) | Acurado (< 15%)? |
|:---:|:---:|:---:|:---:|
| C₁ | 50 | 15,37 | ❌ |
| C₂ | 500 | 12,96 | ✅ |
| C₃ | 1000 | 12,87 | ✅ |
| C₄ | 1500 | 11,39 | ✅ |
| **C₅** | **2520** | **11,11** | ✅ |

---

## Como Reproduzir

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Execute na ordem: `01` → `02` → `03` → `04`. Figuras salvas em `figures/`, tabelas em `results/`.

**Protocolo:** 70% treino / 30% teste, `random_state=42`.

---

## Atividade Bônus — 5G

Atividade A do [5G Datasets Challenge (UNICAMP/INTRIG)](https://github.com/intrig-unicamp/hackathon5G):
predição de qualidade de sinal (`Qual`/RSRQ, dB) com regressão linear.

| Métrica | Valor |
|---------|-------|
| MAE | 3,20 dB |
| RMSE | 3,80 dB |
| R² | 0,29 |

---

## Relatório

O relatório final em PDF está em `relatorio/relatorio.pdf`.

Para recompilar:

```bash
cd relatorio
pdflatex relatorio.tex
pdflatex relatorio.tex   # segunda passagem para referências cruzadas
```

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
