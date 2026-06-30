"""Constantes compartilhadas para relatórios PDF/DOCX."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


AUTORES = "Raquel de Fátima Alves e Victor Guimarães Silva"
ORIENTADOR = "Prof. Rafael Pasquini"
INSTITUICAO = "Universidade Federal de Uberlândia — UFU"
DISCIPLINA = "PGC308A — Estimating Conformance to Service Level Agreements (SLAs)"
TITULO = (
    "Estimativa de Conformidade a SLAs em Serviços de Vídeo sob Demanda "
    "mediante Aprendizado de Máquina"
)
SUBTITULO = "Análise Exploratória, Regressão Linear, Classificação Logística e Extensão 5G"

VARIABLE_DESCRIPTIONS = {
    "all_..idle": "Percentual de tempo em que a CPU do servidor está ociosa (%)",
    "X..memused": "Percentual de memória RAM utilizada no servidor (%)",
    "proc.s": "Taxa de criação de processos no sistema (processos/s)",
    "cswch.s": "Taxa de troca de contexto da CPU (trocas/s)",
    "file.nr": "Quantidade de file handles abertos no sistema operacional",
    "sum_intr.s": "Taxa de interrupções de hardware e software (interrupções/s)",
    "ldavg.1": "Média de carga da CPU nos últimos 60 segundos",
    "tcpsck": "Número de sockets TCP atualmente em uso",
    "pgfree.s": "Taxa de liberação de páginas de memória (páginas/s)",
    "DispFrames": "Taxa de frames de vídeo exibidos no cliente (frames/s — fps)",
}

VARIABLE_LABELS_PT = {
    "all_..idle": "CPU ociosa (%)",
    "X..memused": "Memória usada (%)",
    "proc.s": "Taxa de processos (/s)",
    "cswch.s": "Troca de contexto (/s)",
    "file.nr": "File handles",
    "sum_intr.s": "Interrupções (/s)",
    "ldavg.1": "Carga CPU (1 min)",
    "tcpsck": "Sockets TCP",
    "pgfree.s": "Liberação de páginas (/s)",
    "DispFrames": "Taxa de frames (fps)",
}
