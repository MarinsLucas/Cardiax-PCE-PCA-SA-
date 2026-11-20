# Emuladores PCA-PCE para quantificação de incerteza em um modelo de ventrículo esquerdo

Emuladores (metamodelos) por **PCA + Polynomial Chaos Expansion** treinados sobre 336 simulações
de elementos finitos do ventrículo esquerdo. Depois do treino, os emuladores são usados em um
Monte Carlo (N = 100 000) em que **só a rigidez da fibrose (`C_fib`) varia**, para propagar a
incerteza até:

- volume e pressão do VE ao longo do ciclo;
- fração de ejeção (EF), EDV e ESV;
- deformação (strain) e tensão (stress) da fibra nas 17 regiões AHA.

## Parâmetros de entrada (distribuição uniforme)

| Parâmetro | Faixa |
|---|---|
| `C` | 585 – 715 |
| `b_f` | 5.958 – 7.292 |
| `b_t` | 3.285 – 4.015 |
| `b_fs` | 2.385 – 2.915 |
| `T_ref` | 67 500 – 82 500 |
| `C_fib` | 1 300 – 6 500 |

## Estrutura

```
data/
  uniform_samples.csv       336 amostras dos 6 parâmetros (entrada X)
  volume.csv, pressure.csv  336 simulações × passos de tempo (saídas Y)
  aha/
    strain_ahaN.csv, stress_ahaN.csv   N = 1..17, 336 simulações × 902 instantes
    aha_map.csv, time.csv, simulation_order.csv   metadados da extração
src/
  train/        treino dos emuladores + Monte Carlo (usam o UQCloud)
  plots/        figuras e tabelas a partir de results/
results/        criado ao rodar os scripts (não versionado)
```

## Como funciona (PCA-PCE)

1. Seleciona 90 instantes de tempo (passos 1, 11, …, 891) de cada curva.
2. Divide em treino/validação (264 / 72 simulações, `random_state=42`).
3. Aplica PCA às curvas de treino e mantém as componentes que explicam 99.999 % da variância.
4. Treina um PCE (UQ[py]Lab) para cada componente principal.
5. Reconstrói a curva completa a partir das componentes previstas e valida (NRMSE, R², LOO).
6. Monte Carlo: fixa 5 parâmetros e amostra `C_fib`, avaliando o emulador.

A EF é escalar, então `ef_pce.py` usa um PCE direto (sem PCA), calculado a partir de `volume.csv`.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

O treino usa o [UQCloud](https://uqcloud.ethz.ch) (UQ[py]Lab). Crie uma conta, gere um token e exporte:

```bash
export UQCLOUD_TOKEN='seu_token_aqui'
```

## Como rodar

Todos os comandos são executados **a partir da raiz do repositório**. Os scripts resolvem os
caminhos sozinhos e gravam tudo em `results/`.

### 1. Treino + Monte Carlo

```bash
python src/train/volume_pcapce.py        # -> results/volume/
python src/train/pressure_pcapce.py      # -> results/pressure/
python src/train/ef_pce.py               # -> results/ef/
python src/train/aha_markers_pcapce.py   # -> results/aha/{strain,stress}/AHAxx/   (demorado: 34 emuladores)
```

### 2. Figuras e tabelas

```bash
python src/plots/propagation.py volume
python src/plots/propagation.py pressure
python src/plots/edv_esv.py
python src/plots/pressure_scatter.py
python src/plots/ef_plots.py
python src/plots/aha_cfib_stats.py strain
python src/plots/aha_cfib_stats.py stress
python src/plots/make_latex_table.py results/aha/strain --output results/tabela_strain.tex
python src/plots/make_latex_table.py results/aha/stress --quantity "regional fiber stress" \
       --range-name stress --label tab:pce_stress --output results/tabela_stress.tex
```

As figuras vão para `results/figures/`. As das AHAs ficam em `results/aha/<q>/AHAxx/Cfib/`.

## O que cada script gera

| Script | Lê | Gera |
|---|---|---|
| `train/volume_pcapce.py` | `data/uniform_samples.csv`, `data/volume.csv` | emulador `.pkl`, métricas de validação, 90 gráficos de validação, `mc_Cfib_{X,Y}_volume_pcapce.npy`, estatísticas do MC |
| `train/pressure_pcapce.py` | `data/uniform_samples.csv`, `data/pressure.csv` | idem, para pressão |
| `train/ef_pce.py` | `data/uniform_samples.csv`, `data/volume.csv` | EF das simulações, emulador, validação, `mc_EF_Cfib_samples.csv`, histograma/CDF |
| `train/aha_markers_pcapce.py` | `data/uniform_samples.csv`, `data/aha/*` | por região/quantidade: emulador, métricas, gráficos de validação, MC de `C_fib`; `global_summary/` |
| `plots/propagation.py` | `results/{volume,pressure}/mc_*.npy` | `Grafico_UQ_Propagacao_{VOLUME,PRESSURE}` |
| `plots/edv_esv.py` | `results/volume/mc_*.npy` | `Scatter_EDV_IEEE`, `Scatter_ESV_IEEE`, `EDV_ESV_stats.csv` |
| `plots/pressure_scatter.py` | `results/pressure/mc_*.npy` | `Scatter_Pressure_IEEE` |
| `plots/ef_plots.py` | `results/ef/mc_EF_Cfib_samples.csv` | `Histograma_UQ_EjectionFraction_IEEE`, `Scatter_EF_vs_Stiffness_IEEE` |
| `plots/aha_cfib_stats.py` | `results/aha/<q>/AHAxx/Cfib/mc_Cfib_stats.csv` | `Grafico_UQ_Stats_Cfib` por região |
| `plots/make_latex_table.py` | `results/aha/<q>/AHAxx/metrics_*.csv` | tabela LaTeX de validação |

## Origem dos dados

As simulações foram feitas com o solver de elementos finitos
[cardiax](https://github.com/FISIOCOMP-UFJF/cardiax) (saídas `.h5`; ver
`data/aha/simulation_order.csv`). Os CSVs de `data/aha/` foram extraídos desses arquivos, com a média
por região AHA usando `aha_map.csv` (elemento → região).
