"""
Propagação de incerteza (média + IC 95 %) do Monte Carlo variando só C_fib.

Uso:
    python src/plots/propagation.py volume
    python src/plots/propagation.py pressure
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import scienceplots

# %% 0. CONFIGURAÇÃO PADRÃO (MESMO "DNA" DO visualiza.py)
plt.style.use(['science', 'ieee', 'no-latex'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "serif"],
    "axes.labelsize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "mathtext.fontset": "stix",

    "figure.autolayout": False,
    "figure.constrained_layout.use": False,
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0
})

# Dicionário de Estilos Semânticos
STYLE_CONFIG = {
    'mean_line': {
        'color': 'black',
        'linewidth': 1,
        'linestyle': '-'
    },
    'median_line': {
        'color': '#D55E00',
        'linewidth': 0.8,
        'linestyle': '--'
    },
    'uncertainty_band': {          # intervalo 95 %
        'color': '#0072B2',
        'alpha': 0.3,
        'linewidth': 0
    },
    'inner_band': {                # intervalo 50 % (sobreposto)
        'color': '#0072B2',
        'alpha': 0.5,
        'linewidth': 0
    },
    'bound_lines': {               # P2.5 / P97.5 em linhas
        'color': '#D55E00',
        'linewidth': 0.8,
        'linestyle': '--'
    },
    'realizations': {
        'alpha': 0.3,
        'linewidth': 0.3
    },
    'histogram': {
        'facecolor': '#0072B2',
        'edgecolor': 'black',
        'alpha': 0.6,
        'linewidth': 0.3
    },
    'scatter': {
        'c': '#0072B2',
        's': 10,
        'alpha': 0.5,
        'edgecolors': 'none'
    },
    'labels': {
        'time': 'Time (ms)',
        'volume': 'Volume (mL)',
        'pdf': 'Probability Density',
        'pressure': 'Left Ventricle Pressure (Pa)',
        'cfib': 'Fibrosis Stiffness'
    }
}

LEGEND_KW = dict(loc='best', frameon=True, fancybox=False, edgecolor='black')

# Tamanhos e margens fixas (mesmos valores dos scripts anteriores)
SIZE_WIDE, MARGINS_WIDE = (3.5, 2.625), dict(left=0.16, right=0.96, bottom=0.16, top=0.95)
SIZE_SMALL, MARGINS_SMALL = (2.3, 2.5), dict(left=0.22, right=0.95, bottom=0.18, top=0.95)

# %% 1. CONFIGURAÇÃO DO CONJUNTO DE DADOS
ap = argparse.ArgumentParser()
ap.add_argument('quantity', choices=['volume', 'pressure'])
QUANTITY = ap.parse_args().quantity

ROOT = Path(__file__).resolve().parents[2]
RES_DIR = ROOT / 'results' / QUANTITY
FIG_DIR = ROOT / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    'volume':   dict(x='mc_Cfib_X_volume_pcapce.npy', y='mc_Cfib_Y_volume_pcapce.npy',
                     color='#0072B2'),
    'pressure': dict(x='mc_Cfib_X_pcapce.npy', y='mc_Cfib_Y_pressure_pcapce.npy',
                     color='#D55E00'),
}
CFG = DATASETS[QUANTITY]
X_PATH = RES_DIR / CFG['x']                       # (N, 6)
Y_PATH = RES_DIR / CFG['y']                       # (N, 90)
TAG = QUANTITY.upper()                            # sufixo dos arquivos
YLABEL = STYLE_CONFIG['labels'][QUANTITY]
STYLE_CONFIG['uncertainty_band']['color'] = CFG['color']

# Passos para histogramas e dispersões (usa o mais próximo da grade)
SELECTED_STEPS = [181, 421, 891]
N_TRAJ = 300                             # trajetórias individuais no gráfico 4
N_SCATTER = 5000                         # pontos nos gráficos de dispersão

# %% 2. CARREGAR RESULTADOS DO MONTE CARLO
print("Carregando resultados do Monte Carlo...")

X = np.load(X_PATH)
Y = np.load(Y_PATH)

time_steps = np.arange(1, 901, 10)       # mesmos steps_interest do treino
C_fib = X[:, 5]
N = Y.shape[0]

assert Y.shape[1] == len(time_steps), (
    f"Y tem {Y.shape[1]} colunas, mas time_steps tem {len(time_steps)}"
)
assert X.shape[0] == N, "X e Y têm números de amostras diferentes"

print(f"Dados carregados com sucesso: {Y.shape}")

# %% 3. ESTATÍSTICAS
mean = Y.mean(axis=0)
std = Y.std(axis=0, ddof=1)
p_lo95, p_hi95 = np.percentile(Y, [2.5, 97.5], axis=0)
p_lo50, p_hi50 = np.percentile(Y, [25, 75], axis=0)
median = np.median(Y, axis=0)

# IC 95 % da MÉDIA (erro padrão) -- muito estreito com N = 100k
ci_mean_half = 1.96 * std / np.sqrt(N)
print(f"Maior meia-largura do IC 95% da média: {ci_mean_half.max():.3e}")

idx_sel = [int(np.argmin(np.abs(time_steps - s))) for s in SELECTED_STEPS]

rng = np.random.default_rng(0)
idx_traj = rng.choice(N, size=min(N_TRAJ, N), replace=False)
idx_sc = rng.choice(N, size=min(N_SCATTER, N), replace=False)


# %% 4. FUNÇÕES AUXILIARES
def new_fig(size, margins):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(**margins)
    return fig, ax


def save_fig(fig, name):
    fig.savefig(FIG_DIR / f'{name}.png', dpi=600)
    fig.savefig(FIG_DIR / f'{name}.pdf')
    plt.close(fig)
    print(f"Salvo: {name}.png / .pdf")


# %% 5. PLOT 1: UNCERTAINTY PROPAGATION (média + bandas de percentil)
fig, ax = new_fig(SIZE_WIDE, MARGINS_WIDE)

ax.fill_between(time_steps, p_lo95, p_hi95, label='95% CI',
                **STYLE_CONFIG['uncertainty_band'])
# ax.fill_between(time_steps, p_lo50, p_hi50, label='50% interval',
#                **STYLE_CONFIG['inner_band'])
# ax.plot(time_steps, median, label='Median', **STYLE_CONFIG['median_line'])
ax.plot(time_steps, mean, label='Mean', **STYLE_CONFIG['mean_line'])

ax.set_xlabel(STYLE_CONFIG['labels']['time'])
ax.set_ylabel(YLABEL)
ax.set_xlim([time_steps[0], time_steps[-1]])
ax.legend(fontsize=7, **LEGEND_KW)

save_fig(fig, f'Grafico_UQ_Propagacao_{TAG}')

# %% 6. PLOT 2: PERCENTIL vs. MÉDIA ± 1,96·σ

print("\n[OK] Gráficos gerados no padrão IEEE (Times New Roman).")