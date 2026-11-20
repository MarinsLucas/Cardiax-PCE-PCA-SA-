from pathlib import Path
import numpy as np
import matplotlib

# Se True, abre as janelas (DEPOIS de salvar). Se False, só salva os arquivos
# (backend sem janela, o mais confiável para gerar as imagens).
SHOW_FIGURES = False
if not SHOW_FIGURES:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import scienceplots

# %% 1. CONFIGURAÇÃO DE ESTILO E RESET DE LAYOUT
plt.style.use(['science', 'ieee', 'no-latex'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
    "font.size": 8,
    "axes.labelsize": 12,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,

    "figure.autolayout": False,
    "figure.constrained_layout.use": False,
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0
})

STYLE_CONFIG = {
    'histogram': {
        'facecolor': '#0072B2',
        'edgecolor': 'black',
        'linewidth': 0.5,
        'alpha': 0.7
    },
    'scatter_points': {
        'c': '#0072B2',
        'alpha': 0.5,
        's': 10,
        'edgecolors': 'none',
        'label': 'Samples'
    },
    'trend_line': {
        'color': '#D55E00',
        'linewidth': 1.5,
        'linestyle': '--',
        'label': 'Trend'
    },
    'stats_lines': {
        'mean_color': 'black',
        'percentile_color': '#D55E00'
    }
}

# Com 100k pontos o scatter vira uma curva sólida; plotar um subconjunto
# deixa a renderização mais leve. Use None para plotar tudo.
N_PLOT = 5000


# %% 2. FUNÇÕES AUXILIARES
def save_fig(fig, basename):
    """Salva PDF e PNG a partir da PRÓPRIA figura (não da 'figura atual')."""
    fig.savefig(f'{basename}.pdf', dpi=600)
    fig.savefig(f'{basename}.png', dpi=600)
    check_image(f'{basename}.png')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def check_image(path):
    """Relê o PNG e avisa se ele estiver em branco."""
    img = plt.imread(path)
    rgb = img[..., :3]
    ink = np.any(rgb < 0.99, axis=-1)
    if img.shape[-1] == 4:
        ink &= img[..., 3] > 0.05
    frac = float(np.mean(ink))
    status = "OK" if frac > 0.005 else "ALERTA: imagem aparentemente EM BRANCO"
    print(f"  {path}: {img.shape[1]}x{img.shape[0]} px, {100 * frac:.2f}% com conteúdo -> {status}")


# %% 3. LOAD DATA
ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
arquivo_dados = ROOT / 'results' / 'ef' / 'mc_EF_Cfib_samples.csv'   # gerado por src/train/ef_pce.py
print(f"Reading data from {arquivo_dados}...")

dados = np.loadtxt(arquivo_dados, delimiter=',', skiprows=1)
X_mc = dados[:, 0]         # Fibrosis Stiffness
Y_mc = dados[:, 1] * 100   # Ejection Fraction (%)

if not (np.all(np.isfinite(X_mc)) and np.all(np.isfinite(Y_mc))):
    raise ValueError(f"{arquivo_dados} contém NaN/Inf.")

N_simulacoes = len(Y_mc)
print(f"Data loaded: {N_simulacoes} samples.")
print(f"  X: [{X_mc.min():.4g}, {X_mc.max():.4g}] | "
      f"EF (%): [{Y_mc.min():.4g}, {Y_mc.max():.4g}]")

if N_PLOT is None or N_PLOT >= N_simulacoes:
    idx_plot = np.arange(N_simulacoes)
else:
    idx_plot = np.random.default_rng(0).choice(N_simulacoes, size=N_PLOT, replace=False)

# %% 4. STATISTICS CALCULATION
media = np.mean(Y_mc)
p5 = np.percentile(Y_mc, 5)
p95 = np.percentile(Y_mc, 95)

# %% 5. VISUALIZATION 1: HISTOGRAM (Ejection Fraction)
fig = plt.figure(figsize=(3.5, 2.5))
ax = fig.add_subplot(111)

ax.hist(Y_mc, bins=30, density=True, **STYLE_CONFIG['histogram'])

ax.axvline(media, color=STYLE_CONFIG['stats_lines']['mean_color'],
           linestyle='--', linewidth=1, label=f'Mean: {media:.1f}%')

ax.axvline(p5, color=STYLE_CONFIG['stats_lines']['percentile_color'],
           linestyle=':', linewidth=1, label='5-95% Percentiles')
ax.axvline(p95, color=STYLE_CONFIG['stats_lines']['percentile_color'],
           linestyle=':', linewidth=1)

ax.set_xlabel('Ejection Fraction (%)')
ax.set_ylabel('Probability Density')

ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', fontsize=6)

fig.subplots_adjust(left=0.18, right=0.96, bottom=0.18, top=0.95)

save_fig(fig, FIG_DIR / 'Histograma_UQ_EjectionFraction_IEEE')
print("Histogram saved.")

# %% 6. VISUALIZATION 2: SCATTER PLOT (Stiffness vs EF)
fig = plt.figure(figsize=(3.5, 2.5))
ax = fig.add_subplot(111)

ax.scatter(X_mc[idx_plot], Y_mc[idx_plot], **STYLE_CONFIG['scatter_points'])

ax.set_xlabel('Fibrosis Stiffness (Pa)')
ax.set_ylabel('Ejection Fraction (%)')

ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black', fontsize=6)

fig.subplots_adjust(left=0.18, right=0.96, bottom=0.18, top=0.95)

save_fig(fig, FIG_DIR / 'Scatter_EF_vs_Stiffness_IEEE')
print("Scatter plot saved.")