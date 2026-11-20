"""
Gráfico de média + IC 95 % do Monte Carlo variando só C_fib, por região AHA.

Lê   results/aha/<quantity>/AHAxx/Cfib/mc_Cfib_stats.csv
Salva Grafico_UQ_Stats_Cfib.png/.pdf na mesma pasta Cfib/.

Uso:
    python src/plots/aha_cfib_stats.py strain            # todas as AHAs
    python src/plots/aha_cfib_stats.py stress --aha 3 4 5 7
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registra os estilos 'science'/'ieee')

# %% 0. CONFIGURAÇÃO PADRÃO
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

BAND_COLOR = {
    'strain': '#D55E00',
    'stress': '#0072B2',
}

MEAN_LINE = {'color': 'black', 'linewidth': 1, 'linestyle': '-'}

OUTPUT_NAME = 'Grafico_UQ_Stats_Cfib'

ROOT = Path(__file__).resolve().parents[2]


def plot_one(stats_path, color):
    data = np.loadtxt(stats_path, delimiter=',', skiprows=1)  # TimeStep, Mean, Std, P2.5, P50, P97.5

    time_steps = data[:, 0]
    mean = data[:, 1]
    p_lo = data[:, 3]
    p_hi = data[:, 5]

    # Sem título e sem rótulos nos eixos (montagem final feita fora do Python)
    fig = plt.figure(figsize=(3.5, 2.625))     # IEEE, coluna simples
    ax = fig.add_subplot(111)

    ax.fill_between(time_steps, p_lo, p_hi, color=color, alpha=0.5, linewidth=0)
    ax.plot(time_steps, mean, **MEAN_LINE)
    ax.set_xlim([time_steps[0], time_steps[-1]])

    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.09, top=0.96)

    out = stats_path.parent / OUTPUT_NAME
    fig.savefig(out.with_suffix('.png'), dpi=600)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    print(f"[OK] {out.relative_to(ROOT)}.png / .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('quantity', choices=['strain', 'stress'])
    ap.add_argument('--aha', type=int, nargs='*', default=list(range(1, 18)),
                    help="IDs das regiões AHA (padrão: 1 a 17)")
    args = ap.parse_args()

    base = ROOT / 'results' / 'aha' / args.quantity
    for aha_id in args.aha:
        stats_path = base / f"AHA{aha_id:02d}" / 'Cfib' / 'mc_Cfib_stats.csv'
        if not stats_path.exists():
            print(f"[AVISO] não encontrado: {stats_path.relative_to(ROOT)}")
            continue
        plot_one(stats_path, BAND_COLOR[args.quantity])


if __name__ == '__main__':
    main()
