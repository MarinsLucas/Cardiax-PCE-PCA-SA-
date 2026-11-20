from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# %% 1. CONFIGURAÇÃO DE ESTILO
plt.style.use(['science', 'ieee', 'no-latex'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
    "font.size": 8,
    "axes.labelsize": 12,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "figure.autolayout": False,
    "figure.constrained_layout.use": False,
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0
})

# %% 2. LEITURA DOS RESULTADOS DO MONTE CARLO (PCE)
print("--- Processando dados... ---")

ROOT = Path(__file__).resolve().parents[2]
RES_DIR = ROOT / 'results' / 'volume'
FIG_DIR = ROOT / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

X_full = np.load(RES_DIR / 'mc_Cfib_X_volume_pcapce.npy')   # (N, 6)
Y_mc = np.load(RES_DIR / 'mc_Cfib_Y_volume_pcapce.npy')     # (N, 90)

X_mc = X_full[:, 5]                          # C_fib (única coluna que varia)

# DEVE ser idêntico ao steps_interest usado no treino do PCE.
# Se você retreinar com range(10, 901, 10), troque aqui também.
time_steps = np.arange(1, 901, 10)

assert Y_mc.shape[1] == len(time_steps), (
    f"Y_mc tem {Y_mc.shape[1]} colunas, mas time_steps tem {len(time_steps)}"
)

print(f"Dados carregados. Amostras: {len(X_mc)}")

# %% 3. EXTRAÇÃO DO EDV e ESV
target_t1 = 180  # Tempo do EDV
target_t2 = 420  # Tempo do ESV

idx_edv = np.argmin(np.abs(time_steps - target_t1))
idx_esv = np.argmin(np.abs(time_steps - target_t2))

print(f"EDV: alvo {target_t1} -> passo usado {time_steps[idx_edv]}")
print(f"ESV: alvo {target_t2} -> passo usado {time_steps[idx_esv]}")

Y_EDV = Y_mc[:, idx_edv]
Y_ESV = Y_mc[:, idx_esv]

# Grandezas derivadas (mesmas amostras, então a correlação é preservada)
Y_SV = Y_EDV - Y_ESV
Y_EF = 100.0 * Y_SV / Y_EDV

# %% 4. ESTATÍSTICAS E INTERVALOS
def summarize(name, v):
    p_lo, p50, p_hi = np.percentile(v, [2.5, 50, 97.5])
    return [name, v.mean(), v.std(ddof=1), p_lo, p50, p_hi, v.min(), v.max()]

rows = [
    summarize('EDV', Y_EDV),
    summarize('ESV', Y_ESV),
    summarize('SV', Y_SV),
    summarize('EF(%)', Y_EF),
]

print("\n{:8s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
    "", "Mean", "Std", "P2.5", "P50", "P97.5", "Min", "Max"))
for r in rows:
    print("{:8s} ".format(r[0]) + " ".join(f"{x:10.3f}" for x in r[1:]))

with open(RES_DIR / 'EDV_ESV_stats.csv', 'w') as f:
    f.write('Quantity,Mean,Std,P2.5,P50,P97.5,Min,Max\n')
    for r in rows:
        f.write(r[0] + ',' + ','.join(f'{x:.6e}' for x in r[1:]) + '\n')

np.savetxt(
    RES_DIR / 'EDV_ESV_samples.csv',
    np.column_stack((X_mc, Y_EDV, Y_ESV, Y_SV, Y_EF)),
    delimiter=',',
    header='C_fib,EDV,ESV,SV,EF_percent',
    comments='',
    fmt='%.8e'
)

# %% 5. CÁLCULO DA "MESMA PROPORÇÃO" (COMMON SPAN)
min_edv, max_edv = Y_EDV.min(), Y_EDV.max()
min_esv, max_esv = Y_ESV.min(), Y_ESV.max()

max_range = max(max_edv - min_edv, max_esv - min_esv) * 1.1

mid_edv = (max_edv + min_edv) / 2
mid_esv = (max_esv + min_esv) / 2

ylim_edv = [mid_edv - max_range / 2, mid_edv + max_range / 2]
ylim_esv = [mid_esv - max_range / 2, mid_esv + max_range / 2]

print(f"\nSpan visual calculado: {max_range:.2f} mL")

# %% 6. FUNÇÃO DE PLOTAGEM
# Com 100k pontos o scatter vira uma curva sólida; plotar um subconjunto
# deixa o PNG mais leve. Use N_PLOT = None para plotar tudo.
N_PLOT = 5000
rng = np.random.default_rng(0)
idx_plot = (np.arange(len(X_mc)) if N_PLOT is None
            else rng.choice(len(X_mc), size=min(N_PLOT, len(X_mc)), replace=False))


def plot_scatter_ieee(x, y, filename, ylims, color, ylabel_text=None):
    fig = plt.figure(figsize=(2.3, 2.5))
    ax = fig.add_subplot(111)

    ax.scatter(x[idx_plot], y[idx_plot], alpha=0.5, c=color,
               edgecolors='none', s=10)

    ax.set_ylim(ylims)
    ax.set_xlabel('Fibrosis Stiffness (Pa)')

    if ylabel_text:
        ax.set_ylabel(ylabel_text)

    fig.subplots_adjust(left=0.22, right=0.95, bottom=0.18, top=0.95)

    plt.savefig(FIG_DIR / filename, dpi=600)
    plt.close()
    print(f"Salvo: {filename}")


# %% 7. GERAR GRÁFICOS
plot_scatter_ieee(X_mc, Y_EDV, 'Scatter_EDV_IEEE.png',
                  ylim_edv, 'royalblue', ylabel_text='End Diastolic Volume (mL)')

plot_scatter_ieee(X_mc, Y_ESV, 'Scatter_ESV_IEEE.png',
                  ylim_esv, '#CC79A7', ylabel_text='End Systolic Volume (mL)')

print("\n[OK] Gráficos gerados com sucesso.")