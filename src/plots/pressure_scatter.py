from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# %% 1. CONFIGURAÇÃO DE ESTILO (IDENTICA AOS ANTERIORES)
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
print("--- Processando dados de PRESSÃO... ---")

# Arquivos gerados por src/train/pressure_pcapce.py
ROOT = Path(__file__).resolve().parents[2]
RES_DIR = ROOT / 'results' / 'pressure'
FIG_DIR = ROOT / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
X_PATH = RES_DIR / 'mc_Cfib_X_pcapce.npy'               # (N, 6)
Y_PATH = RES_DIR / 'mc_Cfib_Y_pressure_pcapce.npy'      # (N, 90)

X_full = np.load(X_PATH)
Y_mc = np.load(Y_PATH)

X_mc = X_full[:, 5]                      # C_fib (única coluna que varia)

# DEVE ser idêntico ao steps_interest usado no treino do PCE de pressão.
time_steps = np.arange(1, 901, 10)

assert Y_mc.shape[1] == len(time_steps), (
    f"Y_mc tem {Y_mc.shape[1]} colunas, mas time_steps tem {len(time_steps)}"
)
assert X_mc.shape[0] == Y_mc.shape[0], "X e Y têm números de amostras diferentes"

print(f"Dados carregados. Amostras: {len(X_mc)}")

# %% 3. EXTRAÇÃO DOS DADOS (Instante 300)
target_t = 300
idx_300 = np.argmin(np.abs(time_steps - target_t))

Y_Pressao = Y_mc[:, idx_300]

print(f"Dados extraídos: alvo t = {target_t} -> passo usado {time_steps[idx_300]}.")
print(f"Pressão: média={Y_Pressao.mean():.2f}, std={Y_Pressao.std(ddof=1):.2f}, "
      f"P2.5={np.percentile(Y_Pressao, 2.5):.2f}, P97.5={np.percentile(Y_Pressao, 97.5):.2f}")

# %% 4. PLOTAGEM (PADRÃO TÉCNICO)
# Com 100k pontos o scatter vira uma curva sólida; plotar um subconjunto
# deixa o arquivo mais leve (mesmo critério do EDV_ESV.py).
# Use N_PLOT = None para plotar tudo.
N_PLOT = 5000
rng = np.random.default_rng(0)
idx_plot = (np.arange(len(X_mc)) if N_PLOT is None
            else rng.choice(len(X_mc), size=min(N_PLOT, len(X_mc)), replace=False))

# Tamanho (2.3, 2.5) para ser o 3º da linha
fig = plt.figure(figsize=(2.3, 2.5))
ax = fig.add_subplot(111)

cor_pontos = 'crimson'

# Scatter Plot (s=10 para consistência com os anteriores)
ax.scatter(X_mc[idx_plot], Y_Pressao[idx_plot], alpha=0.5, c=cor_pontos,
           edgecolors='none', s=10)

# Labels
ax.set_xlabel('Fibrosis Stiffness (Pa)')
ax.set_ylabel('Peak Systolic Pressure (Pa)')

# --- MARGENS FIXAS (PARA OS 3 GRÁFICOS FICAREM ALINHADOS) ---
fig.subplots_adjust(left=0.22, right=0.95, bottom=0.18, top=0.95)

# Salvar
arquivo_final = FIG_DIR / 'Scatter_Pressure_IEEE.pdf'
plt.savefig(arquivo_final, dpi=600)
plt.savefig(arquivo_final.with_suffix('.png'), dpi=600)
plt.close()

print(f"\n[SUCESSO] Gráfico salvo: {arquivo_final}")