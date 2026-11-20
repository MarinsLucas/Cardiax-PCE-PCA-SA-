import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from sklearn.model_selection import train_test_split
from uqpylab import sessions

# ============================================================
# 0. CONFIGURAÇÃO
# ============================================================

from pathlib import Path

# --- Caminhos (relativos à raiz do repositório) ---
ROOT = Path(__file__).resolve().parents[2]
X_PATH = ROOT / 'data' / 'uniform_samples.csv'
V_PATH = ROOT / 'data' / 'volume.csv'
OUT_DIR = ROOT / 'results' / 'ef'
os.makedirs(OUT_DIR, exist_ok=True)


def out(name):
    return os.path.join(OUT_DIR, name)


# --- Parâmetros (mesmos do volume.py) ---
parameter_names = ['C', 'b_f', 'b_t', 'b_fs', 'T_ref', 'C_fib']

parameter_limits = [
    [585, 715],                   # C
    [5.958, 7.292],               # b_f
    [3.285, 4.015],               # b_t
    [2.385, 2.915],               # b_fs
    [75000 * 0.9, 75000 * 1.1],   # T_ref
    [1300, 6500],                 # C_fib
]

# --- Definição da EF ---
#   'max_min'     : EDV = max(V[:EDV_WINDOW]), ESV = min(V inteiro)  (como no ef.py)
#   'fixed_times' : EDV = V(T_EDV), ESV = V(T_ESV)                   (como no edv_esv_mc.py)
EF_MODE = 'max_min'
EDV_WINDOW = 600
T_EDV = 180
T_ESV = 420

# --- Treino / validação (mesmo split do volume.py) ---
N_VAL = 36 * 2
SEED_SPLIT = 42

# --- PCE (mesma configuração do volume.py) ---
PCE_DEGREES = np.arange(2, 10).tolist()
PCE_QNORM = 0.75

# --- Monte Carlo: só C_fib varia ---
N_MC = 100_000
SEED_MC = 42
fixed_values = {
    'C': 650.0,
    'b_f': 6.625,
    'b_t': 3.650,
    'b_fs': 2.650,
    'T_ref': 75000.0,
}
idx_cfib = parameter_names.index('C_fib')

# ============================================================
# 1. ESTILO DOS GRÁFICOS (PADRÃO IEEE)
# ============================================================

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
    "savefig.pad_inches": 0,
})

# ============================================================
# 2. CARREGAR DADOS E CALCULAR A EF
# ============================================================

print("=" * 70)
print("CARREGANDO DADOS E CALCULANDO EF")
print("=" * 70)

X_raw = np.loadtxt(X_PATH, delimiter=',', skiprows=1, usecols=(1, 2, 3, 4, 5, 6))
V_full = np.loadtxt(V_PATH, delimiter=',')

print(f"X_raw: {X_raw.shape} | V_full: {V_full.shape}")

if X_raw.shape[0] != V_full.shape[0]:
    raise ValueError("uniform_samples.csv e volume.csv têm números de linhas diferentes.")

# Os limites do PCE precisam conter os dados de treino
for j, name in enumerate(parameter_names):
    lo, hi = parameter_limits[j]
    dmin, dmax = X_raw[:, j].min(), X_raw[:, j].max()
    tol = 1e-9 * (hi - lo)
    if dmin < lo - tol or dmax > hi + tol:
        raise ValueError(
            f"'{name}': dados em [{dmin:.6g}, {dmax:.6g}] estão fora dos limites "
            f"[{lo}, {hi}]. Corrija parameter_limits."
        )
    cover = (dmax - dmin) / (hi - lo)
    flag = "  <-- ATENÇÃO: dados cobrem pouco do intervalo" if cover < 0.5 else ""
    print(f"{name:6s}: dados [{dmin:.6g}, {dmax:.6g}] | limites [{lo}, {hi}]{flag}")


def compute_ef(V):
    if EF_MODE == 'max_min':
        edv = V[:, :EDV_WINDOW].max(axis=1)
        esv = V.min(axis=1)
    elif EF_MODE == 'fixed_times':
        edv = V[:, T_EDV - 1]          # volume.csv tem passos 1..900 (colunas 0..899)
        esv = V[:, T_ESV - 1]
    else:
        raise ValueError("EF_MODE deve ser 'max_min' ou 'fixed_times'.")
    return (edv - esv) / edv, edv, esv


EF_all, EDV_all, ESV_all = compute_ef(V_full)
EF_all = EF_all.reshape(-1, 1)

print(f"EF (simulações): min={EF_all.min():.4f}, max={EF_all.max():.4f}, "
      f"média={EF_all.mean():.4f}")

np.savetxt(out('EF_calculada_simulacoes.csv'),
           np.column_stack((X_raw, EDV_all, ESV_all, EF_all)),
           delimiter=',', comments='',
           header=','.join(parameter_names) + ',EDV,ESV,EF')

X_train, X_val, Y_train, Y_val_true = train_test_split(
    X_raw, EF_all, test_size=N_VAL, random_state=SEED_SPLIT
)
print(f"Treino: {X_train.shape} | Validação: {X_val.shape}")

# ============================================================
# 3. UQCLOUD: TREINO DO PCE, VALIDAÇÃO E MONTE CARLO
# ============================================================

token = os.environ.get('UQCLOUD_TOKEN')
if not token:
    raise RuntimeError("Defina a variável de ambiente UQCLOUD_TOKEN com o seu token.")

mySession = sessions.cloud(host='https://uqcloud.ethz.ch', token=token)
uq = mySession.cli
mySession.reset()

try:
    # ---------- Input (6 marginais uniformes) ----------
    marginals = [
        {'Name': n, 'Type': 'Uniform', 'Parameters': lim}
        for n, lim in zip(parameter_names, parameter_limits)
    ]
    myInput = uq.createInput({'Marginals': marginals})

    # ---------- PCE ----------
    MetaOpts = {
        'Type': 'Metamodel',
        'MetaType': 'PCE',
        'Method': 'LARS',
        'TruncOptions': {'qNorm': PCE_QNORM},
        'Degree': PCE_DEGREES,
        'Input': myInput,
        'ExpDesign': {
            'X': X_train.tolist(),
            'Y': Y_train.tolist(),
            'Sampling': 'User',
        },
    }

    print("\nTreinando PCE para a EF...")
    myPCE = uq.createModel(MetaOpts)
    print("[SUCESSO] PCE treinado.")

    # ---------- Validação ----------
    Y_val_pred = np.asarray(uq.evalModel(myPCE, X_val)).reshape(-1, 1)

    loo_error = float(np.asarray(myPCE['Error']['LOO']).flatten()[0])

    with open(out('emulador_EF_6D.pkl'), 'wb') as f:
        pickle.dump(myPCE, f)

    # ---------- Monte Carlo: só C_fib varia ----------
    print(f"\nMonte Carlo com {N_MC} amostras (só C_fib varia)...")
    uq.rng(SEED_MC, 'twister')

    CfibInput = uq.createInput({
        'Marginals': [{
            'Name': 'C_fib',
            'Type': 'Uniform',
            'Parameters': parameter_limits[idx_cfib],
        }]
    })

    C_fib_mc = np.asarray(uq.getSample(CfibInput, N_MC, 'MC')).reshape(-1)

    X_mc = np.empty((N_MC, len(parameter_names)))
    for j, name in enumerate(parameter_names):
        X_mc[:, j] = C_fib_mc if name == 'C_fib' else fixed_values[name]

    for name, val in fixed_values.items():
        lo, hi = parameter_limits[parameter_names.index(name)]
        if not (lo <= val <= hi):
            raise ValueError(f"{name}={val} fora de [{lo}, {hi}]")

    EF_mc = np.asarray(uq.evalModel(myPCE, X_mc)).reshape(-1)
    print("Monte Carlo finalizado.")

finally:
    mySession.quit()

# ============================================================
# 4. MÉTRICAS DE VALIDAÇÃO
# ============================================================

res = Y_val_true - Y_val_pred
rmse_val = float(np.sqrt(np.mean(res ** 2)))
val_range = float(np.max(Y_val_true) - np.min(Y_val_true)) or 1e-12
nrmse_val = rmse_val / val_range
ss_tot = float(np.sum((Y_val_true - Y_val_true.mean()) ** 2))
r2_val = 1.0 - float(np.sum(res ** 2)) / ss_tot if ss_tot > 1e-15 else np.nan

print("\n" + "=" * 70)
print("VALIDAÇÃO DO PCE (EF)")
print("=" * 70)
print(f"RMSE  = {rmse_val:.4e}")
print(f"NRMSE = {nrmse_val:.4e}")
print(f"R²    = {r2_val:.6f}")
print(f"LOO   = {loo_error:.4e}")

np.savetxt(out('metricas_validacao_EF.csv'),
           np.array([[rmse_val, nrmse_val, r2_val, loo_error]]),
           delimiter=',', comments='',
           header='RMSE,NRMSE,R2,LOO', fmt='%.8e')

np.savetxt(out('validacao_EF_true_vs_pred.csv'),
           np.column_stack((Y_val_true, Y_val_pred)),
           delimiter=',', comments='', header='EF_sim,EF_pred', fmt='%.8e')

# ============================================================
# 5. RESULTADOS DO MONTE CARLO -> CSV
# ============================================================

if not np.all(np.isfinite(EF_mc)):
    raise ValueError("EF_mc contém NaN/Inf.")

n_out = int(np.sum((EF_mc < 0) | (EF_mc > 1)))
if n_out:
    print(f"[AVISO] {n_out} amostras de EF fora de [0, 1] ({100 * n_out / N_MC:.3f} %).")

np.savetxt(out('mc_EF_Cfib_samples.csv'),
           np.column_stack((C_fib_mc, EF_mc)),
           delimiter=',', comments='', header='C_fib,EF', fmt='%.8e')

# mc_EF_Cfib_samples.csv (C_fib, EF como fração 0-1) é lido por src/plots/ef_plots.py

mean = EF_mc.mean()
std = EF_mc.std(ddof=1)
p2_5, p5, p50, p95, p97_5 = np.percentile(EF_mc, [2.5, 5, 50, 95, 97.5])
ci_mean_half = 1.96 * std / np.sqrt(N_MC)

stats_names = ['Mean', 'Std', 'Min', 'P2.5', 'P5', 'P50', 'P95', 'P97.5', 'Max', 'CI95_mean_halfwidth']
stats_vals = [mean, std, EF_mc.min(), p2_5, p5, p50, p95, p97_5, EF_mc.max(), ci_mean_half]

np.savetxt(out('mc_EF_Cfib_stats.csv'),
           np.array(stats_vals).reshape(1, -1),
           delimiter=',', comments='', header=','.join(stats_names), fmt='%.8e')

print("\n" + "=" * 70)
print(f"ESTATÍSTICAS DA EF (Monte Carlo, N={N_MC})")
print("=" * 70)
for n_, v_ in zip(stats_names, stats_vals):
    print(f"{n_:22s}: {v_:.6f}  ({100 * v_:.3f} %)" if n_ != 'CI95_mean_halfwidth'
          else f"{n_:22s}: {v_:.3e}")

# ============================================================
# 6. GRÁFICOS
# ============================================================

EF_pct = 100 * EF_mc
mean_p, p2_5_p, p97_5_p = 100 * mean, 100 * p2_5, 100 * p97_5

# ---------- 6.1 Validação: simulado vs predito ----------
fig = plt.figure(figsize=(2.5, 2.5))
ax = fig.add_subplot(111)

vt, vp = 100 * Y_val_true, 100 * Y_val_pred
lo_, hi_ = min(vt.min(), vp.min()), max(vt.max(), vp.max())
buff = (hi_ - lo_) * 0.1 or 0.5
limits = [lo_ - buff, hi_ + buff]

ax.plot(limits, limits, 'k--', lw=1, label='Perfect Fit', zorder=1)
ax.scatter(vt, vp, c='#0072B2', s=15, alpha=0.7, edgecolors='none',
           label='Validation Data', zorder=2)

txt = '\n'.join((
    r'$\mathrm{NRMSE} = %.2e$' % nrmse_val,
    r'$R^2 = %.4f$' % r2_val,
    r'$\epsilon_{\mathrm{LOO}} = %.2e$' % loo_error,
))
ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=6, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                  edgecolor='lightgray', linewidth=0.5))

ax.set_xlabel('Simulated EF (%)')
ax.set_ylabel('Predicted EF (%)')
ax.set_xlim(limits)
ax.set_ylim(limits)
ax.grid(True, linestyle=':', alpha=0.5)
fig.subplots_adjust(left=0.22, right=0.95, bottom=0.18, top=0.95)
plt.savefig(out('Validacao_EF_PCE_IEEE.pdf'), dpi=600)
plt.savefig(out('Validacao_EF_PCE_IEEE.png'), dpi=600)
plt.close()

# ---------- 6.2 Histograma da EF com intervalo de 95 % ----------
fig = plt.figure(figsize=(2.5, 2.5))
ax = fig.add_subplot(111)

ax.hist(EF_pct, bins=60, density=True, color='#56B4E9',
        edgecolor='none', alpha=0.8)
ax.axvline(mean_p, color='k', lw=1.2, label=f'Mean: {mean_p:.2f}')
ax.axvline(p2_5_p, color='#D55E00', ls='--', lw=1,
           label=f'P2.5: {p2_5_p:.2f}')
ax.axvline(p97_5_p, color='#D55E00', ls='--', lw=1,
           label=f'P97.5: {p97_5_p:.2f}')

ax.set_xlabel('EF (%)')
ax.set_ylabel('PDF')
ax.legend(loc='best')
ax.grid(True, linestyle=':', alpha=0.5)
fig.subplots_adjust(left=0.22, right=0.95, bottom=0.18, top=0.95)
plt.savefig(out('MC_EF_histograma_IEEE.pdf'), dpi=600)
plt.savefig(out('MC_EF_histograma_IEEE.png'), dpi=600)
plt.close()

# ---------- 6.3 EF vs C_fib (subconjunto) com faixa de 95 % ----------
rng = np.random.default_rng(0)
idx_plot = rng.choice(N_MC, size=5000, replace=False)

fig = plt.figure(figsize=(2.5, 2.5))
ax = fig.add_subplot(111)

ax.axhspan(p2_5_p, p97_5_p, color='#CC79A7', alpha=0.15, lw=0)
ax.scatter(C_fib_mc[idx_plot], EF_pct[idx_plot], c='#CC79A7', s=10,
           alpha=0.5, edgecolors='none')
ax.axhline(mean_p, color='k', lw=1)

ax.set_xlabel('Fibrosis Stiffness')
ax.set_ylabel('EF (%)')
ax.grid(True, linestyle=':', alpha=0.5)
fig.subplots_adjust(left=0.22, right=0.95, bottom=0.18, top=0.95)
plt.savefig(out('MC_EF_vs_Cfib_IEEE.pdf'), dpi=600)
plt.savefig(out('MC_EF_vs_Cfib_IEEE.png'), dpi=600)
plt.close()

# ---------- 6.4 CDF empírica ----------
fig = plt.figure(figsize=(2.5, 2.5))
ax = fig.add_subplot(111)

sorted_ef = np.sort(EF_pct)
cdf = np.arange(1, N_MC + 1) / N_MC
ax.plot(sorted_ef, cdf, color='#0072B2', lw=1.2)
ax.axvline(p2_5_p, color='#D55E00', ls='--', lw=1)
ax.axvline(p97_5_p, color='#D55E00', ls='--', lw=1)

ax.set_xlabel('EF (%)')
ax.set_ylabel('CDF')
ax.set_ylim(0, 1)
ax.grid(True, linestyle=':', alpha=0.5)
fig.subplots_adjust(left=0.22, right=0.95, bottom=0.18, top=0.95)
plt.savefig(out('MC_EF_CDF_IEEE.pdf'), dpi=600)
plt.savefig(out('MC_EF_CDF_IEEE.png'), dpi=600)
plt.close()

print(f"\n[OK] Tudo salvo em '{OUT_DIR}/'.")