from uqpylab import sessions, display_util
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA  # [PCA-PCE]
from pathlib import Path

# ------------------------------------------------------------
# Caminhos (relativos à raiz do repositório)
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results" / "pressure"
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(OUT_DIR)   # todas as saídas deste script vão para results/pressure/


# ============================================================
# 0. CONFIGURAÇÃO
# ============================================================

# ------------------------------------------------------------
# Parâmetros de entrada
# ------------------------------------------------------------

parameter_names = [
    'C',
    'b_f',
    'b_t',
    'b_fs',
    'T_ref',
    'C_fib'
]

# Limites das 6 variáveis
parameter_limits = [
    [585, 715],          # C
    [5.958, 7.292],      # b_f
    [3.285, 4.015],      # b_t
    [2.385, 2.915],      # b_fs
    [75000*0.9, 75000*1.1],    # T_ref
    [1300, 6500]         # C_fib
]

# [PCA-PCE] Fração da variância explicada mantida pela PCA
VAR_THRESHOLD = 0.99999 #999


# ============================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================

print("=" * 70)
print("CARREGANDO DADOS")
print("=" * 70)

X_raw = np.loadtxt(
    DATA_DIR / 'uniform_samples.csv',
    delimiter=',',
    skiprows=1,
    usecols=(1, 2, 3, 4, 5, 6)
)

print(f"Dimensão X_raw: {X_raw.shape}")

if X_raw.shape[1] != 6:
    raise ValueError(
        f"Esperados 6 parâmetros, mas X_raw possui "
        f"{X_raw.shape[1]} colunas."
    )


# ------------------------------------------------------------
# Saídas Y = Pressure
# ------------------------------------------------------------

Y_raw_full = np.loadtxt(
    DATA_DIR / 'pressure.csv',
    delimiter=','
)

print(f"Dimensão Y_raw_full: {Y_raw_full.shape}")


# ============================================================
# 2. SELEÇÃO DOS INSTANTES DE TEMPO
# ============================================================

# Tempos de interesse
steps_interest = [
    i for i in range(1, 901, 10)
]

print("\nInstantes utilizados:")
print(steps_interest)
indices_python = [
    i - 1 for i in steps_interest
]

# Selecionar somente os tempos desejados
Y_raw_selected = Y_raw_full[:, indices_python]

print(
    f"\nDimensão Y selecionado: "
    f"{Y_raw_selected.shape}"
)

print(
    f"Número de outputs do PCE: "
    f"{Y_raw_selected.shape[1]}"
)


# ============================================================
# 3. SEPARAÇÃO TREINO / VALIDAÇÃO
# ============================================================

# [PCA-PCE] As curvas completas (900 passos) são separadas junto,
# com as mesmas linhas de treino/validação.
X_train, X_val, Y_train, Y_val_true, Y_train_full, Y_val_full = train_test_split(
    X_raw,
    Y_raw_selected,
    Y_raw_full,
    test_size=36*2,
    random_state=42
)

print("\n" + "=" * 70)
print("SEPARAÇÃO TREINO / VALIDAÇÃO")
print("=" * 70)

print(f"Total de amostras : {len(X_raw)}")
print(f"Treinamento X     : {X_train.shape}")
print(f"Treinamento Y     : {Y_train.shape}")
print(f"Validação X       : {X_val.shape}")
print(f"Validação Y       : {Y_val_true.shape}")


# ============================================================
# [PCA-PCE] 3b. PCA NAS CURVAS COMPLETAS DE TREINO
# ============================================================

print("\n" + "=" * 70)
print("PCA DAS CURVAS DE TREINO")
print("=" * 70)

pca = PCA(
    n_components=VAR_THRESHOLD,
    svd_solver='full'
).fit(Y_train_full)

n_pc = pca.n_components_

# Scores das componentes principais (saídas dos PCEs)
Z_train = pca.transform(Y_train_full)

print(f"Componentes mantidas : {n_pc}")
print(
    f"Variância explicada  : "
    f"{100 * pca.explained_variance_ratio_.sum():.6f}%"
)

# Erro de truncamento da PCA (limite inferior do erro do PCA-PCE)
Y_val_proj = pca.inverse_transform(
    pca.transform(Y_val_full)
)[:, indices_python]

print(
    f"NRMSE médio só da PCA (validação): "
    f"{np.mean(np.sqrt(np.mean((Y_val_true - Y_val_proj)**2, axis=0)) / np.ptp(Y_val_true, axis=0)):.6e}"
)


# ============================================================
# 4. CONEXÃO COM UQCLOUD
# ============================================================

print("\n" + "=" * 70)
print("CONECTANDO AO UQCLOUD")
print("=" * 70)

myToken = os.environ.get('UQCLOUD_TOKEN')
if not myToken:
    raise RuntimeError(
        "Defina a variável de ambiente UQCLOUD_TOKEN com o seu token do UQCloud:\n"
        "    export UQCLOUD_TOKEN='SEU_TOKEN_AQUI'"
    )

UQCloud_instance = 'https://uqcloud.ethz.ch'

mySession = sessions.cloud(
    host=UQCloud_instance,
    token=myToken
)

uq = mySession.cli

# mySession.reset()

display_util.load_plt_defaults()

uq_colors = display_util.get_uq_color_order()


# ============================================================
# 5. DEFINIÇÃO DAS 6 MARGINAIS
# ============================================================

print("\n" + "=" * 70)
print("DEFININDO AS 6 MARGINAIS")
print("=" * 70)

marginals = []

for name, limits in zip(
    parameter_names,
    parameter_limits
):

    marginal = {
        'Name': name,
        'Type': 'Uniform',
        'Parameters': limits
    }

    marginals.append(marginal)

    print(
        f"{name:15s}: "
        f"Uniform({limits[0]}, {limits[1]})"
    )


InputOpts = {
    'Marginals': marginals
}

myInput = uq.createInput(
    InputOpts
)


# ============================================================
# 6. CONFIGURAÇÃO DO PCE
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURANDO PCE")
print("=" * 70)

MetaOpts = {
    'Type': 'Metamodel',
    'MetaType': 'PCE',
    'Method': 'LARS', 

    # Truncamento hiperbolico
    'TruncOptions': {
        'qNorm': 0.75
    },

    # Graus candidatos
    'Degree': np.arange(2, 10).tolist(),

    'Input': myInput
}

# Dados de treinamento
MetaOpts['ExpDesign'] = {
    'X': X_train.tolist(),
    'Y': Z_train.tolist(),   # [PCA-PCE] scores das componentes
    'Sampling': 'User'
}


# ============================================================
# 7. TREINAMENTO
# ============================================================

print("\n" + "=" * 70)
print("TREINANDO PCE")
print("=" * 70)

myPCE = uq.createModel(
    MetaOpts
)

print("[SUCESSO] PCE treinado.")


# ============================================================
# 8. PREDIÇÃO DA VALIDAÇÃO
# ============================================================

print("\n" + "=" * 70)
print("VALIDAÇÃO")
print("=" * 70)

# [PCA-PCE] Predição: PCE -> scores -> reconstrução da curva completa
# -> seleção dos mesmos 90 instantes do script original
def predict_pressure(X):
    Z = np.asarray(
        uq.evalModel(myPCE, X)
    ).reshape(len(X), -1)
    return pca.inverse_transform(Z)[:, indices_python]


Y_val_pred = predict_pressure(
    X_val
)

print(
    f"Predição realizada: "
    f"{Y_val_pred.shape}"
)


# ============================================================
# 9. EXTRAÇÃO DAS INFORMAÇÕES DO PCE
# ============================================================

print("\n" + "=" * 70)
print("EXTRAINDO INFORMAÇÕES DO PCE")
print("=" * 70)


pce_data = myPCE['PCE']

error_data = myPCE['Error']


# Listas para armazenar os resultados
list_degrees = []
list_num_coeffs = []
list_loo = []


# ------------------------------------------------------------
# Número de outputs
# ------------------------------------------------------------

# [PCA-PCE] Um PCE por componente principal
n_outputs = n_pc


# ============================================================
# 10. GRAU, COEFICIENTES E LOO
# ============================================================

for i in range(n_outputs):

    # --------------------------------------------------------
    # COEFICIENTES E BASE
    # --------------------------------------------------------

    if isinstance(pce_data, list):

        model_step = pce_data[i]

        coeffs = np.asarray(
            model_step['Coefficients']
        ).flatten()

        basis_inds = np.asarray(
            model_step['Basis']['Indices']
        )

    else:

        coeffs = np.asarray(
            pce_data['Coefficients']
        )

        # Caso os coeficientes estejam organizados
        # como [termos x outputs]
        if coeffs.ndim > 1:
            coeffs = coeffs[:, i]

        coeffs = coeffs.flatten()

        basis_inds = np.asarray(
            pce_data['Basis']['Indices']
        )


    # --------------------------------------------------------
    # NÚMERO DE COEFICIENTES NÃO-ZERO
    # --------------------------------------------------------

    tolerance = 1e-15

    n_coeffs = np.sum(
        np.abs(coeffs) > tolerance
    )

    list_num_coeffs.append(
        n_coeffs
    )


    # --------------------------------------------------------
    # GRAU MÁXIMO DO PCE
    # --------------------------------------------------------

    if basis_inds.ndim == 1:

        basis_inds = basis_inds.reshape(
            -1,
            1
        )

    # Soma dos índices polinomiais de cada termo
    total_degree_each_term = np.sum(
        basis_inds,
        axis=1
    )

    degree_selected = np.max(
        total_degree_each_term
    )

    list_degrees.append(
        degree_selected
    )


    # --------------------------------------------------------
    # ERRO LOO
    # --------------------------------------------------------

    if isinstance(error_data, list):

        val_loo = error_data[i]['LOO']

        if isinstance(val_loo, list):

            # Pode ser [[valor]]
            val_loo = np.asarray(
                val_loo
            ).flatten()[0]

        else:

            val_loo = float(
                val_loo
            )

    elif isinstance(error_data, dict):

        val_loo = error_data['LOO'][i]

    else:

        val_loo = np.nan

    list_loo.append(
        float(val_loo)
    )


# Converter para arrays
arr_degrees = np.asarray(
    list_degrees
)

arr_num_coeffs = np.asarray(
    list_num_coeffs
)

arr_loo = np.asarray(
    list_loo
).flatten()


# ============================================================
# 11. MÉTRICAS DE VALIDAÇÃO
# ============================================================

print("\n" + "=" * 70)
print("CALCULANDO MÉTRICAS DE VALIDAÇÃO")
print("=" * 70)


# ------------------------------------------------------------
# Resíduos
# ------------------------------------------------------------

residuals = (
    Y_val_true
    - Y_val_pred
)


# ------------------------------------------------------------
# MSE
# ------------------------------------------------------------

mse_val = np.mean(
    residuals**2,
    axis=0
)


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

rmse_val = np.sqrt(
    mse_val
)


# ------------------------------------------------------------
# NRMSE
#
# NRMSE = RMSE / (max(Y) - min(Y))
# ------------------------------------------------------------

val_range = (
    np.max(
        Y_val_true,
        axis=0
    )
    -
    np.min(
        Y_val_true,
        axis=0
    )
)

# Evitar divisão por zero
val_range = np.where(
    val_range == 0,
    1e-12,
    val_range
)

nrmse_val = (
    rmse_val
    / val_range
)


# ------------------------------------------------------------
# R²
# ------------------------------------------------------------

ss_res = np.sum(
    residuals**2,
    axis=0
)

ss_tot = np.sum(
    (
        Y_val_true
        -
        np.mean(
            Y_val_true,
            axis=0
        )
    )**2,
    axis=0
)

r2_val = np.zeros_like(
    rmse_val
)

mask_valid = (
    ss_tot > 1e-15
)

r2_val[mask_valid] = (
    1
    -
    (
        ss_res[mask_valid]
        /
        ss_tot[mask_valid]
    )
)


# ============================================================
# 12. VERIFICAÇÃO
# ============================================================

print("\nVerificação das dimensões:")

print(
    f"Steps       : {len(steps_interest)}"
)

print(
    f"Componentes : {n_pc}"
)

print(
    f"Graus       : {len(arr_degrees)} (por componente)"
)

print(
    f"Coeficientes: {len(arr_num_coeffs)}"
)

print(
    f"LOO         : {len(arr_loo)}"
)

print(
    f"RMSE        : {len(rmse_val)}"
)

print(
    f"NRMSE       : {len(nrmse_val)}"
)

print(
    f"R²          : {len(r2_val)}"
)


# ============================================================
# 13. SALVAR MÉTRICAS EM CSV
# ============================================================

# [PCA-PCE] Grau, coeficientes e LOO são definidos por componente,
# e as métricas de validação por instante de tempo: dois arquivos.
metrics_data = np.column_stack(
    (
        steps_interest,
        rmse_val,
        nrmse_val,
        r2_val
    )
)


filename_metrics = (
    'metrics_pressure_PCAPCE_detailed.csv'
)


header_txt = (
    'TimeStep,'
    'RMSE_Validation,'
    'NRMSE_Validation,'
    'R2_Validation'
)


np.savetxt(
    filename_metrics,
    metrics_data,
    delimiter=',',
    header=header_txt,
    comments='',
    fmt=[
        '%d',
        '%.6e',
        '%.6e',
        '%.6f'
    ]
)

np.savetxt(
    'metrics_pressure_PCAPCE_components.csv',
    np.column_stack(
        (
            np.arange(1, n_pc + 1),
            pca.explained_variance_ratio_,
            arr_degrees,
            arr_num_coeffs,
            arr_loo
        )
    ),
    delimiter=',',
    header='Component,Explained_Variance,Degree,Num_Coeffs,LOO_Error',
    comments='',
    fmt=['%d', '%.6e', '%d', '%d', '%.6e']
)


print(
    f"\n[SUCESSO] Métricas salvas em:"
    f"\n{filename_metrics}"
    f"\nmetrics_pressure_PCAPCE_components.csv"
)


# ============================================================
# 14. RESUMO DAS MÉTRICAS
# ============================================================

print("\n" + "=" * 70)
print("RESUMO DAS MÉTRICAS")
print("=" * 70)

print(
    f"Componentes (PCEs)      : "
    f"{n_pc}"
)

print(
    f"Coeficientes totais     : "
    f"{np.sum(arr_num_coeffs)}"
)

print(
    f"Grau médio              : "
    f"{np.mean(arr_degrees):.2f}"
)

print(
    f"Coeficientes médios     : "
    f"{np.mean(arr_num_coeffs):.2f}"
)

print(
    f"Erro LOO médio          : "
    f"{np.mean(arr_loo):.6e}"
)

print(
    f"RMSE médio              : "
    f"{np.mean(rmse_val):.6e}"
)

print(
    f"NRMSE médio             : "
    f"{np.mean(nrmse_val):.6e}"
)

print(
    f"R² médio                : "
    f"{np.mean(r2_val):.6f}"
)

print("=" * 70)


# [PCA-PCE] Tabela por componente
print("\n" + "=" * 70)
print(
    f"{'PC':>4} "
    f"{'Var. expl.':>12} "
    f"{'Degree':>8} "
    f"{'NCoef':>8} "
    f"{'LOO':>14}"
)
print("=" * 70)

for k in range(n_pc):

    print(
        f"{k + 1:4d} "
        f"{pca.explained_variance_ratio_[k]:12.4e} "
        f"{arr_degrees[k]:8d} "
        f"{arr_num_coeffs[k]:8d} "
        f"{arr_loo[k]:14.4e}"
    )

print("=" * 70)


# Tabela por instante de tempo (mesmos 90 instantes do script original)
print("\n" + "=" * 60)
print(
    f"{'Step':>6} "
    f"{'RMSE':>14} "
    f"{'NRMSE':>14} "
    f"{'R²':>10}"
)
print("=" * 60)

for i, step in enumerate(steps_interest):

    print(
        f"{step:6d} "
        f"{rmse_val[i]:14.4e} "
        f"{nrmse_val[i]:14.4e} "
        f"{r2_val[i]:10.6f}"
    )

print("=" * 60)


# ============================================================
# 16. GRÁFICOS DE VALIDAÇÃO
# ============================================================

output_folder = (
    "graficos_validacao_PCAPCE_pressure"
)

if not os.path.exists(
    output_folder
):

    os.makedirs(
        output_folder
    )

    print(
        f"\nPasta '{output_folder}' criada."
    )


print(
    f"\nIniciando geração de "
    f"{len(steps_interest)} gráficos..."
)


for i, step in enumerate(
    steps_interest
):

    # Dados do instante
    real = Y_val_true[:, i]

    pred = Y_val_pred[:, i]


    # --------------------------------------------------------
    # Limites
    # --------------------------------------------------------

    local_min = min(
        np.min(real),
        np.min(pred)
    )

    local_max = max(
        np.max(real),
        np.max(pred)
    )

    margin = (
        local_max - local_min
    ) * 0.05

    if margin == 0:

        margin = 0.01

    lims = [
        local_min - margin,
        local_max + margin
    ]


    # --------------------------------------------------------
    # Figura
    # --------------------------------------------------------

    plt.figure(
        figsize=(6, 6)
    )


    # --------------------------------------------------------
    # Scatter
    # --------------------------------------------------------

    plt.scatter(
        real,
        pred,
        c='blue',
        alpha=0.6,
        edgecolors='k',
        zorder=5,
        label='Amostras'
    )


    # --------------------------------------------------------
    # Linha ideal
    # --------------------------------------------------------

    plt.plot(
        lims,
        lims,
        'r--',
        linewidth=2,
        label='Ideal',
        zorder=1
    )


    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

    metrics_text = (
        f'RMSE = {rmse_val[i]:.2e}\n'
        f'NRMSE = {nrmse_val[i]:.2e}\n'
        f'R² = {r2_val[i]:.4f}\n'
        f'PCA-PCE ({n_pc} PCs)'
    )


    plt.text(
        0.05,
        0.93,
        metrics_text,
        transform=plt.gca().transAxes,
        bbox=dict(
            facecolor='white',
            alpha=0.8,
            edgecolor='gray'
        )
    )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    plt.title(
        f'Validação: Passo {step}',
        fontweight='bold'
    )

    plt.xlabel(
        'Simulação Real'
    )

    plt.ylabel(
        'Predição PCA-PCE'
    )


    # --------------------------------------------------------
    # Eixos
    # --------------------------------------------------------

    plt.axis(
        'equal'
    )

    plt.xlim(
        lims
    )

    plt.ylim(
        lims
    )

    plt.grid(
        True,
        linestyle='--',
        alpha=0.5
    )

    plt.legend(
        loc='lower right'
    )


    # --------------------------------------------------------
    # Salvar
    # --------------------------------------------------------

    filename = (
        f"{output_folder}/"
        f"validacao_step_{step:04d}.png"
    )

    plt.savefig(
        filename,
        dpi=100,
        bbox_inches='tight'
    )

    plt.close()


    if (i + 1) % 10 == 0:

        print(
            f"Gerado "
            f"{i + 1}/{len(steps_interest)}..."
        )


print(
    f"\n[SUCESSO] Gráficos salvos em:"
    f"\n{output_folder}/"
)


# ============================================================
# 17. SALVAR O MODELO
# ============================================================

print("\n" + "=" * 70)
print("SALVANDO MODELO PCE")
print("=" * 70)

nome_arquivo_modelo = (
    'pressure_pcapce_emulator.pkl'
)


with open(
    nome_arquivo_modelo,
    'wb'
) as f:

    # [PCA-PCE] O emulador precisa do PCE e da PCA
    pickle.dump(
        {
            'pce': myPCE,
            'pca': pca,
            'indices_python': indices_python
        },
        f
    )


print(
    f"[SUCESSO] Emulador salvo em:"
    f"\n{nome_arquivo_modelo}"
)

# ============================================================
# TESTE DIRETO DA INFLUÊNCIA DO C_fib
# ============================================================

N_test = 200

C_fib_test = np.linspace(
    1300,
    6500,
    N_test
)

X_test = np.zeros((N_test, 6))

# Cinco parâmetros fixos
X_test[:, 0] = 650.0
X_test[:, 1] = 6.625
X_test[:, 2] = 3.650
X_test[:, 3] = 2.650
X_test[:, 4] = 75000

# Somente C_fib varia
X_test[:, 5] = C_fib_test


# Avaliar PCE
Y_test = predict_pressure(
    X_test
)

Y_test = np.asarray(Y_test)

print("Shape:", Y_test.shape)

plt.figure(figsize=(7, 4))

plt.plot(
    C_fib_test,
    Y_test[:, -1],
    linewidth=2
)

plt.xlabel('C_fib')
plt.ylabel('Pressure (Pa)')
plt.title('Influência de C_fib no último passo')

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    'PCAPCE_Cfib_ultimo_passo.png',
    dpi=300
)

plt.show()



# ============================================================
# 18. FINALIZAR SESSÃO
# ============================================================

# ============================================================
# 18. MONTE CARLO: SÓ C_fib VARIA (5 PARÂMETROS FIXOS)
# ------------------------------------------------------------
# Cole este bloco ANTES de "mySession.quit()", usando o mesmo
# `uq`, `myPCE`, `parameter_names`, `parameter_limits` e
# `steps_interest` do script de treinamento.
# ============================================================

print("\n" + "=" * 70)
print("MONTE CARLO EM C_fib")
print("=" * 70)

N_MC = 100_000
CHUNK = 10_000          # avalia em lotes para não estourar a requisição ao UQCloud
SEED = 42

# Valores fixos dos 5 primeiros parâmetros (ajuste se quiser).
# Aqui uso o ponto médio de cada intervalo.
fixed_values = {
    'C':     650.0,
    'b_f':   6.625,
    'b_t':   3.650,
    'b_fs':  2.650,
    'T_ref': 75000.0,
}

idx_cfib = parameter_names.index('C_fib')

# Checagem: valores fixos dentro dos limites usados no treino
for name, val in fixed_values.items():
    lo, hi = parameter_limits[parameter_names.index(name)]
    if not (lo <= val <= hi):
        raise ValueError(
            f"{name}={val} fora do intervalo de treino [{lo}, {hi}]"
        )

# ------------------------------------------------------------
# Amostragem MC de C_fib com UQpylab (Uniforme, igual ao treino)
# ------------------------------------------------------------
uq.rng(SEED, 'twister')

CfibInput = uq.createInput({
    'Marginals': [{
        'Name': 'C_fib',
        'Type': 'Uniform',
        'Parameters': parameter_limits[idx_cfib]
    }]
})

C_fib_mc = np.asarray(
    uq.getSample(CfibInput, N_MC, 'MC')
).reshape(-1)

# ------------------------------------------------------------
# Monta a matriz completa (N_MC x 6) na ordem do treino
# ------------------------------------------------------------
X_mc = np.empty((N_MC, len(parameter_names)))

for j, name in enumerate(parameter_names):
    if name == 'C_fib':
        X_mc[:, j] = C_fib_mc
    else:
        X_mc[:, j] = fixed_values[name]

print(f"X_mc: {X_mc.shape}")
print(f"C_fib min/max amostrado: {C_fib_mc.min():.1f} / {C_fib_mc.max():.1f}")

# ------------------------------------------------------------
# Avalia o PCE (90 saídas) em lotes
# ------------------------------------------------------------
Y_chunks = []

for start in range(0, N_MC, CHUNK):
    stop = min(start + CHUNK, N_MC)
    Y_chunk = predict_pressure(X_mc[start:stop])  # [PCA-PCE]
    Y_chunks.append(Y_chunk)
    print(f"Avaliadas {stop}/{N_MC} amostras")

Y_mc = np.vstack(Y_chunks)          # (N_MC, 90)
print(f"Y_mc: {Y_mc.shape}")

# ------------------------------------------------------------
# Salva resultados
# ------------------------------------------------------------
np.save('mc_Cfib_X_pcapce.npy', X_mc)
np.save('mc_Cfib_Y_pressure_pcapce.npy', Y_mc)        # ~72 MB em float64

# Estatísticas por instante de tempo
stats = np.column_stack((
    steps_interest,
    Y_mc.mean(axis=0),
    Y_mc.std(axis=0, ddof=1),
    np.percentile(Y_mc, 2.5, axis=0),
    np.percentile(Y_mc, 50, axis=0),
    np.percentile(Y_mc, 97.5, axis=0),
))

np.savetxt(
    'mc_Cfib_stats_pcapce.csv',
    stats,
    delimiter=',',
    header='TimeStep,Mean,Std,P2.5,P50,P97.5',
    comments='',
    fmt=['%d', '%.6e', '%.6e', '%.6e', '%.6e', '%.6e']
)

# ------------------------------------------------------------
# Gráfico: média e banda de 95 % ao longo do tempo
# ------------------------------------------------------------
plt.figure(figsize=(8, 4.5))
plt.fill_between(steps_interest, stats[:, 3], stats[:, 5],
                 alpha=0.3, label='Intervalo 95 %')
plt.plot(steps_interest, stats[:, 1], linewidth=2, label='Média')
plt.xlabel('Passo de tempo')
plt.ylabel('Pressure (Pa)')
plt.title(f'Monte Carlo (N={N_MC}) variando apenas C_fib (PCA-PCE)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('mc_Cfib_bandas_pcapce.png', dpi=300)
plt.close()

print("[SUCESSO] Monte Carlo concluído e salvo.")

mySession.quit()

print("\n" + "=" * 70)
print("PROCESSO FINALIZADO")
print("=" * 70)
