from uqpylab import sessions, display_util

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA  # [PCA-PCE]

from pathlib import Path
import pickle
import os
import csv


# ============================================================
# 0. CONFIGURAÇÃO GERAL
# ============================================================

# ------------------------------------------------------------
# Diretórios
# ------------------------------------------------------------

# Diretório onde estão:
#   uniform_samples.csv
#   strain_aha1.csv
#   strain_aha2.csv
#   ...
#   stress_aha1.csv
#   stress_aha2.csv
#
ROOT = Path(__file__).resolve().parents[2]   # raiz do repositório
DATA_DIR = ROOT / "data" / "aha"

# Pasta principal dos resultados
OUTPUT_ROOT = ROOT / "results" / "aha"


# ------------------------------------------------------------
# AHA markers
# ------------------------------------------------------------

AHA_IDS = range(1, 18)


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


parameter_limits = [
    [585, 715],              # C
    [5.958, 7.292],          # b_f
    [3.285, 4.015],          # b_t
    [2.385, 2.915],          # b_fs
    [75000 * 0.9, 75000 * 1.1],  # T_ref
    [1300, 6500]             # C_fib
]


# ------------------------------------------------------------
# PCE
# ------------------------------------------------------------

# 90 instantes:
# 1, 11, 21, ..., 891
steps_interest = [
    i for i in range(1, 901, 10)
]

# Os CSVs dos AHA têm 902 colunas:
#   coluna 0       -> estado inicial (t = 0), zero em todas as simulações
#   colunas 1..900 -> passos 1..900
#   coluna 901     -> coluna final, zero em todas as simulações
# Logo, o passo k está na coluna k.
# Se algum CSV começar direto no passo 1 (900 colunas), use 0.
FIRST_STEP_COLUMN = 1

indices_python = [
    FIRST_STEP_COLUMN + (i - 1) for i in steps_interest
]

N_OUTPUTS = len(steps_interest)

# [PCA-PCE] Curvas completas: passos 1..900. Índices dos 90 instantes
# de interesse dentro das curvas completas.
N_STEPS_FULL = 900
indices_in_full = [
    i - 1 for i in steps_interest
]

# [PCA-PCE] Fração da variância explicada mantida pela PCA
# (mesmo valor para todas as QoIs)
VAR_THRESHOLD = 0.99999


# ------------------------------------------------------------
# Validação
# ------------------------------------------------------------

# Equivalente ao seu código:
# test_size = 36*2 = 72
VALIDATION_SIZE = 72

RANDOM_STATE = 42


# ------------------------------------------------------------
# Monte Carlo
# ------------------------------------------------------------

N_MC = 100_000
MC_CHUNK = 10_000
MC_SEED = 42


# ------------------------------------------------------------
# Se True, salva X_mc e Y_mc.
#
# Isso gera arquivos grandes:
# Y_mc tem 100.000 x 90 valores.
#
# Para 34 PCEs isso pode ocupar vários GB.
# ------------------------------------------------------------

SAVE_RAW_MC = False


# ============================================================
# 1. FUNÇÕES AUXILIARES
# ============================================================

def print_separator(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def create_parameter_input(uq):
    """
    Cria o Input de 6 parâmetros (uniformes).
    """

    marginals = [
        {
            'Name': name,
            'Type': 'Uniform',
            'Parameters': limits
        }
        for name, limits in zip(parameter_names, parameter_limits)
    ]

    return uq.createInput({'Marginals': marginals})


def find_aha_file(prefix, aha_id):
    """
    Aceita strain_aha01.csv e strain_aha1.csv.
    """

    for candidate in (
        DATA_DIR / f"{prefix}_aha{aha_id:02d}.csv",
        DATA_DIR / f"{prefix}_aha{aha_id}.csv"
    ):
        if candidate.exists():
            return candidate

    return DATA_DIR / f"{prefix}_aha{aha_id:02d}.csv"


def load_input_data():
    """
    Carrega os 6 parâmetros de entrada.
    """

    filename = ROOT / "data" / "uniform_samples.csv"

    print(f"Carregando: {filename}")

    X_raw = np.loadtxt(
        filename,
        delimiter=",",
        skiprows=1,
        usecols=(1, 2, 3, 4, 5, 6)
    )

    if X_raw.shape[1] != 6:
        raise ValueError(
            f"Esperados 6 parâmetros, "
            f"mas X possui {X_raw.shape[1]} colunas."
        )

    print(f"X_raw: {X_raw.shape}")

    return X_raw


def load_output_data(filename):
    """
    Carrega um strain_ahaXX.csv ou stress_ahaXX.csv
    e seleciona os 90 instantes de interesse.
    """

    print(f"Carregando: {filename}")

    Y_full = np.loadtxt(
        filename,
        delimiter=","
    )

    if Y_full.ndim == 1:
        Y_full = Y_full.reshape(1, -1)

    print(f"Y completo: {Y_full.shape}")

    if Y_full.shape[1] <= max(indices_python):
        raise ValueError(
            f"{filename} possui somente "
            f"{Y_full.shape[1]} colunas, mas é necessário "
            f"o índice {max(indices_python)}."
        )

    Y_selected = Y_full[:, indices_python]

    print(
        f"Y selecionado: {Y_selected.shape}"
    )

    # Colunas sem variação entre simulações (ex.: t=0) não servem para o PCE
    n_const = int(np.sum(np.std(Y_selected, axis=0) < 1e-12))

    if n_const > 0:
        print(
            f"[AVISO] {n_const} das colunas selecionadas são constantes "
            f"entre as simulações. Verifique FIRST_STEP_COLUMN."
        )

    if Y_selected.shape[1] != N_OUTPUTS:
        raise ValueError(
            f"Esperados {N_OUTPUTS} outputs, "
            f"mas foram encontrados {Y_selected.shape[1]}."
        )

    # [PCA-PCE] Curvas completas (passos 1..900) para a PCA
    Y_steps = Y_full[
        :,
        FIRST_STEP_COLUMN:FIRST_STEP_COLUMN + N_STEPS_FULL
    ]

    print(f"Y curvas completas: {Y_steps.shape}")

    return Y_selected, Y_steps


def remove_invalid_samples(X, Y, Y_steps):
    """
    Remove simulações que possuem NaN/inf em qualquer
    instante utilizado.

    Isso é importante porque os CSVs dos AHA podem possuir
    linhas correspondentes a simulações ausentes.
    """

    valid_X = np.all(np.isfinite(X), axis=1)
    valid_Y = np.all(np.isfinite(Y), axis=1)

    # [PCA-PCE] A PCA usa a curva completa: NaN em qualquer passo invalida
    valid_Y = valid_Y & np.all(np.isfinite(Y_steps), axis=1)

    valid = valid_X & valid_Y

    removed = np.sum(~valid)

    print(
        f"Amostras válidas : {np.sum(valid)}"
    )

    print(
        f"Amostras removidas: {removed}"
    )

    return X[valid], Y[valid], Y_steps[valid], valid


def extract_pce_information(myPCE, n_outputs=N_OUTPUTS):
    """
    Extrai:
        - grau
        - número de coeficientes
        - LOO

    para cada output.
    """

    pce_data = myPCE['PCE']
    error_data = myPCE['Error']

    list_degrees = []
    list_num_coeffs = []
    list_loo = []

    for i in range(n_outputs):  # [PCA-PCE] um PCE por componente

        # ----------------------------------------------------
        # Coeficientes e base
        # ----------------------------------------------------

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

            if coeffs.ndim > 1:
                coeffs = coeffs[:, i]

            coeffs = coeffs.flatten()

            basis_inds = np.asarray(
                pce_data['Basis']['Indices']
            )

        # ----------------------------------------------------
        # Número de coeficientes não-zero
        # ----------------------------------------------------

        tolerance = 1e-15

        n_coeffs = np.sum(
            np.abs(coeffs) > tolerance
        )

        list_num_coeffs.append(
            n_coeffs
        )

        # ----------------------------------------------------
        # Grau máximo
        # ----------------------------------------------------

        if basis_inds.ndim == 1:
            basis_inds = basis_inds.reshape(-1, 1)

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

        # ----------------------------------------------------
        # LOO
        # ----------------------------------------------------

        if isinstance(error_data, list):

            val_loo = error_data[i]['LOO']

            if isinstance(val_loo, list):
                val_loo = np.asarray(
                    val_loo
                ).flatten()[0]

            else:
                val_loo = float(val_loo)

        elif isinstance(error_data, dict):

            val_loo = error_data['LOO'][i]

        else:

            val_loo = np.nan

        list_loo.append(
            float(val_loo)
        )

    return (
        np.asarray(list_degrees),
        np.asarray(list_num_coeffs),
        np.asarray(list_loo).flatten()
    )


def calculate_validation_metrics(Y_true, Y_pred):
    """
    Calcula:
        MSE
        RMSE
        NRMSE
        R²
    """

    residuals = Y_true - Y_pred

    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    mse = np.mean(
        residuals ** 2,
        axis=0
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(mse)

    # --------------------------------------------------------
    # NRMSE
    # --------------------------------------------------------

    value_range = (
        np.max(Y_true, axis=0)
        -
        np.min(Y_true, axis=0)
    )

    value_range = np.where(
        value_range == 0,
        1e-12,
        value_range
    )

    nrmse = rmse / value_range

    # --------------------------------------------------------
    # R²
    # --------------------------------------------------------

    ss_res = np.sum(
        residuals ** 2,
        axis=0
    )

    ss_tot = np.sum(
        (
            Y_true
            -
            np.mean(Y_true, axis=0)
        ) ** 2,
        axis=0
    )

    r2 = np.zeros_like(rmse)

    mask_valid = ss_tot > 1e-15

    r2[mask_valid] = (
        1
        -
        (
            ss_res[mask_valid]
            /
            ss_tot[mask_valid]
        )
    )

    return mse, rmse, nrmse, r2


def save_metrics(
    output_dir,
    degrees,
    num_coeffs,
    loo,
    mse,
    rmse,
    nrmse,
    r2,
    explained_variance
):
    """
    [PCA-PCE] Salva:
      - métricas de validação por instante (metrics_detailed.csv)
      - grau, coeficientes e LOO por componente (metrics_components.csv)
    """

    metrics = np.column_stack(
        (
            steps_interest,
            mse,
            rmse,
            nrmse,
            r2
        )
    )

    np.savetxt(
        output_dir / "metrics_detailed.csv",
        metrics,
        delimiter=",",
        header=(
            "TimeStep,"
            "MSE_Validation,"
            "RMSE_Validation,"
            "NRMSE_Validation,"
            "R2_Validation"
        ),
        comments="",
        fmt=[
            "%d",
            "%.6e",
            "%.6e",
            "%.6e",
            "%.8f"
        ]
    )

    np.savetxt(
        output_dir / "metrics_components.csv",
        np.column_stack(
            (
                np.arange(1, len(degrees) + 1),
                explained_variance,
                degrees,
                num_coeffs,
                loo
            )
        ),
        delimiter=",",
        header="Component,Explained_Variance,Degree,Num_Coeffs,LOO_Error",
        comments="",
        fmt=["%d", "%.6e", "%d", "%d", "%.6e"]
    )

    return metrics


def save_summary(
    output_dir,
    degrees,
    num_coeffs,
    loo,
    mse,
    rmse,
    nrmse,
    r2,
    n_valid
):
    """
    Salva um resumo geral daquele AHA.
    """

    filename = (
        output_dir /
        "summary.csv"
    )

    with open(
        filename,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Metric",
            "Mean",
            "Median",
            "Min",
            "Max"
        ])

        data = {
            "Degree": degrees,
            "Num_Coeffs": num_coeffs,
            "LOO": loo,
            "MSE": mse,
            "RMSE": rmse,
            "NRMSE": nrmse,
            "R2": r2
        }

        for name, values in data.items():

            writer.writerow([
                name,
                np.nanmean(values),
                np.nanmedian(values),
                np.nanmin(values),
                np.nanmax(values)
            ])

        writer.writerow([])
        writer.writerow([
            "N_valid_samples",
            n_valid
        ])

        # [PCA-PCE] Grau, coeficientes e LOO acima são por componente
        writer.writerow([
            "N_components",
            len(degrees)
        ])
        writer.writerow([
            "Total_Num_Coeffs",
            int(np.sum(num_coeffs))
        ])


def create_validation_plots(
    output_dir,
    Y_true,
    Y_pred,
    rmse,
    nrmse,
    r2,
    loo,
    quantity_name,
    aha_id
):
    """
    Cria um scatter plot para cada instante.
    """

    folder = (
        output_dir /
        "validation_plots"
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Gerando {N_OUTPUTS} gráficos "
        f"de validação..."
    )

    for i, step in enumerate(
        steps_interest
    ):

        real = Y_true[:, i]
        pred = Y_pred[:, i]

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

        plt.figure(
            figsize=(6, 6)
        )

        plt.scatter(
            real,
            pred,
            alpha=0.6,
            edgecolors='k',
            zorder=5,
            label='Amostras'
        )

        plt.plot(
            lims,
            lims,
            'r--',
            linewidth=2,
            label='Ideal',
            zorder=1
        )

        # [PCA-PCE] LOO é por componente, não por instante
        metrics_text = (
            f'RMSE = {rmse[i]:.2e}\n'
            f'NRMSE = {nrmse[i]:.2e}\n'
            f'R² = {r2[i]:.4f}\n'
            f'PCA-PCE ({len(loo)} PCs)'
        )

        plt.text(
            0.05,
            0.93,
            metrics_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='gray'
            )
        )

        plt.title(
            f'{quantity_name.capitalize()} - '
            f'AHA {aha_id} - Passo {step}',
            fontweight='bold'
        )

        plt.xlabel(
            f'{quantity_name.capitalize()} real'
        )

        plt.ylabel(
            'Predição PCA-PCE'
        )

        plt.axis(
            'equal'
        )

        plt.xlim(lims)
        plt.ylim(lims)

        plt.grid(
            True,
            linestyle='--',
            alpha=0.5
        )

        plt.legend(
            loc='lower right'
        )

        filename = (
            folder /
            f"validation_step_{step:04d}.png"
        )

        plt.savefig(
            filename,
            dpi=120,
            bbox_inches='tight'
        )

        plt.close()


def create_metrics_plot(
    output_dir,
    rmse,
    nrmse,
    r2,
    loo,
    quantity_name,
    aha_id
):
    """
    Gera um gráfico resumindo as principais métricas
    ao longo do tempo.
    """

    folder = (
        output_dir /
        "metrics_plots"
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 12)
    )

    axes[0].plot(
        steps_interest,
        rmse,
        linewidth=2
    )

    axes[0].set_ylabel(
        "RMSE"
    )

    axes[0].grid(
        True,
        alpha=0.3
    )

    axes[1].plot(
        steps_interest,
        nrmse,
        linewidth=2
    )

    axes[1].set_ylabel(
        "NRMSE"
    )

    axes[1].grid(
        True,
        alpha=0.3
    )

    axes[2].plot(
        steps_interest,
        r2,
        linewidth=2
    )

    axes[2].set_ylabel(
        "R²"
    )

    axes[2].grid(
        True,
        alpha=0.3
    )

    axes[2].set_xlabel(
        "Time step"
    )

    # [PCA-PCE] LOO por componente principal
    axes[3].semilogy(
        np.arange(1, len(loo) + 1),
        loo,
        'o-',
        linewidth=2
    )

    axes[3].set_ylabel(
        "LOO"
    )

    axes[3].set_xlabel(
        "PCA component"
    )

    axes[3].set_xticks(
        np.arange(1, len(loo) + 1)
    )

    axes[3].grid(
        True,
        alpha=0.3
    )

    fig.suptitle(
        f'{quantity_name.capitalize()} - AHA {aha_id} - '
        f'Métricas do PCA-PCE',
        fontsize=14,
        fontweight='bold'
    )

    fig.tight_layout()

    filename = (
        folder /
        "metrics_vs_time.png"
    )

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches='tight'
    )

    plt.close(fig)


def run_cfib_monte_carlo(
    uq,
    myInput,
    predict_fn,  # [PCA-PCE] PCE -> scores -> curva reconstruída
    output_dir,
    quantity_name,
    aha_id
):
    """
    Monte Carlo variando somente C_fib.

    Os outros 5 parâmetros permanecem fixos no valor central.
    """

    print_separator(
        f"MONTE CARLO - C_fib - "
        f"{quantity_name} AHA {aha_id}"
    )

    folder = (
        output_dir /
        "Cfib"
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Valores fixos
    # --------------------------------------------------------

    fixed_values = {
        'C': 650.0,
        'b_f': 6.625,
        'b_t': 3.650,
        'b_fs': 2.650,
        'T_ref': 75000.0,
    }

    idx_cfib = parameter_names.index(
        'C_fib'
    )

    # --------------------------------------------------------
    # Verificar limites
    # --------------------------------------------------------

    for name, value in fixed_values.items():

        lo, hi = parameter_limits[
            parameter_names.index(name)
        ]

        if not (lo <= value <= hi):

            raise ValueError(
                f"{name}={value} fora do intervalo "
                f"[{lo}, {hi}]"
            )

    # --------------------------------------------------------
    # Amostragem
    # --------------------------------------------------------

    uq.rng(
        MC_SEED,
        'twister'
    )

    # Amostra os 6 parâmetros a partir do Input de 6 dimensões e
    # usa somente a coluna C_fib. NÃO cria outro Input: um Input de
    # 1 dimensão viraria o Input "atual" da sessão e quebraria os
    # próximos createModel.
    X_sample = np.asarray(
        uq.getSample(
            myInput,
            N_MC,
            'MC'
        )
    )

    C_fib_mc = X_sample[:, idx_cfib].reshape(-1)

    # --------------------------------------------------------
    # Montar X
    # --------------------------------------------------------

    X_mc = np.empty(
        (
            N_MC,
            len(parameter_names)
        )
    )

    for j, name in enumerate(
        parameter_names
    ):

        if name == 'C_fib':

            X_mc[:, j] = C_fib_mc

        else:

            X_mc[:, j] = fixed_values[name]

    print(
        f"X_mc: {X_mc.shape}"
    )

    # --------------------------------------------------------
    # Avaliação em blocos
    # --------------------------------------------------------

    Y_chunks = []

    for start in range(
        0,
        N_MC,
        MC_CHUNK
    ):

        stop = min(
            start + MC_CHUNK,
            N_MC
        )

        Y_chunk = predict_fn(
            X_mc[start:stop]
        )

        Y_chunks.append(
            Y_chunk
        )

        print(
            f"MC: {stop}/{N_MC}"
        )

    Y_mc = np.vstack(
        Y_chunks
    )

    print(
        f"Y_mc: {Y_mc.shape}"
    )

    # --------------------------------------------------------
    # Salvar dados brutos somente se solicitado
    # --------------------------------------------------------

    if SAVE_RAW_MC:

        np.save(
            folder / "mc_Cfib_X.npy",
            X_mc
        )

        np.save(
            folder / "mc_Cfib_Y.npy",
            Y_mc
        )

    # --------------------------------------------------------
    # Estatísticas
    # --------------------------------------------------------

    stats = np.column_stack(
        (
            steps_interest,
            Y_mc.mean(axis=0),
            Y_mc.std(
                axis=0,
                ddof=1
            ),
            np.percentile(
                Y_mc,
                2.5,
                axis=0
            ),
            np.percentile(
                Y_mc,
                50,
                axis=0
            ),
            np.percentile(
                Y_mc,
                97.5,
                axis=0
            )
        )
    )

    np.savetxt(
        folder /
        "mc_Cfib_stats.csv",
        stats,
        delimiter=",",
        header=(
            "TimeStep,"
            "Mean,"
            "Std,"
            "P2.5,"
            "P50,"
            "P97.5"
        ),
        comments="",
        fmt=[
            "%d",
            "%.8e",
            "%.8e",
            "%.8e",
            "%.8e",
            "%.8e"
        ]
    )

    # --------------------------------------------------------
    # Gráfico: média + intervalo 95%
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
    )

    plt.fill_between(
        steps_interest,
        stats[:, 3],
        stats[:, 5],
        alpha=0.3,
        label='Intervalo 95%'
    )

    plt.plot(
        steps_interest,
        stats[:, 1],
        linewidth=2,
        label='Média'
    )

    plt.xlabel(
        'Time step'
    )

    plt.ylabel(
        quantity_name.capitalize()
    )

    plt.title(
        f'Monte Carlo - {quantity_name.capitalize()} '
        f'- AHA {aha_id}\n'
        f'Variando somente C_fib'
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        folder /
        "mc_Cfib_bands.png",
        dpi=250
    )

    plt.close()

    # --------------------------------------------------------
    # Gráfico de influência no último instante
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        C_fib_mc,
        Y_mc[:, -1],
        '.',
        alpha=0.15
    )

    plt.xlabel(
        'C_fib'
    )

    plt.ylabel(
        quantity_name.capitalize()
    )

    plt.title(
        f'Influência de C_fib - '
        f'{quantity_name.capitalize()} '
        f'AHA {aha_id} - '
        f'último passo'
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        folder /
        "PCAPCE_Cfib_final_step.png",
        dpi=250
    )

    plt.close()


# ============================================================
# 2. PROCESSAR UM ARQUIVO
# ============================================================

def process_single_output(
    uq,
    myInput,
    X_raw,
    filename,
    quantity_name,
    aha_id
):
    """
    Treina e analisa um PCE para um único:
        strain_ahaXX.csv
    ou
        stress_ahaXX.csv
    """

    print_separator(
        f"PROCESSANDO {quantity_name.upper()} "
        f"- AHA {aha_id}"
    )

    # --------------------------------------------------------
    # Diretório
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_ROOT
        /
        quantity_name
        /
        f"AHA{aha_id:02d}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Carregar Y
    # --------------------------------------------------------

    Y_raw, Y_steps_raw = load_output_data(
        filename
    )

    # --------------------------------------------------------
    # Verificar número de amostras
    # --------------------------------------------------------

    if len(X_raw) != len(Y_raw):

        raise ValueError(
            f"Número de amostras incompatível:\n"
            f"X = {len(X_raw)}\n"
            f"Y = {len(Y_raw)}\n"
            f"Arquivo = {filename}"
        )

    # --------------------------------------------------------
    # Remover NaN
    # --------------------------------------------------------

    X, Y, Y_steps, valid_mask = remove_invalid_samples(
        X_raw,
        Y_raw,
        Y_steps_raw
    )

    n_valid = len(X)

    if n_valid <= VALIDATION_SIZE:

        raise ValueError(
            f"AHA {aha_id} / {quantity_name}: "
            f"somente {n_valid} amostras válidas. "
            f"São necessárias mais de "
            f"{VALIDATION_SIZE}."
        )

    # --------------------------------------------------------
    # Salvar quais simulações foram utilizadas
    # --------------------------------------------------------

    np.savetxt(
        output_dir /
        "valid_simulation_indices.csv",
        np.where(valid_mask)[0] + 1,
        delimiter=",",
        header="Simulation_ID",
        comments="",
        fmt="%d"
    )

    # --------------------------------------------------------
    # Train / validation
    # --------------------------------------------------------

    # [PCA-PCE] As curvas completas são separadas junto (mesmas linhas)
    X_train, X_val, Y_train, Y_val_true, Y_steps_train, Y_steps_val = train_test_split(
        X,
        Y,
        Y_steps,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    print(
        f"Treinamento : X={X_train.shape}, "
        f"Y={Y_train.shape}"
    )

    print(
        f"Validação   : X={X_val.shape}, "
        f"Y={Y_val_true.shape}"
    )

    # --------------------------------------------------------
    # [PCA-PCE] PCA nas curvas completas de treino
    # --------------------------------------------------------

    pca = PCA(
        n_components=VAR_THRESHOLD,
        svd_solver='full'
    ).fit(Y_steps_train)

    n_pc = pca.n_components_

    Z_train = pca.transform(Y_steps_train)

    Y_val_proj = pca.inverse_transform(
        pca.transform(Y_steps_val)
    )[:, indices_in_full]

    _, _, nrmse_pca, _ = calculate_validation_metrics(
        Y_val_true,
        Y_val_proj
    )

    print(f"PCA: {n_pc} componentes, variância explicada "
          f"{100 * pca.explained_variance_ratio_.sum():.6f}%")
    print(f"NRMSE médio só da PCA (validação): {np.mean(nrmse_pca):.6e}")

    # --------------------------------------------------------
    # Configuração PCE
    # --------------------------------------------------------

    # Garante que o Input 6D é o Input atual da sessão
    myInput = create_parameter_input(uq)

    MetaOpts = {

        'Type': 'Metamodel',

        'MetaType': 'PCE',

        'Method': 'LARS',

        'TruncOptions': {
            'qNorm': 0.75
        },

        'Degree': np.arange(
            2,
            10
        ).tolist(),

        'Input': myInput
    }

    MetaOpts['ExpDesign'] = {

        'X': X_train.tolist(),

        'Y': Z_train.tolist(),  # [PCA-PCE] scores das componentes

        'Sampling': 'User'
    }

    # --------------------------------------------------------
    # Treinar
    # --------------------------------------------------------

    print(
        "Treinando PCE..."
    )

    myPCE = uq.createModel(
        MetaOpts
    )

    print(
        "[SUCESSO] PCE treinado."
    )

    # --------------------------------------------------------
    # Validação
    # --------------------------------------------------------

    # [PCA-PCE] PCE -> scores -> curva completa -> 90 instantes
    def predict_fn(Xn):
        Z = np.asarray(
            uq.evalModel(myPCE, Xn)
        ).reshape(len(Xn), -1)
        return pca.inverse_transform(Z)[:, indices_in_full]

    Y_val_pred = predict_fn(
        X_val
    )

    print(
        f"Predição: {Y_val_pred.shape}"
    )

    # --------------------------------------------------------
    # Informações do PCE
    # --------------------------------------------------------

    degrees, num_coeffs, loo = (
        extract_pce_information(
            myPCE,
            n_outputs=n_pc
        )
    )

    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

    mse, rmse, nrmse, r2 = (
        calculate_validation_metrics(
            Y_val_true,
            Y_val_pred
        )
    )

    # --------------------------------------------------------
    # Salvar métricas
    # --------------------------------------------------------

    save_metrics(
        output_dir,
        degrees,
        num_coeffs,
        loo,
        mse,
        rmse,
        nrmse,
        r2,
        pca.explained_variance_ratio_
    )

    save_summary(
        output_dir,
        degrees,
        num_coeffs,
        loo,
        mse,
        rmse,
        nrmse,
        r2,
        n_valid
    )

    # --------------------------------------------------------
    # Gráficos de validação
    # --------------------------------------------------------

    create_validation_plots(
        output_dir,
        Y_val_true,
        Y_val_pred,
        rmse,
        nrmse,
        r2,
        loo,
        quantity_name,
        aha_id
    )

    # --------------------------------------------------------
    # Gráfico das métricas
    # --------------------------------------------------------

    create_metrics_plot(
        output_dir,
        rmse,
        nrmse,
        r2,
        loo,
        quantity_name,
        aha_id
    )

    # --------------------------------------------------------
    # Salvar PCE
    # --------------------------------------------------------

    model_filename = (
        output_dir
        /
        f"{quantity_name}_AHA{aha_id:02d}_PCAPCE.pkl"
    )

    with open(
        model_filename,
        'wb'
    ) as f:

        # [PCA-PCE] O emulador precisa do PCE e da PCA
        pickle.dump(
            {
                'pce': myPCE,
                'pca': pca,
                'indices_in_full': indices_in_full
            },
            f
        )

    print(
        f"Modelo salvo: {model_filename}"
    )

    # --------------------------------------------------------
    # Monte Carlo C_fib
    # --------------------------------------------------------

    run_cfib_monte_carlo(
        uq,
        myInput,
        predict_fn,
        output_dir,
        quantity_name,
        aha_id
    )

    # --------------------------------------------------------
    # Resumo
    # --------------------------------------------------------

    result = {

        'Quantity': quantity_name,

        'AHA': aha_id,

        'N_valid': n_valid,

        # [PCA-PCE]
        'N_components': n_pc,

        'Total_Num_Coeffs': int(np.sum(num_coeffs)),

        'Mean_NRMSE_PCA_only': np.mean(nrmse_pca),

        'Mean_Degree': np.mean(
            degrees
        ),

        'Mean_Num_Coeffs': np.mean(
            num_coeffs
        ),

        'Mean_LOO': np.mean(
            loo
        ),

        'Mean_MSE': np.mean(
            mse
        ),

        'Mean_RMSE': np.mean(
            rmse
        ),

        'Mean_NRMSE': np.mean(
            nrmse
        ),

        'Mean_R2': np.mean(
            r2
        ),

        'Median_NRMSE': np.median(
            nrmse
        ),

        'Min_R2': np.min(
            r2
        ),

        'Max_R2': np.max(
            r2
        )
    }

    print("\nResumo:")

    for key, value in result.items():

        print(
            f"{key:20s}: {value}"
        )

    return result


# ============================================================
# 3. SALVAR RESUMO GLOBAL
# ============================================================

def save_global_summary(
    results,
    quantity_name
):

    folder = (
        OUTPUT_ROOT /
        "global_summary"
    )

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        folder /
        f"{quantity_name}_summary_all_AHAs.csv"
    )

    if len(results) == 0:
        return

    fieldnames = list(
        results[0].keys()
    )

    with open(
        filename,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print(
        f"\nResumo global salvo em:\n"
        f"{filename}"
    )


# ============================================================
# 4. MAIN
# ============================================================

def main():

    print_separator(
        "PIPELINE PCA-PCE - STRAIN / STRESS - 17 AHAs"
    )

    # --------------------------------------------------------
    # Carregar X
    # --------------------------------------------------------

    X_raw = load_input_data()

    # --------------------------------------------------------
    # Conectar ao UQCloud
    # --------------------------------------------------------

    print_separator(
        "CONECTANDO AO UQCLOUD"
    )

    token = os.environ.get('UQCLOUD_TOKEN')

    if not token:

        raise RuntimeError(
            "A variável de ambiente UQCLOUD_TOKEN "
            "não está definida.\n\n"
            "Execute antes:\n"
            "export UQCLOUD_TOKEN='SEU_TOKEN_AQUI'"
        )

    UQCloud_instance = (
        "https://uqcloud.ethz.ch"
    )

    mySession = sessions.cloud(
        host=UQCloud_instance,
        token=token
    )

    uq = mySession.cli

    display_util.load_plt_defaults()

    # --------------------------------------------------------
    # Definir marginais
    # --------------------------------------------------------

    marginals = []

    for name, limits in zip(
        parameter_names,
        parameter_limits
    ):

        marginals.append({

            'Name': name,

            'Type': 'Uniform',

            'Parameters': limits
        })

    InputOpts = {
        'Marginals': marginals
    }

    myInput = uq.createInput(
        InputOpts
    )

    # --------------------------------------------------------
    # Resultados globais
    # --------------------------------------------------------

    strain_results = []
    stress_results = []

    # ========================================================
    # LOOP STRAIN
    # ========================================================

    print_separator(
        "PROCESSANDO STRAIN"
    )

    for aha_id in AHA_IDS:

        filename = find_aha_file("strain", aha_id)

        if not filename.exists():

            print(
                f"[AVISO] Arquivo não encontrado: "
                f"{filename}"
            )

            continue

        try:

            result = process_single_output(
                uq,
                myInput,
                X_raw,
                filename,
                "strain",
                aha_id
            )

            strain_results.append(
                result
            )

        except Exception as e:

            print(
                f"\n[ERRO] STRAIN AHA {aha_id}"
            )

            print(
                repr(e)
            )

            # Continua para o próximo AHA
            continue

    # ========================================================
    # LOOP STRESS
    # ========================================================

    print_separator(
        "PROCESSANDO STRESS"
    )

    for aha_id in AHA_IDS:

        filename = find_aha_file("stress", aha_id)

        if not filename.exists():

            print(
                f"[AVISO] Arquivo não encontrado: "
                f"{filename}"
            )

            continue

        try:

            result = process_single_output(
                uq,
                myInput,
                X_raw,
                filename,
                "stress",
                aha_id
            )

            stress_results.append(
                result
            )

        except Exception as e:

            print(
                f"\n[ERRO] STRESS AHA {aha_id}"
            )

            print(
                repr(e)
            )

            continue

    # ========================================================
    # RESUMOS
    # ========================================================

    print_separator(
        "SALVANDO RESUMOS GLOBAIS"
    )

    save_global_summary(
        strain_results,
        "strain"
    )

    save_global_summary(
        stress_results,
        "stress"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print_separator(
        "PROCESSO FINALIZADO"
    )

    print(
        f"Strain processados: "
        f"{len(strain_results)}/{len(AHA_IDS)}"
    )

    print(
        f"Stress processados: "
        f"{len(stress_results)}/{len(AHA_IDS)}"
    )

    print(
        f"\nResultados em:\n"
        f"{OUTPUT_ROOT.resolve()}"
    )

    mySession.quit()


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()