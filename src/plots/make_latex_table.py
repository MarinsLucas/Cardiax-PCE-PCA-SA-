"""
Gera a tabela LaTeX de métricas de validação dos emuladores PCA-PCE por região AHA.

Estrutura esperada (gerada por src/train/aha_markers_pcapce.py):
    <ROOT>/AHA01/metrics_detailed.csv     (NRMSE e R2 por instante de tempo)
    <ROOT>/AHA01/metrics_components.csv   (LOO por componente principal)
    ...
    <ROOT>/AHA17/...

NRMSE e R2 são médias no tempo; "Mean LOO" é a média do erro LOO dos PCEs
das componentes principais.

Uso:
    python src/plots/make_latex_table.py results/aha/strain --output results/tabela_strain.tex
    python src/plots/make_latex_table.py results/aha/stress --quantity "regional fiber stress" \
        --range-name stress --label tab:pce_stress --output results/tabela_stress.tex
"""
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AHA_NAMES = {
    1: 'Basal Anterior',
    2: 'Basal Anteroseptal',
    3: 'Basal Inferoseptal',
    4: 'Basal Inferior',
    5: 'Basal Inferolateral',
    6: 'Basal Anterolateral',
    7: 'Mid Anterior',
    8: 'Mid Anteroseptal',
    9: 'Mid Inferoseptal',
    10: 'Mid Inferior',
    11: 'Mid Inferolateral',
    12: 'Mid Anterolateral',
    13: 'Apical Anterior',
    14: 'Apical Septal',
    15: 'Apical Inferior',
    16: 'Apical Lateral',
    17: 'Apex',
}

# Colunas do CSV de métricas
COL_STEP = 'TimeStep'
COL_NRMSE = 'NRMSE_Validation'
COL_R2 = 'R2_Validation'
COL_LOO = 'LOO_Error'


# ------------------------------------------------------------
# Formatação
# ------------------------------------------------------------
def sci(x):
    """2.53e-04 -> $2.53 \\times 10^{-4}$"""
    if not np.isfinite(x) or x == 0:
        return f"${x:g}$"
    mant, exp = f"{x:.2e}".split('e')
    return rf"${mant} \times 10^{{{int(exp)}}}$"


def fixed(x):
    return f"{x:.4f}"


# ------------------------------------------------------------
# Leitura
# ------------------------------------------------------------
def find_aha_dirs(root):
    found = {}
    for p in Path(root).iterdir():
        if p.is_dir():
            m = re.fullmatch(r'AHA[_\s-]*0*(\d+)', p.name, re.IGNORECASE)
            if m:
                found[int(m.group(1))] = p
    return dict(sorted(found.items()))


def find_metrics_csv(aha_dir, pattern):
    files = sorted(aha_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo '{pattern}' em {aha_dir}")
    if len(files) > 1:
        print(f"[AVISO] {len(files)} arquivos casam com '{pattern}' em {aha_dir}; "
              f"usando {files[0].name}")
    return files[0]


def summarize(csv_path, exclude_steps):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # PCA-PCE: o LOO não é por instante, e sim por componente (metrics_components.csv)
    loo_override = None
    if COL_LOO not in df.columns:
        comp_csv = csv_path.parent / 'metrics_components.csv'
        if not comp_csv.exists():
            raise ValueError(f"{csv_path}: sem coluna '{COL_LOO}' e sem {comp_csv.name}")
        loo_override = pd.read_csv(comp_csv)[COL_LOO].to_numpy(dtype=float)
        df[COL_LOO] = np.nan

    missing = [c for c in (COL_NRMSE, COL_R2, COL_LOO) if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: colunas ausentes {missing}")

    if exclude_steps:
        if COL_STEP not in df.columns:
            raise ValueError(f"{csv_path}: sem coluna '{COL_STEP}' para excluir passos")
        df = df[~df[COL_STEP].isin(exclude_steps)]

    if df[[COL_NRMSE, COL_R2]].isna().any().any():
        print(f"[AVISO] NaN em {csv_path}; usando médias que ignoram NaN.")

    nrmse = df[COL_NRMSE].to_numpy(dtype=float)
    r2 = df[COL_R2].to_numpy(dtype=float)
    loo = loo_override if loo_override is not None else df[COL_LOO].to_numpy(dtype=float)

    # Avisa se um único passo domina as estatísticas
    if np.nanmax(nrmse) > 10 * np.nanmedian(nrmse):
        i_max = int(np.nanargmax(nrmse))
        step_txt = int(df[COL_STEP].iloc[i_max]) if COL_STEP in df.columns else i_max
        print(f"[AVISO] {csv_path.parent.name}: NRMSE máximo ({np.nanmax(nrmse):.2e}) "
              f"é >10x a mediana ({np.nanmedian(nrmse):.2e}), no passo {step_txt}.")

    return {
        'mean_nrmse': np.nanmean(nrmse),
        'max_nrmse': np.nanmax(nrmse),
        'mean_r2': np.nanmean(r2),
        'mean_loo': np.nanmean(loo),
        'n_steps': len(df),
    }


# ------------------------------------------------------------
# Montagem da tabela
# ------------------------------------------------------------
def build_table(rows, quantity, range_name, label):
    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Validation metrics for {" + quantity + r"} emulators across the 17 AHA "
        r"segments. Values represent temporal statistics, where NRMSE is normalized by the "
        + range_name + r" range.}"
    )
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{@{}lcccc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{AHA Region} & \textbf{Mean NRMSE} & \textbf{Max NRMSE} & "
        r"\textbf{Mean $R^2$} & \textbf{Mean LOO} \\ \midrule"
    )

    for k, (aha_id, s) in enumerate(rows):
        name = AHA_NAMES.get(aha_id, '')
        region = f"{aha_id} ({name})" if name else f"{aha_id}"
        end = r" \\ \bottomrule" if k == len(rows) - 1 else r" \\"
        lines.append(
            f"{region} & {sci(s['mean_nrmse'])} & {sci(s['max_nrmse'])} & "
            f"{fixed(s['mean_r2'])} & {sci(s['mean_loo'])}{end}"
        )

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help="pasta que contém as subpastas AHA{ID}")
    ap.add_argument('--pattern', default='metrics_detailed.csv',
                    help="padrão do CSV dentro de cada AHA (padrão: metrics_detailed.csv)")
    ap.add_argument('--exclude-steps', type=int, nargs='*', default=[],
                    help="TimeStep(s) a excluir das estatísticas (ex.: 1)")
    ap.add_argument('--quantity', default='regional fiber strain')
    ap.add_argument('--range-name', default='strain')
    ap.add_argument('--label', default='tab:pce_strain')
    ap.add_argument('--output', default='tabela_pce.tex')
    args = ap.parse_args()

    aha_dirs = find_aha_dirs(args.root)
    if not aha_dirs:
        raise FileNotFoundError(f"Nenhuma subpasta AHA{{ID}} em {args.root}")
    if len(aha_dirs) != 17:
        print(f"[AVISO] {len(aha_dirs)} regiões encontradas (esperado 17): {list(aha_dirs)}")

    rows = []
    for aha_id, d in aha_dirs.items():
        csv_path = find_metrics_csv(d, args.pattern)
        rows.append((aha_id, summarize(csv_path, set(args.exclude_steps))))
        print(f"AHA {aha_id:2d}: {csv_path.relative_to(args.root)}  "
              f"({rows[-1][1]['n_steps']} passos)")

    table = build_table(rows, args.quantity, args.range_name, args.label)

    Path(args.output).write_text(table, encoding='utf-8')
    print(f"\n[OK] Tabela salva em '{args.output}'.\n")
    print(table)


if __name__ == '__main__':
    main()