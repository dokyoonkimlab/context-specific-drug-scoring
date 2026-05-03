# -*- coding: utf-8 -*-
"""
1_build_DDN.py
--------------
Constructs the Disease-Disease Network (DDN) from UK Biobank PheWAS summary statistics.

Steps:
    1. Load PheWAS summary statistics (pickle format) for each disease.
    2. Filter diseases by case count (n_cases > 500) and SNP p-value (< 1e-5).
    3. Apply LD-pruned SNP list to retain independent SNPs.
    4. Build binary disease-SNP matrix.
    5. Save outputs for downstream use.

Inputs:
    - UKB_Summary_Data_info_n500.csv     : disease list with case counts
    - ./UKB_Summary_Data/<phecode>       : per-disease PheWAS summary (pickle)
    - LD_pruned_snp_list.prune.in        : LD-pruned SNP list
                                           (LD pruning: window=50kb, step=5kb, r2=0.5)

Outputs:
    - final_used_disease_snp.pkl         : {final_disease, final_snp}
    - final_disease_pruned_snp_matrix.pkl: binary disease x SNP matrix (m x p)
    - Final_Union_Set_of_Pruned_SNPs.csv : union set of pruned SNPs
"""

import os
import pickle
import numpy as np
import pandas as pd

# ── User settings ─────────────────────────────────────────────────────────────
DATA_DIR        = './UKB_Summary_Data'
DISEASE_INFO    = 'UKB_Summary_Data_info_n500.csv'
PRUNED_SNP_FILE = './Data_Bulk/LD_pruned_snp_list.prune.in'
PVAL_THRESHOLD  = 1e-5
# ──────────────────────────────────────────────────────────────────────────────


def load_disease_snp(disease_list, data_dir, pval_th):
    """
    Load per-disease PheWAS summary statistics and filter by p-value threshold.

    Parameters
    ----------
    disease_list : array-like
        List of disease phecode identifiers.
    data_dir : str
        Directory containing per-disease summary statistic files (pickle format).
    pval_th : float
        P-value threshold for genome-wide significance.

    Returns
    -------
    used_disease : list of str
        Diseases passing the p-value filter.
    used_snp : list of np.ndarray
        Significant SNP lists for each disease.
    """
    used_disease, used_snp = [], []
    for disease_id in disease_list:
        filepath = os.path.join(data_dir, disease_id)
        if not os.path.exists(filepath):
            print(f"[SKIP] File not found: {filepath}")
            continue
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            sub_data = data[data['P'] <= pval_th]
            if sub_data.empty:
                continue
            used_disease.append(disease_id)
            used_snp.append(sub_data['SNP'].to_numpy())
        except Exception as e:
            print(f"[ERROR] {filepath}: {e}")
    return used_disease, used_snp


def apply_ld_pruning(used_snp, pruned_snp_file):
    """
    Retain only LD-pruned SNPs from each disease's SNP list.

    Parameters
    ----------
    used_snp : list of np.ndarray
        Raw significant SNP lists per disease.
    pruned_snp_file : str
        Path to file containing LD-pruned SNP IDs (one per line).

    Returns
    -------
    pruned_used_snp : list of np.ndarray
        LD-pruned SNP lists per disease.
    pruned_list : np.ndarray
        Full LD-pruned SNP reference list.
    """
    pruned_list = pd.read_csv(pruned_snp_file, header=None).to_numpy()
    pruned_used_snp = []
    for snp_array in used_snp:
        idx = np.where(np.isin(pruned_list, snp_array))
        pruned_used_snp.append(pruned_list[idx])
    return pruned_used_snp, pruned_list


def build_disease_snp_matrix(disease_list, snp_lists, union_snp_set):
    """
    Build a binary disease x SNP indicator matrix.

    Parameters
    ----------
    disease_list : list of str
        Disease identifiers (rows).
    snp_lists : list of np.ndarray
        SNP lists per disease.
    union_snp_set : np.ndarray
        Union set of all SNPs (columns), shape (p, 1).

    Returns
    -------
    matrix : np.ndarray, shape (m, p)
        Binary matrix; 1 if disease i has SNP j, else 0.
    """
    m = len(disease_list)
    p = len(union_snp_set)
    matrix = np.zeros((m, p), dtype=np.uint8)
    for i, snp_array in enumerate(snp_lists):
        snp_array = snp_array.reshape(-1, 1)
        idx = np.where(np.isin(union_snp_set, snp_array))[0]
        matrix[i, idx] = 1
        print(f"[{i+1}/{m}] {disease_list[i]}: {int(matrix[i].sum())} SNPs")
    return matrix


def main():
    # 1. Load disease list
    disease_info = pd.read_csv(DISEASE_INFO)
    disease_list = disease_info['Phe_'].to_numpy()
    print(f"Total diseases to process: {len(disease_list)}")

    # 2. Load summary stats and filter by p-value
    used_disease, used_snp = load_disease_snp(disease_list, DATA_DIR, PVAL_THRESHOLD)
    print(f"Diseases passing p-value filter (< {PVAL_THRESHOLD}): {len(used_disease)}")

    # 3. Apply LD pruning
    pruned_used_snp, _ = apply_ld_pruning(used_snp, PRUNED_SNP_FILE)

    # 4. Remove diseases with zero pruned SNPs
    final_disease, final_snp = [], []
    for disease_id, snp_array in zip(used_disease, pruned_used_snp):
        if snp_array.size > 0:
            final_disease.append(disease_id)
            final_snp.append(snp_array)
    print(f"Diseases after removing zero-SNP cases: {len(final_disease)}")

    # 5. Build union SNP set
    union_snp = final_snp[0].copy()
    for snp_array in final_snp[1:]:
        union_snp = np.union1d(union_snp, snp_array)
    union_snp = union_snp.reshape(-1, 1)
    print(f"Union pruned SNP set size: {len(union_snp)}")

    # 6. Build binary disease-SNP matrix
    matrix = build_disease_snp_matrix(final_disease, final_snp, union_snp)

    # 7. Save outputs
    with open('final_used_disease_snp.pkl', 'wb') as f:
        pickle.dump({'final_disease': final_disease, 'final_snp': final_snp}, f)

    with open('final_disease_pruned_snp_matrix.pkl', 'wb') as f:
        pickle.dump(matrix, f)

    pd.DataFrame(union_snp, columns=['SNP']).to_csv('Final_Union_Set_of_Pruned_SNPs.csv')

    print("\n[DONE] Outputs saved:")
    print("  - final_used_disease_snp.pkl")
    print("  - final_disease_pruned_snp_matrix.pkl")
    print(f"  - Final matrix shape: {matrix.shape}  (diseases x SNPs)")


if __name__ == '__main__':
    main()
