# -*- coding: utf-8 -*-
"""
4_comparison_benchmark.py
--------------------------
Benchmark experiment: direct integration of disease-protein-drug network.

Constructs a three-layered heterogeneous network:
    Disease (DDN) -- Gene/Protein (PPI) -- Drug (Dr)

with direct connections between the disease layer and the protein layer
via disease-gene association edges. Label propagation is then applied
directly from the index disease node to the drug layer.

Usage:
    python 4_comparison_benchmark.py --disease RA --mu 0.5

    # Hyperparameter search
    python 4_comparison_benchmark.py --disease RA --search_mu

Inputs:
    - final_disease_snp_gene.pkl         : output from 2_snp_to_gene_mapping.py
    - final_disease_pruned_snp_matrix.pkl: binary disease x SNP matrix
    - combined_ppi_net.pkl               : PPI adjacency matrix (n x n)
    - ./STRING_DB/used_full_protein_symbol.csv
    - drug_target_matrix.pkl             : drug-protein matrix (q x n)
    - ./drugbank/drugbank_human.csv
    - ./drug_ground_truth.csv

Outputs:
    - benchmark_scores_<disease>_mu<mu>.csv
    - benchmark_results_summary.csv
"""

import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics


# ── Disease code mapping ───────────────────────────────────────────────────────
DISEASE_CODE = {
    'RA': 'Phe_714.1',
    'AS': 'Phe_495',
    'MS': 'Phe_335',
}

DISEASE_TRUTH_COL = {
    'RA': 'Rheumatoid_Arthritis',
    'AS': 'Asthma',
    'MS': 'Multiple_sclerosis',
}

MU_SEARCH_SPACE = [0.001, 0.01, 0.1, 1.0, 0.5, 10.0, 100.0]

# ── User settings ─────────────────────────────────────────────────────────────
DISEASE_SNP_GENE_PKL = 'final_disease_snp_gene.pkl'
DISEASE_SNP_MATRIX   = 'final_disease_pruned_snp_matrix.pkl'
PPI_NET_PKL          = 'combined_ppi_net.pkl'
PROTEIN_SYMBOL_CSV   = './STRING_DB/used_full_protein_symbol.csv'
DRUG_TARGET_PKL      = 'drug_target_matrix.pkl'
DRUG_INFO_CSV        = './drugbank/drugbank_human.csv'
GROUND_TRUTH_CSV     = './drug_ground_truth.csv'
# ──────────────────────────────────────────────────────────────────────────────


def load_data():
    with open(DISEASE_SNP_GENE_PKL, 'rb') as f:
        d = pickle.load(f)
    final_disease = d['final_disease']
    final_gene    = d['final_gene']

    with open(DISEASE_SNP_MATRIX, 'rb') as f:
        DpS_matrix = pickle.load(f)

    with open(PPI_NET_PKL, 'rb') as f:
        ppi_net = pickle.load(f)

    gene_list = pd.read_csv(PROTEIN_SYMBOL_CSV, index_col=0).to_numpy().squeeze()

    with open(DRUG_TARGET_PKL, 'rb') as f:
        DrT_matrix = pickle.load(f)

    drug_info = pd.read_csv(DRUG_INFO_CSV, index_col=0)
    drug_name = np.union1d(drug_info['drug_name'].to_numpy(),
                           drug_info['drug_name'].to_numpy())

    drug_truth = pd.read_csv(GROUND_TRUTH_CSV, index_col=0)

    return (final_disease, final_gene, DpS_matrix,
            ppi_net, gene_list, DrT_matrix, drug_name, drug_truth)


def build_disease_gene_matrix(final_disease, final_gene, gene_list):
    """
    Build a binary disease-gene association matrix (m x n).

    Parameters
    ----------
    final_disease : list of str
    final_gene    : list of lists
    gene_list     : np.ndarray (n,)

    Returns
    -------
    DG_matrix : np.ndarray, shape (m, n)
    """
    m  = len(final_disease)
    n  = len(gene_list)
    DG = np.zeros((m, n), dtype=np.uint8)
    for i, genes in enumerate(final_gene):
        idx = np.where(np.isin(gene_list, np.array(genes)))[0]
        if len(idx) > 0:
            DG[i, idx] = 1
    return DG


def build_full_network(DpS_matrix, DG_matrix, ppi_net, DrT_matrix):
    """
    Construct the three-layered disease-protein-drug adjacency matrix.

    Block structure (m diseases + n proteins + q drugs):

        W_full = [[W^D,          DG,          0     ],
                  [DG^T,         W^P,         W^Dr~P^T],
                  [0^T,          W^Dr~P,      W^Dr  ]]

    where:
        W^D     = DDN similarity (cosine, m x m)
        DG      = disease-gene association (m x n)
        W^P     = PPI adjacency (n x n)
        W^Dr~P  = drug-protein association (q x n)
        W^Dr    = drug-drug cosine similarity (q x q)

    Parameters
    ----------
    DpS_matrix : np.ndarray (m, p) — disease-SNP matrix for DDN similarity
    DG_matrix  : np.ndarray (m, n)
    ppi_net    : np.ndarray (n, n)
    DrT_matrix : np.ndarray (q, n)

    Returns
    -------
    W_full : np.ndarray, shape (m+n+q, m+n+q)
    """
    m = len(DpS_matrix)
    q = len(DrT_matrix)

    W_D    = metrics.pairwise.cosine_similarity(DpS_matrix)   # m x m
    W_Dr   = metrics.pairwise.cosine_similarity(DrT_matrix)   # q x q
    null_D_Dr = np.zeros((m, q))

    row1 = np.hstack([W_D,            DG_matrix,    null_D_Dr])
    row2 = np.hstack([DG_matrix.T,    ppi_net,      DrT_matrix.T])
    row3 = np.hstack([null_D_Dr.T,    DrT_matrix,   W_Dr])

    W_full = np.vstack([row1, row2, row3])
    return W_full, m, len(ppi_net)


def compute_graph_ssl_inverse(W, mu):
    D   = np.diag(W.sum(axis=1))
    L   = D - W
    inv = np.linalg.inv(np.eye(len(W)) + mu * L)
    return inv


def run_benchmark(disease_code, mu,
                  final_disease, final_gene,
                  DpS_matrix, ppi_net, gene_list,
                  DrT_matrix, drug_name, drug_truth,
                  save_scores=True):
    """
    Run the benchmark (direct integration) experiment.
    """
    disease_list = np.array(final_disease)
    idx = np.where(disease_list == disease_code)[0]
    if len(idx) == 0:
        raise ValueError(f"Disease code '{disease_code}' not found.")
    index = idx[0]

    print(f"\n[BENCHMARK] Disease: {disease_code} (index={index}), mu={mu}")

    # Build disease-gene matrix
    print("  Building disease-gene matrix...")
    DG_matrix = build_disease_gene_matrix(final_disease, final_gene, gene_list)

    # Build full three-layered network
    print("  Building full disease-protein-drug network...")
    W_full, n_disease, n_protein = build_full_network(
        DpS_matrix, DG_matrix, ppi_net, DrT_matrix)

    # Compute SSL inverse
    print(f"  Computing (I + mu*L)^{{-1}} for {W_full.shape[0]} nodes...")
    inv_lap_full = compute_graph_ssl_inverse(W_full, mu)
    del W_full

    # Set label for index disease, propagate
    print("  Propagating labels...")
    y        = np.zeros(len(inv_lap_full))
    y[index] = 1.0
    f_full   = inv_lap_full @ y

    # Extract drug layer scores
    drug_start     = n_disease + n_protein
    f_drug         = f_full[drug_start:]

    # Evaluate
    truth_col    = DISEASE_TRUTH_COL[disease_code]
    truth_labels = drug_truth[truth_col].to_numpy()

    fpr, tpr, _ = metrics.roc_curve(truth_labels, f_drug, pos_label=1)
    auc          = metrics.auc(fpr, tpr)

    top_n      = int(len(f_drug) * 0.1)
    sorted_idx = np.argsort(f_drug)[::-1]
    top_truth  = truth_labels[sorted_idx[:top_n]]
    p_top10    = top_truth.sum() / top_n
    baseline   = truth_labels.sum() / len(truth_labels)

    print(f"  AUC        : {auc:.4f}")
    print(f"  P@Top10%   : {p_top10:.4f}  (baseline: {baseline:.4f})")

    if save_scores:
        out_df = pd.DataFrame({
            'drug_name':  drug_name,
            'drug_score': f_drug,
            'truth':      truth_labels,
        }).sort_values('drug_score', ascending=False)
        fname = f"benchmark_scores_{disease_code}_mu{mu}.csv"
        out_df.to_csv(fname, index=False)
        print(f"  Saved: {fname}")

    return auc, p_top10, baseline


def run_mu_search(disease_code, final_disease, final_gene,
                  DpS_matrix, ppi_net, gene_list,
                  DrT_matrix, drug_name, drug_truth):
    print(f"\n[MU SEARCH - BENCHMARK] Disease: {disease_code}")
    print(f"  Search space: {MU_SEARCH_SPACE}\n")

    records = []
    for mu in MU_SEARCH_SPACE:
        auc, p_top10, baseline = run_benchmark(
            disease_code, mu,
            final_disease, final_gene,
            DpS_matrix, ppi_net, gene_list,
            DrT_matrix, drug_name, drug_truth,
            save_scores=False)
        records.append({'disease': disease_code, 'mu': mu,
                        'AUC': auc, 'P@Top10%': p_top10,
                        'Baseline_P@Top10%': baseline})

    df = pd.DataFrame(records)
    print("\n[MU SEARCH RESULTS - BENCHMARK]")
    print(df.to_string(index=False))
    df.to_csv('benchmark_results_mu_search.csv', index=False)
    print("\nSaved: benchmark_results_mu_search.csv")
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark: direct integration drug scoring')
    parser.add_argument('--disease',   type=str, default='RA',
                        choices=['RA', 'AS', 'MS'])
    parser.add_argument('--mu',        type=float, default=0.5)
    parser.add_argument('--search_mu', action='store_true')
    args = parser.parse_args()

    disease_code = DISEASE_CODE[args.disease]

    print("[LOADING DATA]")
    (final_disease, final_gene, DpS_matrix,
     ppi_net, gene_list, DrT_matrix,
     drug_name, drug_truth) = load_data()

    print(f"  Diseases : {len(final_disease)}")
    print(f"  Proteins : {len(gene_list)}")
    print(f"  Drugs    : {len(drug_name)}")

    if args.search_mu:
        run_mu_search(
            disease_code, final_disease, final_gene,
            DpS_matrix, ppi_net, gene_list,
            DrT_matrix, drug_name, drug_truth)
    else:
        run_benchmark(
            disease_code, args.mu,
            final_disease, final_gene,
            DpS_matrix, ppi_net, gene_list,
            DrT_matrix, drug_name, drug_truth)


if __name__ == '__main__':
    main()
