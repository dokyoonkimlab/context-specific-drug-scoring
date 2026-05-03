# -*- coding: utf-8 -*-
"""
3_label_propagation.py
-----------------------
Main experiment: context-specific drug scoring via network-to-network
label propagation (Algorithm 1 in the manuscript).

Pipeline (per index disease):
    Step 1. Compute disease similarity scores on DDN via graph-based SSL.
    Step 2. For each disease pair (index, j):
              - Build shared-protein label vector M^(D->P)_j in {0,1,2}^n
              - Propagate labels from protein layer to drug layer in Dr-PPIN
    Step 3. Combine drug scores weighted by disease similarity scores.
    Step 4. Evaluate AUC and P@Top10% against ground truth.

Usage:
    # Single run with fixed mu
    python 3_label_propagation.py --disease RA --mu 0.5

    # Hyperparameter search over mu
    python 3_label_propagation.py --disease RA --search_mu

Disease codes:
    RA  →  Phe_714.1   (Rheumatoid Arthritis)
    AS  →  Phe_495     (Asthma)
    MS  →  Phe_335     (Multiple Sclerosis)

Inputs:
    - final_disease_snp_gene.pkl         : output from 2_snp_to_gene_mapping.py
    - final_disease_pruned_snp_matrix.pkl: binary disease x SNP matrix
    - combined_ppi_net.pkl               : PPI adjacency matrix (n x n)
    - ./STRING_DB/used_full_protein_symbol.csv : protein symbol list
    - drug_target_matrix.pkl             : drug-protein binary matrix (q x n)
    - ./drugbank/drugbank_human.csv      : drug metadata
    - ./drug_ground_truth.csv            : ground truth disease-drug associations

Outputs:
    - drug_scores_<disease>_mu<mu>.csv   : ranked drug scores
    - results_summary.csv                : AUC and P@Top10% per mu
"""

import os
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

# ── Hyperparameter search space ────────────────────────────────────────────────
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
    """Load all required network data."""
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


def compute_graph_ssl_inverse(W, mu):
    """
    Compute the closed-form inverse for graph-based SSL:
        (I + mu * L)^{-1}
    where L = D - W is the graph Laplacian.

    Parameters
    ----------
    W : np.ndarray, shape (N, N)
        Adjacency / similarity matrix.
    mu : float
        Regularization parameter.

    Returns
    -------
    inv : np.ndarray, shape (N, N)
    """
    D   = np.diag(W.sum(axis=1))
    L   = D - W
    inv = np.linalg.inv(np.eye(len(W)) + mu * L)
    return inv


def build_drppin_inverse(ppi_net, DrT_matrix, drug_name, mu):
    """
    Build the Dr-PPIN block-wise adjacency matrix and compute its SSL inverse.

    Block structure (n proteins + q drugs):
        W^DrP = [[W^P,        (W^Dr~P)^T],
                 [W^Dr~P,      W^Dr      ]]

    Parameters
    ----------
    ppi_net    : np.ndarray (n x n) — protein-protein interaction matrix
    DrT_matrix : np.ndarray (q x n) — drug-protein association matrix (W^Dr~P)
    drug_name  : np.ndarray (q,)    — drug identifiers
    mu         : float              — regularization parameter

    Returns
    -------
    inv_lap_ppi_dr : np.ndarray, shape (n+q, n+q)
    """
    DrNet   = metrics.pairwise.cosine_similarity(DrT_matrix)   # W^Dr (q x q)
    upper   = np.hstack([ppi_net, DrT_matrix.T])               # (n x n+q)
    lower   = np.hstack([DrT_matrix, DrNet])                   # (q x n+q)
    W_DrP   = np.vstack([upper, lower])                        # (n+q x n+q)

    inv_lap = compute_graph_ssl_inverse(W_DrP, mu)
    return inv_lap


def ssl_on_ddn(DpS_matrix, mu):
    """
    Run graph-based SSL on the DDN for all diseases.

    For each disease i, sets y_i = 1 and y_j = 0 (j != i),
    then computes f = (I + mu*L)^{-1} y.

    Parameters
    ----------
    DpS_matrix : np.ndarray, shape (m, p)
        Binary disease x pruned-SNP matrix.
    mu : float

    Returns
    -------
    result_score_f : np.ndarray, shape (m, m)
        result_score_f[:, i] = disease similarity scores when disease i is index.
    """
    sim_DDN = metrics.pairwise.cosine_similarity(DpS_matrix)
    inv_lap = compute_graph_ssl_inverse(sim_DDN, mu)

    m = len(DpS_matrix)
    result_score_f = np.zeros((m, m))
    for i in range(m):
        y    = np.zeros(m)
        y[i] = 1.0
        result_score_f[:, i] = inv_lap @ y

    return result_score_f


def normalize_disease_scores(f_raw, index):
    """
    Normalize disease similarity scores for the index disease.
    Sets the index disease score to 1 and normalizes others to [0, 1]
    relative to the second-highest score.

    Parameters
    ----------
    f_raw : np.ndarray (m,)
    index : int — index of the target disease

    Returns
    -------
    f_norm : np.ndarray (m,)
    """
    f = f_raw.copy()
    f[f == f.max()] = 10.0
    second_max = np.partition(f, -2)[-2]
    mask = f != 10.0
    f[mask] = (f[mask] - f[mask].min()) / (second_max - f[mask].min() + 1e-12)
    f[index] = 1.0
    return f


def build_label_matrix(target_gene, alter_gene, gene_list):
    """
    Build the initial label vector M^(D->P)_j for a disease pair (i, j).

    Scoring scheme:
        2 : protein k is mapped from both disease i and disease j
        1 : protein k is mapped from exactly one disease
        0 : protein k is not mapped from either disease

    Parameters
    ----------
    target_gene : list of str — proteins from index disease i
    alter_gene  : list of str — proteins from disease j
    gene_list   : np.ndarray (n,) — full protein symbol list

    Returns
    -------
    y_ppi : np.ndarray, shape (n, 1)
        Label vector for the protein layer.
    """
    target_gene = np.array(target_gene)
    alter_gene  = np.array(alter_gene)

    in_target = np.isin(gene_list, target_gene).astype(int)
    in_alter  = np.isin(gene_list, alter_gene).astype(int)

    # 2 if both, 1 if either, 0 if neither
    y_ppi = (in_target + in_alter).reshape(-1, 1)   # values in {0, 1, 2}
    return y_ppi


def compute_drug_scores(index, final_gene, gene_list, inv_lap_ppi_dr, drug_name):
    """
    Compute drug scores for all disease pairs relative to the index disease
    via label propagation over Dr-PPIN (Step 2 of Algorithm 1).

    f^Dr = mu * {I + mu*(D^Dr - W^Dr)}^{-1} * W^{Dr~P} * M^{D->P}

    Note: since inv_lap_ppi_dr = (I + mu*L^DrP)^{-1} is pre-computed for the
    full block matrix, we propagate y = [y_ppi; 0] and extract the drug layer.

    Parameters
    ----------
    index         : int — index disease position
    final_gene    : list of lists — gene lists per disease
    gene_list     : np.ndarray (n,)
    inv_lap_ppi_dr: np.ndarray (n+q, n+q)
    drug_name     : np.ndarray (q,)

    Returns
    -------
    drug_score_matrix : np.ndarray, shape (q, m)
        drug_score_matrix[:, j] = drug scores from pair (index, j).
    """
    n_genes = len(gene_list)
    n_drugs = len(drug_name)
    m       = len(final_gene)

    drug_score_matrix = np.zeros((n_drugs, m))

    for j in range(m):
        target_gene = final_gene[index]
        alter_gene  = final_gene[j]

        y_ppi  = build_label_matrix(target_gene, alter_gene, gene_list)
        y_drug = np.zeros((n_drugs, 1))
        y      = np.vstack([y_ppi, y_drug])            # (n+q, 1)

        f      = inv_lap_ppi_dr @ y
        f_drug = f[n_genes:].reshape(-1)

        # Min-max normalization
        f_min, f_max = f_drug.min(), f_drug.max()
        if f_max > f_min:
            f_drug = (f_drug - f_min) / (f_max - f_min)

        drug_score_matrix[:, j] = f_drug

    return drug_score_matrix


def evaluate(drug_scores, truth_labels):
    """
    Compute AUC and Precision@Top10%.

    Parameters
    ----------
    drug_scores  : np.ndarray (q,)
    truth_labels : np.ndarray (q,) — binary ground truth

    Returns
    -------
    auc   : float
    p_top10 : float — precision at top 10% ranked drugs
    baseline_p_top10 : float — random baseline precision at top 10%
    """
    fpr, tpr, _ = metrics.roc_curve(truth_labels, drug_scores, pos_label=1)
    auc         = metrics.auc(fpr, tpr)

    top_n       = int(len(drug_scores) * 0.1)
    sorted_idx  = np.argsort(drug_scores)[::-1]
    top_truth   = truth_labels[sorted_idx[:top_n]]
    p_top10     = top_truth.sum() / top_n
    baseline    = truth_labels.sum() / len(truth_labels)

    return auc, p_top10, baseline


def run_single(disease_code, mu, final_disease, final_gene,
               DpS_matrix, ppi_net, gene_list,
               DrT_matrix, drug_name, drug_truth, save_scores=True):
    """
    Run the full drug scoring pipeline for one disease and one mu value.
    """
    disease_list = np.array(final_disease)
    idx = np.where(disease_list == disease_code)[0]
    if len(idx) == 0:
        raise ValueError(f"Disease code '{disease_code}' not found in disease list.")
    index = idx[0]

    print(f"\n[RUN] Disease: {disease_code} (index={index}), mu={mu}")

    # Step 1: SSL on DDN → disease similarity scores
    print("  Step 1: SSL on DDN...")
    score_matrix = ssl_on_ddn(DpS_matrix, mu)
    f_disease    = normalize_disease_scores(score_matrix[:, index], index)

    # Step 2: Build Dr-PPIN inverse
    print("  Step 2: Building Dr-PPIN inverse (mu={})...".format(mu))
    inv_lap_ppi_dr = build_drppin_inverse(ppi_net, DrT_matrix, drug_name, mu)

    # Step 3: Drug scores per disease pair → f^Dr (q x m)
    print("  Step 3: Label propagation over Dr-PPIN...")
    drug_score_matrix = compute_drug_scores(
        index, final_gene, gene_list, inv_lap_ppi_dr, drug_name)

    # Step 4: Inference combination F = [f^Dr]^T x f^D
    print("  Step 4: Inference combination...")
    final_drug_scores = drug_score_matrix @ f_disease   # (q,)

    # Step 5: Evaluate
    truth_col    = DISEASE_TRUTH_COL[disease_code]
    truth_labels = drug_truth[truth_col].to_numpy()
    auc, p_top10, baseline = evaluate(final_drug_scores, truth_labels)

    print(f"  AUC        : {auc:.4f}")
    print(f"  P@Top10%   : {p_top10:.4f}  (baseline: {baseline:.4f})")

    # Save ranked drug scores
    if save_scores:
        out_df = pd.DataFrame({
            'drug_name':  drug_name,
            'drug_score': final_drug_scores,
            'truth':      truth_labels,
        }).sort_values('drug_score', ascending=False)
        fname = f"drug_scores_{disease_code}_mu{mu}.csv"
        out_df.to_csv(fname, index=False)
        print(f"  Saved: {fname}")

    return auc, p_top10, baseline


def run_mu_search(disease_code, final_disease, final_gene,
                  DpS_matrix, ppi_net, gene_list,
                  DrT_matrix, drug_name, drug_truth):
    """
    Search over mu values and report AUC for each.
    """
    print(f"\n[MU SEARCH] Disease: {disease_code}")
    print(f"  Search space: {MU_SEARCH_SPACE}\n")

    records = []
    for mu in MU_SEARCH_SPACE:
        auc, p_top10, baseline = run_single(
            disease_code, mu,
            final_disease, final_gene,
            DpS_matrix, ppi_net, gene_list,
            DrT_matrix, drug_name, drug_truth,
            save_scores=False,
        )
        records.append({'disease': disease_code, 'mu': mu,
                        'AUC': auc, 'P@Top10%': p_top10,
                        'Baseline_P@Top10%': baseline})

    df = pd.DataFrame(records)
    print("\n[MU SEARCH RESULTS]")
    print(df.to_string(index=False))
    df.to_csv('results_mu_search.csv', index=False)
    print("\nSaved: results_mu_search.csv")
    return df





def main():
    parser = argparse.ArgumentParser(
        description='Context-specific drug scoring via network-to-network label propagation')
    parser.add_argument('--disease',   type=str, default='RA',
                        choices=['RA', 'AS', 'MS'],
                        help='Index disease (default: RA)')
    parser.add_argument('--mu',        type=float, default=0.5,
                        help='Regularization parameter mu (default: 0.5)')
    parser.add_argument('--search_mu', action='store_true',
                        help='Search over mu values: '
                             '{0.001, 0.01, 0.1, 1.0, 0.5, 10, 100}')
    args = parser.parse_args()

    disease_code = DISEASE_CODE[args.disease]

    # Load data
    print("[LOADING DATA]")
    (final_disease, final_gene, DpS_matrix,
     ppi_net, gene_list, DrT_matrix,
     drug_name, drug_truth) = load_data()

    print(f"  Diseases   : {len(final_disease)}")
    print(f"  Proteins   : {len(gene_list)}")
    print(f"  Drugs      : {len(drug_name)}")

    if args.search_mu:
        run_mu_search(
            disease_code, final_disease, final_gene,
            DpS_matrix, ppi_net, gene_list,
            DrT_matrix, drug_name, drug_truth)
    else:
        run_single(
            disease_code, args.mu,
            final_disease, final_gene,
            DpS_matrix, ppi_net, gene_list,
            DrT_matrix, drug_name, drug_truth)


if __name__ == '__main__':
    main()
