# -*- coding: utf-8 -*-
"""
2_snp_to_gene_mapping.py
------------------------
Maps disease-associated SNPs to gene (protein) symbols.

This script bridges the DDN (SNP space, dimension p) and Dr-PPIN
(protein space, dimension n) by annotating each SNP to its
corresponding gene symbol using a pre-built SNP-to-gene mapping table
generated via ANNOVAR with the refGene database.

Inputs:
    - final_used_disease_snp.pkl         : output from 1_build_DDN.py
    - SNP2Gene_Mapping_Data/final_snp_to_gene.csv
                                         : SNP-to-gene mapping table
                                           (columns: SNP_used, gene)
                                           generated via ANNOVAR (refGene)

Outputs:
    - final_disease_snp_gene.pkl         : {final_disease, final_snp, final_gene}
"""

import pickle
import numpy as np
import pandas as pd


# ── User settings ─────────────────────────────────────────────────────────────
DISEASE_SNP_PKL  = 'final_used_disease_snp.pkl'
SNP2GENE_CSV     = './SNP2Gene_Mapping_Data/final_snp_to_gene.csv'
OUTPUT_PKL       = 'final_disease_snp_gene.pkl'
# ──────────────────────────────────────────────────────────────────────────────


def map_snps_to_genes(snp_list, snp_coord, gene_coord):
    """
    Map a list of SNP IDs to gene symbols using a lookup table.

    Parameters
    ----------
    snp_list : array-like
        SNP IDs for a single disease.
    snp_coord : np.ndarray
        Reference SNP IDs from the mapping vocabulary.
    gene_coord : np.ndarray
        Gene symbols corresponding to snp_coord.

    Returns
    -------
    genes : list of str
        Gene symbols mapped from the input SNP list.
        SNPs without a mapping entry are silently skipped.
    """
    genes = []
    for snp_id in snp_list:
        idx = np.where(snp_coord == snp_id)[0]
        if len(idx) > 0:
            genes.append(gene_coord[idx[0]])
    return genes


def main():
    # 1. Load disease-SNP data
    with open(DISEASE_SNP_PKL, 'rb') as f:
        loaded = pickle.load(f)
    final_disease = loaded['final_disease']
    final_snp     = loaded['final_snp']
    print(f"Loaded {len(final_disease)} diseases from {DISEASE_SNP_PKL}")

    # 2. Load SNP-to-gene mapping vocabulary
    mapping = pd.read_csv(SNP2GENE_CSV)
    snp_coord  = mapping['SNP_used'].to_numpy()
    gene_coord = mapping['gene'].to_numpy()
    print(f"SNP-to-gene vocabulary size: {len(snp_coord)}")

    # 3. Map each disease's SNPs to gene symbols
    final_gene = []
    for i, (disease_id, snp_list) in enumerate(zip(final_disease, final_snp)):
        genes = map_snps_to_genes(snp_list, snp_coord, gene_coord)
        final_gene.append(genes)
        print(f"[{i+1}/{len(final_disease)}] {disease_id}: "
              f"{len(snp_list)} SNPs → {len(genes)} genes mapped")

    # 4. Save output
    output = {
        'final_disease': final_disease,
        'final_snp':     final_snp,
        'final_gene':    final_gene,
    }
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(output, f)

    print(f"\n[DONE] Saved {OUTPUT_PKL}")
    print(f"  Diseases: {len(final_disease)}")
    unmapped = sum(1 for g in final_gene if len(g) == 0)
    print(f"  Diseases with zero gene mappings: {unmapped}")


if __name__ == '__main__':
    main()
