# Context-Specific Drug Scoring via Network-to-Network Label Propagation

This repository contains the implementation code for the paper:

> **"Context-specific drug scoring for precision repurposing via synergizing label propagation with heterogeneous networks"**  
> Yonghyun Nam, Sang-Hyuk Jung, Jakob Woerner, et al.  
> *Journal of Big Data* (under revision)

---

## Overview

This framework integrates a data-driven disease-disease network (DDN) and a knowledge-driven drug-target protein network (Dr-PPIN) via network-to-network label propagation for context-specific drug repurposing.

**Pipeline:**
```
1. Build DDN       : PheWAS summary statistics → Disease-Disease Network
2. SNP-to-Gene     : SNP → Protein mapping via ANNOVAR / dbSNP
3. Label propagation over Dr-PPIN → Drug scores per disease pair
4. Inference combination → Final drug scores for index disease
5. Benchmark comparison (direct integration network)
```

---

## Requirements

```bash
pip install numpy pandas scikit-learn biothings_client
```

---

## Data Sources

All datasets used in this study are publicly available:

| Data | Source |
|------|--------|
| PheWAS summary statistics | UK Biobank GWAS Results |
| Protein-protein interactions | BioGRID, STRING (confidence ≥ 0.8) |
| Drug-target associations | DrugBank |
| Disease-drug ground truth | CTD, DrugBank |
| SNP-to-gene annotation | ANNOVAR (refGene database) |

---

## Usage

### Step 1: Build Disease-Disease Network (DDN)
```bash
python 1_build_DDN.py
```
- Input: UKB PheWAS summary statistics (pickle format)
- Filters: n_cases > 500, p-value < 1×10⁻⁵, LD pruning (window=50kb, step=5kb, r²=0.5)
- Output: `final_disease_pruned_snp_matrix.pkl`, `final_used_disease_snp.pkl`

### Step 2: SNP-to-Gene Mapping
```bash
python 2_snp_to_gene_mapping.py
```
- Input: SNP list from Step 1
- Maps SNPs to gene symbols via ANNOVAR refGene annotations
- Output: `final_disease_snp_gene.pkl`

### Step 3: Label Propagation (Main Experiment)
```bash
python 3_label_propagation.py --disease RA --mu 0.5
```
- `--disease`: target disease code (`RA`, `AS`, `MS`)
- `--mu`: regularization parameter (default: 0.5; see hyperparameter search below)
- Output: drug score vector, AUC, P@Top10%

### Step 4: Hyperparameter Search (μ)
```bash
python 3_label_propagation.py --disease RA --search_mu
```
- Searches over μ ∈ {0.001, 0.01, 0.1, 1.0, 0.5, 10, 100}
- Reports AUC for each μ value

### Step 5: Benchmark Comparison
```bash
python 4_comparison_benchmark.py --disease RA --mu 0.5
```
- Builds a three-layered disease-protein-drug network (direct integration)
- Outputs AUC for benchmark model

---

## Hyperparameter

| Parameter | Description | Search space | Default |
|-----------|-------------|--------------|---------|
| μ | Regularization parameter in graph-based SSL | {0.001, 0.01, 0.1, 1.0, 0.5, 10, 100} | 0.5 |

---

## Output

- Drug score ranking for the index disease
- AUC (ROC curve) vs. ground truth from DrugBank/CTD
- P@Top10%: precision at top 10% ranked drugs

---

## Citation

```
@article{nam2025drugscoring,
  title={Context-specific drug scoring for precision repurposing via synergizing label propagation with heterogeneous networks},
  author={Nam, Yonghyun and Jung, Sang-Hyuk and Woerner, Jakob and others},
  journal={Journal of Big Data},
  year={2025}
}
```

---

## Contact

For questions or additional information, please contact the lead author at:  
**dokyoon.kim@pennmedicine.upenn.edu**
