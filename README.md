# FFD-STRA-GAT

## Financial Fraud Detection using Spatiotemporal Risk-Aware Graph Attention Networks

**Author:** Barshneyo Chakraborty
**Project Type:** Research-Oriented Graph Neural Network Framework for Financial Fraud Detection

---

# Overview

Financial fraud detection in large-scale transaction networks requires robust modeling of:

* Node interactions (sender–receiver relationships)
* Temporal transaction dynamics
* Anomaly propagation across transaction graphs

Traditional fraud detection systems rely heavily on static thresholds. Such approaches frequently fail because legitimate high-value transactions and fraudulent high-value transactions may appear identical when evaluated using a fixed cutoff.

FFD-STRA-GAT introduces a graph-based, risk-aware, spatiotemporal framework that combines:

* Graph Attention Networks (GAT)
* Learnable anomaly scoring
* Dynamic tolerance estimation
* Adaptive thresholding
* Graph-based risk propagation

to detect fraudulent behavior in transaction networks.

---

# Key Idea

Instead of relying on fixed transaction limits, FFD-STRA-GAT learns:

* Context-aware tolerance
* Relationship-aware risk
* Dynamic anomaly thresholds
* Attention-weighted transaction importance

allowing fraud decisions to adapt to the surrounding transaction ecosystem.

---

# Graph Representation

Let

G = (V, E)

where:

* V = account nodes
* E = transaction edges

The transaction network is represented as a directed graph.

Adjacency matrix:

A[i,j] ∈ {0,1}

where:

* 0 → no transaction
* 1 → transaction exists

A dense graph yields:

Complexity = O(n²)

while sparse transaction networks reduce complexity to:

Complexity = O(k·n)

---

# Feature Representation

Each transaction is represented using a 17-dimensional feature vector.

## Sender Features

* Trust score
* Account age
* Location score
* Average balance
* Volatility
* Account type

## Receiver Features

* Trust score
* Account age
* Location score
* Average balance
* Volatility
* Account type

## Edge Features

* Transaction amount
* Transaction deviation
* Relationship strength
* Dynamic tolerance

Total features:

17-dimensional feature vector

---

# Transaction Deviation

Deviation from historical behavior is computed as

Δ_ij(t) = |F_ij(t) − Tavg_ij(t)|

where:

* F_ij(t) = current transaction value
* Tavg_ij(t) = historical average transaction value

---

# Running Transaction Statistics

## Running Mean

Tavg_ij(t+1) =
(n_ij(t)·Tavg_ij(t) + F_ij(t+1))
/
(n_ij(t)+1)

## Running Standard Deviation

σ_ij(t)

captures transaction volatility over time.

---

# Category Mismatch

Transactions between different account categories receive higher anomaly weight.

Catmis =

* 1.0 if account types differ
* γ if account types are identical

Current implementation:

γ = 0.5

Future versions will make γ learnable.

---

# Relationship Strength

Relationship strength between two accounts is defined as:

R_ij(t) =
n_ij(t)
/ Σ_k n_ik(t)

This measures how frequently two accounts interact relative to all outgoing interactions.

---

# Learnable Tolerance Model

FFD-STRA-GAT replaces static tolerances with learnable tolerances:

ε_ij(t) =
α_ij(t) ε_i
+
(1−α_ij(t)) ε_j
+
Ψ_ij(t) σ_ij(t)

where:

* α_ij(t) = learnable mixing coefficient
* Ψ_ij(t) = learnable volatility sensitivity
* σ_ij(t) = transaction standard deviation

---

# Unweighted Anomaly Score

AS_ij(t) =
Δ_ij(t) · Catmis
/
ε_ij(t)

Interpretation:

* Generalized z-score
* Scale invariant
* Non-negative
* Monotonically increases with deviation

Unlike classical z-scores, the denominator is learnable and context-aware.

---

# Attention-Based Risk Aggregation

Node embeddings:

h_i = W X_i

Attention score:

Ats_ij(t) = LeakyReLU(aᵀ z_ij)

where

z_ij = [W X_i , W X_j , e_ij]

Attention normalization:

R_ij(t) =
exp(Ats_ij(t))
/
Σ_k exp(Ats_ik(t))

---

# Learnable Weight Update

w_ij(t+1) =
w_ij(t)
· log(1 + exp(AS_ij(t)))
· (1 + λ_ij(t)R_ij(t))

where

λ_ij(t) = Softplus(cᵀ z_ij)

---

# Weighted Anomaly Propagation

Weighted anomaly contribution:

ϕ_ij(t) =
w_ij(t) · AS_ij(t)

Node aggregation:

ρ_i(t) =
log Σ_j exp(ϕ_ij(t))

---

# Dynamic Thresholding

FFD-STRA-GAT replaces static thresholds with adaptive thresholds:

Θ_i(t) =
μ_i(t)
+
k_i(t)σ_i(t)
+
log(|N(i)|)

where:

* μ_i(t) = running mean anomaly
* σ_i(t) = running standard deviation
* k_i(t) = learnable sensitivity parameter

---

# Output Layer

Fraud score:

f_i(t) =
(
Θ_i(t)
−
AS_i(W,t)
)
/
(
σ_i(t)+1
)

Probability:

p_i(t) = sigmoid(f_i(t))

---

# Loss Function

Weighted Binary Cross Entropy:

L =
− Σ_i
[
τ y_i log(p_i)
+
(1−y_i) log(1−p_i)
]
+
ν ||Ω||²

where:

* τ = fraud-class weighting
* ν = L2 regularization coefficient

---

# Distributed Training

FFD-STRA-GAT supports PyTorch Distributed Data Parallel (DDP).

Training procedure:

1. Graph partitioning
2. Local forward propagation
3. Boundary attention exchange
4. Gradient synchronization
5. All-reduce update

---

# Complexity Analysis

Let:

* n = number of nodes
* m = number of edges
* d = embedding dimension
* P = number of workers

## Time Complexity

O((n+m)d / P)

## Communication Complexity

O(mbd + |Ω| log P)

## Space Complexity

O(|Ω| + (n+m)d/P)

## Mini-Batch Complexity

O(Bkd)

where:

* B = batch size
* k = sampled neighbors

---

# Experimental Setup

Configuration:

* Device: CUDA
* Epochs: 250
* Hidden Dimension: 64
* Layers: 3
* Data Split: 60/20/20 (Stratified)

Synthetic Dataset:

* Nodes: 800
* Edges: 5,957
* Fraud Nodes: 111 (13.88%)

Baselines:

* GCN
* GraphSAGE

FFD-STRA-GAT converged in 73 epochs.

Best threshold:

0.740

---

# Results

| Model        | AUC    | AP     | F1     | Precision | Recall | Balanced Accuracy |
| ------------ | ------ | ------ | ------ | --------- | ------ | ----------------- |
| FFD-STRA-GAT | 0.9941 | 0.9415 | 0.9478 | 0.9160    | 0.9820 | 0.9837            |
| GCN          | 0.5432 | 0.1847 | 0.2483 | 0.1421    | 0.9820 | 0.5135            |
| GraphSAGE    | 0.6425 | 0.2431 | 0.2982 | 0.1868    | 0.7387 | 0.6103            |

---

# Parameter Counts

| Model        | Parameters |
| ------------ | ---------- |
| FFD-STRA-GAT | 112,431    |
| GCN          | 9,025      |
| GraphSAGE    | 17,793     |

---

# Why FFD-STRA-GAT Outperforms

GCN and GraphSAGE are generic message-passing architectures that must learn fraud behavior entirely from raw graph signals.

FFD-STRA-GAT incorporates:

* Deviation-based anomaly scoring
* Learnable tolerance estimation
* Dynamic thresholds
* Risk-aware attention
* Anomaly propagation mechanisms

This provides a strong domain-specific inductive bias for fraud detection.

---

# Future Work

* Make γ (category mismatch) learnable
* Make τ (loss weighting) learnable
* Make ν (regularization coefficient) learnable
* Validate on real-world banking datasets
* Scale distributed training to larger worker counts
* Introduce richer edge features
* Explore multi-hop anomaly propagation

---

# Citation

If you use this work, please cite:

Barshneyo Chakraborty,
"FFD-STRA-GAT: Financial Fraud Detection using Spatiotemporal Risk-Aware Graph Attention Networks"
