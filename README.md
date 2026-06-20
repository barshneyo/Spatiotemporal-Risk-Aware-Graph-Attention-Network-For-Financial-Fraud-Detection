FFD-STRA-GAT: Spatiotemporal Risk-Aware Graph Attention Network for Financial Fraud Detection

FFD-STRA-GAT is an advanced spatiotemporal attentional framework designed to detect financial fraud in large-scale transaction networks
. Unlike generic Graph Neural Networks (GNNs) that must learn fraud patterns from scratch, this model integrates domain-informed deviation-driven anomaly scoring with a learnable Graph Attention Network (GAT) extension
.This repository contains the theoretical foundations, algorithmic framework, and experimental benchmarks for a system that replaces brittle, static thresholds with context-aware, learnable risk tolerance


TABLE OF CONTENTS
Core Motivation
Innovation: The Anomaly Score (AS)
Architecture Overview
Mathematical Framework
Algorithmic Design & Scalability
Benchmarks & Performance
Getting Started
Future Work


CORE MOTIVATION
Traditional fraud detection systems rely on static per-account thresholds (e.g., "flag any transfer over $10,000")
. These are obsolete in modern finance: a legitimate high-value transfer by a corporate entity may be flagged identically to a fraudulent transfer, causing high false-positive rates
.FFD-STRA-GAT addresses this by:
Modeling Node Interactions: Capturing relationships between senders and receivers
.Temporal Dynamics: Tracking evolving transaction patterns over time
Risk-Aware Aggregation: Using max aggregation to localize fraud sources rather than diluting signals across a neighborhood

INNOVATION: The Anomaly Score (AS)
The heart of the framework is the Unweighted Minimal Anomaly Score, which acts as a "generalized z-score"
AS_ij(t) = (1/ϵ_ij(t))*(Δ_ij(t)*Catmis)
​
 
Where:
Δ_ij(t)(Transaction Deviation): The absolute difference between the current transaction value and the historical mean
Catmis (Category Mismatch): A penalty applied if transaction categories between nodes do not align
ϵ_ij(t)(Learnable Tolerance): A context-aware scale parameter that replaces the fixed standard deviation used in classical statistics




ARCHITECTURE OVERVIEW : 
The model follows a 5-step end-to-end pipeline:
Node Embedding: Transform raw features X_i into a hidden representation h_i
Attention Scoring: Compute Ats_ij(t) using a learnable attention vector a_T and concatenated feature vectors
Normalization: Use Softmax to determine relationship strength R_ij(t)
Aggregation: Aggregate neighbor features into a node-level risk score ρ_i(t)
Output Layer: Generate the final fraud probability p_i(t)
​by comparing the weighted anomaly score AS_i(W,t) against an adaptive threshold



MATHEMATICAL FRAMEWORK : 
Graph Representation
The system models transactions as a directed graph G=(V,E). To ensure real-world scalability, it utilizes a sparse adjacency matrix, reducing complexity from O(n^2) to O(kn).





LEARNABLE TOLERANCE MODEL:
The tolerance ϵ_ij(t) is not fixed but learned through gradient descent: 
ϵ_ij(t) = α_ij(t)*ϵ_i + (1−α_ij(t))*ϵ_j + Ψ_ij(t)*σ_ij
​This allows the model to adapt sensitivity based on the specific interaction between two nodes





ADAPTIVE THRESHOLDING:
The model discards static cutoffs for a dynamic threshold Θ_i(t) based on running statistics: 
Θ_i(t) = μ_i(t) + k_i(t)*σ_i(t) + log(∣N(i)∣)
μ_i(t), σ_i(t) : Learnable running mean and standard deviation of anomaly scores
k_i(t) : A learnable sensitivity parameter





ALGORITHMIC DESIGN & SCALABILITY:
Distributed Data Parallel (DDP)
For large graphs, FFD-STRA-GAT supports distributed training across P workers. It achieves near-linear speedup by partitioning the graph and performing boundary exchange of attention scores while synchronizing gradients via All-Reduce

COMPLEXITY ANALYSIS:
Time Complexity: O((n+m)d/P)
Communication Complexity: O(mbd+∥Ω∥logP)





BENCHMARKS & PERFORMANCE: 
The model was tested on a synthetic dataset (800 nodes, 5,957 edges, 13.88% fraud) and significantly outperformed standard GNN architectures

FULL METRIC COMPARISON
Model                  FFD-STRA-GAT    GCN       GraphSAGE
AUC-ROC                 0.9941        0.5432      0.6385
Avg. Precision (AP)     0.9415        0.1847      0.2418
F1 Score                0.9478        0.2483      0.2937
Precision               0.9160        0.1421      0.1850
Recall                  0.9820        0.9820      0.7117




KEY VISUALISATIONS
Training Loss: FFD-STRA-GAT converged rapidly in just 73 epochs (3.58s), while GCN and GraphSAGE plateaued at much higher loss levels
Score Separation: Score distribution plots show that FFD-STRA-GAT achieves nearly perfect separation between normal and fraud nodes, whereas baseline models show significant overlap
Confusion Matrices: The model achieved 0.99 accuracy for normal nodes and 0.98 for fraud nodes



CONFIGURATION
Input Features: 17 features (Sender, Receiver, and Edge combined)
Hidden Dimensions: 64
Layers: 3
Initialization: Xavier/Glorot for stable gradient flow

TRAINING
Example initialization of the FFD-STRA-GAT parameter set : 
                    Ω = {W, a, b, c, d, v} initialized using Xavier pattern
                    η = learning rate
repeat:
    compute forward_pass(hi, Ats_ij, Rij, rho_i, AS_i, p_i)
    compute loss L (weighted BCE + L2 regularization)
    update Ω via backpropagation
until convergence



FUTURE WORK
Make Category Mismatch (γ) and Loss Weighting (τ,ν) fully learnable parameters
Validation on real-world, non-synthetic datasets
Exploration of multi-hop propagation and richer edge feature sets

AUTHOR
Barshneyo Chakraborty Date: 20.06.202
