# Mathematics for AI/ML

```
math_for_ai/
├── 01-Mathematical-Foundations/
│   ├── 01-Number-Systems/
│   ├── 02-Sets-and-Logic/
│   ├── 03-Functions-and-Mappings/
│   ├── 04-Summation-and-Product-Notation/
│   ├── 05-Einstein-Summation-and-Index-Notation/
│   └── 06-Proof-Techniques/
├── 02-Linear-Algebra-Basics/
│   ├── 01-Vectors-and-Spaces/
│   ├── 02-Matrix-Operations/
│   ├── 03-Systems-of-Equations/
│   ├── 04-Determinants/
│   ├── 05-Matrix-Rank/
│   └── 06-Vector-Spaces-Subspaces/
├── 03-Advanced-Linear-Algebra/
│   ├── 01-Eigenvalues-and-Eigenvectors/
│   ├── 02-Singular-Value-Decomposition/
│   ├── 03-Principal-Component-Analysis/
│   ├── 04-Linear-Transformations/
│   ├── 05-Orthogonality-and-Orthonormality/
│   ├── 06-Matrix-Norms/
│   ├── 07-Positive-Definite-Matrices/
│   └── 08-Matrix-Decompositions/
├── 04-Calculus-Fundamentals/
│   ├── 01-Limits-and-Continuity/
│   ├── 02-Derivatives-and-Differentiation/
│   ├── 03-Integration/
│   └── 04-Series-and-Sequences/
├── 05-Multivariate-Calculus/
│   ├── 01-Partial-Derivatives-and-Gradients/
│   ├── 02-Jacobians-and-Hessians/
│   ├── 03-Chain-Rule-and-Backpropagation/
│   ├── 04-Optimization-Theory/
│   └── 05-Automatic-Differentiation/
├── 06-Probability-Theory/
│   ├── 01-Introduction-and-Random-Variables/
│   ├── 02-Common-Distributions/
│   ├── 03-Joint-Distributions/
│   ├── 04-Expectation-and-Moments/
│   ├── 05-Concentration-Inequalities/
│   └── 06-Stochastic-Processes/
├── 07-Statistics/
│   ├── 01-Descriptive-Statistics/
│   ├── 02-Estimation-Theory/
│   ├── 03-Hypothesis-Testing/
│   └── 04-Bayesian-Inference/
├── 08-Optimization/
│   ├── 01-Convex-Optimization/
│   ├── 02-Gradient-Descent/
│   ├── 03-Second-Order-Methods/
│   ├── 04-Constrained-Optimization/
│   ├── 05-Stochastic-Optimization/
│   ├── 06-Optimization-Landscape/
│   ├── 07-Adaptive-Learning-Rate/
│   ├── 08-Regularization-Methods/
│   ├── 09-Hyperparameter-Optimization/
│   └── 10-Learning-Rate-Schedules/
├── 09-Information-Theory/
│   ├── 01-Entropy/
│   ├── 02-KL-Divergence/
│   ├── 03-Mutual-Information/
│   ├── 04-Cross-Entropy/
│   └── 05-Fisher-Information/
├── 10-Numerical-Methods/
│   ├── 01-Floating-Point-Arithmetic/
│   ├── 02-Numerical-Linear-Algebra/
│   ├── 03-Numerical-Optimization/
│   ├── 04-Interpolation-and-Approximation/
│   └── 05-Numerical-Integration/
├── 11-Graph-Theory/
│   ├── 01-Graph-Basics/
│   ├── 02-Graph-Representations/
│   ├── 03-Graph-Algorithms/
│   ├── 04-Spectral-Graph-Theory/
│   └── 05-Graph-Neural-Networks/
├── 12-Functional-Analysis/
│   ├── 01-Vector-Spaces/
│   ├── 02-Normed-Spaces/
│   ├── 03-Hilbert-Spaces/
│   └── 04-Kernel-Methods/
├── 13-ML-Specific-Math/
│   ├── 01-Loss-Functions/
│   ├── 02-Activation-Functions/
│   ├── 03-Attention-Mechanisms/
│   ├── 04-Normalization-Techniques/
│   └── 05-Sampling-Methods/
├── 14-Math-for-Specific-Models/
│   ├── 01-Linear-Models/
│   ├── 02-Neural-Networks/
│   ├── 03-Probabilistic-Models/
│   ├── 04-RNN-and-LSTM-Math/
│   ├── 05-Transformer-Architecture/
│   ├── 06-Reinforcement-Learning/
│   ├── 07-Generative-Models/
│   └── 08-CNN-and-Convolution-Math/
├── 15-Math-for-LLMs/
│   ├── 01-Tokenization-Math/
│   ├── 02-Embedding-Space-Math/
│   ├── 03-Attention-Mechanism-Math/
│   ├── 04-Positional-Encodings/
│   ├── 05-Language-Model-Probability/
│   ├── 06-Training-at-Scale/
│   ├── 07-Fine-Tuning-Math/
│   ├── 08-Scaling-Laws/
│   ├── 09-Efficient-Attention-and-Inference/
│   ├── 10-Mixture-of-Experts-and-Routing/
│   ├── 11-Quantization-and-Distillation/
│   ├── 12-RAG-Math-and-Retrieval/
│   └── 13-Serving-and-Systems-Tradeoffs/
├── 16-LLM-Training-Data-Pipeline/
│   ├── 01-Data-Format-Standards/
│   ├── 02-JSONL-Generation/
│   ├── 03-Quality-Checks/
│   ├── 04-Full-Dataset-Assembly/
│   ├── 05-Contamination-and-Dedup-Audits/
│   ├── 06-Documentation-and-Governance/
│   └── 07-Data-Mixture-Optimization/
├── 17-Evaluation-and-Reliability/
│   ├── 01-Capability-Benchmarks/
│   ├── 02-Calibration-and-Uncertainty/
│   ├── 03-Robustness-and-Distribution-Shift/
│   ├── 04-Error-Analysis-and-Ablations/
│   └── 05-Online-Experimentation-and-AB-Testing/
├── 18-Alignment-and-Safety/
│   ├── 01-Instruction-Tuning-and-SFT/
│   ├── 02-Preference-Optimization-RLHF-and-DPO/
│   ├── 03-Red-Teaming-and-Safety-Evaluations/
│   ├── 04-Policy-and-Guardrails/
│   └── 05-Human-in-the-Loop-and-Monitoring/
├── 19-Production-ML-and-MLOps/
│   ├── 01-Data-Versioning-and-Lineage/
│   ├── 02-Experiment-Tracking-and-Reproducibility/
│   ├── 03-Feature-Stores-and-Data-Contracts/
│   ├── 04-Model-Serving-and-Inference-Optimization/
│   ├── 05-Monitoring-Drift-and-Retraining/
│   └── 06-LLM-Evaluation-Observability-and-Guardrails/
└── docs/
```
