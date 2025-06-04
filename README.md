# Cosine-Similarity-Guided-Adaptive-Moment-Estimation-for-Deep-Learning-Optimization




Optimization in deep neural networks remains a critical challenge, directly influencing training efficiency, convergence behavior, and generalization. First-order adaptive methods, such as Adam, are widely employed due to their computational scalability and parameter-wise learning rate adaptation. However, Adam and its variants often suffer from convergence instability in non-convex landscapes, overfitting due to aggressive adaptation, and sensitivity to hyperparameter tuning. While prior works have attempted to mitigate these issues via techniques like decoupled weight decay, momentum scheduling, or bias correction, these approaches primarily refine existing mechanisms without incorporating novel directional insights. In this work, we identify \emph{directional instability} the inability of optimizers to distinguish constructive gradient alignment from detrimental oscillations as a fundamental limitation of current adaptive methods. To address this, we propose CosAdam, a novel optimizer that introduces \emph{directional consistency monitoring} through an exponential moving average of cosine similarity between successive gradients.  At the core of CosAdam is the Directional Consistency Factor (DCF): (\( s_t = \alpha s_{t-1} + (1-\alpha) \frac{g_t \cdot g_{t-1}}{\|g_t\| \|g_{t-1}\|} \))  which adaptively modulates the step size amplifying updates during stable,  which adaptively modulates the step size amplifying updates when gradients are aligned ($s_t \rightarrow 1$) and suppressing them during noise-induced divergence ($s_t \rightarrow -1$). CosAdam maintains $\mathcal{O}(N)$ ($N$ is the total number of model parameters) time and space complexity, with minimal overhead, and integrates seamlessly with AdamW-style decoupled weight decay. Extensive evaluations on CIFAR-10, MNIST, noisy MNIST, and SST-2 benchmarks demonstrate its superiority: achieving 89.09\% accuracy on CIFAR-10 (vs. 81.15\% for Adam and 88.13\% for AdamW), 82.91\% on SST-2, and 96.64\% with a 96.61\% F1-score on noisy MNIST. Empirical and theoretical analyses confirm improved convergence and robustness in noisy and non-stationary regimes. CosAdam emerges as a robust, drop-in replacement for existing optimizers, particularly effective in scenarios demanding directional stability.




## Performance Metrics on CIFAR-10 with CosAdam

| Optimizer | Train Loss | Train Acc (%) | Val Acc (%) | Test Loss | Test Acc (%) | Precision (%) | Recall (%) | F1-Score (%) | Epoch Time (s) |
|-----------|-----------|--------------|------------|----------|-------------|--------------|-----------|-------------|---------------|
| CosAdam   | 0.1678    | 94.16        | 89.26      | 0.3525   | 89.09       | 89.12        | 89.09     | 89.04       | 46.96         |
| AdamW     | 0.1640    | 94.30        | 88.94      | 0.4029   | 88.13       | 88.61        | 88.13     | 87.88       | 40.96         |
| Adam      | 0.4753    | 84.49        | 82.22      | 0.5501   | 81.15       | 82.81        | 81.15     | 81.34       | 40.44         |
| RMSprop   | 0.6906    | 76.79        | 70.72      | 0.8581   | 69.70       | 74.47        | 69.70     | 69.99       | 39.14         |
| SGD       | 0.8780    | 68.80        | 68.04      | 0.9171   | 67.57       | 67.74        | 67.57     | 67.35       | 40.36         |
| Nadam     | 0.4187    | 86.80        | 83.52      | 0.5106   | 82.87       | 83.54        | 82.87     | 82.58       | 40.82         |

## Performance Metrics on MNIST with CosAdam

| Optimizer | Train Loss | Train Acc (%) | Val Acc (%) | Test Loss | Test Acc (%) | Precision (%) | Recall (%) | F1-Score (%) | Epoch Time (s) |
|-----------|-----------|--------------|------------|----------|-------------|--------------|-----------|-------------|---------------|
| CosAdam   | 0.0131    | 99.59        | 98.78      | 0.0310   | 99.13       | 99.13        | 99.12     | 99.13       | 14.76         |
| AdamW     | 0.0155    | 99.51        | 98.66      | 0.0280   | 99.07       | 99.06        | 99.06     | 99.06       | 13.04         |
| Adam      | 0.0730    | 97.97        | 97.92      | 0.0565   | 98.30       | 98.32        | 98.27     | 98.29       | 12.64         |
| RMSprop   | 0.0753    | 97.92        | 97.88      | 0.0554   | 98.43       | 98.42        | 98.43     | 98.42       | 12.66         |
| SGD       | 0.1342    | 96.10        | 96.64      | 0.0967   | 97.08       | 97.08        | 97.06     | 97.07       | 12.64         |
| Nadam     | 0.0725    | 97.98        | 97.96      | 0.0526   | 98.52       | 98.52        | 98.50     | 98.51       | 12.95         |

## Performance Metrics on SST-2 with CosAdam

| Optimizer | Train Loss | Train Acc (%) | Val Acc (%) | Test Loss | Test Acc (%) | Precision (%) | Recall (%) | F1-Score (%) | Epoch Time (s) |
|-----------|-----------|--------------|------------|----------|-------------|--------------|-----------|-------------|---------------|
| CosAdam   | 0.1600    | 94.19        | 91.09      | 0.6019   | 82.91       | 82.91        | 82.91     | 82.91       | 16.50         |
| AdamW     | 0.1607    | 94.28        | 90.76      | 0.6187   | 81.88       | 82.23        | 81.78     | 81.79       | 15.28         |
| Adam      | 0.6456    | 63.45        | 63.80      | 0.6398   | 65.94       | 68.72        | 65.57     | 64.30       | 15.37         |
| RMSprop   | 0.6480    | 62.99        | 63.37      | 0.6422   | 65.94       | 69.05        | 65.55     | 64.15       | 15.24         |
| SGD       | 0.6780    | 57.10        | 57.58      | 0.6810   | 54.47       | 56.64        | 53.88     | 48.85       | 14.95         |
| Nadam     | 0.6457    | 63.39        | 63.73      | 0.6399   | 66.17       | 68.84        | 65.81     | 64.61       | 15.51         |

## Performance Metrics on Noisy MNIST with CosAdam

| Optimizer | Train Loss | Train Acc (%) | Val Acc (%) | Test Loss | Test Acc (%) | Precision (%) | Recall (%) | F1-Score (%) | Epoch Time (s) |
|-----------|-----------|--------------|------------|----------|-------------|--------------|-----------|-------------|---------------|
| CosAdam   | 0.1360    | 95.64        | 95.84      | 0.1042   | 96.64       | 96.61        | 96.63     | 96.61       | 8.59          |
| AdamW     | 0.1481    | 95.33        | 96.18      | 0.1095   | 96.12       | 96.13        | 96.08     | 96.10       | 7.14          |
| Adam      | 0.1438    | 95.41        | 96.22      | 0.1124   | 96.31       | 96.29        | 96.29     | 96.28       | 7.39          |
| RMSprop   | 0.3493    | 88.92        | 90.78      | 0.2732   | 91.38       | 91.34        | 91.27     | 91.29       | 7.24          |
| SGD       | 0.1423    | 95.50        | 95.54      | 0.1046   | 96.58       | 96.56        | 96.56     | 96.55       | 7.19          |
| Nadam     | 0.1453    | 95.40        | 95.76      | 0.1078   | 96.52       | 96.51        | 96.48     | 96.49       | 7.21          |

## CosAdam Hyperparameter Combinations on MNIST

| Comb. | LR     | β₁  | β₂   | WD    | α    | c  | Test Acc (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-------|--------|-----|------|-------|------|----|--------------|---------------|------------|--------------|
| Comb1 | 0.001  | 0.9 | 0.999 | 0.01  | 0.9  | 0.5 | 99.26        | 99.26         | 99.25      | 99.25        |
| Comb2 | 0.001  | 0.95| 0.999 | 0.001 | 0.95 | 0.7 | 99.21        | 99.21         | 99.20      | 99.20        |
| Comb3 | 0.01   | 0.9 | 0.99  | 0.01  | 0.8  | 0.3 | 98.78        | 98.77         | 98.76      | 98.76        |
| Comb4 | 0.001  | 0.9 | 0.99  | 0.005 | 0.85 | 0.5 | 99.05        | 99.04         | 99.04      | 99.04        |
| Comb5 | 0.0005 | 0.8 | 0.999 | 0.0   | 0.9  | 0.6 | 98.90        | 98.90         | 98.88      | 98.89        |
