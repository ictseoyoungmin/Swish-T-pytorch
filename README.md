# Swish-T Family Implementation in PyTorch

A PyTorch Implementation of Swish-T : Enhancing Swish Activation with Tanh Bias for Improved Neural Network Performance.

Swish-T family : ( Swish-T, Swish-T<sub><strong>A</strong></sub>, Swish-T<sub><strong>B</strong></sub>, Swish-T<sub><strong>C</strong></sub> )

## Introduction

In this work, we propose the Swish-T family, which enhances the Swish activation function by introducing an adaptive bias concept based on the input value $x$. The Swish-T family consists of the following variants:

- **Swish-T**: This variant incorporates a tanh bias, offering superior performance but generally exhibits slower learning speeds compared to the standard Swish function.
- **Swish-T<sub><strong>A</strong></sub>**: Simplifies the formula for faster learning speeds, making it efficient for applications requiring quick training times.
- **Swish-T<sub><strong>B</strong></sub>**: Reintroduces the $\beta$ parameter, providing better performance metrics by allowing fine-tuning of the activation's shape.

- **Swish-T<sub><strong>C</strong></sub>**: Effectively controls the bias using the $\beta$ parameter, achieving stable performance across different tasks.

![Swish-T_Cproperty](figs/plot_SiwshT_C.png)

## Comparison of the Swish-T family and various activation functions across different architectures on the CIFAR-10/100 dataset

The table shows the mean Top-1 accuracy and standard deviation of 5 runs.

| **Activation Function** | **RN-18**          | **SF-V2 (1.x)**     | **SF-V2 (2.x)**     | **SENet-18**        | **EN-B0**          | **MN-V2**          | **DN-121**         |
|-------------------------|--------------------|---------------------|---------------------|---------------------|--------------------|--------------------|--------------------|
| **#Params**             | 11.1M              | 1.4M                | 5.5M                | 11.2M               | 3.6M               | 2.3M               | 1.0M               |
| **ReLU**                | 95.54 ± 0.08       | 91.90 ± 0.24        | 91.93 ± 0.21        | 95.25 ± 0.32        | 90.73 ± 0.13       | 92.63 ± 0.19       | 94.82 ± 0.16       |
| **GELU**                | 94.79 ± 0.16       | 93.06 ± 0.13        | 93.11 ± 0.26        | 94.68 ± 0.10        | 92.14 ± 0.15       | 93.72 ± 0.17       | 93.89 ± 0.21       |
| **SiLU**                | 94.22 ± 0.17       | 92.03 ± 0.11        | 92.34 ± 0.19        | 93.99 ± 0.04        | 91.19 ± 0.15       | 92.59 ± 0.16       | 92.75 ± 0.17       |
| **Swish**               | 95.49 ± 0.07       | 94.06 ± 0.15        | 94.01 ± 0.21        | 95.41 ± 0.14        | **93.46 ± 0.13**   | 95.08 ± 0.12       | 94.89 ± 0.11       |
| **Mish**                | 94.36 ± 0.12       | 92.34 ± 0.14        | 92.41 ± 0.08        | 94.09 ± 0.13        | 91.58 ± 0.14       | 92.81 ± 0.21       | 92.78 ± 0.19       |
| **SMU**                 | **95.58 ± 0.03**   | 94.00 ± 0.17        | 94.01 ± 0.12        | 95.48 ± 0.07        | 93.36 ± 0.10       | **95.12 ± 0.11**   | **94.95 ± 0.20**   |
| **SMU-1**               | 95.12 ± 0.16       | 93.94 ± 0.19        | 93.93 ± 0.14        | 94.87 ± 0.15        | 92.98 ± 0.15       | 94.66 ± 0.13       | 94.42 ± 0.18       |
| **Swish-T**             | 95.53 ± 0.14       | 94.15 ± 0.14        | 94.08 ± 0.21        | 95.40 ± 0.13        | 93.22 ± 0.13       | 94.89 ± 0.12       | 94.88 ± 0.18       |
| **Swish-T<sub><strong>B</strong></sub>** | 95.17 ± 0.21       | 93.99 ± 0.18        | 93.91 ± 0.14        | 95.32 ± 0.04        | 93.04 ± 0.28       | 94.72 ± 0.20       | 94.71 ± 0.14       |
| **Swish-T<sub><strong>C</strong></sub>** | 95.29 ± 0.03       | **94.26 ± 0.08**    | **94.27 ± 0.14**    | **95.50 ± 0.10**    | 93.28 ± 0.19       | 94.97 ± 0.14       | **94.95 ± 0.09**   |

| **Activation Function** | **RN-18**          | **SF-V2 (1.x)**     | **SF-V2 (2.x)**     | **SENet-18**        | **EN-B0**          | **MN-V2**          | **DN-121**         |
|-------------------------|--------------------|---------------------|---------------------|---------------------|--------------------|--------------------|--------------------|
| **#Params**             | 11.1M              | 1.4M                | 5.5M                | 11.2M               | 3.6M               | 2.3M               | 1.0M               |
| **ReLU**                | 78.46 ± 0.15       | 71.70 ± 0.41        | 71.92 ± 0.45        | 77.85 ± 0.28        | 67.71 ± 0.49       | 72.65 ± 0.24       | 76.55 ± 0.39       |
| **GELU**                | 78.08 ± 0.26       | 75.04 ± 0.21        | 74.83 ± 0.17        | 76.92 ± 0.14        | 71.93 ± 0.34       | 75.83 ± 0.30       | 74.41 ± 0.32       |
| **SiLU**                | 76.82 ± 0.24       | 73.19 ± 0.36        | 73.71 ± 0.39        | 75.59 ± 0.15        | 70.17 ± 0.35       | 73.49 ± 0.35       | 73.40 ± 0.21       |
| **Swish**               | 78.60 ± 0.28       | 75.69 ± 0.17        | 75.73 ± 0.19        | 78.18 ± 0.20        | 72.92 ± 0.18       | **77.97 ± 0.16**   | 77.00 ± 0.35       |
| **Mish**                | 77.16 ± 0.24       | 73.58 ± 0.28        | 73.45 ± 0.20        | 75.74 ± 0.35        | 71.30 ± 0.09       | 73.39 ± 0.23       | 73.81 ± 0.27       |
| **SMU**                 | 78.85 ± 0.19       | 75.36 ± 0.46        | 75.43 ± 0.20        | **78.28 ± 0.18**    | 72.49 ± 0.19       | 77.84 ± 0.16       | 76.89 ± 0.22       |
| **SMU-1**               | 78.44 ± 0.26       | 76.01 ± 0.20        | 75.90 ± 0.23        | 76.71 ± 0.26        | 72.48 ± 0.11       | 77.46 ± 0.14       | 76.04 ± 0.10       |
| **Swish-T**             | **79.02 ± 0.24**   | 75.79 ± 0.23        | **76.04 ± 0.30**    | 78.15 ± 0.29        | 72.76 ± 0.43       | 77.31 ± 0.14       | 77.00 ± 0.14       |
| **Swish-T<sub><strong>B</strong></sub>** | 77.24 ± 0.99       | 75.84 ± 0.29        | 75.82 ± 0.29        | 77.80 ± 0.31        | **73.21 ± 0.19**   | 77.40 ± 0.13       | 76.96 ± 0.12       |
| **Swish-T<sub><strong>C</strong></sub>** | 78.72 ± 0.15       | **76.06 ± 0.36**    | **76.04 ± 0.30**    | 77.93 ± 0.20        | 72.75 ± 0.30       | 77.60 ± 0.26       | **77.15 ± 0.19**   |


## arXiv 

https://arxiv.org/abs/2407.01012

## Citation
```
@misc{seo2024swishtenhancingswish,
      title={Swish-T : Enhancing Swish Activation with Tanh Bias for Improved Neural Network Performance}, 
      author={Youngmin Seo and Jinha Kim and Unsang Park},
      year={2024},
      eprint={2407.01012},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.01012}, 
}
```
