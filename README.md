# CSC311 — Introduction to Machine Learning

Implementations and experiments for the **CSC311 Summer 2026 Final Project** at the University of Toronto.

## Overview

This project explores machine learning methods for predicting whether a student will correctly answer a mathematical diagnostic question. Using student response data from the Eedi educational platform, the task is to estimate the correctness of student–question pairs based on previously observed responses.

The dataset is represented as a sparse user–question matrix, where rows correspond to students, columns correspond to questions, and observed entries indicate whether a response was correct. Since most student–question pairs are unobserved, the problem can be approached as a collaborative filtering and matrix completion task.

The project focuses on implementing machine learning algorithms, comparing their predictive performance, and investigating modifications that may improve generalization.

## Methods

### Part A — Baseline Models

The project explores the following methods:

- **User-based k-Nearest Neighbors (kNN):** Predicts a student's response using the answers of similar students.
- **Item-based k-Nearest Neighbors:** Predicts responses using patterns of similarity between diagnostic questions.
- **Item Response Theory (IRT):** Models response probabilities using student ability and question difficulty.
- **Matrix Factorization / Neural Networks:** Learns latent representations of students and questions, or reconstructs student response vectors using an autoencoder.
- **Ensemble Learning:** Combines predictions from multiple base models using bootstrap sampling to investigate whether aggregation improves predictive performance.

## Part B — Subject-Aware Item Response Theory

The baseline one-parameter Item Response Theory (IRT) model predicts student response correctness using student ability and question difficulty. However, it does not explicitly account for differences in a student's ability across mathematical subjects.

To address this limitation, I developed **subject-aware extensions of IRT** using the hierarchical subject metadata provided with the Eedi dataset.

### 1. Subject-Specific Ability Deviations

The first extension introduces a student–subject deviation parameter, allowing each student's effective ability to vary across mathematical subjects:

$$
P(c_{ij}=1)=
\sigma\left(\theta_i+\delta_{i,k(j)}-\beta_j\right)
$$

where $\theta_i$ represents overall student ability, $\beta_j$ represents question difficulty, and $\delta_{i,k(j)}$ captures the student's ability deviation for the subject associated with question $j$.

An $L_2$ regularization penalty controls the magnitude of these subject-specific deviations. Experiments investigate different regularization strengths and hierarchy depths to study the trade-off between model flexibility and generalization.

### 2. Hierarchical Subject Modeling

Questions can be associated with subjects at multiple levels of a hierarchy. I evaluated models using a single representative subject as well as models that combine subject information across multiple hierarchy levels.

These experiments investigate whether increasingly fine-grained subject information improves prediction accuracy, and whether the additional parameters lead to overfitting when observations are sparse.

### 3. Autoencoder-Based Latent Subject Representations

A further extension replaces independent student–subject deviations with learned latent representations. Each student has a latent vector $z_i$, and each subject has an embedding $e_s$. The subject effect is modeled using their inner product:

$$
\Delta_{ij} =
\sum_{n=1}^{d} z_i^\top e_{k_n(j)}
$$

The resulting subject effect is incorporated into the IRT prediction model. An autoencoder is used to pretrain the latent representations, which are then refined through IRT optimization. Regularization is applied to the latent parameters to help control overfitting.

Experiments vary the latent dimension and the maximum subject hierarchy depth.

## Results

The baseline IRT model achieved a test accuracy of **70.70%**, outperforming the individual baseline models evaluated in Part A.

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| User-based kNN ($k=11$) | 68.90% | 68.28% |
| Item-based kNN ($k=21$) | 68.98% | 68.11% |
| Autoencoder without regularization | ~69.00% | 68.87% |
| Autoencoder with $L_2$ regularization | 67.90% | 67.70% |
| Baseline IRT | 70.60% | 70.70% |
| Subject-aware IRT, single hierarchy level | 70.94% | 70.99% |
| Hierarchical subject-aware IRT | 70.76% | 71.04% |

The best reported test accuracy was **71.04%**, achieved by the hierarchical subject-aware IRT model incorporating multiple subject levels. This represents an absolute improvement of **0.34 percentage points** over baseline IRT.

The results suggest that subject metadata can provide additional predictive information, although the gains are modest. Increasing hierarchy depth also increases model complexity and can lead to overfitting, highlighting the importance of regularization and validation-based model selection.

The autoencoder-based latent subject model did not consistently outperform the baseline IRT model. This suggests that learning compact latent subject representations was not sufficient to produce a reliable improvement on this dataset.

## Key Findings

- Implemented and compared collaborative filtering, IRT, autoencoder, and ensemble approaches for student-response prediction.
- Extended IRT with student-specific subject effects derived from hierarchical question metadata.
- Investigated the effects of hierarchy depth, latent dimension, and $L_2$ regularization on predictive performance.
- Analyzed the trade-off between subject-specific modeling capacity and generalization under sparse observations.
- Achieved a best reported test accuracy of 71.04% with hierarchical subject-aware IRT.

## Limitations and Future Work

The main limitation was data sparsity: many student–question pairs were unobserved, while deeper subject categories often had relatively few associated observations. This made subject-specific parameters difficult to estimate reliably.

Potential extensions include training on larger datasets, learning subject groupings directly from response patterns, and investigating alternative regularization or parameter-sharing strategies to improve generalization across related subjects.

## Dataset

The project uses the provided Eedi diagnostic-question dataset.

| File | Description |
|---|---|
| `train_data.csv` | Training student responses |
| `valid_data.csv` | Validation responses for model selection |
| `test_data.csv` | Held-out responses for final evaluation |
| `sparse_matrix.npz` | Sparse user–question response matrix |
| `question_meta.csv` | Question metadata, including subject identifiers |
| `student_meta.csv` | Student metadata |

Each response record contains a question ID, user ID, and binary correctness label.

**Note:** The metadata files are optional for the baseline experiments and may be used for additional feature engineering in Part B.

## Technologies

- **Language:** Python
- **Numerical computing:** NumPy, SciPy
- **Data processing:** Pandas
- **Machine learning:** PyTorch
- **Visualization:** Matplotlib
- **Evaluation:** Prediction accuracy, training objectives, and validation performance

## Project Structure

The main implementation files follow the assignment's starter-code organization:

```text
ml-project/
├── data/
│   ├── train_data.csv
│   ├── valid_data.csv
│   ├── test_data.csv
│   └── sparse_matrix.npz
├── knn.py
├── item_response.py
├── matrix_factorization.py
├── neural_network.py
├── ensemble.py
├── utils.py
└── README.md
```

The exact files present depend on the selected model implementation. Matrix factorization and neural networks are alternative choices for the corresponding Part A question.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/raiyanshadow/ml-project.git
   cd ml-project
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   On Windows:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install the required libraries:

   ```bash
   python -m pip install numpy scipy pandas matplotlib scikit-learn torch
   ```

4. Place the provided dataset files in the expected data directory. Ensure that the file paths used by the implementation match their locations.

## Running Experiments

Run each experiment using the entry point implemented in its corresponding Python file. For example, if the script defines a runnable `main()` function:

```bash
python knn.py
python item_response.py
```

For matrix factorization or neural networks, run the implementation selected for your project:

```bash
python matrix_factorization.py
```

or

```bash
python neural_network.py
```

The ensemble experiment can be run with:

```bash
python ensemble.py
```

These commands assume that each file has a runnable entry point. If an implementation requires arguments or is invoked through another script, use the corresponding configuration or entry point instead.

## Evaluation

Model selection is performed using the validation set, while the test set is reserved for final evaluation.

Experiments investigate:

- The effect of hyperparameters, such as the number of neighbors, latent dimension, learning rate, and training iterations.
- Training and validation objectives throughout optimization.
- Differences in predictive accuracy between baseline algorithms.
- Whether ensembling or the proposed Part B modification improves performance.
- Limitations arising from sparse observations, optimization, and generalization to unobserved responses.

Results, plots, selected hyperparameters, and comparisons should be documented alongside the corresponding experiments and in the final project report.

## References

- Wang, Zichao, et al. *Diagnostic Questions: The NeurIPS 2020 Education Challenge.* arXiv preprint arXiv:2007.12061, 2020.
- University of Toronto, CSC311 Summer 2026 Final Project handout.

## Disclaimer

This repository contains coursework implementations and experiments developed for educational purposes. The models are intended to explore machine learning methods for educational response prediction and should not be interpreted as validated assessments of individual students' abilities.
