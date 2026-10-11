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

### Part B — Model Improvement

The second part investigates a modification to an existing baseline model. The goal is to identify a limitation of the original approach, formulate a proposed improvement, and evaluate the modification experimentally.

The analysis compares the modified model against baseline methods, examines the hypothesis motivating the change, and discusses the approach's limitations and possible extensions.

*The specific Part B modification and experimental results should be documented here once finalized.*

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
