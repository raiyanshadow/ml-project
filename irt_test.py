import argparse
import copy
import json
import os
import matplotlib.pyplot as plt

import numpy as np

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_subject_data,
    load_subject_names,
)


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))

def build_subject_index(subject_data, depth):
    """
    Build question -> subject mapping using a hierarchy depth.

    :param subject_data: A dictionary mapping question IDs to their subject lists.
    :param depth: int
    :return: A tuple of (unique_subjects_list, q2s) where q2s is a dictionary mapping question IDs to subject indices.
    """

    unique_subjects = set()

    for _, subj_list in subject_data.items():
        max_depth = min(depth, len(subj_list) - 1)
        unique_subjects.update(subj_list[1:max_depth + 1])

    unique_subjects_list = sorted(unique_subjects)

    subject_to_index = {
        s: i for i, s in enumerate(unique_subjects_list)
    }

    q2s = {}

    for qid, subj_list in subject_data.items():
        max_depth = min(depth, len(subj_list) - 1)
        q2s[qid] = [
            subject_to_index[s]
            for s in subj_list[1:max_depth + 1]
        ]

    return unique_subjects_list, q2s


def subject_training_counts(train_data, q2s, num_subjects):
    """
    Count the number of training examples per subject.
    :param train_data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param q2s: A dictionary mapping question IDs to subject indices.
    :param num_subjects: int
    :return: A numpy array of counts per subject.
    """
    counts = np.zeros(num_subjects)
    for j in train_data["question_id"]:
        for s in q2s[j]:
            counts[s] += 1
    return counts

def baseline_neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.0
    for i, j, c in zip(data["user_id"], data["question_id"], data["is_correct"]):
        if not np.isnan(c):
            diff = theta[i] - beta[j]
            log_lklihood += c * diff - np.logaddexp(0, diff)
    return -log_lklihood


def baseline_update(user_data, question_data, lr, theta, beta):
    """Update theta and beta using gradient descent.
    
    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param user_data: A dictionary, takes keys as values in data[user_id], and value
    is list of non-NaN entries in data as a tuple of corresponding (data[question_id], data[is_correct])
    of the given user_id
    :param question_data: A dictionary, similar to user_data except with keys as values in data[question_id],
    and value is list of non-NaN entries in data as a tuple of corresponding (data[user_id], data[is_correct])
    of the given question_id
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    for i in range(len(theta)):
        grad = 0.0
        if i in user_data:
            for j, c in user_data[i]:
                diff = theta[i] - beta[j]
                grad += c - sigmoid(diff)
        theta[i] += lr * grad

    for j in range(len(beta)):
        grad = 0.0
        if j in question_data:
            for i, c in question_data[j]:
                diff = theta[i] - beta[j]
                grad += sigmoid(diff) - c
        beta[j] += lr * grad

    return theta, beta


def baseline_evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = theta[u] - beta[q]
        pred.append(sigmoid(x) >= 0.5)
    return np.sum(data["is_correct"] == np.array(pred)) / len(data["is_correct"])


def train_baseline(train_data, val_data, lr, iterations):
    """Train IRT model.
    
    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    unique_users = max(train_data["user_id"]) + 1
    unique_questions = max(train_data["question_id"]) + 1
    theta = np.zeros(unique_users)
    beta = np.zeros(unique_questions)

    user_data, question_data = {}, {}
    train_ll = []
    val_ll = []
    for i, j, c in zip(train_data["user_id"], train_data["question_id"], train_data["is_correct"]):
        if not np.isnan(c):
            user_data.setdefault(i, []).append((j, c))
            question_data.setdefault(j, []).append((i, c))

    for _ in range(iterations):
        train_ll.append(
            baseline_neg_log_likelihood(train_data, theta, beta)
        )
        val_ll.append(
            baseline_neg_log_likelihood(val_data, theta, beta)
        )
        theta, beta = baseline_update(user_data, question_data, lr, theta, beta)

    return theta, beta, train_ll, val_ll


def extended_neg_log_likelihood(data, theta, beta, delta, q2s):
    """Compute the negative log-likelihood.
    
    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param delta: Matrix
    :param q2s: A dictionary mapping question IDs to subject indices.
    :return: float
    """
    ll = 0.0

    for u, q, c in zip(
        data["user_id"],
        data["question_id"],
        data["is_correct"],
    ):
        subject_effect = np.sum(delta[u, q2s[q]])
        diff = theta[u] + subject_effect - beta[q]
        ll += c * diff - np.logaddexp(0, diff)

    return -ll

def extended_update(user_data, question_data, lr, theta, beta, delta, q2s, lamb):
    """Update theta and beta using gradient descent.
        
    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta
        delta <- new_delta

    You may optionally replace the function arguments to receive a matrix.

    :param user_data: A dictionary, takes keys as values in data[user_id], and value
    is list of non-NaN entries in data as a tuple of corresponding (data[question_id], data[is_correct])
    of the given user_id
    :param question_data: A dictionary, similar to user_data except with keys as values in data[question_id],
    and value is list of non-NaN entries in data as a tuple of corresponding (data[user_id], data[is_correct])
    of the given question_id
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param delta: Matrix
    :param q2s: A dictionary mapping question IDs to subject indices.
    :param lamb: float
    :return: tuple of vectors
    """
    for i in range(len(theta)):
        grad = 0.0
        if i in user_data:
            for j, c in user_data[i]:
                diff = theta[i] + np.sum(delta[i, q2s[j]]) - beta[j]
                grad += c - sigmoid(diff)
        theta[i] += lr * grad

    for j in range(len(beta)):
        grad = 0.0
        if j in question_data:
            for i, c in question_data[j]:
                diff = theta[i] + np.sum(delta[i, q2s[j]]) - beta[j]
                grad += sigmoid(diff) - c
        beta[j] += lr * grad

    delta_grad = np.zeros_like(delta)
    for i in range(len(theta)):
        if i in user_data:
            for j, c in user_data[i]:
                subjects = q2s[j]
                diff = theta[i] + np.sum(delta[i, subjects]) - beta[j]

                grad = c - sigmoid(diff)

                for s in subjects:
                    delta_grad[i, s] += grad

    delta += lr * (delta_grad - lamb * delta)

    return theta, beta, delta

def evaluate_by_subject(data, theta, beta, delta, q2s, unique_subjects_list, subject_names):
    """Evaluate the model given data and return the accuracy.
    Will also return the sample size (n) for each subject, as well as 
    the subject name for easier interpretation.
    
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param delta: Matrix
    :param q2s: A dictionary mapping question IDs to subject indices.
    :param unique_subjects_list: A list of unique subject IDs.
    :param subject_names: A dictionary mapping subject IDs to names.
    :return: float
    """

    subject_correct = np.zeros(len(unique_subjects_list))
    subject_total = np.zeros(len(unique_subjects_list))

    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        subjects = q2s[q]

        x = theta[u] + np.sum(delta[u, subjects]) - beta[q]
        pred = sigmoid(x) >= 0.5

        for s in subjects:
            subject_correct[s] += (pred == data["is_correct"][i])
            subject_total[s] += 1

    results = {}

    for s in range(len(unique_subjects_list)):
        if subject_total[s] > 0:
            subject_id = unique_subjects_list[s]
            results[s] = {
                "subject_id": subject_id,
                "subject_name": subject_names.get(
                    int(subject_id),
                    "Unknown"
                ),
                "accuracy": subject_correct[s] / subject_total[s],
                "n": int(subject_total[s])
            }

    return results

def print_best_worst(scores, name, min_examples=50):
    """Print the best and worst performing subjects based on accuracy, filtering out subjects with fewer than min_examples.

    :param scores: A dictionary of subject scores, where each key is a subject index and each value is a dictionary containing 'accuracy' and 'n'.
    :param name: str
    :param min_examples: int
    :return: None
    """

    filtered = {
        k:v for k,v in scores.items()
        if v["n"] >= min_examples
    }

    best = max(
        filtered.items(),
        key=lambda x: x[1]["accuracy"]
    )

    worst = min(
        filtered.items(),
        key=lambda x: x[1]["accuracy"]
    )

    print(f"\n{name}")

    print(
        "Best:",
        best[1]["subject_name"],
        f"(id={best[1]['subject_id']})",
        f"accuracy={best[1]['accuracy']:.4f}",
        f"n={best[1]['n']}"
    )

    print(
        "Worst:",
        worst[1]["subject_name"],
        f"(id={worst[1]['subject_id']})",
        f"accuracy={worst[1]['accuracy']:.4f}",
        f"n={worst[1]['n']}"
    )

def print_subject_ranking(scores, name, top_k=10, min_examples=50):
    """Print the k top and bottom performing subjects based on accuracy, filtering out subjects with fewer than min_examples.
    
    :param scores: A dictionary of subject scores, where each key is a subject index and each value is a dictionary containing 'accuracy' and 'n'.
    :param name: str
    :param top_k: int
    :param min_examples: int
    :return: None
    """
    filtered = {
        k:v for k,v in scores.items()
        if v["n"] >= min_examples
    }

    ranked = sorted(
        filtered.items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True
    )

    print(f"\n{name} TOP {top_k}")

    for _, s in ranked[:top_k]:
        print(
            f"{s['subject_name']:35s} "
            f"{s['accuracy']:.4f} "
            f"(n={s['n']})"
        )


    print(f"\n{name} BOTTOM {top_k}")

    for _, s in ranked[-top_k:]:
        print(
            f"{s['subject_name']:35s} "
            f"{s['accuracy']:.4f} "
            f"(n={s['n']})"
        )


def extended_evaluate_stratified(data, theta, beta, delta, q2s, common_subjects, rare_subjects):
    """Return overall accuracy plus accuracy restricted to 'common' vs
    'rare' subject indices (as determined by training-set frequency).
    Also returns the sample size (n) for each bucket, since accuracy on
    a handful of examples is unreliable and should be reported alongside
    the number, not on its own.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param delta: Matrix
    :param q2s: A dictionary mapping question IDs to subject indices.
    :param common_subjects: A set of subject indices considered 'common'.
    :param rare_subjects: A set of subject indices considered 'rare'.
    :return: A dictionary with overall, common, and rare accuracies and sample sizes.
    """
    correct_overall = correct_common = correct_rare = 0
    total_overall = total_common = total_rare = 0

    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        subjects = q2s[q]
        x = theta[u] + np.sum(delta[u, subjects]) - beta[q]
        pred = sigmoid(x) >= 0.5
        is_correct = pred == data["is_correct"][i]

        correct_overall += is_correct
        total_overall += 1

        if any(s in common_subjects for s in subjects):
            correct_common += is_correct
            total_common += 1

        if any(s in rare_subjects for s in subjects):
            correct_rare += is_correct
            total_rare += 1

    return {
        "overall": correct_overall / total_overall,
        "common": correct_common / total_common if total_common else float("nan"),
        "rare": correct_rare / total_rare if total_rare else float("nan"),
        "n_overall": total_overall,
        "n_common": total_common,
        "n_rare": total_rare,
    }


def train_extended(train_data, val_data, lr, iterations, lamb, q2s, num_subjects, seed=311):
    """Train IRT model with extended parameters (theta, beta, delta).

    You may optionally replace the function arguments to receive a matrix.
    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :param lamb: float
    :param q2s: A dictionary mapping question IDs to subject indices.
    :param num_subjects: int
    :param seed: int
    :return: (theta, beta, delta, train_ll, val_ll)
    """
    np.random.seed(seed)
    unique_users = max(train_data["user_id"]) + 1
    unique_questions = max(train_data["question_id"]) + 1

    theta = np.zeros(unique_users)
    beta = np.zeros(unique_questions)
    delta = np.zeros((unique_users, num_subjects))

    user_data, question_data = {}, {}
    train_ll = []
    val_ll = []
    for i, j, c in zip(train_data["user_id"], train_data["question_id"], train_data["is_correct"]):
        if not np.isnan(c):
            user_data.setdefault(i, []).append((j, c))
            question_data.setdefault(j, []).append((i, c))

    for _ in range(iterations):
        train_ll.append(
            extended_neg_log_likelihood(
                train_data,
                theta,
                beta,
                delta,
                q2s,
            )
        )

        val_ll.append(
            extended_neg_log_likelihood(
                val_data,
                theta,
                beta,
                delta,
                q2s,
            )
        )
        theta, beta, delta = extended_update(
            user_data, question_data, lr, theta, beta, delta, q2s, lamb
        )

    return theta, beta, delta, train_ll, val_ll

# SWEEP AND FIXED HYPERPARAMETERS
DEPTH_SWEEP = [1, 2, 3, 4, 5, 6, 7, 8, 9]
LAMBDA_SWEEP = [0, 0.01, 0.05, 0.2, 1, 5, 10, 20, 50]
DEPTH_FIXED = 1
LAMBDA_FIXED = 20

def lambda_run(data_dir="./data", lr=0.01, iterations=50, rare_threshold=1000, out_dir="."):
    """Run a sweep over lambda values for the extended IRT model, keeping depth fixed.
    
    :param data_dir: str
    :param lr: float
    :param iterations: int
    :param rare_threshold: int
    :param out_dir: str
    :return: A dictionary of results for each lambda value.
    """
    train_data = load_train_csv(data_dir)
    val_data = load_valid_csv(data_dir)
    test_data = load_public_test_csv(data_dir)
    subject_data = load_subject_data(data_dir)
    subject_names = load_subject_names(os.path.join(data_dir, "subject_meta.csv"))

    results = {}

    # Base IRT (no delta)
    print("\n=== Baseline IRT (theta, beta only) ===")
    theta_b, beta_b, _, _ = train_baseline(train_data, val_data, lr, iterations)
    train_acc_b = baseline_evaluate(train_data, theta_b, beta_b)
    val_acc_b = baseline_evaluate(val_data, theta_b, beta_b)
    test_acc_b = baseline_evaluate(test_data, theta_b, beta_b)
    print("Train: {:.4f}  Val: {:.4f}  Test: {:.4f}".format(train_acc_b, val_acc_b, test_acc_b))
    results["baseline"] = {
        "lamb": None,
        "train_overall": train_acc_b,
        "val_overall": val_acc_b,
        "test_overall": test_acc_b,
        "train_common": None, "train_rare": None,
        "test_common": None, "test_rare": None,
    }

    # Subject index is fixed at DEPTH_FIXED for the whole sweep
    unique_subjects_list, q2s = build_subject_index(
        subject_data,
        DEPTH_FIXED
    )

    num_subjects = len(unique_subjects_list)

    counts = subject_training_counts(train_data, q2s, num_subjects)
    common_subjects = set(np.where(counts >= rare_threshold)[0])
    rare_subjects = set(np.where(counts < rare_threshold)[0])

    # Report n for common/rare in each split up front, since accuracy on a
    # handful of examples (e.g. test_rare) is only meaningful alongside its
    # sample size. This is the same across all lambdas since depth/subject
    # index is fixed for the whole sweep.
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        n_common = sum(
            1 for q in split_data["question_id"]
            if any(s in common_subjects for s in q2s[q])
        )

        n_rare = sum(
            1 for q in split_data["question_id"]
            if any(s in rare_subjects for s in q2s[q])
        )
        print("{} split: n_common={}, n_rare={}".format(split_name, n_common, n_rare))

    # Extended model, lambda sweep
    for lamb in LAMBDA_SWEEP:
        print("\n==============================")
        print(f"LAMBDA {lamb}")
        print("==============================")

        theta, beta, delta, _, _ = train_extended(
            train_data,
            val_data,
            lr,
            iterations,
            lamb,
            q2s,
            num_subjects
        )

        train_acc = extended_evaluate_stratified(
            train_data,
            theta,
            beta,
            delta,
            q2s,
            common_subjects,
            rare_subjects
        )

        val_acc = extended_evaluate_stratified(
            val_data,
            theta,
            beta,
            delta,
            q2s,
            common_subjects,
            rare_subjects
        )

        test_acc = extended_evaluate_stratified(
            test_data,
            theta,
            beta,
            delta,
            q2s,
            common_subjects,
            rare_subjects
        )

        train_subject_scores = evaluate_by_subject(
            train_data,
            theta,
            beta,
            delta,
            q2s,
            unique_subjects_list,
            subject_names
        )

        val_subject_scores = evaluate_by_subject(
            val_data,
            theta,
            beta,
            delta,
            q2s,
            unique_subjects_list,
            subject_names
        )

        test_subject_scores = evaluate_by_subject(
            test_data,
            theta,
            beta,
            delta,
            q2s,
            unique_subjects_list,
            subject_names
        )

        print(
            f"Lambda {lamb}"
        )

        print(
            f"Subjects: {num_subjects}"
        )

        print(
            "Train:",
            train_acc["overall"]
        )

        print(
            "Validation:",
            val_acc["overall"]
        )

        print(
            "Test:",
            test_acc["overall"]
        )

        print_best_worst(
            train_subject_scores,
            f"LAMBDA {lamb} TRAIN"
        )

        print_best_worst(
            val_subject_scores,
            f"LAMBDA {lamb} VALIDATION"
        )

        print_best_worst(
            test_subject_scores,
            f"LAMBDA {lamb} TEST"
        )

        print_subject_ranking(
            test_subject_scores,
            f"LAMBDA {lamb} TEST"
        )

        n_test_common = sum(
            1 for q in test_data["question_id"]
            if any(s in common_subjects for s in q2s[q])
        )

        n_test_rare = sum(
            1 for q in test_data["question_id"]
            if any(s in rare_subjects for s in q2s[q])
        )

        results[f"lambda_{lamb}"] = {
            "lamb": lamb,
            "num_subjects": num_subjects,
            "train_overall": train_acc["overall"],
            "val_overall": val_acc["overall"],
            "test_overall": test_acc["overall"],
            "train_common": train_acc["common"],
            "train_rare": train_acc["rare"],
            "test_common": test_acc["common"],
            "test_rare": test_acc["rare"],
            "n_test_common": n_test_common,
            "n_test_rare": n_test_rare,
            "subject_scores": {
                "train": train_subject_scores,
                "val": val_subject_scores,
                "test": test_subject_scores,
            }
        }

    # Save results
    json_path = os.path.join(out_dir, "irt_lambda_sweep.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    print("\nSaved results to {}".format(json_path))

    csv_path = os.path.join(out_dir, "irt_lambda_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("config,lamb,train_overall,val_overall,test_overall,train_common,train_rare,test_common,test_rare,n_test_common,n_test_rare\n")
        for name, r in results.items():
            f.write("{},{},{:.4f},{:.4f},{:.4f},{},{},{},{},{},{}\n".format(
                name, r["lamb"], r["train_overall"], r["val_overall"], r["test_overall"],
                "" if r["train_common"] is None else "{:.4f}".format(r["train_common"]),
                "" if r["train_rare"] is None else "{:.4f}".format(r["train_rare"]),
                "" if r["test_common"] is None else "{:.4f}".format(r["test_common"]),
                "" if r["test_rare"] is None else "{:.4f}".format(r["test_rare"]),
                r.get("n_test_common", ""), r.get("n_test_rare", ""),
            ))
    print("Saved results to {}".format(csv_path))

    return results


def depth_run(data_dir="./data", lr=0.01, iterations=50, rare_threshold=1000, out_dir="."):
    """Run a sweep over depth values for the extended IRT model, keeping lambda fixed.
    
    :param data_dir: str
    :param lr: float
    :param iterations: int
    :param rare_threshold: int
    :param out_dir: str
    :return: A dictionary of results for each depth value.
    """
    train_data = load_train_csv(data_dir)
    val_data = load_valid_csv(data_dir)
    test_data = load_public_test_csv(data_dir)
    subject_data = load_subject_data(data_dir)
    subject_names = load_subject_names(os.path.join(data_dir, "subject_meta.csv"))

    results = {}

    # Base IRT (no delta)
    print("\n=== Baseline IRT (theta, beta only) ===")
    theta_b, beta_b, _, _ = train_baseline(train_data, val_data, lr, iterations)
    train_acc_b = baseline_evaluate(train_data, theta_b, beta_b)
    val_acc_b = baseline_evaluate(val_data, theta_b, beta_b)
    test_acc_b = baseline_evaluate(test_data, theta_b, beta_b)
    print("Train: {:.4f}  Val: {:.4f}  Test: {:.4f}".format(train_acc_b, val_acc_b, test_acc_b))
    results["baseline"] = {
        "depth": None,
        "train_overall": train_acc_b,
        "val_overall": val_acc_b,
        "test_overall": test_acc_b,
        "train_common": None, "train_rare": None,
        "test_common": None, "test_rare": None,
    }

    # Extended model, lambda sweep
    for depth in DEPTH_SWEEP:
        print("\n==============================")
        print(f"DEPTH {depth}")
        print("==============================")

        unique_subjects_list, q2s = build_subject_index(
            subject_data,
            depth
        )

        num_subjects = len(unique_subjects_list)

        counts = subject_training_counts(train_data, q2s, num_subjects)
        common_subjects = set(np.where(counts >= rare_threshold)[0])
        rare_subjects = set(np.where(counts < rare_threshold)[0])

        # Report n for common/rare in each split up front, since accuracy on a
        # handful of examples (e.g. test_rare) is only meaningful alongside its
        # sample size.
        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            n_common = sum(
                1 for q in split_data["question_id"]
                if any(s in common_subjects for s in q2s[q])
            )

            n_rare = sum(
                1 for q in split_data["question_id"]
                if any(s in rare_subjects for s in q2s[q])
            )
            print("{} split: n_common={}, n_rare={}".format(split_name, n_common, n_rare))

        theta, beta, delta, _, _ = train_extended(
            train_data,
            val_data,
            lr,
            iterations,
            LAMBDA_FIXED,
            q2s,
            num_subjects
        )


        train_acc = extended_evaluate_stratified(
            train_data,
            theta,
            beta,
            delta,
            q2s,
            common_subjects,
            rare_subjects
        )

        val_acc = extended_evaluate_stratified(
            val_data,
            theta,
            beta,
            delta,
            q2s,
            common_subjects,
            rare_subjects
        )

        test_acc = extended_evaluate_stratified(
            test_data,
            theta,
            beta,
            delta,
            q2s,
            common_subjects,
            rare_subjects
        )

        train_subject_scores = evaluate_by_subject(
            train_data,
            theta,
            beta,
            delta,
            q2s,
            unique_subjects_list,
            subject_names
        )

        val_subject_scores = evaluate_by_subject(
            val_data,
            theta,
            beta,
            delta,
            q2s,
            unique_subjects_list,
            subject_names
        )

        test_subject_scores = evaluate_by_subject(
            test_data,
            theta,
            beta,
            delta,
            q2s,
            unique_subjects_list,
            subject_names
        )


        print(
            f"Depth {depth}"
        )

        print(
            f"Subjects: {num_subjects}"
        )

        print(
            "Train:",
            train_acc["overall"]
        )

        print(
            "Validation:",
            val_acc["overall"]
        )

        print(
            "Test:",
            test_acc["overall"]
        )

        print_best_worst(
            train_subject_scores,
            f"DEPTH {depth} TRAIN"
        )

        print_best_worst(
            val_subject_scores,
            f"DEPTH {depth} VALIDATION"
        )

        print_best_worst(
            test_subject_scores,
            f"DEPTH {depth} TEST"
        )

        print_subject_ranking(
            test_subject_scores,
            f"DEPTH {depth} TEST"
        )

        n_test_common = sum(
            1 for q in test_data["question_id"]
            if any(s in common_subjects for s in q2s[q])
        )

        n_test_rare = sum(
            1 for q in test_data["question_id"]
            if any(s in rare_subjects for s in q2s[q])
        )


        results[f"depth_{depth}"] = {
            "depth": depth,
            "num_subjects": num_subjects,
            "train_overall": train_acc["overall"],
            "val_overall": val_acc["overall"],
            "test_overall": test_acc["overall"],
            "train_common": train_acc["common"],
            "train_rare": train_acc["rare"],
            "test_common": test_acc["common"],
            "test_rare": test_acc["rare"],
            "n_test_common": n_test_common,
            "n_test_rare": n_test_rare,
            "subject_scores": {
                "train": train_subject_scores,
                "val": val_subject_scores,
                "test": test_subject_scores,
            }
        }

    # Save results
    json_path = os.path.join(out_dir, "irt_depth_sweep.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    print("\nSaved results to {}".format(json_path))

    csv_path = os.path.join(out_dir, "irt_depth_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("config,depth,train_overall,val_overall,test_overall,train_common,train_rare,test_common,test_rare,n_test_common,n_test_rare\n")
        for name, r in results.items():
            f.write("{},{},{:.4f},{:.4f},{:.4f},{},{},{},{},{},{}\n".format(
                name, r["depth"], r["train_overall"], r["val_overall"], r["test_overall"],
                "" if r["train_common"] is None else "{:.4f}".format(r["train_common"]),
                "" if r["train_rare"] is None else "{:.4f}".format(r["train_rare"]),
                "" if r["test_common"] is None else "{:.4f}".format(r["test_common"]),
                "" if r["test_rare"] is None else "{:.4f}".format(r["test_rare"]),
                r.get("n_test_common", ""), r.get("n_test_rare", ""),
            ))
    print("Saved results to {}".format(csv_path))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rare_threshold", type=int, default=1000,
                         help="Subjects with fewer than this many training examples are 'rare'.")
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--sweep", type=str, default="lambda", choices=["lambda", "depth"],
                         help="Which sweep to run: 'lambda' sweeps LAMBDA_SWEEP at DEPTH_FIXED, "
                              "'depth' sweeps DEPTH_SWEEP at LAMBDA_FIXED.")
    args = parser.parse_args()

    run_fn = lambda_run if args.sweep == "lambda" else depth_run

    run_fn(
        data_dir=args.data_dir,
        lr=args.lr,
        iterations=args.iterations,
        rare_threshold=args.rare_threshold,
        out_dir=args.out_dir,
    )