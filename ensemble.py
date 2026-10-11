import itertools
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    load_train_csv,
    evaluate
)


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for h and g.                              #
        #####################################################################
        out = inputs
        hidden = torch.sigmoid(self.g(out))
        reconstruction = torch.sigmoid(self.h(hidden))
        out = reconstruction
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out

def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


def generate_datasets(data: dict, num: int):
    """Return `num` bootstrap datasets generated from the original dataset.

    Each returned dataset contains the same number of observations as the
    original dataset, sampled randomly with replacement.

    :param data: Dictionary containing user_id, question_id, and is_correct.
    :param num: Number of bagged datasets to generate.
    :return: A list of `num` dictionaries.
    """
    number_of_points = len(data["user_id"])
    rng = np.random.default_rng(seed=41)

    datasets = [
        {"user_id": [], "question_id": [], "is_correct": []}
        for _ in range(num)
    ]

    for dataset in datasets:
        for _ in range(number_of_points):
            random_index = rng.integers(low=0, high=number_of_points)

            for key in data:
                dataset[key].append(data[key][random_index])

    return datasets

def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def update_theta_beta(user_data, question_data, lr, theta, beta):
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
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

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
        
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data: dict, lr: float, iterations: int, num_users: int, num_questions: int):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :param num_users: int
    :param num_questions: int
    :return: (theta, beta)
    """
    # Initialize theta and beta.
    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)

    # initialize the valid user data and question data for efficient computation
    user_data = {}
    question_data = {}

    for i, j, c in zip(data["user_id"], data["question_id"], data["is_correct"]):
        if not np.isnan(c):
            if i not in user_data:
                user_data[i] = []
            user_data[i].append((j, c))

            if j not in question_data:
                question_data[j] = []
            question_data[j].append((i, c))

    for _ in range(iterations):
        theta, beta = update_theta_beta(user_data, question_data, lr, theta, beta)
    
    return theta, beta


def predict(data, theta, beta):
    predictions = []
    for user_id, question_id in zip(data["user_id"], data["question_id"]):
        pr = sigmoid(theta[user_id] - beta[question_id])
        predictions.append(pr)
    return np.array(predictions)


def bootstrap_users(num_users: int, seed: int):
    """Return list of student IDs with replacement to train neural network with"""
    rng = np.random.default_rng(seed)
    return rng.choice(num_users, size=num_users, replace=True)


def train_autoencoder(model, lr, lamb, train_data, zero_train_data, sampled_users, num_epoch):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        train_loss = 0.0

        for user_id in sampled_users:
            inputs = zero_train_data[user_id].unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Ignore entries that were originally missing.
            nan_mask = torch.isnan((train_data[user_id])).unsqueeze(0)
            target[nan_mask] = output[nan_mask].detach()

            loss = (torch.sum((output - target) ** 2) + (lamb / 2) * model.get_weight_norm())

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epoch}, "f"training loss: {train_loss:.4f}")


def predict_autoencoder(model, zero_train_data, data):
    model.eval()
    predictions = []
    with torch.no_grad():
        for user_id, question_id in zip(data["user_id"], data["question_id"]):
            inputs = zero_train_data[user_id].unsqueeze(0)
            output = model(inputs)
            predictions.append(output[0, question_id].item())
    return np.array(predictions)


def run_ensemble_experiment(
    irt_lr=0.01,
    irt_iterations=50,
    nn_lr=0.01,
    nn_iterations=40,
    lamb=0.01,
    k=50,
    base_path="./data",
    seed1=42,
    seed2=43,
):
    train_data = load_train_csv(base_path)
    zero_train_matrix, train_matrix, valid_data, test_data = load_data(base_path)
    irt_train_data = generate_datasets(train_data, 1)[0]
    num_users, num_questions = train_matrix.shape

    theta, beta = irt(irt_train_data, irt_lr, irt_iterations, num_users, num_questions)

    sampled_users1 = bootstrap_users(num_users, seed=seed1)
    sampled_users2 = bootstrap_users(num_users, seed=seed2)
    torch.manual_seed(seed1)
    model1 = AutoEncoder(num_questions, k)
    torch.manual_seed(seed2)
    model2 = AutoEncoder(num_questions, k)

    train_autoencoder(
        model1, nn_lr, lamb, train_matrix, zero_train_matrix, sampled_users1, nn_iterations
    )
    train_autoencoder(
        model2, nn_lr, lamb, train_matrix, zero_train_matrix, sampled_users2, nn_iterations
    )

    irt_valid_predictions = predict(valid_data, theta, beta)
    nn1_valid_predictions = predict_autoencoder(model1, zero_train_matrix, valid_data)
    nn2_valid_predictions = predict_autoencoder(model2, zero_train_matrix, valid_data)
    ensemble_valid_predictions = (
        irt_valid_predictions + nn1_valid_predictions + nn2_valid_predictions
    ) / 3

    return {
        "valid_accuracy": evaluate(valid_data, ensemble_valid_predictions),
        "irt_valid_accuracy": evaluate(valid_data, irt_valid_predictions),
        "nn1_valid_accuracy": evaluate(valid_data, nn1_valid_predictions),
        "nn2_valid_accuracy": evaluate(valid_data, nn2_valid_predictions),
    }


def sweep_hyperparameters(experiments=None):
    if experiments is None:
        k_values = [25, 50]
        lamb_values = [0.01, 0.02]
        nn_lr_values = [0.01, 0.03]
        nn_iterations_values = [20, 40]
        irt_lr_values = [0.01, 0.02]
        irt_iterations_values = [30, 50]

        experiments = [
            {
                "k": k,
                "lamb": lamb,
                "nn_lr": nn_lr,
                "nn_iterations": nn_iterations,
                "irt_lr": irt_lr,
                "irt_iterations": irt_iterations,
            }
            for k, lamb, nn_lr, nn_iterations, irt_lr, irt_iterations in itertools.product(
                k_values,
                lamb_values,
                nn_lr_values,
                nn_iterations_values,
                irt_lr_values,
                irt_iterations_values,
            )
        ]

    results = []
    for index, experiment in enumerate(experiments, start=1):
        print(
            f"Running experiment {index}/{len(experiments)}: "
            f"k={experiment['k']}, lamb={experiment['lamb']}, "
            f"nn_lr={experiment['nn_lr']}, nn_iterations={experiment['nn_iterations']}, "
            f"irt_lr={experiment['irt_lr']}, irt_iterations={experiment['irt_iterations']}"
        )
        result = run_ensemble_experiment(**experiment)
        print(f"Validation accuracy: {result['valid_accuracy']}")
        results.append({**experiment, **result})

    best_result = max(results, key=lambda item: item["valid_accuracy"])
    print("Best validation result:")
    print(best_result)
    return results, best_result


def main():
    train_data = load_train_csv("./data") 
    # Tensor/matrix format for the autoencoders.
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    irt_train_data = generate_datasets(train_data, 1)[0]
    num_users, num_questions = train_matrix.shape
    # Hyper parameters learned from previous question
    # best from sweep: k:50, lamb: 0.01, nn_lr: 0.03, nn_iterations: 20, irt_lr: 0.02, irt_iterations: 50
    irt_lr = 0.02
    irt_iterations = 50
    nn_lr = 0.03
    nn_iterations = 20
    lamb = 0.01
    k = 50
    # Learning theta and beta for the irt
    theta, beta = irt(irt_train_data, irt_lr, irt_iterations, num_users, num_questions)
    # Creating and training the 2 autoencoders
    sampled_users1 = bootstrap_users(num_users, seed=42)
    sampled_users2 = bootstrap_users(num_users, seed=43)
    torch.manual_seed(42)
    model1 = AutoEncoder(num_questions, k)
    torch.manual_seed(43)
    model2 = AutoEncoder(num_questions, k)
    train_autoencoder(model1, nn_lr, lamb, train_matrix, zero_train_matrix, sampled_users1, nn_iterations)
    train_autoencoder(model2, nn_lr, lamb, train_matrix, zero_train_matrix, sampled_users2, nn_iterations)
    # Calculating validation accuracy
    irt_valid_predictions = predict(valid_data, theta, beta)
    print(f"Validation accuracy for the irt is: {evaluate(valid_data, irt_valid_predictions)}")
    nn1_valid_predictions = predict_autoencoder(model1, zero_train_matrix, valid_data)
    nn2_valid_predictions = predict_autoencoder(model2, zero_train_matrix, valid_data)
    print(f"NN1 Validation Accuracy: {evaluate(valid_data, nn1_valid_predictions)}")
    print("NN 2 validation accuracy:", evaluate(valid_data, nn2_valid_predictions))
    ensemble_valid_predictions = (irt_valid_predictions + nn1_valid_predictions + nn2_valid_predictions) / 3
    valid_accuracy = evaluate(valid_data, ensemble_valid_predictions)
    print("Ensemble validation accuracy:", valid_accuracy)
    # Calculating the test accuracy
    test_predictions_irt = predict(test_data, theta, beta)
    test_predictions_nn1 = predict_autoencoder(model1, zero_train_matrix, test_data)
    test_predictions_nn2 = predict_autoencoder(model2, zero_train_matrix, test_data)
    print(f"Test accuracy for irt model: {evaluate(test_data, test_predictions_irt)}")
    print(f"Test accuracy for nn1 model: {evaluate(test_data, test_predictions_nn1)}")
    print(f"Test accuracy for nn2 model: {evaluate(test_data, test_predictions_nn2)}")
    ensemble_test_predictions = (test_predictions_irt + test_predictions_nn1 + test_predictions_nn2) / 3
    test_accuracy = evaluate(test_data, ensemble_test_predictions)
    print("Ensemble test accuracy:", test_accuracy)


if __name__ == '__main__':
    main()
