import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Same thing as previous function but using the transformed matrix instead
    matrix = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix)
    mat = mat.T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc

def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_choices = (1, 6, 11, 16, 21, 26)
    validation_accuracies_1 = []
    print("For user based knn:")
    for k in k_choices:
        validation_accuracies_1.append(knn_impute_by_user(sparse_matrix, val_data, k))
    print("\n")
    #plt.plot(k_choices, validation_accuracies_1)
    #plt.ylabel("Validation Accuracies")
    #plt.xlabel("K")
    #plt.title("K vs validation acc for user based knn")
    #plt.show()
    # We find that k=11 is the maximum so I print the test accuracy for this
    nbrs = KNNImputer(n_neighbors=11)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(sparse_matrix)
    acc = sparse_matrix_evaluate(test_data, mat)
    print(f"Test Accuracy for k = 11: {acc}\n")
    print("For question based knn:")
    validation_accuracies_2 = []
    for k in k_choices:
        validation_accuracies_2.append(knn_impute_by_item(sparse_matrix, val_data, k))
    plt.plot(k_choices, validation_accuracies_2)
    plt.xlabel("K")
    plt.ylabel("Validation Accuracies")
    plt.title("Item based knn validation accuracies")
    plt.show()
    # We find that k=21 is the maximum so I print the test accuracy for this
    nbrs = KNNImputer(n_neighbors=21)
    transformed_matrix = sparse_matrix.T
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(transformed_matrix)
    mat = mat.T
    acc = sparse_matrix_evaluate(test_data, mat)
    print(f"Test Accuracy for k = 21: {acc}\n")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
