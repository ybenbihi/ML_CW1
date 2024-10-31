import numpy as np
import matplotlib.pyplot as plt


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### PART 1: Loading data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


clean_data = np.loadtxt('./wifi_db/clean_dataset.txt')
noisy_data = np.loadtxt('./wifi_db/noisy_dataset.txt')

# Set a seed to make the results reproducible
seed = 1330
np.random.seed(seed)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Part 2: Creating Decision Trees

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def compute_entropy(labels):
    """
    Computes the entropy of an array of labels.

    Parameters
    ----------
    labels : array-like
        Array containing the labels for each instance.

    Returns
    -------
    float
        The entropy value of the labels.
    """
    _, counts = np.unique(labels, return_counts=True)
    prob = counts / np.sum(counts)
    return -np.sum(prob * np.log2(prob))


def find_split(dataset):
    """
    Finds the best attribute and value for splitting the dataset to maximize information gain.

    Parameters
    ----------
    dataset : ndarray
        A 2D array where each row represents an instance, the last column contains labels, 
        and the other columns are features.

    Returns
    -------
    int
        Index of the best attribute for the split.
    float
        Value of the split point for the best attribute.
    """
    x = dataset[:, :-1]
    y = dataset[:, -1]

    H_total = compute_entropy(y)

    max_gain = 0
    best_attribute = -1
    best_split = None

    for attribute in range(x.shape[1]):
        arr = x[:, attribute]
        uniques = np.unique(arr)

        for split in uniques:
            left_ds = y[arr < split]
            right_ds = y[arr >= split]

            remainder = ((len(left_ds)/len(arr)) * compute_entropy(left_ds)) + ((len(right_ds)/len(arr)) * compute_entropy(right_ds))
            
            if max_gain < H_total - remainder:

                max_gain = H_total - remainder
                best_attribute = attribute
                best_split = split
    
    return best_attribute, best_split


class Node:
    """
    Represents a node in a decision tree.

    Parameters
    ----------
    attribute : int
        The index of the feature used for splitting at this node.
    value : float
        The value of the feature at which the split occurs.
    left : Node
        The left child node (for values less than `value`).
    right : Node
        The right child node (for values greater than or equal to `value`).
    label : int
        The class label if the node is a leaf node.
    """
    def __init__(self, attribute=None, value=None, left=None, right=None, label=None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.label = label
        self.depth = 0
        self.data_points = []  # List to store data points reaching this node


def decision_tree_learning(training_dataset, depth=0):
    """
    Builds a decision tree from the training dataset using a recursive approach.

    Parameters
    ----------
    training_dataset : ndarray
        A 2D array where each row represents an instance, the last column contains labels, 
        and the other columns are features.
    depth : int, optional
        Current depth of the node in the tree (default is 0).

    Returns
    -------
    Node
        The root node of the constructed decision tree.
    """
    labels, counts = np.unique(training_dataset[:, -1], return_counts=True)
    # Create a new node
    node = Node(label=labels[np.argmax(counts)])

    if len(labels) == 1:
        return node, node.depth 
    else:
        node.attribute, node.value = find_split(training_dataset)
        data_left = training_dataset[training_dataset[:, node.attribute] < node.value]
        data_right = training_dataset[training_dataset[:, node.attribute] >= node.value]
        node.left, depth_left = decision_tree_learning(data_left)
        node.right, depth_right = decision_tree_learning(data_right)
        node.depth = max(depth_left, depth_right) + 1
        return node, node.depth


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# BONUS PART:

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def plot_node(ax, node, x, y, dx, depth):
    """
    Recursively plots nodes in the decision tree, labeling nodes with attributes or leaf values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    node : Node
        The current node in the decision tree.
    x : float
        x-coordinate for node placement.
    y : float
        y-coordinate for node placement.
    dx : float
        Horizontal offset for child nodes.
    depth : int
        Depth level of the current node.
    """
    if node.attribute is None:
        ax.text(x, y, f'leaf: {node.label:.0f}', ha='center', bbox=dict(facecolor='white', edgecolor='black'))
    else:
        ax.text(x, y, f'[X{node.attribute} < {node.value:.1f}]', ha='center', bbox=dict(facecolor='white', edgecolor='black'))
        
        # Plot left subtree
        ax.plot([x, x - dx], [y, y - 1], color='orange')
        plot_node(ax, node.left, x - dx, y - 1, dx / 2, depth + 1)
        
        # Plot right subtree
        ax.plot([x, x + dx], [y, y - 1], color='blue')
        plot_node(ax, node.right, x + dx, y - 1, dx / 2, depth + 1)

def visualize_tree(tree):
    """
    Generates a visual representation of the decision tree structure.

    Parameters
    ----------
    tree : Node
        Root node of the decision tree.
    """
    depth = tree.depth
    fig, ax = plt.subplots()
    ax.set_xlim(-2**depth, 2**depth)
    ax.set_ylim(-depth-0.5, 1)
    ax.axis('off')
    plot_node(ax, tree, 0, 0, 2**(depth-1), 0)
    fig.tight_layout()
    plt.savefig("./tree.pdf")
    plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## PART 3: Evaluation

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def predict(node, X):
    """
    Predicts labels for a dataset based on a trained decision tree.

    Parameters
    ----------
    node : Node
        Root node of the trained decision tree.
    X : ndarray
        Feature matrix to predict labels for.

    Returns
    -------
    ndarray
        Array of predicted labels.
    """
    if node.attribute is None:
        return np.full(len(X), node.label)
    
    left_indices = X[:, node.attribute] <= node.value
    right_indices = X[:, node.attribute] > node.value
    
    predictions  = np.zeros(len(X))
    predictions[left_indices] = predict(node.left, X[left_indices])
    predictions[right_indices] = predict(node.right, X[right_indices])

    return predictions  

def evaluate(test_dataset, trained_tree):
    """
    Computes the accuracy and confusion matrix for the decision tree on a test dataset.

    Parameters
    ----------
    test_dataset : ndarray
        Test dataset with instances in rows; last column contains true labels.
    trained_tree : Node
        Trained decision tree for evaluation.

    Returns
    -------
    float
        Accuracy of the predictions.
    ndarray
        Confusion matrix of predictions.
    """
    y_pred = predict(trained_tree, test_dataset[:, :-1])
    y_true = test_dataset[:, -1]
    accuracy = np.mean(y_pred == y_true)
    num_classes = len(np.unique(y_true))
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[int(true_label) - 1, int(pred_label) - 1] += 1
    
    return accuracy, confusion_matrix


def calculate_precision(matrix):
    """
    Calculates precision for each class from the confusion matrix.

    Parameters
    ----------
    matrix : ndarray
        Confusion matrix where matrix[i, j] represents count of class i predicted as class j.

    Returns
    -------
    ndarray
        Precision for each class.
    """
    precisions = np.zeros(matrix.shape[0])
    for index in range(matrix.shape[0]):
        column_sum = np.sum(matrix[:, index])
        precisions[index] = matrix[index, index] / column_sum if column_sum > 0 else 0
    return precisions

def calculate_recall(matrix):
    """
    Calculates recall for each class from the confusion matrix.

    Parameters
    ----------
    matrix : ndarray
        Confusion matrix.

    Returns
    -------
    ndarray
        Recall for each class.
    """
    recalls = np.zeros(matrix.shape[0])
    for index in range(matrix.shape[0]):
        row_sum = np.sum(matrix[index, :])
        recalls[index] = matrix[index, index] / row_sum if row_sum > 0 else 0
    return recalls

def calculate_f1(precisions, recalls):
    """
    Calculates the F1 score for each class.

    Parameters
    ----------
    precisions : ndarray
        Precision values for each class.
    recalls : ndarray
        Recall values for each class.

    Returns
    -------
    ndarray
        F1 scores for each class.
    """
    f1_scores = np.zeros(len(precisions))
    for index in range(len(precisions)):
        denom = precisions[index] + recalls[index]
        f1_scores[index] = 2 * precisions[index] * recalls[index] / denom if denom > 0 else 0
    return f1_scores

def plot_confusion_matrix(matrix):
    """
    Displays a visual plot of the confusion matrix with class labels.

    Parameters
    ----------
    matrix : ndarray
        Confusion matrix with true vs. predicted counts for each class.
    """
    labels = ["Room " + str(i + 1) for i in range(matrix.shape[0])]
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='Blues')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    
    ax.set_title('Confusion Matrix', pad=20)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')

    threshold = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]}", va='center', ha='center',
                    color="white" if matrix[i, j] > threshold else "black")

    plt.tight_layout()
    plt.show()

def cross_validation(dataset, visualize=False):
    """
    Performs 10-fold cross-validation, optionally visualizing the best model.

    Parameters
    ----------
    dataset : ndarray
        Full dataset with instances and labels.
    visualize : bool, optional
        If True, visualizes the best-performing tree.

    Returns
    -------
    float
        Average accuracy across all folds.
    ndarray
        Average confusion matrix across all folds.
    """
    n_folds = 10
    total_accuracy = 0
    fold_len = len(dataset) // n_folds
    cumulative_confusion = np.zeros((len(np.unique(dataset[:, -1])), len(np.unique(dataset[:, -1]))))
    random_order = np.random.permutation(len(dataset))
    shuffled_data = dataset[random_order]

    highest_accuracy = 0
    best_tree_model = None

    for fold in range(n_folds):
        print(f"Processing fold {fold + 1}/{n_folds}")

        test_set = shuffled_data[fold * fold_len : (fold + 1) * fold_len]
        train_set = np.vstack([shuffled_data[:fold * fold_len], shuffled_data[(fold + 1) * fold_len:]])

        tree, _ = decision_tree_learning(train_set)
        fold_accuracy, fold_conf_matrix = evaluate(test_set, tree)
        
        total_accuracy += fold_accuracy
        cumulative_confusion += fold_conf_matrix

        if visualize and fold_accuracy > highest_accuracy:
            highest_accuracy = fold_accuracy
            best_tree_model = tree

    if visualize:
        visualize_tree(best_tree_model)

    return total_accuracy / n_folds, cumulative_confusion / n_folds

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Part 4: Pruning (and evaluation gain)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def store_x(x, node):
    """
    Tracks data points passing through each node during pruning.

    Parameters
    ----------
    x : ndarray
        Data point to assign to the node.
    node : Node
        Current node for data point assignment.
    """
    node.data_points.append(x)  # Append the data point to the node's list
    if node.attribute is not None:
        if x[node.attribute] < node.value:
            store_x(x, node.left)
        else:
            store_x(x, node.right)

def prune_tree(node):
    """
    Prunes the decision tree, converting eligible nodes into leaves based on improvement.

    Parameters
    ----------
    node : Node
        Root node of the subtree to prune.

    Returns
    -------
    bool
        True if the node is converted to a leaf, else False.
    """
    if node.attribute is None:
        return True

    # Recursively prune the sub-trees
    left = prune_tree(node.left)
    right = prune_tree(node.right)

    # Update the depth of the current node
    node.depth = max(node.left.depth, node.right.depth) + 1

    #  Verify if we can prune
    if left and right:
        improvement = 0
        for x in node.data_points:
            if x[node.attribute] < node.value:
                improvement += int(node.label == x[-1]) - int(node.left.label == x[-1])
            else:
                improvement += int(node.label == x[-1]) - int(node.right.label == x[-1])

        if improvement >= 0:
            # Convert the node into a leaf if pruning improves the tree
            node.attribute = None
            node.value = None
            node.left = None
            node.right = None
            node.depth = 0
            return True
    return False

def cross_validation_prune(data, visualize=False):
    """
    Performs 10-fold nested cross-validation with pruning and visualizes the best pruned model.

    Parameters
    ----------
    data : ndarray
        Dataset for cross-validation with labels in the last column.
    visualize : bool, optional
        If True, visualizes the best pruned tree.

    Returns
    -------
    float
        Average accuracy across folds.
    ndarray
        Average confusion matrix across folds.
    """
    k = 10
    total_accuracy = 0
    class_num = len(np.unique(data[:, -1]))
    fold_size = len(data) // k
    confusion_matrix = np.zeros((class_num, class_num))
    depths_before = np.zeros(k)
    depths_after = np.zeros(k)

    np.random.shuffle(data)
    
    for i in range(k):
        internal_accuracy = 0
        internal_con = np.zeros((class_num, class_num))
        print(f"Processing fold {i + 1}/{k}")
        
        # Seperating test set and training + validation set
        test_db = data[int(i * fold_size):int((i + 1) * fold_size), :]
        not_test_db = np.delete(data, np.s_[int(i * fold_size):int((i + 1) * fold_size)], axis=0)

        # Memorize best tree for visualization
        if visualize:
            global_best = 0
            best_tree = None

        # Internal cross-validation on the remaining k-1 folds
        for j in range(k - 1):
            validation_db = not_test_db[int(j * fold_size):int((j + 1) * fold_size), :]
            train_db = np.delete(not_test_db, np.s_[int(j * fold_size):int((j + 1) * fold_size)], axis=0)
            trained_tree, depth_before = decision_tree_learning(train_db)
            for x in validation_db:
                store_x(x, trained_tree)
            prune_tree(trained_tree)
            new_accuracy, con = evaluate(test_db, trained_tree)

            if new_accuracy > internal_accuracy:
                internal_accuracy = new_accuracy
                internal_con = con
                depths_after[i] = trained_tree.depth
                depths_before[i] = depth_before

            if visualize and new_accuracy > global_best:
                global_best = new_accuracy
                best_tree = trained_tree

        total_accuracy += internal_accuracy
        confusion_matrix += internal_con
    
    #  visualize tree if requested
    if visualize:
        visualize_tree(best_tree)

    print('Mean depth :')
    print('     Before pruning : ', np.mean(depths_before))
    print('     After pruning : ', np.mean(depths_after))

    return total_accuracy / k, confusion_matrix / k


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### EXAMPLE

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def part3_results(data, visualize=False):
    accuracy, confusion_matrix = cross_validation(data, visualize=visualize)
    print("Accuracy:", accuracy)
    print("Precision:", calculate_precision(confusion_matrix))
    print("Recall:", calculate_recall(confusion_matrix))
    print("F1-score:", calculate_f1(calculate_precision(confusion_matrix), calculate_recall(confusion_matrix)))
    plot_confusion_matrix(confusion_matrix)


def part4_results(data, visualize=False):
    accuracy, confusion_matrix = cross_validation_prune(data, visualize=visualize)
    print("Accuracy:", accuracy)
    print("Precision:", calculate_precision(confusion_matrix))
    print("Recall:", calculate_recall(confusion_matrix))
    print("F1-score:", calculate_f1(calculate_precision(confusion_matrix), calculate_recall(confusion_matrix)))
    plot_confusion_matrix(confusion_matrix)

part4_results(noisy_data, True)