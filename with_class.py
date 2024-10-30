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


# Computes the entropy given an array of labels
def compute_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    prob = counts / np.sum(counts)
    return -np.sum(prob * np.log2(prob))


def find_split(dataset):
    # We split out dataset, x is the array of features and y is the labels
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
    def __init__(self, attribute=None, value=None, left=None, right=None, label=None, depth =0):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.label = label
        self.depth = 0
        self.data_points = []  # List to store data points reaching this node


# Builds the decision tree from the training dataset
def decision_tree_learning(training_dataset, depth=0):
    labels, counts = np.unique(training_dataset[:, -1], return_counts=True)
    # Create a new node with the most common label
    node = Node(label=labels[np.argmax(counts)])

    if len(labels) == 1:
        return node, node.depth  # Return leaf node if all labels are the same
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
    if node.attribute is None:
        return np.full(len(X), node.label)
    
    left_indices = X[:, node.attribute] <= node.value
    right_indices = X[:, node.attribute] > node.value
    
    predictions  = np.zeros(len(X))
    predictions[left_indices] = predict(node.left, X[left_indices])
    predictions[right_indices] = predict(node.right, X[right_indices])

    return predictions  

def evaluate(test_dataset, trained_tree):
    y_pred = predict(trained_tree, test_dataset[:, :-1])
    y_true = test_dataset[:, -1]
    accuracy = np.mean(y_pred == y_true)
    num_classes = len(np.unique(y_true))
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[int(true_label) - 1, int(pred_label) - 1] += 1
    
    return accuracy, confusion_matrix


# Computes the precision array given the confusion matrix: tp/(tp+fp)
def compute_precision(confusion_matrix):
    precision = np.zeros(len(confusion_matrix))
    for i in range(len(confusion_matrix)):
        precision[i] = confusion_matrix[i][i] / np.sum(confusion_matrix[:, i])
    return precision


# Computes the recall given the confusion matrix: tp/(tp+fn)
def compute_recall(confusion_matrix):
    recall = np.zeros(len(confusion_matrix))
    for i in range(len(confusion_matrix)):
        recall[i] = confusion_matrix[i][i] / np.sum(confusion_matrix[i, :])
    return recall


# Computes the F1 score as 2 * p * r / (p + r) where p is precision and r is recall
def compute_f1_score(precision, recall):
    f1_score = np.zeros(len(precision))
    for i in range(len(precision)):
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    return f1_score


# # Plots the confusion matrix in a similar way to sklearn library
# def plot_confusion_matrix(confusion_matrix):
#     class_names = [("Room " + str(i)) for i in range(len(confusion_matrix))]
#     fig, ax = plt.subplots()
#     im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
#     ax.figure.colorbar(im, ax=ax)

#     ax.set(xticks=np.arange(confusion_matrix.shape[1]),
#            yticks=np.arange(confusion_matrix.shape[0]),
#            xticklabels=class_names, yticklabels=class_names,
#            title='Confusion Matrix',
#            ylabel='True label',
#            xlabel='Predicted label')
    
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#     for i in range(confusion_matrix.shape[0]):
#         for j in range(confusion_matrix.shape[1]):
#             ax.text(j, i, confusion_matrix[i, j],
#                     ha="center", va="center",
#                     color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")

#     fig.tight_layout()
#     plt.show()

def plot_confusion_matrix(matrix):
    labels = ["Room " + str(i + 1) for i in range(matrix.shape[0])]
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='Blues')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    
    ax.set_title('Confusion Matrix', pad=20)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    threshold = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]}", va='center', ha='center',
                    color="white" if matrix[i, j] > threshold else "black")

    plt.tight_layout()
    plt.show()



# # Performs 10-fold cross validation on the dataset
# def cross_validation(data, visualize=False):
#     n = len(data)
#     k = 10
#     accuracy = 0
#     fold_size = n / k
#     confusion_matrix = np.zeros((len(np.unique(data[:, -1])), len(np.unique(data[:, -1]))))

#     indices = np.random.permutation(n)
#     test_db = data[indices]

#     if visualize:
#         global_best = 0
#         best_tree = None

#     for i in range(k):
#         print("Evaluating fold", i)
#         test_data = test_db[int(i * fold_size):int((i + 1) * fold_size), :]
#         train_data = np.delete(test_db, np.s_[int(i * fold_size):int((i + 1) * fold_size)], axis=0)
#         trained_tree, _ = decision_tree_learning(train_data)
#         acc, con = evaluate(test_data, trained_tree)
#         accuracy += acc
#         confusion_matrix += con
#         if visualize and acc > global_best:
#             global_best = acc
#             best_tree = trained_tree

#     if visualize:
#         visualize_tree(best_tree)

#     return accuracy / k, confusion_matrix / k

def cross_validation(dataset, visualize=False):
    n_samples = len(dataset)
    n_folds = 10
    total_accuracy = 0
    fold_len = n_samples // n_folds
    cumulative_confusion = np.zeros((len(np.unique(dataset[:, -1])), len(np.unique(dataset[:, -1]))))
    random_order = np.random.permutation(n_samples)
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


# Stores the data points that reach each node
def store_point(x, node):
    node.data_points.append(x)  # Append the data point to the node's list

    if node.attribute is not None:
        if x[node.attribute] < node.value:
            store_point(x, node.left)
        else:
            store_point(x, node.right)

# Prunes the decision tree; returns True if the resulting node is a leaf
def prune_tree(node):
    if node.attribute is None:
        return True

    # Recursively prune the children
    left = prune_tree(node.left)
    right = prune_tree(node.right)

    # Update the depth of the current node
    node.depth = max(node.left.depth, node.right.depth) + 1

    # If both children are now leaves, check if we can prune the parent
    if left and right:
        improvement = 0
        for x in node.data_points:
            if x[node.attribute] < node.value:
                improvement += int(node.label == x[-1]) - int(node.left.label == x[-1])
            else:
                improvement += int(node.label == x[-1]) - int(node.right.label == x[-1])

        if improvement >= 0:
            # Convert the node into a leaf
            node.attribute = None
            node.value = None
            node.left = None
            node.right = None
            node.depth = 0
            return True
    return False

# Prunes the decision tree using the validation dataset
def prune(validation_db, trained_tree):
    for x in validation_db:
        store_point(x, trained_tree)
    prune_tree(trained_tree)

# Performs 10-fold cross-validation on the dataset, and shows the best pruned tree if visualize=True
def cross_validation_prune(data, visualize=False):
    n = len(data)
    k, accuracy = 10, 0
    class_num = len(np.unique(data[:, -1]))
    fold_size = n // k
    confusion_matrix = np.zeros((class_num, class_num))
    depths_before = np.zeros(k)
    depths_after = np.zeros(k)

    # Randomly shuffle the data
    np.random.shuffle(data)
    
    for i in range(k):
        internal_accuracy = 0
        internal_con = np.zeros((class_num, class_num))
        print('Evaluating fold', i)
        
        # Use one fold as test set
        test_db = data[int(i * fold_size):int((i + 1) * fold_size), :]
        not_test_db = np.delete(data, np.s_[int(i * fold_size):int((i + 1) * fold_size)], axis=0)

        # Record the best tree as an example, which is then visualized and compared with unpruned tree
        if visualize:
            global_best = 0
            best_tree = None

        # Internal cross-validation on the remaining k-1 folds
        for j in range(k - 1):
            validation_db = not_test_db[int(j * fold_size):int((j + 1) * fold_size), :]
            train_db = np.delete(not_test_db, np.s_[int(j * fold_size):int((j + 1) * fold_size)], axis=0)
            trained_tree, depth_before = decision_tree_learning(train_db)
            prune(validation_db, trained_tree)
            new_accuracy, con = evaluate(test_db, trained_tree)

            if new_accuracy > internal_accuracy:
                internal_accuracy = new_accuracy
                internal_con = con
                depths_after[i] = trained_tree.depth
                depths_before[i] = depth_before

            if visualize and new_accuracy > global_best:
                global_best = new_accuracy
                best_tree = trained_tree

        accuracy += internal_accuracy
        confusion_matrix += internal_con

    if visualize:
        visualize_tree(best_tree)

    print('Average maximal depth before pruning:', np.mean(depths_before))
    print('Average maximal depth after pruning:', np.mean(depths_after))

    return accuracy / k, confusion_matrix / k


#==================================================================================================


# SAMPLE SCRIPTS:


def part3_results(data, visualize=False):
    accuracy, confusion_matrix = cross_validation(data, visualize=visualize)
    print("Accuracy:", accuracy)
    print("Precision:", compute_precision(confusion_matrix))
    print("Recall:", compute_recall(confusion_matrix))
    print("F1-score:", compute_f1_score(compute_precision(confusion_matrix), compute_recall(confusion_matrix)))
    plot_confusion_matrix(confusion_matrix)


def part4_results(data, visualize=False):
    accuracy, confusion_matrix = cross_validation_prune(data, visualize=visualize)
    print("Accuracy:", accuracy)
    print("Precision:", compute_precision(confusion_matrix))
    print("Recall:", compute_recall(confusion_matrix))
    print("F1-score:", compute_f1_score(compute_precision(confusion_matrix), compute_recall(confusion_matrix)))
    plot_confusion_matrix(confusion_matrix)

part3_results(noisy_data, True)
part4_results(noisy_data, True)