# Decision Tree Coursework 1

## Overview
In this project, we implemented a decision tree algorithm and use it to classify indoor locations based on Wi-Fi signal strengths collected from a mobile phone. We compared both an unpruned decision tree and then a pruned decision tree.

## Node Class

The `Node` class is a core component of the decision tree structure. Each `Node` represents a decision point or a leaf within the tree. Its primary attributes are:

- **attribute**: The attribute used for splitting data at this node.
- **value**: The feature value at which the split occurs.
- **left** and **right**: References to the left and right child nodes.
- **label**: The class label if this node is a leaf.
- **depth**: The depth of the node within the tree, aiding in pruning and visualization.
- **data_points**: A list of data points reaching this node, which can assist in tracking and evaluating splits.


## Setup Instructions

1. **Navigate to Directory**: Open a terminal and change the directory (`cd`) to where `decision_tree.py` is located.
2. **Dataset Location**: Ensure the directory containing `decision_tree.py` also has a subdirectory named `wifi_db`, with `clean_dataset.txt` and `noisy_dataset.txt` inside.

   > **Note**: The NumPy random seed is set to `1330` for reproducibility of results.

## Running Tests and Modifying Code

At the end of `decision_tree.py`, you can enable or disable specific parts of the code to test configurations:

- **Selecting Dataset**: Uncomment the dataset you wish to test:
  - `dataset = clean_data` for the clean dataset
  - `dataset = noisy_data` for the noisy dataset

- **Steps to Execute**:
  - `step3(dataset, visualization)`: Runs step 3, which includes evaluation, tree visualization, and confusion matrix generation. Set `visualization` to `True` for visual output.
  
  - `step4(dataset, visualization)`: Runs step 4, including tree pruning, evaluation of the pruned tree, and visualizations. Set `visualization` to `True` for visual output.

## Main Functions

- `decision_tree_learning(dataset, depth=0)`: Builds a decision tree using a recursive approach.
- `predict(x, node)`: Predicts labels for dataset `x` using the tree with `node` as the root.
- `evaluate(test_dataset, trained_tree)`: Computes accuracy and generates a confusion matrix on a test dataset.
- `cross_validation(dataset, visualize)`: Performs 10-fold cross-validation.
- `prune_tree(node)`: Prunes the decision tree by converting eligible nodes into leaves if it improves performance.
- `cross_validation_prune(dataset, visualize)`: Performs 10-fold nested cross-validation with pruning.