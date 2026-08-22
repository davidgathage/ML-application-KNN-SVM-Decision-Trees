"""
decision_tree_model.py
======================
Decision Tree on the Financial Inclusion in Africa dataset.

HOW A DECISION TREE WORKS (intuition)
-------------------------------------
A decision tree is a flowchart of yes/no questions learned from the data:

    Does the person have a cellphone?
    ├── No  -> Does the person have formal education?
    │          ├── No  -> predict "No account"
    │          └── Yes -> ...more questions...
    └── Yes -> Is education level university?
               ├── Yes -> predict "Has account"
               └── No  -> ...more questions...

Training means choosing, at each step, the question that best separates
account-holders from non-holders (measured by "Gini impurity" - how mixed
the two classes still are after the split). Prediction means dropping a
new person down the flowchart until they land in a leaf.

Trees are popular because they are easy to read and explain, they handle
mixed data well, and they don't care about feature scaling.

THE OVERFITTING TRAP - AND WHY WE LIMIT THE TREE
------------------------------------------------
Left unrestricted, a tree keeps asking ever-more-specific questions until
every training person sits in their own tiny leaf. It then scores near
100% on data it has seen (it has effectively memorized the dataset) but
much worse on new data. This project makes that visible: compare the
tree's "all samples" accuracy against its 2-fold CV accuracy - the gap
is the memorization.

To keep the tree honest we "prune" it before it grows too specific:

* max_depth=10          : at most 10 questions from root to leaf.
* min_samples_leaf=25   : a leaf must describe at least 25 people, so the
                          tree cannot build rules around single individuals.

A NOTE ON FAIRNESS OF THE COMPARISON
------------------------------------
Trees (and SVMs) offer a class_weight option that penalizes mistakes on
the rare "Has account" class more heavily. We deliberately do NOT use it:
KNN and the ANN have no equivalent option, so switching it on for some
models only would make the four-way accuracy comparison unfair. The
imbalance is instead examined through the confusion matrices.
"""

from sklearn.tree import DecisionTreeClassifier

from prepare_data import RANDOM_STATE, evaluate_model


def build_model():
    """Create the Decision Tree classifier with the pruning choices above."""
    return DecisionTreeClassifier(
        max_depth=10,               # cap the flowchart's depth
        min_samples_leaf=25,        # no rules about fewer than 25 people
        random_state=RANDOM_STATE,  # reproducible tie-breaking
    )


if __name__ == "__main__":
    # evaluate_model runs the shared protocol: identical preprocessing,
    # 2-fold stratified cross-validation, all-samples accuracy, and a
    # JSON results file for the comparison script.
    evaluate_model("Decision Tree", build_model())