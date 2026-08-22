"""
knn_model.py
============
K-Nearest Neighbours (KNN) on the Financial Inclusion in Africa dataset.

HOW KNN WORKS (intuition)
-------------------------
KNN is the "ask your neighbours" algorithm. It does no real training at
all - it simply memorizes every person in the training data. To classify
a NEW person, it:

  1. Measures the distance between the new person and every stored person
     (after preprocessing, each person is a list of numbers, so distance
     is ordinary geometric distance in that number-space);
  2. Finds the K most similar people (the "nearest neighbours");
  3. Takes a vote: if most of those K neighbours have a bank account,
     it predicts "has account", otherwise "no account".

The idea: people who resemble you (same country, similar age, similar
education and job) probably have similar banking status.

KEY CHOICES MADE BELOW
----------------------
* n_neighbors=15 : K, the number of neighbours consulted.
    - Small K (e.g. 1) trusts a single nearest neighbour, so one noisy
      or unusual record can flip the answer -> overfitting.
    - Large K averages over so many people that local patterns blur out
      -> underfitting. An odd-ish middle value like 15 is a sensible
      default for a dataset of ~23,500 rows.
* weights="distance" : closer neighbours get a bigger say in the vote
    than farther ones, instead of all K counting equally.
* Scaling (done in prepare_data.py) is CRITICAL for KNN, because the
    algorithm is literally built on distances: an unscaled column with
    big numbers would dominate every distance calculation.

WHY IT CAN STRUGGLE HERE
------------------------
After one-hot encoding we have ~50 columns. In high-dimensional spaces,
distances between points become less meaningful ("the curse of
dimensionality"), and with 85% of stored neighbours being "No account",
the majority class tends to win many votes.
"""

from sklearn.neighbors import KNeighborsClassifier

from prepare_data import RANDOM_STATE, evaluate_model  # noqa: F401


def build_model():
    """Create the KNN classifier with the choices explained above."""
    return KNeighborsClassifier(
        n_neighbors=15,        # consult the 15 most similar people
        weights="distance",    # nearer neighbours count for more
        n_jobs=-1,             # use every CPU core for the distance maths
    )


if __name__ == "__main__":
    # evaluate_model runs the shared protocol: identical preprocessing,
    # 2-fold stratified cross-validation, all-samples accuracy, and a
    # JSON results file for the comparison script.
    evaluate_model("KNN", build_model())