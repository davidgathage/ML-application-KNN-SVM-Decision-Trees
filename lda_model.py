"""
lda_model.py
============
Linear Discriminant Analysis (LDA) on the Financial Inclusion in Africa
dataset.

HOW LDA WORKS (intuition)
-------------------------
Imagine every respondent as a point in a space with one axis per feature
(after preprocessing, about 50 axes). The two classes - "has account" and
"no account" - form two overlapping clouds of points in that space.

LDA looks for the single direction along which, if you projected every
point onto it, the two clouds would be pushed as far APART as possible
while each cloud stays as TIGHT as possible. In other words, it finds the
line that best separates the classes. A new person is then projected onto
that line and labelled by which side of the dividing point they land on.

How it works:
  * BETWEEN-class spread: push the two class averages far apart.
  * WITHIN-class spread: keep each class compact around its own average.
LDA maximizes the ratio of the first to the second.

HOW LDA DIFFERS FROM THE OTHER FOUR METHODS
-------------------------------------------
* It is a STATISTICAL method with a closed-form solution: there is no
  iterative training loop and no hyperparameter to tune by trial. Given
  the data, the answer is computed directly - so it is extremely fast.
* It assumes each class follows a bell-shaped (Gaussian) spread with the
  SAME shape, and draws a STRAIGHT (linear) boundary between them. When
  that assumption roughly holds, LDA is remarkably strong for its
  simplicity; when the real boundary is very curved, it underfits.
* Because the boundary is a simple straight cut, LDA is one of the models
  LEAST likely to overfit - expect its "all samples" accuracy and its
  cross-validation accuracy to sit very close together.

This makes LDA a useful BASELINE among the methods: if a far more complex
model (the ANN, the SVM) cannot clearly beat this simple linear one, that
tells you the signal in the data is mostly linear.

KEY CHOICES MADE BELOW
----------------------
* solver="svd" : the default and most numerically stable solver. It works
  well with many features and does not need to build certain large
  matrices, so it copes fine with our ~50 one-hot columns.
* No class weighting is used, to keep the comparison with KNN, the ANN,
  the Decision Tree and the SVM fair. The class imbalance is examined
  through the confusion matrix instead.
* LDA does not strictly need feature scaling, but the shared preprocessing
  in prepare_data.py scales anyway. This is harmless here and keeps every
  model on an identical footing.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from prepare_data import evaluate_model


def build_model():
    """Create the LDA classifier with the choices explained above."""
    return LinearDiscriminantAnalysis(
        solver="svd",  # stable default; good with many features
    )


if __name__ == "__main__":
    # evaluate_model runs the shared protocol: identical preprocessing,
    # 2-fold stratified cross-validation, all-samples accuracy, and a
    # JSON results file for the comparison script.
    evaluate_model("LDA", build_model())