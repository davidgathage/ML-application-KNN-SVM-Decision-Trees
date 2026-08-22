"""
svm_model.py
============
Support Vector Machine (SVM) on the Financial Inclusion in Africa dataset.

HOW AN SVM WORKS (intuition)
----------------------------
After preprocessing, every person is a point in a ~50-dimensional space.
An SVM tries to draw a boundary between the "has account" points and the
"no account" points - and not just any boundary, but the one with the
WIDEST possible safety margin to the nearest points on each side. Those
nearest points, the ones that pin the boundary in place, are the
"support vectors" that give the method its name.

THE KERNEL TRICK
----------------
Real data is rarely separable by a flat boundary. The RBF (Radial Basis
Function) kernel lets the SVM behave as if the data had been lifted into
a much richer space where a curved boundary becomes possible - without
ever computing that space explicitly. In practice: kernel="rbf" lets the
SVM learn smooth, curved decision boundaries.

KEY CHOICES MADE BELOW
----------------------
* C=1.0 : the discipline knob. Small C tolerates points on the wrong
    side of the boundary (smoother, simpler boundary); large C tries to
    classify every training point correctly (risking overfitting).
    C=1.0 is the standard middle ground.
* gamma="scale" : how far each training point's influence reaches.
    "scale" sets it automatically from the data's variance - the
    recommended default.
* cache_size=1000 : memory (MB) for the kernel computations - purely a
    speed setting, no effect on results.
* class_weight is deliberately NOT used (it would penalize mistakes on
    the rare "Has account" class more heavily): KNN and the ANN have no
    equivalent option, so using it here would make the four-way accuracy
    comparison unfair. The imbalance is examined via confusion matrices.
* Scaling (from prepare_data.py) is essential: like KNN, the RBF kernel
    is built on distances between points.

A NOTE ON SPEED
---------------
SVM training cost grows roughly with the SQUARE of the number of samples,
so this is by far the slowest of the four models on 23,500 rows (several
minutes, vs seconds for the others). This speed difference is itself a
finding worth reporting in the comparison.
"""

from sklearn.svm import SVC

from prepare_data import RANDOM_STATE, evaluate_model


def build_model():
    """Create the SVM classifier with the choices explained above."""
    return SVC(
        kernel="rbf",               # allow curved decision boundaries
        C=1.0,                      # standard error-tolerance setting
        gamma="scale",              # sensible automatic influence radius
        cache_size=1000,            # speed: bigger kernel cache (MB)
        random_state=RANDOM_STATE,  # reproducibility
    )


if __name__ == "__main__":
    # evaluate_model runs the shared protocol: identical preprocessing,
    # 2-fold stratified cross-validation, all-samples accuracy, and a
    # JSON results file for the comparison script.
    evaluate_model("SVM", build_model())