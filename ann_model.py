"""
ann_model.py
============
Artificial Neural Network (ANN) on the Financial Inclusion in Africa
dataset, using scikit-learn's MLPClassifier (Multi-Layer Perceptron).

HOW AN ANN WORKS (intuition)
----------------------------
An ANN is layers of simple calculating units ("neurons") connected by
weighted links:

  input layer          hidden layer 1   hidden layer 2   output
  (~50 numbers per  ->  64 neurons   ->  32 neurons   -> 1 probability
  person, from the                                       of "has account"
  preprocessing)

Each neuron computes a weighted sum of its inputs and passes it through
a simple non-linear function (ReLU: negatives become 0). Stacking such
layers lets the network learn interactions that a single rule cannot,
e.g. "cellphone access matters more for young urban respondents".

TRAINING = adjusting the weights. The network starts with random weights,
predicts, measures how wrong it is (the "loss"), and nudges every weight
slightly in the direction that reduces the error. This is repeated over
the data many times (epochs) using backpropagation + the Adam optimizer.

KEY CHOICES MADE BELOW
----------------------
* hidden_layer_sizes=(64, 32): two hidden layers, wide then narrower -
    a common funnel shape; big enough to learn patterns in ~50 features,
    small enough to train in seconds on 23,500 rows.
* alpha=1e-3: L2 regularization - a penalty on large weights that
    discourages the network from memorizing individual training rows.
* early_stopping=True: 10% of the training half is set aside as a
    validation set; training stops when the validation score stops
    improving. This prevents wasted epochs and reduces overfitting.
* max_iter=200: an upper limit on training epochs (early stopping
    usually halts well before this).
* Scaling (from prepare_data.py) matters: neural networks train poorly
    when inputs are on wildly different scales.

Note: results of an ANN depend on the random starting weights; fixing
random_state makes the run reproducible.
"""

from sklearn.neural_network import MLPClassifier

from prepare_data import RANDOM_STATE, evaluate_model


def build_model():
    """Create the neural network with the choices explained above."""
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),  # two hidden layers: 64 then 32 neurons
        activation="relu",            # the standard non-linearity
        solver="adam",                # a robust, widely used optimizer
        alpha=1e-3,                   # weight penalty against memorizing
        early_stopping=True,          # stop when validation stops improving
        max_iter=200,                 # hard cap on training epochs
        random_state=RANDOM_STATE,    # reproducible initial weights
    )


if __name__ == "__main__":
    # evaluate_model runs the shared protocol: identical preprocessing,
    # 2-fold stratified cross-validation, all-samples accuracy, and a
    # JSON results file for the comparison script.
    evaluate_model("ANN", build_model())