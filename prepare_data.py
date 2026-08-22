"""
prepare_data.py
===============
Shared data-preparation and evaluation helpers for the Financial Inclusion
in Africa mini-project.

THE PROBLEM
-------------------------------
We want to predict whether a person has a bank account ("Yes" or "No")
using facts about them: which country they live in, whether they live in
a rural or urban area, whether they own a cellphone, their age, education
level, job type, and so on.

This is called a BINARY CLASSIFICATION problem: the model must place each
person into one of exactly two classes (has account / does not).

Every model script in this project (knn_model.py, decision_tree_model.py,
ann_model.py, svm_model.py) imports the functions in this file, so the
data is prepared IDENTICALLY for every algorithm. That matters: if each
model saw differently-prepared data, any accuracy differences could be
caused by the preparation instead of the algorithm, and the comparison
would be unfair.

WHAT "PREPARATION" MEANS HERE
-----------------------------
Machine-learning algorithms work on numbers, not words. Our dataset is
mostly words ("Kenya", "Rural", "Self employed"...), so we must:

1. ONE-HOT ENCODE the categorical (word) columns.
   "country" with 4 possible values becomes 4 new columns of 0s and 1s:
   country_Kenya, country_Rwanda, country_Tanzania, country_Uganda.
   A Kenyan respondent gets 1 in the first column and 0 in the others.
   We do this instead of numbering the countries 1,2,3,4 because numbering
   would falsely tell the model that "Uganda (4) is twice Rwanda (2)".

2. STANDARDIZE the numeric columns (household_size, age_of_respondent).
   Standardizing means rescaling so each column has mean 0 and standard
   deviation 1. Distance-based algorithms (KNN, SVM) and neural networks
   are sensitive to scale: without this, "age" (16-100) would dominate
   "household_size" (1-21) simply because its numbers are bigger.
   Decision Trees do not need scaling, but applying it everywhere keeps
   the pipeline identical across models, and it does not harm the tree.

HOW WE EVALUATE (2-fold cross-validation)
-----------------------------------------
Testing a model on the same data it was trained on is like grading students
on the exact questions they revised: scores look great but mean little.

2-fold cross-validation (CV) fixes this:
  * Split the data into two halves (fold A and fold B).
  * Train on A, test on B  -> first accuracy score.
  * Train on B, test on A  -> second accuracy score.
  * Report both scores and their average.
Every sample gets used for testing exactly once, and it is always tested
by a model that never saw it during training.

We use a STRATIFIED split, which keeps the Yes/No proportion the same in
both folds. Our data is imbalanced (only ~14% "Yes"), so an unlucky random
split could put most "Yes" cases in one fold and distort the results.

We also report the "all samples" (resubstitution) accuracy: train on the
whole dataset and test on that same whole dataset. Comparing it with the
CV accuracy shows how much each algorithm MEMORIZES rather than LEARNS -
a big gap between the two is the classic signature of overfitting.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants shared by every model script
# ---------------------------------------------------------------------------

# Folder layout: the scripts live at the project root, and the dataset
# and outputs live in sibling folders right next to them:
#   ProjectFolder/             <- Path(__file__).parent
#   |-- prepare_data.py        <- these .py files
#   |-- Dataset/financial_inclusion_africa.csv
#   |-- Results/               <- created automatically on first run
DATA_FILE = (Path(__file__).parent / "Dataset"
             / "financial_inclusion_africa.csv")

# Where each model's results are written as a small JSON file, so that
# compare_models.py can later gather them into one table and one chart.
RESULTS_DIR = Path(__file__).parent / "Results"

# A fixed random seed. Anything random (shuffling the CV split, the ANN's
# initial weights) will produce the same "random" numbers every run, so
# results are reproducible - you and I will see identical accuracies.
RANDOM_STATE = 42

# Numeric columns: genuinely quantitative values.
NUMERIC_COLS = ["household_size", "age_of_respondent"]

# Categorical columns: labels with no numeric meaning. Note that "year"
# is included here - it only takes the values 2016/2017/2018 and acts as
# a survey-wave label, not as a quantity we expect to extrapolate.
CATEGORICAL_COLS = [
    "country", "year", "location_type", "cellphone_access",
    "gender_of_respondent", "relationship_with_head", "marital_status",
    "education_level", "job_type",
]


def load_features_and_target():
    """Load the CSV and split it into features (X) and target (y).

    Returns
    -------
    X : DataFrame with the 11 predictor columns.
    y : Series of 0/1 labels (1 = has a bank account).
    """
    df = pd.read_csv(DATA_FILE)

    # "uniqueid" is just a row identifier (uniqueid_1, uniqueid_2, ...).
    # It carries no information about the person, so we drop it. Leaving
    # it in could even let a model "cheat" by memorizing IDs.
    df = df.drop(columns=["uniqueid"])

    # The TARGET is what we want to predict. We convert the text labels
    # to numbers: "Yes" -> 1, "No" -> 0. Most sklearn tools expect this.
    y = df["bank_account"].map({"Yes": 1, "No": 0})

    # The FEATURES are every remaining column except the target itself.
    X = df.drop(columns=["bank_account"])

    return X, y


def make_preprocessor():
    """Build the preprocessing step used in front of every model.

    A ColumnTransformer applies a different treatment to different columns:
      * StandardScaler on the numeric columns (mean 0, std 1);
      * OneHotEncoder on the categorical columns (one 0/1 column per value).

    handle_unknown="ignore" tells the encoder not to crash if, during
    testing, it meets a category value it never saw during training -
    it simply encodes it as all zeros.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLS),
            # sparse_output=False returns an ordinary (dense) table of 0s
            # and 1s rather than a memory-saving "sparse" one. Our data is
            # small, so the memory cost is trivial, and some models - LDA
            # in particular - require dense input. Using it everywhere
            # keeps the preprocessing identical for all models.
            ("categorical",
             OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_COLS),
        ]
    )


def evaluate_model(model_name, model):
    """Run the full, identical evaluation protocol on one algorithm.

    Steps:
      1. Wrap preprocessing + model in a single Pipeline.
      2. 2-fold stratified cross-validation -> honest accuracy.
      3. Fit on ALL samples and score on ALL samples -> resubstitution
         accuracy (shows memorization/overfitting when compared to CV).
      4. Print a human-readable report and save the numbers to JSON.

    The Pipeline is important for correctness: the scaler and encoder are
    fitted ONLY on each fold's training half, then applied to its test
    half. If we scaled the whole dataset first, information about the test
    half (its mean and spread) would leak into training - a subtle but
    real form of cheating called DATA LEAKAGE.
    """
    X, y = load_features_and_target()

    pipeline = Pipeline(steps=[
        ("preprocess", make_preprocessor()),
        ("model", model),
    ])

    # ----- Step 2: 2-fold stratified cross-validation -------------------
    # shuffle=True mixes the rows before splitting (the CSV is ordered by
    # country, so without shuffling fold A would be mostly Kenya/Rwanda
    # and fold B mostly Tanzania/Uganda - a very unfair test).
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)

    # cross_val_predict returns a prediction for every one of the 23,524
    # samples, each made by the fold-model that did NOT train on it.
    # From these we can compute per-fold accuracies, the overall CV
    # accuracy, and a confusion matrix covering all samples.
    start = time.time()
    cv_predictions = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=1)
    cv_seconds = time.time() - start

    # Recover the two per-fold accuracies from the same predictions.
    fold_accuracies = []
    for _, test_idx in cv.split(X, y):
        fold_acc = accuracy_score(y.iloc[test_idx], cv_predictions[test_idx])
        fold_accuracies.append(fold_acc)

    cv_accuracy = accuracy_score(y, cv_predictions)

    # The confusion matrix counts, over all samples:
    #   [[true negatives,  false positives],
    #    [false negatives, true positives]]
    # It reveals WHERE the model errs, which plain accuracy hides.
    cm = confusion_matrix(y, cv_predictions)

    # Precision/recall/F1 per class - important for imbalanced data,
    # where a model can score high accuracy while ignoring the rare class.
    report = classification_report(y, cv_predictions,
                                   target_names=["No account", "Has account"])

    # ----- Step 3: accuracy with all samples (resubstitution) -----------
    # Train on everything, then test on the very same rows. This is NOT an
    # honest estimate of real-world performance - it is deliberately
    # optimistic - but the gap between this number and the CV number
    # tells us how much the algorithm memorizes its training data.
    pipeline.fit(X, y)
    all_samples_accuracy = accuracy_score(y, pipeline.predict(X))

    # A sanity baseline: a "model" that always predicts the majority class
    # ("No account") gets this accuracy without learning anything. Any
    # real model must beat this number to be worth using.
    baseline_accuracy = max(y.mean(), 1 - y.mean())

    # ----- Step 4: report and save --------------------------------------
    print("=" * 64)
    print(f"MODEL: {model_name}")
    print("=" * 64)
    print(f"Fold 1 accuracy:               {fold_accuracies[0]:.4f}")
    print(f"Fold 2 accuracy:               {fold_accuracies[1]:.4f}")
    print(f"2-fold CV accuracy (mean):     {cv_accuracy:.4f}")
    print(f"All-samples (train=test) acc.: {all_samples_accuracy:.4f}")
    print(f"Majority-class baseline:       {baseline_accuracy:.4f}")
    print(f"Cross-validation time:         {cv_seconds:.1f} s")
    print()
    print("Confusion matrix over all 23,524 CV predictions")
    print("(rows = truth, columns = prediction; order: No, Yes):")
    print(cm)
    print()
    print(report)

    RESULTS_DIR.mkdir(exist_ok=True)
    results = {
        "model": model_name,
        "fold_accuracies": [round(a, 4) for a in fold_accuracies],
        "cv_accuracy": round(cv_accuracy, 4),
        "all_samples_accuracy": round(all_samples_accuracy, 4),
        "baseline_accuracy": round(baseline_accuracy, 4),
        "cv_seconds": round(cv_seconds, 1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    out_file = RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

    return results