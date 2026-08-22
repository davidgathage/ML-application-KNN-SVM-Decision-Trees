# Financial Inclusion Prediction with Machine Learning

## Students

| Name | Student number |
|---|---|
| David Gathage | 223405 |
| Aicha Zindamoyen | 134141 |

## The project

This project examines financial inclusion in East Africa. The dataset contains survey data for 23,524 persons from Kenya, Rwanda, Tanzania, and Uganda. Each record shows the location, cellphone access, age, education level, and job type of one person. The target variable shows if the person has a bank account. Five machine-learning methods make this prediction: K-Nearest Neighbors (KNN), Decision Tree, Artificial Neural Network (ANN), Support Vector Machine (SVM), and Linear Discriminant Analysis (LDA). The project compares the classification accuracy of the five methods.

## The approach

All models use the same data preparation and the same evaluation procedure. The scripts remove the row identifier, one-hot encode the nine categorical variables, and standardize the two numeric variables. Each model then completes a stratified 2-fold cross-validation. The procedure trains the model on one half of the data and tests it on the other half. The two halves then change roles, and each sample gets one honest prediction.

Each model also reports its accuracy when the training data and the test data are the same. A large difference between the two scores shows that the model memorizes the data. The dataset is not balanced, because only 14% of the persons have a bank account. Confusion matrices show the effect of this imbalance on each model.

## Folder structure

```
ProjectFolder/
├── Scripts/     (the six Python files)
├── Dataset/     (financial_inclusion_africa.csv)
└── Results/     (made automatically on the first run)
```

## The scripts

| Script | Function |
|---|---|
| `prepare_data.py` | This module holds the shared functions. It loads the dataset, prepares the features, and does the full evaluation procedure. Do not run this file directly. The other scripts import it. |
| `knn_model.py` | This script evaluates the KNN model. The model examines the 15 most similar persons and takes a distance-weighted vote. |
| `decision_tree_model.py` | This script evaluates the Decision Tree model. The model learns a sequence of yes/no questions that divides the two classes. |
| `ann_model.py` | This script evaluates the neural network. Two hidden layers with 64 and 32 neurons learn non-linear patterns in the data. |
| `svm_model.py` | This script evaluates the SVM. The model finds the widest boundary between the two classes. This script is the slowest of the five. |
| `lda_model.py` | This script evaluates the LDA model. The method finds the straight line that best separates the two classes. It is a fast statistical baseline. |
| `compare_models.py` | This script collects the results of the five models. It writes a summary table and three charts to the Results folder. Run this script last. |

## How to run the project

1. Install Python 3.10 or a later version.
2. Install the packages with `pip install -r requirements.txt`.
3. Open a terminal in the Scripts folder.
4. Run `python knn_model.py`.
5. Run `python decision_tree_model.py`.
6. Run `python ann_model.py`.
7. Run `python svm_model.py`.
8. Run `python lda_model.py`.
9. Run `python compare_models.py`.

NOTE: If a results file is not present, `compare_models.py` runs the related model automatically.

## Results summary

The SVM had the best cross-validation accuracy (0.8881). The ANN was second (0.8859), and the Decision Tree was third (0.8818). The LDA model scored 0.8802, and the KNN model was last (0.8733). The KNN model also had the largest memorization gap. The LDA model had the smallest gap, because a straight-line boundary cannot memorize the data. The LDA model also found the most true account holders. The full results are in the Results folder.

## Data source

The dataset comes from national financial-access surveys (2016 to 2018). The Zindi "Financial Inclusion in Africa" challenge released the data for public use.