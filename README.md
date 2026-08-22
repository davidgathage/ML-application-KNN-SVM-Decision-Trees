# ML-application-KNN-SVM-Decision-Trees
Mini project comparing various ML methods including KNN, SVM, ANN and Decision Trees

Problem: Predicting financial inclusion in East Africa. We classify whether an individual has a bank account using demographic and household data from Kenya, Rwanda, Tanzania and Uganda. Dataset: Financial Inclusion in Africa on Kaggle (originally a Zindi competition).
The dataset has a total pf 23,524 respondents across Kenya (6,068), Rwanda, Tanzania and Uganda; 13 columns. There are zero missing values. The target is bank_account (Yes/No)

READING THE MAIN CHART
----------------------
For each model we plot two bars:
* "2-fold CV" - the honest score: every sample was predicted by a model
  that never saw it during training.
* "All samples" - train and test on the same data: a deliberately
  optimistic score. The GAP between the bars is memorization; a model
  with a huge gap (typically KNN and unpruned trees) is overfitting.
The dashed line is the majority-class baseline (~86%): a model that
always answers "No account" scores this without learning anything, so
only performance ABOVE the line represents real learning.