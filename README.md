# Wine_Quality_Prediction
Thesis for Mathematics in Machine Learning course at @Politecnico di Torino. <br>
The work aims to apply the theoretical knowledge acqired during the course on a real case. The analysis and the experiments are performed on UCI Machine Learning Wine Quality Data Set, that can be found at: https://archive.ics.uci.edu/ml/datasets/Wine+Quality . The dataset contains red and white variants of the ”Vinho Verde” wine. The data were collected from May/2004 to February/2007. <br>
## Main goals

Starting from two different sample sets - one focused on red wines and the other one on white wines - binary classification was performed, having wines’ final quality as target variable (Good or Bad wine quality). <br>
Before applying the classification algorithms, the data was preprocessed in different steps:
1. Feature scaling: data was standardized, according to the estimates of mean and standard deviation computed with the bootstrap method; <br>
2. Dimensionality Reduction with PCA to decrease the number of features describing the dataset, retaining at least 90% of its cumulative variance; <br>
3. Oversampling with ADASYN to balance the White Wine data set, which were imbalanced on the positive class. <br>

In order to find the best hyperparameters for each classification algorithm, a grid search with 5-fold cross validation was applied.
Models were trained on both oversampled and non-oversampled training sets. Results were evaluated in terms of accuracy and F1-score.
The proposed methods were:
1. Logistic Regression; <br>
2. Support Vector Machines. <br>
<br>
Subsequently, regression was performed to predict wines' quality over all possible output values.<br>
Before applying the regression models, outliers were found and dropped in order to achieve a better result.<br>
The proposed methods were: <br>
1. Linear Regression; <br>
2. Polynomial Regression; <br>
3. Ridge Regression; <br>
4. Lasso Regression. <br>
