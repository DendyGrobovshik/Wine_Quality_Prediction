# Wine_Quality_Prediction
thesis for Mathematics in Machine Learning course at @Politecnico di Torino. <br>
The work aims to apply the theoretical knowledge acqired during the course on a real case. The analysis and the experiments are performed on UCI Machine Learning Wine Quality Data Set, that can be found at: https://archive.ics.uci.edu/ml/datasets/Wine+Quality . The dataset contains red and white variants of the ”Vinho Verde” wine. The data were collected from May/2004 to February/2007. <br>
## Main goals

Starting from two different sample sets - one focused on red wines and the other one on white wines - binary classification was performed, having wines’ final quality as target variable (Good or Bad wine quality). <br>
Before applying the classification algorithms, the data was preprocessed in different steps:
1. Feature scaling: data was standardized, according to the estimates of mean and standard deviation computed with the bootstrap method; <br>
2. Dimensionality Reduction with PCA to decrease the number of fea- tures describing the dataset, retaining 90% of its cumulative variance; <br>
3. Oversampling with SMOTE to balance the White Wine data set, which were imbalanced on the positive class. <br>
In order to find the best hyperparameters for each classification algorithm, a grid search with 5-fold cross validation was applied.
Models were trained on both oversampled and non-oversampled training sets. Results were evaluated in terms of accuracy and F1-score.
The proposed methods were:
• Logistic Regression; <br>
• K-Nearest Neighbors; <br>
• Support Vector Machines; • Decision Trees; <br>
• Random Forests. <br>
