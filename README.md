# Imputation via Regression for Missing Data in Credit Risk Assessment by Siddharth Nair

## Overview

This project tackles the common problem of missing data in a real-world machine learning scenario. [cite_start]The objective is to implement and evaluate different strategies for handling missing values in the **UCI Credit Card Default Clients Dataset**[cite: 6, 14]. [cite_start]The effectiveness of each method is measured by the performance of a downstream classification task: predicting credit card default[cite: 3, 9].

[cite_start]The core of the assignment is to compare regression-based imputation (both linear and non-linear) against simpler methods like median imputation and listwise deletion[cite: 3].

***

## Problem Statement

[cite_start]As a machine learning engineer on a credit risk assessment project, the task is to handle missing data in several critical feature columns of the client dataset[cite: 5, 7]. [cite_start]The presence of this missing data prevents the direct application of classification algorithms[cite: 8]. [cite_start]This project implements and compares four distinct strategies to address the missing values, ultimately demonstrating how the choice of imputation technique impacts the final classification model's performance[cite: 9, 10].

***

## Dataset

* [cite_start]**Source**: UCI Credit Card Default Clients Dataset[cite: 14].
* [cite_start]**Preprocessing**: To simulate a realistic scenario, **Missing At Random (MAR)** values were artificially introduced into the dataset before any analysis was performed[cite: 15, 18].
    * **7%** of values in the `AGE` column were replaced with NaN.
    * **8%** of values in the `BILL_AMT1` column were replaced with NaN.

***

## Methodology

[cite_start]The project is structured into three main parts: data imputation, model training, and comparative analysis[cite: 17, 31, 38].

### Part A: Data Preprocessing and Imputation

Four datasets were created to handle missing values differently:

1.  [cite_start]**Dataset A (Baseline)**: Missing values were imputed using the **median** of each respective column[cite: 20, 21]. [cite_start]The median is chosen over the mean because it is more robust to outliers and skewed data distributions[cite: 22].
2.  [cite_start]**Dataset B (Linear Regression)**: Missing values in the `BILL_AMT1` column were predicted and imputed using a **Linear Regression** model trained on all other available features[cite: 25, 26].
3.  [cite_start]**Dataset C (Non-Linear Regression)**: The same `BILL_AMT1` column's missing values were imputed using a **K-Nearest Neighbors (KNN) Regressor**, a non-linear model[cite: 29, 30].
4.  [cite_start]**Dataset D (Listwise Deletion)**: As a simple alternative, all rows containing any missing values were completely removed from the dataset[cite: 33]. This resulted in a **14.4% reduction** in the total number of samples.

### Part B: Model Training and Evaluation

For each of the four clean datasets (A, B, C, and D):
1.  [cite_start]The data was split into training (80%) and testing (20%) sets[cite: 32, 34].
2.  [cite_start]Features were standardized using `StandardScaler` to ensure all variables contribute equally to the model's training[cite: 35].
3.  [cite_start]A **Logistic Regression** classifier was trained on each dataset[cite: 36].
4.  [cite_start]The performance of each trained model was evaluated on its respective test set using a full classification report (Accuracy, Precision, Recall, and F1-score)[cite: 37].

***

## Results

[cite_start]The performance of the four models was compared, with a special focus on the F1-score for the positive class (defaulting clients), as this is the most critical metric for a risk assessment task[cite: 39, 40].

![Bar chart showing F1-Score for default detection across four methods: Median, Linear Regression, KNN, and Listwise Deletion. Listwise Deletion has the highest score.](https://i.imgur.com/gYq3Z5E.png)

| Model                               | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| ----------------------------------- | :------: | :-----------------: | :--------------: | :----------------: |
| **D (Listwise Deletion)** | **0.8115** |     **0.7107** |   **0.2507** |    **0.3706** |
| **C (KNN Regression Imputation)** | 0.8087   |       0.6916        |     0.2434       |      0.3601      |
| **B (Linear Regression Imputation)**| 0.8085   |       0.6910        |     0.2427       |      0.3592      |
| **A (Median Imputation)** | 0.8080   |       0.6898        |     0.2396       |      0.3557      |

### Key Findings:
* [cite_start]**Listwise Deletion Performed Best**: Surprisingly, the model trained on the dataset with rows removed (Model D) outperformed all imputation strategies across every key metric[cite: 44].
* [cite_start]**Regression Outperforms Median**: Both linear and non-linear regression imputation methods provided a slight but consistent improvement over the simple median imputation baseline[cite: 41, 42, 43].
* [cite_start]**Linear vs. Non-Linear**: The KNN (non-linear) model performed marginally better than the Linear Regression model, suggesting the presence of weak non-linear relationships between the features[cite: 48].

***

## Conclusion & Recommendation

[cite_start]For this specific dataset and missing data scenario, the recommended strategy is **Listwise Deletion**[cite: 50].

**Justification**:
* [cite_start]**Superior Performance**: It achieved the highest F1-score for predicting defaults, which is the primary business objective[cite: 50].
* **Simplicity**: It is the most straightforward method and avoids the risk of introducing noise or bias through incorrect imputation.
* [cite_start]**Acceptable Data Loss**: While it reduced the dataset size by 14.4%, the performance gain suggests that the removed data might have been less informative or that the missingness pattern itself was a useful signal that was lost during imputation[cite: 46, 50].

[cite_start]In scenarios where data loss is unacceptable or the percentage of missing data is much higher, **KNN Regression Imputation** would be the next best choice, as it is better at capturing complex relationships than linear models or simple statistical measures[cite: 48, 49].

***

## How to Run

1.  **Prerequisites**: Ensure you have Python installed along with the following libraries:
    * `pandas`
    * `numpy`
    * `scikit-learn`
    * `matplotlib`
    * `seaborn`
    * `plotly`

2.  **Dataset**: Place the `UCI_Credit_Card.csv` file in the same directory as the notebook.

3.  **Execution**: Open and run the `A6.ipynb` Jupyter Notebook in a compatible environment (like Jupyter Lab or VS Code). [cite_start]The notebook is self-contained and will execute all steps from data loading to final analysis[cite: 11, 53, 55].
