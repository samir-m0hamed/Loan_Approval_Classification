# 🏦 Loan Approval Classification Project

A machine learning system to predict bank loan approvals using customer data analysis.

## 📌 Overview
This project aims to build a **Classification Model** capable of predicting whether a loan application should be approved or rejected (`Loan Status`). The model relies on a set of financial and personal factors of the applicant, helping to reduce financial risks and accelerate the decision-making process.

## 📂 Dataset
The dataset used contains records of customers along with details of their loan applications. Key columns include:
* **person_age**: Applicant's age.
* **person_income**: Annual income.
* **person_home_ownership**: Housing status (Rent, Own, Mortgage...).
* **person_emp_exp**: Years of employment experience.
* **loan_intent**: Loan purpose (Education, Medical, Personal...).
* **loan_amnt**: Loan amount requested.
* **loan_int_rate**: Interest rate.
* **loan_percent_income**: Loan-to-income ratio.
* **cb_person_cred_hist_length**: Length of credit history.
* **credit_score**: Credit score.
* **previous_loan_defaults_on_file**: Any previous loan defaults?
* **loan_status**: (Target) 1 for Approved, 0 for Rejected.

## 🛠️ Technologies & Libraries
The project was implemented using **Python** and the following libraries:
* **Pandas & NumPy**: For data manipulation and analysis.
* **Plotly (Express & Graph Objects)**: For creating interactive charts and Exploratory Data Analysis (EDA).
* **Scikit-Learn**:
    * **Preprocessing**: `PowerTransformer`, `StandardScaler`, `OneHotEncoder`.
    * **Models**: `SVC` (SVM), `KNeighborsClassifier` (KNN), `GaussianNB` (Naive Bayes).
    * **Metrics**: `accuracy_score`, `confusion_matrix`.

## ⚙️ Methodology

1.  **Exploratory Data Analysis (EDA)**:
    * Understanding data distribution and studying relationships between variables (Correlation Matrix).
2.  **Data Preprocessing**:
    * Handling missing values and encoding categorical data.
    * Using **Power Transformation (Yeo-Johnson)** to improve data distribution and reduce skewness, ensuring better model performance.
3.  **Modeling**:
    * Splitting data into Training and Testing sets to prevent **Data Leakage**.
    * Training three different models: SVM, KNN, and Naive Bayes.

## 📊 Model Results

Three different models were trained and tested to evaluate their performance. The table below shows the final results:

| Model | Training Accuracy | Testing Accuracy |
| :--- | :---: | :---: |
| **Support Vector Machine (SVM)** | **92.30%** | **92.08%** |
| K-Nearest Neighbors (KNN) `n=5` | 93.25% | 90.18% |
| Gaussian Naive Bayes | 81.05% | 80.85% |

### 🏆 Selected Model
Based on the results, the **SVM** model was selected as the final model for this project.
* **Reason:** It achieved the **highest Testing Accuracy (92.08%)**.
* It demonstrated excellent **Generalization**, with a very small gap between Training and Testing accuracy, indicating no overfitting issues compared to the KNN model.

## 🚀 How to Run
1.  Install the required libraries:
    ```bash
    pip install pandas numpy plotly scikit-learn
    ```
2.  Open the `Loan_Approval_Classification.ipynb` file using Jupyter Notebook or Google Colab.
3.  Ensure the data file is in the same directory and run the cells sequentially.

---

## ✍️ Author
Developed by : Samir Mohamed
