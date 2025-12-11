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
    * **Models**: `GaussianNB` (Naive Bayes), `SVC`, `KNeighborsClassifier`.
    * **Metrics**: `accuracy_score`, `confusion_matrix`.

## ⚙️ Methodology

1.  **Exploratory Data Analysis (EDA)**:
    * Understanding data distribution (Univariate Analysis) using histograms.
    * Studying relationships between variables (Correlation Matrix).
    * Analyzing the impact of different features on loan status.

2.  **Data Preprocessing**:
    * Handling Missing Values.
    * Converting categorical variables to numerical (Encoding).
    * **Feature Scaling**: Using `Power Transformation` (Yeo-Johnson) to transform data to follow a **Normal Distribution**, significantly improving the performance of the Naive Bayes algorithm by addressing skewness.

3.  **Modeling**:
    * Splitting data into `Train` and `Test` sets (to prevent data leakage).
    * Training a **Gaussian Naive Bayes** model.
    * Experimenting with other algorithms for comparison (such as KNN and SVM).

4.  **Evaluation**:
    * Measuring model performance using **Accuracy Score**.
    * Analyzing errors using the **Confusion Matrix**.

## 🚀 How to Run
1.  Install the required libraries:
    ```bash
    pip install pandas numpy plotly scikit-learn
    ```
2.  Open the `Loan_Approval_Classification.ipynb` file using Jupyter Notebook or Google Colab.
3.  Ensure the data file `loan_data.csv` is in the same directory.
4.  Run the cells sequentially to view the analysis and results.

## 📊 Results
* Data distribution was improved using `PowerTransformer` to overcome **Skewness**.
* The Naive Bayes model showed promising results after proper data preprocessing (Training accuracy reached around 92%).

---

## 👤 Author
Developed by : Samir Mohamed
