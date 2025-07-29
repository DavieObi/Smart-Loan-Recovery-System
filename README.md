# Smart Loan Recovery System

## Project Overview

The Smart Loan Recovery System is an innovative, data-driven solution designed to optimize and automate the process of recovering overdue loans. Leveraging machine learning, data visualization, and strategic borrower segmentation, this system moves beyond conventional debt collection methods. It assesses borrower risk, predicts repayment behavior, and recommends dynamic, tailored recovery strategies to enhance recovery rates, reduce operational costs, and promote more efficient and ethical debt management.

## Problem Statement

Traditional loan recovery processes often grapple with inefficiencies, high operational costs, and inconsistent success rates. This is largely due to a lack of data-driven insights and a "one-size-fits-all" approach. Lending institutions frequently face challenges in:

  * Proactively identifying high-risk borrowers who are likely to default.
  * Determining the most effective and cost-efficient collection strategy for a diverse range of borrower profiles.
  * Manually managing a large volume of delinquent accounts.
  * Minimizing financial losses from non-performing loans while striving to maintain positive customer relationships.

This project addresses these issues by introducing an intelligent system that automates decision-making and optimizes recovery efforts.

## Project Objectives

The primary objectives of the Smart Loan Recovery System are to:

1.  **Enhance Recovery Efficiency:** Automate and streamline loan recovery operations, thereby reducing manual effort and processing time.
2.  **Improve Recovery Rates:** Utilize predictive analytics and machine learning to identify optimal recovery strategies, leading to higher percentages of recovered loans.
3.  **Minimize Financial Losses:** Implement a robust risk-scoring mechanism to proactively manage delinquent accounts and mitigate the impact of non-performing assets (NPAs).
4.  **Optimize Resource Allocation:** Segment borrowers into distinct risk categories, enabling lenders to prioritize efforts and allocate collection resources more effectively.
5.  **Promote Data-Driven Decisions:** Provide actionable insights through comprehensive data analysis and reporting, supporting informed strategic decision-making in loan recovery.
6.  **Personalize Recovery Approaches:** Assign dynamic, risk-based recovery strategies tailored to individual borrower profiles to improve engagement and success rates.

## Dataset Description (Column Dictionary)

The project utilizes the `loan-recovery.csv` dataset, which contains detailed information about various loans and borrower characteristics. Below is a dictionary describing each column:

  * **`Borrower_ID`**: Unique identifier for each loan borrower.
  * **`Age`**: Age of the borrower in years.
  * **`Gender`**: Gender of the borrower (e.g., Male, Female).
  * **`Employment_Type`**: Type of employment of the borrower (e.g., Salaried, Self-Employed).
  * **`Monthly_Income`**: Monthly income of the borrower.
  * **`Num_Dependents`**: Number of dependents the borrower has.
  * **`Loan_ID`**: Unique identifier for each loan.
  * **`Loan_Amount`**: The total amount of the loan.
  * **`Loan_Tenure`**: The duration of the loan in months.
  * **`Interest_Rate`**: The annual interest rate applied to the loan.
  * **`Loan_Type`**: The type of loan (e.g., Personal, Home, Auto).
  * **`Collateral_Value`**: The value of any collateral provided for the loan (if applicable).
  * **`Outstanding_Loan_Amount`**: The remaining principal amount of the loan.
  * **`Monthly_EMI`**: Equated Monthly Installment - the fixed payment amount made by the borrower each month.
  * **`Payment_History`**: Indicates the borrower's payment behavior (e.g., On-Time, Delayed, Missed).
  * **`Num_Missed_Payments`**: Number of payments missed by the borrower.
  * **`Days_Past_Due`**: Number of days the loan is past its due date.
  * **`Recovery_Status`**: The current status of loan recovery (e.g., Fully Recovered, Partially Recovered, Not Recovered).
  * **`Collection_Attempts`**: Number of attempts made to collect the outstanding amount.
  * **`Collection_Method`**: The method used for collection (e.g., Calls, Email, Settlement Offer, Legal Action).
  * **`Legal_Action_Taken`**: Indicates if legal action has been initiated (Yes/No).

## Methodology and Approach

The project follows a structured methodology, progressing from initial data understanding and exploratory analysis to advanced machine learning for risk scoring and strategy assignment.

### 1\. Data Loading & Initial Inspection

The process begins by loading the `loan-recovery.csv` dataset into a pandas DataFrame. Initial inspection using `df.info()` revealed **500 entries** and **21 columns**, with **no missing values**, indicating a clean dataset ready for analysis. The data includes a mix of numerical (`int64`, `float64`) and categorical (`object`) types. `df.describe()` provided key statistical insights into the distribution, range, and central tendencies of numerical features, such as the wide variability in `Monthly_Income` and `Loan_Amount`.

### 2\. Exploratory Data Analysis (EDA) & Visualization

Several interactive plots were generated using Plotly to uncover relationships and patterns:

#### 2.1. Loan Amount Distribution & Relationship with Monthly Income

  * **Description:** A combined histogram, violin plot, and scatter plot showing the distribution of `Loan_Amount` and its relationship with `Monthly_Income`. The scatter points are colored and sized by `Loan_Amount`, with a density curve overlay.
  * **Insight:** This visualization clearly demonstrates that **higher loan amounts are strongly linked to higher monthly income levels**. This indicates a sensible lending practice where loan sizes are generally proportional to a borrower's financial capacity.

#### 2.2. How Payment History Affects Loan Recovery Status

  * **Description:** A grouped bar chart displaying the distribution of `Recovery_Status` (Fully Recovered, Partially Recovered, Written Off) across different `Payment_History` categories (On-Time, Delayed, Missed).
  * **Insight:** The plot critically shows that **"On-Time" payments overwhelmingly lead to "Fully Recovered" loans**. "Delayed" payments result in a mix of recovered statuses, while **"Missed" payments are highly correlated with "Partially Recovered" or "Written Off" loans**, highlighting payment history as a key predictor of recovery success.

#### 2.3. How Missed Payments Affect Loan Recovery Status

  * **Description:** A box plot showing the distribution of `Num_Missed_Payments` for each `Recovery_Status`, with all individual data points overlaid.
  * **Insight:** This plot quantitatively confirms that **a higher number of missed payments is directly associated with a lower likelihood of full loan recovery and a significantly increased risk of the loan being "Written Off."** The median and spread of missed payments are substantially higher for "Written Off" loans compared to "Fully Recovered" ones.

#### 2.4. How Monthly Income and Loan Amount Affect Loan Recovery

  * **Description:** A scatter plot of `Monthly_Income` vs. `Loan_Amount`, with points colored by `Recovery_Status` and sized by `Loan_Amount`.
  * **Insight:** The plot reveals a nuanced relationship: **"Higher loans may still get recovered if income is high."** While large loan amounts carry inherent risk, a correspondingly high monthly income appears to mitigate this risk, leading to better recovery outcomes. Conversely, "Written Off" loans tend to concentrate in areas with lower income, especially relative to the loan amount.

### 3\. Feature Engineering & Borrower Segmentation

#### 3.1. Data Preprocessing (Scaling)

Numerical features crucial for understanding borrower profiles (`Age`, `Monthly_Income`, `Loan_Amount`, `Loan_Tenure`, `Interest_Rate`, `Collateral_Value`, `Outstanding_Loan_Amount`, `Monthly_EMI`, `Num_Missed_Payments`, `Days_Past_Due`) were selected. These features were then scaled using `StandardScaler` to ensure that features with larger numerical ranges do not unduly influence distance-based algorithms.

#### 3.2. K-Means Clustering for Borrower Segmentation

  * **Application:** The `KMeans` clustering algorithm was applied to the scaled features to group borrowers into distinct segments. An `optimal_k = 4` was chosen, and `n_init=10` was used to improve clustering robustness.
  * **Segment Name Update:** The numerical cluster IDs were mapped to more descriptive, actionable names:
      * **Segment 0:** 'Moderate Income, High Loan Burden'
      * **Segment 1:** 'High Income, Low Default Risk'
      * **Segment 2:** 'Moderate Income, Medium Risk'
      * **Segment 3:** 'High Loan, Higher Default Risk'

#### 3.3. Borrower Segments Visualization

  * **Description:** A scatter plot of `Monthly_Income` vs. `Loan_Amount`, with points colored by their assigned `Borrower_Segment`.
  * **Insight:** This visualization clearly shows that **"Higher loans are clustered in specific income groups,"** validating the effectiveness of the clustering. It identifies distinct borrower archetypes (e.g., low-income/low-loan, high-income/high-loan) that can be targeted with different recovery strategies.

### 4\. Risk Scoring Model Development

To predict the likelihood of a loan being high-risk, a supervised machine learning model was developed:

#### 4.1. Target Variable Creation (`High_Risk_Flag`)

A new binary target variable, `High_Risk_Flag`, was created. Borrowers falling into the `'High Loan, Higher Default Risk'` or `'Moderate Income, High Loan Burden'` segments were flagged as `1` (High Risk), while others were flagged as `0`. This transforms the problem into a clear classification task.

#### 4.2. Feature Selection & Data Splitting

The same numerical features used for clustering (`Age`, `Monthly_Income`, `Loan_Amount`, etc.) were selected as predictors (`X`). The data was then split into an 80% training set and a 20% testing set using `train_test_split`, ensuring stratification (`stratify=y`) to maintain the class distribution of `High_Risk_Flag` in both sets.

#### 4.3. Model Training (Random Forest Classifier)

A `RandomForestClassifier` with `n_estimators=100` was trained on the prepared training data (`X_train`, `y_train`). Random Forest is chosen for its robustness and ability to handle complex relationships in the data.

#### 4.4. Risk Score Generation

The trained model's `predict_proba()` method was used on the test set (`X_test`) to generate a continuous `Risk_Score` for each borrower. This score represents the probability (between 0 and 1) that a borrower belongs to the high-risk class.

### 5\. Dynamic Recovery Strategy Assignment

The culmination of the project is the dynamic assignment of recovery strategies based on the calculated `Risk_Score`:

```python
def assign_recovery_strategy(risk_score):
    if risk_score > 0.75:
        return "Immediate legal notices & aggressive recovery attempts"
    elif 0.50 <= risk_score <= 0.75:
        return "Settlement offers & repayment plans"
    else:
        return "Automated reminders & monitoring"

# Applied to the test DataFrame:
df_test['Recovery_Strategy'] = df_test['Risk_Score'].apply(assign_recovery_strategy)
```

  * **Strategy Tiers:**
      * **High Risk (Risk Score \> 0.75):** "Immediate legal notices & aggressive recovery attempts"
      * **Medium Risk (0.50 \<= Risk Score \<= 0.75):** "Settlement offers & repayment plans"
      * **Low Risk (Risk Score \< 0.50):** "Automated reminders & monitoring"

This step operationalizes the model's predictions, providing clear, actionable recommendations for loan officers or automated systems, ensuring that resources are focused where they are most needed and strategies are tailored to the specific risk profile of each borrower.

## Technologies Used

  * **Python:** The core programming language.
  * **Pandas:** For data manipulation and analysis.
  * **NumPy:** For numerical operations.
  * **Scikit-learn:** For machine learning algorithms (K-Means Clustering, Random Forest Classifier, StandardScaler, train\_test\_split).
  * **Plotly (Express & Graph Objects):** For creating highly interactive and insightful data visualizations.
