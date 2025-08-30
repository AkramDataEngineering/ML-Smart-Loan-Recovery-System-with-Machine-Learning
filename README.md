# ML-Smart-Loan-Recovery-System-with-Machine-Learning
# 🏦 Smart Loan Recovery System with Machine Learning

## 📌 Overview
Loan defaults pose a major challenge for financial institutions by impacting profitability and cash flow. By leveraging **historical loan repayment data, borrower profiles, and payment behaviours**, financial companies can build a **Smart Loan Recovery System** to:
- Optimize collection efforts
- Minimize recovery costs
- Maximize loan repayments

This guide demonstrates how to build a **Smart Loan Recovery System** using **Machine Learning** with Python.

---

## 🗂 Dataset Overview
We use a dataset containing borrower profiles, loan details, and repayment histories. Key attributes include:
- **Demographic Information:** Age, employment type, income level, dependents.
- **Loan Details:** Loan amount, tenure, interest rate, collateral.
- **Repayment History:** Missed payments, days past due, monthly EMI.
- **Collection Efforts:** Methods used, attempts, legal actions.
- **Recovery Status:** Fully recovered, partially recovered, or outstanding.

```python
import pandas as pd

# Load data
df = pd.read_csv("/content/loan recovery.csv")
print(df.head())
```

---

## 🔍 Summary Statistics
```python
df.describe()
```
Analyzing data summary helps understand distributions, detect anomalies, and prepare for feature engineering.

---

## 📊 Loan Amount Distribution & Income Relationship
```python
import plotly.express as px
import plotly.graph_objects as go

fig = px.histogram(df, x='Loan_Amount', nbins=30, marginal="violin", opacity=0.7,
                   title="Loan Amount Distribution & Relationship with Monthly Income",
                   labels={'Loan_Amount': "Loan Amount ($)", 'Monthly_Income': "Monthly Income"})

scatter = px.scatter(df, x='Loan_Amount', y='Monthly_Income',
                     color='Loan_Amount', color_continuous_scale='Viridis',
                     size=df['Loan_Amount'])
for trace in scatter.data:
    fig.add_trace(trace)

fig.update_layout(template="plotly_white")
fig.show()
```
**Insight:** Higher-income borrowers tend to secure larger loans, showing proportionality between income and loan size.

---

## 📈 Payment History Analysis
```python
fig = px.histogram(df, x="Payment_History", color="Recovery_Status", barmode="group",
                   title="How Payment History Affects Loan Recovery")
fig.update_layout(template="plotly_white")
fig.show()
```
- On-time payments strongly correlate with **full recovery**.
- Missed payments increase the risk of write-offs.

---

## 📦 Missed Payments Impact
```python
fig = px.box(df, x="Recovery_Status", y="Num_Missed_Payments",
             color="Recovery_Status", points="all",
             title="How Missed Payments Affect Loan Recovery")
fig.update_layout(template="plotly_white")
fig.show()
```
- Loans with **4+ missed payments** often lead to **partial or no recovery**.

---

## 📊 Income vs Loan Recovery Scatter
```python
fig = px.scatter(df, x='Monthly_Income', y='Loan_Amount',
                 color='Recovery_Status', size='Loan_Amount',
                 title="How Monthly Income & Loan Amount Affect Loan Recovery")
fig.update_layout(template="plotly_white")
fig.show()
```
**Insight:** Higher-income borrowers are more likely to fully repay loans, even with large loan amounts.

---

## 🤖 Borrower Segmentation with K-Means
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features = ['Age', 'Monthly_Income', 'Loan_Amount', 'Loan_Tenure', 'Interest_Rate',
            'Collateral_Value', 'Outstanding_Loan_Amount', 'Monthly_EMI',
            'Num_Missed_Payments', 'Days_Past_Due']

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Borrower_Segment'] = kmeans.fit_predict(df_scaled)
```

---

## 🏷️ Segment Naming
```python
df['Segment_Name'] = df['Borrower_Segment'].map({
    0: 'Moderate Income, High Loan Burden',
    1: 'High Income, Low Default Risk',
    2: 'Moderate Income, Medium Risk',
    3: 'High Loan, Higher Default Risk'
})
```

---

## 🔮 Early Default Detection (Random Forest)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Label high-risk borrowers
df['High_Risk_Flag'] = df['Segment_Name'].apply(lambda x: 1 if x in ['High Loan, Higher Default Risk', 'Moderate Income, High Loan Burden'] else 0)

X = df[features]
y = df['High_Risk_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

risk_scores = rf_model.predict_proba(X_test)[:, 1]
```

---

## 🛠️ Recovery Strategy Automation
```python
def assign_recovery_strategy(score):
    if score > 0.75:
        return "Immediate legal notices & aggressive recovery attempts"
    elif 0.50 <= score <= 0.75:
        return "Settlement offers & repayment plans"
    return "Automated reminders & monitoring"

X_test = X_test.copy()
X_test['Risk_Score'] = risk_scores
X_test['Recovery_Strategy'] = X_test['Risk_Score'].apply(assign_recovery_strategy)
```

---

## 📝 Summary
We:
1. Explored borrower profiles and repayment patterns.
2. Segmented borrowers using **K-Means clustering**.
3. Built a **Random Forest classifier** to detect high-risk borrowers.
4. Automated **personalized recovery strategies**:
   - High risk → Legal action
   - Medium risk → Settlement offers
   - Low risk → Automated reminders

**Impact:**
- Reduced collection costs
- Improved loan recovery rates
- Enabled proactive borrower management
