import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load Dataset
file_path = "Employee_Compensation_Regression_Data_R80.xlsx"
df = pd.read_excel(file_path)

# Check Missing Values
print("Missing Values:")
print(df.isnull().sum())

# Define Variables
y = df["Annual_Salary_AEDK"]
X = df.drop(columns=["Annual_Salary_AEDK"])

# Add constant
X = sm.add_constant(X)

# Fit Regression Model
model = sm.OLS(y, X).fit()

# Print Results
print("\nRegression Coefficients:")
print(model.params)

print("\nR-squared:", model.rsquared)
print("Adjusted R-squared:", model.rsquared_adj)

print("\nP-values:")
print(model.pvalues)

# Check VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(X.shape[1])]

print("\nVIF:")
print(vif_data)

# Save Results
results_df = pd.DataFrame({
    "Coefficient": model.params,
    "P-value": model.pvalues
})

results_df.to_csv("regression_results.csv")

# Predictions
df["Predicted_Salary"] = model.predict(X)
df["Residuals"] = df["Annual_Salary_AEDK"] - df["Predicted_Salary"]

# Actual vs Predicted Plot
plt.figure()
plt.scatter(df["Annual_Salary_AEDK"], df["Predicted_Salary"])
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted")
plt.savefig("actual_vs_predicted.png")
plt.close()

# Residuals vs Fitted Plot
plt.figure()
plt.scatter(df["Predicted_Salary"], df["Residuals"])
plt.axhline(y=0)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.savefig("residuals_vs_fitted.png")
plt.close()

print("Analysis complete.")
