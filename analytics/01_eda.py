import seaborn as sns
import pandas as pd

# Task 1: Load Titanic dataset ONCE
df = sns.load_dataset("titanic")

print("\n========== DATASET INFO ==========")
df.info()

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe(include="all"))

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== MISSING VALUES ==========")
missing = df.isnull().sum()
missing_percent = (df.isnull().mean() * 100).round(2)

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_percent
})

print(missing_report[missing_report["missing_count"] > 0])

# Required offline fallback
df.to_csv("titanic.csv", index=False)

print("\nTitanic dataset saved as titanic.csv")
# =========================
# TASK 2 - MISSING VALUES
# =========================

print("\n========== TASK 2: MISSING VALUE HANDLING ==========")

# Work on a copy of the original loaded dataset
cleaned_df = df.copy()

# 1. Drop rows for columns with less than 5% missing values
low_missing_cols = ["embarked", "embark_town"]

cleaned_df = cleaned_df.dropna(subset=low_missing_cols)

# 2. Impute age using median because its missing percentage
# is between 5% and 30%
age_median = cleaned_df["age"].median()
cleaned_df["age"] = cleaned_df["age"].fillna(age_median)

# 3. Drop deck because more than 30% of its values are missing
cleaned_df = cleaned_df.drop(columns=["deck"])

print("\nMissing-value handling decisions:")
print("embarked: 0.22% missing -> dropped affected rows")
print("embark_town: 0.22% missing -> dropped affected rows")
print(f"age: 19.87% missing -> median imputation (median = {age_median:.2f})")
print("deck: 77.22% missing -> column dropped")

print("\nRemaining missing values:")
print(cleaned_df.isnull().sum())

print("\nCleaned dataset shape:")
print(cleaned_df.shape)
# =========================
# TASK 3 - UNIVARIATE ANALYSIS
# =========================

import matplotlib.pyplot as plt

print("\n========== TASK 3: UNIVARIATE ANALYSIS ==========")

# ---------- IQR OUTLIER FUNCTION ----------
def count_iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[
        (series < lower_bound) | (series > upper_bound)
    ]

    return len(outliers), lower_bound, upper_bound


# ---------- AGE ----------
age_outliers, age_lower, age_upper = count_iqr_outliers(
    cleaned_df["age"]
)

print(f"\nAge IQR outliers: {age_outliers}")
print(f"Age lower bound: {age_lower:.2f}")
print(f"Age upper bound: {age_upper:.2f}")


# ---------- FARE ----------
fare_outliers, fare_lower, fare_upper = count_iqr_outliers(
    cleaned_df["fare"]
)

print(f"\nFare IQR outliers: {fare_outliers}")
print(f"Fare lower bound: {fare_lower:.2f}")
print(f"Fare upper bound: {fare_upper:.2f}")


# ---------- FARE STATISTICS ----------
fare_mean = cleaned_df["fare"].mean()
fare_median = cleaned_df["fare"].median()
fare_mode = cleaned_df["fare"].mode().iloc[0]

print("\nFare statistics:")
print(f"Mean   : {fare_mean:.2f}")
print(f"Median : {fare_median:.2f}")
print(f"Mode   : {fare_mode:.2f}")


# ---------- SKEWNESS INTERPRETATION ----------
if fare_mean > fare_median > fare_mode:
    fare_distribution = "right-skewed"
elif fare_mean < fare_median < fare_mode:
    fare_distribution = "left-skewed"
else:
    fare_distribution = "approximately symmetric"

print(f"\nFare distribution: {fare_distribution}")

print(
    "\nInterpretation: Fare is compared using mean, median, and mode. "
    "A larger mean than median indicates that higher fare values "
    "are pulling the distribution toward the right."
)


# ---------- HISTOGRAMS ----------
plt.figure(figsize=(8, 5))
plt.hist(cleaned_df["age"], bins=30, edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("age_histogram.png")



plt.figure(figsize=(8, 5))
plt.hist(cleaned_df["fare"], bins=30, edgecolor="black")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("fare_histogram.png")



# ---------- BOXPLOTS ----------
plt.figure(figsize=(8, 4))
plt.boxplot(cleaned_df["age"].dropna(), vert=False)
plt.title("Age Box Plot")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig("age_boxplot.png")



plt.figure(figsize=(8, 4))
plt.boxplot(cleaned_df["fare"].dropna(), vert=False)
plt.title("Fare Box Plot")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig("fare_boxplot.png")

# =========================
# TASK 4 - BIVARIATE ANALYSIS
# =========================

import seaborn as sns

print("\n========== TASK 4: BIVARIATE ANALYSIS ==========")

# Survival rate by sex
survival_by_sex = cleaned_df.groupby("sex")["survived"].mean() * 100

print("\nSurvival rate by sex:")
print(survival_by_sex.round(2))


# Survival rate by passenger class
survival_by_class = cleaned_df.groupby("pclass")["survived"].mean() * 100

print("\nSurvival rate by passenger class:")
print(survival_by_class.round(2))


# Survival rate by sex + passenger class
survival_by_sex_class = (
    cleaned_df
    .groupby(["sex", "pclass"])["survived"]
    .mean() * 100
)

print("\nSurvival rate by sex + passenger class:")
print(survival_by_sex_class.round(2))


# -------------------------------------------------
# CORRELATION MATRIX
# Exactly these six columns are required
# -------------------------------------------------

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = cleaned_df[correlation_columns].corr()

print("\n6x6 Correlation Matrix:")
print(correlation_matrix.round(3))


# -------------------------------------------------
# FIND TWO STRONGEST OFF-DIAGONAL CORRELATIONS
# -------------------------------------------------

corr_pairs = []

for i in range(len(correlation_columns)):
    for j in range(i + 1, len(correlation_columns)):
        col1 = correlation_columns[i]
        col2 = correlation_columns[j]
        corr_value = correlation_matrix.loc[col1, col2]

        corr_pairs.append(
            (col1, col2, corr_value, abs(corr_value))
        )

top_two = sorted(
    corr_pairs,
    key=lambda x: x[3],
    reverse=True
)[:2]

print("\nTwo strongest correlations:")

for col1, col2, value, absolute_value in top_two:
    print(
        f"{col1} <-> {col2}: "
        f"correlation = {value:.3f}"
    )


# -------------------------------------------------
# HEATMAP
# -------------------------------------------------

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig("correlation_heatmap.png")



print("\nTask 4 completed successfully.")
# ============================================================
# TASK 5 - MULTIVARIATE DATA STORY
# ============================================================

print("\n========== TASK 5: MULTIVARIATE DATA STORY ==========")

# Chart 1: Survival rate by passenger class
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="pclass", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("task5_survival_by_class.png")


print("""
Chart 1 Interpretation:
Passengers in different classes had noticeably different survival rates.
The chart shows how passenger class was associated with survival.
Higher-class passengers had a different survival pattern compared with lower-class passengers.
""")


# Chart 2: Fare distribution by passenger class
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="pclass", y="fare")
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("task5_fare_by_class.png")


print("""
Chart 2 Interpretation:
Fare values varied substantially across passenger classes.
Higher passenger classes generally had higher fares and a wider range of ticket prices.
This shows a clear relationship between class and fare.
""")


# Chart 3: Age vs Fare with Survival
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.7
)
plt.title("Age vs Fare by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("task5_age_fare_survival.png")


print("""
Chart 3 Interpretation:
Age and fare show different patterns across passengers.
Survival status is distributed across different age and fare combinations.
Higher fares contain a visible mixture of survived and non-survived passengers.
""")


# Chart 4: Survival by Sex
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="sex", y="survived")
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("task5_survival_by_sex.png")

print("""
Chart 4 Interpretation:
Survival rates differed substantially between male and female passengers.
The chart demonstrates that sex was strongly associated with survival outcomes.
This relationship is useful for understanding the multivariate survival story.
""")


print("\nTask 5 completed successfully.")
# Task 6: Exploratory standardization check

age_before_mean = df["age"].mean()
age_before_std = df["age"].std()

fare_before_mean = df["fare"].mean()
fare_before_std = df["fare"].std()

df["age_standardized"] = (
    (df["age"] - age_before_mean) / age_before_std
)

df["fare_standardized"] = (
    (df["fare"] - fare_before_mean) / fare_before_std
)

print("\n===== TASK 6: STANDARDIZATION CHECK =====")

print("\nAge before standardization:")
print("Mean:", round(age_before_mean, 2))
print("Std :", round(age_before_std, 2))

print("\nAge after standardization:")
print("Mean:", round(df["age_standardized"].mean(), 2))
print("Std :", round(df["age_standardized"].std(), 2))

print("\nFare before standardization:")
print("Mean:", round(fare_before_mean, 2))
print("Std :", round(fare_before_std, 2))

print("\nFare after standardization:")
print("Mean:", round(df["fare_standardized"].mean(), 2))
print("Std :", round(df["fare_standardized"].std(), 2))

print("\nTask 6 completed successfully.")
# ============================================================
# TASK 7 - STRATIFIED TRAIN / TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split

print("\n========== TASK 7: TRAIN / TEST SPLIT ==========")

# Use survived as the classification target
X = df.drop(columns=["survived"])
y = df["survived"]

# Stratified split keeps the survived/not-survived ratio similar
# in both training and testing datasets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape :", X_test.shape)

print("\nOverall class distribution:")
print(y.value_counts(normalize=True).round(3))

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True).round(3))

print("""
Why stratification matters:
The Titanic target variable contains two classes: survived and not survived.
Stratification keeps approximately the same class proportion in both train and test sets.
This makes the model evaluation more representative of the original dataset.
""")

print("\nTask 7 completed successfully.")
# ============================================================
# TASK 8 - FEATURE / TARGET SEPARATION
# ============================================================

print("\n========== TASK 8: FEATURE / TARGET SEPARATION ==========")

# Separate input features and target variable
X = df.drop(columns=["survived"])
y = df["survived"]

print("\nFeatures shape:", X.shape)
print("Target shape  :", y.shape)

print("\nFeature columns:")
print(list(X.columns))

print("\nTarget distribution:")
print(y.value_counts())

print("""
Interpretation:
The survived column is used as the target variable.
All remaining columns are used as input features.
This separation prepares the dataset for machine learning.
""")

print("\nTask 8 completed successfully.")
# ============================================================
# TASK 9 - HANDLE CATEGORICAL FEATURES
# ============================================================

print("\n========== TASK 9: CATEGORICAL FEATURES ==========")

categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumber of categorical columns:", len(categorical_columns))

print("""
Interpretation:
Categorical features contain text or category values.
Machine learning models generally require numerical values,
so these columns need to be encoded before model training.
""")

print("\nTask 9 completed successfully.")
# ============================================================
# TASK 10 - ENCODE CATEGORICAL FEATURES
# ============================================================

print("\n========== TASK 10: ENCODING CATEGORICAL FEATURES ==========")

X_encoded = pd.get_dummies(X, drop_first=True)

print("\nOriginal shape:", X.shape)
print("Encoded shape:", X_encoded.shape)

print("\nEncoded feature columns:")
print(list(X_encoded.columns))

print("""
Interpretation:
Categorical columns are converted into numerical columns
using one-hot encoding. This makes the features suitable
for machine learning algorithms.
""")

print("\nTask 10 completed successfully.")
# ============================================================
# TASK 11 - NUMERICAL FEATURE STANDARDIZATION
# ============================================================

from sklearn.preprocessing import StandardScaler

print("\n========== TASK 11: FEATURE STANDARDIZATION ==========")

numerical_columns = X_encoded.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nNumber of numerical features:", len(numerical_columns))

scaler = StandardScaler()

X_scaled = X_encoded.copy()

X_scaled[numerical_columns] = scaler.fit_transform(
    X_encoded[numerical_columns]
)

print("\nScaled feature mean:")
print(X_scaled[numerical_columns].mean().round(2))

print("\nScaled feature standard deviation:")
print(X_scaled[numerical_columns].std().round(2))

print("""
Interpretation:
Standardization transforms numerical features to a common scale.
The features are centered close to mean 0 and scaled to approximately
standard deviation 1. This helps many machine learning algorithms.
""")

print("\nTask 11 completed successfully.")
# ============================================================
# TASK 12 - TRAIN / TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split

print("\n========== TASK 12: TRAIN / TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining features shape:", X_train.shape)
print("Testing features shape :", X_test.shape)

print("\nTraining target shape:", y_train.shape)
print("Testing target shape :", y_test.shape)

print("""
Interpretation:
The dataset is divided into training and testing sets.
80% of the data is used for training and 20% for testing.
Stratification keeps the survived/not-survived proportion
approximately the same in both sets.
""")

print("\nTask 12 completed successfully.")
# ============================================================
# TASK 13 - TRAIN A BASELINE LOGISTIC REGRESSION MODEL
# ============================================================

from sklearn.linear_model import LogisticRegression

print("\n========== TASK 13: LOGISTIC REGRESSION ==========")

model = LogisticRegression(max_iter=1000, random_state=42)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel training completed.")
print("Number of predictions:", len(y_pred))

print("\nTask 13 completed successfully.")
# ========================================================
# TASK 14: MODEL EVALUATION
# ========================================================

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\n========== TASK 14: MODEL EVALUATION ==========")

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("""
Interpretation:
Accuracy shows the overall percentage of correct predictions.
The confusion matrix shows correct and incorrect predictions
for survived and non-survived passengers.
The classification report provides precision, recall, and F1-score.
""")

print("\nTask 14 completed successfully.")
# ========================================================
# TASK 15: SAVE TRAINED MODEL
# ========================================================

import joblib

print("\n========== TASK 15: SAVE TRAINED MODEL ==========")

joblib.dump(model, "logistic_regression_model.pkl")

print("Logistic Regression model saved successfully.")
print("Model file: logistic_regression_model.pkl")

print("\nTask 15 completed successfully.")