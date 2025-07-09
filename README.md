# AI-ML-TESTING (WEEK - 1 TO 4)
# PROJECT-1   Internship Fit Predictor using Classification Algorithms
##CODE##
Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Step 2: Load the 1000-row Dataset
df = pd.read_csv("internship_fit_predictor_1000_dataset.csv")
df.columns = df.columns.str.strip()

#  Step 3: Clean and Standardize Column Names
for col in df.columns:
    if "github" in col.lower():
        df.rename(columns={col: "GitHub_Profile_Score"}, inplace=True)
    if "gpa" in col.lower():
        df.rename(columns={col: "GPA"}, inplace=True)

#  Drop unwanted object-type columns
categorical_cols = ['Skills', 'Preferred Domain', 'Certifications', 'Hackathon Participation']
non_numeric_to_drop = [
    col for col in df.select_dtypes(include='object').columns
    if col not in categorical_cols + ['Selected']
]
df = df.drop(columns=non_numeric_to_drop)

#  Step 4: Encode the target column
df['Selected'] = df['Selected'].map({'Yes': 1, 'No': 0})

#  Step 5: Encode Categorical Columns
label_encoders = {}
for col in ['Skills', 'Preferred Domain', 'Certifications']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 6: Encode Hackathon Participation
df['Hackathon Participation'] = df['Hackathon Participation'].map({'Yes': 1, 'No': 0})

#  Step 7: Normalize Numerical Columns
scaler = StandardScaler()
df[['GPA', 'GitHub_Profile_Score']] = scaler.fit_transform(df[['GPA', 'GitHub_Profile_Score']])

#  Set Seaborn Theme
sns.set_theme(style="whitegrid")

# Step 8: Enhanced Visualizations

# GPA Histogram
plt.figure(figsize=(7, 5))
sns.histplot(df['GPA'], kde=True, color='orange', edgecolor='black', linewidth=1.2)
plt.title(" GPA Distribution", fontsize=14, weight='bold')
plt.xlabel("GPA", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# GitHub Score vs Selection (Boxplot)
plt.figure(figsize=(7, 5))
sns.boxplot(x='Selected', y='GitHub_Profile_Score', data=df, palette="Set2", linewidth=2)
plt.title(" GitHub Score vs Selection", fontsize=14, weight='bold')
plt.xlabel("Selected (0=No, 1=Yes)", fontsize=12)
plt.ylabel("GitHub_Profile_Score", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Skills vs Selection Rate
plt.figure(figsize=(7, 5))
sns.barplot(x='Skills', y='Selected', data=df, palette='Spectral')
plt.title("Skills vs Selection Rate", fontsize=14, weight='bold')
plt.xlabel("Encoded Skills", fontsize=12)
plt.ylabel("Selection Rate", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Internships Completed vs Selection
plt.figure(figsize=(7, 5))
sns.barplot(x='Internships Completed', y='Selected', data=df, palette='coolwarm')
plt.title(" Internships Completed vs Selection", fontsize=14, weight='bold')
plt.xlabel("Internships Completed", fontsize=12)
plt.ylabel("Selection Rate", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Step 8.5: Network Graph from Correlation Matrix
plt.figure(figsize=(9, 7))
correlation_matrix = df.corr()
G = nx.from_pandas_adjacency(correlation_matrix)
pos = nx.spring_layout(G, seed=42)
edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
nx.draw(
    G, pos,
    node_color='lightgreen',
    with_labels=True,
    node_size=1600,
    font_size=10,
    width=2.5,
    edge_color=weights,
    edge_cmap=plt.cm.plasma
)
plt.title(" Network Graph of Correlation Matrix", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

#  Step 9: Prepare Data for Modeling
X = df.drop('Selected', axis=1)
y = df['Selected']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Step 10: Define and Train Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200, class_weight='balanced'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='linear', probability=True, class_weight='balanced'),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight='balanced')
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        "model": model,
        "accuracy": acc,
        "confusion": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }

# Step 11: Accuracy Comparison (Colorful Barplot)
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=[results[m]['accuracy'] for m in results], palette="pastel")
plt.title(" Model Accuracy Comparison", fontsize=14, weight='bold')
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Step 12: Heatmaps of Confusion Matrices (Colorful)
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, (name, res) in enumerate(results.items()):
    sns.heatmap(res['confusion'], annot=True, fmt='d', cmap="rocket", ax=axes[i], cbar=False)
    axes[i].set_title(name, fontsize=10, weight='bold')
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
plt.suptitle("Confusion Matrices for All Models", fontsize=16, weight='bold')
plt.tight_layout()
plt.show()

results[name] = {
    "model": model,
    "accuracy": acc,
    "confusion": confusion_matrix(y_test, y_pred),
    "report": classification_report(y_test, y_pred, output_dict=True)
}

sns.barplot(x=list(results.keys()), y=[results[m]['accuracy'] for m in results], palette="pastel")
#  Print all accuracy scores
print("\n Accuracy Scores of All Models:")
for name, res in results.items():
    print(f"{name}: {res['accuracy']:.4f}")

#  Identify the best performing model
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n Best Performing Model: {best_model[0]} with Accuracy = {best_model[1]['accuracy']:.4f}")

#  Step 13: Test on New Student Data (Robust Version)
def predict_fit(student_data, model_name="KNN"):
    sample_df = pd.DataFrame([student_data])
    sample_df.columns = sample_df.columns.str.strip()

    # Safely encode categorical columns using trained LabelEncoders
    for col in ['Skills', 'Preferred Domain', 'Certifications']:
        le = label_encoders[col]
        original_value = sample_df[col][0]
        if original_value not in le.classes_:
            print(f"⚠️ Warning: Unseen value '{original_value}' in column '{col}'. Assigning as 'Other'")
            if "Other" in le.classes_:
                sample_df[col] = le.transform(["Other"])
            else:
                # Append 'Other' to encoder classes if not already present
                le.classes_ = np.append(le.classes_, 'Other')
                sample_df[col] = le.transform(["Other"])
        else:
            sample_df[col] = le.transform([original_value])

    # Encode Hackathon Participation
    sample_df['Hackathon Participation'] = 1 if sample_df['Hackathon Participation'][0] == 'Yes' else 0

    # Normalize numerical columns
    sample_df[['GPA', 'GitHub_Profile_Score']] = scaler.transform(sample_df[['GPA', 'GitHub_Profile_Score']])

    # Ensure correct feature order
    sample_df = sample_df[X.columns.tolist()]

    # Predict using chosen model
    model = results[model_name]['model']
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(sample_df)[0][1]
    else:
        prob = model.predict(sample_df)[0]  # fallback if no probability support

    print(f"\n🔍 Using Model: {model_name}")
    print(f"🔮 Probability of Selection: {prob:.2f}" if hasattr(model, "predict_proba") else f"🔮 Prediction: {prob}")
    if prob >= 0.5:
        print("✅ The student IS a good fit for the internship.")
    else:
        print("❌ The student is NOT a fit for the internship.")

# Step 14: Run Test Case

print("\n--- Test Case 1 ---")
new_student_1 = {
    'GPA': 8.5,
    'Skills': 'Python, ML',
    'Projects': 4,
    'Preferred Domain': 'Data Science',
    'Certifications': 'Coursera ML',
    'Internships Completed': 5,
    'Hackathon Participation': 'No',
    'GitHub_Profile_Score': 8
}
predict_fit(new_student_1)

# Step 15: Completion
print("\n Project Completed: All models trained and evaluated with visualizations.")
/////#######/////


(WEEK - 2 )
PROJECT 2 -  Project Title: Medical Insurance Cost Prediction using Linear Regression, Ridge,
Lasso, and Random Forest
##CODE##
# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step 2: Load the Dataset
df = pd.read_csv("synthetic_insurance_dataset.csv")
print("Dataset Loaded Successfully!")
print(df.head())

# Step 3: Prepare Features and Target
X = df.drop('charges', axis=1)
y = df['charges']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and Train Models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

print("\n📊 Model Performance Results:")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Results:")
    print(f"  MAE  = {mae:.2f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  R²   = {r2:.4f}")

# Step 6: Cross-Validation on RandomForest
rf_model = models["RandomForest"]
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print("\n🔁 Cross-Validation (Random Forest):")
print("R² Scores per fold:", cv_scores)
print(f"Average R²: {cv_scores.mean():.4f}")

# Step 7: Save Best Model
joblib.dump(rf_model, "best_model_random_forest.pkl")
print("\n💾 Best model saved as 'best_model_random_forest.pkl'")

# Step 8: Predict on New Sample
loaded_model = joblib.load("best_model_random_forest.pkl")
sample = X.iloc[[0]]
predicted = loaded_model.predict(sample)[0]
print(f"\n🔮 Predicted charge for first sample = ₹{predicted:.2f}")

# Step 9: Visualizations
# 1. Correlation Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 2. Feature Importance from Linear Regression
lr_model = models["LinearRegression"]
coeff_df = pd.DataFrame(lr_model.coef_, X.columns, columns=["Coefficient"])
coeff_df = coeff_df.sort_values(by="Coefficient", ascending=False)

plt.figure(figsize=(7, 4))
sns.barplot(x="Coefficient", y=coeff_df.index, data=coeff_df)
plt.title("Feature Importance - Linear Regression")
plt.tight_layout()
plt.show()

# 3. Actual vs Predicted Charges
y_pred_rf = rf_model.predict(X_test)
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges (Random Forest)")
plt.tight_layout()
plt.show()

# 4. Residual Plot
residuals = y_test - y_pred_rf
plt.figure(figsize=(6, 4))
sns.residplot(x=y_pred_rf, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.axhline(0, linestyle='--', color='black')
plt.title("Residual Plot")
plt.xlabel("Predicted Charges")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

/////#####/////

( WEEK -3 )
PROJECT - 3 - Customer Segmentation
##CODE##
Step 1: Install necessary packages
!pip install umap-learn mlxtend --quiet

# 📌 Step 2: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

import umap.umap_ as umap
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

import warnings
warnings.filterwarnings("ignore")

# 📌 Step 3: Generate synthetic dataset
np.random.seed(42)
n_samples = 3000
data = pd.DataFrame({
    'Age': np.random.randint(18, 70, size=n_samples),
    'Annual_Income': np.random.normal(50000, 15000, size=n_samples).astype(int),
    'Spending_Score': np.random.randint(1, 100, size=n_samples),
    'Online_Visits': np.random.poisson(10, size=n_samples),
    'Store_Visits': np.random.poisson(5, size=n_samples),
    'Loyalty_Points': np.random.randint(100, 10000, size=n_samples),
    'Tenure_Years': np.random.uniform(0.5, 15, size=n_samples).round(1),
    'Referral_Count': np.random.poisson(2, size=n_samples),
    'Cart_Abandon_Rate': np.random.uniform(0, 1, size=n_samples).round(2),
    'Product_Return_Rate': np.random.uniform(0, 0.5, size=n_samples).round(2)
})

# 📌 Step 4: Save CSV
data.to_csv("customer_segmentation_dataset.csv", index=False)
print("✅ Dataset saved as 'customer_segmentation_dataset.csv'")

# 📌 Step 5: Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 📌 Step 6: KMeans - Elbow
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, 'bo-')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# 📌 Step 7: KMeans
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)
print(f"✅ KMeans Silhouette Score: {silhouette_score(scaled_data, kmeans_labels):.2f}")

# 📌 Step 8: DBSCAN
dbscan = DBSCAN(eps=1.2, min_samples=5)
db_labels = dbscan.fit_predict(scaled_data)
n_clusters_dbscan = len(set(db_labels)) - (1 if -1 in db_labels else 0)
if n_clusters_dbscan > 1:
    print(f"✅ DBSCAN Silhouette Score: {silhouette_score(scaled_data, db_labels):.2f}")
else:
    print("⚠️ DBSCAN did not form enough clusters for silhouette score.")

# 📌 Step 9: PCA Visualization
pca = PCA(n_components=2)
pca_2d = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(pca_2d, columns=["PCA1", "PCA2"])
pca_df["KMeans"] = kmeans_labels
pca_df["DBSCAN"] = db_labels

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="KMeans", palette='Set2')
plt.title("KMeans Clusters (PCA View)")

plt.subplot(1, 2, 2)
sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="DBSCAN", palette='Set1')
plt.title("DBSCAN Clusters (PCA View)")
plt.tight_layout()
plt.show()

# 📌 Step 10: t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_2d = tsne.fit_transform(scaled_data)
plt.figure(figsize=(8,6))
sns.scatterplot(x=tsne_2d[:,0], y=tsne_2d[:,1], hue=kmeans_labels, palette='Set2')
plt.title("t-SNE Visualization (KMeans Labels)")
plt.grid(True)
plt.show()

# 📌 Step 11: UMAP
reducer = umap.UMAP(random_state=42)
umap_2d = reducer.fit_transform(scaled_data)
plt.figure(figsize=(8,6))
sns.scatterplot(x=umap_2d[:,0], y=umap_2d[:,1], hue=kmeans_labels, palette='Set3')
plt.title("UMAP Visualization (KMeans Labels)")
plt.grid(True)
plt.show()

# 📌 Step 12: Isolation Forest (Outlier Detection)
iso = IsolationForest(contamination=0.02, random_state=42)
iso_preds = iso.fit_predict(scaled_data)
iso_outliers = np.where(iso_preds == -1)[0]
print(f"✅ Isolation Forest Outliers: {len(iso_outliers)}")

# 👉 Visualize Isolation Forest Outliers on PCA
pca_df["Outlier_IF"] = iso_preds
plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="Outlier_IF", palette={1:"green", -1:"red"})
plt.title("Isolation Forest Outlier Detection (Red = Outlier)")
plt.grid(True)
plt.show()

# 📌 Step 13: Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20)
lof_preds = lof.fit_predict(scaled_data)
lof_outliers = np.where(lof_preds == -1)[0]
print(f"✅ LOF Outliers: {len(lof_outliers)}")

# 👉 Visualize LOF Outliers on PCA
pca_df["Outlier_LOF"] = lof_preds
plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="Outlier_LOF", palette={1:"blue", -1:"orange"})
plt.title("LOF Outlier Detection (Orange = Outlier)")
plt.grid(True)
plt.show()

# 📌 Step 14: Market Basket – Prepare Binary Data
binned_df = data.copy()
for col in binned_df.columns:
    binned_df[col] = pd.qcut(binned_df[col], q=4, labels=False)

basket = pd.get_dummies(binned_df.astype(str))

# 📌 Step 15: Apriori
frequent_ap = apriori(basket, min_support=0.05, use_colnames=True)
rules_ap = association_rules(frequent_ap, metric="lift", min_threshold=1)
print(f"✅ Apriori Rules Generated: {rules_ap.shape[0]}")

# 👉 Visualize Apriori Rules
top_ap = rules_ap.sort_values(by='lift', ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(data=top_ap, x="lift", y=top_ap["antecedents"].astype(str))
plt.title("Top 10 Apriori Rules by Lift")
plt.xlabel("Lift")
plt.ylabel("Rule")
plt.tight_layout()
plt.show()

# 📌 Step 16: FP-Growth
frequent_fp = fpgrowth(basket, min_support=0.05, use_colnames=True)
rules_fp = association_rules(frequent_fp, metric="lift", min_threshold=1)
print(f"✅ FP-Growth Rules Generated: {rules_fp.shape[0]}")

# 👉 Visualize FP-Growth Rules
top_fp = rules_fp.sort_values(by='lift', ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(data=top_fp, x="lift", y=top_fp["antecedents"].astype(str))
plt.title("Top 10 FP-Growth Rules by Lift")
plt.xlabel("Lift")
plt.ylabel("Rule")
plt.tight_layout()
plt.show()
reducer = umap.UMAP(random_state=42)
umap_2d = reducer.fit_transform(scaled_data)
plt.figure(figsize=(8,6))
sns.scatterplot(x=umap_2d[:,0], y=umap_2d[:,1], hue=kmeans_labels, palette='Set3')
plt.title("UMAP Visualization (KMeans Labels)")
plt.grid(True)
plt.show()

iso = IsolationForest(contamination=0.02, random_state=42)
iso_preds = iso.fit_predict(scaled_data)
iso_outliers = np.where(iso_preds == -1)[0]
print(f" Isolation Forest Outliers: {len(iso_outliers)}")

pca_df["Outlier_IF"] = iso_preds
plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="Outlier_IF", palette={1:"green", -1:"red"})
plt.title("Isolation Forest Outlier Detection (Red = Outlier)")
plt.grid(True)
plt.show()

lof = LocalOutlierFactor(n_neighbors=20)
lof_preds = lof.fit_predict(scaled_data)
lof_outliers = np.where(lof_preds == -1)[0]
print(f" LOF Outliers: {len(lof_outliers)}")

pca_df["Outlier_LOF"] = lof_preds
plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="Outlier_LOF", palette={1:"blue", -1:"orange"})
plt.title("LOF Outlier Detection (Orange = Outlier)")
plt.grid(True)
plt.show()

binned_df = data.copy()
for col in binned_df.columns:
    binned_df[col] = pd.qcut(binned_df[col], q=4, labels=False)

basket = pd.get_dummies(binned_df.astype(str))

frequent_ap = apriori(basket, min_support=0.05, use_colnames=True)
rules_ap = association_rules(frequent_ap, metric="lift", min_threshold=1)
print(f"Apriori Rules Generated: {rules_ap.shape[0]}")

top_ap = rules_ap.sort_values(by='lift', ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(data=top_ap, x="lift", y=top_ap["antecedents"].astype(str))
plt.title("Top 10 Apriori Rules by Lift")
plt.xlabel("Lift")
plt.ylabel("Rule")
plt.tight_layout()
plt.show()

frequent_fp = fpgrowth(basket, min_support=0.05, use_colnames=True)
rules_fp = association_rules(frequent_fp, metric="lift", min_threshold=1)
print(f" FP-Growth Rules Generated: {rules_fp.shape[0]}")

top_fp = rules_fp.sort_values(by='lift', ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(data=top_fp, x="lift", y=top_fp["antecedents"].astype(str))
plt.title("Top 10 FP-Growth Rules by Lift")
plt.xlabel("Lift")
plt.ylabel("Rule")
plt.tight_layout()
plt.show()

/////#####/////

(WEEK - 4 )
PROJECT - 4 - Medical Insurance Cost Predection using Linear Regression
##CODE##
# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step 2: Load the Dataset
df = pd.read_csv("synthetic_insurance_dataset.csv")
print("✅ Dataset Loaded Successfully!")
print(df.head())

# Step 3: Prepare Features and Target
X = df.drop('charges', axis=1)
y = df['charges']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and Train Models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

print("\n📊 Model Performance Results:")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Results:")
    print(f"  MAE  = {mae:.2f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  R²   = {r2:.4f}")

# Step 6: Cross-Validation on RandomForest
rf_model = models["RandomForest"]
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print("\n🔁 Cross-Validation (Random Forest):")
print("R² Scores per fold:", cv_scores)
print(f"Average R²: {cv_scores.mean():.4f}")

# Step 7: Save Best Model
joblib.dump(rf_model, "best_model_random_forest.pkl")
print("\n💾 Best model saved as 'best_model_random_forest.pkl'")

# Step 8: Predict on New Sample
loaded_model = joblib.load("best_model_random_forest.pkl")
sample = X.iloc[[0]]
predicted = loaded_model.predict(sample)[0]
print(f"\n🔮 Predicted charge for first sample = ₹{predicted:.2f}")

# Step 9: Visualizations
# 1. Correlation Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 2. Feature Importance from Linear Regression
lr_model = models["LinearRegression"]
coeff_df = pd.DataFrame(lr_model.coef_, X.columns, columns=["Coefficient"])
coeff_df = coeff_df.sort_values(by="Coefficient", ascending=False)

plt.figure(figsize=(7, 4))
sns.barplot(x="Coefficient", y=coeff_df.index, data=coeff_df)
plt.title("Feature Importance - Linear Regression")
plt.tight_layout()
plt.show()

# 3. Actual vs Predicted Charges
y_pred_rf = rf_model.predict(X_test)
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges (Random Forest)")
plt.tight_layout()
plt.show()

# 4. Residual Plot
residuals = y_test - y_pred_rf
plt.figure(figsize=(6, 4))
sns.residplot(x=y_pred_rf, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.axhline(0, linestyle='--', color='black')
plt.title("Residual Plot")
plt.xlabel("Predicted Charges")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

/////#####/////
