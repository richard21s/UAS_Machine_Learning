import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report

# Ensure output directories exist
os.makedirs('plots', exist_ok=True)
os.makedirs('output', exist_ok=True)

print("=== LANGKAH 1: DATA ENGINEERING & EXPLORATORY DATA ANALYSIS ===")

# 1. Load Dataset
df = pd.read_csv('online_shoppers_intention.csv')
print(f"Dataset loaded. Shape: {df.shape}")

# Check for duplicates and drop them
duplicates_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates_count}")
if duplicates_count > 0:
    df = df.drop_duplicates()
    print(f"Removed duplicates. New shape: {df.shape}")

# Check for missing values
missing_values = df.isnull().sum().sum()
print(f"Missing values found: {missing_values}")

# Descriptive Statistics of Numerical Features
num_cols = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 
            'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
desc_stats = df[num_cols].describe()
desc_stats.to_csv('output/descriptive_statistics.csv')
print("Descriptive statistics saved to output/descriptive_statistics.csv")

# Save correlation matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Behavioral Features')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png', dpi=300)
plt.close()
print("Correlation matrix plot saved to plots/correlation_matrix.png")

# Categorical Encoding
# Weekend and Revenue are boolean, convert to int
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue_Target'] = df['Revenue'].astype(int)

# Use pd.get_dummies for categorical columns
categorical_cols = ['Month', 'VisitorType', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_encoded['Revenue'] = df_encoded['Revenue_Target']
df_encoded = df_encoded.drop(columns=['Revenue_Target'])

# Prepare Features and Target for Supervised Learning
X_supervised = df_encoded.drop(columns=['Revenue'])
y_supervised = df_encoded['Revenue']

# Scale numerical columns for clustering
scaler_clust = StandardScaler()
X_clust = scaler_clust.fit_transform(df[num_cols])

print("\n=== LANGKAH 2: UNSUPERVISED ANALYSIS (CLUSTERING) ===")

# Run KMeans for K = 2 to 10 to find optimal K
wcss = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_clust)
    wcss.append(kmeans.inertia_)
    
    # Calculate silhouette score (using sample for speed if needed, but 12k is fast enough)
    score = silhouette_score(X_clust, kmeans.labels_, sample_size=3000, random_state=42)
    silhouette_scores.append(score)
    print(f"K={k} | WCSS={kmeans.inertia_:.2f} | Silhouette Score={score:.4f}")

# Plot Elbow and Silhouette curves side-by-side
fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:red'
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('WCSS (Inertia)', color=color)
ax1.plot(K_range, wcss, 'o-', color=color, linewidth=2, label='WCSS')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(K_range, silhouette_scores, 's-', color=color, linewidth=2, label='Silhouette')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Elbow Method and Silhouette Score Analysis')
fig.tight_layout()
plt.savefig('plots/elbow_silhouette.png', dpi=300)
plt.close()
print("Elbow and Silhouette plot saved to plots/elbow_silhouette.png")

# Let's select K=3 as the optimal number of clusters for customer segmentation
optimal_k = 3
print(f"Fitting KMeans with optimal K={optimal_k}...")
kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_opt.fit_transform(X_clust).argmin(axis=1) # wait, fit_predict is simpler
df['Cluster'] = kmeans_opt.fit_predict(X_clust)

# Profiling of clusters
cluster_profile = df.groupby('Cluster')[num_cols + ['Revenue']].mean()
cluster_profile['Session_Count'] = df.groupby('Cluster').size()
cluster_profile['Revenue_Rate'] = df.groupby('Cluster')['Revenue'].mean()
cluster_profile.to_csv('output/cluster_profiles.csv')
print("Cluster profiles saved to output/cluster_profiles.csv")
print(cluster_profile[['Session_Count', 'PageValues', 'BounceRates', 'ExitRates', 'Revenue_Rate']])

# Plot cluster profiling
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.barplot(x='Cluster', y='PageValues', data=df, ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Average Page Values per Cluster')

sns.barplot(x='Cluster', y='BounceRates', data=df, ax=axes[0, 1], palette='viridis')
axes[0, 1].set_title('Average Bounce Rates per Cluster')

sns.barplot(x='Cluster', y='ProductRelated_Duration', data=df, ax=axes[1, 0], palette='viridis')
axes[1, 0].set_title('Average ProductRelated Duration per Cluster')

sns.barplot(x='Cluster', y='Revenue', data=df, ax=axes[1, 1], palette='viridis')
axes[1, 1].set_title('Purchase Intention Rate (Revenue) per Cluster')

plt.suptitle('Profiling of Shopper Clusters', fontsize=16)
plt.tight_layout()
plt.savefig('plots/cluster_profiling.png', dpi=300)
plt.close()
print("Cluster profiling plot saved to plots/cluster_profiling.png")


print("\n=== LANGKAH 3: INTEGRATION ANALYSIS (CLUSTERING VS CLASS LABEL) ===")
# Integration Analysis: Check how clusters match the Revenue label
cm = contingency_matrix(df['Revenue'], df['Cluster'])
print("Contingency Matrix (Revenue vs Cluster):")
print(cm)

# Save contingency matrix to csv
cm_df = pd.DataFrame(cm, index=['Revenue_False', 'Revenue_True'], columns=[f'Cluster_{i}' for i in range(optimal_k)])
cm_df.to_csv('output/contingency_matrix.csv')

ari = adjusted_rand_score(df['Revenue'], df['Cluster'])
nmi = normalized_mutual_info_score(df['Revenue'], df['Cluster'])
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

# Write integration results
with open('output/integration_results.txt', 'w') as f:
    f.write("=== INTEGRATION ANALYSIS ===\n")
    f.write(f"Adjusted Rand Index (ARI): {ari:.4f}\n")
    f.write(f"Normalized Mutual Information (NMI): {nmi:.4f}\n\n")
    f.write("Contingency Matrix (Revenue vs Cluster):\n")
    f.write(cm_df.to_string())
    f.write("\n")


print("\n=== LANGKAH 4: SUPERVISED BENCHMARKING ===")

# Scale all features for supervised models using StandardScaler
scaler_sup = StandardScaler()
X_supervised_scaled = scaler_sup.fit_transform(X_supervised)

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_supervised_scaled, y_supervised, test_size=0.2, random_state=42, stratify=y_supervised)

# Define models
models = {
    'Naive Bayes': GaussianNB(),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Artificial Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, early_stopping=True, random_state=42)
}

results = []

plt.figure(figsize=(10, 8))

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Probability (for ROC AUC)
    if hasattr(model, "predict_proba"):
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]
    else:
        y_prob_train = model.decision_function(X_train)
        y_prob_test = model.decision_function(X_test)
        
    # Metrics - Test
    acc_test = accuracy_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test)
    rec_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)
    auc_test = roc_auc_score(y_test, y_prob_test)
    
    # Metrics - Train
    acc_train = accuracy_score(y_train, y_pred_train)
    prec_train = precision_score(y_train, y_pred_train)
    rec_train = recall_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)
    auc_train = roc_auc_score(y_train, y_prob_train)
    
    results.append({
        'Model': name,
        'Train Accuracy': acc_train,
        'Test Accuracy': acc_test,
        'Train Precision': prec_train,
        'Test Precision': prec_test,
        'Train Recall': rec_train,
        'Test Recall': rec_test,
        'Train F1': f1_train,
        'Test F1': f1_test,
        'Train AUC': auc_train,
        'Test AUC': auc_test
    })
    
    # Plot ROC curve for test set
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_test:.4f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curves - Supervised Comparison (Test Set)')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('plots/roc_curves.png', dpi=300)
plt.close()
print("ROC curves plot saved to plots/roc_curves.png")

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('output/supervised_comparison.csv', index=False)
print("Supervised comparison saved to output/supervised_comparison.csv")
print(results_df)

# Plot performance metrics comparison
metrics_to_plot = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 'Test AUC']
df_plot = results_df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 6))
sns.barplot(x='Metric', y='Score', hue='Model', data=df_plot, palette='Set2')
plt.title('Supervised Performance Comparison on Test Set')
plt.ylim(0, 1.05)
plt.ylabel('Score')
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/supervised_comparison.png', dpi=300)
plt.close()
print("Supervised comparison plot saved to plots/supervised_comparison.png")

print("\n=== PIPELINE RUN COMPLETE ===")
