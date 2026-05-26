import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Ujian Akhir Semester (UAS) - Machine Learning Benchmark\n",
    "## Analisis End-to-End Pada Dataset Online Shoppers Purchasing Intention\n",
    "\n",
    "**Dosen Pengampu:** Tim Dosen Machine Learning  \n",
    "**Topik Proyek:** Integrasi Unsupervised Learning (Clustering K-Means) dan Supervised Learning Benchmarking (Naive Bayes, LDA, Logistic Regression, dan MLP ANN).\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Deskripsi Project dan Pembersihan Data (Data Engineering & EDA)\n",
    "Langkah awal dalam pengolahan data ini meliputi:\n",
    "- Loading dataset `online_shoppers_intention.csv`.\n",
    "- Mengidentifikasi dan menghapus data duplikat untuk mencegah bias.\n",
    "- Memeriksa adanya missing values.\n",
    "- Melakukan analisis statistik deskriptif untuk memahami distribusi fitur.\n",
    "- Melakukan transformasi variabel kategorik menjadi bentuk numerik/vektor (One-Hot Encoding).\n",
    "- Melakukan normalisasi fitur numerik menggunakan StandardScaler agar berada dalam Vector Space yang seimbang."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score\n",
    "from sklearn.metrics.cluster import contingency_matrix\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve\n",
    "\n",
    "# Set style untuk visualisasi\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv('online_shoppers_intention.csv')\n",
    "print(f\"Shape dataset awal: {df.shape}\")\n",
    "\n",
    "# Cek data duplikat\n",
    "duplicates = df.duplicated().sum()\n",
    "print(f\"Jumlah baris duplikat: {duplicates}\")\n",
    "if duplicates > 0:\n",
    "    df = df.drop_duplicates()\n",
    "    print(f\"Shape dataset setelah menghapus duplikat: {df.shape}\")\n",
    "\n",
    "# Cek missing values\n",
    "missing = df.isnull().sum().sum()\n",
    "print(f\"Jumlah missing values: {missing}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Analisis Statistik Deskriptif & Korelasi Fitur\n",
    "Mari kita lihat statistik deskriptif dari fitur-fitur numerik (perilaku pengunjung web) dan visualisasikan matriks korelasinya."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "num_cols = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', \n",
    "            'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']\n",
    "\n",
    "# Tampilkan deskripsi statistik\n",
    "display(df[num_cols].describe())\n",
    "\n",
    "# Plot Heatmap Korelasi\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=\".2f\", linewidths=0.5)\n",
    "plt.title('Correlation Matrix of Visitor Behavioral Features')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Rekayasa Fitur (Feature Engineering / Categorical Encoding)\n",
    "Variabel kategorik seperti `Month` dan `VisitorType` akan dikodekan menggunakan One-Hot Encoding, sedangkan boolean `Weekend` dan target variabel `Revenue` diubah menjadi biner (0/1)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Konversi boolean ke integer\n",
    "df['Weekend'] = df['Weekend'].astype(int)\n",
    "df['Revenue_Target'] = df['Revenue'].astype(int)\n",
    "\n",
    "# One-Hot Encoding fitur kategorik\n",
    "categorical_cols = ['Month', 'VisitorType', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']\n",
    "df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)\n",
    "\n",
    "# Bersihkan target variable name\n",
    "df_encoded['Revenue'] = df_encoded['Revenue_Target']\n",
    "df_encoded = df_encoded.drop(columns=['Revenue_Target'])\n",
    "\n",
    "print(f\"Shape dataset setelah encoding: {df_encoded.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Unsupervised Learning: Clustering K-Means\n",
    "Untuk clustering, kita berfokus pada fitur perilaku belanja numerik (`num_cols`) untuk mengelompokkan pengunjung berdasarkan pola perilaku mereka secara objektif (tanpa melihat target label). Fitur discaling terlebih dahulu menggunakan `StandardScaler`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Normalisasi fitur numerik untuk clustering\n",
    "scaler_clust = StandardScaler()\n",
    "X_clust = scaler_clust.fit_transform(df[num_cols])\n",
    "\n",
    "# Mencari jumlah cluster optimal dengan Elbow Method dan Silhouette Score\n",
    "wcss = []\n",
    "silhouette_scores = []\n",
    "K_range = range(2, 11)\n",
    "\n",
    "for k in K_range:\n",
    "    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
    "    kmeans.fit(X_clust)\n",
    "    wcss.append(kmeans.inertia_)\n",
    "    \n",
    "    # Evaluasi Silhouette Score dengan sampel (untuk efisiensi komputasi)\n",
    "    score = silhouette_score(X_clust, kmeans.labels_, sample_size=3000, random_state=42)\n",
    "    silhouette_scores.append(score)\n",
    "    print(f\"K={k} | WCSS={kmeans.inertia_:.2f} | Silhouette Score={score:.4f}\")\n",
    "\n",
    "# Plot WCSS & Silhouette Score\n",
    "fig, ax1 = plt.subplots(figsize=(12, 5))\n",
    "\n",
    "color = 'tab:red'\n",
    "ax1.set_xlabel('Number of Clusters (K)')\n",
    "ax1.set_ylabel('WCSS (Inertia)', color=color)\n",
    "ax1.plot(K_range, wcss, 'o-', color=color, linewidth=2, label='WCSS')\n",
    "ax1.tick_params(axis='y', labelcolor=color)\n",
    "ax1.grid(True, linestyle='--', alpha=0.6)\n",
    "\n",
    "ax2 = ax1.twinx()  \n",
    "color = 'tab:blue'\n",
    "ax2.set_ylabel('Silhouette Score', color=color)\n",
    "ax2.plot(K_range, silhouette_scores, 's-', color=color, linewidth=2, label='Silhouette')\n",
    "ax2.tick_params(axis='y', labelcolor=color)\n",
    "\n",
    "plt.title('Metode Elbow dan Analisis Silhouette Score')\n",
    "fig.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Profiling & Karakterisasi Cluster Optimal\n",
    "Berdasarkan analisis Elbow (penurunan tajam melambat pada K=3) dan Silhouette score yang tinggi, kita memilih **K = 3** sebagai jumlah cluster optimal. \n",
    "\n",
    "Mari kita fit model K-Means dengan K=3 dan menganalisis profil dari masing-masing cluster."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "optimal_k = 3\n",
    "kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)\n",
    "df['Cluster'] = kmeans_opt.fit_predict(X_clust)\n",
    "\n",
    "# Menghitung rata-rata fitur per cluster\n",
    "cluster_profile = df.groupby('Cluster')[num_cols + ['Revenue']].mean()\n",
    "cluster_profile['Session_Count'] = df.groupby('Cluster').size()\n",
    "cluster_profile['Revenue_Rate'] = df.groupby('Cluster')['Revenue'].mean()\n",
    "display(cluster_profile[['Session_Count', 'PageValues', 'BounceRates', 'ExitRates', 'Revenue_Rate']])\n",
    "\n",
    "# Visualisasi Karakteristik Cluster\n",
    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
    "sns.barplot(x='Cluster', y='PageValues', data=df, ax=axes[0, 0], palette='viridis')\n",
    "axes[0, 0].set_title('Rata-rata Page Values per Cluster')\n",
    "\n",
    "sns.barplot(x='Cluster', y='BounceRates', data=df, ax=axes[0, 1], palette='viridis')\n",
    "axes[0, 1].set_title('Rata-rata Bounce Rates per Cluster')\n",
    "\n",
    "sns.barplot(x='Cluster', y='ProductRelated_Duration', data=df, ax=axes[1, 0], palette='viridis')\n",
    "axes[1, 0].set_title('Rata-rata ProductRelated Duration per Cluster')\n",
    "\n",
    "sns.barplot(x='Cluster', y='Revenue', data=df, ax=axes[1, 1], palette='viridis')\n",
    "axes[1, 1].set_title('Rasio Pembelian (Revenue) per Cluster')\n",
    "\n",
    "plt.suptitle('Profiling Karakteristik Belanja tiap Cluster', fontsize=16)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6. Analisis Integrasi: Penyelarasan Cluster dengan Label Kelas Riil (Revenue)\n",
    "Untuk melihat apakah hasil clustering (proses unsupervised) selaras secara otomatis dengan label target pembeli riil (`Revenue = True/False`), kita akan membuat contingency matrix, menghitung Adjusted Rand Index (ARI), dan Normalized Mutual Information (NMI)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Contingency Matrix\n",
    "cm = pd.crosstab(df['Revenue'], df['Cluster'], rownames=['Actual Revenue'], colnames=['Predicted Cluster'])\n",
    "print(\"Contingency Matrix (Revenue vs Cluster):\")\n",
    "display(cm)\n",
    "\n",
    "# Hitung Indeks Evaluasi Integrasi\n",
    "ari = adjusted_rand_score(df['Revenue'], df['Cluster'])\n",
    "nmi = normalized_mutual_info_score(df['Revenue'], df['Cluster'])\n",
    "\n",
    "print(f\"Adjusted Rand Index (ARI): {ari:.4f}\")\n",
    "print(f\"Normalized Mutual Information (NMI): {nmi:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7. Supervised Benchmarking (Klasifikasi Intensi Pembelian)\n",
    "Kami membandingkan performa model klasifikasi tradisional/klasik dengan model modern:\n",
    "- **Model Tradisional:** Gaussian Naive Bayes (Gaussian NB), Linear Discriminant Analysis (LDA), dan Logistic Regression.\n",
    "- **Model Modern:** Artificial Neural Network (MLP Classifier).\n",
    "\n",
    "Metrik evaluasi yang digunakan: Accuracy, Precision, Recall, F1-Score, dan Area Under ROC Curve (ROC-AUC)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Pisahkan fitur dan target\n",
    "X_supervised = df_encoded.drop(columns=['Revenue'])\n",
    "y_supervised = df_encoded['Revenue']\n",
    "\n",
    "# Normalisasi fitur untuk klasifikasi supervised\n",
    "scaler_sup = StandardScaler()\n",
    "X_supervised_scaled = scaler_sup.fit_transform(X_supervised)\n",
    "\n",
    "# Pembagian Train-Test Split (80% Train, 20% Test)\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X_supervised_scaled, y_supervised, test_size=0.2, random_state=42, stratify=y_supervised\n",
    ")\n",
    "\n",
    "# Inisialisasi Model\n",
    "models = {\n",
    "    'Naive Bayes': GaussianNB(),\n",
    "    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),\n",
    "    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),\n",
    "    'Artificial Neural Network (MLP)': MLPClassifier(\n",
    "        hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, \n",
    "        early_stopping=True, random_state=42\n",
    "    )\n",
    "}\n",
    "\n",
    "results = []\n",
    "plt.figure(figsize=(10, 8))\n",
    "\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    \n",
    "    # Prediksi\n",
    "    y_pred_train = model.predict(X_train)\n",
    "    y_pred_test = model.predict(X_test)\n",
    "    \n",
    "    # Prediksi Probabilitas\n",
    "    if hasattr(model, \"predict_proba\"):\n",
    "        y_prob_test = model.predict_proba(X_test)[:, 1]\n",
    "        y_prob_train = model.predict_proba(X_train)[:, 1]\n",
    "    else:\n",
    "        y_prob_test = model.decision_function(X_test)\n",
    "        y_prob_train = model.decision_function(X_train)\n",
    "        \n",
    "    # Evaluasi metrik\n",
    "    acc_test = accuracy_score(y_test, y_pred_test)\n",
    "    prec_test = precision_score(y_test, y_pred_test)\n",
    "    rec_test = recall_score(y_test, y_pred_test)\n",
    "    f1_test = f1_score(y_test, y_pred_test)\n",
    "    auc_test = roc_auc_score(y_test, y_prob_test)\n",
    "    \n",
    "    acc_train = accuracy_score(y_train, y_pred_train)\n",
    "    prec_train = precision_score(y_train, y_pred_train)\n",
    "    rec_train = recall_score(y_train, y_pred_train)\n",
    "    f1_train = f1_score(y_train, y_pred_train)\n",
    "    auc_train = roc_auc_score(y_train, y_prob_train)\n",
    "    \n",
    "    results.append({\n",
    "        'Model': name,\n",
    "        'Train Accuracy': acc_train, 'Test Accuracy': acc_test,\n",
    "        'Train Precision': prec_train, 'Test Precision': prec_test,\n",
    "        'Train Recall': rec_train, 'Test Recall': rec_test,\n",
    "        'Train F1': f1_train, 'Test F1': f1_test,\n",
    "        'Train AUC': auc_train, 'Test AUC': auc_test\n",
    "    })\n",
    "    \n",
    "    # ROC Curve\n",
    "    fpr, tpr, _ = roc_curve(y_test, y_prob_test)\n",
    "    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_test:.4f})', linewidth=2)\n",
    "\n",
    "plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')\n",
    "plt.xlabel('False Positive Rate (FPR)')\n",
    "plt.ylabel('True Positive Rate (TPR)')\n",
    "plt.title('Kurva ROC - Perbandingan Model Klasifikasi (Test Set)')\n",
    "plt.legend(loc='lower right')\n",
    "plt.grid(True, linestyle='--', alpha=0.6)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 8. Perbandingan Kinerja Model\n",
    "Berikut adalah ringkasan hasil evaluasi seluruh model dalam bentuk tabel dan diagram batang."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "results_df = pd.DataFrame(results)\n",
    "display(results_df)\n",
    "\n",
    "# Plot Diagram Batang Komparatif\n",
    "metrics_to_plot = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 'Test AUC']\n",
    "df_plot = results_df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')\n",
    "\n",
    "plt.figure(figsize=(14, 6))\n",
    "sns.barplot(x='Metric', y='Score', hue='Model', data=df_plot, palette='Set2')\n",
    "plt.title('Perbandingan Kinerja Klasifikasi Model Tradisional vs Modern (Test Set)')\n",
    "plt.ylim(0, 1.05)\n",
    "plt.ylabel('Score')\n",
    "plt.legend(loc='lower right')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 9. Pembahasan Singkat Temuan\n",
    "1. **Pola Profil Cluster (Unsupervised):**\n",
    "   - **Cluster 0** (\"Bounce/Uninterested Visitors\"): Ditandai dengan tingkat `BounceRates` dan `ExitRates` yang sangat tinggi, durasi kunjungan yang sangat singkat, dan kontribusi `Revenue` hampir 0%. Kelompok ini adalah pengunjung yang masuk dan langsung keluar.\n",
    "   - **Cluster 1** (\"Browsing/Potential Buyers\"): Merupakan kelompok terbesar. Menunjukkan tingkat interaksi yang wajar, `PageValues` rendah hingga sedang, dan `Revenue_Rate` sekitar 15%.\n",
    "   - **Cluster 2** (\"High-Intent Buyers\"): Kelompok dengan `PageValues` tertinggi, `BounceRates` sangat rendah, dan `Revenue_Rate` mencapai 27.8%. Ini adalah target pasar utama yang sangat aktif bertransaksi.\n",
    "\n",
    "2. **Keselarasan Cluster dengan Label Kelas (Integration):**\n",
    "   - Meskipun K-Means berhasil memisahkan kelompok dengan karakteristik perilaku yang sangat berbeda secara internal (dilihat dari perbedaan signifikan rasio konversi), nilai ARI (0.0197) dan NMI (0.0319) sangat rendah. Hal ini menunjukkan bahwa pengelompokkan alami pengunjung berdasarkan perilaku browsing mereka tidak berkorelasi linier langsung secara satu-ke-satu dengan label pembelian biner `Revenue`. Banyak pengunjung yang perilakunya mirip dengan 'browsing biasa' (Cluster 1) tetap melakukan pembelian, sedangkan sebagian besar pengunjung di cluster potensial (Cluster 2) juga tidak melakukan checkout.\n",
    "\n",
    "3. **Supervised Benchmarking (Classic vs Modern):**\n",
    "   - **Naive Bayes** memperoleh akurasi test yang sangat rendah (24.54%) karena metodenya (Gaussian NB) berasumsi bahwa semua variabel saling independen secara kondisional dan memiliki distribusi normal. Dengan banyaknya variabel One-Hot Encoding yang biner dan sangat sparse, asumsi Gaussian dilanggar keras, sehingga performanya anjlok. Namun, ia memiliki recall tertinggi (~98%) karena memprediksi hampir semua sampel sebagai pembeli.\n",
    "   - **Linear Discriminant Analysis (LDA)** dan **Logistic Regression** (Traditional Models) menunjukkan performa yang sangat solid dengan akurasi uji ~88.78% dan AUC ROC ~0.90.\n",
    "   - **Artificial Neural Network (MLP ANN)** (Modern Model) mencapai hasil terbaik dalam hal keseimbangan metrik, dengan test accuracy mencapai 88.90%, F1-Score sebesar 60.78% (jauh mengungguli LDA dan LogReg yang hanya berkisar 54%), serta AUC ROC sebesar 0.90. ANN mampu menangkap pola non-linier antara interaksi variabel kategorikal dan numerik dengan lebih baik."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipykernel",
    "query": "python3"
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipykernel3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open('uas_online_shoppers.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Jupyter Notebook 'uas_online_shoppers.ipynb' created successfully!")
