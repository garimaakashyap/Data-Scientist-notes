# Machine Learning Algorithms - Complete Guide

## Table of Contents
1. [Support Vector Machine (SVM)](#support-vector-machine-svm)
2. [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
3. [Decision Trees](#decision-trees)
4. [Random Forests](#random-forests)
5. [XGBoost](#xgboost)
6. [K-Means Clustering](#k-means-clustering)
7. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)

---

## Support Vector Machine (SVM)

### What is SVM?
**Support Vector Machine (SVM)** finds the **best boundary (hyperplane)** that separates classes with maximum margin.

### Key Concepts

**1. Hyperplane:**
- Decision boundary that separates classes
- In 2D: a line
- In 3D: a plane
- In nD: a hyperplane

**2. Margin:**
- Distance between hyperplane and nearest data points
- **Larger margin = Better generalization**

**3. Support Vectors:**
- Data points closest to the hyperplane
- These define the margin

### Visual Example

```
Class 1: ○ ○ ○
         ╱     ╲
        ╱       ╲  ← Margin
       ╱         ╲
      ╱ Hyperplane╲
     ╱             ╲
    ╱               ╲
Class 2: ● ● ● ● ●
```

### Types of SVM

**1. Hard Margin SVM:**
- No misclassifications allowed
- Works only for linearly separable data

**2. Soft Margin SVM:**
- Allows some misclassifications
- Uses **C parameter** to control trade-off
- Higher C = stricter (fewer errors allowed)

**3. Kernel SVM:**
- Handles non-linear boundaries
- Maps data to higher dimensions
- Common kernels: RBF, Polynomial, Sigmoid

### Kernel Trick

**Problem:** Non-linear data can't be separated by a line

**Solution:** Transform data to higher dimension where it becomes linear

**Example:**
```
Original: x₁, x₂ (2D) → Can't separate with line
Transformed: x₁², x₂², √2x₁x₂ (3D) → Can separate with plane
```

### Common Kernels

**1. Linear Kernel:**
```
K(x, y) = x · y
```
- For linearly separable data
- Fastest

**2. Polynomial Kernel:**
```
K(x, y) = (x · y + 1)ᵈ
```
- d = degree
- Captures polynomial relationships

**3. RBF (Gaussian) Kernel:**
```
K(x, y) = exp(-γ||x - y||²)
```
- Most popular
- γ (gamma) controls influence radius
- Can handle complex boundaries

**4. Sigmoid Kernel:**
```
K(x, y) = tanh(α(x · y) + c)
```
- Similar to neural networks

### Python Implementation

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Sample data
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=100, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1,
    random_state=42
)

# Scale features (IMPORTANT for SVM!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Linear SVM
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(X_train, y_train)
linear_pred = linear_svm.predict(X_test)
print(f"Linear SVM Accuracy: {accuracy_score(y_test, linear_pred):.2f}")

# RBF Kernel SVM (non-linear)
rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
rbf_svm.fit(X_train, y_train)
rbf_pred = rbf_svm.predict(X_test)
print(f"RBF SVM Accuracy: {accuracy_score(y_test, rbf_pred):.2f}")

# Polynomial Kernel
poly_svm = SVC(kernel='poly', degree=3, C=1.0)
poly_svm.fit(X_train, y_train)
poly_pred = poly_svm.predict(X_test)
print(f"Polynomial SVM Accuracy: {accuracy_score(y_test, poly_pred):.2f}")

# Get support vectors
print(f"\nNumber of support vectors: {len(rbf_svm.support_vectors_)}")
```

### Hyperparameters

**C (Regularization):**
- Controls trade-off between margin size and classification errors
- **High C:** Smaller margin, fewer errors (may overfit)
- **Low C:** Larger margin, more errors (may underfit)
- Typical range: 0.1 to 100

**Gamma (for RBF kernel):**
- Controls influence of individual training examples
- **High gamma:** Tight boundaries (may overfit)
- **Low gamma:** Smooth boundaries (may underfit)
- Options: 'scale', 'auto', or float value

**Degree (for polynomial kernel):**
- Degree of polynomial
- Typical: 2, 3, 4

### Advantages
✅ Effective in high dimensions
✅ Memory efficient (uses only support vectors)
✅ Versatile (different kernels for different problems)
✅ Works well with clear margin of separation
✅ Handles non-linear data (with kernels)

### Disadvantages
❌ Doesn't perform well with large datasets
❌ Doesn't directly provide probability estimates
❌ Sensitive to feature scaling (must scale!)
❌ Requires careful hyperparameter tuning
❌ Slow on very large datasets

### Use Cases
- Text classification
- Image classification
- Face recognition
- Bioinformatics
- Handwriting recognition
- Gene classification

---

## K-Nearest Neighbors (KNN)

### What is KNN?
A simple algorithm that classifies/predicts based on the **K closest training examples**.

### How It Works

**For Classification:**
1. Find K nearest neighbors
2. Count votes from each class
3. Predict the majority class

**For Regression:**
1. Find K nearest neighbors
2. Predict average (or weighted average) of their values

### Example: Classification

**Problem:** Classify new point as red or blue

**Steps:**
1. Choose K (e.g., K=3)
2. Find 3 nearest neighbors
3. Count: 2 red, 1 blue
4. Predict: **Red** (majority)

### Distance Metrics

**1. Euclidean Distance (most common):**
```
d = √[(x₁-x₂)² + (y₁-y₂)²]
```

**2. Manhattan Distance:**
```
d = |x₁-x₂| + |y₁-y₂|
```

**3. Minkowski Distance:**
```
d = (|x₁-x₂|ᵖ + |y₁-y₂|ᵖ)^(1/p)
```
- p=1: Manhattan
- p=2: Euclidean

### Choosing K

**Small K (e.g., K=1):**
- Very sensitive to noise
- Complex decision boundary
- May overfit

**Large K (e.g., K=20):**
- Smooth decision boundary
- Less sensitive to noise
- May underfit

**Rule of thumb:** K = √n (where n = number of samples)

### Python Implementation

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Sample data: Predict if customer buys product
data = {
    'age': [25, 35, 45, 20, 30, 40, 50, 22, 38, 48],
    'income': [50000, 70000, 90000, 40000, 60000, 80000, 100000, 45000, 75000, 95000],
    'bought_product': [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

# Prepare data
X = df[['age', 'income']]
y = df['bought_product']

# IMPORTANT: Scale features for KNN!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Try different K values
for k in [1, 3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"K={k}: Accuracy = {accuracy:.2f}")

# Best K (let's use K=5)
best_knn = KNeighborsClassifier(n_neighbors=5)
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)
print(f"\nFinal Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Predict for new customer
new_customer = scaler.transform([[30, 65000]])
prediction = best_knn.predict(new_customer)
probability = best_knn.predict_proba(new_customer)
print(f"Prediction: {prediction[0]}")
print(f"Probability: {probability[0]}")

# KNN for Regression
knn_reg = KNeighborsRegressor(n_neighbors=5)
# Use for continuous target variable
```

### Weighted KNN

Instead of simple majority vote, weight neighbors by distance:
- Closer neighbors have more influence
- Weights = 1 / distance

```python
# Weighted KNN
knn_weighted = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance'  # Weight by inverse distance
)
```

### Advantages
✅ Simple to understand and implement
✅ No assumptions about data distribution
✅ Works well for non-linear problems
✅ Can be used for both classification and regression
✅ No training phase (lazy learning)

### Disadvantages
❌ Computationally expensive (stores all training data)
❌ Sensitive to irrelevant features
❌ Sensitive to scale of features (must normalize!)
❌ Performance degrades with high dimensions
❌ Slow prediction for large datasets

### Use Cases
- Recommendation systems
- Image recognition
- Credit scoring
- Pattern recognition
- Medical diagnosis

---

## Decision Trees

### What is a Decision Tree?
A tree-like model that makes decisions by asking a series of questions.

### Structure

**Components:**
- **Root Node:** Top decision
- **Internal Nodes:** Questions/conditions
- **Leaf Nodes:** Final predictions/classes
- **Branches:** Outcomes of decisions

### Example: Should I play tennis?

```
                    Outlook?
                   /    |    \
              Sunny  Overcast  Rain
                |       |        |
            Humidity?  Yes    Wind?
                |              |
            High  Normal    Strong  Weak
             |      |        |      |
            No     Yes      No    Yes
```

### How It Works

**1. Start at root**
**2. Ask question (e.g., "Is age > 30?")**
**3. Follow branch based on answer**
**4. Repeat until leaf node**
**5. Get prediction**

### Splitting Criteria

**1. Gini Impurity:**
```
Gini = 1 - Σ(pᵢ)²

where pᵢ = probability of class i
```
- Lower Gini = purer node
- Range: 0 to 1

**2. Entropy (Information Gain):**
```
Entropy = -Σ(pᵢ × log₂(pᵢ))

Information Gain = Entropy(parent) - Weighted Entropy(children)
```
- Higher information gain = better split

**3. Classification Error:**
```
Error = 1 - max(pᵢ)
```

### How Splitting Works

1. Try all features and all possible split points
2. Calculate impurity for each split
3. Choose split with **lowest impurity** (or highest information gain)
4. Repeat recursively

### Python Implementation

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt

# Sample data: Predict if person will buy product
data = {
    'age': ['young', 'young', 'middle', 'middle', 'middle', 'old', 'old'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes']
}
df = pd.DataFrame(data)

# Convert categorical to numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df_encoded = df.copy()
for col in df.columns:
    df_encoded[col] = le.fit_transform(df[col])

# Prepare data
X = df_encoded.drop('buys_computer', axis=1)
y = df_encoded['buys_computer']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
tree = DecisionTreeClassifier(
    max_depth=3,           # Maximum depth
    min_samples_split=2,    # Min samples to split
    min_samples_leaf=1,     # Min samples in leaf
    criterion='gini',       # or 'entropy'
    random_state=42
)
tree.fit(X_train, y_train)

# Make predictions
y_pred = tree.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Visualize tree
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': tree.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nFeature Importance:\n{feature_importance}")
```

### Hyperparameters

**max_depth:**
- Maximum depth of tree
- Prevents overfitting
- Lower = simpler model
- Typical: 3-10

**min_samples_split:**
- Minimum samples required to split a node
- Higher = simpler model
- Typical: 2-20

**min_samples_leaf:**
- Minimum samples required in a leaf node
- Higher = simpler model
- Typical: 1-10

**max_features:**
- Maximum features considered for split
- Can help with overfitting
- Options: 'sqrt', 'log2', or number

**criterion:**
- 'gini' or 'entropy'
- Gini is faster, entropy may be slightly better

### Advantages
✅ Easy to understand and interpret
✅ Requires little data preparation
✅ Handles both numerical and categorical data
✅ Can model non-linear relationships
✅ Feature importance is available
✅ No feature scaling needed

### Disadvantages
❌ Prone to overfitting
❌ Unstable (small data changes → different tree)
❌ Biased toward features with more levels
❌ Doesn't work well with imbalanced data
❌ Can create overly complex trees

### Use Cases
- Medical diagnosis
- Credit approval
- Customer segmentation
- Quality control
- Fraud detection

---

## Random Forests

### What is Random Forest?
An **ensemble method** that combines multiple decision trees to make better predictions.

### Key Idea

**"Wisdom of the Crowd"**
- One tree might be wrong
- Many trees voting together → more accurate

### How It Works

**Training:**
1. Create many decision trees (e.g., 100)
2. Each tree trained on **random subset** of data (bootstrap sampling)
3. Each split considers **random subset** of features
4. Trees vote for final prediction

**Prediction:**
- **Classification:** Majority vote
- **Regression:** Average of predictions

### Bootstrap Aggregating (Bagging)

**Process:**
1. Sample with replacement from training data
2. Train tree on this sample
3. Repeat many times
4. Combine predictions

**Why it works:**
- Reduces variance
- Each tree sees different data
- Errors average out

### Random Feature Selection

At each split:
- Consider only random subset of features (e.g., √n features)
- Prevents trees from being too similar
- Increases diversity

### Python Implementation

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# Sample data
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    n_redundant=2, n_classes=2, random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Max depth of trees
    min_samples_split=2,   # Min samples to split
    min_samples_leaf=1,    # Min samples in leaf
    max_features='sqrt',   # Features to consider (√n)
    random_state=42
)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': [f'Feature_{i}' for i in range(X.shape[1])],
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Most Important Features:\n{feature_importance.head()}")

# Compare with single decision tree
from sklearn.tree import DecisionTreeClassifier

single_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
single_tree.fit(X_train, y_train)
single_pred = single_tree.predict(X_test)
print(f"\nSingle Tree Accuracy: {accuracy_score(y_test, single_pred):.2f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

### Hyperparameters

**n_estimators:**
- Number of trees
- More trees = better (but slower)
- Typical: 100-500

**max_depth:**
- Maximum depth of each tree
- Prevents overfitting
- Typical: 10-20

**max_features:**
- Number of features to consider per split
- Options: 'sqrt', 'log2', or number
- Typical: 'sqrt' for classification, 'log2' for regression

**min_samples_split:**
- Minimum samples to split node
- Higher = simpler trees

**min_samples_leaf:**
- Minimum samples in leaf
- Higher = simpler trees

### Advantages
✅ Reduces overfitting (compared to single tree)
✅ Handles missing values well
✅ Provides feature importance
✅ Works well with default parameters
✅ Can handle large datasets
✅ Less sensitive to outliers

### Disadvantages
❌ Less interpretable than single tree
❌ Slower than single tree
❌ Requires more memory
❌ Can overfit with noisy data
❌ May not work well with very high-dimensional sparse data

### Use Cases
- Feature selection
- Image classification
- Stock market prediction
- Medical diagnosis
- Customer churn prediction
- Feature importance analysis

---

## XGBoost

### What is XGBoost?
**Extreme Gradient Boosting** - An advanced, optimized implementation of gradient boosting.

### Gradient Boosting Concept

**Idea:** Build trees **sequentially**, where each new tree corrects errors of previous trees.

**Process:**
1. Train first tree
2. Calculate errors (residuals)
3. Train second tree to predict errors
4. Combine predictions
5. Repeat

### Why XGBoost is Powerful

1. **Gradient Boosting:** Learns from mistakes
2. **Regularization:** Prevents overfitting
3. **Parallel Processing:** Fast training
4. **Handles Missing Values:** Automatically
5. **Tree Pruning:** More efficient

### Key Features

**1. Regularization:**
- L1 (Lasso) and L2 (Ridge) regularization
- Controls model complexity

**2. Shrinkage (Learning Rate):**
- Each tree contributes a fraction
- Slower learning = better generalization

**3. Column Sampling:**
- Random subset of features per tree
- Like Random Forest

**4. Tree Pruning:**
- Grows tree, then prunes backward
- More efficient

### Python Implementation

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# Sample data
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    n_redundant=2, n_classes=2, random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
xgb_model = xgb.XGBClassifier(
    n_estimators=100,        # Number of trees
    max_depth=6,             # Max depth
    learning_rate=0.1,       # Shrinkage
    subsample=0.8,           # Row sampling
    colsample_bytree=0.8,    # Column sampling
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': [f'Feature_{i}' for i in range(X.shape[1])],
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Most Important Features:\n{feature_importance.head()}")

# Plot feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(xgb_model, max_num_features=10)
plt.show()
```

### Key Hyperparameters

**n_estimators:**
- Number of boosting rounds
- Typical: 100-1000

**learning_rate (eta):**
- Step size shrinkage
- Lower = slower but more accurate
- Typical: 0.01-0.3

**max_depth:**
- Maximum tree depth
- Typical: 3-10

**subsample:**
- Fraction of samples used per tree
- Typical: 0.6-1.0

**colsample_bytree:**
- Fraction of features used per tree
- Typical: 0.6-1.0

**reg_alpha:**
- L1 regularization
- Higher = more regularization

**reg_lambda:**
- L2 regularization
- Higher = more regularization

### Advantages
✅ Often best performance on structured data
✅ Handles missing values automatically
✅ Fast and efficient
✅ Regularization prevents overfitting
✅ Feature importance available
✅ Works well with default parameters

### Disadvantages
❌ Requires hyperparameter tuning
❌ Can overfit if not tuned properly
❌ Less interpretable
❌ Sensitive to outliers
❌ Requires more memory than simpler models

### Use Cases
- Kaggle competitions (often winner!)
- Click-through rate prediction
- Fraud detection
- Customer churn prediction
- Recommendation systems
- Any structured data problem

### XGBoost vs Random Forest

| Aspect | Random Forest | XGBoost |
|-------|--------------|---------|
| **Method** | Bagging (parallel) | Boosting (sequential) |
| **Speed** | Fast | Slower but optimized |
| **Overfitting** | Less prone | More prone (needs tuning) |
| **Performance** | Good | Often better |
| **Interpretability** | Medium | Medium |

---

## K-Means Clustering

### What is K-Means?
An **unsupervised learning** algorithm that groups similar data points into **K clusters**.

### How It Works

**Algorithm:**
1. **Initialize:** Randomly place K centroids (cluster centers)
2. **Assign:** Assign each point to nearest centroid
3. **Update:** Move centroids to center of their points
4. **Repeat:** Steps 2-3 until convergence

### Step-by-Step Example

**Data:** Points on 2D plane
**K = 3** (want 3 clusters)

**Iteration 1:**
- Random centroids: C₁, C₂, C₃
- Assign points to nearest centroid
- Move centroids to center of assigned points

**Iteration 2:**
- New centroids
- Reassign points
- Update centroids

**Continue until centroids don't move much**

### Distance Metric

**Euclidean Distance:**
```
d = √[(x₁-x₂)² + (y₁-y₂)²]
```

### Choosing K

**1. Elbow Method:**
- Plot K vs Within-Cluster Sum of Squares (WCSS)
- Look for "elbow" in curve
- Elbow = optimal K

**2. Domain Knowledge:**
- Know how many groups you expect
- Example: Customer segments (3-5 groups)

**3. Silhouette Score:**
- Measures how well points fit clusters
- Higher = better
- Range: -1 to 1

### Python Implementation

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data: Customer spending
data = {
    'annual_income': [15000, 25000, 35000, 45000, 55000, 65000, 75000, 85000, 95000, 105000],
    'spending_score': [39, 81, 6, 77, 40, 6, 26, 29, 78, 2]
}
df = pd.DataFrame(data)

# Scale features (IMPORTANT for K-Means!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Find optimal K using Elbow Method
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method')

# Calculate Silhouette Scores
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.tight_layout()
plt.show()

# Best K (let's use K=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Get cluster labels
df['cluster'] = kmeans.labels_

# Centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
print(f"Centroids:\n{pd.DataFrame(centroids, columns=df.columns[:-1])}")

# Visualize clusters
plt.scatter(df['annual_income'], df['spending_score'], 
           c=df['cluster'], cmap='viridis', s=100)
plt.scatter(centroids[:, 0], centroids[:, 1], 
           marker='x', s=200, c='red', linewidths=3, label='Centroids')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('K-Means Clustering (K=3)')
plt.legend()
plt.show()

# Cluster characteristics
print(f"\nCluster Summary:\n{df.groupby('cluster').mean()}")
```

### Limitations

**1. Assumes Spherical Clusters:**
- Works best for circular clusters
- Struggles with elongated or irregular shapes

**2. Sensitive to Initialization:**
- Different starting points → different results
- Solution: Use k-means++ initialization

**3. Requires K to be Specified:**
- Must know number of clusters beforehand

**4. Sensitive to Scale:**
- Must normalize features
- Features with larger scales dominate

### Advantages
✅ Simple and fast
✅ Works well for spherical clusters
✅ Scales to large datasets
✅ Easy to interpret

### Disadvantages
❌ Assumes clusters are spherical
❌ Sensitive to outliers
❌ Requires K to be specified
❌ Sensitive to initialization
❌ Sensitive to scale

### Use Cases
- Customer segmentation
- Image segmentation
- Document clustering
- Market research
- Anomaly detection
- Gene clustering

---

## Principal Component Analysis (PCA)

### What is PCA?
A **dimensionality reduction** technique that finds the most important directions (components) in data.

### Why Use PCA?

**Problems with High Dimensions:**
1. **Curse of Dimensionality:** More features → need exponentially more data
2. **Visualization:** Can't visualize >3 dimensions
3. **Overfitting:** Too many features → model memorizes
4. **Noise:** Many features may be irrelevant

**Solution:** Reduce dimensions while keeping most information

### How PCA Works

**Step 1: Standardize Data**
- Mean = 0, STD = 1
- All features on same scale

**Step 2: Calculate Covariance Matrix**
- Shows relationships between features

**Step 3: Find Eigenvectors and Eigenvalues**
- Eigenvectors = principal components (directions)
- Eigenvalues = variance explained

**Step 4: Choose Top Components**
- Select components with highest eigenvalues
- These explain most variance

**Step 5: Transform Data**
- Project data onto new components

### Key Concepts

**1. Principal Components:**
- New features that are linear combinations of original features
- PC1 explains most variance
- PC2 explains second most, etc.

**2. Explained Variance:**
- How much information each component captures
- Sum of all = 100%

**3. Dimensionality Reduction:**
- Original: n features
- After PCA: k features (k < n)
- Keep components that explain most variance

### Example: 2D to 1D

**Original Data:** 2 features (height, weight)
**After PCA:** 1 component (combines height and weight)

**PC1** might be: 0.7×height + 0.7×weight
- Captures most variation in data

### Python Implementation

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load sample data
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Standardize data (IMPORTANT!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Explained Variance by Component:")
for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%) - Cumulative: {cum_var:.4f} ({cum_var*100:.2f}%)")

# Scree Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.xticks(range(1, len(explained_variance) + 1))

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.legend()
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.tight_layout()
plt.show()

# Reduce to 2D for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# Visualize in 2D
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis', s=50)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}% variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}% variance)')
plt.title('PCA: 4D → 2D Visualization')
plt.colorbar(scatter)
plt.show()

# Component loadings (how original features contribute)
components_df = pd.DataFrame(
    pca_2d.components_.T,
    columns=['PC1', 'PC2'],
    index=feature_names
)
print(f"\nComponent Loadings:\n{components_df}")

# Reduce to keep 95% variance
pca_95 = PCA(n_components=0.95)  # Keep 95% variance
X_pca_95 = pca_95.fit_transform(X_scaled)
print(f"\nOriginal dimensions: {X.shape[1]}")
print(f"Reduced dimensions (95% variance): {X_pca_95.shape[1]}")
```

### How Many Components to Keep?

**1. Scree Plot:**
- Look for "elbow"
- Keep components before elbow

**2. Cumulative Variance:**
- Keep components until cumulative variance ≥ 0.95 (95%)
- Common threshold: 80-95%

**3. Kaiser Criterion:**
- Keep components with eigenvalue > 1

### Advantages
✅ Reduces overfitting
✅ Removes multicollinearity
✅ Improves visualization
✅ Speeds up training
✅ Reduces noise

### Disadvantages
❌ Less interpretable (components are combinations)
❌ Information loss (if too few components)
❌ Assumes linear relationships
❌ Sensitive to scaling

### Use Cases
- Image compression
- Feature extraction
- Data visualization
- Noise reduction
- Speeding up algorithms
- Face recognition

### Important Notes

**1. Always Standardize First:**
- PCA is sensitive to scale
- Features with larger values dominate

**2. Information Loss:**
- Reducing dimensions = losing some information
- Balance between reduction and information retention

**3. Not for All Problems:**
- Works best with linear relationships
- For non-linear, consider t-SNE or UMAP

---

## Algorithm Comparison Summary

### Supervised Learning Algorithms

| Algorithm | Type | Best For | Pros | Cons |
|-----------|------|----------|-----|------|
| **SVM** | Classification | High dimensions, clear margins | Effective, versatile | Slow on large data |
| **KNN** | Both | Non-linear, local patterns | Simple, no assumptions | Slow, sensitive to scale |
| **Decision Tree** | Both | Interpretable rules | Easy to understand | Prone to overfitting |
| **Random Forest** | Both | General purpose | Robust, feature importance | Less interpretable |
| **XGBoost** | Both | Best performance | Often best accuracy | Requires tuning |

### Unsupervised Learning Algorithms

| Algorithm | Type | Best For | Pros | Cons |
|-----------|------|----------|-----|------|
| **K-Means** | Clustering | Spherical clusters | Simple, fast | Assumes spherical |
| **PCA** | Dimensionality Reduction | Linear relationships | Reduces dimensions | Information loss |

### Quick Decision Guide

**Choose SVM if:**
- Clear margin of separation
- High-dimensional data
- Non-linear (with kernel)

**Choose KNN if:**
- Non-linear patterns
- Local patterns matter
- Small to medium dataset

**Choose Decision Tree if:**
- Need interpretability
- Non-linear relationships
- Mixed data types

**Choose Random Forest if:**
- Want robust performance
- Need feature importance
- General purpose

**Choose XGBoost if:**
- Want best performance
- Structured data
- Can tune hyperparameters

**Choose K-Means if:**
- Unlabeled data
- Spherical clusters
- Know number of clusters

**Choose PCA if:**
- Too many features
- Want to visualize
- Reduce dimensionality

---

## Key Takeaways

1. **SVM** finds best separating boundary with maximum margin
2. **KNN** predicts based on nearest neighbors (must scale!)
3. **Decision Trees** make decisions through series of questions
4. **Random Forest** combines many trees (bagging)
5. **XGBoost** builds trees sequentially to correct errors (boosting)
6. **K-Means** groups similar data points into clusters
7. **PCA** reduces dimensions while keeping information

### Best Practices

1. **Always scale features** for distance-based algorithms (KNN, K-Means, PCA, SVM)
2. **Split data** into train/test sets
3. **Tune hyperparameters** for best performance
4. **Evaluate properly** using appropriate metrics
5. **Start simple** (Decision Tree) before complex (XGBoost)
6. **Understand your data** before choosing algorithm
7. **Feature engineering** often more important than algorithm choice
8. **Cross-validation** for reliable performance estimates

---

*Master these algorithms to become a proficient data scientist!*

