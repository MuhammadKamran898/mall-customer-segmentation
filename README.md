# Mall Customer Segmentation

An end-to-end machine learning project that uses **KMeans Clustering** to segment mall customers based on their **Annual Income** and **Spending Score**.

## 📌 Project Overview

The goal of this project is to group customers into different segments based on their purchasing-related characteristics.

The project uses **Unsupervised Machine Learning**, specifically **KMeans Clustering**, to identify customer groups without predefined labels.

The workflow covers the complete machine learning process:

```text
Dataset
   ↓
Data Exploration
   ↓
Feature Selection
   ↓
Feature Scaling
   ↓
Elbow Method
   ↓
Silhouette Score
   ↓
KMeans Clustering
   ↓
Cluster Visualization
   ↓
Cluster Analysis
   ↓
Model Saving
```

## 🎯 Objective

The main objective is to identify meaningful customer segments using:

* Annual Income (k$)
* Spending Score (1-100)

These segments can help understand different customer groups based on their income and spending behavior.

## 🤖 Machine Learning

### Algorithm

**KMeans Clustering**

KMeans is an unsupervised learning algorithm that divides data points into a predefined number of clusters based on their similarity.

### Choosing the Number of Clusters

Two evaluation techniques were used:

* **Elbow Method** — used to examine the relationship between the number of clusters and WCSS (Within-Cluster Sum of Squares).
* **Silhouette Score** — used to evaluate how well-separated and internally consistent the clusters are.

Based on the analysis, **5 clusters** were selected for the final model.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook

## 📊 Dataset

The project uses the **Mall Customers** dataset.

The main features used for clustering are:

| Feature                | Description                                      |
| ---------------------- | ------------------------------------------------ |
| Annual Income (k$)     | Customer's annual income in thousands of dollars |
| Spending Score (1-100) | Customer's spending score                        |

## 🔄 Machine Learning Pipeline

The final model uses a Scikit-learn `Pipeline` containing:

```text
StandardScaler
      ↓
KMeans
```

Using a pipeline ensures that the same preprocessing applied during training is also applied when making predictions.

## 📁 Project Structure

```text
mall-customer-segmentation/
│
├── data/
│   └── Mall_Customers.csv
│
├── notebook/
│   └── Project(3).ipynb
│
├── model/
│   └── customer_segmentation_model.joblib
│
├── api/
│   └── main.py
│
├── streamlit/
│   └── app.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

> The `api` and `streamlit` folders are included as part of the planned deployment structure if those components have been added to the project.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/mall-customer-segmentation.git
```

Move into the project directory:

```bash
cd mall-customer-segmentation
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebook

Open the notebook:

```text
notebook/Project(3).ipynb
```

Run the cells to:

1. Load the dataset
2. Explore the data
3. Select features
4. Scale the features
5. Apply the Elbow Method
6. Calculate Silhouette Scores
7. Train the KMeans model
8. Visualize the clusters
9. Analyze cluster profiles
10. Save the trained model

## 📈 Results

The final model divides customers into **5 clusters** based on their annual income and spending score.

The project also visualizes:

* Customer clusters
* Cluster centroids
* Elbow curve
* Silhouette scores
* Cluster profiles

## 💾 Saved Model

The trained machine learning pipeline is saved using Joblib:

```text
customer_segmentation_model.joblib
```

The saved pipeline contains both:

```text
StandardScaler
+
KMeans
```

This allows the same preprocessing and clustering logic to be reused when making predictions on new customer data.

## 📚 What I Learned

Through this project, I practiced:

* Data loading and exploration with Pandas
* Feature selection
* Feature scaling with `StandardScaler`
* Unsupervised Machine Learning
* KMeans clustering
* Elbow Method
* WCSS
* Silhouette Score
* Cluster visualization
* Cluster analysis
* Scikit-learn Pipelines
* Saving trained ML models with Joblib
* Preparing an ML model for deployment

## 🔮 Future Improvements

Possible future improvements include:

* Build a FastAPI inference API
* Build a Streamlit frontend
* Add real-time customer predictions
* Add interactive cluster visualizations
* Deploy the application
* Add more customer features for segmentation

## 👨‍💻 Author

**Muhammad Kamran Tariq**

Computer Science Student
Interested in Python, Data Science, Machine Learning, and AI.

---

⭐ If you find this project useful, feel free to explore the repository.
