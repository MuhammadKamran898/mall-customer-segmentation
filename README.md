# Mall Customer Segmentation

An end-to-end unsupervised machine-learning project that explores mall customer groups using **K-Means** and **DBSCAN**, evaluates cluster quality, profiles the resulting segments, and serves live K-Means assignments through **FastAPI** with a **Streamlit** interface.

> **Current deployment choice:** K-Means (`k=5`) is used for online predictions. DBSCAN is included as an offline comparison in the notebook because standard scikit-learn DBSCAN does not provide a native `predict()` method for assigning unseen customers to an already fitted cluster.

## Project Highlights

- Performed exploratory data analysis on the 200-row Mall Customers dataset.
- Selected annual income and spending score as the two clustering features.
- Standardized the features with `StandardScaler` before distance-based clustering.
- Applied the Elbow method and silhouette analysis to compare K-Means cluster counts from 2 to 10.
- Compared K-Means and DBSCAN using silhouette score, Davies–Bouldin index, Calinski–Harabasz score, cluster counts, and DBSCAN noise percentage.
- Analysed cluster sizes and profiles, including mean age, annual income, and spending score.
- Saved and reloaded the complete scaler-plus-K-Means pipeline with Joblib.
- Built a FastAPI service with input validation, model-readiness endpoints, model information, and segment descriptions.
- Built a Streamlit UI that calls the API and displays a cluster, segment description, cluster-centre values, and training cluster size.
- Added API contract tests for predictions, input validation, and health endpoints.

## Results

### K-Means: selection of the number of clusters

The notebook fits K-Means for candidate values of `k` and evaluates silhouette scores for `k=2` through `k=10`.

| Number of clusters (`k`) | Silhouette score |
|---:|---:|
| 2 | 0.321271 |
| 3 | 0.466585 |
| 4 | 0.493907 |
| **5** | **0.554657** |
| 6 | 0.539880 |
| 7 | 0.528149 |
| 8 | 0.455215 |
| 9 | 0.457085 |
| 10 | 0.443171 |

The five-cluster configuration achieved the highest silhouette score among the tested values. A silhouette score is an internal clustering metric; it is **not** classification accuracy or the percentage of customers correctly labelled.

### K-Means vs. DBSCAN

The notebook evaluates a defined grid of DBSCAN `eps` and `min_samples` values using the same two standardized features. It selects a comparison candidate from configurations that produce between 2 and 8 clusters and label no more than 20% of observations as noise.

| Metric | K-Means (`k=5`) | DBSCAN (`eps=0.45`, `min_samples=10`) |
|---|---:|---:|
| Number of clusters | 5 | 5 |
| Customers labelled as noise | 0 (0%) | 29 (14.5%) |
| Silhouette score | 0.5547 | 0.5736 |
| Davies–Bouldin index | 0.5722 | 0.4830 |
| Calinski–Harabasz score | 248.6493 | 245.1439 |
| Customers included in the metric calculation | All 200 | 171 non-noise customers |

**Interpretation caveat:** The DBSCAN metrics are calculated only on non-noise customers, while the K-Means metrics include every customer. The scores are therefore not a strict apples-to-apples comparison. DBSCAN's slightly higher silhouette score does not by itself establish that it is the better choice; it leaves 14.5% of customers unassigned as noise. Compare the noise fraction, cluster sizes, visual structure, and intended application as well as internal metrics.

K-Means remains the deployed algorithm because the saved K-Means pipeline supports assigning a new customer's features with `predict()`. Standard DBSCAN does not provide a built-in equivalent for assigning new observations to an existing fitted clustering.

### K-Means cluster profiles

The following profile values were computed from the original fitted five-cluster K-Means model. Cluster IDs are zero-based in the model and API; the Streamlit interface displays cluster numbers as one-based labels for readability.

| API cluster ID | Customers in cluster | Mean age | Mean annual income (k$) | Mean spending score | Descriptive interpretation |
|---:|---:|---:|---:|---:|---|
| 0 | 81 | 42.72 | 55.30 | 49.52 | Middle income, average spending |
| 1 | 39 | 32.69 | 86.54 | 82.13 | High income, high spending |
| 2 | 22 | 25.27 | 25.73 | 79.36 | Low income, high spending |
| 3 | 35 | 41.11 | 88.20 | 17.11 | High income, low spending |
| 4 | 23 | 45.22 | 26.30 | 20.91 | Low income, low spending |

These descriptions are interpretations of the cluster centroids, not ground-truth segment labels provided by the dataset. Cluster identifiers can change when a model is refitted, even where the overall clustering structure remains similar.

## Dataset

The project uses `Mall_Customers.csv`, which contains **200 rows and 5 columns**.

| Column | Description | Used to fit the clustering model? |
|---|---|---|
| `CustomerID` | Customer record identifier | No |
| `Gender` | Recorded gender | No |
| `Age` | Customer age | No; used for post-clustering profile summaries |
| `Annual Income (k$)` | Annual income in thousands of dollars | Yes |
| `Spending Score (1-100)` | Spending score on a 1–100 scale | Yes |

The supplied CSV was inspected and contained no missing values or duplicate rows. Annual income ranges from 15 to 137 (k$), and spending scores range from 1 to 99. The model intentionally uses only annual income and spending score; introducing age or gender into clustering would change the modelling setup and should be evaluated as a separate experiment.

**Dataset provenance:** The supplied files did not include a verified source URL or licence. Add the original dataset source and licence here before publishing or redistributing the dataset publicly.

## Methodology

1. **Load and inspect data:** Read the CSV and check its shape, data types, missing values, and available columns.
2. **Select features:** Use `Annual Income (k$)` and `Spending Score (1-100)` for clustering.
3. **Scale features:** Apply `StandardScaler` so income magnitude does not dominate distance calculations merely because of its numerical scale.
4. **Select K-Means cluster count:** Use WCSS/inertia in the Elbow method and silhouette scores for `k=2..10`.
5. **Fit K-Means:** Train a pipeline containing `StandardScaler` and `KMeans(n_clusters=5, init="k-means++", random_state=42, n_init=10)`.
6. **Compare DBSCAN:** Explore a grid of `eps` and `min_samples` settings on the same scaled features. Report cluster counts and noise, and calculate silhouette, Davies–Bouldin, and Calinski–Harabasz scores on non-noise points where valid.
7. **Interpret clusters:** Visualize cluster assignments and centroids, and calculate counts and profile means.
8. **Save the pipeline:** Serialize the fitted scaler-plus-K-Means pipeline using Joblib, reload it, and verify that its predictions match the in-memory pipeline.
9. **Serve predictions:** Validate a new customer's income and spending score in FastAPI, assign a K-Means cluster, and return the cluster ID, a centroid-based description, centroid values, and training cluster size.

## Architecture

```text
Mall_Customers.csv
        |
        v
Project_mall_improved.ipynb
  EDA + Scaling + Elbow/Silhouette
  K-Means vs DBSCAN + Cluster Profiling
        |
        v
model/customer_segmentation_model.joblib
        |
        v
FastAPI (main.py)  <--- POST /predict
        |
        v
Streamlit UI (app.py)
```

DBSCAN is currently part of the notebook's offline analysis; it is not the model used by the prediction endpoint.

## Project Structure

The improved files are organized to support running the UI and API from the project root:

```text
mall-customer-segmentation/
├── Mall_Customers.csv
├── Project_mall_improved.ipynb
├── model/
│   └── customer_segmentation_model.joblib
├── main.py
├── app.py
├── test_prediction_mall_improved.py
├── requirements_mall_segmentation_improved.txt
└── README.md
```

The notebook can find the CSV in common locations, including the project root and a `data/` directory. The API resolves the model artifact from common adjacent `model/` paths or from the `CUSTOMER_SEGMENTATION_MODEL_PATH` environment variable.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mall-customer-segmentation.git
cd mall-customer-segmentation
```

Replace `YOUR_USERNAME` with your GitHub username and adjust the repository URL to match your repository.

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements_mall_segmentation_improved.txt
```

For reproducible model loading, record the Python and library versions used to train and serve the Joblib artifact. The dependency file currently lists packages without exact version pins; use a tested, compatible scikit-learn environment when loading the saved model.

## Train and save the model

Open `Project_mall_improved.ipynb` and run all cells from top to bottom. The notebook loads `Mall_Customers.csv`, performs the analysis and clustering comparison, fits the original five-cluster K-Means pipeline, and saves the complete fitted pipeline to:

```text
model/customer_segmentation_model.joblib
```

The scaler is included in the pipeline, so incoming raw income and spending-score values are transformed consistently before K-Means assigns a cluster.

## Run the FastAPI backend

From the project root, start the API:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

Interactive API documentation is available at `http://127.0.0.1:8001/docs` while the server is running.

If the saved model is stored elsewhere, set `CUSTOMER_SEGMENTATION_MODEL_PATH` before starting the API.

Windows PowerShell:

```powershell
$env:CUSTOMER_SEGMENTATION_MODEL_PATH = "C:\absolute\path\customer_segmentation_model.joblib"
uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

macOS/Linux:

```bash
export CUSTOMER_SEGMENTATION_MODEL_PATH="/absolute/path/customer_segmentation_model.joblib"
uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Returns basic API information and the deployed algorithm |
| `GET` | `/health/live` | Checks whether the API process responds |
| `GET` | `/health/ready` | Checks whether the model is loaded and ready |
| `GET` | `/health` | Backward-compatible readiness endpoint |
| `GET` | `/model-info` | Reports the active K-Means algorithm, cluster count, features, and DBSCAN deployment status |
| `POST` | `/predict` | Returns the cluster assignment and descriptive details for a customer |

### Example request

```json
{
  "annual_income": 60,
  "spending_score": 50
}
```

### Example response shape

```json
{
  "annual_income": 60.0,
  "spending_score": 50.0,
  "cluster": 0,
  "segment": "Middle-income, average-spending",
  "cluster_center_annual_income": 55.3,
  "cluster_center_spending_score": 49.52,
  "training_cluster_size": 81
}
```

This is an illustrative response schema based on the recorded cluster profile; the actual cluster for an input is determined by the loaded model artifact. Cluster IDs are zero-based in the API, while the UI displays cluster ID + 1.

The API accepts annual income values from 15 to 137 (k$) and spending scores from 1 to 99, matching the observed ranges in the supplied dataset. Requests outside those ranges are rejected by Pydantic validation.

## Run the Streamlit frontend

Open a second terminal with the same virtual environment activated and run:

```bash
streamlit run app.py
```

The UI calls `http://127.0.0.1:8001` by default. To configure another API address, set `API_URL` before launching Streamlit.

Windows PowerShell:

```powershell
$env:API_URL = "http://127.0.0.1:8001"
streamlit run app.py
```

macOS/Linux:

```bash
export API_URL="http://127.0.0.1:8001"
streamlit run app.py
```

The UI displays the assigned cluster, a descriptive label inferred from centroid rankings, the cluster-centre income and spending score, and the number of training customers assigned to the cluster. These labels are interpretations rather than predictions of future customer behaviour.

## Run tests

From the project root:

```bash
python -m pytest test_prediction_mall_improved.py -q
```

The tests exercise the FastAPI contract, including valid predictions, invalid ranges, and health endpoints. For deployment confidence, also verify that the real model artifact loads successfully and test the full Streamlit-to-API workflow in the target environment.

## Limitations and future work

- **Small dataset:** 200 records may not represent customers in other malls, locations, or time periods.
- **Limited feature set:** clustering uses only annual income and spending score. Age and gender are excluded from the feature space in the current model.
- **Unsupervised evaluation:** there are no ground-truth customer-segment labels, so the project uses internal cluster-quality metrics rather than classification accuracy.
- **DBSCAN noise:** the selected candidate labels 14.5% of records as noise; its quality metrics exclude those records, so compare results with this limitation in mind.
- **Deployment choice:** K-Means is used for inference because it supports prediction on new observations; DBSCAN is an offline comparison in the notebook.
- **Data and market context:** the segments are descriptive and do not imply causation or guarantee customer response to a marketing strategy.
- **Production hardening:** package versions should be pinned, model and dataset provenance documented, and tests run in CI before public deployment.
- **Dataset source and licence:** verify and cite the original dataset source before redistribution.

## What I learned

- Exploratory data analysis with pandas and visualisation with Matplotlib.
- Feature scaling and distance-based clustering.
- Choosing a K-Means cluster count using WCSS/Elbow and silhouette analysis.
- Comparing centroid-based K-Means with density-based DBSCAN.
- Evaluating clustering with silhouette, Davies–Bouldin, and Calinski–Harabasz metrics.
- Interpreting cluster centroids, sizes, and noise points without ground-truth labels.
- Building a reusable scikit-learn pipeline and serializing it with Joblib.
- Building a FastAPI model-serving endpoint with Pydantic validation.
- Connecting a Streamlit user interface to a local prediction API.

## Author

**Muhammad Kamran Tariq** 

---

If you find this project useful, feel free to explore the repository.
