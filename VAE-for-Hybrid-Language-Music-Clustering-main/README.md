# VAE for Hybrid-Language Music Lyrics Clustering

Unsupervised clustering of song lyrics using Variational Autoencoders (VAE) to extract latent representations, followed by K-Means clustering and comparison with PCA baseline.

**Goal**: Group similar lyrical styles (e.g., rap, pop, ballad, poetry) using text features from lyrics of various artists.

Current status (Jan 2026): Basic VAE + TF-IDF features + clustering + evaluation metrics implemented (easy task level).

VAE-for-Hybrid-Language-Music-Clustering/
│
├── data/                         # Data directory (NOT fully tracked)
│   └── lyrics/                   # Raw lyrics text files
│       ├── adele.txt
│       ├── amy-winehouse.txt
│       └── .gitkeep              # Keeps empty folder in Git
│
├── notebooks/
│   └── exploratory.ipynb         # Main experiment notebook
│
├── src/                          # Source code
│   ├── __init__.py               # Makes src a Python module
│   ├── dataset.py                # Data loading + TF-IDF
│   ├── vae.py                    # VAE model & training
│   ├── clustering.py             # KMeans + PCA
│   └── evaluation.py             # Clustering metrics
│
├── results/                      # Generated outputs
│   ├── latent_visualization/
│   │   └── .gitkeep              # Keeps folder tracked
│   └── clustering_metrics.csv    # Evaluation results
│
├── .gitignore                    # Ignore large / generated files
├── README.md                     # Project overview & instructions
├── requirements.txt              # Python dependencies
└── LICENSE                       # (optional but recommended)


## Requirements

Python 3.10–3.14 recommended

Install all dependencies:

bash
pip install -r requirements.txt

## How to Run the Project
Recommended Way: Using the Notebook (Interactive & Visual)

Open VS Code → File → Open Folder → select the project root folder
(the one containing notebooks/, src/, etc.)
Open notebooks/exploratory.ipynb
Make sure you have selected the correct Python interpreter:
Ctrl + Shift + P → "Python: Select Interpreter"
Choose your global Python 3.14 or the virtual environment you created

## Run the cells in order (top to bottom):
Cell 1: Imports + path fix
Cell 2: Load & preprocess lyrics
Cell 3: Train VAE & extract latent features
Cell 4: Perform clustering
Cell 5: Unsupervised metrics
Cell 6: (optional) Supervised metrics if you have labels
Cell 7: Visualization & save results

Results will be saved automatically in results/ folder:
clustering_metrics.csv — comparison table
latent_visualization/vae_latent_clusters.png — 2D scatter plot

What Each File Does

## src/dataset.py
Loads all .txt lyric files and converts them into TF-IDF feature matrix (bag-of-words style)

## src/vae.py
Defines Variational Autoencoder model + training loop
Returns trained model that can encode lyrics → latent vectors (mu)

## src/clustering.py
Contains two clustering functions:
perform_clustering: K-Means on VAE latent features
pca_baseline: PCA dimensionality reduction + K-Means (for comparison)

## src/evaluation.py
Implements 6 clustering metrics:
Silhouette Score
Calinski-Harabasz Index
Davies-Bouldin Index
Adjusted Rand Index (ARI)
Normalized Mutual Information (NMI)
Cluster Purity
(First three are unsupervised, last three need ground truth labels)

## notebooks/exploratory.ipynb

Main interactive workflow:
Loads data
Trains VAE
Clusters
Evaluates
Visualizes latent space in 2D
