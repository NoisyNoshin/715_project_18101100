# VAE for Hybrid-Language Music Lyrics Clustering

Unsupervised clustering of song lyrics using Variational Autoencoders (VAE) to extract latent representations, followed by K-Means clustering and comparison with PCA baseline.

**Goal**: Group similar lyrical styles (e.g., rap, pop, ballad, poetry) using text features from lyrics of various artists.

Current status (Jan 2026): Basic VAE + TF-IDF features + clustering + evaluation metrics implemented (easy task level).

The data/ directory contains all dataset-related files and is not fully tracked by Git. Inside it, the lyrics/ subdirectory stores raw song lyrics in text format (for example, adele.txt, amy-winehouse.txt). A .gitkeep file is included to preserve the directory structure even when the actual data files are excluded from version control.

The notebooks/ directory contains Jupyter notebooks used for experimentation and analysis. The main notebook, exploratory.ipynb, is intended for running experiments, visualizing results, and testing different configurations interactively.

All core implementation files are placed in the src/ directory. This directory is treated as a Python module using an __init__.py file. The file dataset.py handles data loading and TF-IDF feature extraction, while vae.py defines and trains the Variational Autoencoder model. Clustering-related methods, including K-Means and a PCA baseline, are implemented in clustering.py, and all clustering evaluation metrics are centralized in evaluation.py.

The results/ directory stores generated outputs from experiments. It includes a latent_visualization/ subdirectory for saved plots such as latent space visualizations, tracked using a .gitkeep file, and a clustering_metrics.csv file that records quantitative evaluation results.

At the root level, the .gitignore file specifies files and directories that should not be tracked by Git, such as large data files and generated outputs. The README.md file provides an overview of the project along with usage instructions, while requirements.txt lists all Python dependencies required to reproduce the experiments. An optional LICENSE file is included to define usage and distribution rights.


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
