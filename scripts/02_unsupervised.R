# ==============================================================================
# Script: 02_unsupervised.R
# Author: Martin Lee Chiah Herh (2401315)
# Tasks: 
# 1. PCA (PVE curve + Scree plot)
# 2. t-SNE (Scatter plot for non-linear patterns)
# 3. ICA (Independent underlying factors)
# 4. k-Means Clustering (Elbow method, Biplot, Profiling, Cross-Tab, Validation)
# ==============================================================================

cat("\n--- Running 02_unsupervised.R ---\n")

# 1. Load Required Libraries
library(Rtsne)       # For t-SNE
library(fastICA)     # For ICA 
library(cluster)     # For clustering algorithms
library(ucimlrepo)   # For fetching original untransformed data for profiling

# Ensure figure output directory exists
if(!dir.exists("outputs/figures/unsupervised")) dir.create("outputs/figures/unsupervised", recursive = TRUE)

# 2. Load the Pre-processed Training & Testing Data
train_data <- readRDS("outputs/data/train_data.rds")
test_data <- readRDS("outputs/data/test_data.rds")

# Separate features from the target variable
train_features <- train_data[, !names(train_data) %in% c("Revenue")]
train_revenue <- train_data$Revenue

test_features <- test_data[, !names(test_data) %in% c("Revenue")]
test_revenue <- test_data$Revenue

# Fetch original unscaled/untransformed data for cluster profiling later
cat("Fetching original untransformed dataset for cluster profiling...\n")
original_data <- fetch_ucirepo(id = 468)$data$original
# Note: You will need to apply the same chronological split (Jan-Oct) to match train_data rows

# ==============================================================================
# PART A: PCA (Linear Patterns)
# ==============================================================================
cat("\nRunning PCA...\n")

# Note: Data is already scaled in 01_data_engineering.r, so scale. = FALSE
pca_res <- prcomp(train_features, center = FALSE, scale. = FALSE)

# [TODO]: Calculate Proportion of Variance Explained (PVE)
# [TODO]: Plot PVE curve + Scree plot and save to outputs/figures/unsupervised/

# ==============================================================================
# PART B: t-SNE (Non-linear Patterns)
# ==============================================================================
cat("Running t-SNE...\n")

# Remove duplicate rows from features before t-SNE if any exist, as t-SNE requires unique rows
# tsne_res <- Rtsne(as.matrix(train_features), dims = 2, check_duplicates = FALSE)

# [TODO]: Create scatter plot to identify clusters and separability of buyers/non-buyers
# [TODO]: Save plot to outputs/figures/unsupervised/

# ==============================================================================
# PART C: ICA (Independent Components)
# ==============================================================================
cat("Running ICA...\n")

# ica_res <- fastICA(train_features, n.comp = 2)

# [TODO]: Extract IC scores to be used for clustering

# ==============================================================================
# PART D: k-Means Clustering
# ==============================================================================
cat("Running k-Means Clustering...\n")

# [TODO]: Find optimal k using the elbow method (WSS vs k)
# [TODO]: Save Elbow plot to outputs/figures/unsupervised/

# [TODO]: Run kmeans() with the optimal k

# [TODO]: Biplot clusters using the two most appropriate dimensions from PCA/t-SNE/ICA
# [TODO]: Save Biplot to outputs/figures/unsupervised/

# [TODO]: Generate Cluster Profiling Table using the mean values of the UNTRANSFORMED features
# [TODO]: Generate Cluster vs Target (Revenue) Cross-Tabulation

# ==============================================================================
# PART E: Validation
# ==============================================================================
cat("Validating clusters on testing set...\n")

# [TODO]: Apply the chosen clustering logic/transformations to test_features
# [TODO]: Perform Cross-Tabulation on the testing set to validate stability

# 5. Save Models 
# Note: t-SNE doesn't generate a traditional "model" that can be used to predict new data 
# with the Rtsne package, but we can save the results anyway for reproducibility.
# saveRDS(list(pca = pca_res, tsne = tsne_res, ica = ica_res, kmeans = kmeans_model), 
#         "outputs/models/unsupervised_models.rds")

cat("\n--- Completed 02_unsupervised.R ---\n")