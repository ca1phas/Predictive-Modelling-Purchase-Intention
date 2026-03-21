# ==============================================================================
# Script: 02_unsupervised.R
# Author: Martin Lee Chiah Herh (2302551)
# Tasks: 
# 1. PCA (PVE curve + Scree plot)
# 2. t-SNE (Scatter plot for non-linear patterns)
# 3. ICA (Independent underlying factors)
# 4. k-Means Clustering (Elbow method, Biplot, Profiling, Cross-Tab, Validation)
# ==============================================================================


# Note: t-SNE doesn't generate a traditional "model" that can be used to predict new data 
# with the Rtsne package, but we can save the results anyway for reproducibility.
# saveRDS(list(pca = pca_res, tsne = tsne_res, ica = ica_res, kmeans = kmeans_model), 
#         "outputs/models/unsupervised_models.rds")


#Dimensional Reduction and Clustering
#1.PCA
cat("\n--- Running 02_unsupervised.R ---\n")

library(Rtsne)
library(fastICA)
if(!dir.exists("outputs/figures/unsupervised")) {
  dir.create("outputs/figures/unsupervised", recursive = TRUE)}

#read data
train_data <- readRDS("outputs/data/train_data.rds")
test_data <- readRDS("outputs/data/test_data.rds")

train_features <- train_data[, !names(train_data) %in% c("Revenue")]
train_revenue <- train_data$Revenue

test_features <- test_data[, !names(test_data) %in% c("Revenue")]
test_revenue <- test_data$Revenue

#PCA
cat("Running PCA...\n")
pca_res <- prcomp(train_features)

#PVE
pve <- (pca_res$sdev^2) / sum(pca_res$sdev^2)
cum_pve <- cumsum(pve)

#PVE Plot
png("outputs/figures/unsupervised/pve_plot.png")
plot(pve, type="b",
     main="PVE Plot",
     xlab="Principal Component",
     ylab="Proportion of Variance Explained")
dev.off()

# Scree Plot
png("outputs/figures/unsupervised/scree_plot.png")
plot(pca_res, type="l", main="Scree Plot")
dev.off()

#PCA Biplot
png("outputs/figures/unsupervised/pca_biplot.png")
biplot(pca_res)
dev.off()


#2.t-SNE
cat("Running t-SNE...\n")
set.seed(1)
tsne_res <- Rtsne(as.matrix(train_features))
png("outputs/figures/unsupervised/tsne_plot.png")
plot(tsne_res$Y,
     col = ifelse(train_revenue == "Yes", "red", "blue"),
     pch = 19,
     xlab = "t-SNE 1",
     ylab = "t-SNE 2",
     main = "t-SNE Scatter Plot")
legend("topright", legend=c("Buyer","Non-Buyer"),
       col=c("red","blue"), pch=19)
dev.off()

#3.ICA
ica_res <- fastICA(train_features, n.comp = 2)
png("outputs/figures/unsupervised/ica_plot.png")
plot(ica_res$S,
     col = ifelse(train_revenue == "Yes", "red", "blue"),
     pch = 19,
     xlab = "IC1",
     ylab = "IC2",
     main = "ICA Scatter Plot")
legend("topright", legend=c("Buyer","Non-Buyer"),
       col=c("red","blue"), pch=19)
dev.off()

#k-means
cat("Running k-Means...\n")
wss <- numeric(10)
for (k in 1:10) {
  wss[k] <- kmeans(train_features, centers = k, nstart = 20)$tot.withinss
}

png("outputs/figures/unsupervised/elbow_plot.png")
plot(1:10, wss, type="b",
     xlab="Number of Clusters",
     ylab="Within Sum of Squares",
     main="Elbow Method")
dev.off()

# choose k=3
kmeans_model <- kmeans(train_features, centers = 3, nstart = 25)
clusters <- kmeans_model$cluster

# Cluster plot using PCA space
pc_scores <- pca_res$x[,1:2]

png("outputs/figures/unsupervised/kmeans_clusters.png")
plot(pc_scores,
     col = clusters,
     pch = 19,
     xlab = "PC1",
     ylab = "PC2",
     main = "k-Means Clusters (PCA Space)")
dev.off()

# Cross-tabulation Cluster vs Revenue
cat("\nCluster vs Revenue (Training Set):\n")
print(table(clusters, train_revenue))
print(prop.table(table(clusters, train_revenue), 1))

#5.Validation on Test Set
cat("Validating on Test Set...\n")

centers <- kmeans_model$centers

test_cluster <- apply(test_features, 1, function(x) {
  which.min(colSums((t(centers) - x)^2))
})

cat("\nCluster vs Revenue (Test Set):\n")
print(table(test_cluster, test_revenue))
print(prop.table(table(test_cluster, test_revenue), 1))

#Save Models
if(!dir.exists("outputs/models")) dir.create("outputs/models", recursive = TRUE)

saveRDS(list(
  pca = pca_res,
  tsne = tsne_res,
  ica = ica_res,
  kmeans = kmeans_model
), "outputs/models/unsupervised_models.rds")

cat("\n--- Completed 02_unsupervised.R ---\n")
