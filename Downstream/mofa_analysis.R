library(matrixStats)
library(readr)
library(tibble)
library(reticulate)
library(dplyr)
library(writexl)
library(ggplot2)
library(ggpubr)
use_condaenv("mofa_env", required = TRUE) 
library(MOFA2)
# =============================================================
# 1. PREPARE DATASETS 
# =============================================================
gexp <- read_tsv("../GBM.transcriptome__ht_hg_u133a__broad_mit_edu__Level_3__gene_rma__data.data.txt")
gexp <- gexp[-1,]
gexp <- column_to_rownames(gexp, var = "Hybridization REF")
colnames(gexp) <- sapply(colnames(gexp), substring, 1, 12)

rma_matrix <- t(gexp) 
storage.mode(rma_matrix) <- "numeric"

rad <- read.csv("../TCGA_GBM_radiomicFeatures.csv")
rad <- column_to_rownames(rad, var = "ID")
rad <- rad[, 2:ncol(rad)] 
NA_threshold <- 0 
na_counts_rad <- colSums(is.na(rad))
columns_to_keep_rad <- na_counts_rad <= NA_threshold
rad <- rad[, columns_to_keep_rad]
radiomics_data <- data.matrix(rad) 
radiomics_scaled_matrix <- scale(radiomics_data)

annotation_df = read.csv("../Dks_7_30_25.csv")
annotation_df = annotation_df[, 2:ncol(annotation_df)]
annotation_df = column_to_rownames(annotation_df, var="id")

# Get the list of common sample IDs (rownames)
common_samples <- intersect(rownames(rma_matrix), rownames(radiomics_scaled_matrix))

# Subset all matrices to keep only common samples
rma_aligned <- rma_matrix[common_samples, ]
rad_aligned <- radiomics_scaled_matrix[common_samples, ]
annotation_aligned <- annotation_df[common_samples, ]
all_gene_vars <- rowVars(t(rma_aligned))
sorted_vars <- sort(all_gene_vars, decreasing = TRUE)

dir.create("MOFA_Preprocessing", showWarnings = FALSE)
pdf("MOFA_Preprocessing/gene_variance_elbow.pdf")
plot(sorted_vars, type = "l", col = "red", lwd = 2,
     xlab = "Genes (Ranked by Variance)", 
     ylab = "Variance",
     main = "Gene Variance Elbow Plot")
abline(v = 2500, col = "blue", lty = 2, lwd = 2)
dev.off()

k_gene_list <- c(2500)

for (K_GENES in k_gene_list) {
  
  message(paste("\n======================================================="))
  message(paste(" STARTING MOFA RUN WITH TOP", K_GENES, "GENES"))
  message(paste("=======================================================\n"))
  
  folder <- paste0("MOFA_outputs_", K_GENES)
  dir.create(folder, showWarnings = FALSE)
  
  # Filter to Top K Genes
  rma_features_in_rows <- t(rma_aligned)
  feature_variances <- rowVars(rma_features_in_rows)
  sorted_features <- sort(feature_variances, decreasing = TRUE)
  top_k_feature_names <- names(sorted_features)[1:K_GENES]
  
  rma_aligned_filtered <- rma_aligned[, top_k_feature_names]
  rma_aligned_filtered <- scale(rma_aligned_filtered)
  
  data_list <- list(
    GeneExpression = t(rma_aligned_filtered), 
    Radiomics = t(rad_aligned)      
  )
  
  # Create the initial MOFA object
  MOFAobject <- create_mofa(data_list)
  
  # Define Options
  data_options <- get_default_data_options(MOFAobject)
  model_options <- get_default_model_options(MOFAobject)
  train_options <- get_default_training_options(MOFAobject)
  
  train_options$seed <- 42
  data_options$scale_views <- FALSE # Skip scaling (already Z-scored)
  model_options$likelihoods <- c(GeneExpression = "gaussian", Radiomics = "gaussian")
  model_options$num_factors <- 20 
  
  # Prepare Object
  MOFAobject.prepared <- prepare_mofa(
    object = MOFAobject,
    data_options = data_options,
    model_options = model_options,
    training_options = train_options  
  )
  
  # Train Object
  outfile_path <- file.path(tempdir(), paste0("GBM_Radiomics_", K_GENES, ".hdf5"))
  MOFAobject.trained <- run_mofa(
    object = MOFAobject.prepared, 
    outfile = outfile_path,
    use_basilisk = FALSE
  )
  
  # Setup Metadata
  metadata_temp <- annotation_aligned
  metadata_temp$sample <- rownames(metadata_temp)
  colnames(metadata_temp) <- c("Swanson D/k", "D/k", "sample")
  samples_metadata(MOFAobject.trained) <- metadata_temp
  
  # =============================================================
  # 3. SAVE OUTPUTS TO DYNAMIC FOLDER
  # =============================================================
  
  pdf(paste0(folder, "/overview.pdf"))
  print(plot_data_overview(MOFAobject.trained))
  dev.off()
  message("Plotting ELBO convergence...")
  pdf(paste0(folder, "/model_convergence.pdf"))
  elbo_values <- MOFAobject.trained@training_stats$elbo
  clean_elbo <- elbo_values[!is.nan(elbo_values)]
  plot(clean_elbo, type = "l", lwd = 2, col = "blue",
       xlab = "Evaluation Step", ylab = "Evidence Lower Bound (ELBO)",
       main = paste("Model Convergence (", K_GENES, "Genes)"))
  dev.off()
  
  total_var_explained <- get_variance_explained(MOFAobject.trained)$r2_total[[1]]
  write.csv(total_var_explained, file=paste0(folder, "/total_variance_explained.csv"))
  
  totalar_by_factor <- get_variance_explained(MOFAobject.trained)$r2_per_factor[[1]]
  write.csv(totalar_by_factor, file=paste0(folder, "/total_variance_explained_by_factors.csv"))
  
  pdf(paste0(folder, "/varbyviewandfactor.pdf"))
  print(plot_variance_explained(MOFAobject.trained, x="view", y="factor"))
  dev.off()
  
  pdf(paste0(folder, "/varbyview.pdf"))
  print(plot_variance_explained(MOFAobject.trained, x="view", y="factor", plot_total = TRUE)[[2]])
  dev.off()
  
  # =============================================================
  # 4. EXTRACT FACTORS AND RUN CORRELATIONS
  # =============================================================
  factor_scores_list <- get_factors(MOFAobject.trained)
  factor_scores_matrix <- factor_scores_list[[1]] 
  num_factors <- ncol(factor_scores_matrix)
  
  d_k_correlations <- data.frame(Factor = character(num_factors), Correlation = numeric(num_factors), Pvalue = numeric(num_factors), stringsAsFactors = FALSE)
  swanson_d_k_correlations <- data.frame(Factor = character(num_factors), Correlation = numeric(num_factors), Pvalue = numeric(num_factors), stringsAsFactors = FALSE)
  
  for (i in 1:num_factors){
    tt <- cor.test(factor_scores_matrix[, i], as.numeric(metadata_temp$`D/k`), method = "pearson")
    tt2 <- cor.test(factor_scores_matrix[, i], as.numeric(metadata_temp$`Swanson D/k`), method = "pearson")
    factor_name <- colnames(factor_scores_matrix)[i]
    
    # Assign directly by column to preserve numeric types
    d_k_correlations$Factor[i] <- factor_name
    d_k_correlations$Correlation[i] <- tt$estimate
    d_k_correlations$Pvalue[i] <- tt$p.value
    
    swanson_d_k_correlations$Factor[i] <- factor_name
    swanson_d_k_correlations$Correlation[i] <- tt2$estimate
    swanson_d_k_correlations$Pvalue[i] <- tt2$p.value
  }
  
  d_k_correlations_final <- d_k_correlations %>%
    dplyr::mutate(FDR = p.adjust(Pvalue, method = "BH")) %>%
    dplyr::arrange(FDR)
  
  swanson_d_k_correlations_final <- swanson_d_k_correlations %>%
    dplyr::mutate(FDR = p.adjust(Pvalue, method = "BH")) %>%
    dplyr::arrange(FDR)
  
  excel_list <- list(
    Dk_Factor_Correlations = d_k_correlations_final,
    Swanson_Dk_Factor_Correlations = swanson_d_k_correlations_final
  )
  
  write_xlsx(x = excel_list, path = paste0(folder, "/MOFA_Analysis_Results.xlsx"))
  
  # Save Weights
  ww <- get_weights(MOFAobject.trained)
  write.csv(ww$GeneExpression, file=paste0(folder, "/geneexp_factor_weights.csv"))
  write.csv(ww$Radiomics, file=paste0(folder, "/radiomics_factor_weights.csv"))
  
  message("Generating Factor Correlation Plot...")
  pdf(paste0(folder, "/factor_correlation.pdf"), width = 8, height = 8)
  print(plot_factor_cor(MOFAobject.trained))
  dev.off()
  
  message(paste("FINISHED:", folder))
}

factor6 <- ww$Radiomics[, "Factor6"]
sorted_factor6 <- factor6[order(abs(factor6), decreasing = TRUE)]
output_df <- data.frame(
  Feature = names(sorted_factor6),
  Factor6_Value = sorted_factor6
)

write.csv(output_df, file = "sorted_Factor6.csv", row.names = FALSE)


