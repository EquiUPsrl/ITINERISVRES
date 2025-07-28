setwd('/app')
library(optparse)
library(jsonlite)

if (!requireNamespace("caret", quietly = TRUE)) {
	install.packages("caret", repos="http://cran.us.r-project.org")
}
library(caret)
if (!requireNamespace("doFuture", quietly = TRUE)) {
	install.packages("doFuture", repos="http://cran.us.r-project.org")
}
library(doFuture)
if (!requireNamespace("doParallel", quietly = TRUE)) {
	install.packages("doParallel", repos="http://cran.us.r-project.org")
}
library(doParallel)
if (!requireNamespace("dplyr", quietly = TRUE)) {
	install.packages("dplyr", repos="http://cran.us.r-project.org")
}
library(dplyr)
if (!requireNamespace("e1071", quietly = TRUE)) {
	install.packages("e1071", repos="http://cran.us.r-project.org")
}
library(e1071)
if (!requireNamespace("foreach", quietly = TRUE)) {
	install.packages("foreach", repos="http://cran.us.r-project.org")
}
library(foreach)
if (!requireNamespace("future", quietly = TRUE)) {
	install.packages("future", repos="http://cran.us.r-project.org")
}
library(future)
if (!requireNamespace("ggplot2", quietly = TRUE)) {
	install.packages("ggplot2", repos="http://cran.us.r-project.org")
}
library(ggplot2)
if (!requireNamespace("iml", quietly = TRUE)) {
	install.packages("iml", repos="http://cran.us.r-project.org")
}
library(iml)
if (!requireNamespace("kernlab", quietly = TRUE)) {
	install.packages("kernlab", repos="http://cran.us.r-project.org")
}
library(kernlab)
if (!requireNamespace("MASS", quietly = TRUE)) {
	install.packages("MASS", repos="http://cran.us.r-project.org")
}
library(MASS)
if (!requireNamespace("Metrics", quietly = TRUE)) {
	install.packages("Metrics", repos="http://cran.us.r-project.org")
}
library(Metrics)
if (!requireNamespace("nnet", quietly = TRUE)) {
	install.packages("nnet", repos="http://cran.us.r-project.org")
}
library(nnet)
if (!requireNamespace("randomForest", quietly = TRUE)) {
	install.packages("randomForest", repos="http://cran.us.r-project.org")
}
library(randomForest)
if (!requireNamespace("readr", quietly = TRUE)) {
	install.packages("readr", repos="http://cran.us.r-project.org")
}
library(readr)
if (!requireNamespace("xgboost", quietly = TRUE)) {
	install.packages("xgboost", repos="http://cran.us.r-project.org")
}
library(xgboost)
if (!requireNamespace("scales", quietly = TRUE)) {
	install.packages("scales", repos="http://cran.us.r-project.org")
}
library(scales)
if (!requireNamespace("tidyr", quietly = TRUE)) {
	install.packages("tidyr", repos="http://cran.us.r-project.org")
}
library(tidyr)
if (!requireNamespace("reshape2", quietly = TRUE)) {
	install.packages("reshape2", repos="http://cran.us.r-project.org")
}
library(reshape2)



print('option_list')
option_list = list(

make_option(c("--new_dir_gbm"), action="store", default=NA, type="character", help="my description"),
make_option(c("--new_dir_lm"), action="store", default=NA, type="character", help="my description"),
make_option(c("--new_dir_nn"), action="store", default=NA, type="character", help="my description"),
make_option(c("--new_dir_rf"), action="store", default=NA, type="character", help="my description"),
make_option(c("--new_dir_svm"), action="store", default=NA, type="character", help="my description"),
make_option(c("--id"), action="store", default=NA, type="character", help="task id")
)


opt = parse_args(OptionParser(option_list=option_list))

var_serialization <- function(var){
    if (is.null(var)){
        print("Variable is null")
        exit(1)
    }
    tryCatch(
        {
            var <- fromJSON(var)
            print("Variable deserialized")
            return(var)
        },
        error=function(e) {
            print("Error while deserializing the variable")
            print(var)
            var <- gsub("'", '"', var)
            var <- fromJSON(var)
            print("Variable deserialized")
            return(var)
        },
        warning=function(w) {
            print("Warning while deserializing the variable")
            var <- gsub("'", '"', var)
            var <- fromJSON(var)
            print("Variable deserialized")
            return(var)
        }
    )
}

print("Retrieving new_dir_gbm")
var = opt$new_dir_gbm
print(var)
var_len = length(var)
print(paste("Variable new_dir_gbm has length", var_len))

new_dir_gbm <- gsub("\"", "", opt$new_dir_gbm)
print("Retrieving new_dir_lm")
var = opt$new_dir_lm
print(var)
var_len = length(var)
print(paste("Variable new_dir_lm has length", var_len))

new_dir_lm <- gsub("\"", "", opt$new_dir_lm)
print("Retrieving new_dir_nn")
var = opt$new_dir_nn
print(var)
var_len = length(var)
print(paste("Variable new_dir_nn has length", var_len))

new_dir_nn <- gsub("\"", "", opt$new_dir_nn)
print("Retrieving new_dir_rf")
var = opt$new_dir_rf
print(var)
var_len = length(var)
print(paste("Variable new_dir_rf has length", var_len))

new_dir_rf <- gsub("\"", "", opt$new_dir_rf)
print("Retrieving new_dir_svm")
var = opt$new_dir_svm
print(var)
var_len = length(var)
print(paste("Variable new_dir_svm has length", var_len))

new_dir_svm <- gsub("\"", "", opt$new_dir_svm)
id <- gsub('"', '', opt$id)

{'name': 'conf_base_path', 'assignation': "conf_base_path<-'/tmp/data/'"}
{'name': 'conf_output_path', 'assignation': "conf_output_path<-'/tmp/data/output/'"}

print("Running the cell")
results_svm_path <- file.path(new_dir_svm, "cross_validation_metrics.txt")
results_svm <- read.table(results_svm_path, header = TRUE, sep = "\t")

results_rf_path <- file.path(new_dir_rf, "cross_validation_metrics.txt")
results_rf <- read.table(results_rf_path, header = TRUE, sep = "\t")

results_lm_path <- file.path(new_dir_lm, "cross_validation_metrics.txt")
results_lm <- read.table(results_lm_path, header = TRUE, sep = "\t")

results_gbm_path <- file.path(new_dir_gbm, "cross_validation_metrics.txt")
results_gbm <- read.table(results_gbm_path, header = TRUE, sep = "\t")

results_nn_path <- file.path(new_dir_nn, "cross_validation_metrics.txt")
results_nn <- read.table(results_nn_path, header = TRUE, sep = "\t")


mae_rf <- results_rf$MAE
mae_lm <- results_lm$MAE
mae_gbm <- results_gbm$MAE
mae_svm <- results_svm$MAE
mae_nn <- results_nn$MAE

mae_df <- data.frame(
  rf = mae_rf,
  lm = mae_lm,
  gbm = mae_gbm,
  svm = mae_svm,
  nn = mae_nn
)

print(mae_df)

t_test_results <- list()

colonne <- names(mae_df)

for (i in 1:(length(colonne) - 1)) {
  for (j in (i + 1):length(colonne)) {
    coppia <- paste(colonne[i], "vs", colonne[j])
    
    test <- t.test(mae_df[[colonne[i]]], mae_df[[colonne[j]]])
    
    t_test_results[[coppia]] <- list(p_value = test$p.value, statistic = test$statistic)
  }
}

t_test_results

num_modelli <- length(colonne)
p_matrix <- matrix(NA, nrow = num_modelli, ncol = num_modelli)

rownames(p_matrix) <- colonne
colnames(p_matrix) <- colonne

for (i in 1:(num_modelli - 1)) {
  for (j in (i + 1):num_modelli) {
    coppia <- paste(colonne[i], "vs", colonne[j])
    
    p_matrix[i, j] <- t_test_results[[coppia]]$p_value
    p_matrix[j, i] <- p_matrix[i, j]  # La matrice è simmetrica
  }
}

print(p_matrix)

output_dir <- paste(conf_output_path, "Pairwise_Comparison_Results", sep="")

output_file_path <- file.path(output_dir, "pairwise_comparison_results_t_test.txt")
write.table(p_matrix, file = output_file_path, sep = "\t", row.names = TRUE, col.names = NA)

cat("Pairwise comparison results t-test saved to:", output_file_path, "\n")

shapiro_results <- list()

for (model in names(mae_df)) {
  test_result <- shapiro.test(mae_df[[model]])
  shapiro_results[[model]] <- list(
    W = test_result$statistic,
    p_value = test_result$p.value
  )
}

shapiro_results_df <- do.call(rbind, lapply(shapiro_results, as.data.frame))
colnames(shapiro_results_df) <- c("W", "p-value")

output_dir <- paste(conf_output_path, "Pairwise_Comparison_Results", sep="")

output_file_path <- file.path(output_dir, "shapiro_test_results.txt")
write.table(shapiro_results_df, file = output_file_path, sep = "\t", row.names = TRUE, col.names = NA)

cat("Shapiro-Wilk test results saved to:", output_file_path, "\n")
