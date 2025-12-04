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
if (!requireNamespace("fastICA", quietly = TRUE)) {
	install.packages("fastICA", repos="http://cran.us.r-project.org")
}
library(fastICA)
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
if (!requireNamespace("jsonlite", quietly = TRUE)) {
	install.packages("jsonlite", repos="http://cran.us.r-project.org")
}
library(jsonlite)
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
if (!requireNamespace("tidyr", quietly = TRUE)) {
	install.packages("tidyr", repos="http://cran.us.r-project.org")
}
library(tidyr)
if (!requireNamespace("scales", quietly = TRUE)) {
	install.packages("scales", repos="http://cran.us.r-project.org")
}
library(scales)
if (!requireNamespace("tools", quietly = TRUE)) {
	install.packages("tools", repos="http://cran.us.r-project.org")
}
library(tools)



print('option_list')
option_list = list(

make_option(c("--parameter_file"), action="store", default=NA, type="character", help="my description"),
make_option(c("--prediction_file"), action="store", default=NA, type="character", help="my description"),
make_option(c("--training_file"), action="store", default=NA, type="character", help="my description"),
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

print("Retrieving parameter_file")
var = opt$parameter_file
print(var)
var_len = length(var)
print(paste("Variable parameter_file has length", var_len))

parameter_file <- gsub("\"", "", opt$parameter_file)
print("Retrieving prediction_file")
var = opt$prediction_file
print(var)
var_len = length(var)
print(paste("Variable prediction_file has length", var_len))

prediction_file <- gsub("\"", "", opt$prediction_file)
print("Retrieving training_file")
var = opt$training_file
print(var)
var_len = length(var)
print(paste("Variable training_file has length", var_len))

training_file <- gsub("\"", "", opt$training_file)
id <- gsub('"', '', opt$id)


print("Running the cell")
library(ggplot2)
library(xgboost)
library(Metrics)
library(caret)
library(doParallel)
library(iml)
library(e1071)
library(readr)
library(tidyr)
library(dplyr)


config_base_path <- "/tmp/data/WF4"
output_path = file.path(config_base_path, "output")

input_file <- training_file

dati <- read.table(input_file, header = TRUE, sep = ';')

param_file_gbm <- parameter_file

params <- read.table(param_file_gbm, header = TRUE, sep = ';', fill = TRUE, stringsAsFactors = FALSE)

number <- as.numeric(params$value[params$Parameter == "number"]) #identificare il valore per la cross validation

nrounds_row <- params[params$Parameter == "nrounds", ]
max_depth_row  <- params[params$Parameter == "max_depth",  ]
eta_row  <- params[params$Parameter == "eta",  ]
gamma_row  <- params[params$Parameter == "gamma",  ]
colsample_bytree_row  <- params[params$Parameter == "colsample_bytree",  ]
min_child_weight_row  <- params[params$Parameter == "min_child_weight",  ]
subsample_row  <- params[params$Parameter == "subsample",  ]

nrounds           <- as.numeric(nrounds_row[ , -1])
nrounds           <- nrounds[!is.na(nrounds)]
max_depth         <- as.numeric(max_depth_row[ , -1])
max_depth         <- max_depth[!is.na(max_depth)]
eta               <- as.numeric(eta_row[ , -1])
eta               <- eta[!is.na(eta)]
gamma             <- as.numeric(gamma_row[ , -1])
gamma             <- gamma[!is.na(gamma)]
colsample_bytree  <- as.numeric(colsample_bytree_row[ , -1])
colsample_bytree  <- colsample_bytree[!is.na(colsample_bytree)]
min_child_weight  <- as.numeric(min_child_weight_row[ , -1])
min_child_weight  <- min_child_weight[!is.na(min_child_weight)]
subsample         <- as.numeric(subsample_row[ , -1])
subsample         <- subsample[!is.na(subsample)]


target_variable <- as.character(params$value[params$Parameter == "Target variable"])
target_variable_uom <- as.character(params$value[params$Parameter == "Target variable UoM"])
training_data_percentage <- as.numeric(params$value[params$Parameter == "Training data percentage"])

predictors <- setdiff(names(dati), target_variable)

preProcSteps <- c()
if (tolower(as.character(params$value[params$Parameter == "XGB_Normalization"])) == "true") preProcSteps <- c(preProcSteps, "range")
if (tolower(as.character(params$value[params$Parameter == "XGB_Standardization"])) == "true") preProcSteps <- c(preProcSteps, "center", "scale")
if (tolower(as.character(params$value[params$Parameter == "XGB_ICA"])) == "true") preProcSteps <- c(preProcSteps, "ica")
if (tolower(as.character(params$value[params$Parameter == "XGB_PCA"])) == "true") preProcSteps <- c(preProcSteps, "pca")
cat("Selected Preprocessing: ", paste(preProcSteps, collapse = ", "), "\n")

metric <- 'MAE'
if (tolower(as.character(params$value[params$Parameter == "metric_value_MAE"])) == "true") metric <- 'MAE'
if (tolower(as.character(params$value[params$Parameter == "metric_value_RMSE"])) == "true") metric <- 'RMSE'
if (tolower(as.character(params$value[params$Parameter == "metric_value_Rsquared"])) == "true") metric <- 'Rsquared'

set.seed(123)  # Seme globale per riproducibilità
train_index <- createDataPartition(dati[[target_variable]], p = training_data_percentage, list = FALSE)
train_data <- dati[train_index, ]
test_data <- dati[-train_index, ]
cat("Size of training set:", nrow(train_data), "\n")
cat("Size of test set:", nrow(test_data), "\n")

cat("=== DEBUG LOG: Train vs Test columns ===\n")
cat("Train columns (raw): ", paste(names(train_data), collapse = ", "), "\n")
cat("Test columns (raw):  ", paste(names(test_data), collapse = ", "), "\n")


seeds <- vector(mode = "list", length = number + 1)
set.seed(123)
for (i in 1:number) seeds[[i]] <- sample.int(1000, size = 5)
seeds[[number + 1]] <- sample.int(1000, 1)


cat("detectCores() = ", detectCores())
ncores <- 1
cat("ncores = ", ncores)
cl <- makePSOCKcluster(ncores)
registerDoParallel(cl)



test_features <- test_data[, !(names(test_data) %in% target_variable)]

test_matrix <- data.matrix(test_features)



features_train <- train_data[, setdiff(names(train_data), target_variable), drop = FALSE]
y_train        <- train_data[[target_variable]]

features_test  <- test_data[, setdiff(names(test_data), target_variable), drop = FALSE]
y_test         <- test_data[[target_variable]]

cat("Features (before preprocess):", paste(colnames(features_train), collapse = ", "), "\n")
cat("Train rows/cols:", dim(features_train), " Test rows/cols:", dim(features_test), "\n")

if (!is.null(preProcSteps) && length(preProcSteps) > 0) {
  preProcObj <- preProcess(features_train, method = preProcSteps)
  features_train_proc <- predict(preProcObj, features_train)
  features_test_proc  <- predict(preProcObj, features_test)
  cat("Train columns after preprocessing: ", paste(names(features_train_proc), collapse = ", "), "\n")
  cat("Test columns after preprocessing:  ", paste(names(features_test_proc), collapse = ", "), "\n")
} else {
  features_train_proc <- features_train
  features_test_proc  <- features_test
}

ok_train <- complete.cases(features_train_proc)
ok_test  <- complete.cases(features_test_proc)

if (!all(ok_train)) {
  cat("Warning: rimosse", sum(!ok_train), "righe dal train per NA/Inf dopo preprocess\n")
  features_train_proc <- features_train_proc[ok_train, , drop = FALSE]
  y_train <- y_train[ok_train]
}
if (!all(ok_test)) {
  cat("Warning: rimosse", sum(!ok_test), "righe dal test per NA/Inf dopo preprocess\n")
  features_test_proc <- features_test_proc[ok_test, , drop = FALSE]
  y_test <- y_test[ok_test]
}

cat("Features after preprocess (train):", paste(colnames(features_train_proc), collapse = ", "), "\n")
cat("Features after preprocess (test) :", paste(colnames(features_test_proc), collapse = ", "), "\n")
cat("Train dims proc:", dim(features_train_proc), " Test dims proc:", dim(features_test_proc), "\n")

if (!identical(colnames(features_train_proc), colnames(features_test_proc))) {
  stop("ERROR: colonne di train e test diverse dopo preprocess. Controlla preProcSteps e le colonne numeriche.")
}

metric_map <- function(m) {
  m_up <- toupper(as.character(m))
  if (m_up == "MAE") return("mae")
  if (m_up == "RMSE") return("rmse")
  if (m_up == "RSQUARED" || m_up == "RSQUARED" || m_up == "RSQUARED") return("rmse") # use rmse for CV, compute R2 later
  return("rmse")
}
eval_metric_xgb <- metric_map(metric)
cat("Using eval_metric for xgboost CV:", eval_metric_xgb, "\n")

dtrain <- xgb.DMatrix(data = as.matrix(features_train_proc), label = as.numeric(y_train))
dtest  <- xgb.DMatrix(data = as.matrix(features_test_proc),  label = as.numeric(y_test))

results_gbm <- list()
best_model_gbm <- NULL
best_metric_val <- Inf

for (n_value in nrounds) {
  for (depth_value in max_depth) {
    for (eta_value in eta) {
      for (gamma_value in gamma) {
        for (colsample_value in colsample_bytree) {
          for (min_child_value in min_child_weight) {
            for (subsample_value in subsample) {

              cat("------------------------------------------------\n")
              cat(sprintf("Params: nrounds=%s max_depth=%s eta=%s gamma=%s colsample=%s min_child_weight=%s subsample=%s\n",
                          n_value, depth_value, eta_value, gamma_value, colsample_value, min_child_value, subsample_value))

              params_xgb <- list(
                booster = "gbtree",
                objective = "reg:squarederror",
                eta = eta_value,
                max_depth = as.integer(depth_value),
                gamma = gamma_value,
                colsample_bytree = colsample_value,
                min_child_weight = min_child_value,
                subsample = subsample_value,
                eval_metric = eval_metric_xgb
              )

              nfold_use <- min(number, nrow(features_train_proc))
              if (nfold_use < 2) {
                warning("nfold < 2, skipping CV and using full nrounds for training")
                best_iter <- n_value
                rmse_cv_val <- NA
              } else {
                early_stop <- 10
                if (n_value <= early_stop) early_stop <- max(1, floor(n_value/2))

                cv <- tryCatch({
                  xgb.cv(
                    params = params_xgb,
                    data = dtrain,
                    nrounds = n_value,
                    nfold = nfold_use,
                    verbose = FALSE,
                    early_stopping_rounds = early_stop,
                    showsd = TRUE
                  )
                }, error = function(e) {
                  warning("xgb.cv failed: ", e$message)
                  return(NULL)
                })

                if (is.null(cv)) {
                  best_iter <- n_value
                  rmse_cv_val <- NA
                } else {
                  best_iter <- cv$best_iteration
                  if (is.null(best_iter) || length(best_iter) == 0) best_iter <- n_value
                  em_col <- grep("^test_.*_mean$", colnames(cv$evaluation_log), value = TRUE)
                  if (length(em_col) >= 1) {
                    rmse_cv_val <- cv$evaluation_log[[em_col[1]]][best_iter]
                  } else {
                    rmse_cv_val <- NA
                  }
                }
              }

              cat("Using best_iter =", best_iter, " CV-metric:", rmse_cv_val, "\n")

              model_final <- xgb.train(
                params = params_xgb,
                data = dtrain,
                nrounds = best_iter,
                verbose = 0
              )

              key <- paste0("nrounds_", n_value, "_depth_", depth_value, "_eta_", eta_value,
                            "_gamma_", gamma_value, "_col_", colsample_value)
              results_gbm[[key]] <- list(params = params_xgb, best_iter = best_iter, cv_metric = rmse_cv_val, model = model_final)

              metric_comp_val <- rmse_cv_val
              if (is.na(metric_comp_val)) metric_comp_val <- Inf

              if (metric_comp_val < best_metric_val) {
                best_metric_val <- metric_comp_val
                best_model_gbm <- model_final
                best_info <- list(key = key, params = params_xgb, best_iter = best_iter, cv_metric = rmse_cv_val)
              }

            } # end subsample
          } # end min_child
        } # end colsample
      } # end gamma
    } # end eta
  } # end depth
} # end nrounds

cat("------------------------------------------------\n")
cat("Best (CV) metric value:", best_metric_val, "\n")
cat("Best model info:\n")
print(best_info)

preds_test <- predict(best_model_gbm, dtest)

mae_test  <- tryCatch({ Metrics::mae(y_test, preds_test) }, error = function(e) NA)
rmse_test <- tryCatch({ Metrics::rmse(y_test, preds_test) }, error = function(e) NA)
r2_test   <- tryCatch({
  1 - sum((y_test - preds_test)^2) / sum((y_test - mean(y_test))^2)
}, error = function(e) NA)

cat(sprintf("Test MAE: %s  RMSE: %s  R2: %s\n", signif(mae_test,6), signif(rmse_test,6), signif(r2_test,6)))






stopCluster(cl)

output_base_dir <- output_path
model_dir <- file.path(output_base_dir, "Extreme_Gradient_Boosting_Model_2")

if (!dir.exists(model_dir)) {
  dir.create(model_dir, recursive = TRUE)
}




cat("===== DEBUGGING FEATURES FOR XGB PREDICTION =====\n")
cat("Class of best_model_gbm:", class(best_model_gbm), "\n")

num_features_model <- best_model_gbm$nfeatures
cat("Number of features expected by XGB model:", num_features_model, "\n")

cat("Columns in features_test_proc (test set preprocessato):\n")
print(colnames(features_test_proc))

cat("Dimensions of features_test_proc:", dim(features_test_proc), "\n")

cat("Summary of features_test_proc:\n")
print(summary(features_test_proc))

na_count <- sum(is.na(features_test_proc))
cat("Number of NA values in features_test_proc:", na_count, "\n")
if (na_count > 0) {
  cat("Warning: NA values detected. XGBoost cannot handle NA by default.\n")
}

cat("First 5 rows of features_test_proc:\n")
print(head(features_test_proc, 5))

cat("===============================================\n")

                      

results_gbm_test <- data.frame(
  Actual = test_data[[target_variable]],
  Predicted_gbm = preds_test
)

plot_gbm_test <- ggplot(data = results_gbm_test, aes(x = Actual, y = Predicted_gbm)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(title = "Extreme Gradient Boosting - Test Set",
       x = paste("Observed", target_variable, "(", target_variable_uom, ")"),
       y = paste("Predicted", target_variable, "(", target_variable_uom, ")")) +
  theme_minimal()

print(plot_gbm_test)
plot_path_gbm <- file.path(model_dir, "gbm_plot_test_set.png")
ggsave(filename = plot_path_gbm, plot = plot_gbm_test, width = 8, height = 6)
cat("Chart saved in: ", plot_path_gbm, "\n")



train_gbm_preds <- predict(best_model_gbm, newdata = dtrain)

results_gbm_training_df <- data.frame(
  Actual = train_data[[target_variable]],
  Predicted_gbm = train_gbm_preds
)

plot_gbm_training <- ggplot(data = results_gbm_training_df, aes(x = Actual, y = Predicted_gbm)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(title = "Extreme Gradient Boosting - Training Set",
       x = paste("Observed", target_variable, "(", target_variable_uom, ")"),
       y = paste("Predicted", target_variable, "(", target_variable_uom, ")")) +
  theme_minimal()

print(plot_gbm_training)
plot_path_gbm_training <- file.path(model_dir, "gbm_plot_training_set.png")
ggsave(filename = plot_path_gbm_training, plot = plot_gbm_training, width = 8, height = 6)
cat("Chart saved in: ", plot_path_gbm_training, "\n")



params_output_file <- file.path(model_dir, "model_parameters_description.txt")

final_nrounds <- best_info$best_iter
final_max_depth <- best_info$params$max_depth
final_eta <- best_info$params$eta
final_gamma <- best_info$params$gamma
final_colsample_bytree <- best_info$params$colsample_bytree
final_min_child_weight <- best_info$params$min_child_weight
final_subsample <- best_info$params$subsample

variabile_target <- target_variable
trasformazioni_applicate <- if (exists("preProcSteps") && length(preProcSteps) > 0) {
  paste(preProcSteps, collapse = ", ")
} else "None"

r_squared_training <- cor(results_gbm_training_df$Actual, results_gbm_training_df$Predicted_gbm)^2
mae_training <- mean(abs(results_gbm_training_df$Actual - results_gbm_training_df$Predicted_gbm))
rmse_training <- sqrt(mean((results_gbm_training_df$Actual - results_gbm_training_df$Predicted_gbm)^2))

r_squared_test <- cor(results_gbm_test$Actual, results_gbm_test$Predicted_gbm)^2
mae_test <- mean(abs(results_gbm_test$Actual - results_gbm_test$Predicted_gbm))
rmse_test <- sqrt(mean((results_gbm_test$Actual - results_gbm_test$Predicted_gbm)^2))

parametri_testo <- paste(
  "Description of the Optimised Extreme Gradient Boosting Model:\n",
  "Target Variable:", variabile_target, "\n",
  "Type of feature transformation applied:", trasformazioni_applicate, "\n",
  "nrounds:", final_nrounds, "\n",
  "Max Depth:", final_max_depth, "\n",
  "Learning Rate (Eta):", final_eta, "\n",
  "Gamma:", final_gamma, "\n",
  "Colsample Bytree:", final_colsample_bytree, "\n",
  "Min Child Weight:", final_min_child_weight, "\n",
  "Subsample:", final_subsample, "\n",
  "\nPerformance metrics on the training set:\n",
  "R²:", r_squared_training, "\n",
  "Mean Absolute Error (MAE):", mae_training, "\n",
  "Root Mean Squared Error (RMSE):", rmse_training, "\n",
  "\nPerformance metrics on the test set:\n",
  "R²:", r_squared_test, "\n",
  "Mean Absolute Error (MAE):", mae_test, "\n",
  "Root Mean Squared Error (RMSE):", rmse_test, "\n"
)

writeLines(parametri_testo, con = params_output_file)

            

model_path_gbm = file.path(model_dir, "best_model.xgb")
xgb.save(best_model_gbm, model_path_gbm)
cat("Extreme Gradient Boosting Model saved in: ", model_path_gbm, "\n")

results_gbm <- best_model_gbm$resample
print(results_gbm)




prediction_data <- read_delim(prediction_file, delim = ";")

head(prediction_data)


cat("Original prediction_data columns:\n")
print(colnames(prediction_data))

if (!is.null(preProcObj)) {
  cat("Applying preProcess object from training to prediction data...\n")
  features_prediction_proc <- predict(preProcObj, prediction_data)
  preprocessed_train <- predict(preProcObj, train_data[, predictors])
} else {
  features_prediction_proc <- prediction_data
  preprocessed_train <- colnames(train_data[, predictors])
}

cat("Columns after preprocessing for prediction:\n")
print(colnames(features_prediction_proc))


feature_names <- colnames(preprocessed_train)

cat("Columns after preprocessing for train:\n")
print(feature_names)

prediction_matrix <- as.matrix(features_prediction_proc[, feature_names, drop = FALSE])
predictions <- predict(best_model_gbm, prediction_matrix)

                      

prediction_data$Predicted <- predictions

output_path <- file.path(model_dir, "predictions_with_inputs.txt")
write.table(prediction_data, file = output_path,
            row.names = FALSE, col.names = TRUE, sep = ";")

cat("Table with input data and forecasts saved in: ", output_path, "\n")






model_info <- list(
    model_file = model_path_gbm,
    preProcess = preProcObj,
    train_data = train_data,
    test_data = test_data,
    predictors = predictors,
    target_variable = target_variable,
    target_variable_uom = target_variable_uom
)

saveRDS(model_info, file = file.path(model_dir, "model_info.rds"))
# capturing outputs
print('Serialization of model_dir')
file <- file(paste0('/tmp/model_dir_', id, '.json'))
writeLines(toJSON(model_dir, auto_unbox=TRUE), file)
close(file)
print('Serialization of target_variable')
file <- file(paste0('/tmp/target_variable_', id, '.json'))
writeLines(toJSON(target_variable, auto_unbox=TRUE), file)
close(file)
print('Serialization of target_variable_uom')
file <- file(paste0('/tmp/target_variable_uom_', id, '.json'))
writeLines(toJSON(target_variable_uom, auto_unbox=TRUE), file)
close(file)
