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
if (!requireNamespace("remotes", quietly = TRUE)) {
	install.packages("remotes", repos="http://cran.us.r-project.org")
}
library(remotes)
if (!requireNamespace("tidyr", quietly = TRUE)) {
	install.packages("tidyr", repos="http://cran.us.r-project.org")
}
library(tidyr)



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




if(!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")

remotes::install_version("xgboost", version = "1.7.11.1", repos = "http://cran.us.r-project.org")

cat("R.version: \n")
R.version.string
cat("caret version: \n")
packageVersion("caret")
cat("xgboost version: \n")
packageVersion("xgboost")





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

apply_pca <- as.logical(tolower(as.character(params$value[params$Parameter == "GB_PCA"])))

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
library(caret)  # Assicurati che il pacchetto caret sia installato
train_index <- createDataPartition(dati[[target_variable]], p = training_data_percentage, list = FALSE)
train_data <- dati[train_index, ]
test_data <- dati[-train_index, ]
cat("Size of training set:", nrow(train_data), "\n")
cat("Size of test set:", nrow(test_data), "\n")


seeds <- vector(mode = "list", length = number + 1)
set.seed(123)
for (i in 1:number) seeds[[i]] <- sample.int(1000, size = 5)
seeds[[number + 1]] <- sample.int(1000, 1)


cat("detectCores() = ", detectCores())
ncores <- 1
cat("ncores = ", ncores)
cl <- makePSOCKcluster(ncores)
registerDoParallel(cl)

results_gbm <- list()
best_model_gbm <- NULL
best_metric <- Inf

train_data_clean <- as.data.frame(lapply(train_data, function(x) {
  if (inherits(x, "ALTREP")) as.vector(x) else x
}))

for (n_value in nrounds) {
    for (depth_value in max_depth) {
        for (eta_value in eta) {
            for (gamma_value in gamma) {
                for (colsample_value in colsample_bytree) {
                    for (min_child_value in min_child_weight) {
                        for (subsample_value in subsample) {
                            cat("Running XGB with nrounds =", n_value, "max_depth =", depth_value, "\n")
                            
                            tuneGrid_gbm <- expand.grid(
                                nrounds = n_value,
                                max_depth = depth_value,
                                eta = eta_value,
                                gamma = gamma_value,
                                colsample_bytree = colsample_value,
                                min_child_weight = min_child_value,
                                subsample = subsample_value
                            )
                            
                            ctrl <- trainControl(
                                method = "cv",
                                number = number,
                                seeds = seeds,
                                allowParallel = TRUE,
                                verboseIter = TRUE
                            )
                            
                            model_gbm <- train(
                                as.formula(paste(target_variable, "~ .")),
                                data = train_data_clean,
                                method = "xgbTree",
                                trControl = ctrl,
                                tuneGrid = tuneGrid_gbm,
                                preProcess = preProcSteps,
                                verbose = FALSE
                            )
                            
                            results_gbm[[paste0("nrounds_", n_value, "_depth_", depth_value)]] <- model_gbm
                            metric_value <- min(model_gbm$results[[metric]])
                            
                            if (metric_value < best_metric) {
                                best_metric <- metric_value
                                best_model_gbm <- model_gbm
                            }
                        }
                    }
                }
            }
        }
    }
}

cat("Best model:\n")
print(best_model_gbm)

stopCluster(cl)

output_base_dir <- output_path
model_dir <- file.path(output_base_dir, "Extreme_Gradient_Boosting_Model")

if (!dir.exists(model_dir)) {
  dir.create(model_dir, recursive = TRUE)
}  # <-- Close the if statement here



predictions_gbm_test <- predict(best_model_gbm, newdata = test_data)
results_gbm_test <- data.frame(Actual = test_data[[target_variable]], Predicted_gbm = predictions_gbm_test)

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

train_gbm_preds <- predict(best_model_gbm, newdata = train_data)
results_gbm_training_df <- data.frame(Actual = train_data[[target_variable]], Predicted_gbm = train_gbm_preds)

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
final_nrounds <- best_model_gbm$bestTune$nrounds
final_max_depth <- best_model_gbm$bestTune$max_depth
final_eta <- best_model_gbm$bestTune$eta
final_gamma <- best_model_gbm$bestTune$gamma
final_colsample_bytree <- best_model_gbm$bestTune$colsample_bytree
final_min_child_weight <- best_model_gbm$bestTune$min_child_weight
final_subsample <- best_model_gbm$bestTune$subsample

variabile_target <- target_variable
trasformazioni_applicate <- if (exists("preProcSteps") && length(preProcSteps) > 0) {
    paste(preProcSteps, collapse = ", ")
} else {
    "None"
}

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

model_path_gbm <- file.path(model_dir, "best_model.rds")
saveRDS(best_model_gbm, model_path_gbm)
cat("Extreme Gradient Boosting Model saved in: ", model_path_gbm, "\n")


results_gbm <- best_model_gbm$resample  # This will give you a data frame with metrics for each fold
print(results_gbm)



library(e1071)  # Per il modello SVM
library(readr)  # Per leggere i file

model_path <- file.path(model_dir, "best_model.rds")
best_model_gbm <- readRDS(model_path)

data_path <- prediction_file
prediction_data <- read_delim(data_path, delim = ";")

head(prediction_data)

predictions <- predict(best_model_gbm, prediction_data)

prediction_data$Predicted <- predictions

output_path <- file.path(model_dir, "predictions_with_inputs.txt")
write.table(prediction_data, file = output_path, row.names = FALSE, col.names = TRUE, sep = ";")

cat("Table with input data and forecasts saved in: ", output_path, "\n")

saveRDS(train_data, file = file.path(model_dir, "train_data.rds"))
saveRDS(test_data,  file = file.path(model_dir, "test_data.rds"))
