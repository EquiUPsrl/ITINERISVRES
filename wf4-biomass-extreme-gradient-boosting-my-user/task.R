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
if (!requireNamespace("scales", quietly = TRUE)) {
	install.packages("scales", repos="http://cran.us.r-project.org")
}
library(scales)
if (!requireNamespace("tidyr", quietly = TRUE)) {
	install.packages("tidyr", repos="http://cran.us.r-project.org")
}
library(tidyr)
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
cat("R version: ", R.version.string, "\n")

cat("R_ENABLE_ALTREP =", Sys.getenv("R_ENABLE_ALTREP"), "\n")

library(ggplot2)
library(xgboost)
library(Metrics)
library(caret)
library(doParallel)
library(iml)
library(e1071)
library(readr)
library(tidyr)

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

set.seed(123)  # Global seed for reproducibility
library(caret)  # Make sure the caret package is installed
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
                                data = train_data,
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



library(e1071)
library(readr)

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

model_info <- list(
    model_file = model_path,
    preProcess = NULL,
    train_data = train_data,
    test_data = test_data,
    predictors = predictors,
    target_variable = target_variable,
    target_variable_uom = target_variable_uom
)

saveRDS(model_info, file = file.path(model_dir, "model_info.rds"))




library(caret)
library(ggplot2)
library(iml)
library(xgboost)

config_base_path <- "/tmp/data/WF4"


model_info_path <- file.path(model_dir, "model_info.rds")
model_info <- readRDS(model_info_path)

model_file <- model_info$model_file
preProcess <- model_info$preProcess
train_data <- model_info$train_data
test_data <- model_info$test_data
predictors <- model_info$predictors
target_variable <- model_info$target_variable

cat("Model file: ", model_file, "\n")

ext <- tools::file_ext(model_file)

cat("Ext file: ", ext, "\n")

data_for_predictor <- train_data[, predictors, drop = FALSE]
test_data_proc <- test_data[, predictors, drop = FALSE]

if (ext == "xgb") {
    message("Loading XGBoost binary model: ", model_file)

    print(file.info(model_file)$size)

    
    best_model <- xgboost::xgb.load(model_file)

    if (!is.null(preProcess)) {
        message("Applying preProcess to training data")
        data_for_predictor <- predict(preProcess, train_data[, predictors])
        test_data_proc <- predict(preProcess, test_data[, predictors])
        predictors <- colnames(data_for_predictor)
    }
} else if (ext == "rds") {
    message("Loading RDS model: ", model_file)
    best_model <- readRDS(model_file)
} else {
    stop("Unsupported model file type: ", ext)
}





datasets <- c("train", "test")

for (ds in datasets) {

    cat("current dataset: ", ds)

    dataset = data_for_predictor
    if (ds == "test") {
        dataset = test_data_proc
    }

    shap_dir <- file.path(model_dir, paste("SHAP", ds, sep = "_"))
    if (!dir.exists(shap_dir)) {
      dir.create(shap_dir, recursive = TRUE)
    }

    message("Columns in data_for_predictor: ", paste(colnames(data_for_predictor), collapse = ", "))
    message("Columns in dataset row: ", paste(colnames(dataset), collapse = ", "))

    cat(predictors, "\n")

    if (inherits(best_model, "xgb.Booster")) {
      predict_fun <- function(model, newdata) {
        xgb_data <- as.matrix(newdata)
        predict(model, xgb_data)
      }
    } else {
      predict_fun <- function(model, newdata) {
        predict(model, newdata)
      }
    }

    
    predictor_model <- Predictor$new(best_model, data = data_for_predictor, y = train_data[[target_variable]], predict.fun = predict_fun)
    

    shap_values_all <- lapply(1:nrow(dataset), function(i) {
        
        x_interest <- dataset[i, predictors, drop = FALSE]

        str(x_interest)
        
        Shapley$new(predictor_model, x.interest = x_interest)
    })
    
    
    shap_values_matrix <- sapply(shap_values_all, function(shap_obj) {
      if (!is.null(shap_obj$results)) {
        return(as.numeric(shap_obj$results$phi))  # Assuming that 'phi' contains SHAP values
      } else {
        return(rep(NA, length(predictors)))  # Returns NA if there are no SHAP values
      }
    })
    
    shap_values_matrix <- t(shap_values_matrix)
    
    shap_values_df <- as.data.frame(shap_values_matrix)
    
    colnames(shap_values_df) <- predictors  # 'predictors' è la lista dei nomi delle variabili
    
    mean_absolute_shap <- apply(shap_values_df, 2, function(x) mean(abs(x), na.rm = TRUE))
    
    sorted_shap <- sort(mean_absolute_shap, decreasing = TRUE)
    
    
    shap_df_sorted <- data.frame(
      Variable = names(sorted_shap),
      MeanAbsSHAP = sorted_shap
    )
    
    shap_data_file <- file.path(shap_dir, "shap_data_importance.csv")  
    write.csv(shap_df_sorted, shap_data_file, row.names = FALSE)
    
    bar_plot <- ggplot(shap_df_sorted, aes(x = reorder(Variable, -MeanAbsSHAP), y = MeanAbsSHAP)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      labs(
        title = "Feature Impact on Model Output (SHAP Values) ",
        x = "Features",
        y = "Mean SHAP value"
      ) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    
    png_file <- file.path(shap_dir, "shap_importance_bar_plot.png")          
    ggsave(filename = png_file, 
           plot = bar_plot, width = 8, height = 6)
    
    shap_data <- data.frame(
      SHAP = unlist(lapply(shap_values_all, function(shap_obj) shap_obj$results$phi)),
      Variable = rep(names(sorted_shap), each = nrow(shap_values_df)),
      FeatureValue = unlist(lapply(shap_values_all, function(shap_obj) shap_obj$results$feature.value)),
      FeatureNumericValue = unlist(lapply(shap_values_all, function(shap_obj) {
        as.numeric(sub(".*=", "", shap_obj$results$feature.value))  # Extracts only the numeric value of the variable
      }))
    )
    
    shap_data_file <- file.path(shap_dir, "shap_data_summary_plot.csv")  
    write.csv(shap_data, shap_data_file, row.names = FALSE)
    
    shap_data$Variable <- factor(shap_data$Variable, levels = names(sorted_shap))
    
    shap_summary_plot <- ggplot(shap_data, aes(x = SHAP, y = Variable)) +
      geom_point(aes(color = FeatureNumericValue), alpha = 0.7) +  # Use the variable value for the color
      scale_color_gradientn(
        colors = c("red", "yellow", "green"),  # Colors: red for low intensity, yellow for neutral, green for high intensity
        values = scales::rescale(c(min(shap_data$FeatureNumericValue), median(shap_data$FeatureNumericValue), max(shap_data$FeatureNumericValue))),  # Rescale based on the range for each variable
        limits = c(min(shap_data$FeatureNumericValue), max(shap_data$FeatureNumericValue)),  # Bounds between the minimum and maximum SHAP value for each variable
        breaks = c(min(shap_data$FeatureNumericValue), median(shap_data$FeatureNumericValue), max(shap_data$FeatureNumericValue)),  # Color scale breaking points
        labels = c("Low", "Neutral", "High")  # Labels for the extreme values of the coloring
      ) +
      labs(
        title = "SHAP Summary Plot of Feature Importance",
        x = "SHAP value",
        y = "Feature",
        color = "Feature value"
      ) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      theme(legend.position = "right")
    
    png_file <- file.path(shap_dir, "shap_summary_plot.png")
    ggsave(filename = png_file, 
           plot = shap_summary_plot, width = 8, height = 6)
}

output_dir <- config_base_path





library(caret)
library(ggplot2)
library(tidyr)

config_base_path <- "/tmp/data/WF4"


model_info_path <- file.path(model_dir, "model_info.rds")
model_info <- readRDS(model_info_path)

model_file <- model_info$model_file
test_data <- model_info$test_data
target_variable <- model_info$target_variable
uom <- model_info$target_variable_uom

cat("Model file: ", model_file, "\n")

best_model  <- readRDS(model_file)


predictions_test <- predict(best_model, newdata = test_data)

results_test_df <- data.frame(
  Observed = test_data[[target_variable]],
  Predicted = predictions_test
)

results_test_long <- results_test_df %>%
  pivot_longer(cols = c("Observed", "Predicted"),
               names_to = "Type",
               values_to = "Value")

violin_plot <- ggplot(results_test_long, aes(x = Type, y = Value, fill = Type)) +
  geom_violin(trim = FALSE) +  # Show all distribution
  scale_fill_manual(values = c("blue", "red")) +  # Set colors for actual and predicted data
  labs(title = paste("Observed vs Predicted Values ", target_variable),
       x = " Data Type",
       y = paste("Value of ", target_variable, "(", uom, ")")) +
  theme_minimal() +
  theme(legend.position = "none")  # Remove the legend

print(violin_plot)

plot_path_violin_test <- file.path(model_dir, "violin_plot_test_set.png")
ggsave(filename = plot_path_violin_test, plot = violin_plot, width = 8, height = 6)

cat("Violin plot saved in:", plot_path_violin_test, "\n")

output_dir <- config_base_path
# capturing outputs
print('Serialization of output_dir')
file <- file(paste0('/tmp/output_dir_', id, '.json'))
writeLines(toJSON(output_dir, auto_unbox=TRUE), file)
close(file)
