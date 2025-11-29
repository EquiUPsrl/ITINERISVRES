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
library(e1071)    # For Support Vector Machines (SVM)
library(caret)    # For machine learning and cross-validation
library(ggplot2)  # For visualization
library(MASS)     # Contains various datasets and useful functions
library(dplyr)    # For data manipulation
library(doParallel)  # For parallelization

library(Metrics)
library(iml)
library(readr)
library(tidyr)

config_base_path <- "/tmp/data/WF4"
output_path = file.path(config_base_path, "output")

input_file <- training_file
param_file <- parameter_file

if (!file.exists(input_file)) stop("Input file does not exist.")
if (!file.exists(param_file)) stop("Parameter file does not exist.")

dati <- read.table(input_file, header = TRUE, sep = ';', fill = TRUE)

params <- read.table(param_file, header = TRUE, sep = ';', fill = TRUE)

number_row <- params[params$Parameter == "number", ]
gamma_row  <- params[params$Parameter == "gamma_svm",  ]
cost_row  <- params[params$Parameter == "cost",  ]

number <- as.numeric(number_row[ , -1])
number <- number[!is.na(number) & number > 0]
gamma  <- as.numeric(gamma_row[ , -1])
gamma <- gamma[!is.na(gamma) & gamma > 0]
cost  <- as.numeric(cost_row[ , -1])
cost <- cost[!is.na(cost) & cost > 0]


target_variable <- as.character(params$value[params$Parameter == "Target variable"])
target_variable_uom <- as.character(params$value[params$Parameter == "Target variable UoM"])
apply_pca <- as.logical(tolower(as.character(params$value[params$Parameter == "SVM_PCA"])))
training_data_percentage <- as.numeric(params$value[params$Parameter == "Training data percentage"])

cat("target_variable:", target_variable, "\n")

print(paste("number:", number))
print(paste("gamma:", gamma))
print(paste("cost:", cost))

str(dati)

predictors <- setdiff(names(dati), target_variable)

preProcSteps <- c()
if (tolower(as.character(params$value[params$Parameter == "SVM_Normalization"])) == "true") preProcSteps <- c(preProcSteps, "range")
if (tolower(as.character(params$value[params$Parameter == "SVM_Standardization"])) == "true") preProcSteps <- c(preProcSteps, "center", "scale")
if (tolower(as.character(params$value[params$Parameter == "SVM_ICA"])) == "true") preProcSteps <- c(preProcSteps, "ica")
if (tolower(as.character(params$value[params$Parameter == "SVM_PCA"])) == "true") preProcSteps <- c(preProcSteps, "pca")
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
set.seed(123)  # Set seed for reproducibility
for (i in 1:number) {
    seeds[[i]] <- sample.int(1000, size = 5)  # Generate a vector of 5 random seeds for each fold
}
seeds[[number + 1]] <- sample.int(1000, 1)  # Single seed for the final evaluation

ncores <- 1 
cl <- makePSOCKcluster(ncores)  # Use all cores except one
registerDoParallel(cl)

ctrl <- trainControl(
    method = "cv",                # Use cross-validation
    number = number,              # Number of folds
    seeds = seeds,                # Seeds for each fold
    allowParallel = TRUE,         # Allow parallel processing
    verboseIter = TRUE            # Display iterations
)

formula_svm <- as.formula(paste(target_variable, "~ .")) 

results <- list()  # List to save models
best_model_svm <- NULL  # To save the best model
best_metric <- Inf  # Set an initial very high metric (for minimization)
final_gamma <- NULL

cost_values <- cost  # Ensure 'cost' is predefined
gamma_values <- gamma  # Ensure 'gamma' is predefined

for (c_value in cost_values) {
    for (g_value in gamma_values) {
        cat("Running SVM model with cost =", c_value, ", gamma =", g_value, "\n")
        
        model_svm <- train(formula_svm, 
                           data = train_data, 
                           method = "svmRadial",   # Method: SVM with radial kernel
                           trControl = ctrl,
                           tuneGrid = expand.grid(sigma = g_value, C = c_value),
                           preProcess = preProcSteps,
        
        results[[paste0("cost_", c_value, "_gamma_", g_value)]] <- model_svm
        
        metric_value <- min(model_svm$results[[metric]])  # Example: if optimizing MAE
        
        if (metric_value < best_metric) {
            best_metric <- metric_value
            best_model_svm <- model_svm
            final_gamma <- g_value
        }
    }
}

cat("Best model:\n")
print(best_model_svm)

stopCluster(cl)


output_base_dir <- output_path
model_dir <- file.path(output_base_dir, "SVM_Model")
if (!dir.exists(model_dir)) {
  dir.create(model_dir, recursive = TRUE)
}

train_svm_preds <- predict(best_model_svm, train_data)
results_svm_training_df <- data.frame(Actual = train_data[[target_variable]], Predicted = train_svm_preds)

plot_svm_training <- ggplot(data = results_svm_training_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(title = paste("SVM Radial - Training Set per", target_variable),
       x = paste("Observed", target_variable, "(", target_variable_uom, ")"),
       y = paste("Predicted", target_variable, "(", target_variable_uom, ")")) +
  theme_minimal()

print(plot_svm_training)

plot_path_svm_training <- file.path(model_dir, "svm_plot_training_set.png")
ggsave(filename = plot_path_svm_training, plot = plot_svm_training, width = 8, height = 6)
cat("Chart saved in:", plot_path_svm_training, "\n")

predictions_svm_test <- predict(best_model_svm, newdata = test_data)

results_svm_test_df <- data.frame(Actual = test_data[[target_variable]], Predicted = predictions_svm_test)

plot_svm_test <- ggplot(data = results_svm_test_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(title = paste("SVM Radial - Test Set per", target_variable),
       x = paste("Observed", target_variable, "(", target_variable_uom, ")"),
       y = paste("Predicted", target_variable, "(", target_variable_uom, ")")) +
  theme_minimal()

print(plot_svm_test)

plot_path_svm_test <- file.path(model_dir, "svm_plot_test_set.png")
ggsave(filename = plot_path_svm_test, plot = plot_svm_test, width = 8, height = 6)
cat("Chart saved in:", plot_path_svm_test, "\n")

params_output_file_svm <- file.path(model_dir, "model_parameters_description.txt")

final_cost <- best_model_svm$bestTune$C

variabile_target <- target_variable
trasformazioni_applicate <- if (exists("preProcSteps") && length(preProcSteps) > 0) {
  paste(preProcSteps, collapse = ", ")
} else {
  "None"
}

r_squared_training_svm <- cor(results_svm_training_df$Actual, results_svm_training_df$Predicted)^2
mae_training_svm <- mean(abs(results_svm_training_df$Actual - results_svm_training_df$Predicted))
rmse_training_svm <- sqrt(mean((results_svm_training_df$Actual - results_svm_training_df$Predicted)^2))

r_squared_test_svm <- cor(results_svm_test_df$Actual, results_svm_test_df$Predicted)^2
mae_test_svm <- mean(abs(results_svm_test_df$Actual - results_svm_test_df$Predicted))
rmse_test_svm <- sqrt(mean((results_svm_test_df$Actual - results_svm_test_df$Predicted)^2))

parametri_testo_svm <- paste(
  "Description of the Optimised SVM Model:\n",
  "Target Variable:", variabile_target, "\n",
  "Type of feature transformation applied:", trasformazioni_applicate, "\n",
  "Cost:", final_cost, "\n",
  "Gamma:", final_gamma, "\n",
  "\nMetriche di performance sul training set:\n",
  "R²:", r_squared_training_svm, "\n",
  "Mean Absolute Error (MAE):", mae_training_svm, "\n",
  "Root Mean Squared Error (RMSE):", rmse_training_svm, "\n",
  "\nPerformance metrics on the test set:\n",
  "R²:", r_squared_test_svm, "\n",
  "Mean Absolute Error (MAE):", mae_test_svm, "\n",
  "Root Mean Squared Error (RMSE):", rmse_test_svm, "\n"
)

writeLines(parametri_testo_svm, con = params_output_file_svm)


model_path_svm <- file.path(model_dir, "best_model.rds")
saveRDS(best_model_svm, model_path_svm)



results_svm <- best_model_svm$resample  # This will give you a data frame with metrics for each fold
print(results_svm)


library(e1071)  # Per il modello SVM
library(readr)  # Per leggere i file

model_path <- file.path(model_dir, "best_model.rds")
best_model_svm <- readRDS(model_path)

data_path <- prediction_file
prediction_data <- read_delim(data_path, delim = ";")

head(prediction_data)

predictions <- predict(best_model_svm, prediction_data)

prediction_data$Predicted <- predictions

output_path <- file.path(model_dir, "predictions_with_inputs.txt")
write.table(prediction_data, file = output_path, row.names = FALSE, col.names = TRUE, sep = ";")

cat("Tabella con dati di input e previsioni salvata in:", output_path, "\n")

saveRDS(train_data, file = file.path(model_dir, "train_data.rds"))
saveRDS(test_data,  file = file.path(model_dir, "test_data.rds"))
# capturing outputs
print('Serialization of model_dir')
file <- file(paste0('/tmp/model_dir_', id, '.json'))
writeLines(toJSON(model_dir, auto_unbox=TRUE), file)
close(file)
print('Serialization of predictors')
file <- file(paste0('/tmp/predictors_', id, '.json'))
writeLines(toJSON(predictors, auto_unbox=TRUE), file)
close(file)
print('Serialization of target_variable')
file <- file(paste0('/tmp/target_variable_', id, '.json'))
writeLines(toJSON(target_variable, auto_unbox=TRUE), file)
close(file)
print('Serialization of target_variable_uom')
file <- file(paste0('/tmp/target_variable_uom_', id, '.json'))
writeLines(toJSON(target_variable_uom, auto_unbox=TRUE), file)
close(file)
