setwd('/app')
library(optparse)
library(jsonlite)

if (!requireNamespace("caret", quietly = TRUE)) {
	install.packages("caret", repos="http://cran.us.r-project.org")
}
library(caret)
if (!requireNamespace("doParallel", quietly = TRUE)) {
	install.packages("doParallel", repos="http://cran.us.r-project.org")
}
library(doParallel)
if (!requireNamespace("e1071", quietly = TRUE)) {
	install.packages("e1071", repos="http://cran.us.r-project.org")
}
library(e1071)
if (!requireNamespace("fastICA", quietly = TRUE)) {
	install.packages("fastICA", repos="http://cran.us.r-project.org")
}
library(fastICA)
if (!requireNamespace("ggplot2", quietly = TRUE)) {
	install.packages("ggplot2", repos="http://cran.us.r-project.org")
}
library(ggplot2)
if (!requireNamespace("Metrics", quietly = TRUE)) {
	install.packages("Metrics", repos="http://cran.us.r-project.org")
}
library(Metrics)
if (!requireNamespace("randomForest", quietly = TRUE)) {
	install.packages("randomForest", repos="http://cran.us.r-project.org")
}
library(randomForest)
if (!requireNamespace("readr", quietly = TRUE)) {
	install.packages("readr", repos="http://cran.us.r-project.org")
}
library(readr)



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
library(caret)
library(randomForest)
library(doParallel)
library(Metrics)
library(ggplot2)
library(fastICA)

config_base_path <- "/tmp/data/WF4"
output_path = file.path(config_base_path, "output")

input_file <- training_file
param_file <- parameter_file

if (!file.exists(input_file) || !file.exists(param_file)) {
    stop("One or both input files do not exist.")
}

dati <- read.table(input_file, header = TRUE, sep = ';', fill = TRUE)
params <- read.table(param_file, header = TRUE, sep = ';', fill = TRUE)

number <- as.numeric(params$value[params$Parameter == "number"])
target_variable <- as.character(params$value[params$Parameter == "Target variable"])
target_variable_uom <- as.character(params$value[params$Parameter == "Target variable UoM"])
apply_pca <- as.logical(tolower(as.character(params$value[params$Parameter == "RF_PCA"])))
training_data_percentage <- as.numeric(params$value[params$Parameter == "Training data percentage"])

predictors <- setdiff(names(dati), target_variable)


ntree_row <- params[params$Parameter == "ntree", ]
mtry_row  <- params[params$Parameter == "mtry",  ]

ntree_values <- as.numeric(ntree_row[ , -1])
ntree_values <- ntree_values[!is.na(ntree_values)]
mtry_values  <- as.numeric(mtry_row[ , -1])
mtry_values <- mtry_values[!is.na(mtry_values)]

cat("ntree_values:\n")
print(ntree_values)
cat("mtry_values:\n")
print(mtry_values)


if (any(is.na(ntree_values)) || length(ntree_values) == 0) {
    stop("Invalid ntree values.")
}
if (any(is.na(mtry_values)) || length(mtry_values) == 0) {
    stop("Invalid mtry values.")
}

if (!(target_variable %in% colnames(dati))) {
    stop("Target variable does not exist in the data.")
}

if (any(is.na(dati[[target_variable]]))) {
    stop("Target variable contains missing values.")
}

preProcSteps <- c()
if (tolower(as.character(params$value[params$Parameter == "RF_Normalization"])) == "true") preProcSteps <- c(preProcSteps, "range")
if (tolower(as.character(params$value[params$Parameter == "RF_Standardization"])) == "true") preProcSteps <- c(preProcSteps, "center", "scale")
if (tolower(as.character(params$value[params$Parameter == "RF_ICA"])) == "true") preProcSteps <- c(preProcSteps, "ica")
if (tolower(as.character(params$value[params$Parameter == "RF_PCA"])) == "true") preProcSteps <- c(preProcSteps, "pca")
cat("Selected Preprocessing:", paste(preProcSteps, collapse = ", "), "\n")

metric <- 'MAE'
if (tolower(as.character(params$value[params$Parameter == "metric_value_MAE"])) == "true") metric <- 'MAE'
if (tolower(as.character(params$value[params$Parameter == "metric_value_RMSE"])) == "true") metric <- 'RMSE'
if (tolower(as.character(params$value[params$Parameter == "metric_value_Rsquared"])) == "true") metric <- 'Rsquared'

cat("metric_value: ", metric, "\n")

set.seed(123)  # Seme globale per riproducibilità
library(caret)  # Assicurati che il pacchetto caret sia installato
train_index <- createDataPartition(dati[[target_variable]], p = training_data_percentage, list = FALSE)
train_data <- dati[train_index, ]
test_data <- dati[-train_index, ]
cat("Size of training set:", nrow(train_data), "\n")
cat("Size of test set:", nrow(test_data), "\n")

formula_rf <- as.formula(paste(target_variable, "~ ."))

num_models <- length(mtry_values)
seeds <- vector(mode = "list", length = number + 1)
for (i in 1:number) {
    seeds[[i]] <- sample.int(1000, num_models)  # Vettore di semi di lunghezza pari al numero di modelli
}
seeds[[number + 1]] <- sample.int(1000, 1)  # L'ultimo elemento deve contenere almeno un intero

ctrl <- trainControl(method = "cv", number = number, seeds = seeds, verboseIter = TRUE)

ncores <- 1 
cl <- makePSOCKcluster(ncores)  # Use all cores except one
registerDoParallel(cl)

results <- list()  # List to save models
best_model_rf <- NULL  # To save the best model
best_metric <- Inf  # Set an initial very high metric (for minimization)

for (ntree in ntree_values) {
    for (m_value in mtry_values) {
        cat("Running the Random Forest model with ntree =", ntree, 
            ", mtry =", m_value, "\n")

        tuneGrid_rf <- expand.grid(mtry = m_value)

        ctrl <- trainControl(method = "cv",
                             number = number,         # Number of folds for CV
                             seeds = seeds,           # Seed for reproducibility
                             allowParallel = TRUE,    # Allow parallel processing
                             verboseIter = TRUE)      # Print iteration details

        model_rf <- train(as.formula(paste(target_variable, "~ .")), 
                          data = train_data, 
                          method = "rf",       # Method: Random Forest
                          trControl = ctrl, 
                          tuneGrid = tuneGrid_rf,
                          preProcess=preProcSteps,
                          ntree = ntree,       # Use the current ntree value

        results[[paste0("ntree_", ntree, "_mtry_", m_value)]] <- model_rf

        metric_value <- min(model_rf$results[[metric]])  # Example: if optimizing MAE
        cat("calculated metric_value:", metric_value, "\n")
        
        if (metric_value < best_metric) {
            best_metric <- metric_value
            best_model_rf <- model_rf
        }
    }
}

cat("Best model:\n")
print(best_model_rf)

stopCluster(cl)

output_base_dir <- output_path
model_dir <- file.path(output_base_dir, "Random_Forest_Model")
if (!dir.exists(model_dir)) {
  dir.create(model_dir, recursive = TRUE)
}

train_rf_preds <- predict(best_model_rf, train_data)
results_rf_training_df <- data.frame(Actual = train_data[[target_variable]], Predicted = train_rf_preds)

plot_rf_training <- ggplot(data = results_rf_training_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(title = paste("Random Forest - Training Set for", target_variable),
       x = paste("Observed", target_variable, "(", target_variable_uom, ")"),
       y = paste("Predicted", target_variable, "(", target_variable_uom, ")")) +
  theme_minimal()

print(plot_rf_training)

plot_path_rf <- file.path(model_dir, "rf_plot_training_set.png")
ggsave(filename = plot_path_rf, plot = plot_rf_training, width = 8, height = 6)
cat("Chart saved in: ", plot_path_rf, "\n")

predictions_rf_test <- predict(best_model_rf, newdata = test_data)

results_rf_test_df <- data.frame(Actual = test_data[[target_variable]], Predicted = predictions_rf_test)

plot_rf_test <- ggplot(data = results_rf_test_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(title = paste("Random Forest - Test Set for", target_variable),
       x = paste("Observed", target_variable, "(", target_variable_uom, ")"),
       y = paste("Predicted", target_variable, "(", target_variable_uom, ")")) +
  theme_minimal()

print(plot_rf_test)

plot_path_rf <- file.path(model_dir, "rf_plot_test_set.png")
ggsave(filename = plot_path_rf, plot = plot_rf_test, width = 8, height = 6)
cat("Chart saved in: ", plot_path_rf, "\n")

params_output_file <- file.path(model_dir, "model_parameters_description.txt")

final_ntree <- best_model_rf$finalModel$ntree
final_mtry <- best_model_rf$bestTune$mtry

variabile_target <- target_variable
trasformazioni_applicate <- if (exists("preProcSteps") && length(preProcSteps) > 0) {
  paste(preProcSteps, collapse = ", ")
} else {
  "None"
}

r_squared_training <- cor(results_rf_training_df$Actual, results_rf_training_df$Predicted)^2
mae_training <- mean(abs(results_rf_training_df$Actual - results_rf_training_df$Predicted))
rmse_training <- sqrt(mean((results_rf_training_df$Actual - results_rf_training_df$Predicted)^2))

r_squared_test <- cor(results_rf_test_df$Actual, results_rf_test_df$Predicted)^2
mae_test <- mean(abs(results_rf_test_df$Actual - results_rf_test_df$Predicted))
rmse_test <- sqrt(mean((results_rf_test_df$Actual - results_rf_test_df$Predicted)^2))

parametri_testo <- paste(
  "Description of the Optimised Random Forest Model:\n",
  "Target Variable:", variabile_target, "\n",
  "Type of feature transformation applied:", trasformazioni_applicate, "\n",
  "ntree:", final_ntree, "\n",
  "mtry:", final_mtry, "\n",
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
cat("Parameter description file saved in: ", params_output_file, "\n")


model_path_rf <- file.path(model_dir, "best_model.rds")
saveRDS(best_model_rf, model_path_rf)
cat("Model saved in:", model_path_rf, "\n")


results_rf <- best_model_rf$resample

print(results_rf)


library(e1071)  # Per il modello SVM
library(readr)  # Per leggere i file

model_path <- file.path(model_dir, "best_model.rds")
best_model_rf <- readRDS(model_path)

data_path <- prediction_file
prediction_data <- read_delim(data_path, delim = ";")

head(prediction_data)

predictions <- predict(best_model_rf, prediction_data)

prediction_data$Predicted <- predictions

output_path <- file.path(model_dir, "predictions_with_inputs.txt")
write.table(prediction_data, file = output_path, row.names = FALSE, col.names = TRUE, sep = ";")

cat("Table with input data and forecasts saved in: ", output_path, "\n")

saveRDS(train_data, file = file.path(model_dir, "train_data.rds"))
saveRDS(test_data,  file = file.path(model_dir, "test_data.rds"))
