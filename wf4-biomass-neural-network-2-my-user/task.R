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
library(caret)
library(nnet)
library(ggplot2)
library(Metrics)
library(doParallel)

config_base_path <- "/tmp/data/WF4"
output_path = file.path(config_base_path, "output")

input_file <- training_file
param_file <- parameter_file

dati <- read.table(input_file, header = TRUE, sep = ';', fill = TRUE)

params <- read.table(param_file, header = TRUE, sep = ';', fill = TRUE)

cat("Extract parameters\n")
number_row <- params[params$Parameter == "number", ]
size_row  <- params[params$Parameter == "size",  ]
decay_row  <- params[params$Parameter == "decay",  ]
maxit_row  <- params[params$Parameter == "maxit",  ]

number <- as.numeric(number_row[ , -1]) #questo può essere eliminato da qui?
number <- number[!is.na(number)] #questo può essere eliminato da qui?
size  <- as.numeric(size_row[ , -1])
size   <- size[!is.na(size)]
decay  <- as.numeric(decay_row[ , -1])
decay  <- decay[!is.na(decay)]
maxit  <- as.numeric(maxit_row[ , -1])
maxit  <- maxit[!is.na(maxit)]


target_variable <- as.character(params$value[params$Parameter == "Target variable"])
target_variable_uom <- as.character(params$value[params$Parameter == "Target variable UoM"])
apply_pca <- as.logical(tolower(as.character(params$value[params$Parameter == "NN_PCA"])))
training_data_percentage <- as.numeric(params$value[params$Parameter == "Training data percentage"])

cat("Target variable:", target_variable, "\n")
cat("Sizes:", size, "\n")
cat("Decays:", decay, "\n")
cat("Max iterations:", maxit, "\n")

if (!(target_variable %in% colnames(dati))) {
    stop("Target variable does not exist in the data.")
}

if (any(is.na(dati[[target_variable]]))) {
    stop("Target variable contains missing values.")
}

preProcSteps <- c()
if (tolower(as.character(params$value[params$Parameter == "NN_Normalization"])) == "true") preProcSteps <- c(preProcSteps, "range")
if (tolower(as.character(params$value[params$Parameter == "NN_Standardization"])) == "true") preProcSteps <- c(preProcSteps, "center", "scale")
if (tolower(as.character(params$value[params$Parameter == "NN_ICA"])) == "true") preProcSteps <- c(preProcSteps, "ica")
if (tolower(as.character(params$value[params$Parameter == "NN_PCA"])) == "true") preProcSteps <- c(preProcSteps, "pca")
cat("Selected Preprocessing: ", paste(preProcSteps, collapse = ", "), "\n")

metric <- 'MAE'
if (tolower(as.character(params$value[params$Parameter == "metric_value_MAE"])) == "true") metric <- 'MAE'
if (tolower(as.character(params$value[params$Parameter == "metric_value_RMSE"])) == "true") metric <- 'RMSE'
if (tolower(as.character(params$value[params$Parameter == "metric_value_Rsquared"])) == "true") metric <- 'Rsquared'

predictors <- setdiff(names(dati), target_variable)

set.seed(123)  # Seme globale per riproducibilità
library(caret)  # Assicurati che il pacchetto caret sia installato
train_index <- createDataPartition(dati[[target_variable]], p = training_data_percentage, list = FALSE)
train_data <- dati[train_index, ]
test_data <- dati[-train_index, ]
cat("Size of training set:", nrow(train_data), "\n")
cat("Size of test set:", nrow(test_data), "\n")

seeds <- vector(mode = "list", length = number + 1)
set.seed(123)
for (i in 1:number) {
    seeds[[i]] <- sample.int(1000, size = 5)
}
seeds[[number + 1]] <- sample.int(1000, 1)

ncores <- 1 
cl <- makePSOCKcluster(ncores)  # Use all cores except one
registerDoParallel(cl)

results <- list()  # List to save models
best_model_nn <- NULL  # To save the best model
best_metric <- Inf  # Set an initial very high metric (for minimization)
final_maxit <- NULL

for (s_value in size) {
    for (d_value in decay) {
        for (m_value in maxit) {
            cat("Running Neural Network model with size =", s_value, 
                ", decay =", d_value, ", maxit =", m_value, "\n")
            
            tuneGrid_nn <- expand.grid(size = s_value, decay = d_value)

            ctrl <- trainControl(method = "cv",
                                 number = number,
                                 seeds = seeds,
                                 allowParallel = TRUE,  # Allow parallel processing
                                 verboseIter = TRUE)

            model_nn <- train(as.formula(paste(target_variable, "~ .")), 
                              data = train_data, 
                              method = "nnet",   # Method: Neural Network
                              trControl = ctrl, 
                              tuneGrid = tuneGrid_nn,
                              preProcess = preProcSteps,
                              linout = TRUE, 
                              trace = FALSE, 
                              maxit = m_value)  # Use the current maxit value

            results[[paste0("size_", s_value, "_decay_", d_value, "_maxit_", m_value)]] <- model_nn

            metric_value <- min(model_nn$results[[metric]])  # Example: if optimizing MAE

            cat("min metric_value = ", metric_value, " (", metric, ")")
            
            if (metric_value < best_metric) {
                best_metric <- metric_value
                best_model_nn <- model_nn
                final_maxit <- m_value
            }
        }
    }
}

cat("Best model:\n")
print(best_model_nn)

stopCluster(cl)

output_base_dir <- output_path
model_dir <- file.path(output_base_dir, "Neural_Network_Model")
if (!dir.exists(model_dir)) {
  dir.create(model_dir, recursive = TRUE)
}

predictions_nn_test <- predict(best_model_nn, newdata = test_data)

results_nn_test_df <- data.frame(Actual = test_data[[target_variable]], Predicted_nn = predictions_nn_test)

plot_nn_test <- ggplot(data = results_nn_test_df, aes(x = Actual, y = Predicted_nn)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(title = "Neural Network - Test Set",
       x = paste("Observed", target_variable, "(", target_variable_uom, ")"),
       y = paste("Predicted", target_variable, "(", target_variable_uom, ")")) +
  theme_minimal()

print(plot_nn_test)

plot_path_nn <- file.path(model_dir, "nn_plot_test_set.png")
ggsave(filename = plot_path_nn, plot = plot_nn_test, width = 8, height = 6)
cat("Chart saved in: ", plot_path_nn, "\n")

train_nn_preds <- predict(best_model_nn, newdata = train_data)

results_nn_training_df <- data.frame(Actual = train_data[[target_variable]], Predicted_nn = train_nn_preds)

plot_nn_training <- ggplot(data = results_nn_training_df, aes(x = Actual, y = Predicted_nn)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(
    title = "Neural Network - Training Set",
    x = paste("Observed", target_variable, "(", target_variable_uom, ")"),
    y = paste("Predicted", target_variable, "(", target_variable_uom, ")")
  ) +
  theme_minimal()

print(plot_nn_training)

plot_path_nn <- file.path(model_dir, "nn_plot_training_set.png")
ggsave(filename = plot_path_nn, plot = plot_nn_training, width = 8, height = 6)
cat("Chart saved in: ", plot_path_nn, "\n")

params_output_file <- file.path(model_dir, "model_parameters_description.txt")

final_size <- best_model_nn$bestTune$size
final_decay <- best_model_nn$bestTune$decay

variabile_target <- target_variable
trasformazioni_applicate <- if (exists("preProcSteps") && length(preProcSteps) > 0) {
  paste(preProcSteps, collapse = ", ")
} else {
  "None"
}

r_squared_training <- cor(results_nn_training_df$Actual, results_nn_training_df$Predicted_nn)^2
mae_training <- mean(abs(results_nn_training_df$Actual - results_nn_training_df$Predicted_nn))
rmse_training <- sqrt(mean((results_nn_training_df$Actual - results_nn_training_df$Predicted_nn)^2))

r_squared_test <- cor(results_nn_test_df$Actual, results_nn_test_df$Predicted_nn)^2
mae_test <- mean(abs(results_nn_test_df$Actual - results_nn_test_df$Predicted_nn))
rmse_test <- sqrt(mean((results_nn_test_df$Actual - results_nn_test_df$Predicted_nn)^2))

parametri_testo <- paste(
  "Description of the Optimised Neural Network Model:\n",
  "Target Variable:", variabile_target, "\n",
  "Type of feature transformation applied:", trasformazioni_applicate, "\n",
  "Size:", final_size, "\n",
  "Decay:", final_decay, "\n",
  "Maxit:", final_maxit, "\n",
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






results_nn <- best_model_nn$resample

model_path_nn <- file.path(model_dir, "best_model.rds")
saveRDS(best_model_nn, model_path_nn)

cat("Model saved in: ", model_path_nn, "\n")


library(e1071)  # Per il modello SVM
library(readr)  # Per leggere i file

model_path <- file.path(model_dir, "best_model.rds")
best_model_nn <- readRDS(model_path)

data_path <- prediction_file
prediction_data <- read_delim(data_path, delim = ";")

head(prediction_data)

predictions <- predict(best_model_nn, prediction_data)

prediction_data$Predicted <- predictions

output_path <- file.path(model_dir, "predictions_with_inputs.txt")
write.table(prediction_data, file = output_path, row.names = FALSE, col.names = TRUE, sep = ";")

cat("Table with input data and forecasts saved in: ", output_path, "\n")


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
