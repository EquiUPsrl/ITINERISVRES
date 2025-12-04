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
library(iml)
library(e1071)
library(readr)
library(tidyr)
library(caret)

config_base_path <- "/tmp/data/WF4"
output_path = file.path(config_base_path, "output")

input_file <- training_file

dati <- read.table(input_file, header = TRUE, sep = ';')


param_file_gbm <- parameter_file

params <- read.table(param_file_gbm, header = TRUE, sep = ';', fill = TRUE, stringsAsFactors = FALSE)

number <- as.numeric(params$value[params$Parameter == "number"])  # Numero di fold per cross-validation
target_variable <- as.character(params$value[params$Parameter == "Target variable"])
target_variable_uom <- as.character(params$value[params$Parameter == "Target variable UoM"])
apply_pca <- as.logical(tolower(as.character(params$value[params$Parameter == "LM_PCA"])))
training_data_percentage <- as.numeric(params$value[params$Parameter == "Training data percentage"])


str(dati)

predictors <- setdiff(names(dati), target_variable)

preProcSteps <- c()
if (tolower(as.character(params$value[params$Parameter == "LM_Normalization"])) == "true") preProcSteps <- c(preProcSteps, "range")
if (tolower(as.character(params$value[params$Parameter == "LM_Standardization"])) == "true") preProcSteps <- c(preProcSteps, "center", "scale")
if (tolower(as.character(params$value[params$Parameter == "LM_ICA"])) == "true") preProcSteps <- c(preProcSteps, "ica")
if (tolower(as.character(params$value[params$Parameter == "LM_PCA"])) == "true") preProcSteps <- c(preProcSteps, "pca")
cat("Selected Preprocessing:", paste(preProcSteps, collapse = ", "), "\n")

set.seed(123)  # Seme globale per riproducibilità
train_index <- createDataPartition(dati[[target_variable]], p = training_data_percentage, list = FALSE)
train_data <- dati[train_index, ]
test_data <- dati[-train_index, ]
cat("Size of training set:", nrow(train_data), "\n")
cat("Size of test set:", nrow(test_data), "\n")


seeds <- vector(mode = "list", length = number + 1)
for (i in 1:number) {
  seeds[[i]] <- as.integer(sample.int(1000, 1))
}
seeds[[number + 1]] <- as.integer(sample.int(1000, 1))

ctrl <- trainControl(
  method = "cv", 
  number = number, 
  seeds = seeds, 
  verboseIter = TRUE
)

formula_rf <- as.formula(paste(target_variable, "~ ."))

model_lm <- train(formula_rf, 
                  data = train_data, 
                  method = "lm", 
                  preProcess = preProcSteps,
                  trControl = ctrl)

print(model_lm)

output_base_dir <- output_path
model_dir <- file.path(output_base_dir, "Linear_Model")
if (!dir.exists(model_dir)) {
  dir.create(model_dir, recursive = TRUE)
}

predictions_lm_test <- predict(model_lm, newdata = test_data)

results_lm <- data.frame(Actual = test_data[[target_variable]], Predicted = predictions_lm_test)

plot_lm_test <- ggplot(data = results_lm, aes(x = Actual, y = Predicted)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(title = "Linear Model - Test Set",
       x = paste("Observed", target_variable, "(", target_variable_uom, ")"),
       y = paste("Predicted", target_variable, "(", target_variable_uom, ")")) +
  theme_minimal()

print(plot_lm_test)

plot_path_lm <- file.path(model_dir, "lm_plot_test_set.png")
ggsave(filename = plot_path_lm, plot = plot_lm_test, width = 8, height = 6)
cat("Chart saved in: ", plot_path_lm, "\n")


train_lm_preds <- predict(model_lm, newdata = train_data)

results_lm_train <- data.frame(Actual = train_data[[target_variable]], Predicted = train_lm_preds)

plot_lm_training <- ggplot(data = results_lm_train, aes(x = Actual, y = Predicted)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(
    title = "Linear Model - Training Set",
    x = paste("Observed", target_variable, "(", target_variable_uom, ")"),
    y = paste("Predicted", target_variable, "(", target_variable_uom, ")")
  ) +
  theme_minimal()

print(plot_lm_training)
plot_path_lm_training <- file.path(model_dir, "lm_plot_training_set.png")  # Percorso del grafico
ggsave(filename = plot_path_lm_training, plot = plot_lm_training, width = 8, height = 6)
cat("Chart saved in: ", plot_path_lm_training, "\n")


params_output_file <- file.path(model_dir, "model_parameters_description.txt")

variabile_target <- target_variable  # Sostituisci con il nome della tua variabile target
trasformazioni_applicate <- if (exists("preProcSteps") && length(preProcSteps) > 0) {
  paste(preProcSteps, collapse = ", ")
} else {
  "None"
}

dati_training <- train_data  # Assicurati di avere i dati di training disponibili

r_squared_training <- R2(train_lm_preds, dati_training[[variabile_target]])

mae_training <- mean(abs(train_lm_preds - dati_training[[variabile_target]]))

mse_training <- mean((train_lm_preds - dati_training[[variabile_target]]) ^ 2)
rmse_training <- sqrt(mse_training)

dati_test <- test_data  # Assicurati di avere i dati di test disponibili

r_squared_test <- R2(predictions_lm_test, dati_test[[variabile_target]])


mae_test <- mean(abs(predictions_lm_test - dati_test[[variabile_target]]))

mse_test <- mean((predictions_lm_test - dati_test[[variabile_target]]) ^ 2)
rmse_test <- sqrt(mse_test)

parametri_testo <- paste(
  "Description of the Multiple Linear Regression Model:\n",
  "Target Variable:", variabile_target, "\n",
  "Type of feature transformation applied:", trasformazioni_applicate, "\n",
  "\nPerformance metrics on the training set:\n",
  "R²:", round(r_squared_training, 4), "\n",
  "Mean Absolute Error (MAE):", round(mae_training, 4), "\n",
  "Root Mean Squared Error (RMSE):", round(rmse_training, 4), "\n",
  "\nPerformance metrics on the test set:\n",
  "R²:", round(r_squared_test, 4), "\n",
  "Mean Absolute Error (MAE):", round(mae_test, 4), "\n",
  "Root Mean Squared Error (RMSE):", round(rmse_test, 4), "\n"
)


writeLines(parametri_testo, con = params_output_file)

cat("Parameter description file saved in: ", params_output_file, "\n")

output_path_lm <- file.path(model_dir, "result_model.txt")
writeLines(capture.output(print(model_lm)), output_path_lm)

model_path_lm <- file.path(model_dir, "best_model.rds")
saveRDS(model_lm, model_path_lm)
cat("Model saved in: ", model_path_lm, "\n")


results_lm <- model_lm$resample  # Questo ti darà un dataframe con le metriche per ogni fold

print(results_lm)


model_path <- file.path(model_dir, "best_model.rds")
model_lm <- readRDS(model_path)

data_path <- prediction_file
prediction_data <- read_delim(data_path, delim = ";")

head(prediction_data)

predictions <- predict(model_lm, prediction_data)

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
# capturing outputs
print('Serialization of model_dir')
file <- file(paste0('/tmp/model_dir_', id, '.json'))
writeLines(toJSON(model_dir, auto_unbox=TRUE), file)
close(file)
