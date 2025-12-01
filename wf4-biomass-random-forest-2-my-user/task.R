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



print('option_list')
option_list = list(

make_option(c("--parameter_file"), action="store", default=NA, type="character", help="my description"),
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
        cat("Running the Random Forest model with ntree =", ntree, ", mtry =", m_value, "\n")

        tuneGrid_rf <- expand.grid(mtry = m_value)

        ctrl <- trainControl(method = "cv",
                             number = number,
                             seeds = seeds,
                             allowParallel = TRUE,
                             verboseIter = TRUE)

        model_rf <- train(as.formula(paste(target_variable, "~ .")),
                          data = train_data,
                          method = "rf",
                          trControl = ctrl,
                          tuneGrid = tuneGrid_rf,
                          preProcess=preProcSteps,
                          ntree = ntree,
    }
}


stopCluster(cl)
