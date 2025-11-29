setwd('/app')
library(optparse)
library(jsonlite)

if (!requireNamespace("jsonlite", quietly = TRUE)) {
	install.packages("jsonlite", repos="http://cran.us.r-project.org")
}
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
if (!requireNamespace("ggplot2", quietly = TRUE)) {
	install.packages("ggplot2", repos="http://cran.us.r-project.org")
}
library(ggplot2)
if (!requireNamespace("iml", quietly = TRUE)) {
	install.packages("iml", repos="http://cran.us.r-project.org")
}
library(iml)
if (!requireNamespace("Metrics", quietly = TRUE)) {
	install.packages("Metrics", repos="http://cran.us.r-project.org")
}
library(Metrics)
if (!requireNamespace("readr", quietly = TRUE)) {
	install.packages("readr", repos="http://cran.us.r-project.org")
}
library(readr)
if (!requireNamespace("tidyr", quietly = TRUE)) {
	install.packages("tidyr", repos="http://cran.us.r-project.org")
}
library(tidyr)
if (!requireNamespace("xgboost", quietly = TRUE)) {
	install.packages("xgboost", repos="http://cran.us.r-project.org")
}
library(xgboost)
if (!requireNamespace("nnet", quietly = TRUE)) {
	install.packages("nnet", repos="http://cran.us.r-project.org")
}
library(nnet)
if (!requireNamespace("fastICA", quietly = TRUE)) {
	install.packages("fastICA", repos="http://cran.us.r-project.org")
}
library(fastICA)
if (!requireNamespace("randomForest", quietly = TRUE)) {
	install.packages("randomForest", repos="http://cran.us.r-project.org")
}
library(randomForest)
if (!requireNamespace("dplyr", quietly = TRUE)) {
	install.packages("dplyr", repos="http://cran.us.r-project.org")
}
library(dplyr)
if (!requireNamespace("MASS", quietly = TRUE)) {
	install.packages("MASS", repos="http://cran.us.r-project.org")
}
library(MASS)
if (!requireNamespace("scales", quietly = TRUE)) {
	install.packages("scales", repos="http://cran.us.r-project.org")
}
library(scales)



print('option_list')
option_list = list(

make_option(c("--params_path"), action="store", default=NA, type="character", help="my description"),
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

print("Retrieving params_path")
var = opt$params_path
print(var)
var_len = length(var)
print(paste("Variable params_path has length", var_len))

params_path <- gsub("\"", "", opt$params_path)
id <- gsub('"', '', opt$id)


print("Running the cell")
library(jsonlite)

training_url = ''
prediction_url = ''
parameter_url = ''

cat(paste("File dei parametri:", params_path))

if (file.exists(params_path)) {
    params <- fromJSON(params_path)
    
    training_url   = params$param_training_file
    prediction_url = params$param_prediction_file
    parameter_url  = params$param_parameter_file
    
    cat("✅ Parametri caricati correttamente.\n")
} else {
    stop("❌ Parameter file not found, aborting the task: ", params_path)
}
# capturing outputs
print('Serialization of parameter_url')
file <- file(paste0('/tmp/parameter_url_', id, '.json'))
writeLines(toJSON(parameter_url, auto_unbox=TRUE), file)
close(file)
print('Serialization of prediction_url')
file <- file(paste0('/tmp/prediction_url_', id, '.json'))
writeLines(toJSON(prediction_url, auto_unbox=TRUE), file)
close(file)
print('Serialization of training_url')
file <- file(paste0('/tmp/training_url_', id, '.json'))
writeLines(toJSON(training_url, auto_unbox=TRUE), file)
close(file)
