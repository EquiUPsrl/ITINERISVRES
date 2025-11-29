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

make_option(c("--remote_parameter_file"), action="store", default=NA, type="character", help="my description"),
make_option(c("--remote_prediction_file"), action="store", default=NA, type="character", help="my description"),
make_option(c("--remote_training_file"), action="store", default=NA, type="character", help="my description"),
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

print("Retrieving remote_parameter_file")
var = opt$remote_parameter_file
print(var)
var_len = length(var)
print(paste("Variable remote_parameter_file has length", var_len))

remote_parameter_file <- gsub("\"", "", opt$remote_parameter_file)
print("Retrieving remote_prediction_file")
var = opt$remote_prediction_file
print(var)
var_len = length(var)
print(paste("Variable remote_prediction_file has length", var_len))

remote_prediction_file <- gsub("\"", "", opt$remote_prediction_file)
print("Retrieving remote_training_file")
var = opt$remote_training_file
print(var)
var_len = length(var)
print(paste("Variable remote_training_file has length", var_len))

remote_training_file <- gsub("\"", "", opt$remote_training_file)
id <- gsub('"', '', opt$id)


print("Running the cell")
config_base_path <- "/tmp/data/WF4"
input_path = file.path(config_base_path, 'input')

training_url = remote_training_file
prediction_url = remote_prediction_file
parameter_url = remote_parameter_file

download_to_folder <- function(url, folder, filename = NULL, binary = TRUE) {
  if (!dir.exists(folder)) {
    dir.create(folder, recursive = TRUE)
  }
  
  if (is.null(filename)) {
    filename <- basename(url)
  }
  
  dest_file <- file.path(folder, filename)
  
  mode <- if (binary) "wb" else "w"
  
  download.file(url, destfile = dest_file, mode = mode)

  cat("File saved: ", dest_file, '\n')
  
  return(dest_file)
}

training_file <- download_to_folder(
  url = training_url,
  folder = input_path
)

prediction_file <- download_to_folder(
  url = prediction_url,
  folder = input_path
)

parameter_file <- download_to_folder(
  url = parameter_url,
  folder = input_path
)



file_paths <- c(
  training_file,
  prediction_file
)

for (file_path in file_paths) {
    cat("\n==============================\n")
    cat("📄 File:", basename(file_path), "\n")
    cat("==============================\n")
    
    prima_riga <- readLines(file_path, n = 1)
    num_tabs <- length(gregexpr(";", prima_riga)[[1]])
    
    if (num_tabs == 0) {
        cat("❌ No ; separator found: file ignored.\n")
        next
    }
    cat("✅ Separator detected with ", num_tabs + 1, "columns.\n")
    
    dati <- tryCatch({
        read.table(file_path, header = TRUE, sep = ";", stringsAsFactors = FALSE)
    }, error = function(e) {
        cat("❌ Errore nella lettura del file:", e$message, "\n")
        return(NULL)
    })
    
    if (is.null(dati)) next
    
    is_empty <- dati == ""
    is_NA <- is.na(dati)
    righe_valide <- rep(TRUE, nrow(dati))
    righe_valide <- righe_valide & !apply(
      is_empty,
      MARGIN = 1,  # Applica la funzione riga per riga
      FUN = function(x) any(x, na.rm = TRUE)  # Verifica se almeno un valore è TRUE ignorando eventuali NA
    )
    righe_valide <- righe_valide & !apply(
      is_NA,
      MARGIN = 1,  # Analizza ogni riga
      FUN = function(x) any(x, na.rm = TRUE)  # Verifica se almeno un valore è TRUE (=NA trovato)
    )
    
    suppressWarnings({
    dati_num <- as.data.frame(lapply(dati, as.numeric))
    })
    is_non_num <- is.na(dati_num) & !is.na(as.matrix(dati))
    righe_valide <- righe_valide & !apply(
      is_non_num,
      MARGIN = 1,  # Analizza ogni riga
      FUN = function(x) any(x, na.rm = TRUE)  # Verifica se almeno un valore è TRUE (=NA trovato)
    )
    
    n_eliminate <- sum(!righe_valide)
    cat("Total rows:", nrow(dati), "\n")
    cat("Deleted rows for invalid values:", n_eliminate, "\n")
    cat("Remaining rows:", sum(righe_valide), "\n")
    
    dati_puliti <- dati[righe_valide, , drop = FALSE]
    
    write.table(dati_puliti, file_path, sep = ";", row.names = FALSE, quote = FALSE)
    cat("✅ File overwritten with clean data.\n")
}

training_file <- training_file
prediction_file <- prediction_file
# capturing outputs
print('Serialization of parameter_file')
file <- file(paste0('/tmp/parameter_file_', id, '.json'))
writeLines(toJSON(parameter_file, auto_unbox=TRUE), file)
close(file)
print('Serialization of prediction_file')
file <- file(paste0('/tmp/prediction_file_', id, '.json'))
writeLines(toJSON(prediction_file, auto_unbox=TRUE), file)
close(file)
print('Serialization of training_file')
file <- file(paste0('/tmp/training_file_', id, '.json'))
writeLines(toJSON(training_file, auto_unbox=TRUE), file)
close(file)
