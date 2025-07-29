setwd('/app')
library(optparse)
library(jsonlite)

if (!requireNamespace("jsonlite", quietly = TRUE)) {
	install.packages("jsonlite", repos="http://cran.us.r-project.org")
}
library(jsonlite)



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

{'name': 'conf_base_path', 'assignation': "conf_base_path=''"}
{'name': 'conf_output_path', 'assignation': "conf_output_path=''"}

print("Running the cell")
library(jsonlite)

param_training_file = ''
param_prediction_file = ''
param_parameter_file = ''
conf_base_path = ''
conf_output_path = ''

if (file.exists(params_path)) {
  params <- fromJSON(param_file)

  param_training_file   = params$param_training_file
  param_prediction_file = params$param_prediction_file
  param_parameter_file  = params$param_parameter_file
  conf_base_path        = params$conf_base_path
  conf_output_path      = params$conf_output_path

  cat("✅ Parametri caricati correttamente.\n")
} else {
  stop(paste("❌ File dei parametri non trovato:", param_file))
}
