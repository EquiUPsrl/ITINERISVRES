setwd('/app')
library(optparse)
library(jsonlite)




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

{'name': 'conf_base_path', 'assignation': "conf_base_path='/tmp/data/'"}
{'name': 'conf_output_path', 'assignation': "conf_output_path='/tmp/data/output/'"}

print("Running the cell")
cat(training_file, '\n')
cat(prediction_file, '\n')
cat(parameter_file, '\n')
