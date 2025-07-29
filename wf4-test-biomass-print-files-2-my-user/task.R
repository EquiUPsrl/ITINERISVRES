setwd('/app')
library(optparse)
library(jsonlite)




print('option_list')
option_list = list(

make_option(c("--parameter_url"), action="store", default=NA, type="character", help="my description"),
make_option(c("--prediction_url"), action="store", default=NA, type="character", help="my description"),
make_option(c("--training_url"), action="store", default=NA, type="character", help="my description"),
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

print("Retrieving parameter_url")
var = opt$parameter_url
print(var)
var_len = length(var)
print(paste("Variable parameter_url has length", var_len))

parameter_url <- gsub("\"", "", opt$parameter_url)
print("Retrieving prediction_url")
var = opt$prediction_url
print(var)
var_len = length(var)
print(paste("Variable prediction_url has length", var_len))

prediction_url <- gsub("\"", "", opt$prediction_url)
print("Retrieving training_url")
var = opt$training_url
print(var)
var_len = length(var)
print(paste("Variable training_url has length", var_len))

training_url <- gsub("\"", "", opt$training_url)
id <- gsub('"', '', opt$id)


print("Running the cell")
cat(training_url, '\n')
cat(prediction_url, '\n')
cat(parameter_url, '\n')
