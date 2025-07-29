setwd('/app')
library(optparse)
library(jsonlite)




print('option_list')
option_list = list(

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

id <- gsub('"', '', opt$id)


print("Running the cell")
training_url <- param_training_file

prediction_url <- param_prediction_file

parameter_url <- param_parameter_file
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
