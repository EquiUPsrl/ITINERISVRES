setwd('/app')
library(optparse)
library(jsonlite)

if (!requireNamespace("ggplot2", quietly = TRUE)) {
	install.packages("ggplot2", repos="http://cran.us.r-project.org")
}
library(ggplot2)
if (!requireNamespace("tidyr", quietly = TRUE)) {
	install.packages("tidyr", repos="http://cran.us.r-project.org")
}
library(tidyr)



print('option_list')
option_list = list(

make_option(c("--model_dir"), action="store", default=NA, type="character", help="my description"),
make_option(c("--target_variable"), action="store", default=NA, type="character", help="my description"),
make_option(c("--target_variable_uom"), action="store", default=NA, type="character", help="my description"),
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

print("Retrieving model_dir")
var = opt$model_dir
print(var)
var_len = length(var)
print(paste("Variable model_dir has length", var_len))

model_dir <- gsub("\"", "", opt$model_dir)
print("Retrieving target_variable")
var = opt$target_variable
print(var)
var_len = length(var)
print(paste("Variable target_variable has length", var_len))

target_variable <- gsub("\"", "", opt$target_variable)
print("Retrieving target_variable_uom")
var = opt$target_variable_uom
print(var)
var_len = length(var)
print(paste("Variable target_variable_uom has length", var_len))

target_variable_uom <- gsub("\"", "", opt$target_variable_uom)
id <- gsub('"', '', opt$id)


print("Running the cell")
library(ggplot2)
library(tidyr)

test_data  <- readRDS(file.path(model_dir, "test_data.rds"))
best_model  <- readRDS(file.path(model_dir, "best_model.rds"))
uom <- target_variable_uom

predictions_test <- predict(best_model, newdata = test_data)

results_test_df <- data.frame(
  Observed = test_data[[target_variable]],
  Predicted = predictions_test
)

results_test_long <- results_test_df %>%
  pivot_longer(cols = c("Observed", "Predicted"),
               names_to = "Type",
               values_to = "Value")

violin_plot <- ggplot(results_test_long, aes(x = Type, y = Value, fill = Type)) +
  geom_violin(trim = FALSE) +  # Mostra tutta la distribuzione
  scale_fill_manual(values = c("blue", "red")) +  # Imposta colori per dati reali e predetti
  labs(title = paste("Observed vs Predicted Values ", target_variable),
       x = " Data Type",
       y = paste("Value of ", target_variable, "(", uom, ")")) +
  theme_minimal() +
  theme(legend.position = "none")  # Rimuovi la legenda

print(violin_plot)

plot_path_violin_test <- file.path(model_dir, "violin_plot_test_set.png")
ggsave(filename = plot_path_violin_test, plot = violin_plot, width = 8, height = 6)

cat("Violin plot saved in:", plot_path_violin_test, "\n")
