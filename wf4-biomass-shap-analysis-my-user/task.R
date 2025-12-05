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

make_option(c("--model_dir"), action="store", default=NA, type="character", help="my description"),
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
id <- gsub('"', '', opt$id)


print("Running the cell")
library(caret)
library(ggplot2)
library(iml)
library(xgboost)

config_base_path <- "/tmp/data/WF4"


model_info_path <- file.path(model_dir, "model_info.rds")
model_info <- readRDS(model_info_path)

model_file <- model_info$model_file
preProcess <- model_info$preProcess
train_data <- model_info$train_data
test_data <- model_info$test_data
predictors <- model_info$predictors
target_variable <- model_info$target_variable

cat("Model file: ", model_file, "\n")

ext <- tools::file_ext(model_file)

cat("Ext file: ", ext, "\n")

data_for_predictor <- train_data[, predictors, drop = FALSE]
test_data_proc <- test_data[, predictors, drop = FALSE]

if (ext == "xgb") {
    message("Loading XGBoost binary model: ", model_file)

    print(file.info(model_file)$size)

    
    best_model <- xgboost::xgb.load(model_file)

    if (!is.null(preProcess)) {
        message("Applying preProcess to training data")
        data_for_predictor <- predict(preProcess, train_data[, predictors])
        test_data_proc <- predict(preProcess, test_data[, predictors])
        predictors <- colnames(data_for_predictor)
    }
} else if (ext == "rds") {
    message("Loading RDS model: ", model_file)
    best_model <- readRDS(model_file)
} else {
    stop("Unsupported model file type: ", ext)
}





datasets <- c("train", "test")

for (ds in datasets) {

    cat("current dataset: ", ds)

    dataset = data_for_predictor
    if (ds == "test") {
        dataset = test_data_proc
    }

    shap_dir <- file.path(model_dir, paste("SHAP", ds, sep = "_"))
    if (!dir.exists(shap_dir)) {
      dir.create(shap_dir, recursive = TRUE)
    }

    message("Columns in data_for_predictor: ", paste(colnames(data_for_predictor), collapse = ", "))
    message("Columns in dataset row: ", paste(colnames(dataset), collapse = ", "))

    cat(predictors, "\n")

    if (inherits(best_model, "xgb.Booster")) {
      predict_fun <- function(model, newdata) {
        xgb_data <- as.matrix(newdata)
        predict(model, xgb_data)
      }
    } else {
      predict_fun <- function(model, newdata) {
        predict(model, newdata)
      }
    }

    
    predictor_model <- Predictor$new(best_model, data = data_for_predictor, y = train_data[[target_variable]], predict.fun = predict_fun)
    

    shap_values_all <- lapply(1:nrow(dataset), function(i) {
        
        x_interest <- dataset[i, predictors, drop = FALSE]

        str(x_interest)
        
        Shapley$new(predictor_model, x.interest = x_interest)
    })
    
    
    shap_values_matrix <- sapply(shap_values_all, function(shap_obj) {
      if (!is.null(shap_obj$results)) {
        return(as.numeric(shap_obj$results$phi))  # Assumendo che 'phi' contenga i valori SHAP
      } else {
        return(rep(NA, length(predictors)))  # Ritorna NA se non ci sono valori SHAP
      }
    })
    
    shap_values_matrix <- t(shap_values_matrix)
    
    shap_values_df <- as.data.frame(shap_values_matrix)
    
    colnames(shap_values_df) <- predictors  # 'predictors' è la lista dei nomi delle variabili
    
    mean_absolute_shap <- apply(shap_values_df, 2, function(x) mean(abs(x), na.rm = TRUE))
    
    sorted_shap <- sort(mean_absolute_shap, decreasing = TRUE)
    
    
    shap_df_sorted <- data.frame(
      Variable = names(sorted_shap),
      MeanAbsSHAP = sorted_shap
    )
    
    shap_data_file <- file.path(shap_dir, "shap_data_importance.csv")  
    write.csv(shap_df_sorted, shap_data_file, row.names = FALSE)
    
    bar_plot <- ggplot(shap_df_sorted, aes(x = reorder(Variable, -MeanAbsSHAP), y = MeanAbsSHAP)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      labs(
        title = "Feature Impact on Model Output (SHAP Values) ",
        x = "Features",
        y = "Mean SHAP value"
      ) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    
    png_file <- file.path(shap_dir, "shap_importance_bar_plot.png")          
    ggsave(filename = png_file, 
           plot = bar_plot, width = 8, height = 6)
    
    shap_data <- data.frame(
      SHAP = unlist(lapply(shap_values_all, function(shap_obj) shap_obj$results$phi)),
      Variable = rep(names(sorted_shap), each = nrow(shap_values_df)),
      FeatureValue = unlist(lapply(shap_values_all, function(shap_obj) shap_obj$results$feature.value)),
      FeatureNumericValue = unlist(lapply(shap_values_all, function(shap_obj) {
        as.numeric(sub(".*=", "", shap_obj$results$feature.value))  # Estrae solo il valore numerico della variabile
      }))
    )
    
    shap_data_file <- file.path(shap_dir, "shap_data_summary_plot.csv")  
    write.csv(shap_data, shap_data_file, row.names = FALSE)
    
    shap_data$Variable <- factor(shap_data$Variable, levels = names(sorted_shap))
    
    shap_summary_plot <- ggplot(shap_data, aes(x = SHAP, y = Variable)) +
      geom_point(aes(color = FeatureNumericValue), alpha = 0.7) +  # Usa il valore della variabile per il colore
      scale_color_gradientn(
        colors = c("red", "yellow", "green"),  # Colori: rosso per bassa intensità, giallo per neutrale, verde per alta intensità
        values = scales::rescale(c(min(shap_data$FeatureNumericValue), median(shap_data$FeatureNumericValue), max(shap_data$FeatureNumericValue))),  # Rescale in base al range per ciascuna variabile
        limits = c(min(shap_data$FeatureNumericValue), max(shap_data$FeatureNumericValue)),  # Limiti tra il valore minimo e massimo SHAP per ogni variabile
        breaks = c(min(shap_data$FeatureNumericValue), median(shap_data$FeatureNumericValue), max(shap_data$FeatureNumericValue)),  # Punti di rottura della scala di colori
        labels = c("Low", "Neutral", "High")  # Etichette per i valori estremi della colorazione
      ) +
      labs(
        title = "SHAP Summary Plot of Feature Importance",
        x = "SHAP value",
        y = "Feature",
        color = "Feature value"
      ) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      theme(legend.position = "right")
    
    png_file <- file.path(shap_dir, "shap_summary_plot.png")
    ggsave(filename = png_file, 
           plot = shap_summary_plot, width = 8, height = 6)
}

output_dir <- config_base_path
# capturing outputs
print('Serialization of output_dir')
file <- file(paste0('/tmp/output_dir_', id, '.json'))
writeLines(toJSON(output_dir, auto_unbox=TRUE), file)
close(file)
