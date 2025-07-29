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
if (!requireNamespace("scales", quietly = TRUE)) {
	install.packages("scales", repos="http://cran.us.r-project.org")
}
library(scales)
if (!requireNamespace("tidyr", quietly = TRUE)) {
	install.packages("tidyr", repos="http://cran.us.r-project.org")
}
library(tidyr)
if (!requireNamespace("reshape2", quietly = TRUE)) {
	install.packages("reshape2", repos="http://cran.us.r-project.org")
}
library(reshape2)



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
library(caret)
library(ggplot2)

config_output_path = '/tmp/data/output/'

input_file <- training_file

dati <- read.table(input_file, header = TRUE, sep = '\t')


param_file_gbm <- parameter_file

params <- read.table(param_file_gbm, header = TRUE, sep = '\t', fill = TRUE, stringsAsFactors = FALSE)

number <- as.numeric(params$value[params$Parameter == "number"])  # Numero di fold per cross-validation
target_variable <- as.character(params$value[params$Parameter == "Target variable"])
apply_pca <- as.logical(tolower(as.character(params$value[params$Parameter == "LM_PCA"])))


normalize <- as.logical(tolower(as.character(params$value[params$Parameter == "LM_Normalization"])) )
standardize <- as.logical(tolower(as.character(params$value[params$Parameter == "LM_Standardization"])) )
robust_scaling <- as.logical(tolower(as.character(params$value[params$Parameter == "LM_RobustScaling"])) )
loose_scaling <- as.logical(tolower(as.character(params$value[params$Parameter == "LM_LooseScaling"])) )

normalize <- ifelse(is.na(normalize), FALSE, normalize)
standardize <- ifelse(is.na(standardize), FALSE, standardize)
robust_scaling <- ifelse(is.na(robust_scaling), FALSE, robust_scaling)
loose_scaling <- ifelse(is.na(loose_scaling), FALSE, loose_scaling)
apply_pca <- ifelse(is.na(apply_pca), FALSE, apply_pca)  # PCA di default su FALSE

cat("target_variable:", target_variable, "\n")
cat("normalize:", normalize, "\n")
cat("standardize:", standardize, "\n")
cat("apply_pca:", apply_pca, "\n")  
cat("robust_scaling:", robust_scaling, "\n")
cat("loose_scaling:", loose_scaling, "\n")

str(dati)

predictors <- setdiff(names(dati), target_variable)

set.seed(123)  # Seme globale per riproducibilità
library(caret)  # Assicurati che il pacchetto caret sia installato
train_index <- createDataPartition(dati[[target_variable]], p = 0.9, list = FALSE)
train_data <- dati[train_index, ]
test_data <- dati[-train_index, ]
cat("Dimensione del training set:", nrow(train_data), "\n")
cat("Dimensione del test set:", nrow(test_data), "\n")

preProcSteps <- c()
if (normalize) {
    preProcSteps <- c(preProcSteps, "range")
    cat("Trasformazione applicata: Normalizzazione (range)\n")
}
if (standardize) {
    preProcSteps <- c(preProcSteps, "center", "scale")
    cat("Trasformazione applicata: Standardizzazione (centering e scaling)\n")
}
if (robust_scaling) {
    preProcSteps <- c(preProcSteps, "YeoJohnson")
    cat("Trasformazione applicata: Scaling robusto (Yeo-Johnson)\n")
}
if (loose_scaling) {
    preProcSteps <- c(preProcSteps, "range")
    cat("Trasformazione applicata: Scaling allentato (range)\n")
}

if (apply_pca) {
    preProcSteps <- c(preProcSteps, "pca")
    cat("Trasformazione applicata: PCA (Analisi delle Componenti Principali)\n")
}

predictors <- setdiff(names(train_data), target_variable)

if (length(preProcSteps) > 0) {
    preProcValues <- preProcess(train_data[, predictors], method = preProcSteps)
    
    train_data_transformed <- predict(preProcValues, train_data[, predictors])

    test_data_transformed <- predict(preProcValues, test_data[, predictors])
    
    if (apply_pca) {
        pca_model <- preProcValues$rotation
        pca_variance <- attr(preProcValues, "importance")[, 1]  # La varianza spiegata
        cumulative_variance <- cumsum(pca_variance)  # Calcola la varianza cumulativa
        
        num_components <- which(cumulative_variance >= 0.90)[1]  # Primo componente che supera il 90%
        
        if (!is.na(num_components)) {
            cat("Numero di componenti principali da mantenere per il 90% di varianza:", num_components, "\n")
            
            train_data_transformed <- train_data_transformed[, 1:num_components]
            test_data_transformed <- test_data_transformed[, 1:num_components]
        } else {
            warning("Nessun componente principale trovato che spieghi almeno il 90% della varianza.")
        }
    }
} else {
    train_data_transformed <- train_data[, predictors]
    test_data_transformed <- test_data[, predictors]
}

train_data_final <- cbind(train_data_transformed, train_data[, target_variable, drop = FALSE])
test_data_final <- cbind(test_data_transformed, test_data[, target_variable, drop = FALSE])

cat("Prime 5 righe di train_data_final:\n")
print(head(train_data_final, 5))

cat("Prime 5 righe di test_data_final:\n")
print(head(test_data_final, 5))


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
                  data = train_data_final, 
                  method = "lm", 
                  trControl = ctrl)

print(model_lm)

output_base_dir <- config_output_path
new_dir_lm <- file.path(output_base_dir, "LinearModel")
if (!dir.exists(new_dir_lm)) {
  dir.create(new_dir_lm, recursive = TRUE)
}

predictions_lm_test <- predict(model_lm, newdata = test_data_final)

results_lm <- data.frame(Actual = test_data_final[[target_variable]], Predicted = predictions_lm_test)

plot_lm_test <- ggplot(data = results_lm, aes(x = Actual, y = Predicted)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(title = "Linear Model - Test Set",
       x = paste("Observed", target_variable),
       y = paste("Predicted", target_variable)) +
  theme_minimal()

print(plot_lm_test)

plot_path_lm <- file.path(new_dir_lm, "lm_plot_test_set.png")
ggsave(filename = plot_path_lm, plot = plot_lm_test, width = 8, height = 6)

cat("Grafico salvato in:", plot_path_lm, "\n")



train_lm_preds <- predict(model_lm, newdata = train_data_final)

results_lm_train <- data.frame(Actual = train_data_final[[target_variable]], Predicted = train_lm_preds)

plot_lm_training <- ggplot(data = results_lm_train, aes(x = Actual, y = Predicted)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(
    title = "Linear Model - Training Set",
    x = paste("Observed", target_variable),
    y = paste("Predicted", target_variable)
  ) +
  theme_minimal()

print(plot_lm_training)
plot_path_lm_training <- file.path(new_dir_lm, "lm_plot_training_set.png")  # Percorso del grafico
ggsave(filename = plot_path_lm_training, plot = plot_lm_training, width = 8, height = 6)


params_output_file <- file.path(new_dir_lm, "descrizione_parametri_modello.txt")

variabile_target <- target_variable  # Sostituisci con il nome della tua variabile target

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
  "Descrizione del modello di regressione lineare:\n",
  "Variabile target:", variabile_target, "\n",
  "\nMetriche di performance sul training set:\n",
  "R²:", round(r_squared_training, 4), "\n",
  "Mean Absolute Error (MAE):", round(mae_training, 4), "\n",
  "Root Mean Squared Error (RMSE):", round(rmse_training, 4), "\n",
  "\nMetriche di performance sul test set:\n",
  "R²:", round(r_squared_test, 4), "\n",
  "Mean Absolute Error (MAE):", round(mae_test, 4), "\n",
  "Root Mean Squared Error (RMSE):", round(rmse_test, 4), "\n"
)

writeLines(parametri_testo, con = params_output_file)

cat("File di descrizione dei parametri salvato in:", params_output_file, "\n")



print(importance_values)

output_base_dir <- config_output_path
new_dir_lm <- file.path(output_base_dir, "LinearModel")
if (!dir.exists(new_dir_lm)) {
  dir.create(new_dir_lm, recursive = TRUE)
}

output_path_lm <- file.path(new_dir_lm, "result_model_lm.txt")
writeLines(capture.output(print(model_lm)), output_path_lm)

model_path_lm <- file.path(new_dir_lm, "model_lm.rds")
saveRDS(model_lm, model_path_lm)

cat("Modello salvato in:", model_path_lm, "\n")
write.csv(as.data.frame(importance_values$importance), file = importance_file)

cat("L'importanza delle variabili è stata salvata in:", importance_file, "\n")


results_lm <- model_lm$resample  # Questo ti darà un dataframe con le metriche per ogni fold
output_metrics_path <- file.path(new_dir_lm, "cross_validation_metrics.txt")
write.table(results_lm, file = output_metrics_path, sep = "\t", row.names = FALSE)

print(results_lm)




print("SHAP Analisi")


library(iml)

predictor_svm <- Predictor$new(model_lm, data = train_data_final[, predictors], y = train_data_final[[target_variable]])

shap_values_all <- lapply(1:nrow(test_data_final), function(i) {
  Shapley$new(predictor_svm, x.interest = test_data_final[i, predictors, drop = FALSE])
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

library(ggplot2)

shap_df_sorted <- data.frame(
  Variable = names(sorted_shap),
  MeanAbsSHAP = sorted_shap
)

bar_plot <- ggplot(shap_df_sorted, aes(x = reorder(Variable, -MeanAbsSHAP), y = MeanAbsSHAP)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(
    title = "Media Assoluta dei Valori SHAP per Variabile",
    x = "Variabile",
    y = "Media Assoluta SHAP"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


png_file <- file.path(new_dir_lm, "shap_importance_bar_plot.png")          
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
    title = "SHAP Summary Plot con Gradiente di Colore Separato per Ogni Variabile",
    x = "Valore SHAP (phi)",
    y = "Variabili",
    color = "Intensità della variabile"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(legend.position = "right")

png_file <- file.path(new_dir_lm, "shap_summary_plot.png")
ggsave(filename = png_file, 
       plot = shap_summary_plot, width = 8, height = 6)





 


library(iml)

predictor_svm <- Predictor$new(model_lm, data = train_data_final[, predictors], y = train_data_final[[target_variable]])

shap_values_all <- lapply(1:nrow(test_data_final), function(i) {
  Shapley$new(predictor_svm, x.interest = test_data_final[i, predictors, drop = FALSE])
})





shap_values_matrix <- sapply(shap_values_all, function(shap_obj) {
  if (!is.null(shap_obj$results)) {
    return(as.numeric(shap_obj$results$phi.var))  # Assumendo che 'phi.var' contenga i valori SHAP
  } else {
    return(rep(NA, length(predictors)))  # Ritorna NA se non ci sono valori SHAP
  }
})

shap_values_matrix <- t(shap_values_matrix)

shap_values_df <- as.data.frame(shap_values_matrix)

colnames(shap_values_df) <- predictors  # 'predictors' è la lista dei nomi delle variabili

mean_absolute_shap <- apply(shap_values_df, 2, function(x) mean(abs(x), na.rm = TRUE))

sorted_shap <- sort(mean_absolute_shap, decreasing = TRUE)

library(ggplot2)

shap_df_sorted <- data.frame(
  Variable = names(sorted_shap),
  MeanAbsSHAP = sorted_shap
)

bar_plot <- ggplot(shap_df_sorted, aes(x = reorder(Variable, -MeanAbsSHAP), y = MeanAbsSHAP)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(
    title = "Media Assoluta dei Valori SHAP per Variabile",
    x = "Variabile",
    y = "Media Assoluta SHAP"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

                            
png_file <- file.path(new_dir_lm, "shap_importance_bar_plot_phi_var.png")
ggsave(filename = png_file, 
       plot = bar_plot, width = 8, height = 6)

shap_data <- data.frame(
  SHAP = unlist(lapply(shap_values_all, function(shap_obj) shap_obj$results$phi.var)),
  Variable = rep(names(sorted_shap), each = nrow(shap_values_df)),
  FeatureValue = unlist(lapply(shap_values_all, function(shap_obj) shap_obj$results$feature.value)),
  FeatureNumericValue = unlist(lapply(shap_values_all, function(shap_obj) {
    as.numeric(sub(".*=", "", shap_obj$results$feature.value))  # Estrae solo il valore numerico della variabile
  }))
)

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
    title = "SHAP Summary Plot con Gradiente di Colore Separato per Ogni Variabile",
    x = "Valore SHAP (phi.var)",
    y = "Variabili",
    color = "Intensità della variabile"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(legend.position = "right")

png_file <- file.path(new_dir_lm, "shap_summary_plot_phi_var.png")
ggsave(filename = png_file, 
       plot = shap_summary_plot, width = 8, height = 6)


library(ggplot2)
library(tidyr)

predictions_svm_test <- predict(model_lm, newdata = test_data_final)

results_svm_test_df <- data.frame(
  Actual = test_data_final[[target_variable]],
  Predicted_svm = predictions_svm_test
)

results_svm_test_long <- results_svm_test_df %>%
  pivot_longer(cols = c("Actual", "Predicted_svm"),
               names_to = "Type",
               values_to = "Value")

violin_plot_svm <- ggplot(results_svm_test_long, aes(x = Type, y = Value, fill = Type)) +
  geom_violin(trim = FALSE) +  # Mostra tutta la distribuzione
  scale_fill_manual(values = c("blue", "red")) +  # Imposta colori per dati reali e predetti
  labs(title = paste("Confronto tra Dati Osservati e Predetti per", target_variable),
       x = "Tipo di Dato",
       y = paste("Valore di", target_variable)) +
  theme_minimal() +
  theme(legend.position = "none")  # Rimuovi la legenda

print(violin_plot_svm)

plot_path_violin_svm_test <- file.path(new_dir_lm, "violin_plot_test_set_svm.png")
ggsave(filename = plot_path_violin_svm_test, plot = violin_plot_svm, width = 8, height = 6)

cat("Grafico a violino SVM salvato in:", plot_path_violin_svm_test, "\n")

library(e1071)  # Per il modello SVM
library(readr)  # Per leggere i file

model_path <- file.path(new_dir_lm, "model_lm.rds")
model_lm <- readRDS(model_path)

data_path <- prediction_file
prediction_data <- read_delim(data_path, delim = "\t")

head(prediction_data)

predictions <- predict(model_lm, prediction_data)

prediction_data$Predicted <- predictions

output_path <- file.path(new_dir_lm, "predictions_with_inputs.txt")
write.table(prediction_data, file = output_path, row.names = FALSE, col.names = TRUE, sep = "\t")

cat("Tabella con dati di input e previsioni salvata in:", output_path, "\n")
# capturing outputs
print('Serialization of new_dir_lm')
file <- file(paste0('/tmp/new_dir_lm_', id, '.json'))
writeLines(toJSON(new_dir_lm, auto_unbox=TRUE), file)
close(file)
