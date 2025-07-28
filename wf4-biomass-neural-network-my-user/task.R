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

{'name': 'conf_base_path', 'assignation': "conf_base_path<-'/tmp/data/'"}
{'name': 'conf_output_path', 'assignation': "conf_output_path<-'/tmp/data/output/'"}

print("Running the cell")
library(caret)
library(nnet)
library(ggplot2)
library(Metrics)
library(doParallel)

input_file <- training_file
param_file <- parameter_file

cat("Leggo Training data.txt\n")
dati <- read.table(input_file, header = TRUE, sep = '\t', fill = TRUE)

cat("Leggo Parameter data.txt\n")
params <- read.table(param_file, header = TRUE, sep = '\t', fill = TRUE)

cat("Extract parameters\n")

number <- as.numeric(params$value[params$Parameter == "number"])
size <- as.numeric(unlist(strsplit(as.character(params$value[params$Parameter == "size"]), ",")))  # Size for hidden layer
decay <- as.numeric(unlist(strsplit(as.character(params$value[params$Parameter == "decay"]), ",")))  # Decay values
maxit <- as.numeric(unlist(strsplit(as.character(params$value[params$Parameter == "maxit"]), ",")))    # Max iterations

target_variable <- as.character(params$value[params$Parameter == "Target variable"])
apply_pca <- as.logical(tolower(as.character(params$value[params$Parameter == "NN_PCA"])))

if (is.na(size) || length(size) == 0) size <- c(5)  # Default to 5 if invalid
if (is.na(decay) || length(decay) == 0) decay <- c(0.01)  # Default to 0.01 if invalid
if (is.na(maxit) || length(maxit) == 0) maxit <- c(200)  # Default to 200 if invalid

cat("Target variable:", target_variable, "\n")
cat("Sizes:", size, "\n")
cat("Decays:", decay, "\n")
cat("Max iterations:", maxit, "\n")

normalize <- as.logical(tolower(as.character(params$value[params$Parameter == "NN_Normalization"])) )
standardize <- as.logical(tolower(as.character(params$value[params$Parameter == "NN_Standardization"])) )
robust_scaling <- as.logical(tolower(as.character(params$value[params$Parameter == "NN_RobustScaling"])) )
loose_scaling <- as.logical(tolower(as.character(params$value[params$Parameter == "NN_LooseScaling"])) )

normalize <- ifelse(is.na(normalize), FALSE, normalize)
standardize <- ifelse(is.na(standardize), FALSE, standardize)
robust_scaling <- ifelse(is.na(robust_scaling), FALSE, robust_scaling)
loose_scaling <- ifelse(is.na(loose_scaling), FALSE, loose_scaling)
apply_pca <- ifelse(is.na(apply_pca), FALSE, apply_pca)  # PCA di default su FALSE

if (!(target_variable %in% colnames(dati))) {
    stop("La variabile target non esiste nei dati.")
}

if (any(is.na(dati[[target_variable]]))) {
    stop("La variabile target contiene valori mancanti.")
}

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
set.seed(123)
for (i in 1:number) {
    seeds[[i]] <- sample.int(1000, size = 5)
}
seeds[[number + 1]] <- sample.int(1000, 1)

ncores <- 1 
cl <- makePSOCKcluster(ncores)  # Use all cores except one
registerDoParallel(cl)

results <- list()  # List to save models
best_model_nn <- NULL  # To save the best model
best_metric <- Inf  # Set an initial very high metric (for minimization)

for (s_value in size) {
    for (d_value in decay) {
        for (m_value in maxit) {
            cat("Esecuzione del modello Neural Network con size =", s_value, 
                ", decay =", d_value, ", maxit =", m_value, "\n")
            
            tuneGrid_nn <- expand.grid(size = s_value, decay = d_value)

            ctrl <- trainControl(method = "cv",
                                 number = number,
                                 seeds = seeds,
                                 allowParallel = TRUE,  # Allow parallel processing
                                 verboseIter = TRUE)

            model_nn <- train(as.formula(paste(target_variable, "~ .")), 
                              data = train_data_final, 
                              method = "nnet",   # Method: Neural Network
                              trControl = ctrl, 
                              tuneGrid = tuneGrid_nn, 
                              linout = TRUE, 
                              trace = FALSE, 
                              maxit = m_value)  # Use the current maxit value

            results[[paste0("size_", s_value, "_decay_", d_value, "_maxit_", m_value)]] <- model_nn

            metric_value <- min(model_nn$results$MAE)  # Example: if optimizing MAE
            
            if (metric_value < best_metric) {
                best_metric <- metric_value
                best_model_nn <- model_nn
            }
        }
    }
}

cat("Sommario del miglior modello:\n")
print(best_model_nn)

stopCluster(cl)
output_base_dir <- conf_output_path
new_dir_nn <- file.path(output_base_dir, "NeuralNetwork_Model")
if (!dir.exists(new_dir_nn)) {
  dir.create(new_dir_nn, recursive = TRUE)
}

predictions_nn_test <- predict(best_model_nn, newdata = test_data_final)

results_nn_test_df <- data.frame(Actual = test_data_final[[target_variable]], Predicted_nn = predictions_nn_test)

plot_nn_test <- ggplot(data = results_nn_test_df, aes(x = Actual, y = Predicted_nn)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(title = "Neural Network - Test Set",
       x = paste("Observed", target_variable),
       y = paste("Predicted", target_variable)) +
  theme_minimal()

print(plot_nn_test)

plot_path_nn <- file.path(new_dir_nn, "nn_plot_test_set.png")
ggsave(filename = plot_path_nn, plot = plot_nn_test, width = 8, height = 6)

cat("Grafico salvato in:", plot_path_nn, "\n")

train_nn_preds <- predict(best_model_nn, newdata = train_data_final)

results_nn_training_df <- data.frame(Actual = train_data_final[[target_variable]], Predicted_nn = train_nn_preds)

plot_nn_training <- ggplot(data = results_nn_training_df, aes(x = Actual, y = Predicted_nn)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = "dashed") +
  labs(
    title = "Neural Network - Training Set",
    x = paste("Observed", target_variable),
    y = paste("Predicted", target_variable)
  ) +
  theme_minimal()

print(plot_nn_training)

plot_path_nn <- file.path(new_dir_nn, "nn_plot_training_set.png")
ggsave(filename = plot_path_nn, plot = plot_nn_training, width = 8, height = 6)

params_output_file <- file.path(new_dir_nn, "descrizione_parametri_modello.txt")

final_size <- best_model_nn$bestTune$size
final_decay <- best_model_nn$bestTune$decay

variabile_target <- target_variable
trasformazioni_applicate <- if (exists("preProcSteps") && length(preProcSteps) > 0) {
  paste(preProcSteps, collapse = ", ")
} else {
  "Nessuna"
}

r_squared_training <- cor(results_nn_training_df$Actual, results_nn_training_df$Predicted_nn)^2
mae_training <- mean(abs(results_nn_training_df$Actual - results_nn_training_df$Predicted_nn))
rmse_training <- sqrt(mean((results_nn_training_df$Actual - results_nn_training_df$Predicted_nn)^2))

r_squared_test <- cor(results_nn_test_df$Actual, results_nn_test_df$Predicted_nn)^2
mae_test <- mean(abs(results_nn_test_df$Actual - results_nn_test_df$Predicted_nn))
rmse_test <- sqrt(mean((results_nn_test_df$Actual - results_nn_test_df$Predicted_nn)^2))

parametri_testo <- paste(
  "Descrizione del modello Neural Network ottimizzato:\n",
  "Variabile target:", variabile_target, "\n",
  "Trasformazioni applicate:", trasformazioni_applicate, "\n",
  "Size:", final_size, "\n",
  "Decay:", final_decay, "\n",
  "\nMetriche di performance sul training set:\n",
  "R²:", r_squared_training, "\n",
  "Mean Absolute Error (MAE):", mae_training, "\n",
  "Root Mean Squared Error (RMSE):", rmse_training, "\n",
  "\nMetriche di performance sul test set:\n",
  "R²:", r_squared_test, "\n",
  "Mean Absolute Error (MAE):", mae_test, "\n",
  "Root Mean Squared Error (RMSE):", rmse_test, "\n"
)

writeLines(parametri_testo, con = params_output_file)

cat("File di descrizione dei parametri salvato in:", params_output_file, "\n")



print(importance_df)

write.csv(importance_df, file = importance_file, row.names = FALSE)

cat("L'importanza delle variabili è stata salvata in:", importance_file, "\n")

results_nn <- best_model_nn$resample
output_metrics_path <- file.path(new_dir_nn, "cross_validation_metrics.txt")
write.table(results_nn, file = output_metrics_path, sep = "\t", row.names = FALSE)

model_path_nn <- file.path(new_dir_nn, "best_model_nn.rds")
saveRDS(best_model_nn, model_path_nn)

cat("Risultati di cross-validation salvati in:", output_metrics_path, "\n")
cat("Modello salvato in:", model_path_nn, "\n")




print("SHAP Analisi")


library(iml)

predictor_svm <- Predictor$new(best_model_nn, data = train_data_final[, predictors], y = train_data_final[[target_variable]])

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

plot_path_nn <- file.path(new_dir_nn, "shap_importance_bar_plot.png")
ggsave(filename = plot_path_nn, 
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

plot_path_nn <- file.path(new_dir_nn, "shap_summary_plot.png")
ggsave(filename = plot_path_nn, 
       plot = shap_summary_plot, width = 8, height = 6)


 



library(iml)

predictor_svm <- Predictor$new(best_model_nn, data = train_data_final[, predictors], y = train_data_final[[target_variable]])

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

plot_path_nn <- file.path(new_dir_nn, "shap_importance_bar_plot_phi_var.png")
ggsave(filename = plot_path_nn, 
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

plot_path_nn <- file.path(new_dir_nn, "shap_summary_plot_phi_var.png")
ggsave(filename = plot_path_nn, 
       plot = shap_summary_plot, width = 8, height = 6)


library(ggplot2)
library(tidyr)

predictions_svm_test <- predict(best_model_nn, newdata = test_data_final)

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

plot_path_violin_svm_test <- file.path(new_dir_nn, "violin_plot_test_set_svm.png")
ggsave(filename = plot_path_violin_svm_test, plot = violin_plot_svm, width = 8, height = 6)

cat("Grafico a violino SVM salvato in:", plot_path_violin_svm_test, "\n")

library(e1071)  # Per il modello SVM
library(readr)  # Per leggere i file

model_path <- file.path(new_dir_nn, "best_model_nn.rds")
best_model_nn <- readRDS(model_path)

data_path <- prediction_file
prediction_data <- read_delim(data_path, delim = "\t")

head(prediction_data)

predictions <- predict(best_model_nn, prediction_data)

prediction_data$Predicted <- predictions

output_path <- file.path(new_dir_nn, "predictions_with_inputs.txt")
write.table(prediction_data, file = output_path, row.names = FALSE, col.names = TRUE, sep = "\t")

cat("Tabella con dati di input e previsioni salvata in:", output_path, "\n")
# capturing outputs
print('Serialization of new_dir_nn')
file <- file(paste0('/tmp/new_dir_nn_', id, '.json'))
writeLines(toJSON(new_dir_nn, auto_unbox=TRUE), file)
close(file)
