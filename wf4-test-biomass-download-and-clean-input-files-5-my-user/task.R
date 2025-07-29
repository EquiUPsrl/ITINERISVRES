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

{'name': 'conf_base_path', 'assignation': "conf_base_path='/tmp/data/'"}
{'name': 'conf_output_path', 'assignation': "conf_output_path='/tmp/data/output/'"}

print("Running the cell")
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

training_url <- param_training_file

training_file <- download_to_folder(
  url = training_url,
  folder = conf_base_path
)

prediction_url <- param_prediction_file
prediction_file <- download_to_folder(
  url = prediction_url,
  folder = conf_base_path
)

parameter_url <- param_parameter_file
parameter_file <- download_to_folder(
  url = parameter_url,
  folder = conf_base_path
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
    num_tabs <- length(gregexpr("\t", prima_riga)[[1]])
    
    if (num_tabs == 0) {
    cat("❌ Nessun separatore TAB trovato: file ignorato.\n")
    next
    }
    cat("✅ Separatore TAB rilevato con", num_tabs + 1, "colonne.\n")
    
    dati <- tryCatch({
    read.table(file_path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
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
    cat("Righe totali:", nrow(dati), "\n")
    cat("Righe eliminate per valori non validi:", n_eliminate, "\n")
    cat("Righe rimanenti:", sum(righe_valide), "\n")
    
    dati_puliti <- dati[righe_valide, , drop = FALSE]
    
    write.table(dati_puliti, file_path, sep = "\t", row.names = FALSE, quote = FALSE)
    cat("✅ File sovrascritto con dati puliti.\n")
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
