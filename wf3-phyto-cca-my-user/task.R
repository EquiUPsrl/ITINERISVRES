setwd('/app')
library(optparse)
library(jsonlite)

if (!requireNamespace("readr", quietly = TRUE)) {
	install.packages("readr", repos="http://cran.us.r-project.org")
}
library(readr)
if (!requireNamespace("vegan", quietly = TRUE)) {
	install.packages("vegan", repos="http://cran.us.r-project.org")
}
library(vegan)



print('option_list')
option_list = list(

make_option(c("--abio_file"), action="store", default=NA, type="character", help="my description"),
make_option(c("--bio_file_filtered"), action="store", default=NA, type="character", help="my description"),
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

print("Retrieving abio_file")
var = opt$abio_file
print(var)
var_len = length(var)
print(paste("Variable abio_file has length", var_len))

abio_file <- gsub("\"", "", opt$abio_file)
print("Retrieving bio_file_filtered")
var = opt$bio_file_filtered
print(var)
var_len = length(var)
print(paste("Variable bio_file_filtered has length", var_len))

bio_file_filtered <- gsub("\"", "", opt$bio_file_filtered)
id <- gsub('"', '', opt$id)


print("Running the cell")
library(vegan)


config_base_path <- "/tmp/data/WF3/"

config_input_path <- paste(config_base_path, "data", sep="")
config_output_path <- paste(config_base_path, "output", sep="")

bio_path  <- bio_file_filtered
abio_path <- abio_file

output_base_dir <- config_output_path
cca_dir <- file.path(output_base_dir, "CCA")
if (!dir.exists(cca_dir)) {
  dir.create(cca_dir, recursive = TRUE)
}





library(readr)

bio  <- read_csv(bio_path,  show_col_types = FALSE)
abio <- read_csv(abio_path, show_col_types = FALSE)

cat("Dimensions bio:  ", dim(bio),  "\n")
cat("Dimensions abio: ", dim(abio), "\n")

rownames(bio)  <- bio[[grep("^ID$", names(bio))]]
rownames(abio) <- abio[[grep("^ID$", names(abio))]]

bio$ID  <- NULL
abio$ID <- NULL

bio[]  <- lapply(bio,  as.numeric)
abio[] <- lapply(abio, as.numeric)

common_ids <- intersect(rownames(bio), rownames(abio))

bio2  <- bio[common_ids, , drop = FALSE]
abio2 <- abio[common_ids, , drop = FALSE]

keep <- complete.cases(bio2) & complete.cases(abio2)

bio3  <- bio2[keep, , drop = FALSE]
abio3 <- abio2[keep, , drop = FALSE]

keep2 <- rowSums(bio3) > 0

bio3  <- bio3[keep2, , drop = FALSE]
abio3 <- abio3[keep2, , drop = FALSE]

stopifnot(nrow(bio3) > 0, identical(rownames(bio3), rownames(abio3)))





cat("Samples after NA / all-zero removal:", nrow(bio3), "\n")



make_var_table <- function(mod, prefix = "CCA", n_axes = 4, file_out) {
  eig <- mod$CCA$eig
  total_constr <- sum(eig)
  if (length(eig) == 0 || total_constr == 0) {
    warning("No eigenvalue found.")
    return(NULL)
  }
  n <- min(n_axes, length(eig))
  axes_ids <- paste0(prefix, 1:n)
  prop     <- eig[1:n] / total_constr
  cum_prop <- cumsum(prop)
  var_table <- data.frame(
    Axis       = axes_ids,
    Eigenvalue = eig[1:n],
    Proportion = prop,
    Cumulative = cum_prop,
    stringsAsFactors = FALSE
  )
  write.csv(var_table, file_out, row.names = FALSE)
  cat("Saved:", file_out, "\n")
  invisible(var_table)
}



bio_use <- bio3
abio_use <- abio3

cat("Bio dim: ", dim(bio_use),  "\n")
cat("Abio dim:", dim(abio_use), "\n")
        
mod_cca1 <- cca(bio_use ~ ., data = abio_use)
print(mod_cca1)

set.seed(123)
cat("\n=== CCA1: Global test (499 permutations) ===\n")
print(anova(mod_cca1, permutations = 499))
cat("\n=== CCA1: Test by axis (499 permutations) ===\n")
print(anova(mod_cca1, by = "axis", permutations = 499))

cca1_variance_path <- file.path(cca_dir, "cca1_axis_variance_axis1_4.csv")
make_var_table(mod_cca1, prefix = "CCA", n_axes = 4, file_out = cca1_variance_path)

env_sc1 <- scores(mod_cca1, display = "bp", choices = 1:2, scaling = 2)

env_df1 <- as.data.frame(env_sc1)
colnames(env_df1) <- c("CCA1", "CCA2")
env_df1$Variable <- rownames(env_df1)
env_df1 <- env_df1[, c("Variable", "CCA1", "CCA2")]

cca1_env_loadings_path <- file.path(cca_dir, "cca1_env_loadings_axes1_2.csv")
write.csv(env_df1, cca1_env_loadings_path, row.names = FALSE)
cat("Saved: ", cca1_env_loadings_path, "\n")

sites_sc1 <- scores(mod_cca1, display = "sites", choices = 1:2, scaling = 2)

site_df1 <- data.frame(
  SampleID = rownames(sites_sc1),
  CCA1_raw = as.numeric(sites_sc1[, 1]),
  CCA2_raw = as.numeric(sites_sc1[, 2]),
  stringsAsFactors = FALSE
)

site_df1$CCA1_std <- as.numeric(scale(site_df1$CCA1_raw))
site_df1$CCA2_std <- as.numeric(scale(site_df1$CCA2_raw))

cca1_site_scores_path <- file.path(cca_dir, "cca1_site_scores_axes1_2.csv")
write.csv(site_df1, cca1_site_scores_path, row.names = FALSE)
cat("Saved: ", cca1_site_scores_path,"\n")



bio_use <- log(bio3 + 1)
abio_use <- abio3

mod_cca2 <- cca(bio_use ~ ., data = abio_use)
print(mod_cca2)

set.seed(123)
cat("\n=== CCA2: Global test (499 permutations) ===\n")
print(anova(mod_cca2, permutations = 499))
cat("\n=== CCA2: Test by axis (499 permutations) ===\n")
print(anova(mod_cca2, by = "axis", permutations = 499))

cca2_variance_path <- file.path(cca_dir, "cca2_axis_variance_axis1_4_test.csv")
make_var_table(mod_cca2, prefix = "CCA", n_axes = 4, file_out = cca2_variance_path)

sites_sc2 <- scores(mod_cca2, display = "sites", choices = 1:2, scaling = 2)

site_df2 <- data.frame(
  SampleID = rownames(sites_sc2),
  CCA1_raw = as.numeric(sites_sc2[, 1]),
  CCA2_raw = as.numeric(sites_sc2[, 2]),
  stringsAsFactors = FALSE
)

site_df2$CCA1_std <- as.numeric(scale(site_df2$CCA1_raw))
site_df2$CCA2_std <- as.numeric(scale(site_df2$CCA2_raw))

cca2_site_scores_path <- file.path(cca_dir, "cca2_site_scores_axes1_2.csv")
write.csv(site_df2, cca2_site_scores_path, row.names = FALSE)
cat("Saved: ", cca2_site_scores_path, "\n")
# capturing outputs
print('Serialization of cca1_env_loadings_path')
file <- file(paste0('/tmp/cca1_env_loadings_path_', id, '.json'))
writeLines(toJSON(cca1_env_loadings_path, auto_unbox=TRUE), file)
close(file)
print('Serialization of cca2_site_scores_path')
file <- file(paste0('/tmp/cca2_site_scores_path_', id, '.json'))
writeLines(toJSON(cca2_site_scores_path, auto_unbox=TRUE), file)
close(file)
