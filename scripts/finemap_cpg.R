library(susieR)
library(argparse)

parser <- ArgumentParser(description = "Load args CpG SuSiE")
parser$add_argument("--cpg", type="character", help="ID of the CpG fine-map")
parser$add_argument("--data-dir", type="character", help="Directory for fine-mapping input data", default="~/data/finemap-godmc/data/finemapping_tmp/")
parser$add_argument("--out-dir", type="character", help="Output directory for SuSiE results",
                    default="~/data/finemap-godmc/data/susie_results/")

args <- parser$parse_args()
cpg <- args$cpg
data_dir <- args$data_dir

set.seed(42)

df <- read.csv(paste0(data_dir, cpg, '.csv'))
N <- round(median(df$samplesize))
R <- as.matrix(read.csv(paste0(data_dir, cpg, '_LD.txt'), header = FALSE))

fitted_z = susie_rss(z = df$Z, R = R, n = N, L = 8)

df$pip <- fitted_z$pip

# posterior mean effect size
post_mean_beta <- colSums(fitted_z$alpha * fitted_z$mu)
df$post_mean_beta <- post_mean_beta

# posterior SD of beta
df$post_sd_beta <- sqrt(pmax(colSums(fitted_z$alpha * fitted_z$mu2) - post_mean_beta^2, 0))

# credible set IDs
df$cs_id <- NA_character_
if (!is.null(fitted_z$sets$cs)) {
  for (i in seq_along(fitted_z$sets$cs)) {
    idx <- fitted_z$sets$cs[[i]]
    df$cs_id[idx] <- ifelse(
      is.na(df$cs_id[idx]),
      paste0("CS", i),
      paste(df$cs_id[idx], paste0("CS", i), sep = ";")
    )
  }
}

write.csv(df, paste0(args$out_dir, cpg, '_susie.csv'), row.names = FALSE)
