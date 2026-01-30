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

# get posterior mean effect size
post_mean_beta <- colSums(fitted_z$alpha * fitted_z$mu)
df$post_mean_beta <- post_mean_beta

write.csv(df, paste0(args$out_dir, cpg, '_susie.csv'), row.names = FALSE)
