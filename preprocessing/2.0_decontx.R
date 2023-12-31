library(Matrix)
species <- 'human'

counts <- readMM(sprintf("../data/intermediate_files/1_preprocessed_%s_data.mtx", species))
samples <- read.csv(sprintf("../data/intermediate_files/1_preprocessed_%s_data_sample_ids.csv", species))
counts <- as(counts, "dgCMatrix")

library(celda)
res <- decontX(x=t(counts), 
               batch=samples$sample)

write.csv(res$contamination, sprintf('../data/intermediate_files/2_decontx_%s_contamination_proportion.csv', species))
writeMM(res$decontXcounts, sprintf('../data/intermediate_files/2_decontx_%s_data.mtx', species))