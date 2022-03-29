# Run Cibersort
# ===
hpc = 'local' #local cesga

if (hpc == 'cesga') {
  cbsrt = '/home/ulc/co/jlb/git/bulk-deconv/cibersort.r'
  lmd22.path = '/mnt/netapp2/Store_uni/home/ulc/co/jlb/bulk-deConv/data/LM22.txt'
} else if (hpc == 'local'){
  cbsrt = '~/git/bulk-deConv/cibersort.r'
  lmd22.path = '~/projects/bulk-deConv/data/signatures/LM22.txt'
}

source(cbsrt)

# Arguments
# ===
setwd('~/projects/redes-tf/deconvolution/')
outDir = '~/projects/redes-tf/deconvolution/results/'
nperm = 1000
# signature for blood cells
lmd22 = read.table(lmd22.path, sep = '\t', header = T, row.names = 1)
# bulk data (mixture)

bulk.files = list.files(pattern = 'tsv')
pb = txtProgressBar(min = 0,
                    max = length(bulk.files),
                    style = 3,
                    width = 50,
                    char = '=')

for (i in seq_along(bulk.files)) {
  
  # Load bulk (already prepared)
  bulk = read.table(bulk.files[i], header = T, row.names = 1)
  # run CIBERSORT!
  print(paste0('Deconvoluting ', bulk.files[i], ' ...'))
  results <- CIBERSORT(sig_matrix = lmd22,
                       mixture_file = bulk,
                       perm=nperm,
                       QN=TRUE,
                       absolute = F,
                       abs_method = 'sig.score')
  
  # Save CIBERSORT results!
  saveRDS(results, 
          file = paste0(outDir, 'cibsersort_', gsub('.tsv', '.rds', bulk.files[i])))
  
  setTxtProgressBar(pb, i)
}
