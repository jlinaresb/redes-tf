require(GSVA)

# Load expression data
# ===
inDir = '~/projects/redes-tf/data/expression/'
setwd(inDir)
load('expression_Raul.RData')
data = as.matrix(exp.PREC)

# Import signature
# ===
signature = colnames(read.csv('~/projects/redes-tf/data/train.csv'))[-1]

# Select only signature in expression data
# ===
data = data[signature,]

# Run GSVA!
# ===
methods = c('gsva', 'plage', 'ssgsea', 'zscore')
res = list()
for (i in methods) {
  gsva = gsva(expr = data,
       gset.idx.list = list(signature),
       method = i,
       kcdf = 'Gaussian')
  gsva = as.data.frame(t(gsva))
  names(gsva) = i
  res[[i]] = gsva
}
res = as.data.frame(res)


# Save results! 
save(res, file = '~/projects/redes-tf/results/gsva_signature.RData')
