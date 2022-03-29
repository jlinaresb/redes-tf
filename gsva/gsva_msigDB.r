# # # Signature correlation
# # # ===
require(GSVA)
require(qusage)
require(corrplot)
# 
# # Load expression data
# # ===
# inDir = '~/projects/redes-tf/data/expression/'
# setwd(inDir)
# load('expression_Raul.RData')
# data = as.matrix(exp.PREC)
# 
# # Import signature
# # ===
# signature = colnames(read.csv('~/projects/redes-tf/data/train.csv'))[-1]
# 
# # Read MSigDB signatures
# # ===
# c7 = read.gmt('~/projects/redes-tf/data/signatures/c7.immunesigdb.v7.5.1.symbols.gmt')
# c2 = read.gmt('~/projects/redes-tf/data/signatures/c2.cp.reactome.v7.5.1.symbols.gmt')
# head(c7)
# 
# c7$FCBF = signature
# 
# 
# save(c7, data, file = '~/projects/redes-tf/data/signatures/C7_toRun.RData')

load('~/projects/redes-tf/data/signatures/C2_toRun.RData')

# Run GSVA!
# ===
methods = 'ssgsea'
gsva = gsva(expr = data,
            gset.idx.list = c2,
            method = methods,
            kcdf = 'Gaussian')
gsva = as.data.frame(t(gsva))

saveRDS(gsva, file = '~/projects/redes-tf/results/gsva/msigDB_c2_ssgsea.rds')

# Corrplot
# ===
# cor = cor(gsva)
# cor = cor[,which(colnames(cor) == 'FCBF')]
# dim(cor)
# 
# res = data.frame(
#   FCBF = cor,
#   row.names = names(cor)
# )
# 
# saveRDS(res, file = '~/projects/redes-tf/results/gsva/msigDB_c2_gsva.rds')
# plot = corrplot(cor(gsva), type = 'lower', order = 'hclust')

