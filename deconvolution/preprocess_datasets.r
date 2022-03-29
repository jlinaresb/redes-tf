# Deconvolution
load('~/projects/redes-tf/data/expression/expression_Raul.RData')

# In train
# ===
train = exp.DAT1
train = cbind.data.frame(gene = rownames(train), train)
rownames(train) = 1:nrow(train)

write.table(train, file = '~/projects/redes-tf/deconvolution/train_deconvolution.tsv' , quote = F, col.names = T, row.names = F)


# In validation
# ===
validation = exp.DAT2
validation = cbind.data.frame(gene = rownames(validation), validation)
rownames(validation) = 1:nrow(validation)

write.table(validation, file = '~/projects/redes-tf/deconvolution/validation_deconvolution.tsv' , quote = F, col.names = T, row.names = F)
