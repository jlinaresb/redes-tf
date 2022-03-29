# Deconvolution


# In train
# ===
train = read.csv('~/projects/redes-tf/data/train.csv')
test = read.csv('~/projects/redes-tf/data/test.csv')
train = rbind.data.frame(train, test)
rm(test)

train = subset(train, select = -c(target))
train = t(train)
train = cbind.data.frame(gene = rownames(train), train)
rownames(train) = 1:nrow(train)

write.table(train, file = '~/projects/redes-tf/deconvolution/train_deconvolution.tsv' , quote = F, col.names = T, row.names = F)


# In validation
# ===
validation = read.csv('~/projects/redes-tf/data/validation.csv')
validation = subset(validation, select = -c(target))
validation = t(validation)
validation = cbind.data.frame(gene = rownames(validation), validation)
rownames(validation) = 1:nrow(validation)

write.table(validation, file = '~/projects/redes-tf/deconvolution/validation_deconvolution.tsv' , quote = F, col.names = T, row.names = F)
