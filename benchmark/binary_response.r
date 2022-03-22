setwd('~/projects/redes-tf/output/binary/')
files = list.files()
bmrs = list()
for (i in seq_along(files)) {
  bmrs[[i]] = readRDS(files[i])
}
bmr = mlr::mergeBenchmarkResults(bmrs)


# Get models
glmnet = readRDS(files[1])
glmnet.df = as.data.frame(glmnet)
rf = readRDS(files[2])
rf.df = as.data.frame(rf)
svm = readRDS(files[3])
svm.df = as.data.frame(svm)

# Select the best
# In glmnet
glmnet.m = mlr::getBMRModels(glmnet)
best.glmnet = glmnet.m[[1]][[1]][[which.max(rf.df$acc)]]

# In RF
rf.m = mlr::getBMRModels(rf)
best.rf = rf.m[[1]][[1]][[which.max(rf.df$acc)]]

# In SVM
svm.m = mlr::getBMRModels(svm)
best.svm = svm.m[[1]][[1]][[which.max(svm.df$acc)]]


# Validation
validation = read.csv('../../data/validation.csv')
validation = mlr::makeClassifTask(data = validation, target = 'target')
predict(best.glmnet, validation)$data
predict(best.rf, validation)$data
predict(best.svm, validation)$data




# PCA 
train = read.csv('../../data/train.csv')
train$target = 'train'

validation = read.csv('../../data/validation.csv')
validation$target = 'validation'

names(train) == names(validation)
mat = rbind.data.frame(train, validation)
meta = data.frame(
  id = rownames(mat),
  cohort = mat$target,
  row.names = rownames(mat)
)
mat = subset(mat, select = -c(target))
mat = t(mat)

require(PCAtools)
fit = pca(mat = mat, metadata = meta)
colnames(mat)
rownames(meta)
pairsplot(fit, colby = 'cohort')
biplot(fit, colby = 'cohort', lab = NULL)
