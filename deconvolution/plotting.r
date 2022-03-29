require(ggpubr)
load('~/projects/redes-tf/data/expression/expression_Raul.RData')

# Load train data
# ===
train = readRDS('~/projects/redes-tf/deconvolution/results/cibsersort_train_deconvolution.rds')
train.clin = clin.DAT1
train = as.data.frame(train)
train$target = train.clin$Response

train$target[which(train$target == 'moderate_responder')] = 'responder'

train = train[, -c(23, 24, 25)]
# train = train[which(train$target == 'responder' | train$target == 'non_responder'),]



# Load validation data
# ============
validation = readRDS('~/projects/redes-tf/deconvolution/results/cibsersort_validation_deconvolution.rds')
validation.clin = clin.DAT2
validation = as.data.frame(validation)
validation$target = validation.clin$Response

validation = validation[, -c(23, 24, 25)]
validation = validation[which(validation$target == 'responder' | validation$target == 'non_responder'),]

rm(list = setdiff(ls(), c('train','validation')))

# Plotting train
# ====
tr = reshape2::melt(train)
ggboxplot(tr, x = 'target', y = 'value', fill = 'target', facet.by = 'variable', title = 'Train') + stat_compare_means(label.y.npc = 0.6)


# Plotting validation
# ====
val = reshape2::melt(validation)
ggboxplot(val, x = 'target', y = 'value', fill = 'target', facet.by = 'variable', title = 'Validation') + stat_compare_means(label.y.npc = 0.6)
