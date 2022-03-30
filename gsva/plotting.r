setwd('~/projects/redes-tf/results/gsva/')

gsva = readRDS('msigDB_tmod_ssgsea.rds')

# require(RcmdrMisc)
# 
# cor.adj = rcorr.adjust(gsva, type = 'spearman', use = 'complete.obs')
# r = as.data.frame(cor.adj$R$r)
# p = as.data.frame(cor.adj$R$P)
# 
# res = data.frame(
#   cor = r[,which(colnames(r) == 'fcbf')],
#   p = p[,which(colnames(p) == 'fcbf')],
#   row.names = rownames(r)
# )
# 
# head(as.matrix(res))
# res$log10padj = -log10(res$p)
# head(res)


res = list()
for (i in c(1:604)) {
  c = cor.test(gsva[,i], gsva[, 605], method = 'spearman')
  res[[i]] = data.frame(
    id = names(gsva)[i],
    pvalue = c$p.value,
    spearman = c$estimate
  )
}
require(data.table)
res = as.data.frame(rbindlist(res))
rownames(res) = 1:nrow(res)

data("tmod")

modules = tmod$MODULES
res = res[match(intersect(modules$ID, res$id), res$id),]
modules = modules[match(intersect(modules$ID, res$id), modules$ID),]

stopifnot(res$modules == modules$ID)

res = cbind.data.frame(res, 
                 title = modules$Title,
                 category = modules$Category,
                 annotated = modules$Annotated,
                 sourceID = modules$SourceID)

# dc = res[which(res$annotated == 'Yes' & res$sourceID == 'DC'), ]
li = res[which(res$annotated == 'Yes' & res$sourceID == 'LI'), ]

hist(li$spearman, 10)

cats = names(table(li$category))
# cat = 'TF targets'
alpha = 0.001
thres = 0.5
# Plotting
# ===
p = list()
for (i in seq_along(cats)) {
  li.im = li[which(li$category == cats[i]),]
  li.im$log10pval = -log10(li.im$pvalue)
  li.im$color = as.factor(ifelse(li.im$pvalue < alpha, 'pvalue < 0.001', 'pvalue > 0.001'))
  p[[i]] = ggscatter(li.im,
                x="spearman", y="log10pval",
                color = 'color',
                palette = c('#440154FF', '#22A884FF'),
                label = "title",
                label.select = li.im$title[which(li.im$pvalue<alpha & abs(li.im$spearman) > thres)],
                repel = T,
                label.rectangle = T,
                show.legend.text = F,
                title = cats[i]) + 
    theme(legend.position = 'none', title = element_text(vjust = 0.5)) +
    geom_vline(xintercept = c(-thres, thres), linetype = 'dotted', color = '#3B528BFF', size = 0.5)
  
}
ggarrange(p[[1]], p[[2]], p[[3]], p[[4]], p[[5]], p[[6]], ncol = 3, nrow = 2)


# 
data(tmod)
tmod = tmod$MODULES2GENES
tmod$fcbf = fcbf

save(tmod, data, file = '~/projects/redes-tf/results/gsva/tmod_toRun.RData')
load('~/projects/redes-tf/results/gsva/tmod_toRun.RData')
fcbf = tmod$FCBF



# Scatter
# ===
toPlot = data.frame(
  fcbf = gsva$FCBF,
  Tcelldif = gsva$`T cell differentiation`,
  nt.met = gsva$`nucleotide metabolism`,
  Tcell = gsva$`T cell`,
  TcellactIII = gsva$`T cell activation (III)`,
  
  CD1 = gsva$`CD1 and other DC receptors`
)

p1 = ggscatter(data = toPlot, x = 'fcbf', y = 'Tcelldif', add = 'reg.line', conf.int = T,
          add.params = list(color = viridis(3)[1], fill = 'lightgray'), title = 'T cell differentiation') +
  stat_cor(method = 'spearman') + xlab('FCBF signature') + ylab('ssGSEA score')

p2 = ggscatter(data = toPlot, x = 'fcbf', y = 'CD1', add = 'reg.line', conf.int = T,
             add.params = list(color = viridis(3)[1], fill = 'lightgray'), title = 'CD1 and other DC receptors') +
  stat_cor(method = 'spearman') + xlab('FCBF signature') + ylab('ssGSEA score')

p3 = ggscatter(data = toPlot, x = 'fcbf', y = 'nt.met', add = 'reg.line', conf.int = T,
          add.params = list(color = viridis(3)[1], fill = 'lightgray'), title = 'Nucleotide metabolism') +
  stat_cor(method = 'spearman') + xlab('FCBF signature') + ylab('ssGSEA score')

p4 = ggscatter(data = toPlot, x = 'fcbf', y = 'Tcell', add = 'reg.line', conf.int = T,
          add.params = list(color = viridis(3)[1], fill = 'lightgray'), title = 'T cell') +
  stat_cor(method = 'spearman') + xlab('FCBF signature') + ylab('ssGSEA score')


ggarrange(p1, p2, p3, p4, ncol = 4, nrow = 1)



hist(res$FCBF, 10)


