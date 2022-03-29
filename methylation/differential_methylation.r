require(ELMER)

# Load data
# ===
met = load('~/projects/redes-tf/data/methylation/methylation_jose.RData')
exp = load('~/projects/redes-tf/data/expression/expression_Raul.RData')

names(B.precisesads) = make.names(names(B.precisesads))
names(exp.PREC) = make.names(names(exp.PREC))

# Match patients
# ===
pats = intersect(colnames(B.precisesads), colnames(exp.PREC))
met = B.precisesads[, pats]
exp = exp.PREC[, pats]

# Change exp to ENSEMBL
# ===
require(org.Hs.eg.db)
genes = mapIds(org.Hs.eg.db,
       keys = rownames(exp), column = 'ENSEMBL', keytype = 'ALIAS')
rownames(exp) = unname(genes)
length(unique(genes))
xx = genes[complete.cases(genes)]
length(unique(xx))



meta = data.frame(
  primary = make.names(rownames(PREC.res)),
  group = PREC.res$scaled
)
meta = meta[match(pats, meta$primary), ]




# Create MAE object
# ===
mae = createMAE(
        exp = exp,
        met = met,
        colData = meta,
        met.platform = '450K',
        genome = 'hg38')


sig.diff = get.diff.meth(data = met,
                         group.col = '',
                         group1 = '',
                         group2 = '',
                         minSubgroupFrac = 0.2,
                         sig.dif = 0.3,
                         diff.dir = "hypo",
                         cores = 1,
                         dir.out = "result",
                         pvalue = 0.01)


PREC.res$scaled
table(PREC.res$scaled)
