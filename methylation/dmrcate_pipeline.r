require(DMRcate)

# Load data
met = load('~/projects/redes-tf/data/methylation/methylation_jose.RData')

# Differential methylation
type <- factor(PREC.res$scaled)
design <- model.matrix(~type)
myannotation <- cpg.annotate(datatype = "array",
                             object = as.matrix(B.precisesads),
                             what = 'Beta',
                             arraytype = "450K",
                             analysis.type = "differential",
                             design = design,
                             coef = 2,
                             fdr = 0.1)

dmrcoutput = dmrcate(myannotation, lambda = 1000, C = 2)


# Extract ranges
results = extractRanges(dmrcoutput, genome = 'hg19')

# Plot results
groups = c(responder = 'magenta', non_responder = 'forestgreen')
cols = groups[as.character(type)]
DMR.plot(ranges = results, 
        dmr=2,
        CpGs=as.matrix(B.precisesads),
        what="Beta",
        arraytype = "450K",
        phen.col=cols, 
        genome="hg38")


# Functional analysis
require(missMethyl)
enrichment = goregion(results[1:100], all.cpg = rownames(B.precisesads),
                      collection = 'GO', array.type = '450K')
enrichment = enrichment[order(enrichment$P.DE), ]
head(enrichment, 20)

# Plot heatmap
require(ComplexHeatmap)
Heatmap(B.precisesads)