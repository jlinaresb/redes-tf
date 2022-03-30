# Pathway enrichment
require(HTSanalyzeR2)
library(org.Hs.eg.db)
library(KEGGREST)
library(igraph)

## prepare input for analysis
gsea = readRDS('~/projects/redes-tf/results/methylation/gsea.rds')
phenotype <- as.vector(gsea$score)
names(phenotype) <- rownames(gsea)

## specify the gene sets type you want to analyze
GO_MF <- GOGeneSets(species="Hs", ontologies=c("MF"))
PW_KEGG <- KeggGeneSets(species="Hs")
MSig <- MSigDBGeneSets(species = "Hs", collection = 'H')
ListGSC <- list(GO_MF=GO_MF, PW_KEGG=PW_KEGG, MSig=MSig)

## iniate a *GSCA* object
gsca <- GSCA(listOfGeneSetCollections=ListGSC, 
             geneList=phenotype)

## preprocess
gsca1 <- preprocess(gsca, species="Hs", initialIDs="SYMBOL",
                    keepMultipleMappings=TRUE, duplicateRemoverMethod="max",
                    orderAbsValue=FALSE)

## analysis
if (requireNamespace("doParallel", quietly=TRUE)) {
  doParallel::registerDoParallel(cores=4)
}  ## support parallel calculation using multiple cores
gsca2 <- analyze(gsca1, 
                 para=list(pValueCutoff=0.05, pAdjustMethod="BH",
                           nPermutations=100, minGeneSetSize=180,
                           exponent=1), 
                 doGSOA = FALSE)

## append gene sets terms
gsca3 <- appendGSTerms(gsca2,
                       goGSCs=c("GO_MF"),
                       keggGSCs=c("PW_KEGG"),
                       msigdbGSCs = c("MSig"))

## draw GSEA plot for a specific gene set
viewEnrichMap(gsca3,
              resultName = 'GSEA.results',
              gscs = 'GO_MF',
              allSig = T,
              gsNameType = 'term')

viewEnrichMap(gsca3,
              resultName = 'GSEA.results',
              gscs = 'PW_KEGG',
              allSig = T,
              gsNameType = 'term')

viewEnrichMap(gsca3,
              resultName = 'GSEA.results',
              gscs = 'MSig',
              allSig = T,
              gsNameType = 'term')
