#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(edgeR); library(limma); library(jsonlite)})

args <- commandArgs(trailingOnly=TRUE)
here <- normalizePath(ifelse(length(args), args[[1]], "benchmarks/drug_discovery"))
prep <- file.path(here,"work","prepared"); manifests <- file.path(here,"results","manifests")
out <- file.path(here,"results","differential_expression"); dir.create(out,recursive=TRUE,showWarnings=FALSE)
members <- read.csv(file.path(manifests,"contrast_members.csv"),check.names=FALSE,stringsAsFactors=FALSE)
sensitivity <- read.csv(file.path(manifests,"GSE211204_unseen_subject_sensitivity.csv"),check.names=FALSE,stringsAsFactors=FALSE)
sensitivity$contrast_id <- "GSE211204_ULLS_unseen"
members <- rbind(members,sensitivity[,names(members)])
samples <- read.csv(file.path(manifests,"sample_manifest.csv"),check.names=FALSE,stringsAsFactors=FALSE)
canonical <- read.csv("/home/walt/bridge-rna/data/ensembl/canonical_genes.csv",stringsAsFactors=FALSE)$gene_symbol

run_one <- function(cid) {
  cm <- members[members$contrast_id==cid,]; gse <- unique(cm$dataset); stopifnot(length(gse)==1)
  sm <- merge(cm,samples[samples$dataset==gse,c("sample_id","matrix_col")],by="sample_id",all.x=TRUE,sort=FALSE)
  sm <- sm[match(cm$sample_id,sm$sample_id),]; condition <- factor(sm$role,levels=c("control","case"))
  if (gse=="GSE189524") {
    x <- read.csv(gzfile(file.path(prep,paste0(gse,"_tpm.csv.gz"))),row.names=1,check.names=FALSE)
    x <- as.matrix(x[,sm$sample_id,drop=FALSE]); x <- x[rowSums(x)>0,,drop=FALSE]; x <- log2(x+0.5)
    design <- model.matrix(~condition); fit <- eBayes(lmFit(x,design),trend=TRUE,robust=TRUE)
    tt <- topTable(fit,coef="conditioncase",number=Inf,sort.by="none")
    ans <- data.frame(gene=rownames(tt),logFC=tt$logFC,t=tt$t,pvalue=tt$P.Value,fdr=tt$adj.P.Val,eligible=TRUE)
  } else {
    counts <- read.csv(gzfile(file.path(prep,paste0(gse,"_counts.csv.gz"))),row.names=1,check.names=FALSE)
    counts <- round(as.matrix(counts[,sm$matrix_col,drop=FALSE])); storage.mode(counts)<-"integer"
    covars <- data.frame(condition=condition)
    if (cid %in% c("GSE297090_gamma","GSE297090_proton")) {covars$donor<-factor(sm$donor); covars$state<-factor(sm$state); design<-model.matrix(~donor+state+condition,covars)
    } else if (cid=="GSE184119_10Gy") {covars$cell_type<-factor(sm$cell_type); design<-model.matrix(~cell_type+condition,covars)
    } else if (cid %in% c("GSE211204_ULLS","GSE211204_ULLS_unseen","GSE113165_bedrest")) {covars$donor<-factor(sm$donor); design<-model.matrix(~donor+condition,covars)
    } else if (cid=="GSE276529_GIOP") {covars$age<-as.numeric(sm$age_numeric); design<-model.matrix(~age+condition,covars)
    } else design<-model.matrix(~condition,covars)
    y<-DGEList(counts); keep<-filterByExpr(y,design=design); y<-y[keep,,keep.lib.sizes=FALSE]; y<-calcNormFactors(y)
    y<-estimateDisp(y,design,robust=TRUE); fit<-glmQLFit(y,design,robust=TRUE); qlf<-glmQLFTest(fit,coef="conditioncase")
    tt<-topTags(qlf,n=Inf,sort.by="none")$table
    ans<-data.frame(gene=rownames(tt),logFC=tt$logFC,t=sign(tt$logFC)*sqrt(pmax(tt$F,0)),pvalue=tt$PValue,fdr=p.adjust(tt$PValue,"BH"),eligible=TRUE)
  }
  all <- merge(data.frame(gene=canonical),ans,by="gene",all.x=TRUE,sort=FALSE); all$eligible[is.na(all$eligible)]<-FALSE; all$t[is.na(all$t)]<-0; all$pvalue[is.na(all$pvalue)]<-1; all$fdr[is.na(all$fdr)]<-1
  all <- all[match(canonical,all$gene),]; con<-gzfile(file.path(out,paste0(cid,".csv.gz")),"w"); write.csv(all,con,row.names=FALSE); close(con)
  data.frame(contrast_id=cid,dataset=gse,n_case=sum(condition=="case"),n_control=sum(condition=="control"),eligible_genes=sum(all$eligible),design=paste(colnames(design),collapse=";"))
}

summary <- do.call(rbind,lapply(unique(members$contrast_id),function(cid){cat(format(Sys.time()),cid,"\n"); flush.console(); run_one(cid)}))
write.csv(summary,file.path(out,"contrast_summary.csv"),row.names=FALSE)
write(toJSON(list(method="edgeR quasi-likelihood for count matrices; limma-trend on donor-collapsed log2(TPM+0.5) for GSE189524",edgeR=as.character(packageVersion("edgeR")),limma=as.character(packageVersion("limma")),contrasts=summary),pretty=TRUE,auto_unbox=TRUE),file.path(out,"provenance.json"))
