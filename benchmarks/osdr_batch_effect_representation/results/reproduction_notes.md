# Reproduction notes

- Sanders et al. report 112 liver samples: 57 FLT and 55 GC. This pipeline matches those totals exactly.
- The seven local GeneLab unnormalized count tables were restricted to FLT and respective GC samples.
- Counts were inner-joined on version-stripped Ensembl IDs, retaining 51,929 genes.
- The paper states DESeq2 v1.30.1 median-of-ratios normalization, followed by R `prcomp()` v4.1.0. Local R does not have DESeq2, so its default `ratio` size-factor algorithm was reproduced directly: genes containing any zero were excluded from geometric-mean estimation; each sample size factor is the median ratio to gene geometric means. 13,609 genes contributed to size-factor estimation.
- Before PCA, we apply `log2(DESeq2-normalized counts + 1)`, then centered, unscaled PCA matching `prcomp(center=TRUE, scale.=FALSE)`. The article does not explicitly state the log step, but it is required to reproduce Figure 1's reported variance: our PC1/PC2 are 24.69%/12.73%, versus 25.2%/12.75% in the paper. This inferred step is disclosed rather than presented as directly documented. No batch correction is applied.
- Mission assignments for multi-mission OSD-168 and OSD-245 use sample-name mission/return annotations consistent with Table 1: OSD-168 RR3 versus RR1 NASA; OSD-245 ISS-T as RR3 versus LAR as RR6.
- The supplementary payload filenames are declared in the publisher XML, but direct publisher downloads returned 404 during this run. The local OSDR sample metadata plus paper Table 1 were therefore used; this limitation is explicit.
- BridgeRNA inputs are independently generated as mouse-annotation TPM, mapped to the frozen 15,165 one-to-one vocabulary, then natural log1p(TPM). The exact same 112 samples are used, but this representation necessarily uses the model's native preprocessing rather than DESeq2-normalized full counts.
- BridgeRNA is frozen; sample embeddings are 512-D mean-pooled contextual representations. No correction, alignment, fine-tuning, or target-label use occurs.
