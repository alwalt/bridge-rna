"""Shared preprocessing + embedding generation for ARCHS4-trained ExpressionPerformer.

Use this package to turn TCGA, GTEx, or OSDR expression matrices into 512-D
sample embeddings using the exact same gene vocabulary alignment and
normalization the model was trained with (log1p(TPM) by default).
"""

from .vocab import load_canonical_genes, norm_gene
from .model import load_expression_performer
from .transform import align_to_vocab, apply_preprocessing
from .encode import encode_matrix

__all__ = [
    "load_canonical_genes",
    "norm_gene",
    "load_expression_performer",
    "align_to_vocab",
    "apply_preprocessing",
    "encode_matrix",
]
