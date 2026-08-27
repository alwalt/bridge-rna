"""Shared preprocessing + embedding generation for ARCHS4-trained ExpressionPerformer.

Use this package to turn TCGA, GTEx, or OSDR expression matrices into 512-D
sample embeddings using the exact same gene vocabulary alignment and
normalization the model was trained with (log1p(TPM) by default).
"""

from .vocab import load_canonical_genes, norm_gene
from .transform import align_to_vocab, apply_preprocessing
from .encode import encode_matrix
from .cohorts import Archs4Cohort, available_cohorts, load_archs4_cohort


def load_expression_performer(*args, **kwargs):
    """Lazily import the model implementation; cohort access does not require it."""
    from .model import load_expression_performer as _load_expression_performer

    return _load_expression_performer(*args, **kwargs)

__all__ = [
    "load_canonical_genes",
    "norm_gene",
    "load_expression_performer",
    "align_to_vocab",
    "apply_preprocessing",
    "encode_matrix",
    "Archs4Cohort",
    "available_cohorts",
    "load_archs4_cohort",
]
