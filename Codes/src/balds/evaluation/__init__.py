"""Array-based evaluators. Model-dependent loss calculation is imported lazily."""
from .lds import compute_lds, lds_of_scores, predicted_influence, spearman
from .stats import bootstrap_ci, holm_bonferroni, paired_wilcoxon

def __getattr__(name):
    if name == "compute_query_losses":
        from .ground_truth import compute_query_losses
        return compute_query_losses
    raise AttributeError(name)
