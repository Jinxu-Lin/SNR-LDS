"""Plain-text result rendering (no heavy deps; the Presentation layer's views)."""
from __future__ import annotations

from typing import Sequence


def render_table(headers: Sequence[str], rows: Sequence[Sequence], *, title: str = "") -> str:
    """Render a left/right-aligned text table."""
    cols = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(str(cols[r][c])) for r in range(len(cols))) for c in range(len(headers))]
    def line(cells):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))
    out = []
    if title:
        out.append(title)
    out.append(line(headers))
    out.append("  ".join("-" * w for w in widths))
    out.extend(line(r) for r in rows)
    return "\n".join(out)


def render_lds(result: dict, *, method: str, dataset: str, query_type: str) -> str:
    """One-line LDS summary with CI for an evaluate result."""
    text = (f"LDS  {dataset}/{query_type}  {method}: "
            f"{result['mean_lds']:.4f}  CI=[{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]  "
            f"(Q={result['n_queries']})")
    if result.get("lds_by_kappa"):
        cells = [f"{key}:{value['mean_lds']:.4f}"
                 for key, value in result["lds_by_kappa"].items()]
        text += "\nLDS@kappa  " + "  ".join(cells)
    return text


def render_snr_lds(result: dict, *, method: str, dataset: str, query_type: str) -> str:
    """Render the new SNR result without relabelling archived BA-LDS values."""
    headers = ["dataset/track", "method", "Full LDS", "SNR-LDS", "valid", "fit failed", "empty"]
    row = [f"{dataset}/{query_type}", method,
           "NA" if result.get("full_lds") is None else f"{result['full_lds']:.4f}",
           "NA" if result.get("snr_lds") is None else f"{result['snr_lds']:.4f}",
           result.get("n_valid", 0), result.get("n_fit_failed", 0), result.get("n_empty", 0)]
    return render_table(headers, [row], title="SNR-LDS (snr_zero_mean_gaussian_v1, strict |x|/sigma > zeta)")


def render_inject(result: dict) -> str:
    """Concept, tier, overall and chance rows for an INJECT result."""
    metrics = result["metrics"]
    headers = ["group", *metrics]

    def values(name, summary):
        return [name, *[
            f"{summary[m]['mean']:.4f} [{summary[m]['ci_lo']:.4f},{summary[m]['ci_hi']:.4f}]"
            for m in metrics]]

    rows = [values(f"concept:{name}", summary)
            for name, summary in sorted(result["per_concept"].items())]
    rows += [values(f"tier:{name}", summary)
             for name, summary in sorted(result["per_tier"].items())]
    rows.append(values("overall", result["overall"]))
    rows.append(["chance", *[f"{float(result['chance'][m]):.4f}" for m in metrics]])
    return render_table(
        headers, rows,
        title=(f"INJECT {result['dataset']}/{result['split']} {result['method']} "
               f"k={result['k']} λ={result['best_lam']}"))
