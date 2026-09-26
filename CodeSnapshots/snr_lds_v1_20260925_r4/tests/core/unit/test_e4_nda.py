import torch

from balds.attribution.nda import extract_patches, patch_match_scores, nda_scores


def _direct(query, train, *, scale, std, patch, mask_value, topk=None):
    q = extract_patches(query[None], patch, mask_value=mask_value)[0]
    y = extract_patches(train * scale, patch, mask_value=mask_value)
    out = torch.zeros(len(train), dtype=torch.float64)
    for qp in q:
        logits = []
        for image in y:
            logits.extend([-(qp.double() - location.double()).square().sum() / (2 * std**2)
                           for location in image])
        weights = torch.softmax(torch.stack(logits), dim=0).view(len(train), -1)
        if topk is not None:
            weights = weights.topk(topk, dim=1).values
        out += weights.sum(dim=1)
    return out.float()


def test_dense_chunked_and_direct_nda_match_with_global_denominator():
    train = torch.tensor([
        [[[0.0, 0.2], [0.4, 0.6]]],
        [[[0.6, 0.4], [0.2, 0.0]]],
        [[[0.1, 0.3], [0.5, 0.7]]],
    ])
    query = torch.tensor([[[0.2, 0.1], [0.5, 0.4]]])
    kwargs = dict(signal_scale=0.8, noise_std=0.6, patch_size=3, mask_value=9.0)
    direct = _direct(query, train, scale=0.8, std=0.6, patch=3, mask_value=9.0)
    dense = patch_match_scores(query, train, **kwargs)
    chunked = patch_match_scores(query, train, train_chunk=1, query_patch_chunk=1, **kwargs)
    # The vectorised implementation forms float32 distances before the stable
    # float64 reduction, matching the official convolution path; the scalar
    # enumerator forms distances in float64.
    assert torch.allclose(dense, direct, rtol=2e-4, atol=2e-4)
    assert torch.allclose(chunked, dense, rtol=2e-6, atol=2e-6)


def test_spatial_topk_and_extreme_logits_stay_finite():
    train = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]],
                          [[[1.0, 1.0], [1.0, 1.0]]]])
    query = torch.zeros(1, 2, 2)
    kwargs = dict(signal_scale=1.0, noise_std=1e-3, patch_size=1, mask_value=1e3,
                  spatial_topk=1)
    direct = _direct(query, train, scale=1.0, std=1e-3, patch=1,
                     mask_value=1e3, topk=1)
    actual = patch_match_scores(query, train, train_chunk=1, **kwargs)
    assert torch.allclose(actual, direct, rtol=1e-6, atol=1e-6)
    assert torch.isfinite(actual).all()
    # Equal training images remain exact natural ties (no jitter).
    tied = patch_match_scores(query, train[:1].repeat(2, 1, 1, 1), **kwargs)
    assert tied[0] == tied[1]


def test_per_time_recipes_manual_aggregation_and_original_rng_positions():
    g = torch.Generator().manual_seed(32)
    train, queries = torch.randn(3, 1, 3, 3, generator=g), torch.randn(2, 1, 3, 3, generator=g)
    recipes = [dict(patch_size=1, second_patch_size=2, second_projected_size=1, two_scale_alpha=0.2),
               dict(patch_size=3, second_patch_size=4, second_projected_size=2, two_scale_alpha=0.8)]
    bars, ids, positions = [0.8, 0.3], [12, 19], [1, 4]
    kwargs = dict(alpha_bars=bars, patch_size=1, variant="two_scale", seed=7,
                  timestep_recipes=recipes, query_ids=ids, timestep_indices=positions,
                  mask_value=9.0)
    full = nda_scores(train, queries, **kwargs)
    manual = torch.zeros_like(full)
    for qi, clean in enumerate(queries):
        for ti, bar in enumerate(bars):
            gen = torch.Generator().manual_seed(((7*1000003 + ids[qi])*1000033 + positions[ti]) % (2**63-1))
            noisy = bar**0.5*clean + (1-bar)**0.5*torch.randn(clean.shape, generator=gen)
            recipe = recipes[ti]
            common = dict(signal_scale=bar**0.5, noise_std=(1-bar)**0.5, mask_value=9.0)
            first = patch_match_scores(noisy, train, patch_size=recipe["patch_size"], **common)
            second = patch_match_scores(noisy, train, patch_size=recipe["second_patch_size"],
                                         projected_size=recipe["second_projected_size"], **common)
            a = recipe["two_scale_alpha"]
            manual[:, qi] += (a*first + (1-a)*second)/2
    assert torch.allclose(full, manual, atol=1e-6, rtol=2e-6)
    chunks = []
    for qi in range(2):
        times = [nda_scores(train, queries[qi:qi+1], **{
            **kwargs, "alpha_bars": [bars[ti]], "query_ids": [ids[qi]],
            "timestep_recipes": [recipes[ti]], "timestep_indices": [positions[ti]],
            "train_chunk": 1, "query_patch_chunk": 2}) for ti in range(2)]
        chunks.append(sum(times)/2)
    assert torch.allclose(full, torch.cat(chunks, dim=1), atol=2e-5, rtol=2e-5)


def test_scalar_recipe_compatibility_and_bad_projection():
    import pytest
    train = torch.zeros(2, 1, 2, 2)
    kwargs = dict(alpha_bars=[0.8, 0.3], patch_size=1, seed=2)
    old = nda_scores(train, train[:1], **kwargs)
    new = nda_scores(train, train[:1], timestep_recipes=[{}, {}], **kwargs)
    assert torch.equal(old, new)
    with pytest.raises(ValueError):
        nda_scores(train, train[:1], **{**kwargs, "variant": "two_scale"},
                   second_patch_size=21, second_projected_size=10)
