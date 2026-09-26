"""Render accepted a4 tables to stdout as JSON; no experiment is executed."""
import json
from pathlib import Path

SOURCE = Path('/path/to/CFA/_Data/results/snr_lds_20260925/a4/tables/summary.json')
METHODS = [
    ('pixel_dot', 'Raw pixel (dot product)'), ('pixel_cos', 'Raw pixel (cosine)'),
    ('clip_dot', 'CLIP similarity (dot product)'), ('clip_cos', 'CLIP similarity (cosine)'),
    ('grad_dot_T100', 'Gradient (dot product)'), ('grad_cos_T100', 'Gradient (cosine)'),
    ('tracincp_T100', 'TracInCP'), ('gas_T100', 'GAS'),
    ('journey_trak_T100', 'Journey-TRAK'), ('relative_if_T100', 'Relative IF'),
    ('renorm_if_T100', 'Renormalized IF'), ('trak_T100', 'TRAK'),
    ('dtrak_T100', 'D-TRAK'), ('das_native_sq', 'DAS'),
    ('ekfac_if', 'EK-FAC IF'), ('fmas_raw', 'FMAS'),
]
data = json.loads(SOURCE.read_text())
records = {(r['dataset'], r['track'], r['method']): r for r in data['records']}
end = r' \\'

def cell(ds, track, method, sd):
    if method == 'journey_trak_T100' and track == 'val':
        return r'\textnormal{n/a}'
    r = records.get((ds, track, method))
    if r is None:
        return r'\textemdash'
    if r['snr_lds'] is None:
        return r'\textnormal{NA}'
    mean = r['snr_lds'] * 100
    if abs(mean) < .005:
        mean = 0.0
    value = f'{mean:.2f}'
    if sd and r['snr_lds_std'] is not None:
        value += r'\pm' + f"{100*r['snr_lds_std']:.2f}"
    return '$' + value + '$'

def table(datasets, titles, sd, caption, labels):
    lines = [f'% Generated from {SOURCE}; rule={data["rule"]}',
             r'\begin{table}[t]', r'\centering\small', r'\setlength{\tabcolsep}{5pt}',
             r'\begin{tabular}{lrrrr}', r'\toprule',
             r' & \multicolumn{2}{c}{'+titles[0]+r'} & \multicolumn{2}{c}{'+titles[1]+'}'+end,
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
             'Method & Validation & Generation & Validation & Generation'+end, r'\midrule']
    for i, (method, name) in enumerate(METHODS):
        if i in (4, 8, 14):
            lines.append(r'\midrule')
        lines.append(name+' & '+' & '.join(cell(ds,t,method,show_sd) for ds,show_sd in zip(datasets,sd) for t in ('val','gen'))+end)
    lines += [r'\bottomrule', r'\end{tabular}', r'\caption{'+caption+'}']
    lines += [r'\label{'+label+'}' for label in labels]
    return '\n'.join(lines+[r'\end{table}', ''])

out = {}
out['snr_cifar.tex'] = table(
    ['cifar2_5k','cifar2_das'], ['CIFAR-2 FM','CIFAR-2 DDPM'], [True,False],
    r'SNR-LDS (\%) with $\zeta=3$ on CIFAR-2 FM and DDPM. '
    r'FM reports means and sample standard deviations across three model seeds; DDPM uses one model. Each model uses 100 queries per track, with means computed over method-specific valid queries. '
    r'NA denotes no valid noise-scale fits, not a zero score. Valid empty selections give LDS zero. '
    r'Dashes denote five missing DDPM score inputs; n/a denotes the unassessed Journey-TRAK validation track. Coverage is reported in Table~\ref{tab:snr-coverage}.',
    ['tab:exp-snr-cifar','tab:exp-c2'])
out['snr_platforms.tex'] = table(
    ['artbench2_256','cifar10_v2'], ['ArtBench-2','CIFAR-10 FM'], [False,True],
    r'SNR-LDS (\%) with $\zeta=3$ on ArtBench-2 and CIFAR-10 FM. '
    r'ArtBench-2 uses one model; CIFAR-10 FM reports means and sample standard deviations across three model seeds. Each model uses 100 queries per track, with means computed over method-specific valid queries. '
    r'ArtBench-2 EK-FAC IF retains per-track oracle damping selection. '
    r'NA denotes no valid fits; n/a denotes the unassessed Journey-TRAK validation track.',
    ['tab:exp-snr-platforms','tab:exp-platforms'])

datasets = ['cifar2_5k','cifar10_v2','artbench2_256','cifar2_das']
lines = [r'\begin{table}[t]',r'\centering\scriptsize', r'\setlength{\tabcolsep}{3pt}',
         r'\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrrrr}',r'\toprule',
         r' & \multicolumn{2}{c}{C2 FM} & \multicolumn{2}{c}{C10 FM} & \multicolumn{2}{c}{AB2} & \multicolumn{2}{c}{C2 DDPM}'+end,
         'Method & Val & Gen & Val & Gen & Val & Gen & Val & Gen'+end,r'\midrule']
for method, name in METHODS:
    cells=[]
    for ds in datasets:
        for track in ('val','gen'):
            r=records.get((ds,track,method))
            cells.append('n/a' if method=='journey_trak_T100' and track=='val' else
                         r'\textemdash' if r is None else f"{r['n_valid']}/{r['n_total']}")
    lines.append(name+' & '+' & '.join(cells)+end)
lines += [r'\bottomrule',r'\end{tabular}}',
          r'\caption{Valid noise-scale fits / available queries for the new SNR rule. '
          r'CIFAR FM counts pool the three seeds; shared pixel/CLIP validation inputs are counted once. '
          r'Valid empty selections are included in the numerator and assigned zero LDS. '
          r'All pixel/CLIP queries fail the prescribed noise-scale fit; they are not missing input files.}',
          r'\label{tab:snr-coverage}',r'\end{table}','']
out['snr_coverage.tex']='\n'.join(lines)

sens={(r['dataset'],r['track'],r['method'],r['zeta']):r for r in data['sensitivity']}
lines=[r'\begin{table}[t]',r'\centering\scriptsize',r'\setlength{\tabcolsep}{5pt}',
       r'\begin{tabular}{lllrrrr}',r'\toprule',r'Setting & Track & Method & $\zeta=1$ & $2$ & $3$ & $4$'+end,r'\midrule']
for ds,title in zip(datasets,['C2 FM','C10 FM','AB2','C2 DDPM']):
    for track in ('gen','val'):
        for method,name in [('fmas_raw','FMAS'),('ekfac_if','EK-FAC IF'),('dtrak_T100','D-TRAK'),('das_native_sq','DAS')]:
            cells=[f"{sens[ds,track,method,z]['snr_lds']*100:.2f}" for z in (1,2,3,4)]
            lines.append(' & '.join([title,track,name]+cells)+end)
    lines.append(r'\midrule' if ds!=datasets[-1] else r'\bottomrule')
lines += [r'\end{tabular}',r'\caption{SNR-LDS (\%) sensitivity for the four core methods. '
          r'Thresholds share the same fitted noise scale and valid queries within each method. '
          r'FM means average three model seeds; AB2 and DDPM use one. The default remains $\zeta=3$, independent of these results.}',
          r'\label{tab:snr-sensitivity}',r'\end{table}','']
out['snr_sensitivity.tex']='\n'.join(lines)
print(json.dumps(out))
