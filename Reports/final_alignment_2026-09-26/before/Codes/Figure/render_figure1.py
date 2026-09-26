"""Render compact Figure 1 from filed data; no scoring or LDS recomputation."""
from pathlib import Path
import os
import json
from statistics import mean
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2]
DATA=Path(os.environ.get("BALDS_DATA_ROOT", str(ROOT/"_Data"))).expanduser().resolve()
DEST=ROOT/"Paper/ICLR/Figures/section31"
J=DATA/'results/paper_figures/legacy'
METHODS=[('fmas_raw','FMAS','#668394'),('ekfac_if','EK-FAC IF','#B1947B'),('dtrak_T100','D-TRAK','#819985'),('das_T100','DAS (linear)','#9B879F')]
SEEDS=[42,123,456]
head=json.loads((J/'fig2_head_profile.json').read_text())
power=json.loads((J/'fig1_power_c2.json').read_text())
settings=json.loads((J/'fig_score_magnitude.json').read_text())
paired=json.loads((DEST/'square_heads.json').read_text())
plt.rcParams.update({'font.family':'serif','font.size':6.5,'axes.labelsize':6.5,'axes.titlesize':7,'legend.fontsize':6.5,'axes.linewidth':.5,'lines.linewidth':1.05,'pdf.fonttype':42})
fig,axes=plt.subplots(1,4,figsize=(7.2,1.65))
fig.subplots_adjust(left=.065,right=.985,bottom=.26,top=.86,wspace=.52)
curves={}
for m,label,color in METHODS:
    densities=[]
    for seed in SEEDS:
        raw=np.load(DATA/f'scores/{m}/cifar2_5k/seed_{seed}/scores.npy').astype(float)
        x=raw/np.sqrt(np.mean(raw**2,axis=0,keepdims=True))
        edges=np.linspace(-settings['plot_range_rms'],settings['plot_range_rms'],8002)
        width=edges[1]-edges[0]; counts,_=np.histogram(x.ravel(),bins=edges)
        densities.append(gaussian_filter1d(counts.astype(float),settings['bandwidth_rms']/width,mode='constant')/(x.size*width))
    grid=(edges[1:]+edges[:-1])/2;density=np.mean(densities,axis=0);valid=density>1e-4
    axes[0].plot(grid[valid],density[valid],color=color)
    h=np.asarray(head[f'{m}|gen|abs']).mean(axis=0)
    power_y=np.asarray([power[f'{m}|gen|{seed}|signed']['lds'][:21] for seed in SEEDS]).mean(axis=0)
    axes[1].plot(np.arange(21)/5,100*power_y,color=color)
    rs=[r for r in paired['records'] if r['method']==m and r['track']=='gen']
    native=[np.mean([r['native_head']['mean'] for r in rs if r['head_pct']==pct]) for pct in paired['head_grid']]
    axes[2].plot(paired['head_grid'],100*np.array(native),color=color,marker='o',ms=1.8)
    for name,style in [('original_same_head','-'),('square_same_head','--')]:
        y=[100*np.mean([r[name]['mean'] for r in rs if r['head_pct']==pct]) for pct in paired['head_grid']]
        axes[3].plot(paired['head_grid'],y,color=color,ls=style)
    curves[m]={'magnitude_head_lds':h.tolist(),'signed_power_lds':power_y.tolist(),'native_head_lds':native}
axes[0].plot(grid,np.exp(-grid**2/2)/np.sqrt(2*np.pi),color='#B8B8B8',ls='--',lw=.7)
axes[0].set(xlim=(-8,8),ylim=(1e-4,3),yscale='log',xlabel='Score / query RMS',ylabel='Density',title='(a) Score distribution')
axes[0].set_xticks([-8,0,8]);axes[0].set_yticks([1e-4,1e-2,1])
axes[1].set(xlim=(0,4),ylim=(8,55),xlabel='Signed power p',ylabel='LDS',title='(b) Signed reweighting')
axes[1].set_xticks([0,1,2,3,4]);axes[1].set_yticks([10,30,50]);axes[1].axvline(1,color='#BBBBBB',ls=':',lw=.6)
axes[2].set(xlim=(5,100),xlabel='Support head (%)',ylabel='LDS',title='(c) Native aggregation')
axes[2].set_xticks([5,50,100]);axes[2].set_yticks([40,45,50])
axes[3].set(xlim=(5,100),xscale='log',xlabel='Support head (%)',ylabel='LDS',title='(d) Matched-head squaring')
axes[3].set_xticks([5,10,20,50,100]);axes[3].xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'));axes[3].minorticks_off();axes[3].set_yticks([40,45,50])
axes[3].legend(handles=[Line2D([],[],color='#555555',ls='-',label='Original'),Line2D([],[],color='#555555',ls='--',label='Squared')],loc='lower left',fontsize=4.8,frameon=False,handlelength=1.8,labelspacing=.15,borderpad=.1)
for ax in axes:
    ax.spines[['top','right']].set_visible(False);ax.grid(color='#EBEBEB',lw=.4);ax.set_axisbelow(True);ax.tick_params(length=2,pad=1.5)
fig.savefig(DEST/'curve-row.pdf')
fig.savefig(DEST/'curve-row.png',dpi=240)
plt.close(fig)
# Two short, native booktabs tables; values retain the original aggregation.
selected=[5,20,50,100];grid_pct=[5,10,20,30,40,50,60,70,80,90,100]
tex=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}',r'\toprule',r'Head & 5\% & 20\% & 50\% & 100\% \\',r'\midrule']
for m,label,_ in METHODS:
    vals=[f"{100*curves[m]['magnitude_head_lds'][grid_pct.index(pct)]:.2f}" for pct in selected]
    tex.append(' & '.join([label.replace(' (linear)','')]+vals)+r' \\')
tex += [r'\bottomrule',r'\end{tabular*}']
(DEST/'magnitude-head-table.tex').write_text('\n'.join(tex)+'\n')
r16=Path(os.environ.get('BALDS_R16_ROOT', str(DATA/'results/narrative_wave_20260921/analysis/m2a_r16_20260921')))
stats=json.loads((r16/'fmas_raw/gen/statistics.json').read_text())
rows=[]
for band in range(5):
    entries=[r for r in stats['bands'] if r['band']==band]
    rows.append([mean(r['rep_median'] for r in entries),100*mean(r['lds_mean'] for r in entries)])
tex=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}',r'\toprule',r'Band (\%) & 0--1 & 1--5 & 5--20 & 20--50 & 50--100 \\',r'\midrule']
for j,label in enumerate(['Rep','LDS']):tex.append(' & '.join([label]+[f'{r[j]:.2f}' for r in rows])+r' \\')
tex += [r'\bottomrule',r'\end{tabular*}']
(DEST/'repeatability.tex').write_text('\n'.join(tex)+'\n')
(DEST/'figure1_sources.json').write_text(json.dumps({'source':'existing CIFAR-2 generation scores and results/paper_figures/legacy','seeds':SEEDS,'curves':curves,'magnitude_table_head_pct':selected,'r16_fmas_rows_rep_lds_x100':rows,'paired_source':'square_heads.json','panel_map':{'a':'density','b':'signed_power','c':'native_head','d':'matched_square','e':'magnitude_head_table','f':'fmas_repeatability_table'},'lds_display_scale':100},indent=2))
print('Rendered compact curve row and two booktabs tables.')
# Expanded validation track, rendered from the same recomputation.
fig,axs=plt.subplots(1,2,figsize=(7,2.25))
for m,label,color in METHODS:
    rs=[r for r in paired['records'] if r['method']==m and r['track']=='val']
    x=paired['head_grid']
    for ax,name,style in [(axs[0],'native_head','-'),(axs[1],'original_same_head','-'),(axs[1],'square_same_head','--')]:
        y=[100*np.mean([r[name]['mean'] for r in rs if r['head_pct']==p]) for p in x]
        ax.plot(x,y,color=color,ls=style,label=label.replace(' (linear)','') if name!='square_same_head' else None)
for ax,title in zip(axs,['Native-score aggregation','Original vs. square on the same head']):
    ax.set(title=title,xlabel='Support head (%)',ylabel='LDS (×100)',xlim=(5,100))
    ax.set_xticks([5,50,100]);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False)
fig.tight_layout(rect=(0,0,1,.86));fig.savefig(DEST/'square-heads-validation.pdf',bbox_inches='tight');plt.close(fig)
tex=[r'\begin{tabular}{llrrrr}',r'\toprule',r' & & \multicolumn{2}{c}{Full aggregation} & \multicolumn{2}{c}{Same support head (5\%)} \\',r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',r'Method & Queries & Original & Squared & Original & Squared \\',r'\midrule']
for m,label,_ in METHODS:
    for track in ['gen','val']:
        vals=[]
        for pct in [100,5]:
            for name in ['original_same_head','square_same_head']:
                z=[100*r[name]['mean'] for r in paired['records'] if r['method']==m and r['track']==track and r['head_pct']==pct]
                vals.append(f'${np.mean(z):.2f}\\pm{np.std(z,ddof=1):.2f}$')
        tex.append(' & '.join([label.replace(' (linear)',''),'Gen' if track=='gen' else 'Val']+vals)+r' \\')
tex.extend([r'\bottomrule',r'\end{tabular}'])
(DEST/'square-heads-table.tex').write_text('\n'.join(tex)+'\n')
