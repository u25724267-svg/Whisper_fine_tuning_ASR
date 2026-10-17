"""Exploratory paired cluster intervals and figures for the Chapter 4 review.

Run with the project's Python (NumPy), plus Matplotlib for --figures.
No training, inference, or input-artifact mutation is performed.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTRASTS = [('C1','C0'),('C3','C0'),('A2','C0'),('A3','C3'),('A3','A2'),
             ('S2','C0'),('S3','C1'),('S3','S2'),('S4','S2'),('S5','S3')]
CODES = ['C0','C1','C3','A2','A3','S2','S3','S4','S5']
NAMES = ['C0  Random','C1  SortaGrad','C3  SNR–duration','A2  Waveform/random',
         'A3  Waveform/C3','S2  LB/random','S3  LB/SortaGrad',
         'S4  LD/random','S5  LD/SortaGrad']


def analyse():
    evidence_path = HERE/'chapter4_evidence_codex.json'
    evidence = json.loads(evidence_path.read_text())
    sources = {}
    results = []
    for split in ['test','fleurs_test']:
        manifest = Path('/ext_data/casper/asr_data')/(
            'fleurs_shona_corrected_v2' if split.startswith('fleurs') else 'waxal_shona_speaker_disjoint_v2')/'test.jsonl'
        raw = manifest.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == evidence['manifest_hashes'][split]
        sources[str(manifest)] = hashlib.sha256(raw).hexdigest()
        meta = {str(r['id']):r for r in map(json.loads,raw.splitlines())}
        field = 'source_id' if split.startswith('fleurs') else 'speaker_id'
        clusters = sorted({str(r[field]) for r in meta.values()})
        assert len(clusters) == (348 if split.startswith('fleurs') else 24)
        index = {c:i for i,c in enumerate(clusters)}
        errors = {}; reference_counts = None; refs = None
        for code in CODES:
            seed_errors = []
            for record in evidence['data']['Medium'][code]:
                path = ROOT/record['source']/('fleurs_corrected_v2' if split.startswith('fleurs') else 'item_predictions')/(split+'.jsonl')
                raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
                assert digest == record['prediction_hashes'][split]
                sources[str(path.relative_to(ROOT))] = digest
                records = [json.loads(line) for line in raw.splitlines()]
                rows = {str(r['id']):r for r in records}
                assert len(rows)==len(records) and rows.keys()==meta.keys()
                these_refs = {k:(r['reference'],r['reference_words']) for k,r in rows.items()}
                if refs is None: refs=these_refs
                else: assert refs==these_refs
                e = np.zeros(len(clusters)); n = np.zeros(len(clusters))
                for uid,r in rows.items():
                    i = index[str(meta[uid][field])]
                    if field=='speaker_id': assert r['speaker_id']==meta[uid][field]
                    e[i] += sum(r[k] for k in ['substitutions','deletions','insertions'])
                    n[i] += r['reference_words']
                assert np.isclose(100*e.sum()/n.sum(),record[split+'_wer'],rtol=0,atol=1e-10)
                if reference_counts is None: reference_counts=n
                else: assert np.array_equal(reference_counts,n)
                seed_errors.append(e)
            errors[code]=np.array(seed_errors)
        # One shared resample of evaluation clusters across all fixed checkpoints.
        # Average seed-wise corpus-WER differences; do not treat seed x item pairs
        # as independent recordings or resample the three seeds as a population.
        delta = np.array([(errors[c]-errors[t]).mean(axis=0) for t,c in CONTRASTS])
        assert np.allclose((errors['C0']-errors['C0']).sum(axis=1),0)
        rng = np.random.default_rng(20261016 if split=='test' else 20261017)
        replicates = 100_000
        draws = []
        for start in range(0,replicates,2000):
            weights = rng.multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),
                                      size=min(2000,replicates-start))
            draws.append(100*(weights@delta.T)/(weights@reference_counts)[:,None])
        draws=np.concatenate(draws)
        for j,(t,c) in enumerate(CONTRASTS):
            seed_effects = 100*(errors[c]-errors[t]).sum(axis=1)/reference_counts.sum()
            effect = float(seed_effects.mean())
            signs = np.array(list(itertools.product([-1,1],repeat=3)))
            signflip = float(np.mean(np.abs((signs*seed_effects).mean(axis=1))>=abs(effect)-1e-12))
            results.append(dict(split=split,treatment=t,control=c,improvement_pp=effect,
                seed_improvements_pp=seed_effects.tolist(),positive_seeds=int(sum(seed_effects>0)),
                ci95_pp=np.quantile(draws[:,j],[.025,.975]).tolist(),
                ci_bonferroni20_pp=np.quantile(draws[:,j],[.05/(2*20),1-.05/(2*20)]).tolist(),
                exact_seed_signflip_p=signflip,clusters=len(clusters),cluster_field=field,
                recordings=len(meta),reference_words=int(reference_counts.sum()),replicates=replicates))
    report=dict(method='Paired cluster percentile bootstrap, fixed-checkpoint mean over three matched seed records',
        interpretation='Exploratory post-review analysis of 20 existing within-Medium test contrasts; no training-seed population interval. Bonferroni intervals are approximate bootstrap family intervals, not correction for historical adaptive selection.',
        seed_test='Two-sided exact sign-flip sensitivity on three paired seed differences, requiring sign symmetry; minimum attainable p=0.25. Not a replacement for cluster inference.',
        evidence_data_sha256=hashlib.sha256(json.dumps(evidence['data'],sort_keys=True).encode()).hexdigest(),
        source_hashes=sources,results=results)
    (HERE/'chapter4_uncertainty_codex.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in results:
        print(r['split'],r['treatment'],r['control'],round(r['improvement_pp'],3),
              [round(x,3) for x in r['ci95_pp']], [round(x,3) for x in r['ci_bonferroni20_pp']])


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads((HERE/'chapter4_evidence_codex.json').read_text())['data']
    target=HERE/'figures';target.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':200})
    colors={'Base':'#24669b','Medium':'#b64d16'}
    fig,axes=plt.subplots(1,2,figsize=(10.2,5.0),sharey=True,layout='constrained')
    for ax,key,title in zip(axes,['test_wer','fleurs_test_wer'],['WAXAL test','FLEURS test']):
        for model,offset in [('Base',-.13),('Medium',.13)]:
            for i,c in enumerate(CODES):
                vals=np.array([r[key] for r in data[model][c]])
                ax.scatter(vals,np.full(len(vals),i+offset),s=17,color=colors[model],alpha=.45,zorder=3)
                ax.errorbar(vals.mean(),i+offset,xerr=None if c=='C3' else vals.std(ddof=1),
                    marker='D',markersize=4.5,color=colors[model],capsize=3,linewidth=1.2,
                    label=model if i==0 else None,zorder=4)
        ax.set_title(title,fontweight='bold');ax.set_xlabel('Word error rate (%)')
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,['Base (RTX 4090)','Medium (A100)'],loc='outside upper center',ncol=2,frameon=False)
    axes[0].set_yticks(range(len(CODES)),NAMES);axes[0].invert_yaxis()
    axes[0].set_xlim(20.5,33.5);axes[1].set_xlim(34,66)
    for ext in ['pdf','png']:fig.savefig(target/f'chapter4_wer_conditions_codex.{ext}')
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10.2,4.3),layout='constrained')
    offsets={'Base':{'C0':(-13,-15),'C1':(4,-13),'C3':(-24,8),'A2':(-6,-17),'A3':(7,4),
                     'S2':(-22,-13),'S3':(8,-9),'S4':(-25,6),'S5':(7,5)},
             'Medium':{'C0':(-12,-16),'C1':(-10,9),'C3':(6,-12),'A2':(-25,-13),'A3':(-28,5),
                       'S2':(5,-13),'S3':(-26,-13),'S4':(-25,8),'S5':(6,4)}}
    for ax,model in zip(axes,['Base','Medium']):
        x0=np.mean([r['test_wer'] for r in data[model]['C0']]);y0=np.mean([r['fleurs_test_wer'] for r in data[model]['C0']])
        ax.axvline(x0,color='.65',lw=.8,ls='--');ax.axhline(y0,color='.65',lw=.8,ls='--')
        for c in CODES:
            x=np.mean([r['test_wer'] for r in data[model][c]]);y=np.mean([r['fleurs_test_wer'] for r in data[model][c]])
            ax.scatter(x,y,s=48 if c=='C0' else 32,marker='s' if c=='C0' else 'o',color=colors[model],zorder=3)
            ax.annotate(c,(x,y),xytext=offsets[model][c],textcoords='offset points',fontsize=9,
                        arrowprops=dict(arrowstyle='-',lw=.5,color='.5'),zorder=4)
        ax.set_title(model+(' (RTX 4090)' if model=='Base' else ' (A100)'),fontweight='bold')
        ax.set_xlabel('WAXAL test WER (%)');ax.set_ylabel('FLEURS test WER (%)');ax.grid(alpha=.15)
        ax.margins(x=.2,y=.2)
    for ext in ['pdf','png']:fig.savefig(target/f'chapter4_domain_scatter_codex.{ext}')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--figures',action='store_true');args=parser.parse_args()
    figures() if args.figures else analyse()
