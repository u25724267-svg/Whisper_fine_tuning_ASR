"""Additional Chapter 4 figures proposed in chapter4_critique.md.
Reads only the audited evidence/uncertainty JSON; no training or inference.
Run: PYTHONPATH=/tmp/ch4_review_plot .venv/bin/python documents/thesis/figures_chapter4_critique.py
"""
from pathlib import Path
import json, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures'
EV = json.loads((HERE / 'chapter4_evidence_codex.json').read_text())
UN = json.loads((HERE / 'chapter4_uncertainty_codex.json').read_text())

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '.35', 'axes.labelcolor': '.15',
                     'xtick.color': '.3', 'ytick.color': '.3',
                     'grid.color': '.9', 'grid.linewidth': .6})
MODEL = {'Base': '#24669b', 'Medium': '#b64d16'}      # matches existing Chapter 4 figures
ORDER = {'Random': '#2a78d6', 'SortaGrad': '#eb6834', 'SNR-duration': '#1baf7a', 'Other': '#9a9a96'}
INK2 = '.35'


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=200)
    plt.close(fig)


def mean(model, code, key):
    return st.mean(r[key] for r in EV['data'][model][code])


# 1. Forest plot of the ten matched contrasts: Medium paired intervals, Base point estimates.
CONTRASTS = [('C1', 'C0'), ('C3', 'C0'), ('A2', 'C0'), ('A3', 'C3'), ('A3', 'A2'),
             ('S2', 'C0'), ('S3', 'C1'), ('S3', 'S2'), ('S4', 'S2'), ('S5', 'S3')]
GROUPS = ['Curriculum alone', 'Curriculum alone', 'Waveform aug.', 'Waveform aug.', 'Waveform aug.',
          'SpecAugment', 'SpecAugment', 'SpecAugment', 'SpecAugment', 'SpecAugment']
iv = {(r['split'], r['treatment'], r['control']): r for r in UN['results']}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True, layout='constrained')
for ax, (split, key, title) in zip(axes, [('test', 'test_wer', 'WAXAL test (in-domain)'),
                                          ('fleurs_test', 'fleurs_test_wer', 'FLEURS test (external)')]):
    ax.axvline(0, color='.45', lw=.9, zorder=1)
    for i, (t, c) in enumerate(CONTRASTS):
        y = len(CONTRASTS) - 1 - i
        r = iv[(split, t, c)]
        lo, hi = r['ci_bonferroni20_pp']
        ax.plot([lo, hi], [y, y], color=MODEL['Medium'], lw=1.0, alpha=.35, solid_capstyle='round', zorder=2)
        lo, hi = r['ci95_pp']
        ax.plot([lo, hi], [y, y], color=MODEL['Medium'], lw=2.4, solid_capstyle='round', zorder=3)
        ax.plot(r['improvement_pp'], y, 'o', ms=6, color=MODEL['Medium'], mec='white', mew=1.2, zorder=4)
        b = mean('Base', c, key) - mean('Base', t, key)
        ax.plot(b, y, 'D', ms=5.5, mfc='white', mec=MODEL['Base'], mew=1.4, zorder=4)
    for y in (4.5, 7.5):
        ax.axhline(y, color='.85', lw=.6)
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
    ax.set_xlabel('Improvement in test WER (pp), control − treatment\n← treatment worse      treatment better →')
    ax.grid(axis='x')
axes[0].set_yticks(range(len(CONTRASTS)))
axes[0].set_yticklabels([f'{t} vs {c}' for t, c in reversed(CONTRASTS)])
for y, g in [(9, 'Curriculum'), (6.5, 'Waveform'), (2, 'SpecAugment')]:
    axes[0].annotate(g, xy=(0, y), xycoords=('axes fraction', 'data'), xytext=(-62, 0),
                     textcoords='offset points', rotation=90, va='center', ha='center', color=INK2, fontsize=8)
handles = [plt.Line2D([], [], color=MODEL['Medium'], marker='o', lw=2.4, mec='white', label='Medium mean (95% paired cluster interval)'),
           plt.Line2D([], [], color=MODEL['Medium'], lw=1, alpha=.35, label='Medium 99.75% (20-comparison sensitivity)'),
           plt.Line2D([], [], color=MODEL['Base'], marker='D', mfc='white', lw=0, label='Base mean (no interval)')]
fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False, fontsize=8)
save(fig, 'chapter4_effects_forest')

# 2. Validation-WER learning curves: ordering family is the colour; SortaGrad epoch-one cost.
def family(code):
    if code in ('C1', 'S3', 'S5'): return 'SortaGrad'
    if code in ('C3', 'A3', 'C2'): return 'SNR-duration'
    if code in ('C0', 'A2', 'S2', 'S4'): return 'Random'
    return 'Other'

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), layout='constrained')
for ax, model in zip(axes, ['Base', 'Medium']):
    cells = EV['data'][model]
    for code, rows in sorted(cells.items(), key=lambda kv: family(kv[0]) != 'Other'):
        curves = [[e['eval_wer'] for e in r.get('epoch_evaluations', [])] for r in rows]
        curves = [c for c in curves if len(c) == 3]
        if not curves:
            continue
        m = [st.mean(c[i] for c in curves) for i in range(3)]
        f = family(code)
        ax.plot([1, 2, 3], m, '-o', color=ORDER[f], lw=1.6 if f != 'Other' else 1.1, ms=4,
                alpha=1 if f != 'Other' else .8, zorder=3 if f != 'Other' else 2)
        if f == 'SortaGrad':
            ax.annotate(code, (1, m[0]), xytext=(6, 0), textcoords='offset points', va='center',
                        fontsize=8, color='.2')
    ax.set_title(f'{model} ({"RTX 4090" if model == "Base" else "A100"})', fontsize=10, fontweight='bold', loc='left')
    ax.set_xticks([1, 2, 3]); ax.set_xlim(.8, 3.25)
    ax.set_xlabel('Epoch (end-of-epoch validation)')
    ax.grid(axis='y')
axes[0].set_ylabel('WAXAL validation WER (%), seed mean')
fig.legend(handles=[plt.Line2D([], [], color=v, marker='o', lw=1.6, ms=4,
                               label={'Random': 'Random order (C0, A2, S2, S4)',
                                      'SortaGrad': 'SortaGrad (C1, S3, S5)',
                                      'SNR-duration': 'SNR/duration order (C2, C3, A3)',
                                      'Other': 'Dynamic/paced, Base only (C4–C7)'}[k])
                    for k, v in ORDER.items()],
           loc='outside lower center', ncol=4, frameon=False, fontsize=8)
save(fig, 'chapter4_learning_curves')

# 3. Where the Medium "training time" goes: optimiser steps vs in-loop validation generation.
def split_time(rows):
    ev = [sum(e.get('eval_runtime', 0) for e in r['epoch_evaluations']) / 60 for r in rows if r.get('runtime_min')]
    tot = [r['runtime_min'] for r in rows if r.get('runtime_min')]
    return st.mean(t - e for t, e in zip(tot, ev)), st.mean(ev)

hist = EV['historical_medium_rtx']
bars = [('Base C0, RTX 4090', split_time(EV['data']['Base']['C0'])),
        ('Medium C0, RTX 4090', split_time([r for r in hist if r['condition'] == 'C0'])),
        ('Medium C0, A100', split_time(EV['data']['Medium']['C0'])),
        ('Medium C1, A100', split_time(EV['data']['Medium']['C1'])),
        ('Medium S2, A100', split_time(EV['data']['Medium']['S2'])),
        ('Medium S3, A100', split_time(EV['data']['Medium']['S3']))]
fig, ax = plt.subplots(figsize=(8.2, 3.2), layout='constrained')
for i, (label, (train, ev)) in enumerate(reversed(bars)):
    ax.barh(i, train, color='#2a78d6', height=.62, edgecolor='white', linewidth=2)
    ax.barh(i, ev, left=train, color='#eb6834', height=.62, edgecolor='white', linewidth=2)
    ax.text(train + ev + 4, i, f'{train + ev:.0f} min', va='center', fontsize=8, color='.2')
ax.set_yticks(range(len(bars))); ax.set_yticklabels([b[0] for b in reversed(bars)])
ax.set_xlabel('Mean recorded training-loop minutes (three epochs)')
ax.set_xlim(0, 420); ax.grid(axis='x')
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color='#2a78d6', label='Optimiser steps (remainder)'),
                   plt.Rectangle((0, 0), 1, 1, color='#eb6834', label='Three in-loop validation generations')],
          loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8)
save(fig, 'chapter4_runtime_decomposition')
for label, (t, e) in bars:
    print(f'{label:22} train {t:6.1f}  eval {e:6.1f}  total {t + e:6.1f}')
