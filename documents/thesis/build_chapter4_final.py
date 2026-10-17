"""Build figures, tables and analysis facts for chapter4_final.tex.

Reads the audited evidence (chapter4_evidence_codex.json), the paired intervals
(chapter4_uncertainty_codex.json), saved per-utterance predictions and summaries,
and the Medium WAXAL zero-shot summary. No training or inference is performed.

Run: PYTHONPATH=/tmp/ch4_review_plot MPLCONFIGDIR=/tmp/ch4_mplconfig \
     .venv/bin/python documents/thesis/build_chapter4_final.py
Scope follows chapter3_final.tex: C4, C4R and C7 are excluded.
"""
from pathlib import Path
import csv, hashlib, json, re, statistics as st

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / 'figures'
DATA = Path('/ext_data/casper/asr_data')
EVIDENCE = next(p for p in (HERE / 'chapter4_evidence_codex.json', HERE / 'Inactive/chapter4_evidence_codex.json') if p.exists())
EV = json.loads(EVIDENCE.read_text())
UN = json.loads((HERE / 'chapter4_uncertainty_codex.json').read_text())
ZERO_MEDIUM_WAXAL = ROOT / 'artifacts/experiment_outputs/baselines/whisper_medium_zero_shot_waxal_v1/summary.json'

COMMON = ['C0', 'C1', 'C3', 'A2', 'A3', 'S2', 'S3', 'S4', 'S5']
BASE_ONLY = ['C2', 'C5', 'C6']
NAMES = {'C0': 'Random order', 'C1': 'SortaGrad', 'C2': 'SNR order', 'C3': 'SNR+duration order',
         'C5': 'Loss-based (dynamic)', 'C6': 'WER-based (dynamic)', 'A2': 'Waveform + random',
         'A3': 'Waveform + SNR+duration', 'S2': 'Light masks + random', 'S3': 'Light masks + SortaGrad',
         'S4': 'Strong masks + random', 'S5': 'Strong masks + SortaGrad'}
DATA_RUNS = {m: {c: rows for c, rows in EV['data'][m].items() if c in COMMON + BASE_ONLY}
             for m in ['Base', 'Medium']}

MODEL = {'Base': '#24669b', 'Medium': '#b64d16'}
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # validated categorical order
GREY = '#9a9a96'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '.35', 'axes.labelcolor': '.15',
                     'xtick.color': '.3', 'ytick.color': '.3',
                     'grid.color': '.9', 'grid.linewidth': .6})
FACTS = {}


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=200)
    plt.close(fig)


def mean(model, code, key):
    return st.mean(r[key] for r in DATA_RUNS[model][code])


def summary(record, split):
    sub = 'fleurs_corrected_v2' if split.startswith('fleurs') else 'item_predictions'
    s = json.loads((ROOT / record['source'] / sub / 'summary.json').read_text())['splits'][split]
    assert s['predictions_sha256'] == record['prediction_hashes'][split]
    return s


def sdi_rates(s):
    n = s['reference_words']
    return [100 * s[k] / n for k in ('substitutions', 'deletions', 'insertions')]


# ---------------------------------------------------------------- zero-shot
zero = {m: EV['zero_shot'][m]['splits'] for m in ['Base', 'Medium']}
zm = json.loads(ZERO_MEDIUM_WAXAL.read_text())
for split, v in zm['splits'].items():
    assert v['manifest_sha256'] == EV['manifest_hashes'][split]
    assert abs(v['wer'] - 100 * (v['substitutions'] + v['deletions'] + v['insertions']) / v['reference_words']) < 1e-9
    zero['Medium'][split] = v
FACTS['medium_waxal_zero_sha256'] = hashlib.sha256(ZERO_MEDIUM_WAXAL.read_bytes()).hexdigest()

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True, layout='constrained')
rows = [('Base', 'zero'), ('Base', 'fine'), ('Medium', 'zero'), ('Medium', 'fine')]
for ax, (split, title) in zip(axes, [('test', 'WAXAL test (in-domain)'), ('fleurs_test', 'FLEURS test (external)')]):
    for i, (model, stage) in enumerate(reversed(rows)):
        if stage == 'zero':
            rates = sdi_rates(zero[model][split])
        else:
            rates = list(np.mean([sdi_rates(summary(r, split)) for r in DATA_RUNS[model]['C0']], axis=0))
        FACTS[f'sdi_{model}_{stage}_{split}'] = rates
        left = 0
        for j, v in enumerate(rates):
            ax.barh(i, v, left=left, color=CAT[j], height=.64, edgecolor='white', linewidth=1.5)
            left += v
        ax.text(left + 3, i, f'{left:.1f}', va='center', fontsize=8, color='.2')
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
    ax.set_xlabel('Word errors per 100 reference words (bar length = WER %)')
    ax.set_xlim(0, 185); ax.grid(axis='x')
axes[0].set_yticks(range(4))
axes[0].set_yticklabels([f'{m} {"zero-shot" if s == "zero" else "fine-tuned (C0)"}' for m, s in reversed(rows)])
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CAT[j], label=l)
                    for j, l in enumerate(['Substitutions', 'Deletions', 'Insertions'])],
           loc='outside lower center', ncol=3, frameon=False, fontsize=8)
save(fig, 'ch4_zero_shot_errors')

# ------------------------------------------------------- Base curriculum screen
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True, layout='constrained')
codes = ['C0', 'C1', 'C2', 'C3', 'C5', 'C6']
for ax, (key, title) in zip(axes, [('test_wer', 'WAXAL test'), ('fleurs_test_wer', 'FLEURS test')]):
    ref = mean('Base', 'C0', key)
    ax.axvline(ref, color='.55', ls='--', lw=.9, zorder=1)
    for i, c in enumerate(codes):
        y = len(codes) - 1 - i
        vals = [r[key] for r in DATA_RUNS['Base'][c]]
        identical = c in ('C2', 'C3')
        if not identical:
            ax.scatter(vals, [y] * len(vals), s=18, color=MODEL['Base'], alpha=.4, zorder=2, linewidths=0)
        ax.plot(st.mean(vals), y, 'D', ms=6, color=MODEL['Base'], mec='white', mew=1, zorder=3)
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
    ax.set_xlabel('Word error rate (%)'); ax.grid(axis='x')
axes[0].set_yticks(range(len(codes)))
axes[0].set_yticklabels([f'{c}  {NAMES[c]}' for c in reversed(codes)])
save(fig, 'ch4_base_curricula')

# ------------------------------------------- overview of shared conditions
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True, layout='constrained')
for ax, (key, title) in zip(axes, [('test_wer', 'WAXAL test (in-domain)'), ('fleurs_test_wer', 'FLEURS test (external)')]):
    for i, c in enumerate(COMMON):
        for model, off in [('Base', -.17), ('Medium', .17)]:
            y = len(COMMON) - 1 - i + off
            vals = [r[key] for r in DATA_RUNS[model][c]]
            if c != 'C3':
                ax.scatter(vals, [y] * 3, s=16, color=MODEL[model], alpha=.4, linewidths=0, zorder=2)
            ax.plot(st.mean(vals), y, 'D', ms=5.5, color=MODEL[model], mec='white', mew=1, zorder=3)
    for y in (3.5, 5.5):
        ax.axhline(y, color='.85', lw=.6)
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
    ax.set_xlabel('Word error rate (%)'); ax.grid(axis='x')
axes[0].set_yticks(range(len(COMMON)))
axes[0].set_yticklabels([f'{c}  {NAMES[c]}' for c in reversed(COMMON)])
fig.legend(handles=[plt.Line2D([], [], color=MODEL[m], marker='D', lw=0, ms=5.5, label=f'Whisper {m} (mean; dots = runs)') for m in MODEL],
           loc='outside lower center', ncol=2, frameon=False, fontsize=8)
save(fig, 'ch4_overview')

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), layout='constrained')
for ax, model in zip(axes, ['Base', 'Medium']):
    x0, y0 = mean(model, 'C0', 'test_wer'), mean(model, 'C0', 'fleurs_test_wer')
    ax.axvline(x0, color='.6', lw=.8, ls='--'); ax.axhline(y0, color='.6', lw=.8, ls='--')
    for c in COMMON:
        x, y = mean(model, c, 'test_wer'), mean(model, c, 'fleurs_test_wer')
        ax.scatter(x, y, s=46 if c == 'C0' else 30, marker='s' if c == 'C0' else 'o', color=MODEL[model], zorder=3)
        off = (5, -11) if (model, c) in (('Base', 'C3'), ('Medium', 'C3')) else (5, 4)
        ax.annotate(c, (x, y), xytext=off, textcoords='offset points', fontsize=8, color='.2')
    ax.set_title(f'Whisper {model}', fontsize=10, fontweight='bold', loc='left')
    ax.set_xlabel('WAXAL test WER (%)'); ax.grid()
axes[0].set_ylabel('FLEURS test WER (%)')
save(fig, 'ch4_domain_scatter')

# ---------------------------------------------------------- learning curves
FAMILY = {'C0': 0, 'A2': 0, 'S2': 0, 'S4': 0, 'C1': 1, 'S3': 1, 'S5': 1, 'C2': 2, 'C3': 2, 'A3': 2, 'C5': 3, 'C6': 3}
FAMCOL = [CAT[0], CAT[1], CAT[2], GREY]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), layout='constrained')
curves = {}
for ax, model in zip(axes, ['Base', 'Medium']):
    for code, runs in sorted(DATA_RUNS[model].items(), key=lambda kv: FAMILY[kv[0]] != 3):
        cs = [[e['eval_wer'] for e in r.get('epoch_evaluations', [])] for r in runs]
        cs = [c for c in cs if len(c) == 3]
        m = [st.mean(c[i] for c in cs) for i in range(3)]
        curves[f'{model}_{code}'] = m
        f = FAMILY[code]
        ax.plot([1, 2, 3], m, '-o', color=FAMCOL[f], lw=1.6 if f != 3 else 1.1, ms=4, zorder=3 if f != 3 else 2)
        if f == 1:
            ax.annotate(code, (1, m[0]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=8, color='.2')
    ax.set_title(f'Whisper {model}', fontsize=10, fontweight='bold', loc='left')
    ax.set_xticks([1, 2, 3]); ax.set_xlim(.8, 3.25)
    ax.set_xlabel('Epoch'); ax.grid(axis='y')
axes[0].set_ylabel('WAXAL validation WER (%)')
fig.legend(handles=[plt.Line2D([], [], color=FAMCOL[i], marker='o', lw=1.6, ms=4, label=l) for i, l in enumerate(
    ['Random order (C0, A2, S2, S4)', 'SortaGrad (C1, S3, S5)', 'SNR-based order (C2, C3, A3)', 'Dynamic, Base only (C5, C6)'])],
    loc='outside lower center', ncol=4, frameon=False, fontsize=8)
save(fig, 'ch4_learning_curves')
FACTS['curves'] = curves
sel = [r.get('best_checkpoint', '') for m in DATA_RUNS for rs in DATA_RUNS[m].values() for r in rs]
FACTS['checkpoint_selection'] = {k: sel.count(k) for k in set(sel)}
FACTS['epoch1_eval_minutes_medium'] = {c: st.mean(r['epoch_evaluations'][0]['eval_runtime'] / 60 for r in DATA_RUNS['Medium'][c])
                                       for c in COMMON}

# ----------------------------------------------------------- forest plot
CONTRASTS = [('C1', 'C0'), ('C3', 'C0'), ('A2', 'C0'), ('A3', 'C3'), ('A3', 'A2'),
             ('S2', 'C0'), ('S3', 'C1'), ('S3', 'S2'), ('S4', 'S2'), ('S5', 'S3')]
iv = {(r['split'], r['treatment'], r['control']): r for r in UN['results']}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True, layout='constrained')
for ax, (split, key, title) in zip(axes, [('test', 'test_wer', 'WAXAL test (in-domain)'),
                                          ('fleurs_test', 'fleurs_test_wer', 'FLEURS test (external)')]):
    ax.axvline(0, color='.45', lw=.9, zorder=1)
    for i, (t, c) in enumerate(CONTRASTS):
        y = len(CONTRASTS) - 1 - i
        r = iv[(split, t, c)]
        ax.plot(r['ci_bonferroni20_pp'], [y, y], color=MODEL['Medium'], lw=1.0, alpha=.35, zorder=2)
        ax.plot(r['ci95_pp'], [y, y], color=MODEL['Medium'], lw=2.4, solid_capstyle='round', zorder=3)
        ax.plot(r['improvement_pp'], y, 'o', ms=6, color=MODEL['Medium'], mec='white', mew=1.2, zorder=4)
        ax.plot(mean('Base', c, key) - mean('Base', t, key), y, 'D', ms=5.5, mfc='white',
                mec=MODEL['Base'], mew=1.4, zorder=4)
    for y in (4.5, 7.5):
        ax.axhline(y, color='.85', lw=.6)
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
    ax.set_xlabel('Reduction in test WER (pp)\n← treatment worse      treatment better →')
    ax.grid(axis='x')
axes[0].set_yticks(range(len(CONTRASTS)))
axes[0].set_yticklabels([f'{t} vs {c}' for t, c in reversed(CONTRASTS)])
fig.legend(handles=[
    plt.Line2D([], [], color=MODEL['Medium'], marker='o', lw=2.4, mec='white', label='Medium: mean and 95% interval'),
    plt.Line2D([], [], color=MODEL['Medium'], lw=1, alpha=.35, label='Medium: 99.75% interval'),
    plt.Line2D([], [], color=MODEL['Base'], marker='D', mfc='white', lw=0, label='Base: mean only')],
    loc='outside lower center', ncol=3, frameon=False, fontsize=8)
save(fig, 'ch4_effects_forest')

# ----------------------------------------- per-utterance Medium analyses
def load_medium(split):
    manifest = DATA / ('fleurs_shona_corrected_v2' if split.startswith('fleurs') else 'waxal_shona_speaker_disjoint_v2') / 'test.jsonl'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == EV['manifest_hashes'][split]
    meta = {str(r['id']): r for r in map(json.loads, manifest.read_text().splitlines())}
    field = 'source_id' if split.startswith('fleurs') else 'speaker_id'
    items = {}
    for code in COMMON:
        runs = []
        for rec in DATA_RUNS['Medium'][code]:
            sub = 'fleurs_corrected_v2' if split.startswith('fleurs') else 'item_predictions'
            p = ROOT / rec['source'] / sub / f'{split}.jsonl'
            raw = p.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == rec['prediction_hashes'][split]
            runs.append({str(r['id']): r for r in map(json.loads, raw.splitlines())})
        items[code] = runs
    return meta, field, items


def cluster_bootstrap(E, N, n_rep=100_000, seed=0):
    """E: contrasts x clusters error-differences (seed-averaged); N: clusters reference words."""
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, n_rep, 5000):
        w = rng.multinomial(len(N), np.full(len(N), 1 / len(N)), size=min(5000, n_rep - start))
        out.append(100 * (w @ E.T) / (w @ N)[:, None])
    return np.concatenate(out)


INTERACTIONS = {'waveform x SNR+duration': (('A2', 'A3'), ('C0', 'C3')),
                'light masks x SortaGrad': (('S2', 'S3'), ('C0', 'C1'))}
FACTS['interactions'] = {}
for split in ['test', 'fleurs_test']:
    meta, field, items = load_medium(split)
    clusters = sorted({str(m[field]) for m in meta.values()}); idx = {c: i for i, c in enumerate(clusters)}
    def per_cluster(rows):
        e = np.zeros(len(clusters)); n = np.zeros(len(clusters))
        for uid, r in rows.items():
            i = idx[str(meta[uid][field])]
            e[i] += r['substitutions'] + r['deletions'] + r['insertions']; n[i] += r['reference_words']
        return e, n
    E = {c: np.array([per_cluster(rs)[0] for rs in items[c]]) for c in COMMON}
    N = per_cluster(items['C0'][0])[1]
    deltas = np.array([((E[a] - E[b]) - (E[c] - E[d])).mean(axis=0) for (a, b), (c, d) in INTERACTIONS.values()])
    draws = cluster_bootstrap(deltas, N, seed=31 if split == 'test' else 37)
    for j, name in enumerate(INTERACTIONS):
        FACTS['interactions'][f'{split}:{name}'] = dict(estimate_pp=float(100 * deltas[j].sum() / N.sum()),
                                                        ci95_pp=np.quantile(draws[:, j], [.025, .975]).tolist())

# Difficulty bins (WAXAL test only: SNR proxy and duration terciles of the test set)
meta, field, items = load_medium('test')
acoustic = {r['id']: r for r in csv.DictReader(open(DATA / 'waxal_shona_speaker_disjoint_v2/acoustic_metadata.csv')) if r['split'] == 'test'}
assert acoustic.keys() == meta.keys()
speakers = sorted({m['speaker_id'] for m in meta.values()}); sidx = {s: i for i, s in enumerate(speakers)}
BIN_CODES = ['C1', 'C3', 'A2', 'S2', 'S3']
bins_out = {}
for axis_name, value, labels in [('snr', lambda u: float(acoustic[u]['snr_proxy_db']), ['Low SNR', 'Mid SNR', 'High SNR (60 dB cap)']),
                                 ('duration', lambda u: float(meta[u]['duration']), ['Short', 'Medium', 'Long'])]:
    vals = {u: value(u) for u in meta}
    cuts = np.quantile(list(vals.values()), [1 / 3, 2 / 3])
    bin_of = {u: int(np.searchsorted(cuts, v, side='right')) for u, v in vals.items()}
    res = []
    for b in range(3):
        ids = [u for u in meta if bin_of[u] == b]
        def per_spk(rows):
            e = np.zeros(len(speakers)); n = np.zeros(len(speakers))
            for u in ids:
                r = rows[u]; i = sidx[meta[u]['speaker_id']]
                e[i] += r['substitutions'] + r['deletions'] + r['insertions']; n[i] += r['reference_words']
            return e, n
        E = {c: np.array([per_spk(rs)[0] for rs in items[c]]) for c in ['C0'] + BIN_CODES}
        N = per_spk(items['C0'][0])[1]
        deltas = np.array([(E['C0'] - E[c]).mean(axis=0) for c in BIN_CODES])
        draws = cluster_bootstrap(deltas, N, n_rep=20_000, seed=100 + b)
        res.append(dict(label=labels[b], utterances=len(ids), range=[float(min(vals[u] for u in ids)), float(max(vals[u] for u in ids))],
                        c0_wer=float(100 * E['C0'].sum(axis=1).mean() / N.sum()),
                        effects={c: dict(pp=float(100 * deltas[k].sum() / N.sum()),
                                         ci95=np.quantile(draws[:, k], [.025, .975]).tolist()) for k, c in enumerate(BIN_CODES)}))
    bins_out[axis_name] = res
FACTS['bins'] = bins_out

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True, layout='constrained')
for ax, (axis_name, title) in zip(axes, [('snr', 'By estimated SNR (test-set terciles)'), ('duration', 'By utterance duration (test-set terciles)')]):
    res = bins_out[axis_name]
    ax.axhline(0, color='.45', lw=.9)
    for k, c in enumerate(BIN_CODES):
        x = np.arange(3) + (k - 2) * .1
        y = [r['effects'][c]['pp'] for r in res]
        lo = [r['effects'][c]['pp'] - r['effects'][c]['ci95'][0] for r in res]
        hi = [r['effects'][c]['ci95'][1] - r['effects'][c]['pp'] for r in res]
        ax.errorbar(x, y, yerr=[lo, hi], fmt='o', ms=5, color=CAT[k], elinewidth=1.2, capsize=0, mec='white', mew=.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{r['label']}\nC0 WER {r['c0_wer']:.1f}%" for r in res])
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left'); ax.grid(axis='y')
axes[0].set_ylabel('Reduction in WER vs C0 (pp)\n← worse   better →')
fig.legend(handles=[plt.Line2D([], [], color=CAT[k], marker='o', lw=0, ms=5, label=f'{c} {NAMES[c]}') for k, c in enumerate(BIN_CODES)],
           loc='outside lower center', ncol=5, frameon=False, fontsize=8)
save(fig, 'ch4_difficulty_bins')

# --------------------------------------------------------- runtime split
def split_time(runs):
    runs = [r for r in runs if r.get('runtime_min')]
    ev = [sum(e.get('eval_runtime', 0) for e in r['epoch_evaluations']) / 60 for r in runs]
    return st.mean(r['runtime_min'] - e for r, e in zip(runs, ev)), st.mean(ev)

hist = EV['historical_medium_rtx']
bars = [('Base C0 (RTX 4090)', split_time(DATA_RUNS['Base']['C0'])),
        ('Base C6 (RTX 4090)', split_time(DATA_RUNS['Base']['C6'])),
        ('Medium C0 (RTX 4090)*', split_time([r for r in hist if r['condition'] == 'C0'])),
        ('Medium C0 (A100)', split_time(DATA_RUNS['Medium']['C0'])),
        ('Medium C1 (A100)', split_time(DATA_RUNS['Medium']['C1']))]
FACTS['runtime_split'] = {l: dict(steps=t, validation=e) for l, (t, e) in bars}
fig, ax = plt.subplots(figsize=(8.2, 2.9), layout='constrained')
for i, (label, (t, e)) in enumerate(reversed(bars)):
    ax.barh(i, t, color=CAT[0], height=.62, edgecolor='white', linewidth=2)
    ax.barh(i, e, left=t, color=CAT[1], height=.62, edgecolor='white', linewidth=2)
    ax.text(t + e + 4, i, f'{t + e:.0f} min', va='center', fontsize=8, color='.2')
ax.set_yticks(range(len(bars))); ax.set_yticklabels([b[0] for b in reversed(bars)])
ax.set_xlabel('Mean training-loop minutes over three epochs'); ax.set_xlim(0, 420); ax.grid(axis='x')
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CAT[0], label='Training updates (incl. curriculum scoring)'),
                   plt.Rectangle((0, 0), 1, 1, color=CAT[1], label='End-of-epoch validation decoding')],
          loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8)
save(fig, 'ch4_runtime')

# ------------------------------------------------------------- tables
def cell(model, code, key, digits=2):
    vals = [r[key] for r in DATA_RUNS[model][code]]
    if code in ('C2', 'C3') and key != 'runtime_min':
        return f'${st.mean(vals):.{digits}f}^{{\\dagger}}$'
    return f'${st.mean(vals):.{digits}f} \\pm {st.stdev(vals):.{digits}f}$'


def row(fields):
    return ' & '.join(fields) + r' \\'


tables = {}
zrows = []
for corpus, prefix in [('WAXAL', ''), ('FLEURS', 'fleurs_')]:
    for model in ['Base', 'Medium']:
        z = zero[model][prefix + 'test']
        fw, fc = mean(model, 'C0', prefix + 'test_wer'), mean(model, 'C0', prefix + 'test_cer')
        zrows.append(row([corpus, model, f"{z['wer']:.1f}", f"{z['cer']:.1f}", f'{fw:.1f}', f'{fc:.1f}', f"{z['wer'] - fw:.1f}"]))
        FACTS[f'gain_{model}_{corpus}'] = dict(zero_wer=z['wer'], zero_cer=z['cer'], ft_wer=fw, ft_cer=fc,
                                                gain=z['wer'] - fw, rel=100 * (z['wer'] - fw) / z['wer'])
tables['zero'] = '\n'.join(zrows)
tables['conditions'] = '\n'.join(row([c, NAMES[c], *[r'\checkmark' if c in DATA_RUNS[m] else '--' for m in ['Base', 'Medium']]])
                                 for c in ['C0', 'C1', 'C2', 'C3', 'C5', 'C6', 'A2', 'A3', 'S2', 'S3', 'S4', 'S5'])
tables['waxal'] = '\n'.join(row([c] + [cell(m, c, k) for m in ['Base', 'Medium'] for k in ['validation_wer', 'test_wer', 'test_cer']]) for c in COMMON)
tables['fleurs'] = '\n'.join(row([c] + [cell(m, c, k) for m in ['Base', 'Medium'] for k in ['fleurs_test_wer', 'fleurs_test_cer']]) for c in COMMON)
tables['basecur'] = '\n'.join(row([c, cell('Base', c, 'validation_wer'), cell('Base', c, 'test_wer'), cell('Base', c, 'test_cer'),
                                   cell('Base', c, 'fleurs_test_wer'), cell('Base', c, 'runtime_min', 1)]) for c in ['C0', 'C1', 'C2', 'C3', 'C5', 'C6'])


def eff(model, t, c, key):
    v = mean(model, c, key) - mean(model, t, key)
    if model == 'Base':
        return f'${v:+.2f}$'
    r = iv[(key.removesuffix('_wer'), t, c)]
    assert abs(r['improvement_pp'] - v) < 1e-10
    return f"${v:+.2f}\\;[{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}]$"


tables['effects'] = '\n'.join(row([f'{t} vs {c}'] + [eff(m, t, c, k) for k in ['test_wer', 'fleurs_test_wer'] for m in ['Base', 'Medium']])
                              for t, c in CONTRASTS)
tables['time'] = '\n'.join(row([c, cell('Base', c, 'runtime_min', 1),
                                cell('Medium', c, 'runtime_min', 1) if c in DATA_RUNS['Medium'] else '--']) for c in COMMON + ['C5', 'C6'])
tables['migration'] = '\n'.join(row([r['condition'], str(r['seed']), f"{r['test_wer']:.2f}", f"{new['test_wer']:.2f}",
                                     f"{r['test_wer'] - new['test_wer']:+.2f}"])
                                for r in hist for new in DATA_RUNS['Medium'][r['condition']] if new['seed'] == r['seed'])

chapter = HERE / 'chapter4_final.tex'
if chapter.exists():
    text = chapter.read_text()
    for key, table in tables.items():
        text, n = re.subn(rf'(% BEGIN AUTO {key}\n).*?(% END AUTO {key})', lambda m: m[1] + table + '\n' + m[2], text, flags=re.S)
        assert n == 1, (key, n)
    chapter.write_text(text)
(HERE / 'chapter4_final_facts.json').write_text(json.dumps(FACTS, indent=1) + '\n')
print('figures and facts written; tables rendered' if chapter.exists() else 'figures and facts written; chapter not found')
