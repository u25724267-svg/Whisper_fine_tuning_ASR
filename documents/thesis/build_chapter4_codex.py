"""Rebuild the Codex Chapter 4 tables from saved artifacts; no training or inference.
Run with .venv/bin/python documents/thesis/build_chapter4_codex.py.
Requires the local A100 W&B files for the sampled memory audit. Uses a scratch
copy because W&B's scanner opens files in update mode. Narrative is not rewritten;
review it after changing evidence, especially the pending Medium zero-shot result.
"""
from pathlib import Path
import json, hashlib, statistics as st, shutil
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/experiment_outputs'
SOURCES={}
def read(p):
    p=Path(p); raw=p.read_bytes()
    SOURCES[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

names={'C0':('rq1','c0_random'), 'C1':('rq1','c1_sortagrad'), 'C2':('rq1','c2_snr'),
       'C3':('rq1','c3_snr_duration'),'C4':('rq1','c4_acoustic_cumulative'),
       'C4R':('rq1','c4r_random_pacing'),'C5':('rq1','c5_s2s_margin'),
       'C6':('rq1','c6_wer_margin'),'C7':('rq1','c7_acoustic_s2s_hybrid'),
       'A2':('rq2','a2_waveform_random'),'A3':('rq2','a3_waveform_c3'),
       'S2':('rq2','s2_specaug_lb_random'),'S3':('rq2','s3_specaug_lb_sortagrad'),
       'S4':('rq2','s4_specaug_ld_random'),'S5':('rq2','s5_specaug_ld_sortagrad')}
COMMON=['C0','C1','C3','A2','A3','S2','S3','S4','S5']
D={}; HASHES={}; missing=[]
def run(model,code,seed,prefix):
    family,name=names[code]
    d=OUT/prefix/family/(f'{name}_seed{seed}'+('_v2' if code=='C0' else ''))
    summary=read(d/'item_predictions/summary.json')
    results=read(d/'all_results.json'); manifest=read(d/'run_manifest.json'); cfg=read(d/'experiment_config.json')
    assert manifest['effective_seed']==seed and results['epoch']==3
    expected_model='openai/whisper-base' if model=='Base' else 'openai/whisper-medium'
    assert cfg['model']['id']==manifest['model']['id']==expected_model
    r=dict(model=model,condition=code,seed=seed,source=str(d.relative_to(ROOT)),
           gpu=manifest['runtime']['gpu'], runtime_min=results['train_runtime']/60,
           validation_loss=results['validation_loss'],prediction_hashes={})
    if prefix=='medium' and code=='C1' and seed==43:
        r.update(gpu='NVIDIA GeForce RTX 4090',runtime_min=None,recovery_gpu='NVIDIA A100-SXM4-40GB')
    if (d/'trainer_state.json').exists():
        state=read(d/'trainer_state.json'); r['best_checkpoint']=state.get('best_model_checkpoint','').split('/')[-1]
        r['epoch_evaluations']=[h for h in state['log_history'] if 'eval_wer' in h]
        if model=='Medium': assert state['global_step']==6906 and state['epoch']==3
    summaries=[summary]
    if (d/'fleurs_corrected_v2/summary.json').exists(): summaries.append(read(d/'fleurs_corrected_v2/summary.json'))
    for summ in summaries:
        for split,v in summ['splits'].items():
            assert v['rows']=={'validation':1715,'test':1671,'fleurs_validation':393,'fleurs_test':925}[split]
            assert abs(v['wer']-100*sum(v[k] for k in ['substitutions','deletions','insertions'])/v['reference_words'])<1e-9
            assert HASHES.setdefault(split,v['manifest_sha256'])==v['manifest_sha256']
            r.update({f'{split}_{k}':v[k] for k in ['wer','cer']})
            r['prediction_hashes'][split]=v['predictions_sha256']
            # Archived absolute paths may predate migration; use mirror-relative files.
            f=d/('fleurs_corrected_v2' if split.startswith('fleurs') else 'item_predictions')/(split+'.jsonl')
            if f.exists():
                raw=f.read_bytes(); assert hashlib.sha256(raw).hexdigest()==v['predictions_sha256'],f
                SOURCES[str(f.relative_to(ROOT))]=v['predictions_sha256']
            else: missing.append(str(f.relative_to(ROOT)))
    if model=='Medium':
        assert r['gpu']=='NVIDIA A100-SXM4-40GB'
        base=read(OUT/family/(f'{name}_seed{seed}'+('_v2' if code=='C0' else ''))/'experiment_config.json')
        for k in ['data','training','curriculum','augmentation','waveform_augmentation']:
            assert cfg.get(k)==base.get(k),(d,k)
    return r

for model in ['Base','Medium']:
    D[model]={}
    for code in (names if model=='Base' else COMMON):
        rows=[]
        for seed in ([42] if code in ['C4','C4R','C7'] else [42,43,44]):
            prefix='' if model=='Base' else ('medium_a100' if code=='C0' or code=='C1' and seed!=44 else 'medium')
            rows.append(run(model,code,seed,prefix))
        D[model][code]=rows
        for split in ['validation','test']:
            if code in ['C2','C3']: assert len({r['prediction_hashes'][split] for r in rows})==1
historical=[run('Medium RTX',c,s,'medium') for c,seeds in [('C0',[42,43,44]),('C1',[42,43])] for s in seeds]
for rows in D['Medium'].values(): assert all('fleurs_test_wer' in r for r in rows)
zero={model:read(OUT/'baselines'/name/'summary.json') for model,name in [('Base','whisper_base_zero_shot_waxal_fleurs_v1'),('Medium','whisper_medium_zero_shot_fleurs_v1')]}
for model,s in zero.items():
    for split,v in s['splits'].items():
        assert v['manifest_sha256']==HASHES[split]
        assert abs(v['wer']-100*sum(v[k] for k in ['substitutions','deletions','insertions'])/v['reference_words'])<1e-9
        baseline_dir='whisper_base_zero_shot_waxal_fleurs_v1' if model=='Base' else 'whisper_medium_zero_shot_fleurs_v1'
        prediction=OUT/'baselines'/baseline_dir/(split+'.jsonl')
        if prediction.exists():
            assert hashlib.sha256(prediction.read_bytes()).hexdigest()==v['predictions_sha256']
            SOURCES[str(prediction.relative_to(ROOT))]=v['predictions_sha256']
        else:
            missing.append(str(prediction.relative_to(ROOT)))

for relative in ['documents/data/medium_a100_controls_2026-10-15.md',
                 'documents/data/whisper_medium_rq1_rq2_protocol.md',
                 'artifacts/experiment_outputs/rq1/comparisons/c3_vs_c0_test.json',
                 'artifacts/experiment_outputs/medium/diagnostics/memory_clean_batch6_20261006_v2.json',
                 'artifacts/experiment_outputs/medium/diagnostics/memory_ld_batch6_20261006.json']:
    SOURCES[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()

telemetry=[]
for p in sorted(Path('/ext_data/casper/asr_experiment_outputs/medium_a100').glob('rq1/*/wandb/wandb/run-*/*.wandb')):
    name=p.parts[-5]; device=1 if name in ['c0_random_seed43_v2','c1_sortagrad_seed42'] else 0
    meta=read(p.parent/'files/wandb-metadata.json')
    shutil.copyfile(p,'/tmp/ch4_stats_copy.wandb')
    SOURCES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    ds=DataStore(); ds.open_for_scan('/tmp/ch4_stats_copy.wandb'); samples=[]
    while True:
        data=ds.scan_data()
        if data is None: break
        record=pb.Record(); record.ParseFromString(data)
        if record.WhichOneof('record_type')=='stats':
            for item in record.stats.item:
                if item.key==f'gpu.{device}.memoryAllocatedBytes':
                    samples.append((record.stats.timestamp.seconds+record.stats.timestamp.nanos/1e9,json.loads(item.value_json)))
    assert samples
    telemetry.append(dict(run=name,device_index=device,device_uuid=meta['gpu_nvidia'][device]['uuid'],
        samples=len(samples),sampled_max_gib=max(v for t,v in samples)/2**30,
        median_interval_s=st.median(b[0]-a[0] for a,b in zip(samples,samples[1:])),
        first_timestamp=samples[0][0],last_timestamp=samples[-1][0],source=str(p),
        scope='Assigned physical device memory sampled across the tracked job, including training and final evaluation; not process-only or allocator peak.'))

audit=dict(data=D,historical_medium_rtx=historical,zero_shot=zero,telemetry=telemetry,
           sources_sha256=SOURCES,missing_local_predictions=missing,manifest_hashes=HASHES)

# Render only marked numeric blocks; prose requires review after new results.
import csv, re
HERE=ROOT/'documents/thesis'
def mean(model,code,key): return st.mean(r[key] for r in D[model][code])
def cell(model,code,key):
    rows=D[model][code]; vals=[r[key] for r in rows]
    if code in ['C2','C3']:
        mark='E' if model=='Base' and code=='C3' and key.startswith('fleurs_') else 'D'
        return f'${st.mean(vals):.2f}^{{\\mathrm{{{mark}}}}}$'
    if len(vals)==1: return f'${vals[0]:.2f}^{{\\mathrm{{S}}}}$'
    return f'${st.mean(vals):.2f} \\pm {st.stdev(vals):.2f}$'
def row(fields):return ' & '.join(fields)+r' \\'
tables={}
tables['migration']='\n'.join(row([r['condition'],str(r['seed']),f"{r['test_wer']:.2f}",
    f"{new['test_wer']:.2f}",f"{r['validation_wer']-new['validation_wer']:+.2f}",
    f"{r['test_wer']-new['test_wer']:+.2f}",
    '--' if r['runtime_min'] is None else f"{r['runtime_min']:.2f}",f"{new['runtime_min']:.2f}"])
    for r in historical for new in D['Medium'][r['condition']] if new['seed']==r['seed'])
zero_rows=[]
for corpus,prefix in [('WAXAL',''),('FLEURS','fleurs_')]:
    for model in ['Base','Medium']:
        z=zero[model]['splits']; keys=[prefix+'validation',prefix+'test']
        vals=[z[keys[0]]['wer'],z[keys[1]]['wer'],z[keys[1]]['cer']] if all(k in z for k in keys) else None
        zero_rows.append(row([corpus,model,'Zero-shot']+([f'{v:.2f}' for v in vals] if vals else [r'\textit{Pending}']*3)+['--']))
        fine=[mean(model,'C0',keys[0]+'_wer'),mean(model,'C0',keys[1]+'_wer'),mean(model,'C0',keys[1]+'_cer')]
        zero_rows.append(row([corpus,model,'Fine-tuned']+[f'{v:.2f}' for v in fine]+([f'{vals[1]-fine[1]:.2f}'] if vals else [r'\textit{Pending}'])))
tables['zero']='\n'.join(zero_rows)
for title,prefix in [('waxal',''),('fleurs','fleurs_')]:
    tables[title]='\n'.join(row([code]+[cell(model,code,prefix+key) for model in ['Base','Medium']
        for key in ['validation_wer','test_wer','test_cer']]) for code in COMMON)
tables['screen']='\n'.join(row([c]+[cell('Base',c,k) for k in ['validation_wer','test_wer','test_cer','fleurs_test_wer']])
    for c in ['C2','C4','C4R','C5','C6','C7'])
uncertainty_path=HERE/'chapter4_uncertainty_codex.json'
uncertainty=read(uncertainty_path)
intervals={(r['split'],r['treatment'],r['control']):r for r in uncertainty['results']}
def effect_cell(model,treatment,control,key):
    value=mean(model,control,key)-mean(model,treatment,key)
    if model=='Base':return f'${value:+.2f}$'
    result=intervals[(key.removesuffix('_wer'),treatment,control)]
    assert abs(result['improvement_pp']-value)<1e-10
    lo,hi=result['ci95_pp']
    return f'${value:+.2f}\\;[{lo:+.2f}, {hi:+.2f}]$'
tables['effects']='\n'.join(row([f'{t} versus {c}']+[effect_cell(m,t,c,k)
    for k in ['test_wer','fleurs_test_wer'] for m in ['Base','Medium']])
    for t,c in [('C1','C0'),('C3','C0'),('A2','C0'),('A3','C3'),('A3','A2'),
                ('S2','C0'),('S3','C1'),('S3','S2'),('S4','S2'),('S5','S3')])
def timing(model,code):
    if code not in D[model]: return '--'
    v=[r['runtime_min'] for r in D[model][code]]
    return f'${st.mean(v):.2f} \\pm {st.stdev(v):.2f}$'
tables['time']='\n'.join(row([c]+[timing(m,c) for m in ['Base','Medium']]) for c in COMMON+['C5','C6'])

chapter=HERE/'chapter4_codex.tex'
text=chapter.read_text()
for key,table in tables.items():
    text,n=re.subn(rf'(% BEGIN CODEX {key}\n).*?(% END CODEX {key})',lambda m:m[1]+table+'\n'+m[2],text,flags=re.S)
    assert n==1,(key,n)
chapter.write_text(text)
audit.update(tables=tables,reporting_date='2026-10-16',
    primary_run_count=sum(len(rows) for cells in D.values() for rows in cells.values()),
    historical_migration_runs=len(historical),
    cohort_rule='Base original RTX; Medium A100 C0 all seeds, A100 C1 42/43 plus original A100 C1 44; original A100 remaining cells.',
    medium_waxal_zero_shot_status='In progress according to the user; no completed summary at audit.',
    source_chapter='documents/thesis/chapter4_results.tex',
    source_chapter_sha256=hashlib.sha256((HERE/'chapter4_results.tex').read_bytes()).hexdigest(),
    builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(HERE/'chapter4_evidence_codex.json').write_text(json.dumps(audit,indent=2)+'\n')
rows=[r for cells in D.values() for group in cells.values() for r in group]+historical
keys=sorted(set().union(*(r.keys() for r in rows))-{'epoch_evaluations','prediction_hashes'})
with (HERE/'chapter4_per_run_codex.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
print(f'Checked {len(rows)} run records, including {len(historical)} historical migration runs; rendered {len(tables)} tables; read {len(SOURCES)} sources.')
