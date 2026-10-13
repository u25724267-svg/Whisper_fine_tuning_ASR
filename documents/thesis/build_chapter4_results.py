"""Rebuild Chapter 4 numeric tables from local, completed experiment artifacts.

Run from any directory with Python 3 (standard library only). Narrative is not
rewritten; inspect it after any change in the evidence. No training is performed.
"""
import csv
import hashlib
import json
import re
import statistics as stats
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUTS = ROOT / "artifacts/experiment_outputs"
SOURCES = {}


def read(path):
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


GROUPS = {
    "C0": ("rq1", "c0_random", [42, 43, 44]),
    "C1": ("rq1", "c1_sortagrad", [42, 43, 44]),
    "C2": ("rq1", "c2_snr", [42, 43, 44]),
    "C3": ("rq1", "c3_snr_duration", [42, 43, 44]),
    "C4": ("rq1", "c4_acoustic_cumulative", [42]),
    "C4R": ("rq1", "c4r_random_pacing", [42]),
    "C5": ("rq1", "c5_s2s_margin", [42, 43, 44]),
    "C6": ("rq1", "c6_wer_margin", [42, 43, 44]),
    "C7": ("rq1", "c7_acoustic_s2s_hybrid", [42]),
    "A2": ("rq2", "a2_waveform_random", [42, 43, 44]),
    "A3": ("rq2", "a3_waveform_c3", [42, 43, 44]),
    "S2": ("rq2", "s2_specaug_lb_random", [42, 43, 44]),
    "S3": ("rq2", "s3_specaug_lb_sortagrad", [42, 43, 44]),
    "S4": ("rq2", "s4_specaug_ld_random", [42, 43, 44]),
    "S5": ("rq2", "s5_specaug_ld_sortagrad", [42, 43, 44]),
    "M0": ("medium/rq1", "c0_random", [42, 43, 44]),
    "M1": ("medium/rq1", "c1_sortagrad", [42]),
}
NAMES = {
    "C0": "Random", "C1": "SortaGrad", "C2": "Strict SNR",
    "C3": "SNR--duration", "C4": "Acoustic pacing", "C4R": "Random pacing",
    "C5": "S2S loss", "C6": "Utterance WER", "C7": "Acoustic--S2S",
    "A2": "Waveform, random", "A3": "Waveform, C3",
    "S2": "LB, random", "S3": "LB, SortaGrad",
    "S4": "LD, random", "S5": "LD, SortaGrad",
    "M0": "Medium, random", "M1": "Medium, SortaGrad",
}
DATA = {}
ROWS = []
for code, (family, name, seeds) in GROUPS.items():
    DATA[code] = []
    for seed in seeds:
        suffix = "_v2" if name == "c0_random" else ""
        directory = OUTPUTS / family / f"{name}_seed{seed}{suffix}"
        summary = read(directory / "item_predictions/summary.json")
        result = read(directory / "all_results.json")
        manifest = read(directory / "run_manifest.json")
        assert manifest["effective_seed"] == seed
        assert result["epoch"] == 3.0
        assert manifest["runtime"]["gpu"] == "NVIDIA GeForce RTX 4090"
        row = {"condition": code, "seed": seed, "source": str(directory.relative_to(ROOT)),
               "runtime_minutes": result["train_runtime"] / 60,
               "validation_loss": result["validation_loss"], "train_loss": result["train_loss"]}
        for split, expected_rows in [("validation", 1715), ("test", 1671)]:
            metric = summary["splits"][split]
            assert metric["rows"] == expected_rows
            assert metric["manifest_sha256"] == manifest["sha256"][f"{split}_manifest"]
            recomputed = 100 * sum(metric[k] for k in ("substitutions", "deletions", "insertions")) / metric["reference_words"]
            assert abs(recomputed - metric["wer"]) < 1e-10
            for key in ("wer", "cer", "predictions_sha256"):
                row[f"{split}_{key}"] = metric[key]
        fleurs = directory / "fleurs_corrected_v2/summary.json"
        if fleurs.exists():
            for split, metric in read(fleurs)["splits"].items():
                for key in ("wer", "cer"):
                    row[f"{split}_{key}"] = metric[key]
        state = directory / "trainer_state.json"
        if state.exists():
            history = read(state)
            row["best_checkpoint"] = (history.get("best_model_checkpoint") or "").split("/")[-1]
            row["epoch_evaluations"] = [x for x in history["log_history"] if "eval_wer" in x]
        DATA[code].append(row)
        ROWS.append(row)


# Verify the deterministic-prediction qualification and retain ancillary sources
# for manually typeset diagnostic/zero-shot tables and the timing caveat.
for code in ("C2", "C3"):
    for split in ("validation", "test"):
        assert len({r[f"{split}_predictions_sha256"] for r in DATA[code]}) == 1
ANCILLARY = [
    "baselines/whisper_base_zero_shot_waxal_fleurs_v1/summary.json",
    "rq1/c0_random_nondisjoint_seed42/item_predictions/summary.json",
    "rq1/c0_random_seed42/item_predictions/summary.json",
    "rq1/c5_s2s_margin_seed42_rerun/train_results.json",
    "medium/diagnostics/memory_clean_batch6_20261006.json",
    "medium/diagnostics/memory_clean_batch6_20261006_v2.json",
    "medium/diagnostics/memory_ld_batch6_20261006.json",
    "rq2/aggregate/rq2_factorial_summary.json",
    "rq2/fleurs_aggregate/rq2_fleurs_factorial_summary.json",
]
for relative in ANCILLARY:
    read(OUTPUTS / relative)
for code in ("S2", "S4"):
    family, name, _ = GROUPS[code]
    read(ROOT / "experiments" / family / f"{name}_seed42" / "config.json")


def values(code, key):
    return [x[key] for x in DATA[code] if key in x]


def cell(code, key, precision=2):
    vals = values(code, key)
    mean = stats.mean(vals)
    # Prediction-identical deterministic repeats do not estimate seed variance.
    if len(vals) == 1 or (code in ("C2", "C3") and key != "runtime_minutes"):
        return f"${mean:.{precision}f}$"
    return f"${mean:.{precision}f} \\pm {stats.stdev(vals):.{precision}f}$"


def metric_rows(codes, external=False):
    keys = [f"{'fleurs_' if external else ''}{split}_{metric}"
            for split, metric in [("validation", "wer"), ("test", "wer"), ("validation", "cer"), ("test", "cer")]]
    return "\n".join(f"{code}: {NAMES[code]} & " + " & ".join(cell(code, key) for key in keys) + r" \\" for code in codes)


TABLES = {
    "rq1": metric_rows(list(GROUPS)[:9]),
    "waveform": metric_rows(["C0", "C3", "A2", "A3"]),
    "specaugment": metric_rows(["C0", "C1", "S2", "S3", "S4", "S5"]),
    "fleurs": metric_rows(["C0", "C1", "C3", "A2", "A3", "S2", "S3", "S4", "S5"], True),
    "medium": metric_rows(["C0", "M0", "M1"]),
    "resources": "\n".join(f"{c}: {NAMES[c]} & {len(DATA[c])} & {cell(c, 'runtime_minutes')} & {cell(c, 'validation_loss', 4)} \\\\" for c in GROUPS),
}
screen_rows = []
for c in list(GROUPS)[1:9]:
    fields = []
    for split in ["validation", "test"]:
        d = read(OUTPUTS / "rq1/comparisons" / f"{c.lower()}_vs_c0_{split}.json")
        lo, hi = d["confidence_interval_95"]
        fields.append(f"${d['observed_improvement']:+.2f}\\;[{lo:+.2f}, {hi:+.2f}]$")
    screen_rows.append(f"{c}: {NAMES[c]} & " + " & ".join(fields) + r" \\")
TABLES["screen"] = "\n".join(screen_rows)
curve_rows = []
for c in ["C0", "C1", "C5", "C6", "A2", "S2", "S4", "M0", "M1"]:
    seed = 43 if c == "C1" else 42
    row = next(r for r in DATA[c] if r["seed"] == seed)
    history = row["epoch_evaluations"]
    assert len(history) == 3
    curve_rows.append(f"{c}, seed {seed} & " + " & ".join(f"${h['eval_wer']:.2f}/{h['eval_loss']:.4f}$" for h in history) + r" \\")
TABLES["curves"] = "\n".join(curve_rows)
TABLES["perseed"] = "\n".join(f"{r['condition']} & {r['seed']} & " + " & ".join(f"{r[k]:.2f}" for k in ["validation_wer", "test_wer", "validation_cer", "test_cer"]) + r" \\" for r in ROWS)

tex = HERE / "chapter4_results_revised.tex"
if tex.exists():
    text = tex.read_text()
    for key, table in TABLES.items():
        pattern = rf"(% BEGIN GENERATED {key}\n).*?(% END GENERATED {key})"
        text, count = re.subn(pattern, lambda m: m[1] + table + "\n" + m[2], text, flags=re.S)
        assert count == 1, (key, count)
    tex.write_text(text)

keys = sorted(set().union(*(r.keys() for r in ROWS)) - {"epoch_evaluations"})
with (HERE / "chapter4_results_per_run.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(ROWS)
evidence = {"reporting_date": "2026-10-13", "sources_sha256": SOURCES,
            "runs": ROWS, "tables": TABLES,
            "warning": "Prediction-identical C2/C3 repeats are not independent accuracy replications. C3 FLEURS has one export. Narrative must be rechecked after regeneration."}
(HERE / "chapter4_results_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
print(f"Verified {len(ROWS)} completed run records; generated {len(TABLES)} tables from {len(SOURCES)} source files.")
