import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "experiments/fleurs_remaining_queue.json"
OUTPUT_ROOT = Path("/ext_data/casper/asr_experiment_outputs")
ARTIFACT_ROOT = ROOT / "artifacts/experiment_outputs"
REQUIRED = (
    "model.safetensors", "config.json", "generation_config.json",
    "preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "normalizer.json", "experiment_config.json",
    "trainer_state.json", "run_manifest.json", "train_results.json",
    "validation_results.json", "test_results.json", "item_predictions/summary.json",
    "item_predictions/validation.jsonl", "item_predictions/test.jsonl",
)
OPTIONAL = ("added_tokens.json", "tokenizer.json", "all_results.json")


def command(arguments, **kwargs):
    return subprocess.run(
        arguments, check=True, text=True, env={**os.environ,
        "CLOUDSDK_CORE_DISABLE_PROMPTS": "1", "CLOUDSDK_STORAGE_PROCESS_COUNT": "2",
        "CLOUDSDK_STORAGE_THREAD_COUNT": "4"}, **kwargs
    )


def capture(arguments):
    return command(arguments, stdout=subprocess.PIPE).stdout


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def entries(queue):
    medium = [(output, group["gcs_root"]) for group in
              queue["medium_primary"]["archive_groups"] for output in group["outputs"]]
    base = [(output, queue["base_primary"]["gcs_root"])
            for output in queue["base_primary"]["outputs"]]
    result = medium + base
    if len(medium) != 11 or len(base) != 14 or len(set(output for output, _ in result)) != 25:
        raise ValueError("Expected exactly 11 Medium and 14 unique Base runs")
    for output, prefix in result:
        if Path(output).is_absolute() or ".." in Path(output).parts or not prefix.startswith("gs://casper/"):
            raise ValueError(f"Unsafe archive entry: {output}")
    return result


def gpu_inventory():
    rows = capture(["nvidia-smi", "--query-gpu=uuid,name,memory.free",
                    "--format=csv,noheader,nounits"]).splitlines()
    devices = []
    for row in rows[:2]:
        uuid, name, memory = [field.strip() for field in row.split(",")]
        if name != "NVIDIA A100-SXM4-40GB" or int(memory) < 20000:
            raise ValueError(f"GPU unsuitable for evaluation: {row}")
        devices.append(uuid)
    if len(devices) != 2:
        raise ValueError("Two A100 evaluation GPUs are required")
    return devices


def require_idle(devices):
    busy = capture(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                    "--format=csv,noheader"]).splitlines()
    if set(devices).intersection(row.strip() for row in busy):
        raise ValueError("An assigned GPU already has an active CUDA process")


def check_protocol(protocol):
    for split in ("validation", "test"):
        manifest = Path(protocol["root"]) / f"{split}.jsonl"
        if digest(manifest) != protocol[f"{split}_sha256"]:
            raise ValueError(f"Changed FLEURS manifest: {split}")
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if len(rows) != protocol[f"{split}_rows"] or not all(Path(row["audio_filepath"]).is_file() for row in rows):
            raise ValueError(f"Missing FLEURS audio or unexpected rows: {split}")


def package(output, archive_root, protocol):
    model_dir = OUTPUT_ROOT / output
    staging = model_dir.with_name(model_dir.name + ".fleurs-restore")
    artifact_dir = ARTIFACT_ROOT / output
    if model_dir.exists() or staging.exists() or (artifact_dir / "fleurs_corrected_v2").exists():
        raise ValueError(f"Existing or partial destination; inspect before proceeding: {output}")
    config_path = ROOT / "experiments" / output / "config.json"
    config = read_json(config_path)
    archived_config = read_json(artifact_dir / "experiment_config.json")
    for section in ("model", "data", "training", "curriculum", "augmentation"):
        if config.get(section) != archived_config.get(section):
            raise ValueError(f"Scientific config differs from archived run: {output}/{section}")
    for key in ("per_device_eval_batch_size", "fp16", "generation_max_length"):
        if config["training"][key] != protocol[key]:
            raise ValueError(f"Inference contract differs: {output}/{key}")
    for key in ("language", "task", "clear_forced_decoder_ids", "clear_suppress_tokens"):
        if config["model"][key] != protocol[key]:
            raise ValueError(f"Model inference contract differs: {output}/{key}")
    prefix = f"{archive_root.rstrip('/')}/{output}/"
    listing = json.loads(capture(["gcloud", "storage", "ls", "--json", prefix + "**"]))
    objects = {}
    for item in listing:
        metadata = item.get("metadata", {})
        if "bucket" not in metadata or "name" not in metadata:
            continue
        url = f"gs://{metadata['bucket']}/{metadata['name']}"
        if url.startswith(prefix):
            objects[url[len(prefix):]] = {
                "url": url + "#" + str(metadata["generation"]),
                "size": int(metadata["size"]), "crc32c": metadata.get("crc32c"),
            }
    for name in REQUIRED:
        if name not in objects or not objects[name]["crc32c"] or objects[name]["size"] <= 0:
            raise ValueError(f"Missing archive file/checksum: {output}/{name}")
    state = read_json(artifact_dir / "trainer_state.json")
    best = Path(state["best_model_checkpoint"]).name
    checkpoint = objects.get(f"{best}/model.safetensors")
    if not checkpoint or any(objects["model.safetensors"][key] != checkpoint[key] for key in ("size", "crc32c")):
        raise ValueError(f"Root weights do not match recorded best checkpoint: {output}/{best}")
    cloud_state = json.loads(capture(["gcloud", "storage", "cat", objects["trainer_state.json"]["url"]]))
    if cloud_state != state:
        raise ValueError(f"Cloud/local checkpoint provenance differs: {output}")
    selected = {name: objects[name] for name in (*REQUIRED, *OPTIONAL) if name in objects}
    print(f"Verified package and best checkpoint: {output} ({best})", flush=True)
    return {"output": output, "config_sha256": digest(config_path),
            "best_checkpoint": best, "files": selected}


def preflight():
    queue = read_json(QUEUE)
    check_protocol(queue["protocol"])
    devices = gpu_inventory()
    require_idle(devices)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(package, output, archive, queue["protocol"])
                   for output, archive in entries(queue)]
        runs = [future.result() for future in futures]
    size = sum(metadata["size"] for run in runs for metadata in run["files"].values())
    if shutil.disk_usage(OUTPUT_ROOT.parent).free < size + 20 * 1024**3:
        raise ValueError("Insufficient space for inference packages plus 20 GiB reserve")
    return {"queue_sha256": digest(QUEUE), "devices": devices, "protocol": queue["protocol"],
            "restore_bytes": size, "runs": runs}


def checksum_values(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower().replace("_", "").replace("-", "") in ("crc32c", "crc32chash"):
                yield item
            else:
                yield from checksum_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from checksum_values(item)


def verify_file(path, metadata):
    if path.stat().st_size != metadata["size"]:
        raise ValueError(f"Restored size mismatch: {path}")
    hashes = json.loads(capture(["gcloud", "storage", "hash", "--skip-md5", "--format=json", str(path)]))
    if metadata["crc32c"] not in list(checksum_values(hashes)):
        raise ValueError(f"Restored checksum mismatch: {path}")


def verify_results(directory, run, protocol):
    summary = read_json(directory / "summary.json")
    config = read_json(ROOT / "experiments" / run["output"] / "config.json")
    if summary["experiment_id"] != config["experiment_id"] or summary["config_sha256"] != run["config_sha256"]:
        raise ValueError("Prediction experiment/config identity mismatch")
    if Path(summary["model_dir"]) != OUTPUT_ROOT / run["output"]:
        raise ValueError("Prediction model directory mismatch")
    if set(summary["splits"]) != {"fleurs_validation", "fleurs_test"}:
        raise ValueError("Unexpected prediction splits")
    for split in ("validation", "test"):
        name = f"fleurs_{split}"
        record = summary["splits"][name]
        predictions = directory / f"{name}.jsonl"
        if record["rows"] != protocol[f"{split}_rows"] or record["manifest_sha256"] != protocol[f"{split}_sha256"]:
            raise ValueError(f"Prediction protocol mismatch: {name}")
        if digest(predictions) != record["predictions_sha256"] or len(predictions.read_text().splitlines()) != record["rows"]:
            raise ValueError(f"Prediction hash/row mismatch: {name}")


def run_lane(lane, plan, work_dir):
    device = plan["devices"][lane]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": device,
                   "HF_HOME": "/ext_data/casper/huggingface_cache", "PYTHONDONTWRITEBYTECODE": "1"}
    python = ROOT / ".venv/bin/python"
    for run in plan["runs"][lane::2]:
        output = run["output"]
        print(f"GPU {lane}: restoring {output}", flush=True)
        model_dir = OUTPUT_ROOT / output
        staging = model_dir.with_name(model_dir.name + ".fleurs-restore")
        if model_dir.exists() or staging.exists():
            raise ValueError(f"Destination appeared after preflight: {output}")
        if digest(ROOT / "experiments" / output / "config.json") != run["config_sha256"]:
            raise ValueError(f"Config changed after preflight: {output}")
        staging.mkdir(parents=True)
        for name, metadata in run["files"].items():
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            command(["gcloud", "storage", "cp", metadata["url"], str(destination), "--quiet"])
            verify_file(destination, metadata)
        for name in ("experiment_config.json", "trainer_state.json", "run_manifest.json", "item_predictions/summary.json"):
            if read_json(staging / name) != read_json(ARTIFACT_ROOT / output / name):
                raise ValueError(f"Restored/local provenance mismatch: {output}/{name}")
        staging.rename(model_dir)
        require_idle([device])
        logs = model_dir / "logs"
        logs.mkdir(exist_ok=True)
        prediction_dir = model_dir / "fleurs_corrected_v2"
        arguments = [str(python), "-u", str(ROOT / "asr.py"), "evaluate_predictions",
                     "--config", str(ROOT / "experiments" / output / "config.json"),
                     "--model-dir", str(model_dir), "--output-dir", str(prediction_dir)]
        for split in ("validation", "test"):
            arguments += ["--manifest", f"fleurs_{split}={plan['protocol']['root']}/{split}.jsonl"]
        print(f"GPU {lane}: evaluating {output}", flush=True)
        with (logs / "fleurs_corrected_v2.log").open("x") as log:
            subprocess.run(arguments, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
        verify_results(prediction_dir, run, plan["protocol"])
        with (logs / "fleurs_mirror.log").open("x") as log:
            subprocess.run([str(python), str(ROOT / "asr.py"), "mirror_experiment_artifacts",
                            "--source-dir", str(model_dir), "--output-dir", str(ARTIFACT_ROOT / output),
                            "--refresh"], check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
        verify_results(ARTIFACT_ROOT / output / "fleurs_corrected_v2", run, plan["protocol"])
        write_json(work_dir / "completed" / f"{output.replace('/', '__')}.json", {"output": output, "status": "completed_and_mirrored"})
        print(f"GPU {lane}: completed and mirrored {output}", flush=True)


def controller(work_dir):
    plan = read_json(work_dir / "plan.json")
    if digest(QUEUE) != plan["queue_sha256"]:
        raise ValueError("Queue changed after preflight")
    locks = []
    for device in plan["devices"]:
        lock = (ARTIFACT_ROOT / "fleurs_remaining" / f"{device}.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    if gpu_inventory() != plan["devices"]:
        raise ValueError("GPU mapping changed after preflight")
    require_idle(plan["devices"])
    check_protocol(plan["protocol"])
    write_json(work_dir / "status.json", {"status": "running", "runs": len(plan["runs"])})
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_lane, lane, plan, work_dir) for lane in (0, 1)]
        for future in futures:
            try:
                future.result()
            except Exception as error:
                errors.append(str(error))
                print(f"LANE FAILED: {error}", flush=True)
    write_json(work_dir / "status.json", {"status": "failed" if errors else "completed", "errors": errors})
    if errors:
        raise RuntimeError("Evaluation lane failed; inspect preserved outputs and logs")


def main():
    parser = argparse.ArgumentParser(description="Restore and evaluate the prepared primary FLEURS backlog")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--controller", action="store_true")
    parser.add_argument("--work-dir", type=Path)
    arguments = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work_dir = arguments.work_dir or ARTIFACT_ROOT / "fleurs_remaining" / stamp
    if arguments.controller:
        controller(work_dir)
        return
    if work_dir.exists():
        raise ValueError(f"Work directory already exists: {work_dir}")
    plan = preflight()
    write_json(work_dir / "plan.json", plan)
    print(f"Preflight passed: {len(plan['runs'])} runs, {plan['restore_bytes'] / 1024**3:.2f} GiB; {work_dir}", flush=True)
    if arguments.preflight_only:
        return
    session = f"fleurs-remaining-{work_dir.name}"
    arguments = [str(ROOT / ".venv/bin/python"), "-u", str(Path(__file__).resolve()),
                 "--controller", "--work-dir", str(work_dir)]
    shell_command = f"set -o pipefail; {shlex.join(arguments)} 2>&1 | tee -a {shlex.quote(str(work_dir / 'controller.log'))}"
    command(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT), shell_command])
    command(["tmux", "set-window-option", "-t", f"={session}", "remain-on-exit", "on"])
    print(f"Launched {session}; log: {work_dir / 'controller.log'}", flush=True)


if __name__ == "__main__":
    main()