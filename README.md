# Whisper fine-tuning for ASR
In this Repo, you can easily fine-tune different variations of the Whisper model to your specific multilingual data based on a simple manifest. 

1. [prepare_data.py](prepare_data.py)     :::: to prepare ".csv" files for train and test
2. [train.py](train.py)                   :::: train and save the fine-tuned Whisper model
3. [decode.py](decode.py)                 :::: decode the test or any evaluation ".wav" file
4. [whisper_transcribe_WER.py](whisper_transcribe_WER.py) ::: another (easier) method for utilizing the Whisper model in transcription.

*** You can use different versions of the [openai Whisper model](https://huggingface.co/openai/whisper-large-v2).


## Auxilary files

The tested training packages are listed in `requirements-training.txt`:
`pip install -r requirements-training.txt`

It would be better to make a new Python environment using `python3 -m venv myenv` , after that, activate the venv using `source myenv/bin/activate` and then install the packages.

To run on the servers by Slurm, you can use the [slurm_run.sh](slurm_run.sh) file.

The "files_test.csv" and "files_train.csv" help us understand better the required files for testing and training.

## Reproducible WAXAL experiments

The completed Whisper Base Shona setup is stored in
`configs/whisper-base-shona-3epochs.json`. It contains the model, manifests,
preprocessing, hyperparameters, output path, and non-secret W&B metadata.

Create the environment file once and add the W&B API key locally:

```bash
cp .env.example .env
```

Validate the configuration without training:

```bash
.venv/bin/python train_full.py \
	--config configs/whisper-base-shona-3epochs.json \
	--dry-run
```

Start or resume it in a connection-independent tmux session:

```bash
./run_full_detached.sh
```

The default output directory resumes its newest checkpoint. To rerun the same
configuration from scratch without deleting or overwriting the original run,
select a new output directory and session name:

```bash
WHISPER_OUTPUT_DIR=output_dir_whisper_base_shona_rerun_01 \
TMUX_SESSION_NAME=whisper-base-shona-rerun-01 \
./run_full_detached.sh
```

Select another experiment without editing Python:

```bash
WHISPER_CONFIG=configs/another-experiment.json \
TMUX_SESSION_NAME=another-experiment \
./run_full_detached.sh
```

The Whisper Medium config requires at least 20 GB of free GPU memory and will
refuse to start rather than risk an out-of-memory failure:

```bash
WHISPER_CONFIG=configs/whisper-medium-shona-3epochs-eval1000.json \
./run_full_detached.sh
```

After training completes, measure the corresponding pretrained model's
zero-shot performance on the same test manifest:

```bash
.venv/bin/python evaluate_zero_shot.py \
	--config configs/whisper-medium-shona-3epochs-eval1000.json
```

The evaluator reports raw and normalized WER for the full test split and for
the subset whose speakers never appear in training, and saves every prediction.

## Combined WAXAL and FLEURS Shona experiment

Prepare the pinned Google FLEURS `sn_zw` corpus, apply the same ASR text
normalization to FLEURS and WAXAL, and launch the reproducible Whisper Base run:

```bash
./run_waxal_fleurs.sh
```

The preparation step lowercases text, removes Unicode punctuation, numbers,
symbols, and control characters, and collapses whitespace. It preserves the
official train, validation, and test boundaries and writes corpus-specific and
combined manifests plus hashes and dataset provenance under
`/ext_data/casper/whisper_data/waxal_fleurs/sna_asr`.

The experiment uses the original Whisper Base three-epoch hyperparameters and
the existing `whisper-shona-multilingual` W&B project. Final metrics are saved
for the combined data and separately for WAXAL and FLEURS validation and test
splits.

Run the FLEURS-only Whisper Base control baseline with the established
1,000-step evaluation protocol:

```bash
./run_fleurs_only.sh
```
