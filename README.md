# GPT-1 Training Demo

An interactive seminar demo of generative pre-training and supervised transfer.
The interface uses a small, real decoder-only Transformer: four layers, four
attention heads, width 128, a 128-token model context and 64-token training
windows. It is a teaching implementation, not a reproduction of GPT-1's scale.

The six sections are:

1. **Data & targets:** tokenization, shifted next-token targets, prediction
   probabilities, causal attention, and Q/K/V inspection.
2. **One update:** forward prediction, cross-entropy loss, backward gradients,
   clipping, an Adam update, and a before/after comparison. A preview restores
   the original state; Apply commits that exact update once.
3. **Convergence:** real background training, measured training/validation loss,
   progress and timing plots.
4. **Use trained weights:** streamed continuation with the current model,
   EOS stopping, optional sliding windows and explicit session saving.
5. **Transfer:** compare a randomly initialized classifier with a copy of the
   language model, using the same labeled sentiment data. "Random" describes
   initialization; that branch is trained on the classification task too.
6. **Audience discussion:** an optional separate audience service, published
   model snapshots and opt-in submissions for discussion.

## What is included

This repository contains source code, browser assets and synthetic test fixtures.
It contains **no trained weights, optimizer state, downloaded datasets, fitted
tokenizer, recorded experiment results, private conversations, login state,
mail credentials or personal server configuration**. Git history starts with
this sanitized source release. Training starts at random initialization, step 0.

Data preparation downloads public source data on your machine. Generated files
belong in ignored directories; do not add them to Git. Test fixtures use invented
texts, metrics and identifiers, and are not recordings of a seminar run.

## Quick start: local CPU training

Python 3.12 and a modern browser are recommended. Node.js is needed only to run
the JavaScript tests. The web assets are checked in; no frontend build is needed.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On Linux/macOS, activate it with `source .venv/bin/activate` instead.

Install the CPU dependencies and prepare a bounded public-data selection:

```text
python -m pip install -r requirements-training.txt --extra-index-url https://download.pytorch.org/whl/cpu
python training_data.py --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 --train-stories 2000 --validation-stories 128
python public_server.py
```

Open **http://127.0.0.1:8768/**. This entry point uses local training and does not
open SSH connections. The first measurements are from a newly initialized model,
not a downloaded checkpoint. Larger datasets can be prepared by increasing the
selection up to 50,000 training stories and 2,000 validation stories.

The short prompts in section 1 are inspection examples. Section 2 selects a real
example from the downloaded training split; it does not train on arbitrary text
entered into the inspection box. CPU training is slower than CUDA; start with
the one-update preview before requesting a long run.

## GPU training

Install a CUDA-enabled PyTorch build compatible with your GPU and driver in a
separate environment. `requirements-gpu.txt` records the CUDA training dependency
versions; install PyTorch from its matching official CUDA wheel index.

```text
python public_server.py --device cuda:0 --precision fp16 --compile
```

CUDA is explicit: an unavailable device is an error, not a silent CPU fallback.
FP16 uses automatic mixed precision and loss scaling; master parameters and Adam
state remain FP32. Inspection and evaluation keep their existing FP32 behavior.
Compilation is optional and its first warmup can take time.

Use **Save current model** to persist your own session under `.training/live/`.
Restarting the default entry point starts fresh; pass `--restore` to explicitly
restore your saved session. Keep checkpoints created by other people out of this
trusted-owner restoration path.

## Optional audience service

The audience service uses a published copy of the LM, separate from ongoing
training. Train first, save your session, then export a new bundle directory:

```text
python export_seminar_model.py --session .training/live/session.pt --tokenizer data/tokenizer.json --output .private/bundles/lm-1
```

If you trained and saved a Transfer pair, export it separately while preserving
the story model:

```text
python export_seminar_model.py --session .training/live/session.pt --base-bundle .private/bundles/lm-1 --output .private/bundles/with-classifier
```

For local audience testing, start the presenter on the fixed publication port:

```text
python public_server.py --port 18769 --restore --seminar-admin-port 18871
python seminar_server.py --public-port 18870 --admin-port 18871 --public-url http://127.0.0.1:18870/seminar-gpt/ --bundle .private/bundles/with-classifier --state .private/audience.sqlite3 --static audience --model-updates .private/models --training-export-url http://127.0.0.1:18769/api/training/audience-export
```

Run these in separate terminals. If no classifier was exported, use the `lm-1`
bundle instead. Open the presenter at http://127.0.0.1:18769/ and the audience page
at http://127.0.0.1:18870/seminar-gpt/.

In section 6, **Sync current LM** publishes the latest idle LM. Subsequent replies
use the new snapshot; existing replies retain their original step and model ID.
Compare the same prompt in a new conversation before and after training.
Classifier publication is independent and remains unchanged by an LM sync.
The static local URL does not create a temporary QR invitation. Temporary sharing
requires the optional `seminar_share.py` integration and a configured tunnel
endpoint, both owned and configured by the operator.

## Optional authenticated hosting

The loopback training entry point is for local use. For a hosted presenter, use
`presenter_server.py` behind your HTTPS reverse proxy, with your own public URL,
explicit email allowlist, private authentication state and SMTP profile. Keep
training and private audience-admin endpoints on loopback.

The mail profile is a private JSON file with exactly `username`, `password` and
`fromAddress`, supplied through `--smtp-config`; it is not part of this repository.
The sender uses Gmail STARTTLS. Give this file owner-only permissions on Linux.

```text
python presenter_server.py --bind 127.0.0.1 --port 18880 --public-url https://demo.example.com --static dist --state-dir .private/presenter --smtp-config .private/smtp.json --allowed-email presenter@example.com --training-port 18769
```

Replace example addresses and the URL with your own configuration. The login
allowlist is explicit, and public anonymous audience access remains separate from
presenter authentication. Optional legacy SSH integrations read the variables in
`.env.example`; the file is a template and is not automatically loaded.

## Tests

```text
npm test
python -m unittest discover -s tests -p "test_*.py"
python tools/check_public_source.py
```

Tests create tiny datasets, weights and databases in temporary directories. The
public-source check scans the Git index, including an initial commit, for excluded
artifacts and known personal configuration. It reports file/line/categories
without printing matched secret values. Run it again before each push.

## Data, method and attribution

- Story language modeling: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
  CDLA-Sharing-1.0. The preparation script records the selected revision and hashes.
- Supervised classification: [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment%2Blabelled%2Bsentences),
  CC-BY-4.0; DOI 10.24432/C57604. A sentence is paired with a positive/negative label.
- Language-model targets come from shifting text by one token; classification
  labels are provided separately. Validation/test data are excluded from tokenizer
  fitting, and test examples are not used by the interactive training loop.
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
  and [Attention Is All You Need](https://arxiv.org/abs/1706.03762) motivate the demo.
- Bundled KaTeX retains its MIT license at `dist/vendor/LICENSE`. See
  `THIRD_PARTY_NOTICES.md` for the data and component notices.

No project-wide license has been selected for the original application code.
Third-party components and datasets retain their own licenses.
