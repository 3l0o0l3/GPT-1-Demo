# Third-party notices

The application source license does not replace the licenses of third-party
software or downloaded datasets. This repository does not include downloaded
training data, trained weights, experiment recordings, or visitor conversations.

## KaTeX

The browser assets in `dist/vendor/` include KaTeX 0.16.22 and its fonts for local
formula rendering. KaTeX is distributed under the MIT License, copyright
2013–2020 Khan Academy and other contributors. The included license notice is
preserved in [`dist/vendor/LICENSE`](dist/vendor/LICENSE).

Upstream: [KaTeX](https://github.com/KaTeX/KaTeX).

## TinyStories

The optional `training_data.py` preparation script downloads bounded training and
validation prefixes from the public
[roneneldan/TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories).
The upstream dataset card identifies the data license as
[Community Data License Agreement – Sharing 1.0](https://cdla.dev/sharing-1-0/).
The dataset contains model-generated stories and is described in
[TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759).

Preparation records the selected revision, source URLs, selection rules, and
hashes in the generated `data/manifest.json`; it saves the dataset card and
license snapshot under `data/licenses/`. These generated files are not bundled
with the source distribution. The six short UI prompts are teaching inputs for
inspection; their presence in a downloaded training subset is not asserted.

## UCI Sentiment Labelled Sentences

The same preparation script downloads
[Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment%2Blabelled%2Bsentences),
contributed by Dimitrios Kotzias (2015), from the UCI Machine Learning Repository.
Dataset DOI: [10.24432/C57604](https://doi.org/10.24432/C57604).
The publisher identifies the data license as
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

The script normalizes text for duplicate detection, removes duplicate or
conflicting groups before splitting, and creates deterministic training,
validation, and test splits. It records these transformations and saves the
publisher page and license text under the generated `data/` directory. The test
split is not used by the training demo. No original reviews or resulting dataset
files are distributed in this repository.

## Installed dependencies

PyTorch, NumPy, Hugging Face Tokenizers, and optional Segno are installed from
their respective distributions. Their own license and notice files remain part
of those distributions; dependency binaries are not vendored here. Node.js is
used to run the built-in JavaScript tests. The optional browser check requires a
separately installed Playwright package and browser.
