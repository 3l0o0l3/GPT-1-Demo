"""Export a trusted, explicitly selected live training save into a new bundle.

The input uses Python checkpoint deserialization and MUST be the owner's trusted
saved session, never an uploaded/public file. Deployment reads the resulting
weights-only format instead and does not need this converter or training data.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import torch

from seminar_model import CONFIG, FrozenLanguageModel, file_sha256, validate_checkpoint, validate_classifier


def export_bundle(session_path, tokenizer_path, output):
    session_path, tokenizer_path, output = Path(session_path), Path(tokenizer_path), Path(output)
    if output.exists():
        raise FileExistsError('Export destination must be a new directory.')
    # Deliberate only for this locally trusted training save, not runtime inputs.
    saved = torch.load(session_path, map_location='cpu', weights_only=False)
    if (not isinstance(saved, dict) or saved.get('liveFormatVersion') != 1 or
            saved.get('config') != CONFIG or not isinstance(saved.get('lm'), dict)):
        raise ValueError('Input is not a compatible saved live session.')
    lm = saved['lm']
    if (lm.get('formatVersion') != 1 or lm.get('config') != CONFIG or
            lm.get('vocabSize') != saved.get('vocabSize') or
            lm.get('dataManifestSha256') != saved.get('dataManifestSha256')):
        raise ValueError('Live save and LM provenance disagree.')
    tokenizer_hash = file_sha256(tokenizer_path)
    if tokenizer_hash != saved.get('tokenizerSha256'):
        raise ValueError('Tokenizer does not match the selected saved session.')
    checkpoint = {
        'formatVersion': 1, 'config': copy.deepcopy(CONFIG), 'vocabSize': saved['vocabSize'],
        'model': {key: value.detach().cpu().clone() for key, value in lm['model'].items()},
        'source': {key: lm[key] for key in ('step', 'seed', 'source')},
        'tokenizerSha256': tokenizer_hash, 'dataManifestSha256': saved['dataManifestSha256'],
        'savedAt': saved['savedAt'],
    }
    validate_checkpoint(checkpoint)
    # Never overwrite the live save or an existing deployment bundle.
    output.mkdir(parents=True, exist_ok=False)
    torch.save(checkpoint, output / 'model.pt')
    shutil.copyfile(tokenizer_path, output / 'tokenizer.json')
    model_hash = file_sha256(output / 'model.pt')
    manifest = {'bundleVersion': 1, 'snapshotId': model_hash,
                'files': {'model.pt': model_hash, 'tokenizer.json': tokenizer_hash}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return FrozenLanguageModel(output).metadata()


def attach_classifier_bundle(session_path, base_bundle, output):
    """Add the saved trained pair without changing the published story weights."""
    base_bundle, output = Path(base_bundle), Path(output)
    if output.exists():
        raise FileExistsError('Export destination must be a new directory.')
    base = FrozenLanguageModel(base_bundle).metadata()
    saved = torch.load(session_path, map_location='cpu', weights_only=False)
    if (not isinstance(saved, dict) or saved.get('liveFormatVersion') != 1
            or saved.get('config') != CONFIG or saved.get('vocabSize') != base['vocabSize']
            or saved.get('tokenizerSha256') != base['tokenizerSha256']):
        raise ValueError('Saved classifier and published tokenizer must be compatible.')
    pair = saved.get('transfer')
    if not isinstance(pair, dict) or not isinstance(pair.get('branches'), dict) or set(pair['branches']) != {'pretrained', 'scratch'}:
        raise ValueError('Save a trained live transfer pair before publishing.')
    if pair.get('optimizerFailure'):
        raise ValueError('Restart and train the failed transfer pair before publishing.')
    frozen = {key: copy.deepcopy(saved[key]) for key in
              ('config', 'vocabSize', 'tokenizerSha256', 'dataManifestSha256', 'savedAt')}
    frozen.update(formatVersion=1, source=copy.deepcopy(pair['source']), step=pair['step'],
                  runId=pair['runId'], branches={name: {key: tensor.detach().cpu().clone()
                    for key, tensor in branch['model'].items()} for name, branch in pair['branches'].items()})
    validate_classifier(frozen, base['vocabSize'], base['tokenizerSha256'])
    output.mkdir(parents=True, exist_ok=False)
    for name in ('model.pt', 'tokenizer.json'):
        shutil.copyfile(base_bundle / name, output / name)
    torch.save(frozen, output / 'classifier.pt')
    manifest = {'bundleVersion': 2, 'snapshotId': base['snapshotId'],
                'files': {name: file_sha256(output / name) for name in ('model.pt', 'tokenizer.json', 'classifier.pt')}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return FrozenLanguageModel(output).metadata()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, help='Trusted owner-created live session.pt')
    parser.add_argument('--tokenizer')
    parser.add_argument('--base-bundle', help='Attach saved classifier to this existing story bundle, preserving story weights')
    parser.add_argument('--output', required=True, help='A new bundle directory; never overwritten')
    arguments = parser.parse_args()
    if bool(arguments.tokenizer) == bool(arguments.base_bundle):
        parser.error('Choose --tokenizer for story export or --base-bundle to attach the classifier.')
    metadata = (attach_classifier_bundle(arguments.session, arguments.base_bundle, arguments.output)
                if arguments.base_bundle else export_bundle(arguments.session, arguments.tokenizer, arguments.output))
    print(json.dumps({'bundle': str(Path(arguments.output).resolve()), 'model': metadata}, indent=2))


if __name__ == '__main__':
    main()
