"""Check exact staged Git blobs without printing matched private values."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


BLOCKED_PARTS = {
    '.openai', '.local', '.runtime', '.training', '.private', 'private', '.venv',
    'venv', 'node_modules', '__pycache__', 'data', 'checkpoints', 'recordings',
    'artifacts', 'test-artifacts', 'models', 'bundles',
}
BLOCKED_SUFFIXES = {
    '.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.npz', '.npy', '.pkl',
    '.pickle', '.jsonl', '.sqlite', '.sqlite3', '.db', '.key', '.pem', '.p12',
    '.pfx', '.log', '.pyc', '.zip', '.tgz', '.pid',
}
TEXT_SUFFIXES = {'.py', '.js', '.mjs', '.json', '.html', '.css', '.md', '.txt', '.yaml', '.yml', '.toml', '.example'}
PATTERNS = {
    'private-key': re.compile(rb'-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----'),
    'github-token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{30,})'),
    'aws-access-key': re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'slack-token': re.compile(rb'\bxox[baprs]-[A-Za-z0-9-]{20,}'),
    'personal-home-path': re.compile(rb'(?:[A-Za-z]:[/\\](?:Users|Codex|Documents)[/\\][^\s\x22\x27]+|/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/)'),
    'private-network-address': re.compile(rb'\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b'),
}
EMAIL = re.compile(rb'[A-Za-z0-9.!#$%&\x27*+/=?^_`{|}~-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)')


def reserved_email(domain):
    domain = domain.lower()
    return (domain in (b'example.com', b'example.org', b'example.net', b'users.noreply.github.com')
            or domain.endswith((b'.example.com', b'.example.org', b'.example.net', b'.test', b'.invalid')))


def staged_blobs(root):
    listing = subprocess.run(['git', 'ls-files', '--stage', '-z'], cwd=root, check=True, capture_output=True).stdout
    entries = []
    for record in listing.split(b'\0'):
        if not record:
            continue
        metadata, name = record.split(b'\t', 1)
        mode, object_id, stage = metadata.split()
        if stage != b'0':
            raise ValueError('Resolve the Git index before checking public source.')
        entries.append((name.decode('utf8'), mode.decode(), object_id.decode()))
    if not entries:
        raise ValueError('The Git index is empty. Stage the intended source files first.')
    request = ''.join(object_id + '\n' for _, _, object_id in entries).encode()
    result = subprocess.run(['git', 'cat-file', '--batch'], cwd=root, input=request, check=True, capture_output=True)
    stream = io.BytesIO(result.stdout)
    for name, mode, object_id in entries:
        header = stream.readline().decode().split()
        if len(header) != 3 or header[0] != object_id or header[1] != 'blob':
            raise ValueError('The Git index contains an unsupported object.')
        raw = stream.read(int(header[2]))
        if stream.read(1) != b'\n':
            raise ValueError('Invalid Git blob stream.')
        yield name, mode, raw


def check(root):
    findings, files = [], 0
    for name, mode, raw in staged_blobs(root):
        files += 1
        path = PurePosixPath(name)
        def add(category, offset=0):
            findings.append({'path': name, 'line': raw[:offset].count(b'\n') + 1, 'category': category})
        if mode != '100644' and mode != '100755':
            add('symlink-or-submodule')
        if (set(path.parts) & BLOCKED_PARTS or path.suffix.lower() in BLOCKED_SUFFIXES
                or '.sqlite' in path.name or path.name.startswith('.env') and path.name != '.env.example'
                or path.name in ('smtp.json', 'credentials.json', 'share.json', 'tunnel-endpoint.json')):
            add('excluded-runtime-or-private-artifact')
        if len(raw) > 2 * 1024 * 1024:
            add('unexpected-large-file')
        # Preserve upstream attribution, including original copyright contacts.
        if name.startswith('dist/vendor/'):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in ('.gitignore', '.env.example'):
            continue
        for category, expression in PATTERNS.items():
            for match in expression.finditer(raw):
                add(category, match.start())
        for match in EMAIL.finditer(raw):
            if not reserved_email(match.group(1)):
                add('non-example-email', match.start())
    return {'passed': not findings, 'stagedFiles': files, 'findings': findings,
            'scope': 'Exact Git index blobs; run a dedicated secret scanner and review before publishing.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    try:
        report = check(arguments.repo)
    except (ValueError, subprocess.CalledProcessError) as error:
        print(json.dumps({'passed': False, 'error': str(error)}))
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
