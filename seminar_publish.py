"""Private, explicit publication of immutable audience models.

The publisher owns its active provider and durable selection. It only pulls a
weights-only export from the fixed local training bridge. The application owns
the inference worker lock; publication uses that same lock for load and swap.
"""
from __future__ import annotations

import base64
import binascii
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import threading
import uuid

from seminar_store import SeminarError


EXPORT_URL = 'http://127.0.0.1:18769/api/training/audience-export'
MAX_EXPORT_BYTES = 16 * 1024 * 1024
FILE_LIMITS = {'model.pt': 12 * 1024 * 1024, 'tokenizer.json': 2 * 1024 * 1024}
POINTER_FIELDS = {'publicationVersion', 'bundle', 'releaseId', 'publishedAt',
                  'lastRequestId', 'expectedReleaseId'}


def _hex(value, length=64):
    return isinstance(value, str) and len(value) == length and all(c in '0123456789abcdef' for c in value)


def _request_id(value):
    try:
        return isinstance(value, str) and str(uuid.UUID(value)) == value
    except (ValueError, AttributeError):
        return False


def release_id(metadata):
    classifier = metadata.get('classifier', {})
    classifier_id = classifier.get('snapshotId') if classifier.get('available') is True else None
    if (not _hex(metadata.get('snapshotId'))
            or (classifier.get('available') is True and not _hex(classifier_id))):
        raise ValueError('The audience model has invalid release provenance.')
    value = {'snapshotId': metadata['snapshotId'], 'classifierSnapshotId': classifier_id}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def _fetch_export():
    """No redirects, retries, arbitrary hosts, or credentials."""
    connection = http.client.HTTPConnection('127.0.0.1', 18769, timeout=10)
    try:
        connection.request('GET', '/api/training/audience-export', headers={'Host': '127.0.0.1:18769'})
        response = connection.getresponse()
        raw = response.read(MAX_EXPORT_BYTES + 1)
        if len(raw) > MAX_EXPORT_BYTES:
            raise ValueError('The export response exceeds its size limit.')
        value = json.loads(raw)
        error = value.get('error') if isinstance(value, dict) else None
        if response.status == 409 and isinstance(error, dict) and error.get('code') == 'training_busy':
            raise SeminarError(409, 'training_busy', 'Training is running. Wait for it to finish before updating the audience model.')
        if response.status != 200:
            raise SeminarError(503, 'export_unavailable', 'The current training weights could not be exported. The audience model was not changed.')
        return value
    except SeminarError:
        raise
    except (OSError, ValueError, http.client.HTTPException) as error:
        raise SeminarError(503, 'export_unavailable', 'Cannot read the current training weights. The audience model was not changed.') from error
    finally:
        connection.close()


def _decode_export(value):
    if (not isinstance(value, dict) or set(value) != {'exportVersion', 'manifest', 'files'}
            or type(value['exportVersion']) is not int or value['exportVersion'] != 1):
        raise ValueError('Unsupported audience export.')
    manifest, files = value['manifest'], value['files']
    if (not isinstance(manifest, dict) or set(manifest) != {'bundleVersion', 'snapshotId', 'files'}
            or type(manifest['bundleVersion']) is not int or manifest['bundleVersion'] != 1
            or not isinstance(manifest['files'], dict) or set(manifest['files']) != set(FILE_LIMITS)
            or not isinstance(files, dict) or set(files) != set(FILE_LIMITS)
            or not all(_hex(digest) for digest in manifest['files'].values())
            or manifest['snapshotId'] != manifest['files']['model.pt']):
        raise ValueError('Invalid export manifest or file names.')
    decoded = {}
    for name, maximum in FILE_LIMITS.items():
        encoded = files[name]
        if not isinstance(encoded, str) or len(encoded) > 4 * ((maximum + 2) // 3):
            raise ValueError('An exported file exceeds its size limit.')
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError('An exported file is not valid base64.') from error
        if not data or len(data) > maximum or hashlib.sha256(data).hexdigest() != manifest['files'][name]:
            raise ValueError('An exported file is empty, oversized, or has a different hash.')
        decoded[name] = data
    if len(json.dumps(value, separators=(',', ':')).encode('utf-8')) > MAX_EXPORT_BYTES:
        raise ValueError('The export response exceeds its size limit.')
    return copy.deepcopy(manifest), decoded


@dataclass(frozen=True)
class _Active:
    provider: object
    bundle: Path
    metadata: dict
    release: str
    published_at: str | None = None
    request: str | None = None
    expected: str | None = None


class SeminarPublisher:
    """Public API: current_model, status(), describe(), sync(body, worker).

    ``loader(path)`` must return an immutable validated provider with metadata().
    ``fetcher()`` supplies the exact weights-only export object. These injectable
    boundaries permit independent CPU/HTTP tests without contacting live services.
    """

    def __init__(self, updates_dir, initial_bundle, export_url=None, *, loader=None,
                 fetcher=None, clock=None, worker_timeout=5):
        if export_url not in (None, EXPORT_URL):
            raise ValueError('Only the fixed loopback training export URL is supported.')
        if loader is None:
            from seminar_model import FrozenLanguageModel
            loader = lambda directory: FrozenLanguageModel(directory, threads=1)
        self.root = Path(updates_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.pointer = self.root / 'active.json'
        self.enabled = export_url is not None
        self.loader, self.fetcher = loader, fetcher or _fetch_export
        self.clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self.worker_timeout = worker_timeout
        self._publish_lock = threading.Lock()
        pointer = None
        bundle = Path(initial_bundle).resolve()
        if self.pointer.exists():
            if self.pointer.stat().st_size > 16384:
                raise ValueError('The audience publication pointer is too large.')
            pointer = json.loads(self.pointer.read_text(encoding='utf-8'))
            self._validate_pointer(pointer)
            bundle = self._bundle_path(pointer['bundle'])
        provider = loader(bundle)
        metadata = provider.metadata()
        identity = release_id(metadata)
        if pointer is not None and identity != pointer['releaseId']:
            raise ValueError('The restored audience release does not match its pointer.')
        self._active = _Active(provider, bundle, metadata, identity,
            pointer['publishedAt'] if pointer else None,
            pointer['lastRequestId'] if pointer else None,
            pointer['expectedReleaseId'] if pointer else None)

    def _bundle_path(self, name):
        if not _hex(name, 32):
            raise ValueError('Invalid controlled model directory.')
        path = (self.root / name).resolve()
        if path.parent != self.root:
            raise ValueError('The model directory is outside the publication root.')
        return path

    @staticmethod
    def _validate_pointer(value):
        if (not isinstance(value, dict) or set(value) != POINTER_FIELDS
                or type(value['publicationVersion']) is not int or value['publicationVersion'] != 1
                or not _hex(value['bundle'], 32) or not _hex(value['releaseId'])
                or not _hex(value['expectedReleaseId']) or not _request_id(value['lastRequestId'])
                or not isinstance(value['publishedAt'], str)):
            raise ValueError('Invalid audience publication pointer.')
        try:
            published = datetime.fromisoformat(value['publishedAt'])
        except ValueError as error:
            raise ValueError('Invalid publication timestamp.') from error
        if published.tzinfo is None:
            raise ValueError('Publication timestamp must include its timezone.')

    @property
    def current_model(self):
        return self._active.provider

    def _status(self, active):
        return {'enabled': self.enabled, 'releaseId': active.release,
                'publishedAt': active.published_at, 'lastRequestId': active.request}

    def status(self):
        return self._status(self._active)

    def describe(self):
        active = self._active
        return {'model': copy.deepcopy(active.metadata), 'modelPublication': self._status(active)}

    def _persist(self, value):
        """Everything fallible precedes replace; an uncommitted candidate is inert."""
        temporary = self.root / ('.active-' + uuid.uuid4().hex + '.tmp')
        with temporary.open('x', encoding='utf-8') as handle:
            json.dump(value, handle, sort_keys=True, separators=(',', ':'))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.pointer)
        # Replace is the commit point. Directory fsync is best-effort where the
        # platform supports it; do not report a committed swap as failed.
        self._sync_directory(self.root)

    @staticmethod
    def _sync_directory(directory):
        if os.name == 'nt':
            return
        try:
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass

    def _candidate(self, active):
        manifest, files = _decode_export(self.fetcher())
        candidate = self._bundle_path(uuid.uuid4().hex)
        candidate.mkdir(exist_ok=False)
        for name, data in files.items():
            (candidate / name).write_bytes(data)
        classifier = active.metadata.get('classifier', {})
        if classifier.get('available') is True:
            if manifest['files']['tokenizer.json'] != active.metadata.get('tokenizerSha256'):
                raise ValueError('The updated language model cannot use the published classifier tokenizer.')
            source = active.bundle / 'classifier.pt'
            if source.stat().st_size > 64 * 1024 * 1024:
                raise ValueError('The published classifier exceeds its size limit.')
            shutil.copyfile(source, candidate / 'classifier.pt')
            digest = hashlib.sha256((candidate / 'classifier.pt').read_bytes()).hexdigest()
            if digest != classifier['snapshotId']:
                raise ValueError('The published classifier changed on disk.')
            manifest['bundleVersion'] = 2
            manifest['files']['classifier.pt'] = digest
        (candidate / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True), encoding='utf-8')
        # The durable pointer must never select files still buffered by Python
        # or the OS. This work is outside the inference worker lock.
        for path in candidate.iterdir():
            with path.open('r+b') as handle:
                os.fsync(handle.fileno())
        self._sync_directory(candidate)
        return candidate

    def sync(self, body, worker):
        if (not isinstance(body, dict) or set(body) != {'requestId', 'expectedReleaseId'}
                or not _request_id(body['requestId']) or not _hex(body['expectedReleaseId'])):
            raise SeminarError(400, 'invalid_input', 'A canonical request ID and the displayed audience release are required.')
        if not self.enabled:
            raise SeminarError(503, 'publication_disabled', 'Model updates are not configured.')
        if not self._publish_lock.acquire(blocking=False):
            raise SeminarError(409, 'publication_busy', 'An audience model update is already running. Refresh its status before trying again.')
        locked = False
        try:
            active = self._active
            if body['requestId'] == active.request:
                if body['expectedReleaseId'] != active.expected:
                    raise SeminarError(409, 'request_conflict', 'This publication request ID was used with a different expected release.')
                return dict(self.describe(), alreadyPublished=True)
            if body['expectedReleaseId'] != active.release:
                raise SeminarError(409, 'stale_release', 'The audience model changed. Refresh its status before updating it.')
            candidate = self._candidate(active)
            locked = worker.acquire(timeout=self.worker_timeout)
            if not locked:
                raise SeminarError(409, 'audience_busy', 'An audience response is still running. The model was not changed; refresh status before trying again.')
            provider = self.loader(candidate)
            metadata = provider.metadata()
            next_active = _Active(provider, candidate, metadata, release_id(metadata),
                                  self.clock(), body['requestId'], body['expectedReleaseId'])
            pointer = {'publicationVersion': 1, 'bundle': candidate.name, 'releaseId': next_active.release,
                       'publishedAt': next_active.published_at, 'lastRequestId': next_active.request,
                       'expectedReleaseId': next_active.expected}
            self._validate_pointer(pointer)
            self._persist(pointer)
            self._active = next_active
            return dict(self.describe(), alreadyPublished=False)
        except SeminarError:
            raise
        except Exception as error:
            raise SeminarError(503, 'publication_failed', 'The current weights could not be published. The audience model was not changed.') from error
        finally:
            if locked:
                worker.release()
            self._publish_lock.release()
