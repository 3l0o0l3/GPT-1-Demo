"""Server-clock invitations, independent of HTTP, model and tunnel processes."""
from contextlib import contextmanager
from datetime import datetime, timezone
import base64
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import threading
import time

from seminar_store import SeminarError


_ORIGIN = re.compile(r'https://[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.trycloudflare\.com\Z')
_TOKEN = re.compile(r'[A-Za-z0-9_-]{43}\Z')
_PATH = re.compile(r'/seminar-gpt/s/([A-Za-z0-9_-]{43})(?:/.*)?\Z')
_DURATIONS = (15, 30, 60, 120, 240)
_PROVIDER = 'Cloudflare temporary tunnel'


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _date(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


class ShareManager:
    """One process owns invitation state; endpoint heartbeat is a public contract.

    State replacement is atomic and durable before a new capability is returned.
    The tunnel supervisor only owns endpoint_path; it never edits invitation state.
    No network service is contacted to render a QR code.
    """

    def __init__(self, state_path, endpoint_path, clock=time.time):
        self.state_path, self.endpoint_path = Path(state_path), Path(endpoint_path)
        self.clock, self.lock = clock, threading.RLock()
        self._qr_url, self._qr_data = None, None
        self._state, self._invalid = None, False
        try:
            state = self._read(self.state_path)
            self._validate_state(state)
            self._state = state
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError):
            self._invalid = True

    @staticmethod
    def _read(path):
        # Both inputs are small, locally owned JSON files, not request bodies.
        with path.open('rb') as handle:
            raw = handle.read(8193)
        if len(raw) > 8192:
            raise ValueError('Invalid invitation configuration size.')
        return json.loads(raw)

    @staticmethod
    def _validate_state(value):
        expected = {'version', 'id', 'origin', 'createdAt', 'expiresAt', 'revokedAt'}
        if not isinstance(value, dict) or set(value) != expected or type(value['version']) is not int or value['version'] != 1:
            raise ValueError('Invalid invitation state.')
        if not isinstance(value['id'], str) or not _TOKEN.fullmatch(value['id']):
            raise ValueError('Invalid invitation capability.')
        if not isinstance(value['origin'], str) or not _ORIGIN.fullmatch(value['origin']):
            raise ValueError('Invalid invitation origin.')
        start, end, revoked = value['createdAt'], value['expiresAt'], value['revokedAt']
        if not _number(start) or not _number(end) or not 0 <= start < end <= 253402300799 or end - start not in tuple(m*60 for m in _DURATIONS):
            raise ValueError('Invalid invitation lifetime.')
        if revoked is not None and (not _number(revoked) or revoked < 0):
            raise ValueError('Invalid invitation revocation.')

    def _endpoint(self, now):
        try:
            value = self._read(self.endpoint_path)
            if not isinstance(value, dict) or set(value) != {'origin', 'updatedAt'}:
                return None
            origin, updated = value['origin'], value['updatedAt']
            if not isinstance(origin, str) or not _ORIGIN.fullmatch(origin):
                return None
            if not _number(updated) or not -5 <= now-updated <= 30:
                return None
            return origin
        except (OSError, ValueError, TypeError):
            return None

    def _save(self, state):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(self.state_path.name+'.'+secrets.token_hex(8)+'.tmp')
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
                json.dump(state, handle, ensure_ascii=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
            # Linux deployment: also persist the directory entry replacement.
            if hasattr(os, 'O_DIRECTORY'):
                directory = os.open(self.state_path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _condition(self, now):
        if self._invalid:
            return 'unavailable'
        state = self._state
        if state is not None:
            if state['revokedAt'] is not None:
                return 'revoked'
            if now >= state['expiresAt']:
                return 'expired'
        endpoint = self._endpoint(now)
        if not endpoint or (state is not None and endpoint != state['origin']):
            return 'unavailable'
        return 'active' if state is not None else 'inactive'

    @staticmethod
    def _url(state):
        return state['origin']+'/seminar-gpt/s/'+state['id']+'/'

    def _qr(self, url):
        if url != self._qr_url:
            import segno
            buffer = io.BytesIO()
            segno.make(url, error='m', micro=False).save(buffer, kind='svg', scale=5,
                                                       border=4, light='#fff', dark='#002e4f')
            self._qr_data = 'data:image/svg+xml;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')
            self._qr_url = url
        return self._qr_data

    def _status(self, now):
        condition, state = self._condition(now), self._state
        url = self._url(state) if condition == 'active' else None
        return dict(state=condition, publicUrl=url,
                    expiresAt=_date(state['expiresAt']) if state else None,
                    remainingSeconds=max(0, math.ceil(state['expiresAt']-now)) if state and condition == 'active' else 0,
                    qrDataUrl=self._qr(url) if url else None, provider=_PROVIDER)

    def status(self):
        with self.lock:
            return self._status(self.clock())

    def create(self, minutes):
        if type(minutes) is not int or minutes not in _DURATIONS:
            raise SeminarError(400, 'invalid_input', 'Choose 15, 30, 60, 120 or 240 minutes.')
        with self.lock:
            now = self.clock()
            origin = self._endpoint(now)
            if not origin:
                raise SeminarError(503, 'share_unavailable', 'The temporary tunnel is not ready. Try again shortly.')
            state = dict(version=1, id=secrets.token_urlsafe(32), origin=origin,
                         createdAt=now, expiresAt=now+minutes*60, revokedAt=None)
            # Verify QR rendering before changing a currently valid invitation.
            self._qr(self._url(state))
            self._save(state)
            self._state, self._invalid = state, False
            return self._status(now)

    def revoke(self):
        with self.lock:
            if self._state is not None and self._state['revokedAt'] is None:
                state = dict(self._state, revokedAt=self.clock())
                self._save(state)
                self._state = state
            return self._status(self.clock())

    def _require(self, invitation_id, now):
        condition = self._condition(now)
        if condition == 'unavailable':
            raise SeminarError(503, 'share_unavailable', 'The temporary invitation is unavailable. Ask the presenter for a new link.')
        if (condition != 'active' or not isinstance(invitation_id, str) or
                not _TOKEN.fullmatch(invitation_id) or
                not secrets.compare_digest(invitation_id, self._state['id'])):
            raise SeminarError(410, 'share_closed', 'This invitation has expired or was closed. Ask the presenter for a new link.')

    def check(self, invitation_id):
        with self.lock:
            self._require(invitation_id, self.clock())

    @contextmanager
    def guard(self, invitation_id):
        """Authorize and serialize a small commit against create/revoke.

        The clock check at entry is the authorization linearization point. Never
        hold this context across model execution or network I/O.
        """
        with self.lock:
            self._require(invitation_id, self.clock())
            yield

    def resolve(self, host, path, origin=None):
        with self.lock:
            now = self.clock()
            state = self._state
            expected_origin = state['origin'] if state else self._endpoint(now)
            if expected_origin and host != expected_origin.removeprefix('https://'):
                raise SeminarError(403, 'invalid_host', 'Use the current seminar invitation.')
            if origin is not None and origin != expected_origin:
                raise SeminarError(403, 'invalid_origin', 'Use the seminar page to send requests.')
            match = _PATH.fullmatch(path) if isinstance(path, str) else None
            if not match:
                raise SeminarError(404, 'not_found', 'Not found.')
            self._require(match.group(1), now)
            url = self._url(state)
            return dict(id=state['id'], publicUrl=url, origin=state['origin'],
                        host=state['origin'].removeprefix('https://'),
                        prefix='/seminar-gpt/s/'+state['id'], secure=True)
