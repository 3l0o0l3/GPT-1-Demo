"""Invitation state/clock boundaries; no tunnel or network access required."""
import base64
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from seminar_share import ShareManager
from seminar_store import SeminarError


class InvitationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.state, self.endpoint = self.root/'invitation.json', self.root/'endpoint.json'
        self.now = 1800000000.0
        self.origin = 'https://example-seminar.trycloudflare.com'
        self.heartbeat()
        self.manager = ShareManager(self.state, self.endpoint, lambda: self.now)

    def tearDown(self):
        self.temp.cleanup()

    def heartbeat(self, **overrides):
        value = dict(origin=self.origin, updatedAt=self.now)
        value.update(overrides)
        self.endpoint.write_text(json.dumps(value), encoding='utf-8')

    def assert_error(self, status, code, callback, *args):
        with self.assertRaises(SeminarError) as caught:
            callback(*args)
        self.assertEqual((caught.exception.status, caught.exception.code), (status, code))

    def active(self, minutes=120):
        status = self.manager.create(minutes)
        path = status['publicUrl'].removeprefix(self.origin)
        resolved = self.manager.resolve('example-seminar.trycloudflare.com', path)
        return status, path, resolved

    def test_create_qr_exact_resolution_and_restart(self):
        self.assertEqual(self.manager.status()['state'], 'inactive')
        status, path, resolved = self.active()
        self.assertEqual(status['state'], 'active')
        self.assertEqual(status['remainingSeconds'], 7200)
        self.assertEqual(resolved['publicUrl'], status['publicUrl'])
        self.assertEqual(resolved['prefix']+'/', path)
        self.assertTrue(resolved['secure'])
        svg = base64.b64decode(status['qrDataUrl'].split(',', 1)[1])
        self.assertIn(b'<svg', svg)
        self.assertNotIn(b'<script', svg)
        self.assertEqual(self.manager.resolve(resolved['host'], path+'api/chat', self.origin), resolved)
        self.manager.check(resolved['id'])
        reloaded = ShareManager(self.state, self.endpoint, lambda: self.now)
        self.assertEqual(reloaded.status(), status)

    def test_invalid_durations_do_not_rotate_active_invitation(self):
        before, _, _ = self.active()
        for value in (None, True, False, '120', 120.0, -1, 0, 14, 16, 241, [], {}):
            self.assert_error(400, 'invalid_input', self.manager.create, value)
        self.assertEqual(self.manager.status(), before)

    def test_expiry_exact_boundary_even_with_existing_identity(self):
        _, path, resolved = self.active(15)
        self.now += 899.5
        self.heartbeat()
        self.assertEqual(self.manager.status()['remainingSeconds'], 1)
        self.manager.check(resolved['id'])
        self.now += .5
        status = self.manager.status()
        self.assertEqual((status['state'], status['remainingSeconds']), ('expired', 0))
        self.assertIsNone(status['publicUrl'])
        self.assertIsNone(status['qrDataUrl'])
        self.assert_error(410, 'share_closed', self.manager.check, resolved['id'])
        self.assert_error(410, 'share_closed', self.manager.resolve, resolved['host'], path)

    def test_revoke_idempotent_persists_and_rotation_never_reopens_old_url(self):
        first, path, resolved = self.active()
        closed = self.manager.revoke()
        self.assertEqual(closed['state'], 'revoked')
        self.assertEqual(self.manager.revoke(), closed)
        self.assertEqual(ShareManager(self.state, self.endpoint, lambda:self.now).status(), closed)
        self.assert_error(410, 'share_closed', self.manager.check, resolved['id'])
        second, _, newer = self.active()
        self.assertNotEqual(first['publicUrl'], second['publicUrl'])
        self.assert_error(410, 'share_closed', self.manager.resolve, resolved['host'], path)
        self.manager.check(newer['id'])

    def test_direct_rotation_invalidates_previous_active_link(self):
        first, _, old = self.active()
        second, _, new = self.active(30)
        self.assertNotEqual(first['publicUrl'], second['publicUrl'])
        self.assertEqual(second['remainingSeconds'], 1800)
        self.assert_error(410, 'share_closed', self.manager.check, old['id'])
        self.manager.check(new['id'])

    def test_host_origin_and_path_allowlist(self):
        _, path, resolved = self.active()
        for host in ('evil.example', 'example-seminar.trycloudflare.com.evil',
                     'example-seminar.trycloudflare.com:443', 'example-seminar.trycloudflare.com@evil'):
            self.assert_error(403, 'invalid_host', self.manager.resolve, host, path)
        for origin in ('null', 'https://evil.example', self.origin+'/', 'http://example-seminar.trycloudflare.com'):
            self.assert_error(403, 'invalid_origin', self.manager.resolve, resolved['host'], path, origin)
        for unknown in ('/', '/admin/status', '/seminar-gpt/', '/seminar-gpt/s/short/', path.replace('/s/', '/%73/')):
            self.assert_error(404, 'not_found', self.manager.resolve, resolved['host'], unknown)
        for invalid_id in (None, {}, [], '', 'x'*43, '中文'*22):
            self.assert_error(410, 'share_closed', self.manager.check, invalid_id)

    def test_heartbeat_stale_missing_malformed_and_new_origin_fail_closed(self):
        _, _, resolved = self.active()
        self.now += 30
        self.manager.check(resolved['id'])
        self.now += .01
        self.assertEqual(self.manager.status()['state'], 'unavailable')
        self.assert_error(503, 'share_unavailable', self.manager.check, resolved['id'])
        self.assert_error(503, 'share_unavailable', self.manager.create, 15)
        self.heartbeat()
        self.manager.check(resolved['id'])
        self.endpoint.unlink()
        self.assert_error(503, 'share_unavailable', self.manager.check, resolved['id'])
        self.endpoint.write_text('{broken')
        self.assertEqual(self.manager.status()['state'], 'unavailable')
        self.heartbeat(origin='https://changed.trycloudflare.com')
        self.assert_error(503, 'share_unavailable', self.manager.check, resolved['id'])
        self.assertTrue(self.manager.create(15)['publicUrl'].startswith('https://changed.trycloudflare.com/'))

    def test_invalid_endpoint_forms_cannot_create_a_share(self):
        for origin in ('http://example.trycloudflare.com', 'https://trycloudflare.com',
                       'https://one.two.trycloudflare.com', 'https://example.trycloudflare.com:443',
                       'https://example.trycloudflare.com/', 'https://user' + '@' + 'example.trycloudflare.com',
                       'https://-bad.trycloudflare.com', 'https://bad-.trycloudflare.com',
                       'https://'+('a'*64)+'.trycloudflare.com', 'https://owner.example'):
            self.heartbeat(origin=origin)
            self.assert_error(503, 'share_unavailable', self.manager.create, 15)
        for updated in (True, None, '1800000000', float('nan'), float('inf'), self.now+5.1):
            self.heartbeat(updatedAt=updated)
            self.assert_error(503, 'share_unavailable', self.manager.create, 15)

    def test_invalid_persisted_state_is_closed_until_explicit_creation(self):
        self.state.write_text('{broken')
        manager = ShareManager(self.state, self.endpoint, lambda:self.now)
        self.assertEqual(manager.status()['state'], 'unavailable')
        self.assert_error(503, 'share_unavailable', manager.check, 'a'*43)
        self.assertEqual(manager.create(15)['state'], 'active')

    def test_failed_atomic_replace_does_not_publish_new_capability(self):
        before, _, resolved = self.active()
        original = self.state.read_bytes()
        with patch('seminar_share.os.replace', side_effect=OSError('controlled write failure')):
            with self.assertRaises(OSError):
                self.manager.create(15)
        self.assertEqual(self.manager.status(), before)
        self.assertEqual(self.state.read_bytes(), original)
        self.assertFalse(list(self.root.glob('*.tmp')))
        self.manager.check(resolved['id'])

    def test_repeated_status_reuses_qr_rendering(self):
        status, _, _ = self.active()
        with patch('segno.make', side_effect=AssertionError('QR rendered again')):
            for _ in range(3):
                self.assertEqual(self.manager.status()['qrDataUrl'], status['qrDataUrl'])

    def test_guard_serializes_commits_against_revocation_and_rejects_closed(self):
        _, _, resolved = self.active()
        started, revoked = threading.Event(), threading.Event()

        def revoke():
            started.set()
            self.manager.revoke()
            revoked.set()

        with self.manager.guard(resolved['id']):
            thread = threading.Thread(target=revoke)
            thread.start()
            self.assertTrue(started.wait(2))
            self.assertFalse(revoked.wait(.05))
        thread.join(timeout=2)
        self.assertTrue(revoked.is_set())
        with self.assertRaises(SeminarError) as caught:
            with self.manager.guard(resolved['id']):
                self.fail('A revoked invitation entered the commit guard')
        self.assertEqual(caught.exception.code, 'share_closed')


if __name__ == '__main__':
    unittest.main()
