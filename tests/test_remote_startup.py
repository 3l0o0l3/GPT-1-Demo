import tempfile
import unittest
from pathlib import Path
import os
from unittest.mock import patch

from remote_training_server import restore_startup_session, configure_compilation_environment, PERSISTENCE_DESCRIPTION


class StartupLab:
    def __init__(self, restore_error=None):
        self.calls = []
        self.restore_error = restore_error

    def restore_live(self):
        self.calls.append(('restore_live',))
        if self.restore_error:
            raise self.restore_error
        return {'step':4000}

    def reset(self, source):
        self.calls.append(('reset',source))
        return {'step':1000}

    def prepare_training_execution(self):
        self.calls.append(('prepare_training_execution',))


class RemoteStartupTests(unittest.TestCase):
    def test_missing_live_session_uses_original_pretrained_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            lab = StartupLab()
            self.assertEqual(restore_startup_session(lab,directory),'pretrained-checkpoint')
            self.assertEqual(lab.calls,[('reset','pretrained'),('prepare_training_execution',)])

    def test_present_live_session_restores_without_resetting_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            live = Path(directory)/'live'
            live.mkdir()
            (live/'session.pt').write_bytes(b'fixture')
            lab = StartupLab()
            self.assertEqual(restore_startup_session(lab,directory),'live-session')
            self.assertEqual(lab.calls,[('restore_live',),('prepare_training_execution',)])

    def test_incompatible_or_missing_during_restore_fails_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            live = Path(directory)/'live'
            live.mkdir()
            (live/'session.pt').write_bytes(b'fixture')
            for error in (ValueError('Tokenizer mismatch.'),FileNotFoundError('Removed during restore.')):
                with self.subTest(error=type(error).__name__):
                    lab = StartupLab(error)
                    with self.assertRaises(type(error)):
                        restore_startup_session(lab,directory)
                    self.assertEqual(lab.calls,[('restore_live',)])
            self.assertIn('Unsaved updates remain in server memory',PERSISTENCE_DESCRIPTION)

    def test_private_compiler_environment_and_failure_before_readiness(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            cache = configure_compilation_environment(Path(directory))
            self.assertEqual(cache, Path(directory) / '.runtime' / 'torchinductor-cache')
            self.assertEqual(os.environ['TORCHINDUCTOR_COMPILE_THREADS'], '1')
            self.assertEqual(os.environ['MAX_JOBS'], '1')
            self.assertEqual(os.environ['TORCHINDUCTOR_CACHE_DIR'], str(cache))
            self.assertTrue(cache.is_dir())
            lab = StartupLab()
            with patch.object(lab, 'prepare_training_execution', side_effect=RuntimeError('compile failed')):
                with self.assertRaisesRegex(RuntimeError, 'compile failed'):
                    restore_startup_session(lab, directory)


if __name__ == '__main__':
    unittest.main()
