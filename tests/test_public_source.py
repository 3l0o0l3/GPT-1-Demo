"""Publication checks use actual Git index blobs, never a guessed export list."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.check_public_source import check


class PublicSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.git('init', '-q')

    def git(self, *arguments):
        return subprocess.run(['git', *arguments], cwd=self.root, check=True, capture_output=True)

    def stage(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf8')
        self.git('add', '--', name)

    def test_checks_staged_bytes_and_detects_later_staged_private_content(self):
        self.stage('settings.py', "email = 'presenter@example.com'\n")
        private_email = 'operator' + '@' + 'organization.org'
        (self.root / 'settings.py').write_text(private_email, encoding='utf8')
        self.assertTrue(check(self.root)['passed'])
        self.git('add', '--', 'settings.py')
        result = check(self.root)
        self.assertFalse(result['passed'])
        self.assertEqual(result['findings'][0]['category'], 'non-example-email')
        self.assertNotIn(private_email, str(result))

    def test_forced_training_artifact_is_rejected(self):
        self.stage('.training/session.pt', 'test-only synthetic bytes')
        result = check(self.root)
        self.assertFalse(result['passed'])
        self.assertEqual(result['findings'][0]['category'], 'excluded-runtime-or-private-artifact')

    def test_example_configuration_and_upstream_attribution_are_allowed(self):
        self.stage('settings.example', 'presenter@example.com\nhttps://demo.example.com\n')
        self.stage('dist/vendor/LICENSE', 'Upstream attribution: ' + 'author' + '@' + 'upstream.org\n')
        self.assertTrue(check(self.root)['passed'])

    def test_secret_match_reports_location_without_value(self):
        token = 'ghp_' + 'Ab12' * 10
        self.stage('config.py', 'value = ' + repr(token))
        report = check(self.root)
        self.assertFalse(report['passed'])
        self.assertTrue(any(item['category'] == 'github-token' for item in report['findings']))
        self.assertNotIn(token, str(report))


if __name__ == '__main__':
    unittest.main()
