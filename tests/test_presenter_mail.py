"""Gmail sender uses only its protected config and verified STARTTLS."""
import json
import os
from pathlib import Path
import ssl
import stat
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from presenter_mail import DEFAULT_ALLOWED_EMAIL,GmailMailer,MailerError,_credential_permissions,_service_acl


class FakeSMTP:
    def __init__(self):self.events=[];self.failure=None;self.refused={};self.message=None
    def __call__(self,host,port,timeout):
        self.events.append(('connect',host,port,timeout));return self
    def __enter__(self):return self
    def __exit__(self,*args):self.events.append(('exit',))
    def ehlo(self):self.events.append(('ehlo',))
    def starttls(self,context):
        self.events.append(('tls',context.verify_mode,context.check_hostname))
        if self.failure=='tls':raise ssl.SSLError('private fixture response')
    def login(self,username,password):
        self.events.append(('login',))
        if self.failure=='login':raise OSError('secret fixture account/password')
    def send_message(self,message,from_addr,to_addrs):
        self.events.append(('send',from_addr,to_addrs));self.message=message
        return self.refused


class MailerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'smtp.json'
        self.config={'username':'sender@example.com','password':'fixture-app-password','fromAddress':'sender@example.com'}
        self.write(self.config);self.smtp=FakeSMTP();self.mailer=GmailMailer(self.path,smtp_factory=self.smtp)
    def tearDown(self):self.temp.cleanup()
    def write(self,value):
        self.path.write_text(json.dumps(value));os.chmod(self.path,0o600)
    def test_starttls_certificate_verification_before_login_fixed_recipient(self):
        self.assertTrue(self.mailer.configured)
        self.mailer.send_code(DEFAULT_ALLOWED_EMAIL,'00123456')
        self.assertEqual(self.smtp.events,[('connect','smtp.gmail.com',587,15),('ehlo',),('tls',ssl.CERT_REQUIRED,True),
                                         ('ehlo',),('login',),('send','sender@example.com',[DEFAULT_ALLOWED_EMAIL]),('exit',)])
        self.assertEqual(self.smtp.message['To'],DEFAULT_ALLOWED_EMAIL)
        self.assertIn('00123456',self.smtp.message.get_content())
        self.assertIn('10 minutes',self.smtp.message.get_content())
        self.assertNotIn(self.config['password'],self.smtp.message.as_string())
    def test_nonallowlisted_or_injected_message_never_connects(self):
        for email,code in (('other@example.org','12345678'),(DEFAULT_ALLOWED_EMAIL+'\r\nBcc: bad@example.org','12345678'),
                           (DEFAULT_ALLOWED_EMAIL,'12345678\n'),(DEFAULT_ALLOWED_EMAIL,12345678)):
            with self.assertRaises(MailerError):self.mailer.send_code(email,code)
        self.assertEqual(self.smtp.events,[])
    def test_second_allowed_address_receives_its_own_message_and_envelope_only(self):
        email='collaborator@example.com'
        self.mailer.send_code('  '+email.upper()+'  ','87654321')
        self.assertEqual(self.smtp.message['To'],email)
        self.assertIsNone(self.smtp.message['Cc']);self.assertIsNone(self.smtp.message['Bcc'])
        self.assertIn(('send','sender@example.com',[email]),self.smtp.events)
        self.assertNotIn(DEFAULT_ALLOWED_EMAIL,self.smtp.message.as_string())
    def test_exact_second_address_and_legacy_override_do_not_expand_to_aliases(self):
        email='collaborator@example.com'
        for rejected in (email+'.evil.example','collaborator+alias@example.com',email+'\r\nBcc: bad@example.org'):
            with self.assertRaises(MailerError):self.mailer.send_code(rejected,'12345678')
        restricted=GmailMailer(self.path,allowed_email=DEFAULT_ALLOWED_EMAIL,smtp_factory=self.smtp)
        with self.assertRaises(MailerError):restricted.send_code(email,'12345678')
        self.assertEqual(self.smtp.events,[])
        restricted.send_code(DEFAULT_ALLOWED_EMAIL,'12345678')
        self.assertEqual(self.smtp.message['To'],DEFAULT_ALLOWED_EMAIL)
    def test_missing_malformed_or_extra_config_fails_closed(self):
        for config in ({},[],{**self.config,'host':'evil.example'},{**self.config,'password':''},
                       {**self.config,'fromAddress':'sender@example.com\r\nBcc: b@example.org'}):
            self.write(config);self.assertFalse(self.mailer.configured)
            with self.assertRaises(MailerError):self.mailer.send_code(DEFAULT_ALLOWED_EMAIL,'12345678')
        self.path.unlink();self.assertFalse(self.mailer.configured)
        self.assertEqual(self.smtp.events,[])
    def test_failed_tls_never_authenticates_or_sends(self):
        self.smtp.failure='tls'
        with self.assertRaises(MailerError) as caught:self.mailer.send_code(DEFAULT_ALLOWED_EMAIL,'12345678')
        self.assertNotIn('private',str(caught.exception))
        self.assertFalse(any(e[0] in ('login','send') for e in self.smtp.events))
    def test_failed_login_or_refused_recipient_is_generic(self):
        self.smtp.failure='login'
        with self.assertRaises(MailerError) as caught:self.mailer.send_code(DEFAULT_ALLOWED_EMAIL,'12345678')
        self.assertNotIn('secret',str(caught.exception))
        self.assertFalse(any(e[0]=='send' for e in self.smtp.events))
        self.smtp.failure=None;self.smtp.refused={DEFAULT_ALLOWED_EMAIL:(550,b'private response')}
        with self.assertRaises(MailerError):self.mailer.send_code(DEFAULT_ALLOWED_EMAIL,'12345678')
    def test_unverified_tls_context_is_refused_before_connection(self):
        insecure=ssl._create_unverified_context()
        mailer=GmailMailer(self.path,smtp_factory=self.smtp,context_factory=lambda:insecure)
        with self.assertRaises(MailerError):mailer.send_code(DEFAULT_ALLOWED_EMAIL,'12345678')
        self.assertEqual(self.smtp.events,[])

    def test_systemd_acl_grants_only_current_service_user(self):
        def acl(uid,permission,extra=None):
            entries=[(1,permission,0xffffffff),(2,permission,uid),(4,0,0xffffffff),
                     (16,permission,0xffffffff),(32,0,0xffffffff)]
            if extra:entries.append(extra)
            return struct.pack('<I',2)+b''.join(struct.pack('<HHI',*entry) for entry in entries)
        file=SimpleNamespace(st_mode=stat.S_IFREG|0o440,st_uid=0,st_gid=0)
        directory=SimpleNamespace(st_mode=stat.S_IFDIR|0o550,st_uid=0,st_gid=0)
        self.assertTrue(_credential_permissions(file,directory,991,978,acl(991,4),acl(991,5)))
        self.assertFalse(_credential_permissions(file,directory,992,978,acl(991,4),acl(991,5)))
        self.assertFalse(_credential_permissions(file,directory,991,978,acl(991,6),acl(991,5)))
        self.assertFalse(_credential_permissions(file,directory,991,978,acl(991,4,(2,4,992)),acl(991,5)))
        self.assertFalse(_credential_permissions(file,directory,991,978,b'bad',acl(991,5)))
        self.assertFalse(_credential_permissions(file,directory,991,978))
        directory.st_mode=stat.S_IFDIR|0o570
        self.assertFalse(_credential_permissions(file,directory,991,978,acl(991,4),acl(991,5)))

    def test_systemd_direct_group_grant_requires_root_owner_and_no_extra_access(self):
        file=SimpleNamespace(st_mode=stat.S_IFREG|0o440,st_uid=0,st_gid=978)
        directory=SimpleNamespace(st_mode=stat.S_IFDIR|0o750,st_uid=0,st_gid=978)
        self.assertTrue(_credential_permissions(file,directory,991,978))
        for field,value in (('st_gid',999),('st_uid',991),('st_mode',stat.S_IFREG|0o460),('st_mode',stat.S_IFREG|0o444)):
            original=getattr(file,field);setattr(file,field,value)
            self.assertFalse(_credential_permissions(file,directory,991,978))
            setattr(file,field,original)

    @unittest.skipUnless(os.name=='posix','POSIX file mode boundary')
    def test_shared_readable_credentials_are_refused(self):
        os.chmod(self.path,0o644)
        self.assertFalse(self.mailer.configured)
        os.chmod(self.path,0o440)
        self.assertFalse(self.mailer.configured)
        with patch.dict(os.environ,{'CREDENTIALS_DIRECTORY':str(self.path.parent)}):
            self.assertFalse(self.mailer.configured)  # An arbitrary directory is not systemd's mount.


if __name__=='__main__':unittest.main()
