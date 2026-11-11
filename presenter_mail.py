"""Fixed Gmail STARTTLS sender; credentials are never part of the release."""
from email.message import EmailMessage
import json
import os
from pathlib import Path
import re
import smtplib
import ssl
import stat
import struct


DEFAULT_ALLOWED_EMAIL = 'presenter@example.com'
DEFAULT_ALLOWED_EMAILS = (DEFAULT_ALLOWED_EMAIL, 'collaborator@example.com')
_EMAIL = re.compile(r'[A-Za-z0-9.!#$%&\'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\Z')


def normalize_allowlist(value):
    """Return exact normalized recipients; a string retains single-address semantics."""
    if isinstance(value,str):
        value=(value,)
    if not isinstance(value,(tuple,list,set,frozenset)) or not 1<=len(value)<=32:
        raise ValueError('Invalid presenter allowlist.')
    addresses=[]
    for address in value:
        if not isinstance(address,str):
            raise ValueError('Invalid presenter allowlist address.')
        address=address.strip().lower()
        if not 1<=len(address)<=254 or not _EMAIL.fullmatch(address):
            raise ValueError('Invalid presenter allowlist address.')
        addresses.append(address)
    return frozenset(addresses)


class MailerError(Exception):
    """Safe error text only; underlying SMTP responses are deliberately omitted."""


def _service_acl(blob, uid, permission, directory=False):
    """Accept only root owner + this service UID, never extra ACL principals."""
    if not isinstance(blob,bytes) or len(blob)!=4+5*8:
        return False
    if struct.unpack_from('<I',blob)[0]!=2:
        return False
    entries=[struct.unpack_from('<HHI',blob,offset) for offset in range(4,len(blob),8)]
    tags={tag for tag,_,_ in entries}
    if tags!={1,2,4,16,32}:  # USER_OBJ, USER, GROUP_OBJ, MASK, OTHER
        return False
    values={tag:(perm,identifier) for tag,perm,identifier in entries}
    undefined=0xffffffff
    owner=values[1][0]
    return (values[1][1]==undefined and owner in ((5,7) if directory else (4,)) and
            values[2]==(permission,uid) and values[4]==(0,undefined) and
            values[16]==(permission,undefined) and values[32]==(0,undefined))


def _credential_permissions(file_info,directory_info,uid,gid,file_acl=None,directory_acl=None):
    """Root-controlled systemd read-only credentials, either group- or ACL-granted."""
    if (not stat.S_ISREG(file_info.st_mode) or stat.S_IMODE(file_info.st_mode)!=0o440 or
            not stat.S_ISDIR(directory_info.st_mode) or file_info.st_uid!=0 or directory_info.st_uid!=0):
        return False
    directory_mode=stat.S_IMODE(directory_info.st_mode)
    if directory_mode & 0o027 or directory_mode & 0o050!=0o050:
        return False
    if file_acl is None and directory_acl is None:
        return file_info.st_gid==gid and directory_info.st_gid==gid
    return (_service_acl(file_acl,uid,4) and
            _service_acl(directory_acl,uid,5,directory=True))


def _systemd_credential(path,file_info):
    """Narrow exception for the credential mount declared by systemd itself."""
    configured=os.environ.get('CREDENTIALS_DIRECTORY')
    if not configured:
        return False
    directory=Path(configured)
    if (not path.is_absolute() or not directory.is_absolute() or path.parent!=directory or
            directory.parent!=Path('/run/credentials') or not directory.name.endswith('.service') or
            directory.is_symlink() or path.resolve(strict=True)!=path):
        return False
    def acl(target):
        import errno
        try:
            return os.getxattr(target,'system.posix_acl_access')
        except OSError as error:
            if error.errno in (errno.ENODATA,getattr(errno,'ENOATTR',errno.ENODATA)):
                return None
            raise
    return _credential_permissions(file_info,directory.stat(),os.geteuid(),os.getegid(),
                                   acl(path),acl(directory))


class GmailMailer:
    def __init__(self, config_path, allowed_email=DEFAULT_ALLOWED_EMAILS,
                 smtp_factory=smtplib.SMTP, context_factory=ssl.create_default_context):
        self.config_path = Path(config_path) if config_path else None
        self.allowed_emails = normalize_allowlist(allowed_email)
        self.smtp_factory, self.context_factory = smtp_factory, context_factory

    def _config(self):
        try:
            path = self.config_path
            if path is None or path.is_symlink() or not path.is_file():
                raise ValueError()
            if os.name == 'posix':
                file_info=path.stat()
                if file_info.st_mode & 0o077 and not _systemd_credential(path,file_info):
                    raise ValueError()
            with path.open('rb') as handle:
                raw = handle.read(8193)
            if len(raw) > 8192:
                raise ValueError()
            value = json.loads(raw)
            if not isinstance(value, dict) or set(value) != {'username','password','fromAddress'}:
                raise ValueError()
            if any(not isinstance(value[k], str) for k in value):
                raise ValueError()
            if (not _EMAIL.fullmatch(value['username']) or not _EMAIL.fullmatch(value['fromAddress']) or
                    not 1 <= len(value['password']) <= 1024 or '\x00' in value['password'] or
                    '\r' in value['password'] or '\n' in value['password']):
                raise ValueError()
            return value
        except (OSError, ValueError, TypeError, UnicodeError):
            raise MailerError('Email delivery is not configured. Contact the presenter administrator.') from None

    @property
    def configured(self):
        try:
            self._config()
            return True
        except MailerError:
            return False

    def send_code(self, email, code, expires_minutes=10):
        if (not isinstance(email,str) or email.strip().lower() not in self.allowed_emails or
                not isinstance(code,str) or not re.fullmatch(r'[0-9]{8}',code) or
                type(expires_minutes) is not int or not 1 <= expires_minutes <= 30):
            raise MailerError('This email request is not permitted.')
        email=email.strip().lower()
        config = self._config()
        message = EmailMessage()
        message['From'] = config['fromAddress']
        message['To'] = email
        message['Subject'] = 'Your GPT Training Lab sign-in code'
        message.set_content(
            'Use this one-time code to sign in to the GPT Training Lab presenter page:\n\n'
            + code + '\n\n'
            + f'The code expires in {expires_minutes} minutes and works only in the browser that requested it.\n'
            + 'If you did not request this code, you can ignore this message.\n'
            + 'Never share the code with another person.\n')
        try:
            context = self.context_factory()
            if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
                raise ValueError('Certificate verification must stay enabled.')
            with self.smtp_factory('smtp.gmail.com',587,timeout=15) as smtp:
                smtp.ehlo()
                smtp.starttls(context=context)
                smtp.ehlo()
                smtp.login(config['username'],config['password'])
                refused = smtp.send_message(message,from_addr=config['fromAddress'],to_addrs=[email])
                if refused:
                    raise MailerError('Email delivery could not be confirmed. Please try again later.')
        except Exception:
            # No SMTP reply, username, password or message body reaches HTTP/logs.
            raise MailerError('Email delivery could not be confirmed. Please try again later.') from None
