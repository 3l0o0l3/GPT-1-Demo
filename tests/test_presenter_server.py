"""Real HTTP boundary tests; controlled mail capture only in temporary fixtures."""
import http.client
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from presenter_auth import AuthService
from presenter_server import PresenterApplication, PresenterHTTPServer


EMAIL='presenter@example.com'
ORIGIN='https://demo.example.com'


class FixtureMail:
    configured=True
    def __init__(self):
        self.sent=[]
    def send_code(self,email,code,expires_minutes=10):
        self.sent.append((email,code))


class PresenterHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.clock=1800000000.
        self.mail=FixtureMail()
        self.auth=AuthService(self.root/'auth.sqlite',self.root/'auth.key',self.mail,clock=lambda:self.clock)
        static=self.root/'dist';static.mkdir()
        for name in ('index.html','training.html','teaching.html','training-app.js','training-jobs.js','presenter-session.js','presenter-login.html','presenter-login.js','presenter-login.css'):
            (static/name).write_text('fixture '+name)
        (static/'secret.env').write_text('never served')
        (static/'vendor'/'fonts').mkdir(parents=True)
        (static/'vendor'/'fonts'/'KaTeX_Size1-Regular.woff2').write_bytes(b'font fixture')
        self.gpu=Mock();self.gpu.request.return_value=(200,{'step':9000})
        self.seminar=Mock();self.seminar.get.return_value={'submissionCount':2};self.seminar.post.return_value={'share':{'state':'revoked'}}
        app=PresenterApplication(self.auth,self.mail,static,ORIGIN)
        self.server=PresenterHTTPServer(('127.0.0.1',0),app,self.gpu,self.seminar)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()
        self.temp.cleanup()

    def request(self,path,body=None,cookie=None,csrf=None,headers=None,method=None):
        outgoing={'Host':'demo.example.com'}
        if body is not None:
            outgoing.update({'Origin':ORIGIN,'Content-Type':'application/json'})
        if cookie:outgoing['Cookie']=cookie
        if csrf:outgoing['X-CSRF-Token']=csrf
        outgoing.update(headers or {})
        conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        conn.request(method or ('POST' if body is not None else 'GET'),path,json.dumps(body) if body is not None else None,outgoing)
        response=conn.getresponse();raw=response.read()
        value=json.loads(raw) if 'application/json' in response.getheader('Content-Type','') and raw else raw.decode()
        cookies=[v.split(';')[0] for k,v in response.getheaders() if k.lower()=='set-cookie']
        result=response.status,value,dict(response.getheaders()),cookies
        conn.close();return result

    def prelogin(self):
        status,value,headers,cookies=self.request('/api/auth/session')
        self.assertEqual(status,200);self.assertFalse(value['authenticated'])
        self.assertIn('Secure',headers['Set-Cookie']);self.assertIn('HttpOnly',headers['Set-Cookie']);self.assertIn('SameSite=Strict',headers['Set-Cookie'])
        return cookies[0],value['csrfToken']

    def login(self,email=EMAIL):
        cookie,csrf=self.prelogin()
        status,value,_,_=self.request('/api/auth/request-code',{'email':email},cookie,csrf)
        self.assertEqual(status,200)
        status,value,_,cookies=self.request('/api/auth/verify',{'challengeId':value['challengeId'],'code':self.mail.sent[-1][1]},cookie,csrf)
        self.assertEqual(status,200);self.assertTrue(value['authenticated'])
        self.assertNotIn('sessionToken',value)
        return next(c for c in cookies if c.startswith('__Host-presenter_session=')),value['csrfToken']

    def test_both_presenters_have_independent_authenticated_http_sessions(self):
        original,original_csrf=self.login()
        additional='collaborator@example.com'
        added,added_csrf=self.login(additional)
        self.assertEqual([email for email,_ in self.mail.sent],[EMAIL,additional])
        self.assertNotEqual(original,added)
        self.assertNotEqual(original_csrf,added_csrf)
        for cookie,email in ((original,EMAIL),(added,additional)):
            self.assertEqual(self.request('/api/auth/session',cookie=cookie)[1]['email'],email)
            self.assertEqual(self.request('/',cookie=cookie)[0],200)
            self.assertEqual(self.request('/api/training/state',cookie=cookie)[0],200)
        self.assertEqual(self.request('/api/training/train',{'steps':10},added,original_csrf)[0],403)
        self.assertEqual(self.request('/api/auth/logout',{},added,added_csrf)[0],200)
        self.assertEqual(self.request('/api/training/state',cookie=added)[0],401)
        self.assertEqual(self.request('/api/training/state',cookie=original)[0],200)

    def test_all_management_surfaces_require_authentication(self):
        for path in ('/','/index.html','/training.html','/teaching.html'):
            status,_,headers,_=self.request(path)
            self.assertEqual(status,302);self.assertEqual(headers['Location'],'/login')
        for path in ('/training-app.js','/training-jobs.js','/presenter-session.js','/vendor/katex.min.js','/api/training/state','/api/training/job','/api/telemetry','/api/seminar/status','/api/seminar/submissions'):
            self.assertEqual(self.request(path)[0],401,path)
        for path in ('/api/training/train','/api/training/job-start','/api/training/job-pause','/api/training/generate-stream','/api/seminar/share','/api/seminar/share/revoke','/api/seminar/submissions/clear'):
            self.assertEqual(self.request(path,{} )[0],401,path)
        self.gpu.request.assert_not_called();self.gpu.stream.assert_not_called();self.seminar.get.assert_not_called();self.seminar.post.assert_not_called()
        for path in ('/login','/presenter-login.js','/presenter-login.css'):
            self.assertEqual(self.request(path)[0],200)

    def test_allowlist_and_real_fixture_code_cookie_flow(self):
        cookie,csrf=self.prelogin()
        denied=self.request('/api/auth/request-code',{'email':'other@example.org'},cookie,csrf)
        self.assertEqual(denied[0],200);self.assertEqual(self.mail.sent,[])
        self.assertEqual(self.request('/api/auth/verify',{'challengeId':denied[1]['challengeId'],'code':'00000000'},cookie,csrf)[0],401)
        session,token=self.login()
        self.assertEqual(self.request('/',cookie=session)[0],200)
        self.assertEqual(self.request('/vendor/fonts/KaTeX_Size1-Regular.woff2',cookie=session)[0],200)
        self.assertEqual(self.request('/training-jobs.js',cookie=session)[0],200)
        self.assertTrue(self.request('/api/auth/session',cookie=session)[1]['authenticated'])
        self.assertEqual(self.request('/api/training/state',cookie=session)[1]['step'],9000)
        self.assertEqual(self.request('/api/seminar/status',cookie=session)[1]['submissionCount'],2)
        self.assertEqual(self.request('/api/seminar/share/revoke',{},session,token)[0],200)

    def test_host_origin_and_csrf_guard_protected_mutations(self):
        session,csrf=self.login()
        for headers in ({'Host':'evil.example'},{'Origin':'https://evil.example'},{'Sec-Fetch-Site':'cross-site'}):
            self.assertEqual(self.request('/api/training/train',{'steps':10},session,csrf,headers)[0],403)
        self.assertEqual(self.request('/api/training/train',{'steps':10},session)[0],403)
        self.assertEqual(self.request('/api/training/train',{'steps':10},session,'bad')[0],403)
        self.gpu.request.assert_not_called()
        self.assertEqual(self.request('/api/training/train',{'steps':10},session,csrf)[0],200)
        self.gpu.request.assert_called_once_with('POST','/api/training/train',b'{"steps": 10}')

    def test_background_job_routes_keep_auth_csrf_and_single_forward(self):
        session,csrf=self.login()
        self.assertEqual(self.request('/api/training/job',cookie=session)[0],200)
        self.gpu.request.assert_called_once_with('GET','/api/training/job',None)
        for route,body in (('job-start',{'kind':'lm','steps':1000,'learningRate':.0003,'requestId':'65db2951-9d10-4c24-b3ab-29770a1b97f5'}),
                           ('job-pause',{'jobId':'job-fixture'})):
            self.gpu.request.reset_mock()
            path='/api/training/'+route
            self.assertEqual(self.request(path,body,session)[0],403)
            self.assertEqual(self.request(path,body,session,csrf,{'Origin':'https://evil.example'})[0],403)
            self.gpu.request.assert_not_called()
            self.assertEqual(self.request(path,body,session,csrf)[0],200)
            self.gpu.request.assert_called_once_with('POST',path,json.dumps(body).encode())

    def test_expiry_logout_and_fixed_static_allowlist(self):
        session,csrf=self.login()
        for path in ('/secret.env','/../auth.key','/%2e%2e/auth.key','/presenter_server.py','//evil.example/x'):
            self.assertIn(self.request(path,cookie=session)[0],(400,404))
        self.assertEqual(self.request('/api/auth/logout',{},session,csrf)[0],200)
        self.assertEqual(self.request('/api/seminar/submissions',cookie=session)[0],401)
        self.clock+=61
        session,csrf=self.login()
        self.clock+=28800
        self.assertEqual(self.request('/api/training/train',{'steps':10},session,csrf)[0],401)
        self.gpu.request.assert_not_called()

    def test_clear_submissions_requires_login_same_origin_and_csrf(self):
        path='/api/seminar/submissions/clear'
        body={'throughCursor':3}
        self.assertEqual(self.request(path,body)[0],401)
        session,csrf=self.login()
        self.assertEqual(self.request(path,body,session)[0],403)
        self.assertEqual(self.request(path,body,session,'invalid')[0],403)
        for headers in ({'Host':'evil.example'},{'Origin':'https://evil.example'},{'Sec-Fetch-Site':'cross-site'}):
            self.assertEqual(self.request(path,body,session,csrf,headers)[0],403)
        self.seminar.post.assert_not_called()
        self.seminar.post.return_value={'deletedCount':3,'submissionCount':0,'lastCursor':3,'clearedThrough':3}
        status,value,_,_=self.request(path,body,session,csrf)
        self.assertEqual(status,200)
        self.assertEqual(value['deletedCount'],3)
        self.seminar.post.assert_called_once_with(path,body)
        self.gpu.request.assert_not_called()

    def test_login_csrf_unknown_fields_and_body_bounds(self):
        cookie,csrf=self.prelogin()
        self.assertEqual(self.request('/api/auth/request-code',{'email':EMAIL},cookie)[0],403)
        self.assertEqual(self.request('/api/auth/request-code',{'email':EMAIL,'to':'other@example.org'},cookie,csrf)[0],400)
        conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        conn.request('POST','/api/auth/request-code',b'',{'Host':'demo.example.com','Origin':ORIGIN,
            'Cookie':cookie,'X-CSRF-Token':csrf,'Content-Type':'application/json','Content-Length':'70000'})
        response=conn.getresponse();self.assertEqual(response.status,413);response.read();conn.close()
        self.assertEqual(self.mail.sent,[])


if __name__=='__main__':unittest.main()


class PresenterConfigurationTests(unittest.TestCase):
    def test_cli_requires_explicit_origin_and_allowlist_before_initializing_auth(self):
        from presenter_server import main
        base=['--static','dist','--state-dir','unused-state','--smtp-config','unused-config']
        for extra in ([],['--public-url','https://demo.example.com'],['--allowed-email','presenter@example.com']):
            with self.subTest(extra=extra), patch('presenter_server.GmailMailer') as mailer, patch('presenter_server.AuthService') as auth, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(base+extra)
            self.assertEqual(raised.exception.code,2)
            mailer.assert_not_called();auth.assert_not_called()

    def test_cli_passes_the_same_explicit_allowlist_to_mailer_and_auth_and_binds_loopback(self):
        from presenter_server import main
        emails=['owner@example.com','colleague@example.com']
        with tempfile.TemporaryDirectory() as directory, patch('presenter_server.GmailMailer') as mailer, patch('presenter_server.AuthService') as auth, patch('presenter_server.PresenterApplication') as application, patch('presenter_server.PresenterHTTPServer') as server:
            main(['--static','dist','--state-dir',directory,'--smtp-config','smtp-config.json','--public-url','https://demo.example.com','--allowed-email',emails[0],'--allowed-email',emails[1]])
            mailer.assert_called_once_with('smtp-config.json',allowed_email=emails)
            self.assertEqual(auth.call_args.kwargs['allowed_email'],emails)
            self.assertIs(auth.call_args.args[2],mailer.return_value)
            self.assertEqual(application.call_args.args[3],'https://demo.example.com')
            self.assertEqual(server.call_args.args[0],('127.0.0.1',18880))
            server.return_value.serve_forever.assert_called_once();server.return_value.server_close.assert_called_once()
