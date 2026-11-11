"""Invitation enforcement through real loopback HTTP, with a controlled model."""
import http.client
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from urllib.parse import urlsplit
import uuid

from seminar_server import AudienceApplication, AudienceHTTPServer
from seminar_share import ShareManager
from seminar_store import SeminarStore
from tests.test_seminar_server import FixtureModel


class InvitationHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.now = 1800000000.0
        self.origin = 'https://seminar-fixture.trycloudflare.com'
        self.endpoint = self.root/'endpoint.json'
        self.heartbeat()
        for name in ('index.html','app.js','style.css'):
            (self.root/name).write_text('audience fixture')
        self.share = ShareManager(self.root/'share.json', self.endpoint, clock=lambda:self.now)
        self.model = FixtureModel()
        self.store = SeminarStore(self.root/'state.sqlite')
        self.app = AudienceApplication(self.store,self.model,None,self.root,share=self.share)
        self.public = AudienceHTTPServer(('127.0.0.1',0),self.app)
        self.private = AudienceHTTPServer(('127.0.0.1',0),self.app,private=True)
        self.threads = []
        for server in (self.public,self.private):
            thread = threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            self.threads.append(thread)

    def heartbeat(self):
        self.endpoint.write_text(json.dumps({'origin':self.origin,'updatedAt':self.now}))

    def tearDown(self):
        self.model.release.set()
        for server in (self.public,self.private):
            server.shutdown()
            server.server_close()
        self.temp.cleanup()

    def request(self,path,data=None,private=False,session=None,headers=None):
        server = self.private if private else self.public
        outgoing = {'Host':f'127.0.0.1:{server.server_port}' if private else urlsplit(self.origin).netloc}
        body = json.dumps(data).encode('utf-8') if data is not None else b''
        if data is not None:
            outgoing.update({'Content-Type':'application/json','Content-Length':str(len(body))})
        if session:
            outgoing.update({'Cookie':session['cookie'],'X-CSRF-Token':session['csrf']})
        outgoing.update(headers or {})
        method = 'POST' if data is not None else 'GET'
        head = f'{method} {path} HTTP/1.1\r\n' + ''.join(f'{key}: {value}\r\n' for key,value in outgoing.items()) + '\r\n'
        # Keep this small fixture request in one write. An early invitation
        # rejection may close before http.client's separate body write arrives.
        with socket.create_connection(('127.0.0.1',server.server_port),timeout=8) as conn:
            conn.sendall(head.encode('ascii') + body)
            with http.client.HTTPResponse(conn) as response:
                response.begin()
                raw = response.read()
                kind = response.getheader('Content-Type','')
                value = ([json.loads(line) for line in raw.splitlines()] if 'ndjson' in kind else
                         json.loads(raw) if 'application/json' in kind else raw.decode())
                return response.status,value,dict(response.getheaders())

    def create(self):
        status,value,_ = self.request('/admin/share',{'durationMinutes':15},private=True)
        self.assertEqual(status,200)
        self.prefix = urlsplit(value['share']['publicUrl']).path
        return value['share']

    def session(self):
        status,value,headers = self.request(self.prefix+'api/session',{})
        self.assertEqual(status,200)
        self.assertIn('Path='+self.prefix,headers['Set-Cookie'])
        self.assertIn('shareExpiresAt',value)
        return dict(cookie=headers['Set-Cookie'].split(';')[0],csrf=value['csrf'])

    def chat(self,session):
        return self.request(self.prefix+'api/chat',{'message':'hello','requestId':str(uuid.uuid4())},session=session)

    def test_private_actions_and_public_host_origin_boundaries(self):
        self.assertIsNone(self.request('/admin/status',private=True)[1]['publicUrl'])
        for headers in ({'Origin':'https://evil.example'},{'Host':'evil.example'},{'Sec-Fetch-Site':'cross-site'}):
            self.assertEqual(self.request('/admin/share',{'durationMinutes':15},private=True,headers=headers)[0],403)
        for body in ({'durationMinutes':1},{'durationMinutes':True},{'durationMinutes':15,'host':'evil'}):
            self.assertEqual(self.request('/admin/share',body,private=True)[0],400)
        share = self.create()
        self.assertTrue(share['qrDataUrl'].startswith('data:image/svg+xml;base64,'))
        self.assertEqual(self.request(self.prefix)[0],200)
        for headers in ({'Origin':'https://evil.example'},{'Host':'demo.example.com'}):
            self.assertEqual(self.request(self.prefix+'api/session',{},headers=headers)[0],403)
        for path in ('/admin/status',self.prefix+'admin/share','/seminar-gpt/',self.prefix+'model.pt'):
            self.assertEqual(self.request(path)[0],404)
        info = self.request(self.prefix+'api/info')[1]
        self.assertNotIn('presenter.example.org',json.dumps(info))
        self.assertNotIn('qrDataUrl',info)

    def test_server_expiry_denies_existing_cookie_and_browser_document(self):
        self.create()
        session = self.session()
        self.now += 900
        self.heartbeat()
        for path,data in (('api/session',{}),('api/new',{}),('api/chat',{'message':'hello','requestId':str(uuid.uuid4())}),('api/submit',{}),('api/info',None),('app.js',None)):
            status,value,_ = self.request(self.prefix+path,data,session=session)
            self.assertEqual((status,value['error']['code']),(410,'share_closed'))
        status,value,_ = self.request(self.prefix,headers={'Accept':'text/html'})
        self.assertEqual(status,410)
        self.assertIn('invitation has closed',value)
        self.assertEqual(self.request('/admin/status',private=True)[1]['share']['state'],'expired')
        self.assertEqual(len(self.model.calls),0)

    def test_rotation_and_revoke_do_not_reopen_old_link(self):
        self.create()
        old = self.prefix
        self.create()
        self.assertNotEqual(old,self.prefix)
        self.assertEqual(self.request(old+'api/session',{})[0],410)
        self.assertEqual(self.request('/admin/share/revoke',{},private=True)[1]['share']['state'],'revoked')
        self.assertEqual(self.request(self.prefix+'api/session',{})[0],410)
        self.assertEqual(self.request('/admin/submissions',private=True)[0],200)

    def test_revoke_during_generation_aborts_without_committing(self):
        self.create()
        session = self.session()
        self.model.release.clear()
        result = []
        worker = threading.Thread(target=lambda:result.append(self.chat(session)))
        worker.start()
        self.assertTrue(self.model.entered.wait(3))
        self.request('/admin/share/revoke',{},private=True)
        self.model.release.set()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        events = result[0][1]
        self.assertEqual(events[-1]['code'],'share_closed')
        self.assertFalse(any(e['type']=='done' for e in events))
        token = session['cookie'].split('=',1)[1]
        self.assertEqual(self.store.current(token)['messages'],[])

    def test_submitted_snapshots_remain_after_close(self):
        self.create()
        session = self.session()
        events = self.chat(session)[1]
        conversation = events[-1]['conversation']
        body = dict(nickname='Fixture',comment='',consent=True,expectedConversationId=conversation['id'],expectedMessageCount=2)
        self.assertEqual(self.request(self.prefix+'api/submit',body,session=session)[0],200)
        self.request('/admin/share/revoke',{},private=True)
        self.assertEqual(self.request('/admin/status',private=True)[1]['submissionCount'],1)
        self.assertEqual(len(self.request('/admin/submissions',private=True)[1]['submissions'][0]['messages']),2)


if __name__=='__main__':
    unittest.main()
