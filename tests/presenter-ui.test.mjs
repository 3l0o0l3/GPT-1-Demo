import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createSessionTransport} from '../dist/presenter-session.js';
import {mountLogin,validCode} from '../dist/presenter-login.js';

const hostedLocation={protocol:'https:',hostname:'demo.example.com',origin:'https://demo.example.com'};
const session={authenticated:true,email:'fixture@example.org',csrfToken:'fixture-session-csrf',expiresAt:'2030-01-01T00:00:00Z'};
const json=(value,status=200)=>new Response(JSON.stringify(value),{status,headers:{'Content-Type':'application/json'}});

test('local mode uses the page origin and never requests authentication',async()=>{
  const calls=[];const transport=createSessionTransport({location:{protocol:'http:',hostname:'localhost',origin:'http://localhost:8768'},fetch:async(...args)=>{calls.push(args);return json({ok:true});},redirect:()=>assert.fail('Unexpected redirect')});
  assert.equal(transport.hosted,false);assert.equal(transport.trainingBase,'/api/training/');assert.equal(transport.telemetryUrl,'/api/telemetry');
  const options={method:'POST',body:'{}'};await transport.sessionFetch(transport.trainingBase+'train',options);
  assert.equal(calls.length,1);assert.equal(calls[0][1],options);
});
test('hosted mutations share one session check and send CSRF with the browser cookie exactly once',async()=>{
  const calls=[];let resolveSession;const waiting=new Promise(resolve=>{resolveSession=resolve;});
  const transport=createSessionTransport({location:hostedLocation,fetch:async(url,options)=>{calls.push({url,options});return url==='/api/auth/session'?waiting:json({ok:true});},redirect:()=>assert.fail('Unexpected redirect')});
  const a=transport.sessionFetch('/api/training/train',{method:'POST',headers:{'Content-Type':'application/json'},body:'{"steps":10}'}),b=transport.sessionFetch('/api/seminar/share',{method:'POST',body:'{}'});
  assert.equal(calls.length,1);resolveSession(json(session));await Promise.all([a,b]);
  assert.equal(transport.trainingBase,'/api/training/');assert.equal(transport.telemetryUrl,'/api/telemetry');assert.equal(calls.length,3);
  for(const call of calls.slice(1)){assert.equal(call.options.credentials,'same-origin');assert.equal(call.options.headers.get('X-CSRF-Token'),session.csrfToken);assert.equal(call.options.redirect,'error');}
});
test('expired session redirects once and never replays a model mutation',async()=>{
  const calls=[],redirects=[];
  const transport=createSessionTransport({location:hostedLocation,fetch:async(url,options)=>{calls.push({url,options});return url==='/api/auth/session'?json(session):json({error:{code:'login_required'}},401);},redirect:url=>redirects.push(url)});
  await assert.rejects(transport.sessionFetch('/api/training/train',{method:'POST',body:'{}'}),error=>error.status===401&&/may already have completed/.test(error.message));
  assert.equal(calls.filter(c=>c.options.method==='POST').length,1);assert.deepEqual(redirects,['/login?reason=session-expired']);
  await assert.rejects(transport.sessionFetch('/api/training/state'),/session has ended/);assert.equal(redirects.length,1);
});
test('missing authentication prevents the mutation, and transport failures never retry it',async()=>{
  let calls=0;const noSession=createSessionTransport({location:hostedLocation,fetch:async()=>{calls++;return json({authenticated:false,csrfToken:'prelogin'});},redirect:()=>{}});
  await assert.rejects(noSession.sessionFetch('/api/seminar/share',{method:'POST',body:'{}'}),/session has ended/);assert.equal(calls,1);
  const requests=[];const failed=createSessionTransport({location:hostedLocation,fetch:async(url,options)=>{requests.push({url,options});if(url==='/api/auth/session')return json(session);throw new TypeError('Connection lost');},redirect:()=>{}});
  await assert.rejects(failed.sessionFetch('/api/training/commit',{method:'POST',body:'{}'}),/Connection lost/);assert.equal(requests.length,2);
});
test('CSRF cannot be sent to another origin and sign out uses the protected session',async()=>{
  const calls=[],redirects=[];const transport=createSessionTransport({location:hostedLocation,fetch:async(url,options)=>{calls.push({url,options});return url==='/api/auth/session'?json(session):json({ok:true});},redirect:url=>redirects.push(url)});
  await assert.rejects(transport.sessionFetch('https://attacker.example/api',{method:'POST'}),/stay on this website/);assert.equal(calls.length,0);
  await transport.logout();assert.equal(calls[1].url,hostedLocation.origin+'/api/auth/logout');assert.equal(calls[1].options.headers.get('X-CSRF-Token'),session.csrfToken);assert.deepEqual(redirects,['/login?reason=signed-out']);
});

function loginFixture({configured=true,verifyStatus=200,initialSession,search=''}={}){
  const ids=['request-form','login-email','request-code','verify-form','login-code','verify-code','change-email','login-status','resend-note','login-notice'];
  const nodes=new Map(ids.map(id=>[id,{id,value:'',hidden:['verify-form','login-notice'].includes(id),disabled:false,textContent:'',events:{},addEventListener(name,callback){this.events[name]=callback;},classList:{toggle(){}},focus(){this.focused=true;},checkValidity(){return this.value.includes('@');}}]));
  const requests=[],destinations=[],intervals=[];let now=1900000000000;
  const document={getElementById:id=>nodes.get(id),addEventListener(){}};
  const fetch=async(url,options)=>{requests.push({url,options});
    if(url.endsWith('/session'))return json(initialSession||{authenticated:false,csrfToken:'prelogin-csrf',smtpConfigured:configured});
    if(url.endsWith('/request-code'))return json({challengeId:'challenge-1',expiresIn:600,retryAfter:60,message:'If this address is approved, a code has been sent.'});
    if(url.endsWith('/verify'))return verifyStatus===200?json(session):json({error:{code:'invalid_code',message:'Invalid or expired code.'}},verifyStatus);
    assert.fail('Unexpected request '+url);
  };
  mountLogin({document,location:{search,replace:url=>destinations.push(url)},fetch,now:()=>now,setInterval:callback=>intervals.push(callback),clearInterval(){}});
  return {nodes,requests,destinations,intervals,advance:ms=>{now+=ms;},async event(id,event='submit'){await nodes.get(id).events[event]({preventDefault(){}});}};
}
const settle=async()=>{for(let i=0;i<12;i++)await Promise.resolve();};

test('login requests an email-bound challenge and verifies one 8-digit code through the real UI handlers',async()=>{
  const view=loginFixture();await settle();assert.equal(view.nodes.get('request-code').disabled,false);
  view.nodes.get('login-email').value='presenter@example.org';await view.event('request-form');
  assert.equal(view.nodes.get('verify-form').hidden,false);assert.equal(view.nodes.get('request-code').disabled,true);
  assert.match(view.nodes.get('login-status').textContent,/If this address is approved/);
  view.nodes.get('login-code').value='1234';await view.event('login-code','input');assert.equal(view.nodes.get('verify-code').disabled,true);
  view.nodes.get('login-code').value='12345678';await view.event('login-code','input');assert.equal(view.nodes.get('verify-code').disabled,false);await view.event('verify-form');
  const posts=view.requests.filter(r=>r.options.method==='POST');assert.equal(posts.length,2);
  assert.deepEqual(JSON.parse(posts[0].options.body),{email:'presenter@example.org'});assert.deepEqual(JSON.parse(posts[1].options.body),{challengeId:'challenge-1',code:'12345678'});
  for(const post of posts)assert.equal(post.options.headers['X-CSRF-Token'],'prelogin-csrf');
  assert.equal(view.nodes.get('login-code').value,'');assert.deepEqual(view.destinations,['/']);
});
test('login cannot send when SMTP is unconfigured; an existing authenticated session opens the workspace',async()=>{
  const unavailable=loginFixture({configured:false});await settle();assert.equal(unavailable.nodes.get('request-code').disabled,true);await unavailable.event('request-form');assert.equal(unavailable.requests.length,1);
  const authenticated=loginFixture({initialSession:session});await settle();assert.deepEqual(authenticated.destinations,['/']);
});
test('bad codes remain on login, clear the input and are never retried automatically',async()=>{
  const view=loginFixture({verifyStatus:401});await settle();view.nodes.get('login-email').value='presenter@example.org';await view.event('request-form');view.nodes.get('login-code').value='12345678';await view.event('login-code','input');await view.event('verify-form');
  assert.equal(view.nodes.get('login-code').value,'');assert.equal(view.nodes.get('verify-code').disabled,true);assert.match(view.nodes.get('login-status').textContent,/Invalid or expired/);assert.equal(view.destinations.length,0);assert.equal(view.requests.filter(r=>r.url.endsWith('/verify')).length,1);
});
test('resend cooldown and code expiry use time without automatic email requests',async()=>{
  const view=loginFixture();await settle();view.nodes.get('login-email').value='presenter@example.org';await view.event('request-form');view.advance(60000);view.intervals[0]();assert.equal(view.nodes.get('request-code').disabled,false);
  view.nodes.get('login-code').value='12345678';await view.event('login-code','input');view.advance(540000);view.intervals[0]();assert.equal(view.nodes.get('verify-code').disabled,true);assert.equal(view.nodes.get('login-code').value,'');assert.match(view.nodes.get('login-status').textContent,/expired/);assert.equal(view.requests.length,2);
});
test('changing email discards the previous challenge, and expired-session notice warns against repeating work',async()=>{
  const view=loginFixture({search:'?reason=session-expired'});await settle();assert.equal(view.nodes.get('login-notice').hidden,false);assert.match(view.nodes.get('login-notice').textContent,/may already have completed/);
  view.nodes.get('login-email').value='one@example.org';await view.event('request-form');view.nodes.get('login-email').value='two@example.org';await view.event('login-email','input');assert.equal(view.nodes.get('verify-form').hidden,true);assert.equal(view.nodes.get('verify-code').disabled,true);
});
test('login assets never embed the whitelist or persist secrets, and accept only eight decimal digits',()=>{
  for(const code of ['12345678','00000000'])assert.equal(validCode(code),true);
  for(const code of ['1234567','123456789','1234 5678','1234567a',12345678])assert.equal(validCode(code),false);
  for(const path of ['presenter-login.html','presenter-login.js','presenter-session.js']){
    const source=readFileSync(new URL('../dist/'+path,import.meta.url),'utf8');assert.doesNotMatch(source,/presenter@example\.com|collaborator@example\.com|localStorage|sessionStorage|innerHTML|insertAdjacentHTML/);
  }
});
