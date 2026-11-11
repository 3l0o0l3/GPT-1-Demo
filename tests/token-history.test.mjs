// Controller contract tests for the production UI; unpublished replay experiments are excluded.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';

const app=readFileSync(new URL('../dist/training-app.js',import.meta.url),'utf8');
const source=app.split('// TOKEN_HISTORY_CONTROLLER_START')[1].split('// TOKEN_HISTORY_CONTROLLER_END')[0].replaceAll('export ','');
const {readTokenHistory,createTokenHistory}=runInNewContext(`${source}\n({readTokenHistory,createTokenHistory});`,{setTimeout,clearTimeout,Date});
const empty=()=>({version:1,available:false,active:false,frames:[]});
const snapshot=({active=true,steps=[100,200,300],probabilities=[.2,.6,.4],runId='recording-a',...extra}={})=>({
  version:1,available:true,active,runId,text:'A cat',textSha256:'a'.repeat(64),seed:42,source:'pretrained',startStep:100,
  intervalSteps:100,capacity:256,droppedFrames:0,context:128,totalTargets:3,truncated:false,
  tokens:[{id:1,piece:'[BOS]'},{id:10,piece:'A'},{id:11,piece:'Ġcat'}],targets:[{id:10,piece:'A'},{id:11,piece:'Ġcat'},{id:2,piece:'[EOS]'}],
  frames:steps.map((step,i)=>({step,targetProbabilities:[probabilities[i]??.3,.1,.05],topPredictions:[{id:10,piece:'A',p:probabilities[i]??.3},{id:12,piece:'Ġdog',p:.3},{id:2,piece:'[EOS]',p:.05}]})),...extra
});
const defer=()=>{let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return{promise,resolve,reject};};
const clone=value=>JSON.parse(JSON.stringify(value));
function fixture(handler=async()=>snapshot()){
  let time=0,nextId=0;const timers=new Map(),requests=[],events=[];
  const controller=createTokenHistory({request:async(path,body)=>{requests.push({path,body,time});return handler(path,body);},onChange:(state,reason)=>events.push({reason,frame:state.frameIndex}),now:()=>time,
    setTimer:(fn,delay)=>{const id=++nextId;timers.set(id,{fn,at:time+delay,delay});return id;},clearTimer:id=>timers.delete(id)});
  const flush=()=>new Promise(resolve=>setImmediate(resolve));
  const tick=async()=>{const entry=[...timers.entries()].sort((a,b)=>a[1].at-b[1].at)[0];assert.ok(entry,'a timer exists');const [id,timer]=entry;timers.delete(id);time=timer.at;timer.fn();await flush();};
  const start=async()=>{controller.setLocked(false);controller.setVisible(true);await tick();};
  return{controller,timers,requests,events,tick,flush,start,advance:ms=>{time+=ms;}};
}

test('all target probabilities are validated without requiring an improvement',()=>{
  const value=snapshot();assert.equal(readTokenHistory(value),value);assert.equal(readTokenHistory(empty()).available,false);
  for(const mutate of [x=>x.frames[1].targetProbabilities.pop(),x=>x.frames[0].topPredictions[1].p=NaN,x=>x.frames[0].targetProbabilities[0]=1.1,x=>x.frames[1].step=100,x=>x.tokens.pop(),x=>x.truncated=true,x=>x.frames[0].step=99]){
    const bad=clone(value);mutate(bad);assert.throws(()=>readTokenHistory(bad),/invalid token recording/);
  }
});

test('reload retrieves the existing current-session recording with GET only',async()=>{
  const f=fixture();await f.start();assert.equal(f.controller.state.snapshot.runId,'recording-a');assert.equal(f.controller.state.frameIndex,2);
  assert.deepEqual(f.requests.map(x=>[x.path,x.body]),[['token-history',undefined]]);f.controller.dispose();
});

test('polling never overlaps and starts no more than once per second',async()=>{
  const gate=defer(),f=fixture(()=>gate.promise);f.controller.setVisible(true);await f.tick();
  const a=f.controller.refresh(),b=f.controller.refresh();assert.equal(f.requests.length,1);assert.equal(f.controller.state.fetching,true);
  gate.resolve(snapshot());await Promise.all([a,b]);await f.flush();await f.controller.refresh();assert.equal(f.requests.length,1);
  await f.tick();assert.equal(f.requests.length,2);assert.ok(f.requests[1].time-f.requests[0].time>=1000);f.controller.dispose();
});

test('hidden panels cancel polling and pause playback; returning reads current history',async()=>{
  const f=fixture();await f.start();f.controller.play();f.controller.setVisible(false);assert.equal(f.controller.state.playing,false);assert.equal(f.timers.size,0);
  await f.controller.refresh();assert.equal(f.requests.length,1);f.advance(1500);f.controller.setVisible(true);await f.tick();assert.equal(f.requests.length,2);f.controller.dispose();
});

test('playback selects genuine frames, including a decreasing probability, with no mutations',async()=>{
  const f=fixture();await f.start();f.controller.setSpeed(4);f.controller.play();assert.equal(f.controller.state.frameIndex,0);
  await f.tick();assert.equal(f.controller.state.frameIndex,1);assert.equal(f.controller.state.snapshot.frames[1].targetProbabilities[0],.6);
  await f.tick();assert.equal(f.controller.state.frameIndex,2);assert.equal(f.controller.state.snapshot.frames[2].targetProbabilities[0],.4);assert.equal(f.controller.state.playing,false);
  assert.equal(f.requests.filter(x=>x.body!==undefined).length,0);f.controller.dispose();
});

test('training lock blocks start and stop but leaves replay and token selection available',async()=>{
  const f=fixture();await f.start();f.controller.setLocked(true);assert.equal(await f.controller.start('Other'),false);assert.equal(await f.controller.stop(),false);
  f.controller.select(2);f.controller.seek(0);f.controller.play();assert.equal(f.controller.state.position,2);assert.equal(f.controller.state.playing,true);assert.equal(f.requests.length,1);f.controller.dispose();
});

test('a paused replay keeps its measured step when recording appends new frames',async()=>{
  let value=snapshot();const f=fixture(()=>value);await f.start();f.controller.seek(1);
  value=snapshot({steps:[100,200,300,400]});await f.tick();assert.equal(f.controller.state.frameIndex,1);assert.equal(f.controller.state.followLatest,false);
  f.controller.follow(true);assert.equal(f.controller.state.frameIndex,3);f.controller.dispose();
});

test('changing selected text elsewhere cannot relabel captured tokens; a new run resets replay',async()=>{
  let value=snapshot();const f=fixture(()=>value);await f.start();f.controller.select(2);f.controller.seek(1);f.controller.play();
  value=snapshot({runId:'new-run',steps:[100],probabilities:[.12],text:'New fixed text'});f.controller.pause();await f.tick();
  assert.equal(f.controller.state.snapshot.text,'New fixed text');assert.equal(f.controller.state.position,0);assert.equal(f.controller.state.frameIndex,0);assert.equal(f.controller.state.playing,false);f.controller.dispose();
});

test('start records only the supplied fixed text and does not issue training requests',async()=>{
  const f=fixture(async(path,body)=>path==='token-history-start'?snapshot({text:body.text,runId:'new-run',steps:[100]}):empty());await f.start();
  assert.equal(await f.controller.start('A fixed observation.'),true);assert.equal(f.controller.state.snapshot.text,'A fixed observation.');
  assert.deepEqual(clone(f.requests.filter(x=>x.body!==undefined).map(x=>[x.path,x.body])),[['token-history-start',{text:'A fixed observation.'}]]);f.controller.dispose();
});

test('a lost POST response is reconciled with GET and never repeats the mutation',async()=>{
  let recorded=false;const f=fixture(async(path)=>{if(path==='token-history-start'){recorded=true;throw new Error('response lost');}return recorded?snapshot({runId:'started-on-server'}):empty();});await f.start();
  assert.equal(await f.controller.start('A cat'),false);assert.equal(f.controller.state.uncertain,true);assert.equal(await f.controller.start('A cat'),false);
  await f.tick();assert.equal(f.controller.state.uncertain,false);assert.equal(f.controller.state.snapshot.runId,'started-on-server');assert.equal(f.requests.filter(x=>x.path==='token-history-start').length,1);f.controller.dispose();
});

test('malformed GET preserves prior observations and marks the connection stale',async()=>{
  let value=snapshot();const f=fixture(()=>value);await f.start();value={...snapshot(),frames:[]};await f.tick();
  assert.equal(f.controller.state.snapshot.frames.length,3);assert.equal(f.controller.state.connected,false);assert.match(f.controller.state.error,/invalid token recording/);f.controller.dispose();
});

test('stop waits for an in-flight GET, publishes the final frame, and sends one POST',async()=>{
  const gate=defer();let slow=false;const f=fixture(async(path)=>path==='token-history-stop'?snapshot({active:false}):slow?gate.promise:snapshot());await f.start();
  slow=true;await f.tick();const stop=f.controller.stop();assert.equal(f.controller.state.pending,true);assert.equal(f.requests.some(x=>x.path==='token-history-stop'),false);
  gate.resolve(snapshot());await stop;assert.equal(f.controller.state.snapshot.active,false);assert.equal(f.requests.filter(x=>x.path==='token-history-stop').length,1);f.controller.dispose();
});

test('page wiring keeps generation defaults',()=>{
  const html=readFileSync(new URL('../dist/index.html',import.meta.url),'utf8');
  assert.equal(html,readFileSync(new URL('../dist/training.html',import.meta.url),'utf8'));
  assert.match(html,/<input[^>]*id="generation-tokens"[^>]*value="1000"/);assert.match(html,/<option value="immediate" selected>Immediate/);
});
