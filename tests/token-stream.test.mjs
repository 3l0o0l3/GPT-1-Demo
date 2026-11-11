import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {readGenerationStream,TokenPlayer} from '../dist/token-stream.js';

const source={model:'live',step:1000,seed:42,source:'random'};
const sampling={temperature:.8,seed:42};
const start={type:'start',text:'Once 🐈',source,sampling,maxNewTokens:48};
const tokenEvents=[
  {type:'token',index:0,token:{id:301,piece:'byte_a',p:.2},continuation:' �'},
  {type:'token',index:1,token:{id:302,piece:'byte_b',p:.3},continuation:' 猫'},
  {type:'token',index:2,token:{id:2,piece:'[EOS]',p:.4},continuation:' 猫'}
];
const result={text:start.text,continuation:' 猫',fullText:'Once 🐈 猫',tokens:tokenEvents.map(e=>e.token),stopReason:'eos',source,sampling};
const done={type:'done',result};
const encode=events=>new TextEncoder().encode(events.map(e=>JSON.stringify(e)+'\n').join(''));
function response(chunks){return new Response(new ReadableStream({start(c){for(const chunk of chunks)c.enqueue(chunk);c.close();}}),{headers:{'content-type':'application/x-ndjson; charset=utf-8'}});}
const flush=()=>new Promise(resolve=>setImmediate(resolve));

class Clock {
  now=0;next=0;timers=new Map();callbacks=[];
  set=(fn,delay)=>{const id=++this.next;this.timers.set(id,{at:this.now+delay,fn});this.callbacks.push(fn);return id;};
  clear=id=>this.timers.delete(id);
  tick(ms){const end=this.now+ms;while(true){const entry=[...this.timers].filter(([,t])=>t.at<=end).sort((a,b)=>a[1].at-b[1].at)[0];if(!entry)break;this.now=entry[1].at;this.timers.delete(entry[0]);entry[1].fn();}this.now=end;}
}
function player(clock,extra={}){return new TokenPlayer({setTimer:clock.set,clearTimer:clock.clear,...extra});}

test('incremental NDJSON survives every UTF-8 byte split and preserves exact tokens/provenance',async()=>{
  const bytes=encode([start,...tokenEvents,done]),events=[];
  const actual=await readGenerationStream(response([...bytes].map(byte=>Uint8Array.of(byte))),{onEvent:event=>events.push(event)});
  assert.deepEqual(actual,result);assert.deepEqual(events,[start,...tokenEvents,done]);
});

test('tokens are delivered before EOF; done is accepted only after stream closure',async()=>{
  let controller;const res=new Response(new ReadableStream({start(c){controller=c;}}),{headers:{'content-type':'application/x-ndjson'}}),events=[];
  const reading=readGenerationStream(res,{onEvent:e=>events.push(e.type)});
  controller.enqueue(encode([start,tokenEvents[0]]));await flush();assert.deepEqual(events,['start','token']);
  controller.enqueue(encode([...tokenEvents.slice(1),done]));await flush();assert.deepEqual(events,['start','token','token','token']);
  controller.close();await reading;assert.equal(events.at(-1),'done');
});

test('EOF and explicit server errors are failures, with no completion event',async()=>{
  for(const tail of [[],[{type:'error',error:{code:'model_failure',message:'GPU stream failed'}}]]){
    const events=[];await assert.rejects(readGenerationStream(response([encode([start,tokenEvents[0],...tail])]),{onEvent:e=>events.push(e.type)}),tail.length?/GPU stream failed/:/before completion/);assert.deepEqual(events,['start','token']);
  }
});

test('rejects out-of-order tokens, mismatched terminal data, trailing events and partial lines',async()=>{
  const cases=[
    [encode([start,{...tokenEvents[0],index:1}])],
    [encode([start,...tokenEvents,{type:'done',result:{...result,continuation:'invented'}}])],
    [encode([start,...tokenEvents,done,{type:'token',...tokenEvents[0]}])],
    [new TextEncoder().encode(JSON.stringify(start))],
    [encode([{...start,maxNewTokens:1001}])]
  ];
  for(const chunks of cases)await assert.rejects(readGenerationStream(response(chunks)));
});

test('HTTP errors and JSON-only success never become simulated streams',async()=>{
  await assert.rejects(readGenerationStream(new Response(JSON.stringify({error:{code:'invalid_request',message:'Maximum 1000 tokens'}}),{status:400})),/Maximum 1000/);
  await assert.rejects(readGenerationStream(new Response(JSON.stringify(result),{headers:{'content-type':'application/json'}})),/not return a token stream/);
});

test('abort cancels pending reads and rejects without accepting late events',async()=>{
  let controller,cancelled=false;const abort=new AbortController();
  const res=new Response(new ReadableStream({start(c){controller=c;},cancel(){cancelled=true;}}),{headers:{'content-type':'application/x-ndjson'}}),events=[];
  const reading=readGenerationStream(res,{signal:abort.signal,onEvent:e=>events.push(e.type)});controller.enqueue(encode([start,tokenEvents[0]]));await flush();abort.abort();
  await assert.rejects(reading,{name:'AbortError'});assert.equal(cancelled,true);assert.deepEqual(events,['start','token']);
});

test('default pace is 8 real tokens/s; decoded text replaces the cumulative continuation',async()=>{
  const clock=new Clock(),observed=[],p=player(clock,{onToken:e=>observed.push(e.token.id)});p.start(start);tokenEvents.forEach(e=>p.enqueue(e));p.networkDone(result);
  assert.equal(p.snapshot().status,'revealing');clock.tick(124);assert.equal(p.snapshot().displayed,0);clock.tick(1);assert.equal(p.snapshot().continuation,' �');
  clock.tick(125);assert.equal(p.snapshot().continuation,' 猫');assert.equal(p.snapshot().status,'revealing');clock.tick(125);
  assert.deepEqual(observed,[301,302,2]);assert.equal((await p.finished).status,'done');assert.equal(p.snapshot().continuation,' 猫');
});

test('default timer adapters do not bind native browser timer calls to the player',()=>{
  const realSet=globalThis.setTimeout,realClear=globalThis.clearTimeout,clock=new Clock();let scheduled=0,cancelled=0;
  globalThis.setTimeout=function(fn,delay){assert.equal(this,undefined,'native setTimeout received an illegal receiver');scheduled++;return clock.set(fn,delay);};
  globalThis.clearTimeout=function(timer){assert.equal(this,undefined,'native clearTimeout received an illegal receiver');cancelled++;return clock.clear(timer);};
  try{const p=new TokenPlayer();p.start(start);p.enqueue(tokenEvents[0]);assert.equal(scheduled,1);p.pause();assert.equal(cancelled,1);p.resume();clock.tick(125);assert.equal(p.snapshot().displayed,1);p.stop();}
  finally{globalThis.setTimeout=realSet;globalThis.clearTimeout=realClear;}
});

test('pause buffers incoming tokens and completion waits for resume and the empty queue',async()=>{
  const clock=new Clock(),p=player(clock);p.start(start);p.enqueue(tokenEvents[0]);p.pause();tokenEvents.slice(1).forEach(e=>p.enqueue(e));p.networkDone(result);clock.tick(10000);
  assert.equal(p.snapshot().status,'paused');assert.equal(p.snapshot().received,3);assert.equal(p.snapshot().displayed,0);assert.equal(p.terminal,null);
  p.resume();clock.tick(375);assert.equal((await p.finished).status,'done');assert.equal(p.snapshot().displayed,3);
});

test('speed changes cancel old timers; immediate flush respects pause',async()=>{
  const clock=new Clock(),p=player(clock);p.start(start);tokenEvents.forEach(e=>p.enqueue(e));p.networkDone(result);const oldTimer=clock.callbacks.at(-1);
  p.setRate(2);oldTimer();assert.equal(p.snapshot().displayed,0);clock.tick(499);assert.equal(p.snapshot().displayed,0);clock.tick(1);assert.equal(p.snapshot().displayed,1);
  p.pause();p.setRate(Infinity);clock.tick(10000);assert.equal(p.snapshot().displayed,1);p.resume();assert.equal((await p.finished).status,'done');assert.equal(p.snapshot().displayed,3);
});

test('stop drops hidden tokens, retains visible partial text, and ignores stale timers/events',async()=>{
  const clock=new Clock(),seen=[],p=player(clock,{onToken:e=>seen.push(e.token.id)});p.start(start);tokenEvents.forEach(e=>p.enqueue(e));clock.tick(125);const lateTimer=clock.callbacks.at(-1);p.stop();lateTimer();p.enqueue(tokenEvents[0]);p.networkDone(result);p.resume();clock.tick(10000);
  const state=await p.finished;assert.equal(state.status,'stopped');assert.equal(state.displayed,1);assert.equal(state.queued,0);assert.equal(state.continuation,' �');assert.deepEqual(seen,[301]);
});

test('failure preserves already revealed text without declaring buffered output complete',async()=>{
  const clock=new Clock(),p=player(clock);p.start(start);tokenEvents.forEach(e=>p.enqueue(e));clock.tick(125);p.fail(new Error('Truncated stream'));clock.tick(10000);p.networkDone(result);
  const state=await p.finished;assert.equal(state.status,'failed');assert.equal(state.displayed,1);assert.equal(state.queued,0);assert.equal(state.error,'Truncated stream');
});

test('independent comparison queues allow the next network stream before first reveal completes',async()=>{
  const clock=new Clock(),a=player(clock,{rate:1}),b=player(clock,{rate:32});a.start(start);tokenEvents.forEach(e=>a.enqueue(e));a.networkDone(result);
  b.start({...start,source:{...source,model:'random'}});tokenEvents.forEach(e=>b.enqueue(e));b.networkDone({...result,source:{...source,model:'random'}});clock.tick(100);
  assert.equal(b.snapshot().status,'done');assert.equal(a.snapshot().displayed,0);clock.tick(2900);assert.equal(a.snapshot().status,'done');
});

function windowRun(count=3,{mode='sliding',windowSize=2,promptTokens=5}={}){
  const context={mode,windowSize,modelContext:128,promptTokens};
  const initial={...start,maxNewTokens:1000,context},events=Array.from({length:count},(_,index)=>({type:'token',index,token:{id:300+index,piece:' token',p:.1},continuation:' token'.repeat(index+1),window:{inputTokens:Math.min(windowSize,promptTokens+index),droppedTokens:Math.max(0,promptTokens+index-windowSize)}}));
  const final={text:initial.text,continuation:events.at(-1)?.continuation??'',fullText:initial.text+(events.at(-1)?.continuation??''),tokens:events.map(e=>e.token),stopReason:'length',source,sampling,context,window:events.at(-1)?.window??null};
  return {initial,events,final};
}

test('accepts 1000 actual sliding tokens across 128 positions and a total stream larger than 1 MiB',async()=>{
  const {initial,events,final}=windowRun(1000,{mode:'sliding',windowSize:128,promptTokens:180}),bytes=encode([initial,...events,{type:'done',result:final}]);
  assert.ok(bytes.length>1048576);let tokens=0;
  const chunks=[];for(let at=0;at<bytes.length;at+=1537)chunks.push(bytes.slice(at,at+1537));
  const actual=await readGenerationStream(response(chunks),{onEvent:e=>{if(e.type==='token')tokens++;}});
  assert.equal(tokens,1000);assert.deepEqual(actual.context,initial.context);assert.deepEqual(actual.window,{inputTokens:128,droppedTokens:1051});assert.equal(actual.tokens.length,1000);
});

test('fixed context keeps the 128-position boundary even with a 1000-token requested budget',async()=>{
  const {initial,events,final}=windowRun(2,{mode:'fixed',windowSize:128,promptTokens:126});
  assert.deepEqual((await readGenerationStream(response([encode([initial,...events,{type:'done',result:final}])]))).window,{inputTokens:127,droppedTokens:0});
  const tooLong=windowRun(3,{mode:'fixed',windowSize:128,promptTokens:126});await assert.rejects(readGenerationStream(response([encode([tooLong.initial,...tooLong.events,{type:'done',result:tooLong.final}])])),/window counts/);
});

test('rejects invalid window contexts, mismatched per-token counts and final provenance',async()=>{
  const {initial,events,final}=windowRun();
  for(const windowSize of [0,129,true,'128'])await assert.rejects(readGenerationStream(response([encode([{...initial,context:{...initial.context,windowSize}}])])),/invalid start event or context/);
  for(const context of [{...initial.context,mode:'other'},{...initial.context,modelContext:256},{...initial.context,promptTokens:0},{...initial.context,mode:'fixed',windowSize:64}])await assert.rejects(readGenerationStream(response([encode([{...initial,context}])])),/invalid start event or context/);
  await assert.rejects(readGenerationStream(response([encode([initial,{...events[0],window:{inputTokens:2,droppedTokens:99}}])])),/window counts/);
  await assert.rejects(readGenerationStream(response([encode([initial,...events,{type:'done',result:{...final,window:{inputTokens:1,droppedTokens:5}}}])])),/does not match/);
  await assert.rejects(readGenerationStream(response([encode([initial,...events,{type:'done',result:{...final,context:{...final.context,windowSize:3}}}])])),/does not match/);
});

test('1 MiB line bound counts UTF-8 bytes rather than decoded characters',async()=>{
  const text='猫'.repeat(400000);assert.ok(text.length<1048576);
  await assert.rejects(readGenerationStream(response([encode([{...start,text}])])),/1 MiB/);
});

test('window counts follow only revealed tokens through pause, resume and stop',async()=>{
  const {initial,events,final}=windowRun(),clock=new Clock(),p=player(clock);p.start(initial);p.pause();events.forEach(e=>p.enqueue(e));p.networkDone(final);
  assert.equal(p.snapshot().window,null);clock.tick(10000);assert.equal(p.snapshot().window,null);p.resume();clock.tick(125);
  assert.deepEqual(p.snapshot().window,{inputTokens:2,droppedTokens:3});p.pause();clock.tick(10000);assert.deepEqual(p.snapshot().window,{inputTokens:2,droppedTokens:3});p.resume();clock.tick(125);
  assert.deepEqual(p.snapshot().window,{inputTokens:2,droppedTokens:4});p.stop();assert.deepEqual((await p.finished).window,{inputTokens:2,droppedTokens:4});assert.equal(p.snapshot().displayed,2);
});

test('HTML entry points agree and expose bounded default-on sliding controls with valid DOM IDs',async()=>{
  const [html,index,js]=await Promise.all(['../dist/training.html','../dist/index.html','../dist/training-app.js'].map(path=>readFile(new URL(path,import.meta.url),'utf8')));
  const hash=s=>createHash('sha256').update(s).digest('hex');assert.equal(hash(html),hash(index));
  const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);assert.equal(new Set(ids).size,ids.length);
  // The retired recording view is optional and returns before reading its removed DOM.
  const optionalView=js.slice(js.indexOf('function createTokenHistoryView(){'),js.indexOf('// SHORT_EXAMPLES_START'));
  assert.match(optionalView,/const panel=\$\('token-history-panel'\);if\(!panel\)return null;/);
  const activeJs=js.replace(optionalView,'');
  for(const [,id]of activeJs.matchAll(/\$\('([^']+)'\)/g))assert.ok(ids.includes(id),`Missing ${id}`);
  const toggle=html.match(/<input\b[^>]*\bid="generation-sliding"[^>]*>/)?.[0];assert.ok(toggle);assert.match(toggle,/\bchecked\b/);
  assert.match(html,/<input\b[^>]*id="generation-window-size"[^>]*min="1"[^>]*max="128"[^>]*value="128"/);
  const windowInput=html.match(/<input\b[^>]*id="generation-window-size"[^>]*>/)[0];assert.doesNotMatch(windowInput,/\bdisabled\b/);
  assert.match(html,/<input\b[^>]*id="generation-temperature"[^>]*value="0\.8"/);
  assert.match(html,/<input\b[^>]*id="generation-tokens"[^>]*min="1"[^>]*max="1000"[^>]*value="1000"/);
  assert.match(js,/generation-window-size'\)\.disabled=liveLocked\|\|!\$\('generation-sliding'\)\.checked/);
});
