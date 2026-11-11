import test from 'node:test';
import assert from 'node:assert/strict';
import {createTokenPlayback} from '../dist/token-playback.js';

const settle=async()=>{for(let i=0;i<5;i++)await Promise.resolve();};
function deferred(){let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return{promise,resolve,reject};}
function clock(){
  let now=0,next=0;const timers=new Map();
  return{setTimer(fn,delay){const id=++next;timers.set(id,{fn,at:now+delay});return id;},clearTimer(id){timers.delete(id);},
    async advance(ms){const end=now+ms;while(true){const entry=[...timers].filter(([,t])=>t.at<=end).sort((a,b)=>a[1].at-b[1].at)[0];if(!entry)break;now=entry[1].at;timers.delete(entry[0]);entry[1].fn();await settle();}now=end;await settle();}};
}
function harness(requestBatch,rate=2){const time=clock(),seen=[],errors=[];const player=createTokenPlayback({requestBatch,rate,onToken:(token,meta)=>seen.push({token,meta}),onError:error=>errors.push(error),...time});return{player,time,seen,errors};}

test('reveals actual token records at the selected rate and drains EOS before completing',async()=>{
  const batch=[{id:1,p:.3},{id:2,p:.6},{id:9,p:1}];const h=harness(async()=>({tokens:batch,finished:true}));
  h.player.start();await settle();assert.equal(h.player.snapshot().phase,'playing');assert.equal(h.seen.length,0);
  await h.time.advance(499);assert.equal(h.seen.length,0);await h.time.advance(1);assert.equal(h.seen.length,1);
  assert.equal(h.player.snapshot().phase,'playing');await h.time.advance(1000);
  assert.deepEqual(h.seen.map(item=>item.token),batch);assert.equal(h.seen[2].meta.finished,true);assert.equal(h.player.snapshot().phase,'complete');
});

test('pause while a request is pending buffers its result and resume neither repeats nor loses it',async()=>{
  const pending=deferred();let calls=0;const h=harness(()=>{calls++;return pending.promise;});h.player.start();h.player.pause();
  pending.resolve({tokens:[{id:4},{id:5}],finished:true});await settle();await h.time.advance(5000);
  assert.equal(h.seen.length,0);assert.equal(h.player.snapshot().queued,2);assert.equal(h.player.snapshot().phase,'paused');
  h.player.start();await h.time.advance(1000);assert.deepEqual(h.seen.map(item=>item.token.id),[4,5]);assert.equal(calls,1);
});

test('pause during reveal is immediate and changing the rate affects resumed intervals',async()=>{
  const h=harness(async()=>({tokens:[{id:1},{id:2},{id:3}],finished:true}));h.player.start();await settle();await h.time.advance(500);
  h.player.pause();await h.time.advance(5000);assert.equal(h.seen.length,1);h.player.setRate(4);h.player.start();
  await h.time.advance(249);assert.equal(h.seen.length,1);await h.time.advance(1);assert.equal(h.seen.length,2);
  h.player.setRate(8);await h.time.advance(125);assert.equal(h.seen.length,3);assert.equal(h.player.snapshot().phase,'complete');
});

test('reset invalidates a late result while preserving the no-overlap request invariant',async()=>{
  const first=deferred(),second=deferred();let calls=0;const h=harness(()=>{calls++;return calls===1?first.promise:second.promise;});
  h.player.start();h.player.reset();h.player.start();assert.equal(calls,1);assert.equal(h.player.snapshot().inFlight,true);
  first.resolve({tokens:[{id:111}],finished:true});await settle();assert.equal(calls,2);assert.equal(h.seen.length,0);
  second.resolve({tokens:[{id:222}],finished:true});await settle();await h.time.advance(500);
  assert.deepEqual(h.seen.map(item=>item.token.id),[222]);
});

test('a new batch is requested only after the previous queue is revealed',async()=>{
  let calls=0;const h=harness(async()=>{calls++;return{tokens:[{id:calls}],finished:calls===2};});
  h.player.start();await settle();assert.equal(calls,1);await h.time.advance(499);assert.equal(calls,1);
  await h.time.advance(1);assert.equal(calls,2);await h.time.advance(500);assert.deepEqual(h.seen.map(item=>item.token.id),[1,2]);
});

test('provider errors stop automatic requests until explicit resume',async()=>{
  let calls=0;const error=new Error('offline');const h=harness(async()=>{calls++;if(calls===1)throw error;return{tokens:[{id:2}],finished:true};});
  h.player.start();await settle();await h.time.advance(5000);assert.equal(calls,1);assert.equal(h.player.snapshot().phase,'error');assert.deepEqual(h.errors,[error]);
  h.player.start();await settle();await h.time.advance(500);assert.deepEqual(h.seen.map(item=>item.token.id),[2]);
});

test('context limit has a distinct completion reason after the final buffered token',async()=>{
  const h=harness(async()=>({tokens:[{id:1}],finished:false,limit:true}));h.player.start();await settle();assert.notEqual(h.player.snapshot().phase,'complete');
  await h.time.advance(500);assert.equal(h.player.snapshot().reason,'limit');assert.equal(h.seen[0].meta.finished,false);assert.equal(h.player.snapshot().phase,'complete');
});

test('reset cancels a reveal timer and empty nonterminal responses stop instead of looping',async()=>{
  const h=harness(async()=>({tokens:[{id:1}],finished:true}));h.player.start();await settle();h.player.reset();await h.time.advance(1000);assert.equal(h.seen.length,0);
  let calls=0;const empty=harness(async()=>{calls++;return{tokens:[],finished:false};});empty.player.start();await settle();assert.equal(empty.player.snapshot().phase,'error');assert.equal(calls,1);
});
