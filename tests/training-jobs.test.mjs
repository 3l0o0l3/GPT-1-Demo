import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {createTrainingJobs,readJob,jobIsActive,trainingMode,runTeachingSteps} from '../dist/training-jobs.js';

const REQUEST='0f317bb0-7b5d-407d-a0a6-27a91a5b6e89',ID='05a53a2c-34a8-4f5b-bb97-14408611bd20';
function job({steps=1000,done=0,status='running',kind='lm',id=ID,requestId=REQUEST,...extra}={}){
  const state={step:100+done,history:[{step:100+done,trainLoss:4.1,valLoss:4.2}],metrics:{trainLoss:4.1,valLoss:4.2}};
  return {id,requestId,kind,status,requestedSteps:steps,completedSteps:done,initialStep:100,currentStep:100+done,learningRate:.0003,
    startedAt:'2026-11-04T00:00:00Z',updatedAt:`2026-11-04T00:00:${String(Math.min(59,Math.floor(done/10))).padStart(2,'0')}Z`,
    elapsedSeconds:done/10,stepsPerSecond:10,liveState:kind==='lm'?state:{...state,step:100},
    transferState:kind==='transfer'?{...state,available:true,branches:[]}:{available:false},error:null,...extra};
}
function fixture(handler=async()=>({job:null})){
  let time=0,serial=0;const timers=new Map(),requests=[],events=[],errors=[];
  const controller=createTrainingJobs({request:async(path,body)=>{requests.push({path,body});return handler(path,body);},
    onUpdate:value=>events.push(value),onError:(error,context)=>errors.push({error,context}),newId:()=>REQUEST,now:()=>time,
    setTimer:(fn,delay)=>{const id=++serial;timers.set(id,{fn,delay});return id;},clearTimer:id=>timers.delete(id)});
  const flush=()=>new Promise(resolve=>setImmediate(resolve));
  const tick=async()=>{const [id,timer]=timers.entries().next().value||[];assert.ok(timer,'a polling timer is scheduled');timers.delete(id);time+=timer.delay;timer.fn();await flush();};
  return{controller,requests,events,errors,timers,tick,flush,advance:ms=>{time+=ms;}};
}
const deferred=()=>{let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return{promise,resolve,reject};};

test('1000 steps is one start POST followed only by 100 ms read-only snapshots',async()=>{
  let done=0;const f=fixture(async(path,body)=>path==='job-start'?{job:job({requestId:body.requestId})}:{job:job({done:done+=250,status:done===1000?'completed':'running'})});
  assert.equal(await f.controller.start({kind:'lm',steps:1000,learningRate:.0003}),true);
  for(let i=0;i<4;i++){assert.equal([...f.timers.values()][0].delay,100);await f.tick();}
  assert.equal(f.controller.state.job.completedSteps,1000);assert.equal(f.controller.state.active,false);
  assert.equal(f.requests.filter(r=>r.body!==undefined).length,1);assert.deepEqual(f.requests[0],{path:'job-start',body:{kind:'lm',steps:1000,learningRate:.0003,requestId:REQUEST}});
  assert.deepEqual(f.events.filter(e=>e.changed).map(e=>e.job.completedSteps),[0,250,500,750,1000]);f.controller.dispose();
});

test('double-click while start is pending cannot submit a second task',async()=>{
  const gate=deferred(),f=fixture(()=>gate.promise),first=f.controller.start({kind:'lm',steps:1000,learningRate:.0003});
  assert.equal(f.controller.state.pending,true);assert.equal(await f.controller.start({kind:'lm',steps:1000,learningRate:.0003}),false);
  assert.equal(f.requests.length,1);gate.resolve({job:job()});await first;assert.equal(f.controller.state.pending,false);f.controller.dispose();
});

test('reload adopts running LM or transfer jobs with no start request',async()=>{
  for(const kind of ['lm','transfer']){const f=fixture(async()=>({job:job({kind,done:430})}));
    await f.controller.discover();assert.equal(f.controller.state.active,true);assert.equal(f.controller.state.job.kind,kind);
    assert.equal(f.events.at(-1).source,'discovery');assert.deepEqual(f.requests,[{path:'job',body:undefined}]);f.controller.dispose();}
});

test('slow progress GETs never overlap, including discovery followed by polling',async()=>{
  const gate=deferred(),f=fixture(()=>gate.promise),discovery=f.controller.discover();
  const a=f.controller.poll(),b=f.controller.poll();assert.equal(f.requests.length,1);
  gate.resolve({job:job()});await Promise.all([discovery,a,b]);assert.equal(f.requests.length,1);assert.equal(f.timers.size,1);f.controller.dispose();
});

test('pause sends one explicit job ID and keeps progress active until server pauses',async()=>{
  const gate=deferred();const f=fixture(async(path)=>path==='job-pause'?gate.promise:{job:job({done:20})});await f.controller.discover();
  const pausing=f.controller.pause();assert.equal(await f.controller.pause(),false);
  gate.resolve({job:job({done:30,status:'pause_requested'})});await pausing;
  assert.equal(f.controller.state.active,true);assert.equal(await f.controller.pause(),false);
  assert.deepEqual(f.requests.filter(r=>r.body),[{path:'job-pause',body:{jobId:ID}}]);f.controller.dispose();
});

test('disconnect retains active progress, retries GET and never repeats POST',async()=>{
  let fail=false;const f=fixture(async()=>{if(fail)throw new TypeError('offline');return{job:job({done:20})};});
  await f.controller.discover();fail=true;await f.tick();assert.equal(f.controller.state.active,true);assert.equal(f.controller.state.connected,false);
  fail=false;await f.tick();assert.equal(f.controller.state.connected,true);assert.equal(f.requests.every(r=>r.body===undefined),true);f.controller.dispose();
});

test('lost start response recovers its existing task without sending start twice',async()=>{
  const f=fixture(async(path)=>{if(path==='job-start')throw new TypeError('response lost');return{job:job({done:80})};});
  assert.equal(await f.controller.start({kind:'lm',steps:1000,learningRate:.0003}),false);assert.equal(f.controller.state.uncertain,true);
  assert.equal(await f.controller.start({kind:'lm',steps:1000,learningRate:.0003}),false);
  await f.tick();assert.equal(f.controller.state.uncertain,false);assert.equal(f.controller.state.job.completedSteps,80);
  assert.equal(f.requests.filter(r=>r.path==='job-start').length,1);f.controller.dispose();
});

test('an unconfirmed start with no matching task stays explicit and does not retry',async()=>{
  const f=fixture(async(path)=>{if(path==='job-start')throw new TypeError('offline');return{job:null};});
  await f.controller.start({kind:'lm',steps:1000,learningRate:.0003});await f.tick();await f.tick();
  assert.equal(f.controller.state.uncertain,true);assert.equal(f.requests.filter(r=>r.body).length,1);f.controller.dispose();
});

test('an explicit server rejection can be refreshed without automatically resubmitting',async()=>{
  const f=fixture(async(path)=>{if(path==='job-start')throw Object.assign(new Error('busy'),{status:409});return{job:job({requestId:'other',done:30})};});
  await f.controller.start({kind:'lm',steps:1000,learningRate:.0003});assert.equal(f.controller.state.uncertain,false);
  await f.tick();assert.equal(f.controller.state.active,true);assert.equal(f.requests.filter(r=>r.body).length,1);f.controller.dispose();
});

test('older or missing progress cannot pretend an active task completed',async()=>{
  let value=job({done:50});const f=fixture(async()=>({job:value}));await f.controller.discover();
  value=job({done:40});await f.tick();assert.equal(f.controller.state.job.completedSteps,50);assert.equal(f.errors.length,1);
  value=null;await f.tick();assert.equal(f.controller.state.active,true);assert.equal(f.errors.length,2);
  await f.controller.discover({reset:true});assert.equal(f.controller.state.job,null);f.controller.dispose();
});

test('failed and paused responses retain real partial steps without fabricated points',async()=>{
  for(const status of ['failed','paused']){const result=job({done:30,status,error:status==='failed'?{code:'training_failed',message:'Finite-check stopped training'}:null});
    const f=fixture(async()=>({job:result}));await f.controller.discover();assert.equal(f.controller.state.active,false);
    assert.deepEqual(f.controller.state.job.liveState.history,[{step:130,trainLoss:4.1,valLoss:4.2}]);f.controller.dispose();}
});

test('unchanged terminal jobs are identifiable so plots do not overwrite later manual changes',async()=>{
  const f=fixture(async()=>({job:job({done:1000,status:'completed'})}));await f.controller.discover();await f.tick();
  assert.equal(f.events[0].source,'discovery');assert.equal(f.events[0].changed,true);assert.equal(f.events.at(-1).changed,false);f.controller.dispose();
});

test('disposing the view cancels polling only; it never pauses server training',async()=>{
  const f=fixture(async()=>({job:job()}));await f.controller.discover();f.controller.dispose();
  assert.equal(f.timers.size,0);await f.controller.poll();assert.deepEqual(f.requests,[{path:'job',body:undefined}]);
});

test('validation rejects malformed progress and mismatched current-step accounting',()=>{
  assert.equal(readJob({job:null}),null);assert.equal(jobIsActive(job()),true);
  for(const value of [{}, {job:job({done:1001})},{job:job({status:'completed'})},{job:job({elapsedSeconds:NaN})},{job:job({currentStep:999})}])assert.throws(()=>readJob(value));
});

test('actual page wiring uses job controller, keeps entry pages equal and supplies accessible progress',()=>{
  const app=readFileSync(new URL('../dist/training-app.js',import.meta.url),'utf8'),html=readFileSync(new URL('../dist/index.html',import.meta.url),'utf8');
  assert.equal(html,readFileSync(new URL('../dist/training.html',import.meta.url),'utf8'));
  assert.match(app,/from '\.\/training-jobs\.js'/);assert.match(app,/trainingJobs\.start\(\{kind:'lm',steps:total/);assert.match(app,/trainingJobs\.start\(\{kind:'transfer',steps:total/);
  assert.doesNotMatch(app,/await api\('(?:train|transfer-train)'/);assert.match(app,/await trainingJobs\.discover\(\{reset:true\}\)/);
  assert.match(app,/lab\.examples\.length&&!jobLocked\(\)/);assert.match(app,/requestAnimationFrame/);
  assert.match(html,/<progress id="job-progress-bar"[^>]*aria-label=/);assert.match(html,/id="transfer-train-1000"/);assert.match(html,/Closing this page does not stop training/);
  assert.match(html,/id="train-5000"/);assert.match(html,/id="job-progress-note"/);assert.match(html,/every 0\.1 s/);assert.match(html,/training-mode-label">Background/);
  assert.doesNotMatch(html,/id="(?:transfer-)?train-(?:10|100|300)"/);assert.doesNotMatch(app,/'(?:transfer-)?train-(?:10|100|300)'/);
  assert.match(html,/id="transfer-train-500"/);assert.doesNotMatch(html,/id="transfer-train-5000"/);assert.doesNotMatch(app,/transfer-train-5000/);
  assert.match(html,/id="preview-update"/);assert.match(html,/id="advance-update"/);
  assert.match(app,/Run 1000 steps and watch the measured updates/);assert.match(app,/Optionally add 500 steps per branch/);
});

async function bootFixture(initialJob,{handler,examples=[{id:'e1',text:'A little story.'}],afterPaint=async()=>{}}={}){
  const html=readFileSync(new URL('../dist/index.html',import.meta.url),'utf8'),source=readFileSync(new URL('../dist/training-app.js',import.meta.url),'utf8').replace(/^import .*;\r?\n/gm,'').replaceAll('export ','');
  const nodes=new Map(),requests=[],timers=new Map();let timerId=0;
  class Element{
    constructor(id=''){this.id=id;this.value='';this.textContent='';this.hidden=false;this.disabled=false;this.checked=false;this.dataset={};this.attrs={viewBox:'0 0 760 310'};this.children=[];this.events={};this.classes=new Set();this.classList={toggle:(key,on)=>on?this.classes.add(key):this.classes.delete(key),add:key=>this.classes.add(key),remove:key=>this.classes.delete(key),contains:key=>this.classes.has(key)};this.style={setProperty(){}};}
    append(...values){for(const value of values){if(value?.parentElement)value.remove();this.children.push(value);if(value&&typeof value==='object')value.parentElement=this;}}replaceChildren(...values){for(const child of this.children)if(child&&typeof child==='object')child.parentElement=null;this.children=[];this.append(...values);}
    get lastChild(){return this.children.at(-1);}remove(){if(this.parentElement){this.parentElement.children=this.parentElement.children.filter(child=>child!==this);this.parentElement=null;}}
    setAttribute(key,value){this.attrs[key]=String(value);}getAttribute(key){return this.attrs[key]??null;}removeAttribute(key){delete this.attrs[key];}
    addEventListener(name,fn){this.events[name]=fn;}querySelectorAll(){return [];}querySelector(){return null;}scrollIntoView(){}focus(){}
  }
  for(const [,id]of html.matchAll(/\bid="([^"]+)"/g))nodes.set(id,new Element(id));
  for(const id of ['training-lr','live-transfer-lr','update-lr'])nodes.get(id).value='.0003';
  const body=new Element('body'),doc={body,addEventListener(){},getElementById:id=>nodes.get(id),querySelectorAll:()=>[],createElement:()=>new Element(),createElementNS:()=>new Element(),createDocumentFragment:()=>new Element()};
  const win={location:{hash:'#demo-convergence',pathname:'/',search:''},history:{state:null,replaceState(){},pushState(){}},localStorage:{getItem:()=>null,setItem(){}},addEventListener(){},matchMedia:()=>({matches:false})};
  const setTimer=(fn,delay)=>{const id=++timerId;timers.set(id,{fn,delay});return id;},clearTimer=id=>timers.delete(id);
  const sessionFetch=async(url,options)=>{const path=url.replace('/api/training/',''),body=options.body?JSON.parse(options.body):undefined;requests.push({path,method:options.method,body});let value=handler?await handler(path,body):undefined;
    if(value!==undefined){}
    else if(path==='job')value={job:initialJob};
    else if(path==='meta')value={architecture:{layers:4,heads:4,width:128,context:128,parameters:1858306},data:{counts:{}},device:'cuda:0',seed:42};
    else if(path==='state')value=initialJob?.liveState||{step:999,history:[],metrics:{}};
    else if(path==='transfer-state')value=initialJob?.transferState||{available:false};
    else if(path==='examples'||path.startsWith('examples?'))value={examples};
    // Short teaching inputs are always available, so idle boot now performs inspection.
    else if(path==='inspect')value={text:body.text,position:body.position,tokens:[],targets:[],target:{id:363,piece:'Lily'},targetProbability:.1,tokenLoss:Math.log(10),probabilities:[]};
    else if(path==='recordings')value={pretraining:{history:[],checkpoints:[]},transfer:{runs:[]}};
    else if(path==='token-history')value={version:1,available:false,active:false,frames:[]};
    else throw new Error('Unexpected request during running-job reload: '+path);
    return{ok:true,status:200,json:async()=>value};};
  const context={document:doc,window:win,sessionFetch,trainingBase:'/api/training/',hosted:true,jobIsActive,trainingMode,
    runTeachingSteps:options=>runTeachingSteps({...options,afterPaint}),
    createTrainingJobs:options=>createTrainingJobs({...options,setTimer,clearTimer,newId:()=>REQUEST}),
    setTimeout:setTimer,clearTimeout:clearTimer,requestAnimationFrame:fn=>setTimer(fn,16),AbortController,console};
  runInNewContext(source+'\nglobalThis.fixture={lab,trainingJobs,boot,trainSteps,trainLiveTransfer,pauseTraining,renderConvergence,renderLiveTransfer,renderJobProgress,resetModel,changeTab,renderInspection,tokenHistoryView,refreshAfterJob};',context);
  for(let i=0;i<20;i++)await new Promise(resolve=>setImmediate(resolve));
  return{...context.fixture,nodes,requests,timers};
}

test('full page boot during a running job stays ready, displays progress and skips GPU inspection',async()=>{
  const f=await bootFixture(job({done:430}));
  assert.equal(f.lab.ready,true,f.nodes.get('lab-status').textContent);assert.equal(f.lab.training,true);
  assert.equal(f.nodes.get('training-job-progress').hidden,false);assert.equal(f.nodes.get('job-progress-bar').value,430);
  assert.match(f.nodes.get('job-count').textContent,/430 \/ 1,000/);assert.equal(f.nodes.get('train-1000').disabled,true);
  assert.equal(f.nodes.get('pause-training').disabled,false);assert.equal(f.nodes.get('story-example').disabled,true);
  assert.equal(f.requests.some(r=>r.path==='inspect'||r.method==='POST'),false);f.trainingJobs.dispose();
});

test('full page boot adopts a transfer task with both per-branch progress and pause controls',async()=>{
  const f=await bootFixture(job({kind:'transfer',done:120}));
  assert.equal(f.lab.ready,true,f.nodes.get('lab-status').textContent);assert.equal(f.lab.transferTraining,true);
  assert.match(f.nodes.get('job-count').textContent,/steps per branch/);assert.equal(f.nodes.get('pause-transfer').disabled,false);
  assert.equal(f.nodes.get('generate-current').disabled,true);assert.equal(f.nodes.get('start-live-transfer').disabled,true);
  assert.equal(f.requests.some(r=>r.method==='POST'),false);f.trainingJobs.dispose();
});

test('small teaching requests use at most 10 steps, show every measured update and await painting',async()=>{
  for(const kind of ['lm','transfer'])for(const steps of [10,100]){
    let current=70,painted=0;const calls=[],points=[];
    const result=await runTeachingSteps({kind,steps,learningRate:.0003,initialState:{step:current,runId:'pair-a'},
      request:async(path,body)=>{assert.equal(painted,calls.length);calls.push({path,body});current+=body.steps;return{step:current,runId:'pair-a'};},
      onChunk:({state,completedSteps})=>points.push([state.step,completedSteps]),afterPaint:async()=>{painted++;}});
    assert.equal(result.completedSteps,steps);assert.equal(result.paused,false);assert.equal(painted,steps/10);
    assert.deepEqual(calls,Array.from({length:steps/10},()=>({path:kind==='lm'?'train':'transfer-train',body:{steps:10,learningRate:.0003}})));
    assert.deepEqual(points,Array.from({length:steps/10},(_,i)=>[80+i*10,10+i*10]));
  }
});

test('teaching pause stops after the in-flight block and never submits remaining requests',async()=>{
  let pause=false,calls=0;const result=await runTeachingSteps({kind:'lm',steps:100,learningRate:.0003,initialState:{step:0},
    request:async()=>{calls++;return{step:10};},onChunk:()=>{pause=true;},shouldPause:()=>pause,afterPaint:async()=>{}});
  assert.equal(calls,1);assert.equal(result.completedSteps,10);assert.equal(result.paused,true);
});

test('lost teaching response and changed transfer pair stop without retrying a POST',async()=>{
  for(const changed of [false,true]){
    let calls=0;await assert.rejects(runTeachingSteps({kind:'transfer',steps:100,learningRate:.0003,initialState:{step:0,runId:'a'},
      request:async()=>{calls++;if(!changed)throw new TypeError('response lost');return{step:10,runId:'b'};},afterPaint:async()=>{}}));
    assert.equal(calls,1);
  }
});

test('10 and 100 route to teaching; 300, 1000 and 5000 route to background',()=>{
  for(const steps of [10,100])assert.equal(trainingMode(steps),'teaching');
  for(const steps of [300,1000,5000])assert.equal(trainingMode(steps),'background');
});

test('5000 steps submits exactly one job and validates its full measured result',async()=>{
  const f=fixture(async(path,body)=>({job:job({steps:5000,done:path==='job-start'?0:5000,status:path==='job-start'?'running':'completed',requestId:body?.requestId||REQUEST})}));
  await f.controller.start({kind:'lm',steps:5000,learningRate:.0003});await f.tick();
  assert.equal(f.controller.state.job.completedSteps,5000);assert.equal(f.requests.filter(r=>r.body).length,1);
  assert.equal(f.requests[0].body.steps,5000);f.controller.dispose();
});

test('poll failures back off to 1, 2, 4 and 8 seconds, then resume 100 ms active polling',async()=>{
  let fail=false;const f=fixture(async()=>{if(fail)throw new TypeError('offline');return{job:job()};});
  await f.controller.discover();fail=true;
  for(const expected of [1000,2000,4000,8000,8000]){await f.tick();assert.equal([...f.timers.values()][0].delay,expected);}
  fail=false;await f.tick();assert.equal([...f.timers.values()][0].delay,100);f.controller.dispose();
});

test('elapsed/status-only snapshots update progress without requesting a plot redraw',async()=>{
  let value=job({done:10});const f=fixture(async()=>({job:value}));await f.controller.discover();
  value={...value,elapsedSeconds:99,updatedAt:'2026-11-04T00:09:00Z'};await f.tick();
  assert.equal(f.events.at(-1).changed,true);assert.equal(f.events.at(-1).snapshotChanged,false);
  value=job({done:20});await f.tick();assert.equal(f.events.at(-1).snapshotChanged,true);f.controller.dispose();
});

test('full page old terminal-job polling preserves teaching busy state, then pause stops at 10',async()=>{
  const old=job({done:1000,status:'completed'}),gate=deferred();let calls=0;
  const f=await bootFixture(old,{examples:[],handler:(path)=>{if(path==='train'){calls++;return gate.promise;}}});
  const teaching=f.trainSteps(100);await new Promise(resolve=>setImmediate(resolve));
  assert.equal(f.lab.teaching.active,true);assert.equal(f.lab.training,true);assert.equal(f.lab.busy,true);
  await f.trainingJobs.poll();assert.equal(f.lab.training,true);assert.equal(f.lab.busy,true);assert.equal(f.lab.live.step,1100);
  assert.equal(f.nodes.get('pause-training').disabled,false);assert.equal(f.nodes.get('train-5000').disabled,true);
  f.pauseTraining();gate.resolve({...old.liveState,step:1110});await teaching;
  assert.equal(calls,1);assert.equal(f.lab.live.step,1110);assert.equal(f.lab.teaching.status,'paused');assert.equal(f.lab.teaching.completedSteps,10);
  assert.equal(f.lab.busy,false);assert.equal(f.lab.training,false);assert.equal(f.requests.some(r=>r.path==='job-start'||r.path==='job-pause'),false);
  await f.trainingJobs.poll();assert.equal(f.lab.live.step,1110);f.trainingJobs.dispose();
});

test('full page teaching failure blocks new training until Retry reads actual state',async()=>{
  const old=job({done:1000,status:'completed'});let actual=old.liveState,calls=0;
  const f=await bootFixture(old,{examples:[],handler:(path)=>{
    if(path==='state')return actual;
    if(path==='train'){calls++;actual={...actual,step:actual.step+10};throw new TypeError('response lost');}
  }});
  await f.trainSteps(100);assert.equal(calls,1);assert.equal(f.lab.ready,false);assert.equal(f.nodes.get('train-1000').disabled,true);
  assert.equal(f.nodes.get('retry-training').hidden,false);assert.match(f.nodes.get('lab-status').textContent,/Use Retry/);
  await f.trainSteps(10);assert.equal(calls,1);await f.boot();assert.equal(f.lab.ready,true);assert.equal(f.lab.live.step,1110);
  assert.equal(f.nodes.get('train-1000').disabled,false);f.trainingJobs.dispose();
});

test('all remaining LM and transfer buttons dispatch one background job without teaching requests',async()=>{
  for(const [kind,steps]of [['lm',1000],['lm',5000],['transfer',500],['transfer',1000]]){
    const f=await bootFixture(null,{examples:[],handler:(path,body)=>path==='job-start'?{job:job({kind:body.kind,steps:body.steps,requestId:body.requestId})}:path==='transfer-state'?{available:true,step:100,branches:[]}:undefined});
    await f.nodes.get(`${kind==='transfer'?'transfer-':''}train-${steps}`).events.click();
    assert.equal(f.requests.filter(r=>r.path==='job-start').length,1);assert.equal(f.requests.some(r=>r.path==='train'||r.path==='transfer-train'),false);
    assert.equal(f.lab.job.kind,kind);assert.equal(f.lab.job.requestedSteps,steps);assert.equal(f.lab.teaching,null);
    assert.equal(kind==='lm'?f.lab.training:f.lab.transferTraining,true);f.trainingJobs.dispose();
  }
});

test('a slow GET schedules a positive delay instead of a zero-delay catch-up storm',async()=>{
  const gate=deferred();let slow=false;const f=fixture(async()=>slow?gate.promise:{job:job()});await f.controller.discover();
  slow=true;const polling=f.controller.poll();f.advance(400);const second=f.controller.poll();assert.equal(f.requests.length,2);
  gate.resolve({job:job({done:10})});await Promise.all([polling,second]);assert.equal([...f.timers.values()][0].delay,25);f.controller.dispose();
});

test('full page 100 teaching steps display each real LM and transfer block',async()=>{
  for(const kind of ['lm','transfer']){
    const old=job({kind,done:1000,status:'completed'});if(kind==='transfer')old.transferState.runId='pair-a';
    let current=1100,f;const observed=[];
    f=await bootFixture(old,{examples:[],afterPaint:async()=>{observed.push([kind==='lm'?f.lab.live.step:f.lab.liveTransfer.step,f.nodes.get('job-progress-bar').value]);},
      handler:(path,body)=>{if(path===(kind==='lm'?'train':'transfer-train')){current+=body.steps;return{...(kind==='lm'?old.liveState:old.transferState),step:current};}}});
    await(kind==='lm'?f.trainSteps(100):f.trainLiveTransfer(100));
    assert.deepEqual(observed,Array.from({length:10},(_,i)=>[1110+i*10,10+i*10]));
    assert.equal(f.lab.teaching.status,'completed');assert.equal(f.lab.busy,false);assert.equal(f.requests.filter(r=>r.method==='POST'&&r.path!=='inspect').length,10);
    assert.equal(f.requests.some(r=>r.path==='job-start'),false);f.trainingJobs.dispose();
  }
});

const executionMetadata={trainingPrecision:'FP32',parameterPrecision:'FP32',optimizer:'Adam',optimizerImplementation:'default',
  lmBackend:'compiled',compileEnabled:true,compileReady:true,compiledShape:[8,64],warmupSeconds:12.3,
  evaluationBackend:'eager',previewBackend:'eager',transferBackend:'eager',optimizerStatePrecision:'FP32',
  evaluationPrecision:'FP32',previewPrecision:'FP32',generationPrecision:'FP32',transferPrecision:'FP32',
  lossScaling:'none',optimizerStepAccounting:'successful-updates-only',maxAttemptsPerUpdate:1};
const elementText=node=>[node.textContent,...node.children.map(elementText)].join(' ');
const metadataFixture=execution=>bootFixture(null,{examples:[],handler:path=>path==='meta'?{
  architecture:{layers:4,heads:4,width:128,context:128,trainContext:64,parameters:1858306},data:{counts:{}},device:'cuda:0',seed:42,execution}:undefined});

test('ready compiled FP32 metadata is visible in live seminar copy and explains exact execution scope',async()=>{
  const f=await metadataFixture(executionMetadata);assert.equal(f.lab.ready,true);
  assert.match(f.nodes.get('engine-detail').textContent,/FP32 · compiled LM training/);
  assert.match(f.nodes.get('convergence-description').textContent,/^FP32 · compiled LM training\. One background task;/);
  const details=elementText(f.nodes.get('model-details-content'));
  assert.match(details,/8 sequences × 64 input tokens/);assert.match(details,/Other batch shapes use eager FP32/);
  assert.match(details,/Model weights and Adam state remain FP32/);assert.match(details,/Adam uses its default implementation/);
  assert.match(details,/Inspection and update previews use eager FP32/);assert.match(details,/Evaluation uses eager FP32/);
  assert.match(details,/Text generation uses FP32/);assert.match(details,/Transfer uses eager FP32/);
  assert.match(details,/12\.3 s/);assert.match(details,/adds no optimizer steps or loss-curve points/);
  assert.equal(f.requests.some(r=>r.method==='POST'&&r.path!=='inspect'),false);f.trainingJobs.dispose();
});

test('recorded convergence never labels old observations compiled; returning live restores runtime label',async()=>{
  for(const trainingPrecision of ['FP32','FP16 AMP']){
    const f=await metadataFixture({...executionMetadata,trainingPrecision});f.nodes.get('mode-recorded').events.click();
    assert.equal(f.nodes.get('convergence-description').textContent,'Actual saved observations. The slider selects stored checkpoints.');
    f.nodes.get('mode-live').events.click();assert.ok(f.nodes.get('convergence-description').textContent.startsWith(`${trainingPrecision} · compiled LM training`));
    assert.equal(f.requests.some(r=>r.method==='POST'&&r.path!=='inspect'),false);f.trainingJobs.dispose();
  }
});

test('missing or unready metadata never invents a compiled execution claim',async()=>{
  for(const execution of [undefined,null,{...executionMetadata,compileReady:false},{...executionMetadata,compileEnabled:false},{...executionMetadata,lmBackend:'eager'}]){
    const f=await metadataFixture(execution),description=f.nodes.get('convergence-description').textContent;
    assert.doesNotMatch(description,/compiled LM training/);assert.doesNotMatch(f.nodes.get('engine-detail').textContent,/compiled LM training/);
    if(!execution){assert.doesNotMatch(description,/FP32/);assert.match(description,/^One background task;/);}
    else if(execution.lmBackend==='eager')assert.match(description,/^FP32 · eager LM training/);
    assert.doesNotMatch(elementText(f.nodes.get('model-details-content')),/Compiler warmup finishes/);f.trainingJobs.dispose();
  }
});

test('FP16 AMP metadata reports actual training precision with FP32 state and auxiliary operations',async()=>{
  for(const compiled of [true,false]){
    const f=await metadataFixture({...executionMetadata,trainingPrecision:'FP16 AMP',lossScaling:'dynamic',maxAttemptsPerUpdate:8,
      lmBackend:compiled?'compiled':'eager',compileEnabled:compiled,compileReady:compiled});
    const description=f.nodes.get('convergence-description').textContent,details=elementText(f.nodes.get('model-details-content'));
    assert.ok(description.startsWith(`FP16 AMP · ${compiled?'compiled':'eager'} LM training`));
    assert.match(f.nodes.get('engine-detail').textContent,/FP16 AMP/);assert.doesNotMatch(description,/FP32 ·/);
    assert.match(details,/Model weights and Adam state remain FP32/);assert.match(details,/Inspection and update previews use eager FP32/);
    assert.match(details,/Evaluation uses eager FP32/);assert.match(details,/Text generation uses FP32/);assert.match(details,/Transfer uses eager FP32/);
    assert.match(details,/Dynamic loss scaling is enabled/);assert.match(details,/Only successful optimizer updates count/);
    assert.match(details,/overflow attempts add no steps or loss-curve points/);assert.doesNotMatch(details,/No FP16/);
    assert.match(details,compiled?/Other batch shapes use eager FP16 AMP/:/Language-model training uses eager FP16 AMP/);
    assert.equal(f.requests.some(r=>r.method==='POST'&&r.path!=='inspect'),false);f.trainingJobs.dispose();
  }
});

test('partial FP16 metadata does not invent optimizer-state, scaling or other-operation precision',async()=>{
  const f=await metadataFixture({trainingPrecision:'FP16 AMP',lmBackend:'eager',parameterPrecision:'FP32'});
  assert.match(f.nodes.get('convergence-description').textContent,/^FP16 AMP · eager LM training/);
  const details=elementText(f.nodes.get('model-details-content'));assert.match(details,/Model weights remain FP32/);
  assert.doesNotMatch(details,/Adam state|loss scaling|Evaluation uses|Text generation uses|Transfer uses|update previews use|successful optimizer/);
  f.trainingJobs.dispose();
});

test('transfer uses its own actual FP16 or FP32 metadata without changing recorded experiment wording',async()=>{
  for(const transferPrecision of ['FP32','FP16 AMP']){
    const f=await metadataFixture({...executionMetadata,trainingPrecision:'FP16 AMP',transferPrecision});
    assert.ok(f.nodes.get('live-transfer-status').textContent.startsWith(`${transferPrecision} · eager transfer training.`));
    assert.ok(elementText(f.nodes.get('model-details-content')).includes(`Transfer uses eager ${transferPrecision}.`));
    assert.doesNotMatch(f.nodes.get('transfer-description').textContent,/FP16|compiled/);
    f.trainingJobs.dispose();
  }
  const f=await metadataFixture({trainingPrecision:'FP16 AMP'});
  assert.doesNotMatch(f.nodes.get('live-transfer-status').textContent,/FP16|FP32/);f.trainingJobs.dispose();
});

test('a transfer pair requiring restart disables every training budget but keeps explicit restart available',async()=>{
  const f=await bootFixture(null,{examples:[],handler:path=>path==='transfer-state'?{available:true,step:100,requiresRestart:true,branches:[]}:undefined});
  assert.equal(f.lab.ready,true);for(const steps of [500,1000])assert.equal(f.nodes.get(`transfer-train-${steps}`).disabled,true);
  assert.match(f.nodes.get('live-transfer-status').textContent,/requires a restart/);assert.match(f.nodes.get('live-transfer-status').textContent,/Reset transfer/);
  assert.equal(f.nodes.get('start-live-transfer').disabled,false);assert.equal(f.nodes.get('start-live-transfer').textContent,'Reset transfer');
  await f.trainLiveTransfer(500);assert.equal(f.requests.some(r=>r.path==='job-start'||r.path==='transfer-train'),false);f.trainingJobs.dispose();
});

const measuredLM=(step=130,lastMeasuredStep=100)=>({step,lastMeasuredStep,evaluationIntervalSteps:100,source:'random',
  metrics:{trainLoss:4.125,valLoss:4.25,evalDescription:'Fixed held-out validation subset.'},
  history:[{step:0,trainLoss:8,valLoss:8.2},{step:lastMeasuredStep,trainLoss:4.125,valLoss:4.25}]});
const measuredTransfer=(step=130,lastMeasuredStep=100)=>({available:true,step,lastMeasuredStep,evaluationIntervalSteps:100,seed:42,
  runId:'pair-evaluation',source:{step:1000,seed:42,source:'pretrained'},auxiliaryWeight:0,
  branches:['pretrained','scratch'].map(name=>({name,metrics:{accuracy:.75,trainLoss:.45,valLoss:.5},
    history:[{step:0,accuracy:.5,trainLoss:.8,valLoss:.8},{step:lastMeasuredStep,accuracy:.75,trainLoss:.45,valLoss:.5}]}))});

test('live LM progress and sampled losses stay separate while only actual new observations redraw the chart',async()=>{
  const live=measuredLM(),f=await bootFixture(job({done:30,liveState:live}),{examples:[]});
  assert.equal(String(f.nodes.get('metric-step').textContent),'130');assert.equal(f.nodes.get('metric-train').textContent,'4.125');
  assert.match(f.nodes.get('metric-measurement').textContent,/Loss measured at step 100\. Current model: step 130/);
  assert.match(f.nodes.get('metric-measurement').textContent,/Evaluation every 100 optimizer steps/);
  assert.equal(f.nodes.get('metric-measurement').classes.has('measurement-lagging'),true);
  assert.match(f.nodes.get('job-progress-note').textContent,/every 0\.1 s/);assert.match(f.nodes.get('job-progress-note').textContent,/every 100 optimizer steps/);
  assert.match(f.nodes.get('job-progress-note').textContent,/block of up to 10 steps/);
  const chart=f.nodes.get('convergence-chart'),oldChildren=chart.children;
  f.lab.live={...live,step:140};f.renderConvergence();
  assert.equal(String(f.nodes.get('metric-step').textContent),'140');assert.equal(chart.children,oldChildren);
  assert.match(f.nodes.get('metric-measurement').textContent,/measured at step 100\. Current model: step 140/);
  assert.doesNotMatch(elementText(chart),/step 130|step 140/);
  f.lab.live={...measuredLM(200,200),history:[...live.history,{step:200,trainLoss:3.9,valLoss:4.1}],metrics:{trainLoss:3.9,valLoss:4.1}};
  f.renderConvergence();assert.notEqual(chart.children,oldChildren);assert.match(elementText(chart),/step 200/);
  assert.equal(f.nodes.get('metric-train').textContent,'3.900');assert.doesNotMatch(f.nodes.get('metric-measurement').textContent,/Current model:/);
  assert.equal(f.nodes.get('metric-measurement').classes.has('measurement-lagging'),false);f.trainingJobs.dispose();
});

test('transfer cards label the measured step while pair progress continues without a new curve point',async()=>{
  const state=measuredTransfer(),f=await bootFixture(job({kind:'transfer',done:30,transferState:state}),{examples:[]});
  assert.match(elementText(f.nodes.get('live-transfer-provenance')),/130/);
  assert.match(f.nodes.get('live-transfer-measurement').textContent,/Metrics measured at fine-tuning step 100\. Current pair: step 130/);
  assert.match(elementText(f.nodes.get('live-transfer-cards')),/Validation accuracy · measured at fine-tuning step 100/);
  assert.doesNotMatch(elementText(f.nodes.get('live-transfer-cards')),/fine-tuning step 130/);
  const accuracy=f.nodes.get('live-accuracy-chart'),loss=f.nodes.get('live-transfer-loss-chart'),oldAccuracy=accuracy.children,oldLoss=loss.children;
  f.lab.liveTransfer={...state,step:170};f.renderLiveTransfer();
  assert.equal(accuracy.children,oldAccuracy);assert.equal(loss.children,oldLoss);
  assert.match(f.nodes.get('live-transfer-measurement').textContent,/Current pair: step 170/);
  assert.doesNotMatch(elementText(accuracy),/step 130|step 170/);f.trainingJobs.dispose();
});

test('long paused and completed nonmultiple tasks retain the last measured metrics for both training kinds',async()=>{
  for(const kind of ['lm','transfer'])for(const status of ['paused','completed']){
    const snapshot=job({kind,status,steps:1000,done:1000,initialStep:90,currentStep:1090,
      liveState:measuredLM(1090,1000),transferState:measuredTransfer(1090,1000)});
    const f=await bootFixture(snapshot,{examples:[]});
    assert.match(f.nodes.get(kind==='lm'?'metric-measurement':'live-transfer-measurement').textContent,/measured at (?:fine-tuning )?step 1000\. Current (?:model|pair): step 1090/);
    assert.doesNotMatch(elementText(f.nodes.get(kind==='lm'?'convergence-chart':'live-transfer-loss-chart')),/step 1090/);
    assert.equal(f.requests.some(request=>request.method==='POST'&&request.path!=='inspect'),false);f.trainingJobs.dispose();
  }
});

test('legacy and explicit-update histories supply the sampled step without inventing the current step or cadence',async()=>{
  const live=measuredLM(137,101);delete live.lastMeasuredStep;delete live.evaluationIntervalSteps;
  const pair=measuredTransfer(137,101);delete pair.lastMeasuredStep;delete pair.evaluationIntervalSteps;
  const f=await bootFixture(null,{examples:[],handler:path=>path==='state'?live:path==='transfer-state'?pair:undefined});
  assert.match(f.nodes.get('metric-measurement').textContent,/measured at step 101\. Current model: step 137/);
  assert.match(f.nodes.get('live-transfer-measurement').textContent,/measured at fine-tuning step 101\. Current pair: step 137/);
  assert.doesNotMatch(f.nodes.get('metric-measurement').textContent,/every 100/);
  assert.match(elementText(f.nodes.get('convergence-chart')),/step 101/);
  f.lab.live={step:137,history:[],metrics:live.metrics};f.renderConvergence();
  assert.match(f.nodes.get('metric-measurement').textContent,/measurement step unavailable/);
  assert.doesNotMatch(f.nodes.get('metric-measurement').textContent,/measured at step 137/);f.trainingJobs.dispose();
});

test('baseline step zero remains a real measurement and recorded mode uses only saved-step copy',async()=>{
  const live=measuredLM(10,0);live.history=live.history.slice(0,1);
  const f=await bootFixture(null,{examples:[],handler:path=>path==='state'?live:undefined});
  assert.match(f.nodes.get('metric-measurement').textContent,/Loss measured at step 0\. Current model: step 10/);
  f.lab.recordings={pretraining:{history:[{step:25,trainLoss:6,valLoss:6.2}],checkpoints:[{step:25,label:'Saved'}]},transfer:{runs:[]}};
  f.nodes.get('mode-recorded').events.click();
  assert.equal(f.nodes.get('metric-measurement').textContent,'Saved loss measured at step 25.');
  assert.doesNotMatch(f.nodes.get('convergence-description').textContent,/every 100/);
  assert.equal(String(f.nodes.get('metric-step').textContent),'25');f.trainingJobs.dispose();
});

test('compiled transfer labels require actual enabled and ready metadata and retain reported precision',async()=>{
  for(const transferPrecision of ['FP32','FP16 AMP']){
    const execution={...executionMetadata,transferPrecision,transferBackend:'compiled-dynamic',transferCompileEnabled:true,
      transferCompileReady:true,transferDynamicLength:true,transferWarmupSeconds:1.25,evaluationIntervalSteps:100};
    const f=await metadataFixture(execution),details=elementText(f.nodes.get('model-details-content'));
    assert.ok(f.nodes.get('live-transfer-status').textContent.startsWith(`${transferPrecision} · compiled transfer training.`));
    assert.ok(details.includes(`Transfer forward passes use compiled ${transferPrecision}.`));
    assert.match(details,/supports variable review lengths/);assert.match(details,/separate weights and Adam states/);
    assert.match(details,/Normal training evaluates metrics every 100 optimizer steps/);
    assert.doesNotMatch(details,new RegExp(`Transfer uses eager ${transferPrecision.replaceAll(' ','\\s')}`));f.trainingJobs.dispose();
  }
  for(const flags of [{transferCompileEnabled:false,transferCompileReady:true},{transferCompileEnabled:true,transferCompileReady:false},{}]){
    const f=await metadataFixture({...executionMetadata,transferPrecision:'FP16 AMP',transferBackend:'compiled-dynamic',...flags});
    assert.doesNotMatch(f.nodes.get('live-transfer-status').textContent,/compiled transfer|eager transfer/);
    assert.doesNotMatch(elementText(f.nodes.get('model-details-content')),/Transfer forward passes use compiled|Transfer compiler warmup/);
    f.trainingJobs.dispose();
  }
});

test('starting or restarting a pair refreshes actual execution metadata with one mutation only',async()=>{
  let available=false,compiled=false;
  const f=await bootFixture(null,{examples:[],handler:(path)=>{
    if(path==='meta')return{architecture:{},data:{},execution:{...executionMetadata,transferBackend:compiled?'compiled-dynamic':'eager',transferCompileEnabled:true,transferCompileReady:compiled}};
    if(path==='transfer-state')return available?measuredTransfer(0,0):{available:false};
    if(path==='transfer-start'){available=true;compiled=true;return measuredTransfer(0,0);}
  }});
  f.nodes.get('live-transfer-seed').value='42';f.nodes.get('live-transfer-auxiliary').value='0';
  for(let index=0;index<2;index++){
    const before=f.requests.length;await f.nodes.get('start-live-transfer').events.click();
    assert.deepEqual(f.requests.slice(before).map(request=>request.path),['transfer-start','meta']);
    assert.match(f.nodes.get('live-transfer-status').textContent,/compiled transfer training/);
    assert.match(f.nodes.get('lab-status').textContent,/Live pair created/);assert.equal(f.lab.ready,true);
  }
  assert.equal(f.requests.filter(request=>request.method==='POST'&&request.path!=='inspect').length,2);f.trainingJobs.dispose();
});

test('metadata read failure preserves successful pair creation, removes stale claims and never replays it',async()=>{
  for(const badMetadata of ['network','invalid']){
    let reads=0,recovered=false;
    const f=await bootFixture(null,{examples:[],handler:path=>{
      if(path==='meta'){
        reads++;if(reads>1&&!recovered){if(badMetadata==='network')throw new TypeError('metadata unavailable');return null;}
        return{architecture:{},data:{},execution:{...executionMetadata,transferBackend:'compiled-dynamic',transferCompileEnabled:true,transferCompileReady:true}};
      }
      if(path==='transfer-start'||path==='transfer-state')return measuredTransfer(0,0);
    }});
    f.nodes.get('live-transfer-seed').value='42';f.nodes.get('live-transfer-auxiliary').value='0';
    await f.nodes.get('start-live-transfer').events.click();
    assert.equal(f.lab.ready,true);assert.equal(f.lab.liveTransfer.available,true);assert.equal(f.lab.meta.execution,null);
    assert.match(f.nodes.get('lab-status').textContent,/^Live pair created/);assert.match(f.nodes.get('lab-status').textContent,/Execution details could not be refreshed/);
    assert.doesNotMatch(f.nodes.get('live-transfer-status').textContent,/compiled transfer|eager transfer/);
    assert.equal(f.requests.filter(request=>request.method==='POST'&&request.path!=='inspect').length,1);
    recovered=true;const before=f.requests.length;await f.nodes.get('refresh-live-transfer').events.click();
    assert.deepEqual(f.requests.slice(before).map(request=>request.path),['transfer-state','meta']);
    assert.match(f.nodes.get('live-transfer-status').textContent,/compiled transfer training/);
    assert.equal(f.requests.filter(request=>request.method==='POST'&&request.path!=='inspect').length,1);f.trainingJobs.dispose();
  }
});

test('reset success is retained when its metadata refresh fails and no reset is replayed',async()=>{
  let reads=0;
  const f=await bootFixture(null,{examples:[],handler:path=>{
    if(path==='meta'){if(++reads>1)throw new TypeError('metadata offline');return{architecture:{},data:{},execution:executionMetadata};}
    if(path==='reset')return measuredLM(0,0);
  }});
  f.nodes.get('reset-seed').value='42';await f.resetModel('random');
  assert.equal(f.lab.ready,true);assert.equal(f.lab.live.step,0);assert.equal(f.lab.meta.execution,null);
  assert.match(f.nodes.get('lab-status').textContent,/^Live model reset from random; actual step 0/);
  assert.match(f.nodes.get('lab-status').textContent,/Execution details could not be refreshed/);
  assert.equal(f.requests.filter(request=>request.method==='POST'&&request.path!=='inspect').length,1);f.trainingJobs.dispose();
});

const timingStats=(mean,p95,count=2)=>({count,mean,p95,min:mean/2,max:p95});
const noTimingStats=()=>({count:0,mean:null,p95:null,min:null,max:null});
function timingSnapshot(kind='lm'){
  const total={cpuMs:timingStats(10,15),gpuMs:timingStats(6,8,1)},phase={cpuMs:timingStats(4,6),gpuMs:timingStats(3,4,1)};
  return{version:1,kind,enabled:true,sampleEvery:20,capacity:200,window:'last-capacity-records',summaryScope:'successful-records-only',gpuSupported:true,pendingGpuSamples:1,
    counters:{observedSteps:40,sampledSteps:3,successfulSamples:3,failedSamples:1,evaluations:1,gpuDroppedSamples:2,eventErrors:1},
    steps:{count:3,successfulCount:2,failedCount:1,total,phases:kind==='lm'?{data:phase,forward_loss:phase,backward:phase,optimizer:phase}:{data:phase,'pretrained.forward_loss':phase,'scratch.forward_loss':phase},
      trend:[{step:20,success:true,cpuMs:5,gpuMs:6},{step:30,success:false,cpuMs:88,gpuMs:88},{step:40,success:true,cpuMs:15,gpuMs:null}]},
    evaluations:{count:1,successfulCount:1,failedCount:0,total,phases:kind==='lm'?{train_evaluation:phase,validation:phase}:{'pretrained.validation':phase,'scratch.validation':phase},trend:[{step:0,success:true,cpuMs:20,gpuMs:null}]}};
}
function openTiming(f,kind='lm'){
  const id=kind==='lm'?'lm-performance':'transfer-performance',panel=f.nodes.get(id);panel.open=true;panel.events.toggle();return f.nodes.get(`${id}-content`);
}
const descendantNodes=node=>[node,...node.children.flatMap(descendantNodes)];

test('compact timing rows retain LM and transfer phase bars from aggregate summaries and actual total trends',async()=>{
  for(const kind of ['lm','transfer']){
    const timing=timingSnapshot(kind);
    for(const group of [timing.steps,timing.evaluations])for(const point of group.trend)assert.deepEqual(Object.keys(point).sort(),['cpuMs','gpuMs','step','success']);
    const f=await bootFixture(null,{examples:[],handler:path=>path===(kind==='lm'?'state':'transfer-state')?{...(kind==='lm'?measuredLM():measuredTransfer()),timing}:undefined});
    const content=openTiming(f,kind),nodes=descendantNodes(content),bars=nodes.filter(node=>node.className==='timing-phase'),charts=nodes.filter(node=>node.attrs.class==='line-chart timing-chart');
    const text=bars.map(elementText).join(' ');assert.match(text,kind==='lm'?/Forward \+ loss/:/Copied LM · Forward \+ loss/);
    assert.match(text,/Mean 4\.000 ms · p95 6\.000 ms/);assert.match(text,/Validation/);
    assert.match(elementText(charts[0]),/CPU wall · step 40 · 15\.00000 ms/);assert.match(elementText(charts[0]),/GPU stream · step 20 · 6\.00000 ms/);
    assert.match(elementText(charts[1]),/CPU wall · step 0 · 20\.00000 ms/);f.trainingJobs.dispose();
  }
});

test('live performance plots report sampled CPU and GPU clocks independently with actual failed and pending counts',async()=>{
  const timing=timingSnapshot(),f=await bootFixture(null,{examples:[],handler:path=>path==='state'?{...measuredLM(),timing}:undefined});
  const content=openTiming(f),text=elementText(content),charts=descendantNodes(content).filter(node=>node.attrs.class==='line-chart timing-chart');
  assert.match(text,/first update after timing reset and every 20th absolute update/);
  assert.match(text,/normal training evaluates every 100 optimizer steps/);assert.match(text,/up to 200 recent records/);
  assert.match(text,/CPU wall time includes waiting/);assert.match(text,/GPU stream time includes dispatch gaps/);assert.match(text,/not GPU busy time/);
  assert.match(text,/do not add phases or CPU \+ GPU totals/);assert.match(text,/n = 2/);assert.match(text,/n = 1/);
  assert.match(text,/1 GPU samples pending; 2 GPU samples skipped; 1 GPU event errors/);
  assert.match(text,/2 successful \/ 3 retained samples · 1 failed/);assert.match(text,/Total update · excludes evaluation/);
  assert.equal(charts.length,2);assert.match(elementText(charts[0]),/CPU wall · step 40 · 15\.00000 ms/);
  assert.doesNotMatch(elementText(charts[0]),/GPU stream · step 40/);assert.doesNotMatch(elementText(charts[0]),/step 30 · 88/);
  assert.match(elementText(charts[1]),/CPU wall · step 0 · 20\.00000 ms/);
  assert.equal(f.requests.some(request=>request.method==='POST'&&request.path!=='inspect'),false);f.trainingJobs.dispose();
});

test('collapsed timing panels defer plotting and repeated progress polls preserve unchanged timing DOM',async()=>{
  const timing=timingSnapshot(),f=await bootFixture(null,{examples:[],handler:path=>path==='state'?{...measuredLM(),timing}:undefined}),content=f.nodes.get('lm-performance-content');
  assert.equal(content.children.length,0);assert.match(f.nodes.get('lm-performance-summary').textContent,/3 updates sampled · 1 evaluations/);
  openTiming(f);const children=content.children;
  f.lab.live={...f.lab.live,step:150,timing:{...JSON.parse(JSON.stringify(timing)),counters:{...timing.counters,observedSteps:150}}};f.renderConvergence();assert.equal(content.children,children);assert.match(elementText(content),/150 update attempts observed/);
  f.nodes.get('lm-performance').open=false;f.lab.live={...f.lab.live,timing:{...timing,pendingGpuSamples:0}};f.renderConvergence();assert.equal(content.children,children);
  openTiming(f);assert.equal(content.children,children);assert.match(elementText(content),/0 GPU samples pending/);
  const before=content.children;f.nodes.get('mode-recorded').events.click();assert.equal(f.nodes.get('lm-performance').hidden,true);assert.equal(content.children,before);
  f.nodes.get('mode-live').events.click();assert.equal(f.nodes.get('lm-performance').hidden,false);assert.equal(content.children,before);f.trainingJobs.dispose();
});

test('transfer performance labels both branches and its own evaluation separately from the LM',async()=>{
  const timing=timingSnapshot('transfer'),f=await bootFixture(null,{examples:[],handler:path=>path==='transfer-state'?{...measuredTransfer(),timing}:undefined});
  const text=elementText(openTiming(f,'transfer'));
  assert.match(text,/Paired fine-tuning update time/);assert.match(text,/Copied LM · Forward \+ loss/);assert.match(text,/Random weights · Forward \+ loss/);
  assert.match(text,/Copied LM · Validation/);assert.match(text,/Random weights · Validation/);
  assert.doesNotMatch(f.nodes.get('lm-performance-summary').textContent,/3 updates sampled/);
  assert.match(f.nodes.get('transfer-performance-summary').textContent,/3 updates sampled/);f.trainingJobs.dispose();
});

test('performance phase bars follow execution order and keep small millisecond measurements visible',async()=>{
  const phase={cpuMs:timingStats(.002,.004),gpuMs:timingStats(.002,.004)},timing=timingSnapshot('transfer');
  timing.steps.phases={'scratch.optimizer':phase,'pretrained.backward':phase,'pretrained.setup':phase,'scratch.forward_loss':phase,
    'pretrained.loss_check':phase,data:phase,'pretrained.optimizer':phase,'pretrained.gradient_check_clip':phase,'pretrained.forward_loss':phase};
  const f=await bootFixture(null,{examples:[],handler:path=>path==='transfer-state'?{...measuredTransfer(),timing}:undefined});
  const text=elementText(openTiming(f,'transfer')),labels=['Data packing + transfer','Copied LM · Update setup','Copied LM · Forward + loss','Copied LM · Loss check','Copied LM · Backward','Copied LM · Gradient check + clipping','Copied LM · Adam + loss scaling','Random weights · Forward + loss','Random weights · Adam + loss scaling'];
  const indices=labels.map(label=>text.indexOf(label));assert.ok(indices.every((index,i)=>index>=0&&(i===0||index>indices[i-1])));
  assert.match(text,/Mean 0\.002 ms · p95 0\.004 ms/);f.trainingJobs.dispose();
});

test('CPU-only and empty timing snapshots never fabricate GPU values or empty statistics',async()=>{
  const timing=timingSnapshot();timing.gpuSupported=false;timing.pendingGpuSamples=0;
  const f=await bootFixture(null,{examples:[],handler:path=>path==='state'?{...measuredLM(),timing}:undefined});
  const content=openTiming(f),text=elementText(content),charts=descendantNodes(content).filter(node=>node.attrs.class==='line-chart timing-chart');
  assert.match(text,/GPU event timing is not supported/);assert.match(elementText(charts[0]),/CPU wall · step 20/);assert.doesNotMatch(elementText(charts[0]),/GPU stream · step/);
  const empty={count:0,successfulCount:0,failedCount:0,total:{cpuMs:noTimingStats(),gpuMs:noTimingStats()},phases:{},trend:[]};
  f.lab.live={...f.lab.live,timing:{...timing,steps:empty,evaluations:empty,counters:{},pendingGpuSamples:0,gpuSupported:true}};f.renderConvergence();
  assert.match(elementText(content),/No successful timing samples are available yet/);assert.match(elementText(content),/No resolved GPU samples/);
  assert.doesNotMatch(elementText(content),/Mean 0\.00 ms/);f.trainingJobs.dispose();
});

test('CPU clock errors leave CPU data missing while resolved GPU spans remain independently visible',async()=>{
  const timing=timingSnapshot();timing.counters.clockErrors=2;timing.steps.total.cpuMs=noTimingStats();timing.steps.phases={};
  timing.steps.trend=[{step:20,success:true,cpuMs:null,gpuMs:6}];
  const f=await bootFixture(null,{examples:[],handler:path=>path==='state'?{...measuredLM(),timing}:undefined}),content=openTiming(f);
  assert.match(elementText(content),/2 CPU clock errors; invalid CPU timings are unavailable/);assert.match(elementText(content),/No valid CPU samples/);
  const chart=descendantNodes(content).find(node=>node.attrs.class==='line-chart timing-chart');
  assert.match(elementText(chart),/GPU stream · step 20 · 6\.00000 ms/);assert.doesNotMatch(elementText(chart),/CPU wall · step/);f.trainingJobs.dispose();
});

test('unsupported, missing and disabled timing metadata clears prior plots without claiming historical timings',async()=>{
  const f=await bootFixture(null,{examples:[],handler:path=>path==='state'?{...measuredLM(),timing:timingSnapshot()}:undefined});
  const content=openTiming(f);assert.match(elementText(content),/Optimizer update time/);
  for(const timing of [undefined,{...timingSnapshot(),version:2},{...timingSnapshot(),kind:'transfer'},{...timingSnapshot(),enabled:false}]){
    f.lab.live={...f.lab.live,timing};f.renderConvergence();const text=elementText(content);
    assert.match(text,/Timing is (?:unavailable|disabled|collection is disabled)|Timing collection is disabled/);
    assert.match(text,/No timing values are inferred/);assert.doesNotMatch(text,/Optimizer update time|Mean \d/);
  }
  f.trainingJobs.dispose();
});


test('reset keeps the new pair when an earlier completion refresh or inspection returns late',async()=>{
  for(const scenario of ['existing-pair','first-initialization','inspection']){
    const oldPair=scenario==='first-initialization'?{available:false}:{...measuredTransfer(500,500),runId:'old-pair'};
    const newPair={...measuredTransfer(0,0),runId:'new-pair'},gate=deferred(),started=deferred();let armed=false,staleInspection;
    const completed=job({done:1000,status:'completed',transferState:oldPair});
    const f=await bootFixture(completed,{examples:[],handler:path=>{
      if(path==='transfer-start')return newPair;
      if(armed&&path==='transfer-state'){
        if(scenario==='inspection')return oldPair;
        started.resolve();return gate.promise;
      }
      if(armed&&scenario==='inspection'&&path==='inspect'){started.resolve();return gate.promise;}
    }});
    const originalInspection=f.lab.inspection;staleInspection={...originalInspection,text:'stale pre-reset inspection'};
    f.nodes.get('live-transfer-seed').value='42';f.nodes.get('live-transfer-auxiliary').value='0';armed=true;
    const refresh=f.refreshAfterJob(completed.id);await started.promise;
    await f.nodes.get('start-live-transfer').events.click();
    assert.equal(f.lab.liveTransfer.runId,'new-pair');assert.equal(f.lab.liveTransfer.step,0);
    gate.resolve(scenario==='inspection'?staleInspection:oldPair);await refresh;
    assert.equal(f.lab.liveTransfer.runId,'new-pair');assert.equal(f.lab.liveTransfer.step,0);
    assert.equal(f.lab.inspection,originalInspection);assert.match(f.nodes.get('live-transfer-step').textContent,/Step 0/);
    assert.equal(f.requests.filter(request=>request.path==='transfer-start').length,1);
    f.trainingJobs.dispose();
  }
});
