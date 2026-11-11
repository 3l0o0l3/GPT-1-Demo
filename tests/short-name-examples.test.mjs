import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';

const app=readFileSync(new URL('../dist/training-app.js',import.meta.url),'utf8');
const examplesCode=app.split('// SHORT_EXAMPLES_START')[1].split('// SHORT_EXAMPLES_END')[0];
const inspectCode=app.slice(app.indexOf('async function inspect('),app.indexOf('async function refreshInspection('));
const previewCode=app.split('\n').find(line=>line.startsWith('async function previewUpdate('));
function fixture({examples=[{id:'small-subset-row-1',text:'A real row in this test training split.'}]}={}){
  const nodes={'inspect-text':{value:'A girl is named Lily.'},'story-example':{value:''},'update-lr':{value:'0.0003'}},requests=[];
  let response={examples},failure=null;
  const context={lab:{live:{step:1000},trainingExample:null},tokenHistoryView:null,$:id=>nodes[id],action:fn=>fn(),api:async(path,body)=>{requests.push({path,body});if(path==='examples'){if(failure)throw failure;return response;}return {};},renderInspection:()=>{},message:()=>{},pauseRecording:()=>{},renderUpdate:()=>{}};
  runInNewContext(examplesCode+'\n'+inspectCode+'\n'+previewCode+'\nglobalThis.subject={examples:SHORT_EXAMPLES,inspect,previewUpdate,loadTrainingExample,trainingExampleFromResponse};',context);
  return {...context.subject,lab:context.lab,nodes,requests,setResponse:value=>{response=value;},failExamples:error=>{failure=error;}};
}

test('six teaching prompts use generic UI keys and initial inspection position hints',async()=>{
  const f=fixture();
  assert.deepEqual(Array.from(f.examples,e=>e.theme),['girl-name','boy-name','animal','food','action','color']);
  assert.deepEqual(Array.from(f.examples,e=>e.text),['A girl is named Lily.','A boy is named Tim.','The animal was a dog.','The cat drank milk.','The bird flew away.','The sky was blue']);
  const positions=[4,4,4,3,2,3];
  for(const [index,example] of f.examples.entries()){await f.inspect(example.text);const {path,body}=f.requests.at(-1);assert.equal(path,'inspect');assert.equal(body.text,example.text);assert.equal(body.position,positions[index]);}
  assert.equal(new Set(f.examples.map(e=>e.id)).size,6);
  for(const example of f.examples){assert.match(example.id,/^teaching-/);assert.equal('sourceExcerpt' in example,false);}
  assert.doesNotMatch(examplesCode,/tinystories-train-/);
});

test('manual text uses a conservative default and explicit token selection is preserved',async()=>{
  const f=fixture();
  await f.inspect();assert.equal(f.requests.at(-1).body.position,4);
  f.nodes['inspect-text'].value='My edited sentence.';await f.inspect();assert.equal(f.requests.at(-1).body.position,0);
  await f.inspect('A girl is named Lily.',0);assert.equal(f.requests.at(-1).body.position,0);
  await f.inspect('My edited sentence.',2);assert.equal(f.requests.at(-1).body.position,2);
});

test('One update uses the current API training row even with a one-row download subset',async()=>{
  const f=fixture();await f.loadTrainingExample();
  assert.equal(f.requests[0].path,'examples');assert.equal(f.requests[0].body,undefined);
  for(const example of f.examples){f.nodes['story-example'].value=example.id;f.nodes['inspect-text'].value='Edited unrelated text';await f.previewUpdate();const {path,body}=f.requests.at(-1);assert.equal(path,'preview');assert.equal(body.exampleId,'small-subset-row-1');assert.equal('text' in body,false);assert.equal(body.learningRate,0.0003);}
  f.setResponse({examples:[{id:'different-download-first-row',text:'A different training row.'},{id:'another-row',text:'Second row.'}]});
  await f.loadTrainingExample();await f.previewUpdate();assert.equal(f.requests.at(-1).body.exampleId,'different-download-first-row');
  assert.match(app,/await loadTrainingExample\(\);lab\.examples=SHORT_EXAMPLES;renderExamples\(\)/);
});

test('empty training examples disable preview instead of using a teaching ID or implicit backend fallback',async()=>{
  const f=fixture({examples:[]});await f.loadTrainingExample();
  assert.equal(f.lab.trainingExample,null);
  await assert.rejects(f.previewUpdate(),/No training example is available/);
  assert.equal(f.requests.filter(request=>request.path==='preview').length,0);
  assert.match(app,/\$\('preview-update'\)\.disabled=locked\|\|!lab\.trainingExample/);
});

test('failed or malformed examples responses clear stale selection and do not silently proceed',async()=>{
  const f=fixture();await f.loadTrainingExample();assert.ok(f.lab.trainingExample);
  const unavailable=new Error('Training examples are unavailable');f.failExamples(unavailable);
  await assert.rejects(f.loadTrainingExample(),error=>error===unavailable);
  assert.equal(f.lab.trainingExample,null);await assert.rejects(f.previewUpdate(),/No training example/);
  for(const response of [null,{}, {examples:null}, {examples:[{}]}, {examples:[{id:'',text:'row'}]}, {examples:[{id:'row',text:''}]}]){
    const malformed=fixture();malformed.setResponse(response);
    await assert.rejects(malformed.loadTrainingExample(),/invalid training examples/);
    assert.equal(malformed.lab.trainingExample,null);
    assert.equal(malformed.requests.filter(request=>request.path==='preview').length,0);
  }
});

test('boot reports an examples API failure and leaves the training UI unavailable',async()=>{
  const bootCode=app.slice(app.indexOf('async function boot(){'),app.indexOf('\nfor(const button of document.querySelectorAll'));
  const nodes={'engine-badge':{textContent:'',className:''}},messages=[],calls=[];
  const context={lab:{ready:false,busy:false,trainingExample:{id:'stale-row',text:'A stale row.'}},
    $:id=>nodes[id],controls:()=>{},message:(text,error)=>messages.push({text,error}),renderMetadata:()=>{},
    trainingJobs:{discover:async()=>{}},api:async path=>{calls.push(path);if(path==='examples')throw new Error('Examples API is unavailable');return {};}};
  runInNewContext(examplesCode+'\n'+bootCode+'\nglobalThis.runBoot=boot;',context);
  await context.runBoot();
  assert.deepEqual(calls,['meta','state','examples']);assert.equal(context.lab.ready,false);assert.equal(context.lab.busy,false);
  assert.equal(context.lab.trainingExample,null);assert.equal(nodes['engine-badge'].textContent,'Training service unavailable');
  assert.deepEqual(messages.at(-1),{text:'Examples API is unavailable',error:true});
});
