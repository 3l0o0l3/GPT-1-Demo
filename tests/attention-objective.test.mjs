import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const app=readFileSync(new URL('../dist/training-app.js',import.meta.url),'utf8');
const code=app.split('// ATTENTION_OBJECTIVE_VIEW_START')[1].split('// ATTENTION_OBJECTIVE_VIEW_END')[0];
const {attentionRowView,transferObjectiveView}=runInNewContext(code+'\n({attentionRowView,transferObjectiveView});');
const objective={schemaVersion:1,step:100,classificationLoss:.4,lmLoss:2,auxiliaryWeight:.5,auxiliaryContribution:1,totalLoss:1.4,sampleCount:32,split:'train',measurement:'fixed-subset-eval'};

test('attention displays raw current-query weights and masks future tokens without normalization',()=>{
  const tokens=Array.from({length:24},(_,index)=>({piece:String(index)})),row=Array(24).fill(.01),mask=Array(24).fill(false);row[0]=0;
  const view=attentionRowView({tokens,position:20,attention:Array.from({length:24},()=>row),mask:Array.from({length:24},()=>mask)});
  assert.equal(view.length,24);assert.equal(view[0].weight,0);assert.equal(view[20].weight,.01);assert.equal(view[20].blocked,false);
  assert.equal(view[21].blocked,true);assert.equal(view[23].blocked,true);
});
test('missing, malformed and nonfinite attention never becomes a fabricated distribution',()=>{
  const value={tokens:[{piece:'A'}],position:0,attention:[[0]],mask:[[false]]};assert.equal(attentionRowView(value)[0].weight,0);
  for(const input of [null,{...value,attention:[]},{...value,mask:[[null]]},{...value,attention:[[NaN]]},{...value,attention:[[1,0]]}])assert.equal(attentionRowView(input),null);
});
test('objective accepts measured terms, including exact zeros and disabled auxiliary LM',()=>{
  const view=transferObjectiveView(objective,100,.5);assert.equal(view.totalLoss,1.4);assert.equal(view.step,100);
  const zero={...objective,classificationLoss:0,lmLoss:null,auxiliaryWeight:0,auxiliaryContribution:0,totalLoss:0};
  assert.equal(transferObjectiveView(zero,100,0).totalLoss,0);assert.equal(transferObjectiveView(zero,100,0).lmLoss,null);
});
test('objective rejects missing snapshots, incorrect measurement provenance, sums, steps and branch weights',()=>{
  for(const value of [null,{}, {...objective,schemaVersion:2},{...objective,step:150},{...objective,split:'validation'},{...objective,measurement:'minibatch'},{...objective,totalLoss:2},{...objective,auxiliaryContribution:.5},{...objective,classificationLoss:NaN},{...objective,sampleCount:0},{...objective,lmLoss:null},{...objective,auxiliaryWeight:0,lmLoss:2,auxiliaryContribution:0,totalLoss:.4}])assert.equal(transferObjectiveView(value,100,.5),null);
  assert.equal(transferObjectiveView(objective,100,0),null);assert.equal(transferObjectiveView(objective,null,.5),null);
});
test('objective renderer distinguishes missing measurements, auxiliary off and lagged measured steps',()=>{
  const make=(tag,cls,text)=>({tag,cls,text,children:[],style:{},attrs:{},append(...children){this.children.push(...children);},setAttribute(key,value){this.attrs[key]=value;}});
  const renderCode=app.slice(app.indexOf('function renderTransferObjective('),app.indexOf('function renderAttention(){'));
  const render=runInNewContext(code+'\n'+renderCode+'\nrenderTransferObjective;',{make,number:value=>value.toFixed(3)});
  const text=node=>[node.text,...node.children.map(text)].filter(x=>x!==undefined).join(' ');
  const missing=render({metrics:{trainLoss:.4}},100,.5);assert.match(text(missing),/Not measured yet/);assert.equal(missing.children.some(n=>n.cls==='objective-bar'),false);
  const measured=render({metrics:{objective}},100,.5);assert.match(text(measured),/Measured step 100/);assert.doesNotMatch(text(measured),/150/);assert.equal(measured.children.find(n=>n.tag==='details').open,undefined);
  const off=render({metrics:{objective:{...objective,lmLoss:null,auxiliaryWeight:0,auxiliaryContribution:0,totalLoss:.4}}},100,0);
  assert.match(text(off),/Auxiliary off/);assert.match(text(off),/Not computed/);
});
