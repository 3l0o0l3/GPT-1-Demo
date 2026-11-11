import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';

const app=readFileSync(new URL('../dist/training-app.js',import.meta.url),'utf8');
const source=app.split('// PRESENTER_PROGRESS_VIEW_START')[1].split('// PRESENTER_PROGRESS_VIEW_END')[0].replaceAll('export ','');
const view=runInNewContext(source+'\npresenterProgressView;');

test('completed and paused tasks do not occupy the live presenter area',()=>{
  for(const kind of ['lm','transfer'])for(const status of ['completed','paused']){
    const result=view({job:{kind,status}});assert.equal(result.kind,kind);assert.equal(result.active,false);
  }
});
test('running, stopping and failed tasks retain their actual training owner',()=>{
  for(const kind of ['lm','transfer'])for(const status of ['running','pause_requested','failed']){
    const result=view({job:{kind,status}});assert.equal(result.kind,kind);assert.equal(result.active,true);
  }
});
test('pending or uncertain transfer start cannot be attributed to the previous LM task',()=>{
  for(const state of [{pending:true},{uncertain:true}]){
    const result=view({job:{kind:'lm',status:'completed'},pendingKind:'transfer',...state});
    assert.equal(result.kind,'transfer');assert.equal(result.active,true);
  }
});
test('connection failures remain actionable while an empty healthy state stays quiet',()=>{
  assert.equal(view().active,false);
  const result=view({job:{kind:'lm',status:'completed'},error:'Disconnected'});
  assert.equal(result.active,true);assert.equal(result.state,'error');assert.equal(result.kind,'lm');
});
test('a confirmed task owns its display after a previous start request',()=>{
  const result=view({job:{kind:'lm',status:'running'},pendingKind:'transfer'});
  assert.equal(result.kind,'lm');assert.equal(result.active,true);
});

test('pending pause of a discovered active task keeps its actual owner',()=>{
  for(const status of ['running','pause_requested']){
    const result=view({job:{kind:'lm',status},pending:true,pendingKind:'transfer'});
    assert.equal(result.kind,'lm');assert.equal(result.active,true);
  }
});
