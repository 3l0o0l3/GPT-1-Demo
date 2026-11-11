import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';

const file=readFileSync(new URL('../dist/training-app.js',import.meta.url),'utf8');
const moduleSource=file.split('// SEMINAR_GUIDE_START')[1].split('// SEMINAR_GUIDE_END')[0].replaceAll('export ','');
function fixture({hash='',preference=null}={}){
  const nodes=new Map(),events={},storage=new Map(),calls=[],history=[];
  if(preference)storage.set('gpt-seminar-mode',preference);
  class Node{
    constructor(id){this.id=id;this.dataset={};this.attributes={};this.events={};this.hidden=false;this.open=false;this.classes=new Set();this.classList={toggle:(name,enabled)=>enabled?this.classes.add(name):this.classes.delete(name)};nodes.set(id,this);}
    addEventListener(type,fn){this.events[type]=fn;}
    setAttribute(key,value){this.attributes[key]=value;}
    removeAttribute(key){delete this.attributes[key];}
    scrollIntoView(){calls.push(['scroll',this.id]);}
    querySelectorAll(){return buttons;}
    querySelector(){return details;}
  }
  const ids=['presentation-guide','seminar-route','seminar-cue','seminar-workspace','seminar-mode-toggle','seminar-stage-number','seminar-stage-name','seminar-stage-description','seminar-stage-paper','seminar-stage-time','seminar-stage-action','seminar-stage-observe','seminar-panel'];
  ids.forEach(id=>new Node(id));
  const buttons=['data','update','convergence','generation','transfer','discussion'].map(id=>{const button=new Node('route-'+id);button.dataset.demoStop=id;return button;});
  const details={open:false},body=new Node('body');
  const doc={body,getElementById:id=>nodes.get(id)};
  const entries=[{url:'/'+hash,state:null}];let entry=0;
  const win={location:{hash,pathname:'/',search:''},localStorage:{getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value)},history:{state:null},addEventListener:(type,fn)=>events[type]=fn,matchMedia:()=>({matches:false}),MutationObserver:class{constructor(fn){this.fn=fn;}observe(){}},fetch:()=>{throw new Error('Guide may not issue network requests');}};
  const loadEntry=()=>{const current=entries[entry];win.history.state=current.state;win.location.hash=current.url.includes('#')?current.url.slice(current.url.indexOf('#')):'';};
  win.history.pushState=(state,title,url)=>{history.push(['push',url]);entries.splice(entry+1);entries.push({url,state});entry++;loadEntry();};
  win.history.replaceState=(state,title,url)=>{history.push(['replace',url]);entries[entry]={url,state};loadEntry();};
  const back=()=>{if(entry>0){entry--;loadEntry();events.popstate();events.hashchange();}};
  const forward=()=>{if(entry<entries.length-1){entry++;loadEntry();events.popstate();events.hashchange();}};
  const context={console};runInNewContext(moduleSource+'\nglobalThis.module={SEMINAR_STOPS,seminarRoute,createSeminarGuide};',context);
  let controller;const navigateTab=tab=>{calls.push(['tab',tab]);controller?.followTab(tab);};
  controller=context.module.createSeminarGuide({document:doc,window:win,navigateTab});
  return{...context.module,controller,doc,win,nodes,buttons,details,calls,history,events,storage,back,forward,click:id=>nodes.get(id).events.click()};
}

test('six stops total exactly ten minutes and resolve known, legacy, and malformed routes',()=>{
  const f=fixture();assert.equal(f.SEMINAR_STOPS.length,6);assert.equal(f.SEMINAR_STOPS.reduce((a,s)=>a+s.seconds,0),600);
  for(const stage of ['data','update','convergence','generation','transfer','discussion'])assert.equal(f.seminarRoute('#demo-'+stage),stage);
  assert.equal(f.seminarRoute('#seminar-panel'),'discussion');assert.equal(f.seminarRoute('#demo-invalid'),'data');assert.equal(f.seminarRoute('#unrelated'),null);
});
test('presentation is the default and a generation deep link selects the existing convergence panel only',()=>{
  const f=fixture({hash:'#demo-generation'});assert.equal(f.doc.body.classes.has('seminar-mode'),true);assert.equal(f.doc.body.dataset.demoStage,'generation');assert.deepEqual(f.calls,[['tab','convergence']]);assert.deepEqual(f.history,[['replace','/#demo-generation']]);assert.match(f.nodes.get('seminar-stage-time').textContent,/5:30–7:00/);
});
test('six route tabs navigate the existing tab owner with no model operations',()=>{
  const f=fixture();for(const id of ['update','convergence','generation','transfer','discussion'])f.click('route-'+id);
  assert.deepEqual(f.calls.filter(x=>x[0]==='tab'),[['tab','data'],['tab','update'],['tab','convergence'],['tab','convergence'],['tab','transfer'],['tab','data']]);
  assert.equal(f.doc.body.dataset.demoStage,'discussion');assert.equal(f.nodes.get('route-discussion').attributes['aria-current'],'step');assert.equal(f.details.open,true);
  f.click('route-transfer');assert.equal(f.doc.body.dataset.demoStage,'transfer');assert.equal(f.win.location.hash,'#demo-transfer');
});
test('legacy audience link shows the discussion without creating an invitation',()=>{
  const f=fixture({hash:'#seminar-panel'});assert.equal(f.details.open,true);assert.equal(f.nodes.get('seminar-workspace').open,false);assert.equal(f.doc.body.dataset.demoStage,'discussion');assert.deepEqual(f.calls,[['tab','data']]);assert.deepEqual(f.history,[['replace','/#seminar-panel']]);
});
test('a fresh page ignores old workspace preference; an explicit mode toggle still works',()=>{
  const f=fixture({preference:'workspace'});assert.equal(f.doc.body.classes.has('seminar-mode'),true);assert.equal(f.nodes.get('seminar-route').hidden,false);assert.equal(f.nodes.get('seminar-workspace').open,false);
  f.nodes.get('seminar-workspace').open=true;
  f.click('seminar-mode-toggle');assert.equal(f.doc.body.classes.has('seminar-mode'),false);assert.equal(f.storage.get('gpt-seminar-mode'),'workspace');assert.equal(f.nodes.get('seminar-workspace').open,true);assert.equal(f.win.location.hash,'');
  f.click('seminar-mode-toggle');assert.equal(f.doc.body.classes.has('seminar-mode'),true);assert.equal(f.storage.get('gpt-seminar-mode'),'presentation');assert.equal(f.nodes.get('seminar-workspace').open,true);
  f.win.location.hash='#demo-update';f.events.hashchange();assert.equal(f.doc.body.classes.has('seminar-mode'),true);assert.equal(f.doc.body.dataset.demoStage,'update');
});
test('step changes retain the visible experiment and its model state',()=>{
  const f=fixture(),workspace=f.nodes.get('seminar-workspace'),retained={prompt:'My story',step:6000};workspace.experimentState=retained;workspace.open='not-a-disclosure';
  for(const id of ['data','generation','transfer','discussion','update']){f.click('route-'+id);assert.equal(workspace.open,'not-a-disclosure');assert.equal(workspace.hidden,false);assert.equal(workspace.experimentState,retained);}
  assert.equal(f.nodes.get('seminar-stage-description').textContent,'One prediction error produces a weight update.');
  f.back();assert.equal(workspace.open,'not-a-disclosure');assert.equal(workspace.experimentState,retained);
});
test('history and existing tab navigation keep the guide in sync without recursion',()=>{
  const f=fixture();f.click('route-generation');assert.equal(f.doc.body.dataset.demoStage,'generation');
  f.win.location.hash='#demo-transfer';f.events.popstate();assert.equal(f.doc.body.dataset.demoStage,'transfer');
  f.controller.followTab('update');assert.equal(f.doc.body.dataset.demoStage,'update');assert.equal(f.win.location.hash,'#demo-update');
  assert.equal(f.nodes.get('route-update').attributes['aria-current'],'step');assert.equal(f.nodes.get('route-transfer').attributes['aria-current'],undefined);
});
test('Back to the initial empty hash restores Data, and Forward restores the selected stop',()=>{
  const f=fixture();f.click('route-update');assert.equal(f.doc.body.dataset.demoStage,'update');
  f.back();assert.equal(f.win.location.hash,'');assert.equal(f.doc.body.dataset.demoStage,'data');assert.equal(f.doc.body.classes.has('seminar-mode'),true);
  f.forward();assert.equal(f.win.location.hash,'#demo-update');assert.equal(f.doc.body.dataset.demoStage,'update');
});
test('workspace history entries retain their mode and current panel across Back and Forward',()=>{
  const f=fixture();f.click('route-transfer');f.click('seminar-mode-toggle');
  assert.equal(f.win.location.hash,'');assert.equal(f.doc.body.classes.has('seminar-mode'),false);assert.equal(f.doc.body.dataset.demoStage,'transfer');
  f.back();assert.equal(f.doc.body.classes.has('seminar-mode'),true);assert.equal(f.doc.body.dataset.demoStage,'transfer');
  f.forward();assert.equal(f.win.location.hash,'');assert.equal(f.doc.body.classes.has('seminar-mode'),false);assert.equal(f.doc.body.dataset.demoStage,'transfer');
  f.controller.followTab('update');f.back();f.forward();assert.equal(f.doc.body.classes.has('seminar-mode'),false);assert.equal(f.doc.body.dataset.demoStage,'update');
});
test('both allowlisted HTML entry points are identical and contain all navigation targets',()=>{
  const a=readFileSync(new URL('../dist/index.html',import.meta.url),'utf8'),b=readFileSync(new URL('../dist/training.html',import.meta.url),'utf8');assert.equal(a,b);
  for(const id of ['presentation-guide','seminar-route','seminar-workspace','seminar-mode-toggle','seminar-stage-name','seminar-stage-description','seminar-stage-action','seminar-stage-observe'])assert.match(a,new RegExp(`id="${id}"`));
  assert.doesNotMatch(a,/id="seminar-(?:previous|next)"/);
  assert.equal((a.match(/data-demo-stop=/g)||[]).length,6);assert.doesNotMatch(a,/<script[^>]*>\s*[^<\s]/);
  assert.doesNotMatch(a,/<details\b[^>]*\bopen(?:[\s=>])/);
  assert.match(a,/<div id="seminar-workspace"/);assert.doesNotMatch(a,/<details id="seminar-workspace"/);assert.doesNotMatch(a,/Explore this step/);
  assert.match(a,/<details class="explanation"><summary>Generation details<\/summary>/);
  const css=readFileSync(new URL('../dist/training-style.css',import.meta.url),'utf8');assert.doesNotMatch(css,/seminar-workspace:not\(\[open\]\)/);
});
