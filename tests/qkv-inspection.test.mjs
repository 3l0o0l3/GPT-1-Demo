import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const app=readFileSync(new URL('../dist/training-app.js',import.meta.url),'utf8');
const attention=app.split('// ATTENTION_OBJECTIVE_VIEW_START')[1].split('// ATTENTION_OBJECTIVE_VIEW_END')[0];
const qkv=app.split('// QKV_INSPECTION_VIEW_START')[1].split('// QKV_INSPECTION_VIEW_END')[0];
const {qkvInspectionView,qkvKeyPosition}=runInNewContext(attention+'\n'+qkv+'\n({qkvInspectionView,qkvKeyPosition});');
function inspection(){
  const width=32,scale=Math.sqrt(width),query=Array(width).fill(0);query[0]=1;
  const keys=[1,2,3].map(first=>[first,...Array(width-1).fill(0)]),scores=[1/scale,2/scale,null];
  const weights=[Math.exp(scores[0])/(Math.exp(scores[0])+Math.exp(scores[1])),Math.exp(scores[1])/(Math.exp(scores[0])+Math.exp(scores[1])),0];
  const values=[Array.from({length:width},(_,i)=>(i-16)/16),Array.from({length:width},(_,i)=>(12-i)/12),Array(width).fill(2)];
  const context=Array.from({length:width},(_,dimension)=>values.reduce((sum,value,i)=>sum+weights[i]*value[dimension],0));
  return {position:1,tokens:[{piece:'A'},{piece:'girl'},{piece:'future'}],attention:[[1,0,0],weights,[.2,.3,.5]],mask:[[false,true,true],[false,false,true],[false,false,false]],attentionDetails:{schemaVersion:1,layer:1,head:1,headDimension:width,queryPosition:1,scale,query,keys,values,scores,weights:[...weights],context}};
}

test('QKV preserves signed 32-dimensional vectors and actual weights for the same query',()=>{
  const input=inspection(),view=qkvInspectionView(input);assert.ok(view);assert.equal(view.query.length,32);assert.equal(view.values[0][0],-1);assert.equal(view.dots[0],1);assert.equal(view.dots[1],2);assert.equal(view.weights[0],input.attention[1][0]);assert.equal(view.context[0],input.attentionDetails.context[0]);
});
test('missing, nonfinite, mismatched or unmasked-future details are unavailable',()=>{
  for(const change of [v=>delete v.attentionDetails,v=>v.attentionDetails.queryPosition=0,v=>v.attentionDetails.query[0]=NaN,v=>v.attentionDetails.values[0][3]=Infinity,v=>v.attentionDetails.weights[0]=.9,v=>v.attentionDetails.scores[0]=100,v=>v.attentionDetails.scores[2]=0,v=>v.attentionDetails.context[0]=100]){const value=inspection();change(value);assert.equal(qkvInspectionView(value),null);}
});
test('a masked future key cannot be selected and floating-point capture tolerance is respected',()=>{
  const value=inspection();value.attentionDetails.scores[0]+=1e-6;value.attentionDetails.context[0]+=1e-6;const view=qkvInspectionView(value);assert.ok(view);assert.equal(qkvKeyPosition(view,2),1);assert.equal(qkvKeyPosition(view,0),0);assert.equal(qkvKeyPosition(view,-1),1);
});
test('vector strips use signed numbers, percentages are confined to attention, and raw values stay folded',()=>{
  const make=(tag,cls,text)=>({tag,cls,text,children:[],style:{},attrs:{},events:{},append(...children){this.children.push(...children);},replaceChildren(...children){this.children=[...children];},setAttribute(key,value){this.attrs[key]=value;},addEventListener(type,fn){this.events[type]=fn;}});
  const box=make('div'),lab={inspection:inspection()},renderSource=app.slice(app.indexOf('let selectedQkvKey=null;'),app.indexOf('function renderAttentionRow(){'));
  const render=runInNewContext(attention+'\n'+qkv+'\n'+renderSource+'\nrenderQkvInspection;',{make,lab,$:()=>box,number:(value,digits)=>value.toFixed(digits),pct:value=>(100*value).toFixed(1)+'%',piece:String});render();
  const all=node=>[node,...node.children.flatMap(all)],nodes=all(box),strips=nodes.filter(node=>node.cls==='qkv-vector-strip');assert.equal(strips.length,5);
  for(const strip of strips){assert.equal(strip.children.length,32);for(const cell of strip.children){assert.doesNotMatch(cell.title,/%/);assert.match(cell.title,/Dimension \d+: -?\d/);}}
  const raw=nodes.find(node=>node.cls==='qkv-raw');assert.equal(raw.open,undefined);assert.equal(all(raw).filter(node=>node.tag==='code').length,160);
  const scoreRows=all(nodes.find(node=>node.cls==='qkv-scores')).find(node=>node.tag==='tbody').children;assert.equal(scoreRows.length,2);assert.equal(scoreRows[0].children[1].text,lab.inspection.attentionDetails.scores[0].toFixed(5));
  const select=nodes.find(node=>node.tag==='select');assert.equal(select.children[2].disabled,true);select.value='2';select.events.change();assert.equal(all(box).find(node=>node.tag==='select').value,'1');
  const text=all(box).map(node=>node.text).filter(Boolean).join(' ');assert.match(text,/Softmax weight/);assert.match(text,/all visible scores/);assert.match(text,/This head’s output/);assert.match(text,/%/);
  lab.inspection={};render();assert.match(box.children[0].text,/not available/);assert.equal(box.children.length,1);
});
test('QKV starts closed inside the live attention panel and both HTML entry points agree',()=>{
  const html=readFileSync(new URL('../dist/index.html',import.meta.url),'utf8');assert.equal(html,readFileSync(new URL('../dist/training.html',import.meta.url),'utf8'));
  const panel=html.slice(html.indexOf('<section id="attention-row-panel"'),html.indexOf('<div class="selected-target">'));assert.match(panel,/<details id="qkv-details" class="qkv-details">/);assert.doesNotMatch(panel,/<details[^>]*\bopen\b/);
});
