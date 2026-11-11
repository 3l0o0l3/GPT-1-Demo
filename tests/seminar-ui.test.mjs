// Synthetic protocol/UI fixtures; these values are not recorded training results or visitor conversations.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {consumeChatStream,validateConversation,canSubmit,stopLabel,submissionRequest,apiError,isShareClosed,classifierAvailable,validateClassification,validateClassificationResponse,modelVersionLabel as audienceModelVersionLabel} from '../audience/app.js';
import {validateShare,remainingLabel,validateInbox,validateClassification as validateSubmittedClassification,validateModelPublication,modelVersionLabel} from '../dist/seminar-panel.js';

const model={step:14000,parameters:1858306,snapshotId:'d'.repeat(64)};
const user={id:'u1',role:'user',content:'A curious cat',createdAt:'2026-11-02T00:00:00Z'};
const assistant={id:'a1',role:'assistant',content:' 猫',tokenCount:2,stopReason:'eos',model};
const conversation={id:'c1',messages:[user,assistant],submittedMessageCount:0};
const classifierSource={step:14000,seed:42,source:'pretrained',modelSha256:'b'.repeat(64)};
const classification=(text='The food was excellent.')=>({text,snapshotId:'a'.repeat(64),branches:[
  {branch:'pretrained',prediction:'positive',probabilities:[{label:'negative',p:.12},{label:'positive',p:.88}],step:1500,source:{...classifierSource},runId:'c'.repeat(32)},
  {branch:'scratch',prediction:'negative',probabilities:[{label:'negative',p:.61},{label:'positive',p:.39}],step:1500,source:{...classifierSource},runId:'c'.repeat(32)}]});
const classifierModel={...model,classifier:{available:true,snapshotId:'a'.repeat(64),step:1500,source:classifierSource,runId:'c'.repeat(32),branches:['pretrained','scratch']}};
function classificationMessages(text='The food was excellent.',index=1){return[{id:'cu'+index,role:'user',kind:'classification',content:text},{id:'ca'+index,role:'assistant',kind:'classification',content:'Pretrained: positive. Random: negative.',classification:classification(text),model:classifierModel,tokenCount:0,stopReason:'classification'}];}
const frames=[{type:'queued'},{type:'start',userMessage:user,model},{type:'token',index:0,content:' 猫'},{type:'token',index:1,content:' 猫'},{type:'done',conversation,assistantMessage:assistant,model}];
const encoded=events=>new TextEncoder().encode(events.map(event=>JSON.stringify(event)).join('\n')+'\n');
function response(events,{chunks,headers={'Content-Type':'application/x-ndjson'}}={}){const bytes=encoded(events);const segments=chunks?chunks(bytes):[bytes];return new Response(new ReadableStream({start(controller){for(const segment of segments)controller.enqueue(segment);controller.close();}}),{headers});}

test('actual chat stream survives individual UTF-8 bytes; EOS and server transcript agree',async()=>{
  const seen=[];const terminal=await consumeChatStream(response(frames,{chunks:bytes=>[...bytes].map(byte=>new Uint8Array([byte]))}),event=>seen.push(event));
  assert.deepEqual(seen,frames);assert.deepEqual(terminal,frames.at(-1));assert.equal(terminal.assistantMessage.stopReason,'eos');
});
test('done is withheld until EOF and trailing data cannot be displayed as successful completion',async()=>{
  const seen=[];await assert.rejects(consumeChatStream(response([...frames,{type:'token',index:2,content:'extra'}]),event=>seen.push(event)),/after completion/);
  assert.equal(seen.some(event=>event.type==='done'),false);
});
test('missing completion, token order, mismatched transcript and invalid role history fail explicitly',async()=>{
  await assert.rejects(consumeChatStream(response(frames.slice(0,-1)),()=>{}),/before completion/);
  const unordered=structuredClone(frames);unordered[2].index=1;await assert.rejects(consumeChatStream(response(unordered),()=>{}),/out-of-order/);
  const wrong=structuredClone(frames);wrong.at(-1).conversation.messages.at(-1).content='invented';await assert.rejects(consumeChatStream(response(wrong),()=>{}),/does not match/);
  const invalid=structuredClone(conversation);invalid.messages[0].role='assistant';assert.throws(()=>validateConversation(invalid),/invalid message/);
});
test('an explicit backend error or a JSON response is never presented as model text',async()=>{
  const seen=[];await assert.rejects(consumeChatStream(response([{type:'queued'},{type:'error',code:'busy',message:'Please retry later.'}]),event=>seen.push(event)),/Please retry later/);
  assert.deepEqual(seen,[{type:'queued'}]);await assert.rejects(consumeChatStream(response([],{headers:{'Content-Type':'application/json'}}),()=>{}),/did not return a token stream/);
});
test('submission requires a completed, unsubmitted version and explicit consent',()=>{
  assert.equal(canSubmit(conversation,false,false,false),false);assert.equal(canSubmit(conversation,true,false,true),false);assert.equal(canSubmit(conversation,false,true,true),false);
  assert.equal(canSubmit(conversation,false,false,true),true);assert.equal(canSubmit({...conversation,submittedMessageCount:2},false,false,true),false);
  assert.equal(canSubmit({...conversation,messages:[]},false,false,true),false);assert.match(stopLabel('eos'),/model ended/);assert.match(stopLabel('length'),/Maximum/);
});
test('submission consent binds to the exact displayed conversation and message version',()=>{
  const request=submissionRequest(conversation,'Visitor','Interesting ending');
  assert.deepEqual(request,{nickname:'Visitor',comment:'Interesting ending',consent:true,expectedConversationId:'c1',expectedMessageCount:2});
  const other={...conversation,id:'c2'};assert.equal(submissionRequest(other,'','').expectedConversationId,'c2');
  const longer={...conversation,messages:[...conversation.messages,{...user,id:'u2'},{...assistant,id:'a2'}]};assert.equal(submissionRequest(longer,'','').expectedMessageCount,4);
  assert.equal(request.expectedMessageCount,2);
});
test('API errors preserve conflict status and stale-consent code for UI recovery',()=>{
  const stale=apiError(409,{error:{code:'stale_conversation',message:'Conversation changed.'}});
  assert.equal(stale.status,409);assert.equal(stale.code,'stale_conversation');assert.equal(stale.message,'Conversation changed.');
  assert.equal(canSubmit(conversation,false,true,true),false);
  assert.equal(canSubmit(conversation,false,false,false),false);
});
test('closed invitations survive both HTTP and in-stream error boundaries',async()=>{
  const closed={error:{code:'share_closed',message:'This invitation has ended.'}};
  await assert.rejects(consumeChatStream(new Response(JSON.stringify(closed),{status:410,headers:{'Content-Type':'application/json'}}),()=>{}),error=>isShareClosed(error)&&error.status===410);
  await assert.rejects(consumeChatStream(response([{type:'queued'},{type:'error',code:'share_closed',message:'This invitation has ended.'}]),()=>{}),error=>isShareClosed(error)&&error.code==='share_closed');
  assert.equal(isShareClosed(apiError(503,{error:{code:'unavailable',message:'Offline'}})),false);
});
const share={state:'active',publicUrl:'https://example-random.trycloudflare.com/seminar-gpt/s/abcdefghijklmnopqrstuvwxyz123456/',expiresAt:'2026-11-02T12:00:00Z',remainingSeconds:7200,qrDataUrl:'data:image/svg+xml;utf8,%3Csvg%2F%3E',provider:'Cloudflare temporary tunnel'};
test('presenter accepts temporary invitations and closed states without stale links',()=>{
  assert.equal(validateShare(share),share);
  for(const state of ['inactive','expired','revoked','unavailable'])assert.equal(validateShare({...share,state,publicUrl:null,qrDataUrl:null,remainingSeconds:0}).state,state);
  assert.throws(()=>validateShare({...share,state:'expired'}),/closed invitation/);
  assert.throws(()=>validateShare({...share,remainingSeconds:0}),/QR code/);
  assert.throws(()=>validateShare({...share,expiresAt:'tomorrow'}),/expiry/);
  assert.equal(remainingLabel(7200),'2h 0m 00s');assert.equal(remainingLabel(61),'1m 01s');assert.equal(remainingLabel(-1),'0m 00s');
});
test('presenter refuses domain leaks, redirects and non-image QR sources',()=>{
  for(const publicUrl of ['https://personal.example/seminar-gpt/s/abcdefghijklmnopqrstuvwxyz/','https://example-random.trycloudflare.com/redirect?target=personal.example','https://example-random.trycloudflare.com.attacker.example/seminar-gpt/s/abcdefghijklmnopqrstuvwxyz/','javascript:alert(1)'])assert.throws(()=>validateShare({...share,publicUrl}),/invitation URL/);
  for(const qrDataUrl of ['https://qr.example/?url=private','data:text/html,%3Cscript%3E','javascript:alert(1)'])assert.throws(()=>validateShare({...share,qrDataUrl}),/QR code/);
});
function presenterFixture(initialState='inactive',options={}){
  const nodes=[],requests=[],intervals=[],events={};let now=Date.now(),focused=null,uuid=0;
  const db={submissions:structuredClone(options.submissions||[]),lastCursor:Math.max(0,...(options.submissions||[]).map(e=>e.cursor)),clearedThrough:0,model:structuredClone(model),publication:options.publication};
  const inbox=()=>({submissionCount:db.submissions.length,lastCursor:db.lastCursor,clearedThrough:db.clearedThrough});
  class Node{constructor(tag){this.tagName=tag;this.children=[];this.events={};this.attributes={};this.dataset={};this.classList={add(){},remove(){}};this.value='';nodes.push(this);}append(...items){this.children.push(...items);}replaceChildren(...items){this.children=items;}addEventListener(name,callback){this.events[name]=callback;}setAttribute(name,value){this.attributes[name]=value;}getAttribute(name){return this.attributes[name]??this[name]??null;}removeAttribute(name){delete this.attributes[name];delete this[name];}focus(){focused=this;}querySelectorAll(tag){const result=[];for(const child of this.children){if(child.tagName===tag)result.push(child);result.push(...child.querySelectorAll(tag));}return result;}}
  const mount=new Node('section');mount.id='seminar-panel';
  const doc={hidden:false,createElement:tag=>new Node(tag),getElementById:id=>nodes.find(n=>n.id===id),addEventListener:(name,callback)=>{events[name]=callback;}};
  const closed={...share,state:initialState,publicUrl:null,qrDataUrl:null,expiresAt:null,remainingSeconds:0};
  const fetch=async(url,requestOptions={})=>{requests.push({url,options:requestOptions});let value;
    const override=await options.respond?.(url,requestOptions,db);if(override)return{ok:override.ok??true,status:override.status??200,json:async()=>structuredClone(override.value)};
    if(requestOptions.method==='POST'&&url.endsWith('/submissions/clear')){const boundary=JSON.parse(requestOptions.body).throughCursor,before=db.submissions.length;db.submissions=db.submissions.filter(e=>e.cursor>boundary);db.clearedThrough=Math.max(db.clearedThrough,boundary);value={...inbox(),deletedCount:before-db.submissions.length};}
    else if(requestOptions.method==='POST'&&url.endsWith('/model/sync')){const request=JSON.parse(requestOptions.body);db.model={...db.model,step:15000,snapshotId:'e'.repeat(64)};db.publication={...db.publication,releaseId:'f'.repeat(64),publishedAt:'2026-11-11T12:00:00Z',lastRequestId:request.requestId};value={model:db.model,modelPublication:db.publication,alreadyPublished:false};}
    else if(requestOptions.method==='POST')value={share:url.endsWith('/revoke')?{...closed,state:'revoked'}:share};
    else if(url.includes('submissions')){const q=new URL(url,'http://localhost').searchParams,after=Number(q.get('after')),limit=Number(q.get('limit')),rows=db.submissions.filter(e=>e.cursor>after).slice(0,limit);value={...inbox(),submissions:rows,nextCursor:rows.at(-1)?.cursor??after,hasMore:db.submissions.filter(e=>e.cursor>after).length>rows.length};}
    else value={publicUrl:null,share:closed,...inbox(),model:db.model,modelPublication:db.publication};
    return {ok:true,json:async()=>structuredClone(value)};};
  const context={document:doc,sessionFetch:fetch,crypto:{randomUUID:()=>`00000000-0000-4000-8000-${String(++uuid).padStart(12,'0')}`},URL,AbortController,Date:class extends Date{static now(){return now;}},setTimeout:()=>1,clearTimeout(){},setInterval:callback=>intervals.push(callback),IntersectionObserver:class{observe(){}},window:{addEventListener(){}},location:{hash:'#seminar-panel'},navigator:{clipboard:{writeText:async()=>{}}}};
  runInNewContext(readFileSync(new URL('../dist/seminar-panel.js',import.meta.url),'utf8').replace(/^import .*;\r?\n/,'').replaceAll('export function ','function '),context);
  return {nodes,requests,intervals,db,events,focused:()=>focused,advance:ms=>{now+=ms;},find:label=>nodes.find(n=>n.textContent===label),list:()=>nodes.find(n=>n.className==='seminar-list'),discussion:()=>nodes.find(n=>n.className==='seminar-discussion'),clear:()=>nodes.find(n=>n.id==='seminar-clear-submissions'),clearNotice:()=>nodes.find(n=>n.className==='seminar-clear-status'),confirmation:()=>nodes.find(n=>n.id==='seminar-clear-confirmation'),confirm:()=>nodes.find(n=>n.id==='seminar-clear-confirm'),cancel:()=>nodes.find(n=>n.id==='seminar-clear-cancel')};
}
const settle=async()=>{for(let i=0;i<16;i++)await Promise.resolve();};
test('presenter create and revoke update the QR and make exactly one POST per click',async()=>{
  const fixture=presenterFixture();await settle();const create=fixture.find('Create invitation'),revoke=fixture.find('Revoke link'),copy=fixture.find('Copy link'),input=fixture.nodes.find(n=>n.id==='seminar-public-link');
  assert.equal(create.disabled,false);assert.equal(copy.disabled,true);fixture.nodes.find(n=>n.id==='seminar-share-duration').value='120';
  await create.events.click();assert.equal(copy.disabled,false);assert.equal(input.value,share.publicUrl);assert.equal(create.textContent,'Replace invitation');
  const qr=fixture.nodes.find(n=>n.className==='seminar-qr');assert.equal(qr.src,share.qrDataUrl);assert.equal(fixture.nodes.find(n=>n.className==='seminar-qr-row').hidden,false);
  await revoke.events.click();assert.equal(copy.disabled,true);assert.equal(input.value,'');assert.equal(qr.src,undefined);assert.equal(fixture.nodes.find(n=>n.className==='seminar-qr-row').hidden,true);
  const posts=fixture.requests.filter(r=>r.options.method==='POST');assert.equal(posts.length,2);assert.deepEqual(posts.map(r=>[r.url,JSON.parse(r.options.body)]),[['/api/seminar/share',{durationMinutes:120}],['/api/seminar/share/revoke',{}]]);
});
test('presenter hides a QR at its displayed deadline without an automatic mutation',async()=>{
  const fixture=presenterFixture();await settle();fixture.nodes.find(n=>n.id==='seminar-share-duration').value='120';await fixture.find('Create invitation').events.click();
  fixture.nodes.find(n=>n.className==='seminar-disclosure').open=true;
  fixture.advance(7201000);fixture.intervals[0]();assert.equal(fixture.find('Copy link').disabled,true);assert.equal(fixture.nodes.find(n=>n.className==='seminar-qr-row').hidden,true);assert.equal(fixture.requests.filter(r=>r.options.method==='POST').length,1);
});
test('audience deep links show the inbox without a mutation and keep supplementary prose collapsed',async()=>{
  const fixture=presenterFixture();await settle();assert.equal(Boolean(fixture.nodes.find(n=>n.className==='seminar-disclosure').open),true);assert.equal(Boolean(fixture.nodes.find(n=>n.className==='explanation').open),false);assert.equal(fixture.requests.filter(r=>r.options.method==='POST').length,0);
});
test('a changed temporary tunnel can recover by creating a new invitation',async()=>{
  const fixture=presenterFixture('unavailable');await settle();const create=fixture.find('Create invitation');assert.equal(create.disabled,false);assert.equal(fixture.find('Copy link').disabled,true);
  fixture.nodes.find(n=>n.id==='seminar-share-duration').value='60';await create.events.click();assert.equal(fixture.find('Copy link').disabled,false);assert.equal(fixture.nodes.find(n=>n.id==='seminar-public-link').value,share.publicUrl);
});
test('presenter mount preserves identical training entry pages; user text has no HTML sink',()=>{
  const index=readFileSync(new URL('../dist/index.html',import.meta.url),'utf8'),training=readFileSync(new URL('../dist/training.html',import.meta.url),'utf8');
  assert.equal(index,training);assert.equal((index.match(/id="seminar-panel"/g)||[]).length,1);assert.match(index,/href="#seminar-panel"/);
  for(const path of ['../audience/app.js','../dist/seminar-panel.js']){const source=readFileSync(new URL(path,import.meta.url),'utf8');assert.doesNotMatch(source,/innerHTML|outerHTML|insertAdjacentHTML|eval\(/);}
});
const submission=cursor=>({cursor,id:'s'+cursor,submittedAt:'2026-11-04T00:00:00Z',nickname:'Visitor '+cursor,comment:'',messages:[{...user,content:'Prompt '+cursor},{...assistant,content:'Reply '+cursor}],model});
const listIds=fixture=>fixture.list().children.filter(e=>e.tagName==='button').map(e=>e.dataset.submissionId);
test('inbox boundaries require safe integers and a valid watermark',()=>{
  assert.equal(validateInbox({lastCursor:5,clearedThrough:2,submissionCount:3}).lastCursor,5);
  for(const value of [{lastCursor:true,clearedThrough:0},{lastCursor:1,clearedThrough:2},{lastCursor:-1,clearedThrough:0},{lastCursor:Number.MAX_SAFE_INTEGER+1,clearedThrough:0},{lastCursor:1,clearedThrough:0,submissionCount:-1}])assert.throws(()=>validateInbox(value),/inbox state/);
  assert.throws(()=>validateInbox({lastCursor:1,clearedThrough:0},{requireCount:true}),/inbox state/);
});
test('clear is disabled for an empty inbox; cancelled confirmation preserves entries without a POST',async()=>{
  const empty=presenterFixture();await settle();assert.equal(empty.clear().disabled,true);await empty.clear().events.click();assert.equal(empty.confirmation().hidden,true);
  const f=presenterFixture('inactive',{submissions:[submission(1),submission(2)]});await settle();await f.find('Refresh inbox').events.click();assert.equal(f.clear().disabled,false);
  await f.clear().events.click();assert.equal(f.confirmation().hidden,false);assert.equal(f.confirmation().getAttribute('role'),'alertdialog');assert.equal(f.confirmation().getAttribute('aria-modal'),'false');assert.equal(f.focused(),f.cancel());assert.equal(f.find('Clear 2 submitted conversations?').textContent,'Clear 2 submitted conversations?');assert.match(f.confirmation().children[1].textContent,/permanently deleted/);assert.match(f.confirmation().children[1].textContent,/Visitors keep their own chats/);assert.deepEqual(listIds(f),['s2','s1']);assert.equal(f.requests.filter(r=>r.options.method==='POST').length,0);
  await f.cancel().events.click();assert.equal(f.confirmation().hidden,true);assert.equal(f.focused(),f.clear());assert.deepEqual(listIds(f),['s2','s1']);assert.equal(f.requests.filter(r=>r.options.method==='POST').length,0);
  assert.doesNotMatch(readFileSync(new URL('../dist/seminar-panel.js',import.meta.url),'utf8'),/window\.confirm|globalThis\.confirm/);
});
test('Escape cancels the in-page confirmation without a POST or transcript changes',async()=>{
  const f=presenterFixture('inactive',{submissions:[submission(1)]});await settle();await f.find('Refresh inbox').events.click();await f.clear().events.click();let prevented=false;
  f.events.keydown({key:'Escape',preventDefault(){prevented=true;}});assert.equal(prevented,true);assert.equal(f.confirmation().hidden,true);assert.deepEqual(listIds(f),['s1']);assert.equal(f.requests.filter(r=>r.options.method==='POST').length,0);assert.equal(f.focused(),f.clear());
});
test('confirmed clear sends one frozen boundary, removes the selected transcript, and keeps newer submissions',async()=>{
  const f=presenterFixture('inactive',{submissions:[submission(1),submission(2)]});await settle();await f.find('Refresh inbox').events.click();assert.deepEqual(listIds(f),['s2','s1']);
  const oldTranscript=f.discussion().children.at(-1);await f.clear().events.click();assert.equal(f.requests.filter(r=>r.options.method==='POST').length,0);f.db.submissions.push(submission(3));f.db.lastCursor=3;await f.find('Refresh inbox').events.click();assert.equal(f.confirm().textContent,'Clear 2 submissions');await f.confirm().events.click();
  const posts=f.requests.filter(r=>r.options.method==='POST');assert.equal(posts.length,1);assert.equal(posts[0].url,'/api/seminar/submissions/clear');assert.deepEqual(JSON.parse(posts[0].options.body),{throughCursor:2});
  assert.deepEqual(listIds(f),['s3']);assert.equal(f.discussion().children.includes(oldTranscript),false);assert.match(f.clearNotice().textContent,/Cleared 2 submitted conversations/);assert.equal(f.find('1 submitted conversation').textContent,'1 submitted conversation');
});
test('a full clear renders the empty state and double clicks cannot issue two clear requests',async()=>{
  let release;const pending=new Promise(resolve=>{release=resolve;});
  const f=presenterFixture('inactive',{submissions:[submission(1)],respond:async(url,options)=>{if(url.endsWith('/clear'))await pending;}});await settle();await f.find('Refresh inbox').events.click();
  await f.clear().events.click();assert.equal(f.requests.filter(r=>r.options.method==='POST').length,0);const first=f.confirm().events.click();assert.equal(f.clear().disabled,true);assert.equal(f.confirmation().hidden,true);await f.confirm().events.click();assert.equal(f.requests.filter(r=>r.url.endsWith('/clear')).length,1);release();await first;
  assert.equal(f.requests.filter(r=>r.url.endsWith('/clear')).length,1);assert.deepEqual(listIds(f),[]);assert.equal(f.clear().disabled,true);assert.equal(f.discussion().children[0].className,'seminar-discussion-empty');assert.match(f.clearNotice().textContent,/Cleared 1 submitted conversation/);
});
test('failed clear retains cached conversations and disables retry until a successful refresh',async()=>{
  const f=presenterFixture('inactive',{submissions:[submission(1)],respond:url=>url.endsWith('/clear')?{ok:false,status:503,value:{error:{message:'Connection unavailable.'}}}:null});await settle();await f.find('Refresh inbox').events.click();
  const oldTranscript=f.discussion().children.at(-1);await f.clear().events.click();await f.confirm().events.click();assert.deepEqual(listIds(f),['s1']);assert.equal(f.discussion().children.at(-1),oldTranscript);assert.equal(f.clear().disabled,true);assert.match(f.clearNotice().textContent,/Refresh inbox to confirm/);
  await f.find('Refresh inbox').events.click();assert.equal(f.clear().disabled,false);
});
test('polling after another presenter clears purges cached entries and continues with monotonic cursors',async()=>{
  const f=presenterFixture('inactive',{submissions:[submission(1),submission(2)]});await settle();await f.find('Refresh inbox').events.click();const oldTranscript=f.discussion().children.at(-1);
  f.db.submissions=[];f.db.clearedThrough=2;await f.find('Refresh inbox').events.click();assert.deepEqual(listIds(f),[]);assert.equal(f.discussion().children.includes(oldTranscript),false);assert.equal(f.clear().disabled,true);
  f.db.submissions.push(submission(3));f.db.lastCursor=3;await f.find('Refresh inbox').events.click();assert.deepEqual(listIds(f),['s3']);assert.equal(f.clear().disabled,false);assert.equal(f.requests.filter(r=>r.options.method==='POST').length,0);
});
test('a stale page fetched before another clear cannot resurrect removed conversations',async()=>{
  let stale=false;const f=presenterFixture('inactive',{submissions:[submission(1)],respond:url=>stale&&url.includes('/submissions?')?{value:{submissionCount:2,lastCursor:2,clearedThrough:0,submissions:[submission(2)],nextCursor:2,hasMore:false}}:null});await settle();await f.find('Refresh inbox').events.click();
  f.db.submissions=[];f.db.clearedThrough=2;f.db.lastCursor=2;stale=true;await f.find('Refresh inbox').events.click();
  assert.deepEqual(listIds(f),[]);assert.equal(f.clear().disabled,true);assert.equal(f.find('0 submitted conversations').textContent,'0 submitted conversations');assert.equal(f.discussion().children[0].className,'seminar-discussion-empty');
});
test('a clear between status and pagination applies the page watermark and exact remaining count',async()=>{
  let clearOnPage=false;const f=presenterFixture('inactive',{submissions:[submission(1),submission(2)],respond:(url,options,db)=>{if(clearOnPage&&url.includes('/submissions?')){db.submissions=[submission(3)];db.lastCursor=3;db.clearedThrough=2;clearOnPage=false;}}});await settle();await f.find('Refresh inbox').events.click();clearOnPage=true;await f.find('Refresh inbox').events.click();
  assert.deepEqual(listIds(f),['s3']);assert.equal(f.find('1 submitted conversation').textContent,'1 submitted conversation');assert.equal(f.requests.filter(r=>r.options.method==='POST').length,0);
});

test('audience and presenter validate complete classifier pairs and reject invalid probabilities or provenance',()=>{
  for(const validate of [validateClassification,validateSubmittedClassification]){
    assert.equal(validate(classification()).branches.length,2);
    for(const mutate of [v=>v.branches.pop(),v=>v.branches[1].branch='pretrained',v=>v.snapshotId='bad',v=>v.branches[0].source.modelSha256='bad',v=>v.branches[0].runId='bad',v=>v.branches[0].step=0,v=>v.branches[0].probabilities[0].p=NaN,v=>v.branches[0].probabilities[1].p=.1,v=>v.branches[0].prediction='negative']){const value=classification();mutate(value);assert.throws(()=>validate(value));}
  }
  assert.equal(classifierAvailable(classifierModel),true);assert.equal(classifierAvailable({...model,classifier:{available:false}}),false);assert.equal(classifierAvailable(model),false);assert.equal(classifierAvailable({...classifierModel,classifier:{available:true}}),false);
});

test('classification response binds the result and probabilities to the stored reviewed text',()=>{
  const messages=classificationMessages(),value={conversation:{id:'c2',messages,submittedMessageCount:0},assistantMessage:structuredClone(messages[1]),model:classifierModel};
  value.assistantMessage.classification.branches.reverse();value.assistantMessage.classification.branches[0].probabilities.reverse();
  assert.equal(validateClassificationResponse(value,messages[0].content),value);
  assert.throws(()=>validateClassificationResponse(value,'A different review'),/does not match/);
  const changed=structuredClone(value);changed.assistantMessage.classification.branches[1].probabilities=[{label:'negative',p:.2},{label:'positive',p:.8}];assert.throws(()=>validateClassificationResponse(changed,messages[0].content),/does not match/);
  const invalid=structuredClone(value.conversation);invalid.messages[0].content='Different text';assert.throws(()=>validateConversation(invalid),/does not match/);
});

const treeText=node=>[node.textContent||'',...(node.children||[]).map(treeText)].join(' ');
function audienceFixture({available=true,respond,initialConversation,shareExpiresAt}={}){
  const nodes=[],byId=new Map(),requests=[],timers=new Map(),events={};let timerId=0,uuid=0;
  class Element{
    constructor(tag='div'){this.tagName=tag;this.children=[];this.events={};this.attributes={};this.dataset={};this.value='';this.textContent='';this.disabled=false;this.hidden=false;this.checked=false;this.scrollHeight=0;this.clientHeight=500;this.scrollTop=0;this.classes=new Set();this.classList={add:key=>this.classes.add(key),remove:key=>this.classes.delete(key),toggle:(key,on)=>on?this.classes.add(key):this.classes.delete(key)};nodes.push(this);}
    append(...items){this.children.push(...items);}replaceChildren(...items){this.children=items;}addEventListener(name,callback){this.events[name]=callback;}setAttribute(name,value){this.attributes[name]=String(value);}getAttribute(name){return this.attributes[name]??null;}focus(){}requestSubmit(){return this.events.submit({preventDefault(){}});}
  }
  const html=readFileSync(new URL('../audience/index.html',import.meta.url),'utf8');for(const[,id]of html.matchAll(/\bid="([^"]+)"/g)){const node=new Element();node.id=id;byId.set(id,node);}
  const suggestions=[];for(const[,prompt]of html.matchAll(/data-prompt="([^"]+)"/g)){const node=new Element('button');node.dataset.prompt=prompt;suggestions.push(node);}
  const doc={hidden:false,getElementById:id=>byId.get(id),createElement:tag=>new Element(tag),querySelectorAll:selector=>selector==='[data-prompt]'?suggestions:[],addEventListener:(name,fn)=>{events[name]=fn;}};
  const db={model:structuredClone(available?classifierModel:{...model,classifier:{available:false}}),conversation:structuredClone(initialConversation||{id:'audience-c1',messages:[],submittedMessageCount:0}),submitted:null};
  const session=()=>({shareExpiresAt,csrf:'csrf-token',model:db.model,conversation:db.conversation,limits:{messageChars:800,maxTurns:10,windowSize:128,maxNewTokens:160}});
  const fetch=async(url,options={})=>{
    const body=options.body?JSON.parse(options.body):undefined;requests.push({url,options,body});const override=await respond?.(url,body,db);if(override)return new Response(JSON.stringify(override.value),{status:override.status||200,headers:{'Content-Type':'application/json'}});
    let value;
    if(url==='api/session')value=session();
    else if(url==='api/info')value={model:db.model,shareExpiresAt};
    else if(url==='api/classify'){const messages=classificationMessages(body.message,db.conversation.messages.length/2+1);db.conversation.messages.push(...messages);value={conversation:db.conversation,assistantMessage:messages[1],model:db.model};}
    else if(url==='api/chat'){
      const index=db.conversation.messages.length/2+1,nextUser={...user,id:'su'+index,content:body.message},nextAssistant={...assistant,id:'sa'+index,content:'The end.',model:db.model};db.conversation.messages.push(nextUser,nextAssistant);
      return response([{type:'queued'},{type:'start',userMessage:nextUser,model:db.model},{type:'token',index:0,content:'The'},{type:'token',index:1,content:'The end.'},{type:'done',conversation:db.conversation,assistantMessage:nextAssistant,model:db.model}]);
    }else if(url==='api/submit'){db.submitted=structuredClone(db.conversation);db.conversation.submittedMessageCount=db.conversation.messages.length;value={submission:{id:'receipt-classifier',messageCount:db.conversation.messages.length}};}
    else if(url==='api/new'){db.conversation={id:'audience-new',messages:[],submittedMessageCount:0};value=session();}
    else throw new Error('Unexpected audience request '+url);
    return new Response(JSON.stringify(value),{headers:{'Content-Type':'application/json'}});
  };
  const context={document:doc,fetch,AbortController,TextDecoder,crypto:{randomUUID:()=>`00000000-0000-4000-8000-${String(++uuid).padStart(12,'0')}`},setTimeout:(fn,ms)=>{const id=++timerId;timers.set(id,{fn,ms});return id;},clearTimeout:id=>timers.delete(id),setInterval(){},console};
  runInNewContext(readFileSync(new URL('../audience/app.js',import.meta.url),'utf8').replaceAll('export ',''),context);
  return{nodes,byId,requests,timers,db,events,get:id=>byId.get(id),input:text=>{byId.get('chat-prompt').value=text;byId.get('chat-prompt').events.input();},send:()=>byId.get('chat-form').events.submit({preventDefault(){}}),submit:()=>byId.get('submit-form').events.submit({preventDefault(){}})};
}
const settleAudience=async()=>{for(let i=0;i<12;i++)await new Promise(resolve=>setImmediate(resolve));};

test('audience classification submits one authenticated request, renders both branches and requires new consent',async()=>{
  const f=audienceFixture();await settleAudience();assert.equal(f.get('mode-classifier').disabled,false);f.get('mode-classifier').events.click();f.input('<b>The food was excellent.</b>');f.get('submit-consent').checked=true;
  await f.send();assert.equal(f.get('submit-consent').checked,false);assert.equal(f.get('submit-chat').disabled,true);
  const posts=f.requests.filter(r=>r.url==='api/classify');assert.equal(posts.length,1);assert.equal(posts[0].options.headers['X-CSRF-Token'],'csrf-token');assert.equal(posts[0].body.message,'<b>The food was excellent.</b>');assert.match(posts[0].body.requestId,/^[a-f0-9-]{36}$/);
  const transcript=treeText(f.get('chat-messages'));assert.match(transcript,/<b>The food was excellent\.<\/b>/);assert.match(transcript,/Pretrained → sentiment/);assert.match(transcript,/Random → sentiment/);assert.match(transcript,/88\.0%/);assert.match(transcript,/61\.0%/);assert.match(transcript,/Fine-tuning step 1500 · source LM step 14000/);assert.match(transcript,/Classifier snapshot: a{64}/);
  assert.equal(f.nodes.filter(n=>n.tagName==='progress').length,4);f.get('submit-consent').checked=true;f.get('submit-consent').events.change();assert.equal(f.get('submit-chat').disabled,false);await f.submit();
  const cards=f.nodes.filter(n=>n.className==='classification-branch');assert.match(treeText(cards[0]),/source LM seed 42/);assert.match(treeText(cards[1]),/trained from random weights/);assert.match(treeText(cards[1]),/Compared against source LM/);assert.doesNotMatch(treeText(cards[1]),/copied LM|initialization seed|source LM seed/);
  const submitted=f.requests.find(r=>r.url==='api/submit');assert.deepEqual(submitted.body,{nickname:'',comment:'',consent:true,expectedConversationId:'audience-c1',expectedMessageCount:2});assert.equal(f.db.submitted.messages[1].kind,'classification');assert.equal(f.get('submit-chat').disabled,true);
});

test('classifier-unavailable metadata disables that mode while story continuation remains usable',async()=>{
  const f=audienceFixture({available:false});await settleAudience();assert.equal(f.get('mode-classifier').disabled,true);assert.match(f.get('classifier-status').textContent,/unavailable/);f.get('mode-classifier').events.click();f.input('Once upon a time');await f.send();
  assert.equal(f.requests.some(r=>r.url==='api/classify'),false);assert.equal(f.requests.filter(r=>r.url==='api/chat').length,1);assert.match(treeText(f.get('chat-messages')),/The end\./);
});

test('lost classification responses require session recovery and never replay the prediction',async()=>{
  const f=audienceFixture({respond:(url,body,db)=>{if(url==='api/classify'){db.conversation.messages.push(...classificationMessages(body.message));throw new TypeError('Connection lost');}}});await settleAudience();f.get('mode-classifier').events.click();f.input('The service was friendly.');await f.send();
  assert.equal(f.get('recover-chat').hidden,false);assert.equal(f.get('chat-prompt').disabled,true);assert.equal(f.get('submit-chat').disabled,true);assert.match(f.get('chat-status').textContent,/will not be repeated/);
  await f.get('recover-chat').events.click();assert.equal(f.requests.filter(r=>r.url==='api/classify').length,1);assert.equal(f.get('recover-chat').hidden,true);assert.equal(f.get('submit-consent').checked,false);assert.match(treeText(f.get('chat-messages')),/Pretrained → sentiment/);
});

test('an explicit unavailable response returns to usable story mode without inventing a result',async()=>{
  const f=audienceFixture({respond:(url,body,db)=>{if(url==='api/classify'){db.model.classifier={available:false};return{status:503,value:{error:{code:'classifier_unavailable',message:'No published classifier.'}}};}}});await settleAudience();f.get('mode-classifier').events.click();f.input('An interesting film.');await f.send();
  assert.equal(f.get('mode-classifier').disabled,true);assert.doesNotMatch(treeText(f.get('chat-messages')),/88\.0%/);assert.equal(f.get('mode-story').disabled,false);f.get('mode-story').events.click();await f.send();assert.equal(f.requests.filter(r=>r.url==='api/chat').length,1);
});

test('classification expiry disables further access and a pending prediction cannot be double-submitted',async()=>{
  let release;const gate=new Promise(resolve=>{release=resolve;});
  const f=audienceFixture({respond:async url=>{if(url==='api/classify'){await gate;return{status:410,value:{error:{code:'share_closed',message:'Invitation closed.'}}};}}});await settleAudience();f.get('mode-classifier').events.click();f.input('A good movie.');const pending=f.send();await f.send();assert.equal(f.requests.filter(r=>r.url==='api/classify').length,1);assert.equal(f.get('submit-chat').disabled,true);
  release();await pending;assert.equal(f.get('mode-story').disabled,true);assert.equal(f.get('mode-classifier').disabled,true);assert.equal(f.get('submit-chat').disabled,true);assert.match(f.get('share-expiry').textContent,/closed/);
});

test('mixed classification and story turns share the explicit version-checked submission without client history injection',async()=>{
  const f=audienceFixture();await settleAudience();f.get('mode-classifier').events.click();f.input('Great food.');await f.send();f.get('mode-story').events.click();f.input('Once upon a time');await f.send();
  assert.equal(f.db.conversation.messages.length,4);assert.equal(f.db.conversation.messages[1].kind,'classification');const chat=f.requests.find(r=>r.url==='api/chat');assert.deepEqual(Object.keys(chat.body).sort(),['message','requestId']);
  f.get('submit-consent').checked=true;f.get('submit-consent').events.change();await f.submit();assert.equal(f.requests.find(r=>r.url==='api/submit').body.expectedMessageCount,4);assert.equal(f.db.submitted.messages.length,4);
});

test('presenter discussion renders submitted classifier probabilities and exact provenance without token-generation labels',async()=>{
  const entry={...submission(1),messages:classificationMessages(),model:classifierModel},f=presenterFixture('inactive',{submissions:[entry]});await settle();await f.find('Refresh inbox').events.click();
  const text=treeText(f.discussion());assert.match(text,/Sentiment classifier comparison/);assert.match(text,/Pretrained → sentiment/);assert.match(text,/Random → sentiment/);assert.match(text,/88\.0%/);assert.match(text,/61\.0%/);assert.match(text,/Fine-tuning step 1500 · source LM step 14000/);assert.match(text,/Classifier snapshot: a{64}/);assert.doesNotMatch(text,/0 tokens/);assert.equal(f.nodes.filter(n=>n.tagName==='progress').length,4);
  const cards=f.nodes.filter(n=>n.className==='classification-branch');assert.match(treeText(cards[0]),/source LM seed 42/);assert.match(treeText(cards[1]),/trained from random weights/);assert.match(treeText(cards[1]),/Compared against source LM/);assert.doesNotMatch(treeText(cards[1]),/copied LM|initialization seed|source LM seed/);
});


const publication={enabled:true,releaseId:'a'.repeat(64),publishedAt:null,lastRequestId:null};
test('publication metadata is explicit; legacy or disabled services cannot sync',async()=>{
  assert.equal(validateModelPublication(undefined),null);assert.equal(validateModelPublication(publication),publication);
  for(const bad of [{...publication,enabled:1},{...publication,releaseId:'bad'},{...publication,lastRequestId:'bad'},{...publication,publishedAt:'yesterday'}])assert.throws(()=>validateModelPublication(bad));
  for(const metadata of [undefined,{...publication,enabled:false}]){const f=presenterFixture('inactive',{publication:metadata});await settle();const button=f.nodes.find(n=>n.id==='seminar-sync-model');assert.equal(button.disabled,true);await button.events.click();assert.equal(f.requests.filter(r=>r.options.method==='POST').length,0);}
});

test('model sync sends one CAS request despite a double click and publishes only actual returned identity',async()=>{
  let release;const gate=new Promise(resolve=>{release=resolve;});const f=presenterFixture('inactive',{publication,respond:async(url)=>{if(url.endsWith('/model/sync'))await gate;}});await settle();
  const button=f.nodes.find(n=>n.id==='seminar-sync-model');const first=button.events.click();await button.events.click();assert.equal(button.disabled,true);release();await first;
  const posts=f.requests.filter(r=>r.options.method==='POST');assert.equal(posts.length,1);assert.equal(posts[0].url,'/api/seminar/model/sync');assert.deepEqual(JSON.parse(posts[0].options.body),{requestId:'00000000-0000-4000-8000-000000000001',expectedReleaseId:publication.releaseId});
  assert.match(f.nodes.find(n=>n.id==='seminar-audience-model').textContent,/15,000.*eeeeeeee/);assert.equal(button.disabled,false);assert.match(f.nodes.find(n=>n.id==='seminar-sync-status').textContent,/updated/);
});

test('a lost sync reply is confirmed by its request ID using GET status without another POST',async()=>{
  const f=presenterFixture('inactive',{publication,respond:(url,options,db)=>{if(url.endsWith('/model/sync')){const request=JSON.parse(options.body);db.model={...model,step:16000,snapshotId:'b'.repeat(64)};db.publication={...publication,releaseId:'c'.repeat(64),lastRequestId:request.requestId};throw new TypeError('Connection lost');}}});await settle();await f.nodes.find(n=>n.id==='seminar-sync-model').events.click();
  assert.equal(f.requests.filter(r=>r.options.method==='POST').length,1);assert.equal(f.requests.at(-1).url,'/api/seminar/status');assert.match(f.nodes.find(n=>n.id==='seminar-sync-status').textContent,/confirmed after checking/);assert.match(f.nodes.find(n=>n.id==='seminar-audience-model').textContent,/16,000.*bbbbbbbb/);
});

test('sync conflicts refresh the release CAS without retry; failed reconciliation stays disabled',async()=>{
  const conflict=presenterFixture('inactive',{publication,respond:(url,options,db)=>{if(url.endsWith('/model/sync')){db.publication={...publication,releaseId:'b'.repeat(64)};return{ok:false,status:409,value:{error:{message:'Release changed.'}}};}}});await settle();await conflict.nodes.find(n=>n.id==='seminar-sync-model').events.click();assert.equal(conflict.requests.filter(r=>r.options.method==='POST').length,1);assert.match(conflict.nodes.find(n=>n.id==='seminar-sync-status').textContent,/conflict/);assert.equal(conflict.db.publication.releaseId,'b'.repeat(64));
  let failed=false;const unavailable=presenterFixture('inactive',{publication,respond:url=>{if(url.endsWith('/model/sync')){failed=true;throw new TypeError('timeout');}if(failed&&url.endsWith('/status'))throw new TypeError('offline');}});await settle();await unavailable.nodes.find(n=>n.id==='seminar-sync-model').events.click();assert.equal(unavailable.requests.filter(r=>r.options.method==='POST').length,1);assert.equal(unavailable.nodes.find(n=>n.id==='seminar-sync-model').disabled,true);assert.match(unavailable.nodes.find(n=>n.id==='seminar-sync-status').textContent,/Could not confirm/);
});

test('audience session, info and reply badges update while old messages keep their own model version',async()=>{
  const f=audienceFixture({shareExpiresAt:'2099-01-01T00:00:00Z'});await settleAudience();assert.match(f.get('model-badge').textContent,/14,000.*dddddddd/);f.input('A cat');await f.send();
  f.db.model={...f.db.model,step:15000,snapshotId:'e'.repeat(64)};f.events.visibilitychange();await settleAudience();assert.match(f.get('model-badge').textContent,/15,000.*eeeeeeee/);
  f.db.model={...f.db.model,step:16000,snapshotId:'f'.repeat(64)};f.input('A dog');await f.send();assert.match(f.get('model-badge').textContent,/16,000.*ffffffff/);
  const transcript=treeText(f.get('chat-messages'));assert.match(transcript,/step 14,000 · dddddddd/);assert.match(transcript,/step 16,000 · ffffffff/);assert.doesNotMatch(transcript,/step 15,000/);
  for(const label of [modelVersionLabel,audienceModelVersionLabel])assert.equal(label(undefined),'step unavailable · snapshot unavailable');
});


test('a late audience info read cannot overwrite the version confirmed by a newer reply',async()=>{
  let release,started;const gate=new Promise(resolve=>{release=resolve;}),waiting=new Promise(resolve=>{started=resolve;});
  const f=audienceFixture({respond:async(url,body,db)=>{if(url==='api/info'){const stale=structuredClone(db.model);started();await gate;return{value:{model:stale}};}}});await settleAudience();
  f.events.visibilitychange();await waiting;f.db.model={...f.db.model,step:17000,snapshotId:'e'.repeat(64)};f.input('A new story');await f.send();release();await settleAudience();
  assert.match(f.get('model-badge').textContent,/17,000.*eeeeeeee/);assert.match(treeText(f.get('chat-messages')),/17,000.*eeeeeeee/);
});


test('enabled publication clears only the obsolete unavailable notice after startup or status recovery',async()=>{
  for(const metadata of [publication,{...publication,enabled:false}]){
    const f=presenterFixture('inactive',{publication:metadata});await settle();const button=f.nodes.find(n=>n.id==='seminar-sync-model'),notice=f.nodes.find(n=>n.id==='seminar-sync-status');
    if(!metadata.enabled){assert.match(notice.textContent,/unavailable/);f.db.publication=publication;await f.find('Refresh inbox').events.click();}
    assert.equal(button.disabled,false);assert.doesNotMatch(notice.textContent,/unavailable/);
    await button.events.click();const completed=notice.textContent;assert.match(completed,/updated/);await f.find('Refresh inbox').events.click();assert.equal(notice.textContent,completed);
  }
});
