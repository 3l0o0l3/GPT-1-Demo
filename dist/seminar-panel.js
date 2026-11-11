import {sessionFetch} from './presenter-session.js';
export function validateClassification(value){
  const object=item=>item!==null&&typeof item==='object'&&!Array.isArray(item),integer=item=>Number.isInteger(item)&&item>=0;
  if(!object(value)||typeof value.text!=='string'||!value.text.trim()||value.text.length>800||!/^[a-f0-9]{64}$/.test(value.snapshotId||'')||!Array.isArray(value.branches)||value.branches.length!==2)throw new Error('Invalid submitted classification.');
  const names=new Set();
  for(const branch of value.branches){
    const source=branch?.source;
    if(!object(branch)||!['pretrained','scratch'].includes(branch.branch)||names.has(branch.branch)||!['negative','positive'].includes(branch.prediction)||!integer(branch.step)||branch.step<1||!object(source)||!integer(source.step)||!integer(source.seed)||typeof source.source!=='string'||source.source.length>100||!/^[a-f0-9]{64}$/.test(source.modelSha256||'')||!/^[a-f0-9]{32}$/.test(branch.runId||'')||!Array.isArray(branch.probabilities)||branch.probabilities.length!==2)throw new Error('Invalid submitted classifier provenance.');
    names.add(branch.branch);const probabilities=new Map();for(const item of branch.probabilities){if(!object(item)||!['negative','positive'].includes(item.label)||probabilities.has(item.label)||!Number.isFinite(item.p)||item.p<0||item.p>1)throw new Error('Invalid submitted probabilities.');probabilities.set(item.label,item.p);}
    if(Math.abs(probabilities.get('negative')+probabilities.get('positive')-1)>1e-4||probabilities.get(branch.prediction)+1e-6<Math.max(...probabilities.values()))throw new Error('Submitted probabilities do not match the prediction.');
  }
  const [first,second]=value.branches;if(first.step!==second.step||first.runId!==second.runId||!['step','seed','source','modelSha256'].every(key=>first.source[key]===second.source[key]))throw new Error('Submitted classifier branches are not from the same saved pair.');
  return value;
}
export function validateShare(value){
  if(!value||!['inactive','active','expired','revoked','unavailable'].includes(value.state)||!Number.isInteger(value.remainingSeconds)||value.remainingSeconds<0)throw new Error('Invalid invitation status.');
  if(value.expiresAt!==null&&(typeof value.expiresAt!=='string'||!Number.isFinite(Date.parse(value.expiresAt))))throw new Error('Invalid invitation expiry.');
  if(value.state==='active'){
    let url;try{url=new URL(value.publicUrl);}catch{throw new Error('Invalid audience invitation URL.');}
    if(url.protocol!=='https:'||!/^[-a-z0-9]+\.trycloudflare\.com$/.test(url.hostname)||url.port||url.username||url.password||url.search||url.hash||!/^\/seminar-gpt\/s\/[A-Za-z0-9_-]{20,128}\/$/.test(url.pathname))throw new Error('Invalid audience invitation URL.');
    if(!value.expiresAt||value.remainingSeconds<1||typeof value.qrDataUrl!=='string'||value.qrDataUrl.length>1000000||!/^data:image\/svg\+xml(?:;charset=utf-8|;utf8)?(?:;base64)?,/.test(value.qrDataUrl))throw new Error('Invalid invitation QR code.');
  }else if(value.publicUrl!==null||value.qrDataUrl!==null)throw new Error('A closed invitation cannot expose an active link.');
  return value;
}
export function remainingLabel(seconds){const total=Math.max(0,Math.ceil(seconds)),hours=Math.floor(total/3600),minutes=Math.floor(total%3600/60),rest=total%60;return `${hours?`${hours}h `:''}${minutes}m ${String(rest).padStart(2,'0')}s`;}
export function validateInbox(value,{requireCount=false}={}){
  if(!value||!Number.isSafeInteger(value.lastCursor)||value.lastCursor<0||!Number.isSafeInteger(value.clearedThrough)||value.clearedThrough<0||value.clearedThrough>value.lastCursor||(requireCount||value.submissionCount!==undefined)&&(!Number.isSafeInteger(value.submissionCount)||value.submissionCount<0))throw new Error('Invalid seminar inbox state.');
  return value;
}
export function validateModelPublication(value){
  if(value===undefined||value===null)return null;
  if(typeof value!=='object'||Array.isArray(value)||typeof value.enabled!=='boolean'||!/^[a-f0-9]{64}$/.test(value.releaseId||'')||(value.publishedAt!==null&&(typeof value.publishedAt!=='string'||!Number.isFinite(Date.parse(value.publishedAt))))||(value.lastRequestId!==null&&(typeof value.lastRequestId!=='string'||!/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i.test(value.lastRequestId))))throw new Error('Invalid model publication status.');
  return value;
}
export function modelVersionLabel(model){
  const step=Number.isSafeInteger(model?.step)&&model.step>=0?model.step.toLocaleString():'unavailable',snapshot=/^[a-f0-9]{64}$/.test(model?.snapshotId||'')?model.snapshotId.slice(0,8):'snapshot unavailable';
  return `step ${step} · ${snapshot}`;
}
const host=typeof document==='undefined'?null:document.getElementById('seminar-panel');
if(host){
  const make=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
  const state={cursor:0,lastCursor:0,clearedThrough:0,entries:new Map(),selected:null,busy:false,clearBusy:false,clearPending:null,ready:false,visible:false,timer:null,model:null,total:0,share:null,shareBusy:false,shareDeadline:0,shareVersion:0,publication:null,syncBusy:false,syncPending:null,modelVersion:0};
  const details=make('details','seminar-disclosure'),summary=make('summary','seminar-summary'),title=make('span',null,'Seminar audience'),count=make('span','seminar-count','Connect audience & discuss submissions');summary.append(title,count);details.append(summary);host.append(details);
  const body=make('div','seminar-body'),intro=make('div','seminar-heading'),heading=make('div'),refresh=make('button','button secondary','Refresh inbox');refresh.type='button';heading.append(make('p','eyebrow','Audience / from experiment to discussion'),make('h2',null,'What did everyone discover?'));intro.append(heading,refresh);body.append(intro);
  const explanation=make('details','explanation'),explanationTitle=make('summary',null,'Audience access and submission details');explanation.append(explanationTitle,make('p','body-copy','Share the audience link. Train in 03, then sync here. The next audience replies use the published LM weights. The frozen sentiment classifiers are published separately.'));
  const sharing=make('section','seminar-sharing'),shareControls=make('div','seminar-share-controls'),durationLabel=make('label',null,'Link valid for'),duration=make('select'),create=make('button','button primary','Create invitation'),revoke=make('button','button secondary','Revoke link');duration.id='seminar-share-duration';durationLabel.htmlFor=duration.id;
  for(const minutes of [15,30,60,120,240]){const option=make('option',null,minutes<60?`${minutes} minutes`:`${minutes/60} ${minutes===60?'hour':'hours'}`);option.value=String(minutes);option.selected=minutes===120;duration.append(option);}create.type=revoke.type='button';create.disabled=revoke.disabled=true;durationLabel.append(duration);shareControls.append(durationLabel,create,revoke);sharing.append(shareControls);
  const linkRow=make('div','seminar-link-row'),linkLabel=make('label','seminar-link-label','Temporary audience link'),linkInput=make('input','seminar-link-input'),copy=make('button','button secondary','Copy link'),open=make('a','button primary','Open audience page ↗');linkInput.readOnly=true;linkInput.id='seminar-public-link';linkInput.placeholder='Create an invitation to share with the room';linkLabel.htmlFor=linkInput.id;copy.type='button';copy.disabled=true;open.target='_blank';open.rel='noopener noreferrer';open.hidden=true;linkRow.append(linkLabel,linkInput,copy,open);sharing.append(linkRow);
  const qrRow=make('div','seminar-qr-row'),qr=make('img','seminar-qr'),qrText=make('div','seminar-qr-text'),expiry=make('p','seminar-share-expiry'),download=make('a','button secondary','Download QR code');qr.alt='Scan to open the temporary audience invitation';qr.width=qr.height=200;download.download='seminar-invitation.svg';qrText.append(make('h3',null,'Scan, explore, submit.'),download);explanation.append(make('p',null,'Visitors open a random temporary address. Your personal domain is not part of this link.'));qrRow.append(qr,qrText);qrRow.hidden=true;sharing.append(expiry,qrRow);
  const shareNotice=make('p','seminar-share-notice');shareNotice.setAttribute('role','status');shareNotice.setAttribute('aria-live','polite');sharing.append(shareNotice);explanation.append(make('p','small-note','Creating a replacement closes the previous invitation. Expiry and Revoke stop audience access; submitted conversations stay in your inbox. Temporary tunnels may be unavailable.'));body.append(sharing,explanation);
  const notice=make('p','seminar-status','Checking the seminar service…');notice.setAttribute('role','status');notice.setAttribute('aria-live','polite');const modelNote=make('p','small-note'),modelRow=make('div','seminar-share-controls'),syncModel=make('button','button secondary','Sync current LM'),syncNotice=make('p','small-note'),modelSource=make('p','small-note');syncModel.type='button';syncModel.id='seminar-sync-model';syncModel.disabled=true;modelNote.id='seminar-audience-model';syncNotice.id='seminar-sync-status';syncNotice.setAttribute('role','status');syncNotice.setAttribute('aria-live','polite');modelRow.append(modelNote,syncModel);body.append(notice,modelRow,syncNotice);explanation.append(modelSource);
  const columns=make('div','seminar-columns'),listSection=make('section','seminar-inbox'),listHeading=make('div','seminar-inbox-heading'),listTitle=make('h3',null,'Submitted conversations'),clear=make('button','button secondary seminar-clear','Clear submissions'),clearNotice=make('p','seminar-clear-status'),list=make('div','seminar-list'),discussion=make('section','seminar-discussion');clear.type='button';clear.id='seminar-clear-submissions';clear.disabled=true;clearNotice.setAttribute('role','status');clearNotice.setAttribute('aria-live','polite');listHeading.append(listTitle,clear);list.setAttribute('aria-label','Submitted conversations');
  const clearConfirmation=make('section','seminar-clear-confirmation'),clearConfirmationTitle=make('h4'),clearConfirmationText=make('p',null,'Only the submitted copies will be permanently deleted. Visitors keep their own chats. Submissions arriving after this panel opens will be kept. This cannot be undone.'),clearConfirmationActions=make('div','seminar-clear-actions'),clearCancel=make('button','button secondary','Cancel'),clearConfirm=make('button','button secondary seminar-clear-confirm','Clear submissions');
  clearConfirmation.id='seminar-clear-confirmation';clearConfirmation.hidden=true;clearConfirmation.setAttribute('role','alertdialog');clearConfirmation.setAttribute('aria-modal','false');clearConfirmationTitle.id='seminar-clear-confirmation-title';clearConfirmationText.id='seminar-clear-confirmation-text';clearConfirmation.setAttribute('aria-labelledby',clearConfirmationTitle.id);clearConfirmation.setAttribute('aria-describedby',clearConfirmationText.id);clearCancel.type=clearConfirm.type='button';clearCancel.id='seminar-clear-cancel';clearConfirm.id='seminar-clear-confirm';clearConfirmationActions.append(clearCancel,clearConfirm);clearConfirmation.append(clearConfirmationTitle,clearConfirmationText,clearConfirmationActions);
  listSection.append(listHeading,clearConfirmation,clearNotice,list);columns.append(listSection,discussion);body.append(columns);details.append(body);
  function empty(){state.selected=null;list.replaceChildren(make('p','small-note','No submissions yet. Only conversations explicitly submitted by visitors appear here.'));discussion.replaceChildren(make('p','seminar-discussion-empty','Choose a submitted conversation to discuss it together.'));}
  empty();
  async function get(path){const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),15000);try{const response=await sessionFetch(`/api/seminar/${path}`,{cache:'no-store',signal:controller.signal});const value=await response.json();if(!response.ok)throw new Error(value.error?.message||`Seminar service unavailable (${response.status}).`);return value;}finally{clearTimeout(timeout);}}
  function syncStatus(text,error=false){syncNotice.textContent=text;syncNotice.classList.remove('error');if(error)syncNotice.classList.add('error');}
  function paintModel(){
    syncModel.disabled=state.syncBusy||state.busy||!state.ready||state.publication?.enabled!==true;
    syncModel.textContent=state.syncBusy?'Syncing current LM…':'Sync current LM';
    modelNote.textContent=`Audience LM · ${modelVersionLabel(state.model)}`;
    if(!state.syncBusy&&!state.publication?.enabled&&!state.syncPending)syncStatus('Model sync is unavailable on this service.');
    else if(!state.syncBusy&&state.publication?.enabled===true&&syncNotice.textContent==='Model sync is unavailable on this service.')syncStatus('');
  }
  function applyModel(model,publication){
    if(!model||typeof model!=='object'||Array.isArray(model))throw new Error('Invalid audience model status.');
    const checked=validateModelPublication(publication);state.model=model;state.publication=checked;
    modelSource.textContent=`Published LM snapshot: ${model.snapshotId||'Unavailable'} · parameters: ${Number.isSafeInteger(model.parameters)?model.parameters.toLocaleString():'Unavailable'}.${model.classifier?.available===true?` Classifier remains a separate frozen pair: fine-tuning step ${model.classifier.step??'—'}, source LM step ${model.classifier.source?.step??'—'}, classifier snapshot ${model.classifier.snapshotId||'Unavailable'}, source LM SHA-256 ${model.classifier.source?.modelSha256||'Unavailable'}, run ${model.classifier.runId||'Unavailable'}.`:model.classifier?.available===false?' No classifier is published.':''}`;
    if(state.syncPending&&checked?.lastRequestId===state.syncPending){state.syncPending=null;syncStatus('Audience model sync confirmed.');}
    paintModel();
  }
  async function syncCurrentModel(){
    if(state.syncBusy||state.busy||!state.ready||state.publication?.enabled!==true)return;
    const requestId=crypto.randomUUID(),expectedReleaseId=state.publication.releaseId;state.syncPending=requestId;state.syncBusy=true;state.modelVersion++;paintModel();inboxControls();syncStatus('Publishing the current LM for the next audience replies…');
    const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),50000);
    try{
      const response=await sessionFetch('/api/seminar/model/sync',{method:'POST',cache:'no-store',headers:{'Content-Type':'application/json'},body:JSON.stringify({requestId,expectedReleaseId}),signal:controller.signal});
      const value=await response.json();if(!response.ok){const error=new Error(value.error?.message||`Model sync failed (${response.status}).`);error.status=response.status;throw error;}
      if(typeof value.alreadyPublished!=='boolean'||!validateModelPublication(value.modelPublication))throw new Error('The model sync response was incomplete.');
      state.modelVersion++;applyModel(value.model,value.modelPublication);state.syncPending=null;syncStatus(value.alreadyPublished?'This model publication was already confirmed.':'Audience LM updated. The next replies use these weights.');
    }catch(error){
      if(error.status===409)state.syncPending=null;
      syncStatus('Checking publication status; the sync request will not be repeated…');state.modelVersion++;
      try{
        const value=await get('status');applyStatus(value,state.shareVersion,state.modelVersion);state.ready=true;state.modelVersion++;
        if(error.status===409)syncStatus('Sync was not applied (conflict). Status refreshed; you may sync again when ready.',true);
        else if(state.publication?.lastRequestId===requestId){state.syncPending=null;syncStatus('Audience model sync confirmed after checking status.');}
        else syncStatus('Sync not confirmed. No request was repeated. Refresh inbox to check before trying again.',true);
      }catch{state.ready=false;syncStatus('Could not confirm model sync. Use Refresh inbox to check its status before trying again.',true);}
    }finally{clearTimeout(timeout);state.syncBusy=false;paintModel();inboxControls();}
  }
  async function mutateShare(action,data){
    if(state.shareBusy)return;state.shareBusy=true;state.shareVersion++;paintShare();shareNotice.classList.remove('error');shareNotice.textContent=action==='share'?'Creating your temporary invitation…':'Closing audience access…';
    const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),20000);
    try{const response=await sessionFetch(`/api/seminar/${action}`,{method:'POST',cache:'no-store',headers:{'Content-Type':'application/json'},body:JSON.stringify(data),signal:controller.signal});const value=await response.json();if(!response.ok)throw new Error(value.error?.message||`Invitation request failed (${response.status}).`);applyShare(value.share);shareNotice.textContent=action==='share'?'Invitation created. Share the link or QR code below.':'Invitation revoked. Previous links and QR codes no longer grant access.';}
    catch(error){shareNotice.classList.add('error');shareNotice.textContent=`${error.name==='AbortError'?'The invitation request timed out.':error.message} Use Refresh inbox to check its current state before trying again.`;}
    finally{clearTimeout(timeout);state.shareBusy=false;paintShare();}
  }
  function paintShare(){
    const share=state.share,seconds=Math.max(0,Math.ceil((state.shareDeadline-Date.now())/1000)),active=share?.state==='active'&&seconds>0;
    create.disabled=state.shareBusy||!share;create.textContent=share?.state==='active'?'Replace invitation':'Create invitation';duration.disabled=state.shareBusy;revoke.disabled=state.shareBusy||share?.state!=='active';copy.disabled=!active||state.shareBusy;open.hidden=!active;qrRow.hidden=!active;
    if(active){if(linkInput.value!==share.publicUrl)linkInput.value=share.publicUrl;open.href=share.publicUrl;if(qr.getAttribute('src')!==share.qrDataUrl)qr.src=share.qrDataUrl;download.href=share.qrDataUrl;expiry.textContent=`Access ends ${new Date(share.expiresAt).toLocaleString('en-GB',{day:'numeric',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',timeZoneName:'short'})} · ${remainingLabel(seconds)} remaining`;}
    else{linkInput.value='';open.removeAttribute('href');qr.removeAttribute('src');download.removeAttribute('href');expiry.textContent=({inactive:'No active invitation. Choose a duration to open audience access.',expired:'This invitation has expired. Create a new one to reopen access.',revoked:'Audience access is closed. Submitted conversations remain available.',unavailable:'Temporary invitation is unavailable. Try creating a new invitation, or refresh to check the connection.',active:'This invitation has reached its expiry time. Refresh to confirm the server status.'})[share?.state]||'Checking invitation status…';}
  }
  function applyShare(value){state.share=validateShare(value);state.shareDeadline=Date.now()+value.remainingSeconds*1000;paintShare();}
  function time(value){const date=new Date(value);return Number.isNaN(date.getTime())?'Unknown time':date.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});}
  function stopLabel(reason){return {eos:'EOS · model ended',length:'Maximum token limit',context:'Context limit'}[reason]||'Completed response';}
  function classificationCards(result){
    validateClassification(result);const comparison=make('div','classification-comparison');
    for(const name of ['pretrained','scratch']){
      const branch=result.branches.find(item=>item.branch===name),card=make('section','classification-branch');card.append(make('h4',null,name==='pretrained'?'Pretrained → sentiment':'Random → sentiment'),make('strong','classification-prediction',branch.prediction==='positive'?'Positive':'Negative'));
      for(const label of ['negative','positive']){const probability=branch.probabilities.find(item=>item.label===label).p,row=make('div','classification-probability'),track=make('progress');track.max=1;track.value=probability;track.setAttribute('aria-label',`${name}: ${label}, ${(100*probability).toFixed(1)} percent`);row.append(make('span',null,label==='positive'?'Positive':'Negative'),track,make('span',null,`${(100*probability).toFixed(1)}%`));card.append(row);}
      card.append(make('p','small-note',name==='pretrained'?`Fine-tuning step ${branch.step} · source LM step ${branch.source.step} · source LM seed ${branch.source.seed}`:`Fine-tuning step ${branch.step} · trained from random weights`));const provenance=make('details','classification-provenance');provenance.append(make('summary',null,'Exact source and frozen snapshot'),make('p',null,`${name==='pretrained'?'Source LM':'Compared against source LM'}: ${branch.source.source} · step ${branch.source.step} · seed ${branch.source.seed} · SHA-256: ${branch.source.modelSha256}`),make('p',null,`Run: ${branch.runId}`),make('p',null,`Classifier snapshot: ${result.snapshotId}`));card.append(provenance);comparison.append(card);
    }
    return comparison;
  }
  function select(entry){
    state.selected=entry.id;for(const button of list.querySelectorAll('button'))button.setAttribute('aria-pressed',String(button.dataset.submissionId===entry.id));
    const header=make('div','seminar-transcript-heading'),name=entry.nickname||'Anonymous';header.append(make('p','eyebrow','Selected for discussion'),make('h3',null,name),make('p','small-note',`${time(entry.submittedAt)} · submission ${entry.id.slice(0,8)}`));discussion.replaceChildren(header);
    if(entry.comment){const note=make('div','seminar-submission-note');note.append(make('strong',null,'What caught their attention'),make('p',null,entry.comment));discussion.append(note);}
    const transcript=make('div','seminar-transcript');
    for(const message of entry.messages){const classification=message.role==='assistant'&&message.kind==='classification',card=make('article',`seminar-message ${message.role==='user'?'visitor':'model'}`);card.append(make('h4',null,message.role==='user'?(message.kind==='classification'?'Visitor review':'Visitor'):classification?'Sentiment classifier comparison':'Trained model'));if(classification)card.append(classificationCards(message.classification));else{card.append(make('p','seminar-message-content',message.content));if(message.role==='assistant')card.append(make('p','small-note',`${stopLabel(message.stopReason)} · ${message.tokenCount??'—'} tokens · ${modelVersionLabel(message.model)}`));}transcript.append(card);}
    discussion.append(transcript);
  }
  function renderList(){
    if(state.selected&&!state.entries.has(state.selected))state.selected=null;
    if(!state.entries.size){empty();return;}list.replaceChildren();
    const entries=[...state.entries.values()].sort((a,b)=>b.cursor-a.cursor);
    for(const entry of entries){const button=make('button','seminar-entry');button.type='button';button.dataset.submissionId=entry.id;button.setAttribute('aria-pressed',String(state.selected===entry.id));const name=make('span','seminar-entry-name',entry.nickname||'Anonymous'),meta=make('span','seminar-entry-meta',`${time(entry.submittedAt)} · ${entry.messages.length/2} experiments`),preview=make('span','seminar-entry-preview',entry.comment||entry.messages[0]?.content||'Submitted conversation');button.append(name,meta,preview);button.addEventListener('click',()=>select(entry));list.append(button);}
    if(!state.selected)select(entries[0]);
  }
  function inboxControls(){refresh.disabled=state.busy||state.clearBusy||state.syncBusy;clear.disabled=state.busy||state.clearBusy||!!state.clearPending||!state.ready||state.total===0;clear.textContent=state.clearBusy?'Clearing…':'Clear submissions';clearConfirm.disabled=state.busy||state.clearBusy||!state.ready||!state.clearPending;}
  function applyInbox(value){
    validateInbox(value);const oldWatermark=state.clearedThrough,oldLast=state.lastCursor;
    state.clearedThrough=Math.max(oldWatermark,value.clearedThrough);state.lastCursor=Math.max(oldLast,value.lastCursor);state.cursor=Math.max(state.cursor,state.clearedThrough);
    if(value.submissionCount!==undefined&&value.clearedThrough>=oldWatermark&&value.lastCursor>=oldLast)state.total=value.submissionCount;
    let removed=false;for(const [id,entry]of state.entries)if(entry.cursor<=state.clearedThrough){state.entries.delete(id);removed=true;}
    if(removed)renderList();
    count.textContent=`${state.total} submitted ${state.total===1?'conversation':'conversations'}`;
  }
  function applyStatus(value,shareVersion,modelVersion=state.modelVersion){
    validateInbox(value,{requireCount:true});if(!value.model)throw new Error('Invalid seminar status response.');
    if(!state.shareBusy&&state.shareVersion===shareVersion)applyShare(value.share);
    applyInbox(value);if(modelVersion===state.modelVersion)applyModel(value.model,value.modelPublication);
  }
  function validEntry(entry){
    if(!(entry&&Number.isSafeInteger(entry.cursor)&&entry.cursor>0&&typeof entry.id==='string'&&typeof entry.submittedAt==='string'&&typeof entry.nickname==='string'&&typeof entry.comment==='string'&&Array.isArray(entry.messages)&&entry.messages.length>0&&entry.messages.length<=20&&entry.messages.every(m=>m&&['user','assistant'].includes(m.role)&&typeof m.content==='string')))return false;
    try{for(const [index,message]of entry.messages.entries())if(message.role==='assistant'&&(message.kind==='classification'||message.classification!==undefined)){if(message.kind!=='classification'||validateClassification(message.classification).text!==entry.messages[index-1]?.content)return false;}return true;}catch{return false;}
  }
  async function refreshInbox(includeEntries=true){
    if(state.busy||state.clearBusy||state.syncBusy)return;state.busy=true;inboxControls();paintModel();
    try{
      const shareVersion=state.shareVersion,modelVersion=state.modelVersion;applyStatus(await get('status'),shareVersion,modelVersion);let more=false,added=0;
      if(includeEntries){
        for(let page=0;page<4;page++){
          const after=state.cursor,value=await get(`submissions?after=${after}&limit=50`);
          validateInbox(value);if(!Array.isArray(value.submissions)||!Number.isSafeInteger(value.nextCursor)||typeof value.hasMore!=='boolean')throw new Error('Invalid seminar inbox response.');
          let previous=after;for(const entry of value.submissions){if(!validEntry(entry)||entry.cursor<=previous||entry.cursor>value.lastCursor)throw new Error('Invalid submission ordering.');previous=entry.cursor;}
          if(value.nextCursor!==previous||(value.hasMore&&!value.submissions.length))throw new Error('Invalid seminar inbox cursor.');
          applyInbox(value);for(const entry of value.submissions)if(entry.cursor>state.clearedThrough){state.entries.set(entry.id,entry);added++;}
          state.cursor=Math.max(state.cursor,value.nextCursor);more=value.hasMore;if(!more)break;
        }
        if(added)renderList();
      }
      state.ready=true;notice.classList.remove('error');notice.textContent=includeEntries?`${state.entries.size} of ${state.total} submitted conversations loaded. ${more?'Loading more on the next refresh.':'Refreshes every 3 seconds while this panel is visible.'}`:'Audience service is ready. Expand this panel to view submissions.';
    }catch(error){state.ready=false;notice.classList.add('error');notice.textContent=error.name==='AbortError'?'The seminar connection timed out. Use Refresh inbox to reconnect.':`${error.message} Local training remains available.`;}
    finally{state.busy=false;inboxControls();paintModel();}
  }
  function openClearConfirmation(){
    if(state.busy||state.clearBusy||state.clearPending||!state.ready||state.total===0)return;
    state.clearPending={throughCursor:state.lastCursor,total:state.total};clearConfirmationTitle.textContent=`Clear ${state.total} submitted ${state.total===1?'conversation':'conversations'}?`;clearConfirm.textContent=`Clear ${state.total} ${state.total===1?'submission':'submissions'}`;clearConfirmation.hidden=false;inboxControls();clearCancel.focus?.({preventScroll:true});
  }
  function cancelClearConfirmation(){
    if(!state.clearPending||state.clearBusy)return;
    state.clearPending=null;clearConfirmation.hidden=true;inboxControls();clear.focus?.({preventScroll:true});
  }
  async function clearSubmissions(){
    if(state.busy||state.clearBusy||!state.ready||!state.clearPending)return;
    const {throughCursor}=state.clearPending;state.clearPending=null;clearConfirmation.hidden=true;
    state.clearBusy=true;inboxControls();clearNotice.classList.remove('error');clearNotice.textContent='Clearing submitted copies…';
    const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),20000);let succeeded=false;
    try{
      const response=await sessionFetch('/api/seminar/submissions/clear',{method:'POST',cache:'no-store',headers:{'Content-Type':'application/json'},body:JSON.stringify({throughCursor}),signal:controller.signal}),value=await response.json();
      if(!response.ok)throw new Error(value.error?.message||`Clear request failed (${response.status}).`);
      validateInbox(value,{requireCount:true});if(!Number.isSafeInteger(value.deletedCount)||value.deletedCount<0||value.clearedThrough<throughCursor)throw new Error('Invalid clear confirmation from the seminar service.');
      applyInbox(value);renderList();clearNotice.textContent=`Cleared ${value.deletedCount} submitted ${value.deletedCount===1?'conversation':'conversations'}. Visitors’ own chats and newer submissions were kept.`;succeeded=true;
    }catch(error){state.ready=false;clearNotice.classList.add('error');clearNotice.textContent=`${error.name==='AbortError'?'The clear request timed out.':error.message} Use Refresh inbox to confirm the result before trying again.`;}
    finally{clearTimeout(timeout);state.clearBusy=false;inboxControls();}
    if(succeeded)await refreshInbox();
  }
  const active=()=>details.open&&state.visible&&!document.hidden;
  function schedule(){clearTimeout(state.timer);state.timer=null;if(active())state.timer=setTimeout(async()=>{if(active())await refreshInbox();schedule();},3000);}
  details.addEventListener('toggle',()=>{if(active())refreshInbox();schedule();});
  const observer=new IntersectionObserver(entries=>{state.visible=entries.some(e=>e.isIntersecting);schedule();},{rootMargin:'100px'});observer.observe(host);
  document.addEventListener('visibilitychange',schedule);
  refresh.addEventListener('click',()=>refreshInbox());syncModel.addEventListener('click',syncCurrentModel);
  clear.addEventListener('click',openClearConfirmation);clearCancel.addEventListener('click',cancelClearConfirmation);clearConfirm.addEventListener('click',clearSubmissions);
  document.addEventListener('keydown',event=>{if(event.key==='Escape'&&state.clearPending&&!state.clearBusy){event.preventDefault();cancelClearConfirmation();}});
  create.addEventListener('click',()=>mutateShare('share',{durationMinutes:Number(duration.value)}));revoke.addEventListener('click',()=>mutateShare('share/revoke',{}));
  copy.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(linkInput.value);copy.textContent='Copied';setTimeout(()=>{copy.textContent='Copy link';},1800);}catch{linkInput.focus();linkInput.select();notice.textContent='The link is selected. Copy it with Ctrl+C or Command+C.';}});
  window.addEventListener('hashchange',()=>{if(location.hash==='#seminar-panel')details.open=true;});
  if(location.hash==='#seminar-panel')details.open=true;
  setInterval(()=>{if(!document.hidden&&details.open)paintShare();},1000);
  refreshInbox(false);
}
