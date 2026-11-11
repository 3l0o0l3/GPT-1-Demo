export function modelVersionLabel(model){
  const step=Number.isSafeInteger(model?.step)&&model.step>=0?model.step.toLocaleString():'unavailable',snapshot=/^[a-f0-9]{64}$/.test(model?.snapshotId||'')?model.snapshotId.slice(0,8):'snapshot unavailable';
  return `step ${step} · ${snapshot}`;
}
const object=value=>value!==null&&typeof value==='object'&&!Array.isArray(value);
const integer=value=>Number.isInteger(value)&&value>=0;
const sourceValid=value=>object(value)&&integer(value.step)&&integer(value.seed)&&typeof value.source==='string'&&value.source.length<=100&&/^[a-f0-9]{64}$/.test(value.modelSha256||'');
const sameSource=(a,b)=>['step','seed','source','modelSha256'].every(key=>a[key]===b[key]);
export function classifierAvailable(model){const value=model?.classifier;return Boolean(value?.available===true&&integer(value.step)&&value.step>0&&sourceValid(value.source)&&/^[a-f0-9]{32}$/.test(value.runId||'')&&/^[a-f0-9]{64}$/.test(value.snapshotId||'')&&Array.isArray(value.branches)&&value.branches.length===2&&new Set(value.branches).size===2&&value.branches.every(branch=>['pretrained','scratch'].includes(branch)));}
export function validateClassification(value){
  if(!object(value)||typeof value.text!=='string'||!value.text.trim()||value.text.length>800||!/^[a-f0-9]{64}$/.test(value.snapshotId||'')||!Array.isArray(value.branches)||value.branches.length!==2)throw new Error('The server returned an invalid classification.');
  const names=new Set();
  for(const branch of value.branches){
    if(!object(branch)||!['pretrained','scratch'].includes(branch.branch)||names.has(branch.branch)||!['negative','positive'].includes(branch.prediction)||!integer(branch.step)||branch.step<1||!sourceValid(branch.source)||!/^[a-f0-9]{32}$/.test(branch.runId||'')||!Array.isArray(branch.probabilities)||branch.probabilities.length!==2)throw new Error('The classifier branch or provenance is invalid.');
    names.add(branch.branch);const probabilities=new Map();
    for(const item of branch.probabilities){if(!object(item)||!['negative','positive'].includes(item.label)||probabilities.has(item.label)||!Number.isFinite(item.p)||item.p<0||item.p>1)throw new Error('The classifier probabilities are invalid.');probabilities.set(item.label,item.p);}
    if(Math.abs(probabilities.get('negative')+probabilities.get('positive')-1)>1e-4||probabilities.get(branch.prediction)+1e-6<Math.max(...probabilities.values()))throw new Error('The classifier probabilities do not match its prediction.');
  }
  const [first,second]=value.branches;if(first.step!==second.step||first.runId!==second.runId||!sameSource(first.source,second.source))throw new Error('The classifier branches are not from the same saved pair.');
  return value;
}
const classificationIdentity=value=>JSON.stringify([value.text,value.snapshotId,...[...value.branches].sort((a,b)=>a.branch.localeCompare(b.branch)).map(branch=>[branch.branch,branch.prediction,branch.step,branch.runId,branch.source.step,branch.source.seed,branch.source.source,branch.source.modelSha256,...[...branch.probabilities].sort((a,b)=>a.label.localeCompare(b.label)).map(item=>[item.label,item.p])])]);
export function validateClassificationResponse(value,text){
  if(!object(value)||!object(value.model)||!object(value.assistantMessage)||value.assistantMessage.kind!=='classification')throw new Error('The classification response is incomplete.');
  const conversation=validateConversation(value.conversation),last=conversation.messages.at(-1),result=validateClassification(value.assistantMessage.classification);
  if(!last||last.kind!=='classification'||last.id!==value.assistantMessage.id||last.content!==value.assistantMessage.content||result.text!==text||classificationIdentity(last.classification)!==classificationIdentity(result))throw new Error('The saved conversation does not match the classification.');
  const model=value.model.classifier,branch=result.branches[0];if(!classifierAvailable(value.model)||model.snapshotId!==result.snapshotId||model.step!==branch.step||model.runId!==branch.runId||!sameSource(model.source,branch.source))throw new Error('The classification does not match the published snapshot.');
  return value;
}
export function validateConversation(value){
  if(!object(value)||typeof value.id!=='string'||!Array.isArray(value.messages)||value.messages.length>20||!integer(value.submittedMessageCount)||value.submittedMessageCount>value.messages.length)throw new Error('The server returned an invalid conversation.');
  const ids=new Set();
  for(const [index,message]of value.messages.entries()){
    if(!object(message)||typeof message.id!=='string'||ids.has(message.id)||message.role!==(index%2?'assistant':'user')||typeof message.content!=='string'||message.content.length>20000)throw new Error('The server returned an invalid message.');
    if(message.role==='assistant'&&(message.kind==='classification'||message.classification!==undefined)){
      if(message.kind!=='classification'||validateClassification(message.classification).text!==value.messages[index-1].content)throw new Error('The classification does not match its review.');
    }
    ids.add(message.id);
  }
  if(value.messages.length%2)throw new Error('The server returned an incomplete exchange.');
  return value;
}
export function stopLabel(reason){return {eos:'EOS · the model ended its response',length:'Maximum token limit reached',context:'Context limit reached'}[reason]||'Response complete';}
export function canSubmit(conversation,busy,recovery,consent){return Boolean(conversation&&conversation.messages.length>=2&&conversation.messages.length>conversation.submittedMessageCount&&!busy&&!recovery&&consent);}
export function submissionRequest(conversation,nickname,comment){
  validateConversation(conversation);
  return {nickname,comment,consent:true,expectedConversationId:conversation.id,expectedMessageCount:conversation.messages.length};
}
export function apiError(status,value){const error=new Error(value?.error?.message||`Request failed (${status}).`);error.status=status;error.code=value?.error?.code;return error;}
export function isShareClosed(error){return error?.code==='share_closed'||error?.status===410;}

export async function consumeChatStream(response,onEvent){
  if(!response.ok){let error;try{error=await response.json();}catch{}throw apiError(response.status,error);}
  if(!response.headers.get('content-type')?.includes('application/x-ndjson')||!response.body)throw new Error('The server did not return a token stream.');
  const reader=response.body.getReader(),decoder=new TextDecoder('utf-8',{fatal:true});
  let buffer='',bytes=0,started=false,queued=false,tokenCount=0,lastContent='',terminal=null;
  function accept(line){
    let event;try{event=JSON.parse(line);}catch{throw new Error('The token stream contained invalid JSON.');}
    if(!object(event)||terminal)throw new Error('The token stream continued after completion.');
    if(event.type==='queued'){
      if(queued||started)throw new Error('Unexpected queue event.');queued=true;
    }else if(event.type==='start'){
      if(started||!object(event.userMessage)||event.userMessage.role!=='user'||typeof event.userMessage.content!=='string'||!object(event.model))throw new Error('Invalid stream start.');started=true;
    }else if(event.type==='token'){
      if(!started||event.index!==tokenCount||tokenCount>=160||typeof event.content!=='string'||event.content.length>20000)throw new Error('Invalid or out-of-order token.');tokenCount++;lastContent=event.content;
    }else if(event.type==='done'){
      if(!started||tokenCount===0||!object(event.assistantMessage)||event.assistantMessage.role!=='assistant'||event.assistantMessage.content!==lastContent||event.assistantMessage.tokenCount!==tokenCount||!['eos','length','context'].includes(event.assistantMessage.stopReason))throw new Error('The completed response does not match the received tokens.');
      validateConversation(event.conversation);
      const last=event.conversation.messages.at(-1);
      if(!last||last.id!==event.assistantMessage.id||last.content!==lastContent)throw new Error('The saved conversation does not match the response.');
      terminal=event;return;
    }else if(event.type==='error'){
      throw apiError(event.code==='share_closed'?410:503,{error:{code:event.code,message:typeof event.message==='string'?event.message:'Generation failed.'}});
    }else throw new Error('Unknown token stream event.');
    onEvent(event);
  }
  try{
    while(true){const {done,value}=await reader.read();if(done)break;bytes+=value.byteLength;if(bytes>4*1024*1024)throw new Error('The token stream exceeded its size limit.');buffer+=decoder.decode(value,{stream:true});let newline;while((newline=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,newline);buffer=buffer.slice(newline+1);if(line.length>131072)throw new Error('The token event exceeded its size limit.');if(line.trim())accept(line);}if(buffer.length>131072)throw new Error('The token event exceeded its size limit.');}
    buffer+=decoder.decode();if(buffer.trim())throw new Error('The token stream ended inside an event.');if(!terminal)throw new Error('Connection ended before completion. Recover the conversation before retrying.');onEvent(terminal);return terminal;
  }catch(error){try{await reader.cancel();}catch{}throw error;}finally{reader.releaseLock();}
}

if(typeof document!=='undefined'){
  const $=id=>document.getElementById(id),make=(tag,className,text)=>{const element=document.createElement(tag);if(className)element.className=className;if(text!==undefined)element.textContent=text;return element;};
  const state={csrf:null,conversation:null,model:null,modelVersion:0,limits:null,mode:'story',busy:false,recovery:false,abort:null,pending:null,closed:false,expiresAt:null,serverOffset:0,checkingShare:false,lastShareCheck:0};
  function status(text,error=false){$('chat-status').textContent=text;$('chat-status').classList.toggle('error',error);}
  function closeShare(){state.closed=true;state.recovery=false;state.abort?.abort();$('submit-consent').checked=false;$('share-expiry').hidden=false;$('share-expiry').textContent='This invitation has closed. Ask the presenter for a new QR code or link.';$('share-expiry').classList.add('closed');status('Audience access has ended. Previously submitted conversations remain in the presenter’s inbox.',true);controls();}
  function showExpiry(){
    if(state.closed||!state.expiresAt)return;
    const seconds=Math.max(0,Math.ceil((Date.parse(state.expiresAt)-Date.now()-state.serverOffset)/1000));$('share-expiry').hidden=false;$('share-expiry').textContent=seconds?`Temporary invitation · access ends ${new Date(state.expiresAt).toLocaleString('en-GB',{day:'numeric',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',timeZoneName:'short'})} · about ${Math.ceil(seconds/60)} minutes remaining`:'Checking whether this invitation is still open…';
  }
  function useExpiry(value){if(value.shareExpiresAt!==undefined&&value.shareExpiresAt!==null){if(typeof value.shareExpiresAt!=='string'||!Number.isFinite(Date.parse(value.shareExpiresAt)))throw new Error('The invitation expiry could not be read.');state.expiresAt=value.shareExpiresAt;showExpiry();}}
  function serverClock(response){const serverDate=Date.parse(response.headers.get('date'));if(Number.isFinite(serverDate))state.serverOffset=serverDate-Date.now();}
  async function checkShare(){
    if(state.closed||state.checkingShare)return;state.checkingShare=true;state.lastShareCheck=Date.now();
    const modelVersion=state.modelVersion,controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),10000);
    try{const response=await fetch('api/info',{cache:'no-store',credentials:'same-origin',signal:controller.signal});serverClock(response);const value=await response.json();if(!response.ok)throw apiError(response.status,value);useExpiry(value);if(!state.busy&&state.modelVersion===modelVersion&&object(value.model)){useModel(value.model);renderMode();}}
    catch(error){if(isShareClosed(error))closeShare();}
    finally{clearTimeout(timeout);state.checkingShare=false;}
  }
  function controls(){
    const ready=Boolean(state.csrf&&state.conversation),turnLimit=ready&&state.conversation.messages.length/2>=(state.limits?.maxTurns??10),classifier=classifierAvailable(state.model),locked=state.busy||!ready||state.recovery||state.closed||(state.mode==='classification'&&!classifier);
    $('chat-prompt').disabled=locked||turnLimit;$('send-chat').disabled=locked||turnLimit||!$('chat-prompt').value.trim();$('new-chat').disabled=state.busy||!ready||state.closed;$('stop-chat').hidden=!state.abort||state.closed;
    for(const button of document.querySelectorAll('[data-prompt]'))button.disabled=locked||turnLimit;
    $('mode-story').disabled=state.busy||!ready||state.closed;$('mode-classifier').disabled=state.busy||!ready||state.closed||!classifier;
    $('submit-chat').disabled=state.closed||!canSubmit(state.conversation,state.busy,state.recovery,$('submit-consent').checked);
    for(const id of ['visitor-name','discussion-note','submit-consent'])$(id).disabled=state.busy||state.closed;
    $('recover-chat').hidden=!state.recovery||state.closed;$('recover-chat').disabled=state.busy||state.closed;
    $('message-count').textContent=`${$('chat-prompt').value.length} / ${state.limits?.messageChars??800}`;
  }
  function renderMode(){
    const classify=state.mode==='classification',classifier=classifierAvailable(state.model),metadata=state.model?.classifier;
    $('mode-story').setAttribute('aria-pressed',String(!classify));$('mode-classifier').setAttribute('aria-pressed',String(classify));
    $('chat-title').textContent=classify?'Compare sentiment classifiers':'Your conversation';
    $('model-note').textContent=classify?'Compare two fine-tuned heads on the same English review: one started from random weights, the other from the pretrained language model. Your review is used only for prediction, never for training.':'This tiny model learned from English stories and is not instruction-tuned. It may continue your text instead of answering, repeat itself, or lose the thread. That is part of the experiment.';
    $('classifier-status').textContent=classifier?`Frozen classifier · ${metadata.step.toLocaleString()} fine-tuning steps per branch · copied LM step ${metadata.source.step.toLocaleString()}. Compare both branches, then choose whether to submit the result.`:'Sentiment classification is unavailable in this published snapshot. Story continuation is still available.';
    $('classifier-status').hidden=!classify&&classifier;
    $('chat-prompt').placeholder=classify?'Write a short English review to classify…':'Write a short English prompt…';
    $('chat-prompt').setAttribute('aria-label',classify?'Your review':'Your message');$('send-chat').textContent=classify?'Compare both classifiers':'Send message ↗';
    for(const button of document.querySelectorAll('[data-prompt]'))button.hidden=classify;
    $('generation-limits').textContent=classify?`Both frozen heads predict negative or positive; probabilities are model estimates. Up to ${state.limits?.maxTurns??10} completed experiments per conversation. Classification turns are excluded from later story prompts.`:`Each response uses the latest ${state.limits?.windowSize??128} tokens and generates at most ${state.limits?.maxNewTokens??160} new tokens. EOS ends it immediately. Up to ${state.limits?.maxTurns??10} experiments per conversation.`;
    controls();
  }
  async function api(action,data={}){
    const response=await fetch(`api/${action}`,{method:'POST',credentials:'same-origin',cache:'no-store',headers:{'Content-Type':'application/json',...(state.csrf?{'X-CSRF-Token':state.csrf}:{})},body:JSON.stringify(data)});
    serverClock(response);
    let value;try{value=await response.json();}catch{throw new Error('The server response could not be read.');}
    if(!response.ok)throw apiError(response.status,value);return value;
  }
  function modelText(model){return `Published LM · ${modelVersionLabel(model)}`;}
  function paintModel(){ $('model-badge').textContent=modelText(state.model); }
  function useModel(model){state.model=model;state.modelVersion++;paintModel();}
  function classificationCards(result){
    validateClassification(result);const comparison=make('div','classification-comparison');
    for(const name of ['pretrained','scratch']){
      const branch=result.branches.find(item=>item.branch===name),card=make('section','classification-branch');
      card.append(make('h3',null,name==='pretrained'?'Pretrained → sentiment':'Random → sentiment'),make('strong','classification-prediction',branch.prediction==='positive'?'Positive':'Negative'));
      for(const label of ['negative','positive']){const value=branch.probabilities.find(item=>item.label===label).p,row=make('div','classification-probability'),track=make('progress');track.max=1;track.value=value;track.setAttribute('aria-label',`${name}: ${label}, ${(100*value).toFixed(1)} percent`);row.append(make('span',null,label==='positive'?'Positive':'Negative'),track,make('span',null,`${(100*value).toFixed(1)}%`));card.append(row);}
      card.append(make('p','message-meta',name==='pretrained'?`Fine-tuning step ${branch.step} · source LM step ${branch.source.step} · source LM seed ${branch.source.seed}`:`Fine-tuning step ${branch.step} · trained from random weights`));
      const provenance=make('details','classification-provenance');provenance.append(make('summary',null,'Exact source and frozen snapshot'),make('p',null,`${name==='pretrained'?'Source LM':'Compared against source LM'}: ${branch.source.source} · step ${branch.source.step} · seed ${branch.source.seed} · SHA-256: ${branch.source.modelSha256}`),make('p',null,`Run: ${branch.runId}`),make('p',null,`Classifier snapshot: ${result.snapshotId}`));card.append(provenance);comparison.append(card);
    }
    return comparison;
  }
  function makeMessage(message,pending=false){
    const card=make('article',`message ${message.role}${pending?' pending':''}`),head=make('div','message-head',message.role==='user'?'YOU':'TINY GPT'),content=make('div','message-content',message.content),meta=make('p','message-meta');
    if(message.role==='assistant'&&!pending&&message.kind==='classification'){
      head.textContent='SENTIMENT CLASSIFIER COMPARISON';card.append(head,classificationCards(message.classification));return{card,content,meta};
    }
    if(message.role==='assistant'&&!pending)meta.textContent=`${stopLabel(message.stopReason)} · ${message.tokenCount??'—'} tokens · ${modelVersionLabel(message.model)}`;
    if(message.role==='user'&&message.kind==='classification')head.textContent='YOUR REVIEW';
    card.append(head,content);if(message.role==='assistant')card.append(meta);return {card,content,meta};
  }
  function follow(work){const box=$('chat-messages'),atEnd=box.scrollHeight-box.clientHeight-box.scrollTop<80;work();if(atEnd)box.scrollTop=box.scrollHeight;}
  function renderConversation(){
    const box=$('chat-messages');box.replaceChildren();
    if(!state.conversation.messages.length){const classify=state.mode==='classification',empty=make('div','empty-chat');empty.append(make('span','empty-mark','g'),make('h3',null,classify?'Try an English review.':'Start with a few words.'),make('p',null,classify?'Compare how the two trained heads interpret the same sentence.':'A short English story prompt is a good place to begin.'));box.append(empty);}
    else for(const message of state.conversation.messages)box.append(makeMessage(message).card);
    box.scrollTop=box.scrollHeight;$('prompt-suggestions').hidden=state.conversation.messages.length>0;
    if(state.conversation.messages.length===0)$('submit-status').textContent='Complete a story exchange or classification to submit.';
    else if(state.conversation.submittedMessageCount>=state.conversation.messages.length)$('submit-status').textContent='This conversation version has been submitted. A new experiment can be shared again.';
    else $('submit-status').textContent='Your completed experiments are ready to share if you choose.';
  }
  function useSession(value){
    if(typeof value.csrf!=='string'||!value.csrf||!object(value.model)||!object(value.limits))throw new Error('The session response is incomplete.');
    state.conversation=validateConversation(value.conversation);state.csrf=value.csrf;useModel(value.model);state.limits=value.limits;state.recovery=false;state.pending=null;
    useExpiry(value);
    $('submit-consent').checked=false;
    paintModel();$('chat-prompt').maxLength=value.limits.messageChars??800;
    renderConversation();renderMode();
  }
  async function openSession(){
    if(state.busy||state.closed)return;state.busy=true;controls();status('Opening your anonymous session…');
    try{useSession(await api('session'));status('Connected. Try a short English prompt.');}
    catch(error){if(isShareClosed(error))closeShare();else{state.recovery=true;status(`${error.message} Use Reconnect to try again.`,true);}}
    finally{state.busy=false;controls();}
  }
  async function send(event){
    event.preventDefault();const message=$('chat-prompt').value.trim();if(!message||$('send-chat').disabled)return;
    if(state.mode==='classification')return classify(message);
    $('submit-consent').checked=false;
    state.busy=true;state.abort=new AbortController();controls();status('Sending your prompt…');
    const user=makeMessage({role:'user',content:message}),assistant=makeMessage({role:'assistant',content:''},true);state.pending=assistant;
    if(!state.conversation.messages.length)$('chat-messages').replaceChildren();$('chat-messages').append(user.card,assistant.card);$('chat-messages').scrollTop=$('chat-messages').scrollHeight;
    try{
      const response=await fetch('api/chat',{method:'POST',credentials:'same-origin',cache:'no-store',headers:{'Content-Type':'application/json','X-CSRF-Token':state.csrf},body:JSON.stringify({message,requestId:crypto.randomUUID()}),signal:state.abort.signal});
      await consumeChatStream(response,frame=>{
        if(frame.type==='queued')status('Waiting for the model. Other visitors may be generating.');
        if(frame.type==='start'){useModel(frame.model);status('The trained model is generating…');}
        if(frame.type==='token')follow(()=>{assistant.content.textContent=frame.content;assistant.meta.textContent=`${frame.index+1} actual tokens received`;});
        if(frame.type==='done'){state.conversation=frame.conversation;useModel(frame.model);state.pending=null;$('chat-prompt').value='';renderConversation();renderMode();status(`${stopLabel(frame.assistantMessage.stopReason)}. You can continue or submit this conversation.`);}
      });
    }catch(error){
      assistant.card.classList.add('partial');assistant.card.classList.remove('pending');
      if(state.closed||isShareClosed(error)){assistant.meta.textContent='Generation stopped because this invitation closed. This partial response was not submitted.';closeShare();}
      else{assistant.meta.textContent='Unconfirmed response · recover the saved conversation before continuing.';state.recovery=true;status(error.name==='AbortError'?'Request stopped. Recover the conversation to check what was saved.':`${error.message} Recover the conversation to check what was saved.`,true);}
    }finally{state.abort=null;state.busy=false;controls();}
  }
  async function classify(message){
    if(state.busy||state.closed||!classifierAvailable(state.model))return;
    state.busy=true;state.abort=new AbortController();$('submit-consent').checked=false;controls();status('Comparing the two frozen sentiment classifiers…');
    const user=makeMessage({role:'user',kind:'classification',content:message}),pending=make('p','classification-pending','Classifying the review with both frozen heads…');
    if(!state.conversation.messages.length)$('chat-messages').replaceChildren();$('chat-messages').append(user.card,pending);
    const timeout=setTimeout(()=>state.abort?.abort(),60000);
    try{
      const response=await fetch('api/classify',{method:'POST',credentials:'same-origin',cache:'no-store',headers:{'Content-Type':'application/json','X-CSRF-Token':state.csrf},body:JSON.stringify({message,requestId:crypto.randomUUID()}),signal:state.abort.signal});serverClock(response);
      let value;try{value=await response.json();}catch{throw new Error('The classification response could not be read.');}if(!response.ok)throw apiError(response.status,value);
      validateClassificationResponse(value,message);if(state.closed)return;if(state.abort.signal.aborted)throw Object.assign(new Error('Classification stopped.'),{name:'AbortError'});
      state.conversation=value.conversation;useModel(value.model);state.recovery=false;$('chat-prompt').value='';renderConversation();renderMode();status('Both classifications are complete. Compare their probabilities and submit if the result is worth discussing.');
    }catch(error){
      if(state.closed||isShareClosed(error))closeShare();
      else if(error.code==='classifier_unavailable'){useModel({...state.model,classifier:{available:false}});renderConversation();renderMode();status('The classifier is unavailable. You can switch to Story continuation or reconnect to check the published snapshot.',true);}
      else{state.recovery=true;pending.textContent='Unconfirmed classification. Recover the saved conversation before continuing.';status(`${error.name==='AbortError'?'Classification was stopped or timed out.':error.message} Recover the conversation to check what was saved. The request will not be repeated automatically.`,true);}
    }finally{clearTimeout(timeout);state.abort=null;state.busy=false;controls();}
  }
  $('chat-form').addEventListener('submit',send);
  $('chat-prompt').addEventListener('input',controls);
  $('chat-prompt').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();if(!$('send-chat').disabled)$('chat-form').requestSubmit();}});
  for(const button of document.querySelectorAll('[data-prompt]'))button.addEventListener('click',()=>{$('chat-prompt').value=button.dataset.prompt;controls();$('chat-prompt').focus();});
  $('stop-chat').addEventListener('click',()=>state.abort?.abort());$('recover-chat').addEventListener('click',openSession);$('submit-consent').addEventListener('change',controls);
  for(const [id,mode]of [['mode-story','story'],['mode-classifier','classification']])$(id).addEventListener('click',()=>{if($(id).disabled)return;state.mode=mode;if(!state.conversation.messages.length)renderConversation();renderMode();$('chat-prompt').focus();});
  $('new-chat').addEventListener('click',async()=>{
    if(state.busy||state.closed)return;state.busy=true;controls();
    try{useSession(await api('new'));$('chat-prompt').value='';$('discussion-note').value='';$('submit-consent').checked=false;$('submit-status').className='status';status('New conversation. Submitted copies stay in the seminar inbox.');}
    catch(error){if(isShareClosed(error))closeShare();else status(error.message,true);}finally{state.busy=false;controls();}
  });
  $('submit-form').addEventListener('submit',async event=>{
    event.preventDefault();if($('submit-chat').disabled)return;state.busy=true;controls();$('submit-status').className='status';$('submit-status').textContent='Submitting this conversation for discussion…';
    try{
      const value=await api('submit',submissionRequest(state.conversation,$('visitor-name').value.trim(),$('discussion-note').value.trim()));
      if(!object(value.submission)||typeof value.submission.id!=='string'||value.submission.messageCount!==state.conversation.messages.length)throw new Error('The submission receipt could not be verified. Recover the conversation to check its status.');
      state.conversation.submittedMessageCount=value.submission.messageCount;$('submit-status').className='status success';$('submit-status').textContent=`${value.alreadySubmitted?'Already submitted':'Submitted'} · receipt ${value.submission.id.slice(0,8)}. The presenter can now open this conversation for discussion.`;
    }catch(error){
      $('submit-status').className='status error';$('submit-status').textContent=error.message;
      if(isShareClosed(error))closeShare();
      if(error.status===409&&error.code==='stale_conversation'){
        state.recovery=true;$('submit-consent').checked=false;
        const instruction='This conversation changed in another tab. Recover it, review the current messages, and agree to share again before submitting.';
        $('submit-status').textContent=instruction;status(instruction,true);
      }
    }
    finally{state.busy=false;controls();}
  });
  setInterval(()=>{if(document.hidden||state.closed)return;showExpiry();const due=state.expiresAt&&Date.parse(state.expiresAt)<=Date.now()+state.serverOffset;if(Date.now()-state.lastShareCheck>=(due?3000:30000))checkShare();},1000);
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)checkShare();});
  openSession();
}
