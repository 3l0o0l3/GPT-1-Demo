import {createTokenPlayback} from './token-playback.js';
const $ = id => document.getElementById(id);
const state = {online:false, busy:false, prepared:null, generated:[], selected:-1, finished:false, epoch:0,autoStarting:false};
let playback,autoRun=0;
const presets = {
  continuation:{mode:'raw',text:'A cat is an'},
  classification:{mode:'instruction',text:'Classify a cat as animal, plant, or vehicle. Answer with one word.'},
  translation:{mode:'instruction',text:'Translate this sentence into German. Output only the translation: The robot learns from experience.'},
  summary:{mode:'instruction',text:'Summarize this paragraph in one short sentence:\nA robot learns to sort parts by observing examples. At first, its predictions are often wrong. After training, it recognizes useful patterns and sorts new parts more accurately.'}
};

function element(tag, className, text){const node=document.createElement(tag);if(className)node.className=className;if(text!==undefined)node.textContent=text;return node;}
function status(message,error=false){$('status').textContent=message;$('status').classList.toggle('error',error);}
function requestError(error){
  if(['model_unavailable','ssh_unavailable','network_error'].includes(error.code)){
    state.online=false;$('connection-badge').textContent='Connection unavailable';$('connection-badge').className='badge offline';
  }
  status(error.message,true);
}
function visiblePiece(piece){return piece === '' ? '⟨end⟩' : piece.replaceAll(' ','▁').replaceAll('\n','↵').replaceAll('\t','⇥');}
function percent(value){return `${(value*100).toFixed(value<.001?3:1)}%`;}
function decodeTokens(tokens,final=false){
  const bytes=tokens.flatMap(token=>token.bytes || []);
  return new TextDecoder('utf-8').decode(new Uint8Array(bytes),{stream:!final});
}
function currentIds(){return [...(state.prepared?.tokens.map(token=>token.id)||[]),...state.generated.map(token=>token.id)];}

async function api(path,body){
  const controller=new AbortController();const timeout=setTimeout(()=>controller.abort(),130000);
  try{
    const response=await fetch(path,{method:body===undefined?'GET':'POST',headers:body===undefined?{}:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),signal:controller.signal,cache:'no-store'});
    let data;try{data=await response.json();}catch{throw new Error('The local gateway returned an unreadable response. Restart the local demo launcher.');}
    if(!response.ok){const error=new Error(data.error?.message||`Request failed (${response.status}).`);error.code=data.error?.code;error.status=response.status;throw error;}
    return data;
  }catch(error){
    if(error.name==='AbortError')throw new Error('The request timed out. The model may still be completing it; retry after it becomes available.');
    if(error instanceof TypeError){const unavailable=new Error('Cannot reach the local gateway. Start the local demo launcher, then retry.');unavailable.code='network_error';throw unavailable;}
    throw error;
  }finally{clearTimeout(timeout);}
}

function controls(){
  const blocked=state.busy||!state.online;
  const auto=playback?.snapshot()||{active:false,queued:0,inFlight:false,phase:'idle'};
  const autoLocked=auto.active||auto.queued>0||auto.inFlight||state.autoStarting;
  for(const id of ['preset','mode','prompt'])$(id).disabled=state.busy;
  for(const id of ['temperature','seed'])$(id).disabled=state.busy||autoLocked;
  $('prepare').disabled=blocked||!$('prompt').value.trim();
  const canGenerate=!blocked&&!autoLocked&&state.prepared&&!state.finished&&currentIds().length<2048;
  $('next').disabled=!canGenerate;$('batch').disabled=!canGenerate;
  $('auto-generate').disabled=blocked||auto.active||state.autoStarting||state.finished||!$('prompt').value.trim()||(state.prepared&&currentIds().length>=2048);
  $('auto-generate').textContent=auto.phase==='paused'||auto.phase==='error'?'Resume':'Auto generate';
  $('pause-auto').disabled=!auto.active&&!state.autoStarting;
  $('retry').hidden=state.online;$('retry').disabled=state.busy;
  $('experiment').setAttribute('aria-busy',String(state.busy||auto.inFlight));
  $('prepare').textContent=state.prepared?'Restart context':'Prepare context';
}

function emptyChart(){
  const box=element('div','empty-state');const bars=element('span','empty-bars');bars.setAttribute('aria-hidden','true');
  for(let i=0;i<3;i++)bars.append(element('i'));
  box.append(bars,element('h3',null,'A prediction has alternatives.'),element('p',null,'Press Next token after preparing the context. Each bar will show a probability reported by Qwen.'));
  $('probabilities').replaceChildren(box);$('step-label').textContent='Awaiting a token';
  document.querySelector('.chart-key span:last-child').textContent='Top candidates + remaining vocabulary';
  $('selected-probability').textContent='—';$('selected-surprise').textContent='—';
}

function invalidate(){
  autoRun++;state.autoStarting=false;playback.reset();
  state.epoch++;state.prepared=null;state.generated=[];state.selected=-1;state.finished=false;
  $('input-count').textContent='Not prepared';$('input-tokens').replaceChildren();
  $('formatted-prompt').textContent='Prepare a context to inspect the input.';
  $('mode-note').textContent=$('mode').value==='raw'?'The next token continues this exact prefix.':'The server applies Qwen’s chat template. This demonstrates prompting, not fine-tuning.';
  emptyChart();renderOutput();renderFormulas();controls();status('Context changed. Prepare it before generating.');
}

function renderInput(){
  const prepared=state.prepared;
  $('input-count').textContent=`${prepared.tokens.length} tokens`;
  const chips=prepared.tokens.map(token=>{const chip=element('span','token-chip',visiblePiece(token.piece));chip.append(element('small',null,`ID ${token.id}`));return chip;});
  $('input-tokens').replaceChildren(...chips);$('formatted-prompt').textContent=prepared.formattedPrompt;
}

function renderOutput(){
  const mode=state.prepared?.mode||$('mode').value;
  const text=state.prepared?.userText||$('prompt').value;
  document.querySelector('.chart-context').textContent=mode==='raw'?'Context at this step':'Context at this step · chat template applied';
  $('prefix-display').textContent=mode==='raw'?text:'';
  $('output-text').textContent=decodeTokens(state.generated,state.finished);
  $('output-placeholder').hidden=state.generated.length>0;
  $('generated-count').textContent=`${state.generated.length} ${state.generated.length===1?'token':'tokens'}`;
  $('output-title').textContent=mode==='raw'?'The continuation':'The answer';
  $('output-hint').textContent=state.generated.length?'Select a token below to inspect that exact prediction. ▁ marks a space; ⟨end⟩ marks the end of the answer.':'Generate one token to see the model’s actual output.';
  const chips=state.generated.map((token,index)=>{
    const chip=element('button',`token-chip${index===state.selected?' selected':''}`,visiblePiece(token.piece));
    chip.type='button';chip.setAttribute('aria-pressed',String(index===state.selected));
    chip.setAttribute('aria-label',`Token ${index+1}: ${token.piece||'end of answer'}, ID ${token.id}`);
    chip.append(element('small',null,`${index+1} · ID ${token.id}`));chip.addEventListener('click',()=>{state.selected=index;renderOutput();renderPrediction();});return chip;
  });
  $('generated-tokens').replaceChildren(...chips);
  if(state.selected<0)$('selected-context').textContent=mode==='raw'?text:`${text}\nAssistant: `;
}

function probabilityRow(label,p,{chosen=false,remaining=false,title=''}={}){
  const row=element('div',`prob-row${chosen?' chosen':''}${remaining?' remaining':''}`);
  const name=element('span','prob-label',label);name.title=title||label;
  const track=element('div','prob-track');track.setAttribute('aria-hidden','true');
  const fill=element('div','prob-fill');fill.style.width=`${Math.max(0,Math.min(1,p))*100}%`;if(p===0)fill.style.minWidth='0';track.append(fill);
  row.append(name,track,element('span','prob-value',percent(p)));return row;
}

function renderPrediction(){
  const token=state.generated[state.selected];if(!token)return;
  const previous=decodeTokens(state.generated.slice(0,state.selected));
  const context=state.prepared.mode==='raw'?state.prepared.userText+previous:`${state.prepared.userText}\nAssistant: ${previous}`;
  $('selected-context').textContent=context;$('selected-context').title=state.prepared.formattedPrompt+previous;
  document.querySelector('.chart-context').textContent=state.prepared.mode==='raw'?'Context at this step':'Context at this step · chat template applied';
  $('step-label').textContent=`Token ${state.selected+1} of ${state.generated.length}`;
  const candidates=[...token.candidates];const chosenOutside=!candidates.some(candidate=>candidate.id===token.id);
  if(chosenOutside)candidates.push({id:token.id,piece:token.piece,p:token.p,logprob:token.logprob});
  const rows=candidates.map(candidate=>probabilityRow(visiblePiece(candidate.piece),candidate.p,{chosen:candidate.id===token.id,title:`Token ID ${candidate.id}${candidate.id===token.id?' · chosen':''}`}));
  const remaining=Math.max(0,Math.min(1,1-candidates.reduce((sum,candidate)=>sum+candidate.p,0)));
  rows.push(probabilityRow('All other tokens',remaining,{remaining:true}));$('probabilities').replaceChildren(...rows);
  document.querySelector('.chart-key span:last-child').textContent=chosenOutside?'Top candidates + chosen token + remainder':'Top candidates + remaining vocabulary';
  $('selected-probability').textContent=percent(token.p);$('selected-surprise').textContent=(-token.logprob).toFixed(3);
  renderFormulas(token);
}

function renderFormulas(token){
  if(!window.katex)return;
  const options={throwOnError:false,displayMode:true};
  window.katex.render(String.raw`p_\theta(x_t\mid x_{<t})=\big[\operatorname{softmax}(z_t)\big]_{x_t}`,$('probability-formula'),options);
  const formula=token?String.raw`-\ln p(x_t\mid x_{<t})=${(-token.logprob).toFixed(3)}\;\text{nats}`:String.raw`\text{surprise}=-\ln p(\text{chosen token})`;
  window.katex.render(formula,$('surprise-formula'),options);
}

async function prepare({fromAuto=false}={}){
  if(state.busy)return false;
  if(!fromAuto){autoRun++;state.autoStarting=false;}
  playback.reset();
  state.busy=true;controls();status('Tokenizing the context on Qwen…');const epoch=state.epoch;
  try{
    const text=$('prompt').value;const result=await api('/api/prepare',{text,mode:$('mode').value});
    if(epoch!==state.epoch)return false;
    state.prepared={...result,userText:text};state.generated=[];state.selected=-1;state.finished=false;
    renderInput();emptyChart();renderOutput();renderFormulas();
    status(result.tokens.length>=2048?'Context is at the 2048-token demo limit. Shorten the text before generating.':`${result.tokens.length} real token IDs ready. Generate the next token to inspect its alternatives.`);
    return true;
  }catch(error){requestError(error);return false;}finally{state.busy=false;controls();}
}

async function generate(steps){
  const auto=playback.snapshot();
  if(state.busy||!state.prepared||state.finished||auto.active||auto.queued||auto.inFlight||state.autoStarting)return;
  const seed=Number($('seed').value);if(!$('seed').value||!Number.isInteger(seed)||seed<0||seed>2147483647){status('Enter a whole-number random seed between 0 and 2147483647.',true);return;}
  const ids=currentIds();if(ids.length>=2048){status('The sequence reached the 2048-token demo limit. Restart with a shorter context.',true);controls();return;}
  state.busy=true;controls();const epoch=state.epoch;const temperature=Number($('temperature').value);
  status(steps===1?'Qwen is computing the next token…':`Qwen is generating up to ${steps} tokens. The batch will appear when complete…`);
  try{
    const result=await api('/api/generate',{tokens:ids,steps:Math.min(steps,2048-ids.length),temperature,seed});if(epoch!==state.epoch)return;
    if(!Array.isArray(result.tokens)||result.probabilityKind!=='pre-sampling')throw new Error('The model response does not contain supported token probabilities.');
    const start=state.generated.length;state.generated.push(...result.tokens.map(token=>({...token,temperature})));state.finished=result.finished;
    state.selected=result.tokens.length?start:state.selected;renderOutput();renderPrediction();
    const finish=state.finished?'The model ended its answer. Restart context to try again.':currentIds().length>=2048?'The sequence reached the demo limit. Restart with a shorter context.':'Each chosen token is now part of the next context.';
    status(`${result.tokens.length} token${result.tokens.length===1?'':'s'} returned by Qwen. ${finish}`);
  }catch(error){requestError(error);}finally{state.busy=false;controls();}
}

async function requestAutomaticBatch(){
  if(!state.prepared)throw new Error('Prepare a context before automatic generation.');
  const ids=currentIds(),remaining=2048-ids.length;
  if(remaining<=0)return{tokens:[],finished:false,limit:true};
  const seed=Number($('seed').value),temperature=Number($('temperature').value);
  if(!$('seed').value||!Number.isInteger(seed)||seed<0||seed>2147483647)throw new Error('Enter a whole-number random seed between 0 and 2147483647.');
  const result=await api('/api/generate',{tokens:ids,steps:Math.min(16,remaining),temperature,seed});
  if(!Array.isArray(result.tokens)||result.probabilityKind!=='pre-sampling')throw new Error('The model response does not contain supported token probabilities.');
  return{tokens:result.tokens.map(token=>({...token,temperature})),finished:result.finished,limit:!result.finished&&ids.length+result.tokens.length>=2048};
}

function updatePlayback(view){
  const rateLabel=`${view.rate} ${view.rate===1?'token':'tokens'}/s`;
  const bufferLabel=`${view.queued} ${view.queued===1?'token':'tokens'} buffered`;
  let message='Automatic display is off. Choose a speed, then start.';
  if(state.autoStarting)message='Preparing the context for automatic generation…';
  else if(view.phase==='waiting')message=`Waiting for Qwen… Display target: ${rateLabel}.`;
  else if(view.phase==='playing')message=`Displaying ${rateLabel} · ${bufferLabel}.`;
  else if(view.phase==='paused')message=view.inFlight?'Paused. The pending batch may finish into the buffer.':`Paused · ${bufferLabel}. Resume to continue.`;
  else if(view.phase==='error')message='Automatic display stopped after an error. Check the message above.';
  else if(view.phase==='complete'){
    message=view.reason==='limit'?'Reached the 2048-token demo limit. Restart with a shorter context.':'Answer complete. Every returned token has been displayed.';
    if(view.reason==='eos'){state.finished=true;renderOutput();}
    status(message);
  }
  $('playback-status').textContent=message;controls();
}

async function startAutomatic(){
  if(state.busy||!state.online||state.finished)return;
  const intent=++autoRun;state.autoStarting=true;updatePlayback(playback.snapshot());
  if(!state.prepared){
    const ready=await prepare({fromAuto:true});
    if(intent!==autoRun)return;
    if(!ready){state.autoStarting=false;updatePlayback(playback.snapshot());return;}
  }
  if(intent!==autoRun)return;
  state.autoStarting=false;status('Automatic display uses real buffered tokens. The speed is a presentation setting.');playback.start();
}

function pauseAutomatic(){
  autoRun++;state.autoStarting=false;playback.pause();
  if(playback.snapshot().phase==='idle')$('playback-status').textContent='Paused before generation. Press Auto generate when ready.';
  controls();
}

playback=createTokenPlayback({
  requestBatch:requestAutomaticBatch,rate:Number($('display-rate').value),
  onToken(token,meta){state.generated.push(token);state.selected=state.generated.length-1;if(meta.finished)state.finished=true;renderOutput();renderPrediction();},
  onState:updatePlayback,onError(error){requestError(error);controls();}
});

async function connect(){
  if(state.busy)return;state.busy=true;controls();status('Checking the Qwen service through SSH home…');
  try{
    const model=await api('/api/model');if(!model.online)throw new Error('The model is not online.');
    state.online=true;$('connection-badge').textContent='Connected to home';$('connection-badge').className='badge online';
    $('model-name').textContent=model.model.split(/[\\/]/).pop();$('model-name').title=model.model;status('Qwen is connected. Prepare a context to begin.');
  }catch(error){state.online=false;$('connection-badge').textContent='Model unavailable';$('connection-badge').className='badge offline';$('model-name').textContent='Start the local demo launcher';status(error.message,true);}
  finally{state.busy=false;controls();}
  if(state.online&&!state.prepared)await prepare();
}

$('preset').addEventListener('change',()=>{const preset=presets[$('preset').value];$('mode').value=preset.mode;$('prompt').value=preset.text;invalidate();});
$('mode').addEventListener('change',invalidate);$('prompt').addEventListener('input',invalidate);
$('temperature').addEventListener('input',()=>{$('temperature-value').textContent=Number($('temperature').value)===0?'0 · greedy':Number($('temperature').value).toFixed(1);});
$('prepare').addEventListener('click',()=>prepare());$('next').addEventListener('click',()=>generate(1));$('batch').addEventListener('click',()=>generate(16));$('retry').addEventListener('click',connect);
$('auto-generate').addEventListener('click',startAutomatic);$('pause-auto').addEventListener('click',pauseAutomatic);
$('display-rate').addEventListener('change',()=>playback.setRate(Number($('display-rate').value)));
emptyChart();renderOutput();renderFormulas();window.addEventListener('load',()=>renderFormulas(state.generated[state.selected]));connect();
