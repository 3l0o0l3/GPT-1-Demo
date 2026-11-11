import {readGenerationStream,TokenPlayer} from './token-stream.js';
import {sessionFetch,trainingBase,hosted} from './presenter-session.js';
import {createTrainingJobs,jobIsActive,trainingMode,runTeachingSteps} from './training-jobs.js';
const $=id=>document.getElementById(id);
const trainingEndpoint=trainingBase;
const lab={ready:false,busy:false,tab:'data',meta:null,live:null,examples:[],inspection:null,inspectionStep:null,trace:null,stage:0,maxStage:0,mode:'live',recordings:null,checkpoint:0,playing:false,timer:null,training:false,pause:false,seed:null,liveUseAvailable:false,liveUseError:null,liveTransfer:null,transferTraining:false,transferPause:false,saveReceipt:null,saveError:null,saving:false,generation:null,job:null,jobPending:false,jobConnected:false,jobDiscovered:false,jobUncertain:false,jobError:null,teaching:null};
let seminarGuide=null,tokenHistoryView=null;
lab.trainingExample=null;
const branches={scratch:{label:'From random weights',caption:'Classification only · λ = 0',color:'#8a96a0'},pretrained:{label:'After pretraining',caption:'Classification only · λ = 0',color:'#00549f'},auxiliary:{label:'Pretraining + auxiliary LM',caption:'Classification + LM · λ = 0.5',color:'#197a67'}};
const make=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const number=(value,digits=3)=>finite(value)?value.toFixed(digits):'—';
const pct=value=>finite(value)?`${(100*value).toFixed(value<.001?3:1)}%`:'—';
const precise=value=>finite(value)?(value!==0&&Math.abs(value)<.001?value.toExponential(4):value.toFixed(5)):'—';
const piece=value=>String(value??'').replaceAll('Ġ','▁').replaceAll('Ċ','↵').replaceAll(' ','▁').replaceAll('\n','↵').replaceAll('\t','⇥')||'⟨empty⟩';
const message=(text,error=false)=>{$('lab-status').textContent=text;$('lab-status').classList.toggle('error',error);};
function math(id,source){if(window.katex)window.katex.render(source,$(id),{throwOnError:false,displayMode:true});else $(id).textContent=source;}
function empty(id,text){$(id).replaceChildren(make('p','empty-message',text));}

async function api(path,body,{optional=false}={}){
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),optional?8000:180000);
  try{
    const response=await sessionFetch(`${trainingEndpoint}${path}`,{method:body===undefined?'GET':'POST',headers:body===undefined?{}:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),cache:'no-store',signal:controller.signal});
    let value;try{value=await response.json();}catch{throw new Error('The training service returned an unreadable response. Restart the local launcher.');}
    if(!response.ok){const error=new Error(value.error?.message||`Training request failed (${response.status}).`);error.code=value.error?.code;error.status=response.status;if(response.status>=500&&!optional){lab.ready=false;controls();}throw error;}return value;
  }catch(error){if(error.name==='AbortError'){if(!optional){lab.ready=false;controls();}throw new Error('The GPU request timed out and may have completed. Use Retry to refresh the actual model state before continuing.');}if(error instanceof TypeError){if(!optional){lab.ready=false;controls();}throw new Error(hosted?'Cannot reach the configured training runtime. Use Retry to refresh model state before repeating an action.':'Cannot reach the local training runtime. Start public_server.py, then use Retry.');}throw error;}
  finally{clearTimeout(timer);}
}

let jobPaint=null;
function queueJobPlots(){
  if(jobPaint!==null)return;
  jobPaint=requestAnimationFrame(()=>{jobPaint=null;if(lab.ready){renderConvergence();renderLiveTransfer();renderUpdate();}});
}
// PRESENTER_PROGRESS_VIEW_START
export function presenterProgressView({job=null,pending=false,uncertain=false,error=null,pendingKind=null}={}){
  const actualActive=job?.status==='running'||job?.status==='pause_requested';
  const kind=(pending||uncertain)&&pendingKind&&!actualActive?pendingKind:job?.kind||pendingKind;
  const state=uncertain?'uncertain':pending?'pending':error?'error':job?.status||'idle';
  return {kind:['lm','transfer'].includes(kind)?kind:'unknown',active:['pending','uncertain','error','running','pause_requested','failed'].includes(state),state};
}
// PRESENTER_PROGRESS_VIEW_END
function renderJobProgress(){
  const teaching=lab.teaching&&(lab.teaching.active||!jobIsActive(lab.job))?lab.teaching:null,job=teaching||lab.job,box=$('training-job-progress');box.hidden=!job&&!lab.jobPending&&!lab.jobUncertain;
  const presentation=presenterProgressView({job,pending:lab.jobPending,uncertain:lab.jobUncertain,error:teaching?null:lab.jobError,pendingKind:lab.jobDisplayKind});
  box.dataset.presenterKind=presentation.kind;box.dataset.presenterActive=String(presentation.active);box.dataset.presenterState=presentation.state;
  if(box.hidden)return;
  $('job-kind').textContent=job?`${teaching?'Teaching mode':jobIsActive(job)?'Background training':'Latest background task'} · ${job.kind==='lm'?'Language model':'Paired fine-tuning'}`:'Starting background training';
  $('job-count').textContent=job?`${job.completedSteps.toLocaleString()} / ${job.requestedSteps.toLocaleString()} ${job.kind==='transfer'?'steps per branch':'steps'} · model step ${job.currentStep.toLocaleString()}`:'Waiting for the server to identify the task…';
  $('job-elapsed').textContent=job?`${number(job.elapsedSeconds,1)} s elapsed`:'—';
  $('job-rate').textContent=job?`${number(job.stepsPerSecond,1)} ${job.kind==='transfer'?'paired steps/s':'steps/s'}`:'—';
  const progress=$('job-progress-bar');progress.max=job?.requestedSteps||1;progress.value=job?.completedSteps||0;
  progress.setAttribute('aria-valuetext',job?`${job.completedSteps} of ${job.requestedSteps} real steps completed`:'Waiting for task confirmation');
  const status=!teaching&&lab.jobError?`Connection interrupted: ${lab.jobError} Checking progress without restarting training.`:!teaching&&lab.jobUncertain?'Start not yet confirmed. Checking for the submitted task; the request will not be repeated.':({running:teaching?'Training one block; progress and measured loss are shown separately.':'Training continues on the server.',pause_requested:'Pause requested. Finishing the current block of up to 10 steps.',paused:'Paused at the displayed step. Start another task to add more steps.',completed:'All requested optimizer steps completed.',failed:`Task stopped: ${job?.error?.message||'See the reported training error.'}`})[job?.status]||'Submitting one training task…';
  $('job-status').textContent=status;$('job-status').classList.toggle('error',(!teaching&&Boolean(lab.jobError))||job?.status==='failed');
  const measuredState=job?.kind==='transfer'?lab.liveTransfer:lab.live,interval=measurementInfo(measuredState,job?.kind==='transfer').interval;
  $('job-progress-note').textContent=teaching?'Teaching mode reports each completed block. Loss values keep their measured step until the next scheduled evaluation. Keep this page open; pause finishes the current request.':`One request starts the whole background task. Progress is checked every 0.1 s; ${interval?`loss evaluation runs every ${interval} optimizer steps; `:''}plots redraw only when measured observations change. Closing this page does not stop training; pause finishes the current block of up to 10 steps.`;
  $('job-pause').hidden=teaching?!teaching.active:!jobIsActive(job);
}
function syncTrainingFlags(){
  const teaching=lab.teaching?.active?lab.teaching:null,background=jobIsActive(lab.job)?lab.job:null;
  lab.training=teaching?.kind==='lm'||background?.kind==='lm';lab.transferTraining=teaching?.kind==='transfer'||background?.kind==='transfer';
  lab.pause=teaching?.kind==='lm'?teaching.pause:background?.kind==='lm'&&background.status==='pause_requested';
  lab.transferPause=teaching?.kind==='transfer'?teaching.pause:background?.kind==='transfer'&&background.status==='pause_requested';
}
function pauseTraining(){
  if(lab.teaching?.active){lab.teaching.pause=true;lab.teaching.status='pause_requested';syncTrainingFlags();renderJobProgress();controls();message('Pause requested. Teaching mode finishes the current 10-step request, then stops.');return;}
  return trainingJobs.pause();
}
async function teachSteps(kind,total,learningRate){
  if(lab.busy||jobLocked()||!lab.ready)return;
  const initial=kind==='lm'?lab.live:lab.liveTransfer;
  if(!initial||(kind==='transfer'&&!initial.available))return;
  const run={kind,active:true,pause:false,status:'running',requestedSteps:total,completedSteps:0,initialStep:initial.step,currentStep:initial.step,started:Date.now(),elapsedSeconds:0,stepsPerSecond:0,error:null};
  lab.teaching=run;lab.busy=true;lab.trace=null;if(kind==='lm')lab.mode='live';else clearLivePrediction();
  pauseRecording();syncTrainingFlags();renderUpdate();renderJobProgress();controls();
  const timing=()=>{run.elapsedSeconds=Math.max(0,(Date.now()-run.started)/1000);run.stepsPerSecond=run.elapsedSeconds?run.completedSteps/run.elapsedSeconds:0;};
  try{
    const result=await runTeachingSteps({kind,steps:total,learningRate,initialState:initial,request:(path,body)=>api(path,body),shouldPause:()=>run.pause,
      onChunk:({state,completedSteps})=>{run.completedSteps=completedSteps;run.currentStep=state.step;timing();if(kind==='lm'){lab.live=state;renderConvergence();}else{lab.liveTransfer=state;renderLiveTransfer();}renderJobProgress();message(`Teaching mode: ${completedSteps}/${total} real ${kind==='transfer'?'steps per branch':'steps'} completed. Loss values retain their actual measurement step.`);}});
    run.status=result.paused?'paused':'completed';message(`Teaching mode ${result.paused?'paused':'complete'}: ${result.completedSteps}/${total} real steps; current step ${run.currentStep}.`);
  }catch(error){run.status='failed';run.error={message:error.message};lab.ready=false;message(`${error.message} Teaching mode stopped; no request is retried. Use Retry to read the actual model state before training again.`,true);}
  finally{
    run.active=false;timing();syncTrainingFlags();
    if(jobIsActive(lab.job)){lab.live=lab.job.liveState;lab.liveTransfer=lab.job.transferState;queueJobPlots();}
    else if(lab.ready&&lab.inspection){try{const inspection=await api('inspect',{text:lab.inspection.text,position:lab.inspection.position},{optional:true});if(!jobIsActive(lab.job)){lab.inspection=inspection;lab.inspectionStep=lab.live.step;renderInspection();}}catch(error){if(error.code!=='training_busy')message(`Training progress is retained. Refresh inspection when connected: ${error.message}`,true);}}
    lab.busy=false;syncTrainingFlags();renderJobProgress();controls();
  }
}
async function refreshAfterJob(id){
  if(lab.busy||jobLocked())return;
  const transferRunId=lab.liveTransfer?.runId??null;
  try{
    const live=await api('state',undefined,{optional:true}),transfer=await api('transfer-state',undefined,{optional:true});
    if(lab.job?.id!==id||jobLocked()||lab.busy||(lab.liveTransfer?.runId??null)!==transferRunId)return;
    lab.live=live;lab.liveTransfer=transfer;queueJobPlots();
    const acceptedTransferRunId=lab.liveTransfer?.runId??null;
    if(lab.inspection){
      const inspection=await api('inspect',{text:lab.inspection.text,position:lab.inspection.position},{optional:true});
      if(lab.job?.id===id&&!jobLocked()&&!lab.busy&&(lab.liveTransfer?.runId??null)===acceptedTransferRunId){lab.inspection=inspection;lab.inspectionStep=lab.live.step;renderInspection();}
    }
  }catch(error){if(error.code!=='training_busy')message(`Training status is retained. Refresh model details when connected: ${error.message}`,true);}
}
const trainingJobs=createTrainingJobs({
  request:(path,body)=>api(path,body,{optional:true}),
  onUpdate:event=>{
    const {job,previous}=event,wasConnected=lab.jobConnected;
    lab.job=job;lab.jobPending=event.pending;lab.jobConnected=event.connected;lab.jobUncertain=event.uncertain;lab.jobDiscovered=true;
    if(event.active&&lab.teaching?.active){lab.teaching.pause=true;lab.teaching.status='pause_requested';}
    else if(event.active)lab.teaching=null;
    syncTrainingFlags();
    if(event.connected)lab.jobError=null;
    if(!lab.teaching?.active&&event.snapshotChanged&&job&&(event.active||event.source==='start'||jobIsActive(previous))){
      lab.live=job.liveState;lab.liveTransfer=job.transferState;lab.trace=null;
      if(job.kind==='lm')lab.mode='live';else clearLivePrediction();
      queueJobPlots();
    }
    if(!lab.teaching?.active&&event.active&&(event.changed||!wasConnected))message(`${job.kind==='lm'?'Language-model training':'Paired fine-tuning'} runs in the background: ${job.completedSteps}/${job.requestedSteps} real steps completed. You can navigate freely; closing the page does not stop training.`);
    if(!lab.teaching?.active&&event.changed&&job&&!event.active&&event.source!=='discovery'){
      message(`${job.status==='completed'?'Training complete':job.status==='paused'?'Training paused':'Training stopped'}: ${job.completedSteps}/${job.requestedSteps} real steps; current step ${job.currentStep}.${job.error?` ${job.error.message}`:''}`,job.status==='failed');
      void refreshAfterJob(job.id);
    }
    renderJobProgress();controls();
  },
  onError:(error,{source})=>{lab.jobError=error.message;if(!lab.teaching?.active)message(`${source==='start'?'Start was not confirmed. ':''}${error.message} Status checks will continue; no training request is repeated.`,true);}
});

function jobLocked(){return lab.jobPending||lab.training||lab.transferTraining||lab.jobUncertain||(lab.jobDiscovered&&!lab.jobConnected);}
function controls(){
  const locked=lab.busy||!lab.ready||jobLocked();
  for(const id of ['story-example','inspect-text','inspect-button','preview-update','update-lr','train-1000','train-5000','training-lr','reset-seed','reset-random','reset-pretrained','refresh-recordings','refresh-transfer','review-text','classifier-branch'])$(id).disabled=locked;
  $('preview-update').disabled=locked||!lab.trainingExample;
  $('advance-update').disabled=locked||!lab.trace;
  $('pause-training').disabled=!lab.training||lab.pause||lab.jobPending;
  $('job-pause').disabled=!(lab.training||lab.transferTraining)||lab.jobPending||(lab.teaching?.active?lab.teaching.pause:lab.job?.status==='pause_requested');
  $('mode-live').disabled=lab.training;$('mode-recorded').disabled=lab.training;
  const checkpoints=getCheckpoints();
  $('play-recording').disabled=locked||!checkpoints.length||lab.playing;
  $('pause-recording').disabled=!lab.playing;
  $('checkpoint-slider').disabled=locked||!checkpoints.length;
  $('transfer-seed').disabled=locked||!getRuns().length;
  $('predict-review').disabled=locked||!lab.recordings?.transfer?.available||!$('review-text').value.trim();
  $('retry-training').disabled=lab.busy;
  $('retry-training').hidden=lab.ready&&lab.jobConnected&&!lab.jobUncertain;
  const liveLocked=locked||!lab.liveUseAvailable;
  for(const id of ['generation-prompt','generation-tokens','generation-temperature','generation-seed','generation-sliding','live-transfer-seed','live-transfer-auxiliary','live-transfer-lr','live-review-text','live-classifier-branch','save-live','save-transfer','start-live-transfer'])$(id).disabled=liveLocked;
  $('generation-window-size').disabled=liveLocked||!$('generation-sliding').checked;
  for(const id of ['generate-current','generate-compare'])$(id).disabled=liveLocked||!$('generation-prompt').value.trim();
  const generating=lab.generation?.active&&!lab.generation.stopped;
  $('generation-pause').disabled=!generating;
  $('generation-pause').textContent=lab.generation?.paused&&generating?'Resume display':'Pause display';
  $('generation-pause').setAttribute('aria-pressed',String(Boolean(lab.generation?.paused&&generating)));
  $('generation-stop').disabled=!generating;
  for(const id of ['transfer-train-500','transfer-train-1000'])$(id).disabled=liveLocked||!lab.liveTransfer?.available||lab.liveTransfer.requiresRestart===true;
  $('pause-transfer').disabled=!lab.transferTraining||lab.transferPause||lab.jobPending;
  $('predict-live-review').disabled=liveLocked||!lab.liveTransfer?.available||!(lab.liveTransfer.step>=1)||!$('live-review-text').value.trim();
  $('refresh-live-transfer').disabled=locked;
  for(const button of document.querySelectorAll('[data-stage]'))button.disabled=locked||!lab.trace||Number(button.dataset.stage)>lab.maxStage;
  for(const button of document.querySelectorAll('.shift-pair button'))button.disabled=locked;
  tokenHistoryView?.setLocked(locked);
}

async function action(work,notice){
  if(lab.busy||jobLocked())return;lab.busy=true;controls();if(notice)message(notice);
  try{return await work();}catch(error){message(error.message,true);return null;}finally{lab.busy=false;controls();}
}

function changeTab(tab){
  if(tab!=='convergence')pauseRecording();lab.tab=tab;
  for(const button of document.querySelectorAll('[data-tab]')){const active=button.dataset.tab===tab;button.classList.toggle('active',active);button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;}
  for(const name of ['data','update','convergence','transfer'])$(`panel-${name}`).hidden=name!==tab;
  seminarGuide?.followTab(tab);
  tokenHistoryView?.syncVisibility();
}

function probabilities(id,items,targetId,{remaining=true}={}){
  const container=typeof id==='string'?$(id):id;
  if(!Array.isArray(items)||!items.length){container.replaceChildren(make('p','empty-message','No measured probabilities available.'));return;}
  const rows=items.filter(item=>finite(item.p)).map(item=>{
    const row=make('div',`prob-row${targetId!==undefined&&item.id===targetId?' chosen':''}`),label=make('span','prob-label',piece(item.piece??item.label));label.title=`${item.piece??item.label} · ${item.id===undefined?'':`ID ${item.id}`}`;
    const track=make('div','prob-track'),fill=make('div','prob-fill');fill.style.width=`${Math.max(0,Math.min(1,item.p))*100}%`;track.append(fill);row.append(label,track,make('span','prob-value',pct(item.p)));return row;
  });
  if(remaining){const mass=Math.max(0,Math.min(1,1-items.reduce((sum,item)=>sum+(finite(item.p)?item.p:0),0)));const row=make('div','prob-row remaining'),track=make('div','prob-track'),fill=make('div','prob-fill');fill.style.width=`${mass*100}%`;track.append(fill);row.append(make('span','prob-label','All other tokens'),track,make('span','prob-value',pct(mass)));rows.push(row);}
  container.replaceChildren(...rows);
}

function compiledLmReady(execution){return execution?.lmBackend==='compiled'&&execution.compileEnabled===true&&execution.compileReady===true;}
function trainingPrecision(execution){return ['FP32','FP16 AMP'].includes(execution?.trainingPrecision)?execution.trainingPrecision:'';}
function executionLabel(){
  const execution=lab.meta?.execution,precision=trainingPrecision(execution);if(!precision)return '';
  return compiledLmReady(execution)?`${precision} · compiled LM training`:execution.lmBackend==='eager'?`${precision} · eager LM training`:`${precision} training`;
}
function compiledTransferReady(execution){return execution?.transferBackend==='compiled-dynamic'&&execution.transferCompileEnabled===true&&execution.transferCompileReady===true;}
function transferExecutionLabel(){
  const execution=lab.meta?.execution,precision=execution?.transferPrecision;if(!['FP32','FP16 AMP'].includes(precision))return '';
  return compiledTransferReady(execution)?`${precision} · compiled transfer training`:execution.transferBackend==='eager'?`${precision} · eager transfer training`:`${precision} transfer training`;
}
function renderMetadata(){
  const meta=lab.meta,a=meta.architecture||{},data=meta.data||{},execution=meta.execution,label=executionLabel();
  $('architecture').textContent=`${a.layers??'—'} layers · ${a.heads??'—'} heads · width ${a.width??'—'}`;
  $('engine-detail').textContent=`${finite(a.parameters)?a.parameters.toLocaleString():'—'} parameters · ${meta.host||meta.hostname||'home'} · ${meta.device||'CUDA'}${meta.gpuName?' · NVDIA GPU':''} · seed ${meta.seed??'—'}${label?` · ${label}`:''}`;
  const provenance=document.createDocumentFragment();
  for(const source of data.sources||[]){const title=make('a','source-link',source.name);if(/^https?:\/\//.test(source.url||'')){title.href=source.url;title.target='_blank';title.rel='noopener noreferrer';}provenance.append(title,make('p',null,source.description||''),make('p',null,`License: ${source.license||'See source metadata'}`));}
  const counts=data.counts||{};provenance.append(make('p',null,`TinyStories: ${counts.storiesTrain?.toLocaleString()??'—'} training stories; ${counts.storiesValidation?.toLocaleString()??'—'} validation stories.`),make('p',null,`UCI sentiment: ${counts.sentimentTrain??'—'} train / ${counts.sentimentValidation??'—'} validation / ${counts.sentimentTest??'—'} test. Test data is not used by this page.`),make('p',null,`Split seed: ${data.splitSeed??'—'}. Selection: ${typeof data.selection==='string'?data.selection:JSON.stringify(data.selection||{})}`));
  $('data-provenance').replaceChildren(provenance);
  const modelBlock=make('section'),dataBlock=make('section');modelBlock.append(make('h3',null,'The actual model'),make('p',null,meta.description||'A small decoder-only transformer with genuine PyTorch training.'),make('p',null,`${a.layers} layers, ${a.heads} heads, embedding width ${a.width}, maximum context ${a.context}, vocabulary ${a.vocabSize}. Parameters: ${a.parameters?.toLocaleString()??'—'}.`));
  if(a.trainContext)modelBlock.append(make('p',null,`Live batch-training window: ${a.trainContext} tokens. Inspection supports up to ${a.context} input positions.`));
  if(label){
    const shape=execution.compiledShape,notes=[],precision=trainingPrecision(execution);
    if(compiledLmReady(execution)&&Array.isArray(shape)&&shape.length===2&&shape.every(value=>Number.isSafeInteger(value)&&value>0))notes.push(`Compiled LM training is used for eligible batches of ${shape[0]} sequences × ${shape[1]} input tokens. Other batch shapes use eager ${precision}.`);
    else if(execution.lmBackend==='eager')notes.push(`Language-model training uses eager ${precision}.`);
    if(execution.parameterPrecision==='FP32')notes.push(execution.optimizer==='Adam'&&execution.optimizerStatePrecision==='FP32'?'Model weights and Adam state remain FP32.':'Model weights remain FP32.');
    if(execution.optimizer==='Adam'&&execution.optimizerImplementation==='default')notes.push('Adam uses its default implementation.');
    if(execution.previewBackend==='eager'&&execution.previewPrecision==='FP32')notes.push('Inspection and update previews use eager FP32.');
    if(execution.evaluationBackend==='eager'&&execution.evaluationPrecision==='FP32')notes.push('Evaluation uses eager FP32.');
    if(execution.generationPrecision==='FP32')notes.push('Text generation uses FP32.');
    if(compiledTransferReady(execution)&&['FP32','FP16 AMP'].includes(execution.transferPrecision)){
      notes.push(`Transfer forward passes use compiled ${execution.transferPrecision}.`);
      if(execution.transferDynamicLength===true)notes.push('The compiled transfer forward supports variable review lengths; the comparison branches retain separate weights and Adam states.');
      if(finite(execution.transferWarmupSeconds)&&execution.transferWarmupSeconds>=0)notes.push(`Transfer compiler warmup: ${number(execution.transferWarmupSeconds,1)} s; no optimizer updates are added.`);
    }else if(execution.transferBackend==='eager'&&['FP32','FP16 AMP'].includes(execution.transferPrecision))notes.push(`Transfer uses eager ${execution.transferPrecision}.`);
    if(Number.isSafeInteger(execution.evaluationIntervalSteps)&&execution.evaluationIntervalSteps>0)notes.push(`Normal training evaluates metrics every ${execution.evaluationIntervalSteps} optimizer steps; progress is checked every 0.1 s.`);
    if(execution.lossScaling==='dynamic')notes.push('Dynamic loss scaling is enabled.');
    if(execution.optimizerStepAccounting==='successful-updates-only')notes.push('Only successful optimizer updates count toward the requested steps; overflow attempts add no steps or loss-curve points.');
    if(compiledLmReady(execution))notes.push(`Compiler warmup finishes before training requests are accepted${finite(execution.warmupSeconds)&&execution.warmupSeconds>=0?` (${number(execution.warmupSeconds,1)} s)`:''}; it adds no optimizer steps or loss-curve points.`);
    if(notes.length)modelBlock.append(make('p',null,notes.join(' ')));
  }
  dataBlock.append(make('h3',null,'What these experiments can show'),make('p',null,'This is a scaled demonstration of GPT-1’s method. TinyStories is synthetic public text; UCI sentiment uses labeled human reviews. Neither dataset reproduces GPT-1’s original benchmark setting.'),make('p',null,'Live controls update the current model. Recorded views display actual saved observations. Validation uses the documented held-out subset. A short run may show noisy or weak transfer; pretraining is not guaranteed to win.'));
  $('model-details-content').replaceChildren(modelBlock,dataBlock);
}
async function refreshExecutionMetadata(){
  let warning='';const previous=lab.meta;
  try{
    const metadata=await api('meta',undefined,{optional:true});
    if(!metadata||typeof metadata!=='object'||Array.isArray(metadata))throw new Error('The runtime returned invalid execution metadata.');
    lab.meta=metadata;renderMetadata();
  }
  catch(error){
    // A successful model operation must never be replayed because this GET fails.
    // Drop cached execution claims until a later explicit read confirms them.
    lab.meta={...(previous||{}),execution:null};renderMetadata();
    warning=` Execution details could not be refreshed: ${error.message} Use Refresh live state to retry this read.`;
  }
  renderConvergence();return warning;
}

// TOKEN_HISTORY_CONTROLLER_START
// Saved observations and playback are independent of model ownership and training.
export function readTokenHistory(value){
  const invalid=()=>{throw new Error('The runtime returned an invalid token recording. Existing observations are retained.');};
  const integer=(n,min=0)=>Number.isSafeInteger(n)&&n>=min;
  const probability=n=>typeof n==='number'&&Number.isFinite(n)&&n>=0&&n<=1;
  const token=t=>t&&integer(t.id)&&typeof t.piece==='string'&&t.piece.length<=8192;
  if(!value||value.version!==1||typeof value.available!=='boolean'||typeof value.active!=='boolean'||!Array.isArray(value.frames))invalid();
  if(!value.available){if(value.active||value.frames.length)invalid();return value;}
  if(typeof value.runId!=='string'||!value.runId||typeof value.text!=='string'||!value.text.trim()||value.text.length>8192||!/^([a-f0-9]{64})$/.test(value.textSha256||'')||!integer(value.seed)||typeof value.source!=='string'||!integer(value.startStep)||value.intervalSteps!==100||value.capacity!==256||!integer(value.droppedFrames)||value.context!==128||!integer(value.totalTargets,1)||typeof value.truncated!=='boolean')invalid();
  const n=value.targets?.length;
  if(!Array.isArray(value.tokens)||!Array.isArray(value.targets)||!integer(n,1)||n>128||value.tokens.length!==n||!value.tokens.every(token)||!value.targets.every(token)||value.totalTargets<n||value.truncated!==(value.totalTargets>n)||value.frames.length<1||value.frames.length>256)invalid();
  let previous=-1;
  for(const frame of value.frames){
    if(!frame||!integer(frame.step)||frame.step<=previous||frame.step<value.startStep||!Array.isArray(frame.targetProbabilities)||frame.targetProbabilities.length!==n||!frame.targetProbabilities.every(probability)||!Array.isArray(frame.topPredictions)||frame.topPredictions.length!==n||!frame.topPredictions.every(item=>token(item)&&probability(item.p)))invalid();
    previous=frame.step;
  }
  if(value.frames[0].step!==value.startStep)invalid();
  return value;
}

export function createTokenHistory({request,onChange=()=>{},now=()=>Date.now(),setTimer=(fn,ms)=>setTimeout(fn,ms),clearTimer=id=>clearTimeout(id)}){
  const state={snapshot:null,frameIndex:0,position:0,playing:false,speed:1,followLatest:true,pending:false,fetching:false,connected:false,error:null,uncertain:false,locked:true,visible:false};
  let pollTimer=null,playTimer=null,inflight=null,lastRead=-Infinity,disposed=false;
  const emit=reason=>{if(!disposed)onChange(state,reason);};
  const clearPoll=()=>{if(pollTimer!==null)clearTimer(pollTimer);pollTimer=null;};
  const pause=()=>{if(playTimer!==null)clearTimer(playTimer);playTimer=null;state.playing=false;emit('playback');};
  function schedule(){
    clearPoll();if(disposed||!state.visible||state.pending||inflight)return;
    pollTimer=setTimer(()=>{pollTimer=null;void refresh();},Math.max(0,1000-(now()-lastRead)));
  }
  function accept(raw){
    const next=readTokenHistory(raw),old=state.snapshot,same=next.available&&old?.available&&next.runId===old.runId,selectedStep=old?.frames[state.frameIndex]?.step;
    if(!same){pause();state.position=0;state.followLatest=true;}
    state.snapshot=next;
    if(!next.available)state.frameIndex=0;
    else if(state.followLatest&&!state.playing)state.frameIndex=next.frames.length-1;
    else{const exact=next.frames.findIndex(frame=>frame.step===selectedStep);state.frameIndex=exact>=0?exact:Math.min(state.frameIndex,next.frames.length-1);}
    state.connected=true;state.error=null;state.uncertain=false;emit('snapshot');
  }
  async function refresh(){
    if(disposed||!state.visible||state.pending)return false;
    if(inflight)return inflight;
    if(now()-lastRead<1000){schedule();return false;}
    clearPoll();lastRead=now();state.fetching=true;emit('connection');
    const promise=(async()=>{
      try{accept(await request('token-history'));return true;}
      catch(error){state.connected=false;state.error=error.message;emit('error');return false;}
      finally{state.fetching=false;inflight=null;schedule();emit('connection');}
    })();
    inflight=promise;return promise;
  }
  async function mutate(path,body){
    if(disposed||state.pending||state.locked||state.uncertain)return false;
    state.pending=true;state.error=null;clearPoll();pause();emit('mutation');
    try{if(inflight)await inflight;accept(await request(path,body));return true;}
    catch(error){state.error=`${error.message} Checking the saved recording; the action is not retried.`;state.uncertain=true;emit('error');return false;}
    finally{state.pending=false;schedule();emit('mutation');}
  }
  function tick(){
    if(playTimer!==null)clearTimer(playTimer);playTimer=null;
    if(!state.playing||disposed)return;
    playTimer=setTimer(()=>{
      playTimer=null;const frames=state.snapshot?.frames||[];
      if(state.frameIndex>=frames.length-1){pause();return;}
      state.frameIndex++;emit('frame');
      if(state.frameIndex>=frames.length-1)pause();else tick();
    },1000/state.speed);
  }
  const api={state,refresh,
    setLocked(value){const locked=Boolean(value);if(locked===state.locked)return;state.locked=locked;emit('controls');},
    setVisible(value){const visible=Boolean(value);if(visible===state.visible)return;state.visible=visible;if(visible)schedule();else{clearPoll();pause();}},
    start(text){if(typeof text!=='string'||!text.trim()||text.length>8192)return Promise.resolve(false);return mutate('token-history-start',{text});},
    stop(){return state.snapshot?.active?mutate('token-history-stop',{}):Promise.resolve(false);},
    play(){if(disposed||!state.visible||state.pending||(state.snapshot?.frames.length||0)<2)return;state.followLatest=false;if(state.frameIndex>=state.snapshot.frames.length-1)state.frameIndex=0;state.playing=true;emit('frame');tick();},
    pause,
    seek(index){if(!Number.isInteger(index)||index<0||index>=(state.snapshot?.frames.length||0))return;pause();state.followLatest=false;state.frameIndex=index;emit('frame');},
    select(position){if(!Number.isInteger(position)||position<0||position>=(state.snapshot?.targets.length||0))return;state.position=position;emit('position');},
    setSpeed(speed){if(![.5,1,2,4].includes(speed))return;state.speed=speed;if(state.playing)tick();},
    follow(value){pause();state.followLatest=Boolean(value);if(state.followLatest)state.frameIndex=Math.max(0,(state.snapshot?.frames.length||0)-1);emit('frame');},
    dispose(){disposed=true;clearPoll();pause();}
  };
  return api;
}
// TOKEN_HISTORY_CONTROLLER_END

function createTokenHistoryView(){
  const panel=$('token-history-panel');if(!panel)return null;
  let controller=null,drawn='',tileRun=null,tileButtons=[];
  const probabilityLabel=value=>value>0&&value<.00001?'<0.001%':pct(value);
  const visible=()=>lab.tab==='data'&&!document.hidden&&!(document.body.classList.contains('seminar-mode')&&document.body.dataset.demoStage==='discussion');
  function render(state,reason){
    const saved=state.snapshot,available=Boolean(saved?.available),frames=saved?.frames||[],frame=frames[state.frameIndex];
    $('token-history-start').disabled=state.locked||state.pending||state.uncertain||!$('inspect-text').value.trim();
    $('token-history-start').textContent=available?'Replace recording with this text':'Record this text during training';
    $('token-history-stop').disabled=state.locked||state.pending||state.uncertain||!saved?.active;
    $('token-history-play').disabled=state.pending||frames.length<2||state.playing;
    $('token-history-pause').disabled=!state.playing;
    $('token-history-slider').disabled=frames.length<2||state.pending;
    $('token-history-follow').disabled=!available;
    $('token-history-follow').checked=state.followLatest;
    $('token-history-badge').textContent=saved?.active?'RECORDING CURRENT LM':available?'SAVED OBSERVATIONS':'NO RECORDING';
    $('token-history-badge').className=`badge ${saved?.active?'online':'recorded'}`;
    const status=state.pending?'Updating the recording on the server…':state.error?state.error:!saved?'Checking for a current-session recording…':!available?'Start recording, then train in Convergence. The initial observation is captured now; later frames arrive at measured LM steps.':`${frames.length} measured frame${frames.length===1?'':'s'} · ${saved.active?'recording continues on the server':'recording stopped'}${state.playing?' · replaying saved frames':''}. ${state.locked&&saved.active?'Recording controls unlock when the model is idle. ':''}${frames.length===1?'Only the initial observation is available. Train the language model to record changes.':'Playback changes the view only; it never trains or generates text.'}`;
    $('token-history-status').textContent=status;$('token-history-status').classList.toggle('error',Boolean(state.error));
    $('token-history-content').hidden=!available;
    if(!available){drawn='';tileRun=null;tileButtons=[];return;}
    if(['controls','connection','mutation','error','playback'].includes(reason))return;
    const signature=JSON.stringify([saved,state.frameIndex,state.position]);if(signature===drawn)return;drawn=signature;
    $('token-history-source').textContent=saved.text;
    $('token-history-provenance').textContent=`Current-session recording · started at LM step ${saved.startStep} · seed ${saved.seed} · source ${saved.source} · run ${saved.runId}`;
    $('token-history-extent').textContent=saved.truncated?`Showing all ${saved.targets.length} recorded target positions, from ${saved.totalTargets} in the text. The model's 128-position context truncates the rest; EOS is included only if it fits.`:`All ${saved.targets.length} target positions are recorded, including EOS when present. Each tile shows P(observed next token | its true prefix).`;
    $('token-history-retention').textContent=`Initial frame + observations at ${saved.intervalSteps}-step LM evaluation checkpoints, committed single updates, and stop. Up to ${saved.capacity} frames are retained.${saved.droppedFrames?` ${saved.droppedFrames} older intermediate frames have been dropped; the initial frame is retained.`:''} Reload reads the server's in-memory recording; Save live also saves it with the model.`;
    const slider=$('token-history-slider');slider.max=frames.length-1;slider.value=state.frameIndex;
    $('token-history-frame').textContent=`LM step ${frame.step} · frame ${state.frameIndex+1} / ${frames.length}`;
    $('token-history-step').textContent=frame.step;
    if(tileRun!==saved.runId){
      tileRun=saved.runId;tileButtons=saved.targets.map((target,index)=>{
        const button=make('button','token-probability-tile');button.type='button';button.append(make('small',null,`Target ${index+1}`),make('strong',null,piece(target.piece)),make('span','token-probability-value'));
        button.addEventListener('click',()=>controller.select(index));return button;
      });$('token-history-tiles').replaceChildren(...tileButtons);
    }
    tileButtons.forEach((button,index)=>{
      const p=frame.targetProbabilities[index],selected=index===state.position;button.classList.toggle('selected',selected);button.setAttribute('aria-pressed',String(selected));button.setAttribute('aria-label',`Target ${index+1}, ${piece(saved.targets[index].piece)}: ${probabilityLabel(p)} at LM step ${frame.step}. Select its trajectory.`);button.style.setProperty('--token-probability',String(p));button.lastChild.textContent=probabilityLabel(p);button.title=`Input ${index}: ${piece(saved.tokens[index].piece)} → observed next token ${piece(saved.targets[index].piece)} · ID ${saved.targets[index].id} · probability ${precise(p)}`;
    });
    const target=saved.targets[state.position],top=frame.topPredictions[state.position],p=frame.targetProbabilities[state.position],initial=frames[0].targetProbabilities[state.position],difference=100*(p-initial);
    $('token-history-selected').textContent=`Target ${state.position+1}: ${piece(target.piece)}`;
    $('token-history-target-p').textContent=probabilityLabel(p);
    $('token-history-delta').textContent=`${difference>=0?'+':''}${difference.toFixed(2)} percentage points from initial step ${frames[0].step}`;
    $('token-history-best').textContent=piece(top.piece);$('token-history-best-p').textContent=`${probabilityLabel(top.p)} · ID ${top.id}${top.id===target.id?' · matches observed target':''}`;
    $('token-history-prefix').textContent=`True prefix: ${saved.tokens.slice(0,state.position+1).map(token=>piece(token.piece)).join(' ')}`;
    drawChart('token-history-chart',[{label:`Observed ${piece(target.piece)}`,color:'#00549f',data:frames.map(item=>({step:item.step,value:item.targetProbabilities[state.position]}))}],{accuracy:true,selected:frame.step,windowed:true});
    $('token-history-chart').setAttribute('aria-label',`Measured probability of target ${state.position+1}, ${piece(target.piece)}, over ${frames.length} saved LM steps. Selected step ${frame.step}: ${probabilityLabel(p)}.`);
  }
  controller=createTokenHistory({request:(path,body)=>api(path,body,{optional:true}),onChange:render});
  async function mutate(kind){
    if(lab.busy||jobLocked()||!lab.ready)return;
    if(kind==='start'&&controller.state.snapshot?.available&&!window.confirm('Replace this token recording with a new recording of the text currently in the inspection editor? The model and training data stay unchanged.'))return;
    const promise=kind==='start'?controller.start($('inspect-text').value):controller.stop();lab.busy=true;controls();
    try{await promise;}finally{lab.busy=false;controls();}
  }
  $('token-history-start').addEventListener('click',()=>void mutate('start'));
  $('token-history-stop').addEventListener('click',()=>void mutate('stop'));
  $('token-history-play').addEventListener('click',()=>controller.play());$('token-history-pause').addEventListener('click',()=>controller.pause());
  $('token-history-slider').addEventListener('input',()=>controller.seek(Number($('token-history-slider').value)));
  $('token-history-speed').addEventListener('change',()=>controller.setSpeed(Number($('token-history-speed').value)));
  $('token-history-follow').addEventListener('change',()=>controller.follow($('token-history-follow').checked));
  $('inspect-text').addEventListener('input',()=>render(controller.state,'controls'));
  const syncVisibility=()=>controller.setVisible(visible());
  document.addEventListener('visibilitychange',syncVisibility);window.addEventListener('hashchange',syncVisibility);window.addEventListener('popstate',syncVisibility);window.addEventListener('pagehide',()=>controller.setVisible(false));window.addEventListener('pageshow',syncVisibility);
  $('seminar-mode-toggle')?.addEventListener('click',()=>queueMicrotask(syncVisibility));
  render(controller.state,'initial');
  return{setLocked:value=>{controller.setLocked(value);syncVisibility();},syncVisibility};
}

// SHORT_EXAMPLES_START
// Teaching-only prompts. Their UI keys never identify training documents.
const SHORT_EXAMPLES=Object.freeze([
  Object.freeze({id:'teaching-girl-name',theme:'girl-name',label:'Girl name',text:'A girl is named Lily.',position:4}),
  Object.freeze({id:'teaching-boy-name',theme:'boy-name',label:'Boy name',text:'A boy is named Tim.',position:4}),
  Object.freeze({id:'teaching-animal',theme:'animal',label:'Animal',text:'The animal was a dog.',position:4}),
  Object.freeze({id:'teaching-food',theme:'food',label:'Food',text:'The cat drank milk.',position:3}),
  Object.freeze({id:'teaching-action',theme:'action',label:'Action',text:'The bird flew away.',position:2}),
  Object.freeze({id:'teaching-color',theme:'color',label:'Color',text:'The sky was blue',position:3})
]);
// Initial positions are presentation hints; the active tokenizer supplies the
// actual tokens and targets, and users can select any displayed input token.
function defaultInspectionPosition(text){return SHORT_EXAMPLES.find(example=>example.text===text)?.position??0;}
function trainingExampleFromResponse(value){
  if(!value||!Array.isArray(value.examples)||value.examples.some(example=>!example||typeof example.id!=='string'||!example.id.trim()||example.id.length>200||typeof example.text!=='string'||!example.text.trim()))throw new Error('The training service returned invalid training examples. Use Retry before previewing an update.');
  const example=value.examples[0];return example?{id:example.id,text:example.text}:null;
}
async function loadTrainingExample(){lab.trainingExample=null;lab.trainingExample=trainingExampleFromResponse(await api('examples'));}
// SHORT_EXAMPLES_END

function renderExamples(){
  const select=$('story-example');select.replaceChildren(...lab.examples.map(example=>{const option=make('option',null,`${example.label} · ${example.text}`);option.value=example.id;return option;}));
  $('short-example-provenance').replaceChildren(make('p','small-note',lab.trainingExample?'One update uses a separate example from your downloaded training split.':'No training example is available for One update.'));
  if(lab.examples.length)$('inspect-text').value=lab.examples[0].text;
}

async function inspect(text=$('inspect-text').value,position=defaultInspectionPosition(text)){
  if(!text.trim()){message('Choose or enter text to inspect.',true);return;}
  return action(async()=>{lab.inspection=await api('inspect',{text,position});lab.inspectionStep=lab.live?.step;renderInspection();message('This prediction uses the current live weights. Click another input position to change its target.');},'Computing real token probabilities and attention…');
}

async function refreshInspection(){
  if(!lab.inspection)return;
  lab.inspection=await api('inspect',{text:lab.inspection.text,position:lab.inspection.position});lab.inspectionStep=lab.live?.step;renderInspection();
}

function renderInspection(){
  const result=lab.inspection;if(!result)return;
  $('position-label').textContent=`Position ${result.position} · live weights at step ${lab.inspectionStep??'—'}`;
  const pairs=result.tokens.map((token,index)=>{const pair=make('div',`shift-pair${index===result.position?' selected':''}`),button=make('button',null,piece(token.piece));button.type='button';button.title=`Input ${index} · ID ${token.id}`;button.setAttribute('aria-label',`Inspect input position ${index}: ${token.piece}`);button.setAttribute('aria-pressed',String(index===result.position));button.addEventListener('click',()=>inspect(result.text,index));pair.append(make('small',null,String(index)),button,make('span','shift-arrow','↓'),make('span','target-chip',piece(result.targets[index]?.piece)));return pair;});
  $('shifted-tokens').replaceChildren(...pairs);$('target-piece').textContent=piece(result.target.piece);$('target-id').textContent=`Target ID ${result.target.id}`;
  $('data-target-p').textContent=`Observed target: ${pct(result.targetProbability)}`;probabilities('data-probabilities',result.probabilities,result.target.id);
  math('shift-formula',String.raw`\mathcal L_t=-\log p_\theta(x_{t+1}\mid x_{\leq t})=${number(result.tokenLoss)}`);
  renderAttention();controls();
}

// ATTENTION_OBJECTIVE_VIEW_START
function attentionRowView(result){
  const position=result?.position,tokens=result?.tokens,row=result?.attention?.[position],mask=result?.mask?.[position];
  if(!Number.isSafeInteger(position)||position<0||!Array.isArray(tokens)||position>=tokens.length||!Array.isArray(row)||!Array.isArray(mask)||row.length!==tokens.length||mask.length!==tokens.length)return null;
  if(mask.some(value=>typeof value!=='boolean'))return null;
  const cells=tokens.map((token,index)=>({index,piece:token.piece,blocked:index>position||mask[index],weight:row[index]}));
  if(cells.some(cell=>!cell.blocked&&(!Number.isFinite(cell.weight)||cell.weight<0||cell.weight>1)))return null;
  return cells;
}
function transferObjectiveView(objective,measuredStep,expectedWeight){
  if(!objective||objective.schemaVersion!==1||objective.split!=='train'||objective.measurement!=='fixed-subset-eval'||objective.auxiliaryWeight!==expectedWeight||!Number.isSafeInteger(objective.step)||objective.step<0||objective.step!==measuredStep||!Number.isSafeInteger(objective.sampleCount)||objective.sampleCount<1)return null;
  const {classificationLoss,lmLoss,auxiliaryWeight,auxiliaryContribution,totalLoss}=objective;
  const valid=value=>typeof value==='number'&&Number.isFinite(value)&&value>=0;
  if(![classificationLoss,auxiliaryContribution,totalLoss].every(valid)||![0,.5].includes(auxiliaryWeight))return null;
  if(auxiliaryWeight===0?(lmLoss!==null||auxiliaryContribution!==0):!valid(lmLoss))return null;
  const close=(a,b)=>Math.abs(a-b)<=1e-6*Math.max(1,Math.abs(a),Math.abs(b));
  if(!close(auxiliaryContribution,auxiliaryWeight===0?0:auxiliaryWeight*lmLoss)||!close(totalLoss,classificationLoss+auxiliaryContribution))return null;
  return {step:objective.step,classificationLoss,lmLoss,auxiliaryWeight,auxiliaryContribution,totalLoss,sampleCount:objective.sampleCount};
}
// ATTENTION_OBJECTIVE_VIEW_END

// QKV_INSPECTION_VIEW_START
function qkvInspectionView(result){
  const cells=attentionRowView(result),details=result?.attentionDetails;
  if(!cells||!details||details.schemaVersion!==1||details.layer!==1||details.head!==1||details.headDimension!==32||details.queryPosition!==result.position)return null;
  const width=details.headDimension,count=cells.length,vector=value=>Array.isArray(value)&&value.length===width&&value.every(Number.isFinite),close=(a,b)=>Math.abs(a-b)<=1e-5+1e-4*Math.max(Math.abs(a),Math.abs(b));
  if(!vector(details.query)||!vector(details.context)||!Number.isFinite(details.scale)||!close(details.scale,Math.sqrt(width)))return null;
  if(!Array.isArray(details.keys)||details.keys.length!==count||!details.keys.every(vector)||!Array.isArray(details.values)||details.values.length!==count||!details.values.every(vector)||!Array.isArray(details.scores)||details.scores.length!==count||!Array.isArray(details.weights)||details.weights.length!==count)return null;
  const dots=details.keys.map(key=>key.reduce((sum,value,index)=>sum+details.query[index]*value,0));
  for(const cell of cells){
    const weight=details.weights[cell.index],score=details.scores[cell.index];
    if(!Number.isFinite(weight)||weight<0||weight>1||!close(weight,cell.weight))return null;
    if(cell.blocked?(score!==null||weight!==0):(!Number.isFinite(score)||!close(score,dots[cell.index]/details.scale)))return null;
  }
  for(let dimension=0;dimension<width;dimension++){const mixed=details.values.reduce((sum,value,index)=>sum+details.weights[index]*value[dimension],0);if(!close(mixed,details.context[dimension]))return null;}
  return {...details,cells,dots};
}
function qkvKeyPosition(view,requested){return Number.isInteger(requested)&&view.cells[requested]&&!view.cells[requested].blocked?requested:view.cells.find(cell=>cell.index===view.queryPosition&&!cell.blocked)?.index??view.cells.find(cell=>!cell.blocked)?.index??null;}
// QKV_INSPECTION_VIEW_END
let selectedQkvKey=null;
function qkvVector(label,values){
  const box=make('section','qkv-vector'),strip=make('div','qkv-vector-strip'),maximum=Math.max(...values.map(Math.abs),1e-12);box.append(make('h5',null,label));
  strip.setAttribute('role','img');strip.setAttribute('aria-label',`${label}: ${values.length} signed vector components; raw values below.`);
  values.forEach((value,index)=>{const cell=make('span');cell.title=`Dimension ${index}: ${value}`;cell.style.backgroundColor=value<0?`rgba(179,113,25,${.08+.8*Math.abs(value)/maximum})`:`rgba(0,84,159,${.08+.8*Math.abs(value)/maximum})`;strip.append(cell);});box.append(strip);return box;
}
function renderQkvInspection(){
  const box=$('qkv-content'),view=qkvInspectionView(lab.inspection);box.replaceChildren();
  if(!view){box.append(make('p','small-note','Q, K, V details are not available for this inspection.'));return;}
  selectedQkvKey=qkvKeyPosition(view,selectedQkvKey);if(selectedQkvKey===null){box.append(make('p','small-note','No visible context key is available.'));return;}
  const chosen=selectedQkvKey,weight=view.weights[chosen],weighted=view.values[chosen].map(value=>weight*value),label=index=>`${index} · ${piece(lab.inspection.tokens[index].piece)}`;
  box.append(make('p','qkv-description','Q asks what to look for at the current position. K describes each context token for matching. V is the content vector mixed into the output.'));
  const field=make('label','field qkv-key-field','Context token for K and V'),select=make('select');select.setAttribute('aria-label','Context token for K and V');
  for(const cell of view.cells){const option=make('option',null,`${label(cell.index)}${cell.blocked?' · masked':''}`);option.value=String(cell.index);option.disabled=cell.blocked;select.append(option);}select.value=String(chosen);select.addEventListener('change',()=>{selectedQkvKey=Number(select.value);renderQkvInspection();});field.append(select);box.append(field);
  const vectors=make('div','qkv-vectors');vectors.append(qkvVector(`Q · query ${label(view.queryPosition)}`,view.query),qkvVector(`K · key ${label(chosen)}`,view.keys[chosen]),qkvVector(`V · value ${label(chosen)}`,view.values[chosen]));box.append(vectors);
  const chain=make('div','qkv-chain');for(const [title,value]of[['Q · K',number(view.dots[chosen],5)],[`÷ √${view.headDimension}`,number(view.scores[chosen],5)],['Softmax weight',pct(weight)]]){const step=make('div');step.append(make('span',null,title),make('strong',null,value));chain.append(step);}box.append(chain,make('p','qkv-description','Weight = exp(score) / sum of exp(all visible scores). Future positions are masked.'));
  const scoreBox=make('div','qkv-scores'),table=make('table'),header=make('tr');header.append(make('th',null,'Visible token'),make('th',null,'Scaled score'),make('th',null,'Weight'));const head=make('thead');head.append(header);const body=make('tbody');for(const cell of view.cells){if(cell.blocked)continue;const row=make('tr',cell.index===chosen?'selected':'');row.append(make('td',null,label(cell.index)),make('td',null,number(view.scores[cell.index],5)),make('td',null,pct(view.weights[cell.index])));body.append(row);}table.append(head,body);scoreBox.append(table);box.append(scoreBox);
  const outputs=make('div','qkv-outputs');outputs.append(qkvVector('Selected weight × V',weighted),qkvVector('This head’s output · Σ weight × V',view.context));box.append(outputs);
  const raw=make('details','qkv-raw');raw.append(make('summary',null,`All ${view.headDimension} values · Q, K, V and outputs`),make('p','small-note','Vector entries are signed numbers, not percentages. Blue is positive; amber is negative. Each strip scales its own colors. Raw components below are unscaled; displayed scores are rounded.'));
  for(const [title,values]of[['Q',view.query],['K',view.keys[chosen]],['V',view.values[chosen]],['Selected weight × V',weighted],['This head’s output · Σ weight × V',view.context]]){const section=make('section'),grid=make('div','qkv-raw-values');section.append(make('h5',null,title));values.forEach((value,index)=>{const item=make('span');item.append(make('b',null,String(index)),make('code',null,String(value)));grid.append(item);});section.append(grid);raw.append(section);}box.append(raw);
}

function renderAttentionRow(){
  renderQkvInspection();
  const panel=$('attention-row-panel'),row=$('attention-row'),cells=attentionRowView(lab.inspection);panel.hidden=false;
  if(!cells){empty('attention-row','Attention unavailable for this inspection.');return;}
  row.replaceChildren(...cells.map(item=>{const cell=make('div',`attention-token${item.blocked?' masked':''}`);cell.setAttribute('role','listitem');cell.append(make('span',null,piece(item.piece)),make('small',null,item.blocked?'masked':pct(item.weight)));cell.title=`Input ${item.index} · ${item.blocked?'masked':`attention ${item.weight}`}`;if(!item.blocked)cell.style.backgroundColor=`rgba(0,84,159,${.05+.7*item.weight})`;return cell;}));
}
function renderTransferObjective(run,measuredStep,expectedWeight){
  const box=make('div','transfer-objective'),value=transferObjectiveView(run.metrics?.objective,measuredStep,expectedWeight);
  box.append(make('h4',null,'Training objective'));
  if(!value){box.append(make('p','objective-unavailable','Not measured yet'));return box;}
  const bar=make('div','objective-bar');bar.setAttribute('role','img');bar.setAttribute('aria-label',`Classification ${number(value.classificationLoss)} plus auxiliary LM ${number(value.auxiliaryContribution)} equals total ${number(value.totalLoss)}; measured step ${value.step}`);
  for(const [cls,loss]of[['objective-class',value.classificationLoss],['objective-auxiliary',value.auxiliaryContribution]]){const segment=make('span',cls);segment.style.width=`${value.totalLoss>0?100*loss/value.totalLoss:0}%`;bar.append(segment);}
  const values=make('div','objective-values');for(const [label,loss,cls]of[['Class',value.classificationLoss,'objective-class-label'],['λ · LM',value.auxiliaryContribution,'objective-auxiliary-label'],['Total',value.totalLoss,'']]){const item=make('div',cls);item.append(make('span',null,label),make('strong',null,number(loss)));values.append(item);}
  box.append(bar,values,make('p','objective-step',`Measured step ${value.step}${value.auxiliaryWeight===0?' · Auxiliary off':''}`));
  const details=make('details','objective-details');details.append(make('summary',null,'Loss measurement details'),make('p',null,`Raw LM loss: ${value.lmLoss===null?'Not computed (λ = 0)':number(value.lmLoss)} · λ = ${value.auxiliaryWeight}`),make('p',null,`Fixed training subset · ${value.sampleCount} examples · evaluated at step ${value.step}. This is an evaluation of the training objective, not the current minibatch or validation loss.`));box.append(details);return box;
}

function renderAttention(){
  renderAttentionRow();
  const result=lab.inspection,attention=result?.attention,mask=result?.mask;
  if(!Array.isArray(attention)||!attention.length||!Array.isArray(mask)){empty('attention-map','This response did not include an attention matrix.');return;}
  const n=Math.min(attention.length,result.tokens.length),size=Math.min(n,18),start=Math.max(0,Math.min(result.position-Math.floor(size/2),n-size)),end=start+size;
  const grid=make('div','attention-grid');grid.style.gridTemplateColumns=`24px repeat(${size},19px)`;
  grid.append(make('span','attention-axis','q / k'));for(let col=start;col<end;col++)grid.append(make('span','attention-axis',String(col)));
  let max=0;for(let row=start;row<end;row++)for(let col=start;col<end;col++)if(!mask[row]?.[col]&&finite(attention[row]?.[col]))max=Math.max(max,attention[row][col]);
  for(let row=start;row<end;row++){grid.append(make('span','attention-axis',String(row)));for(let col=start;col<end;col++){const blocked=mask[row]?.[col]===true,value=attention[row]?.[col],cell=make('span',`attention-cell${blocked?' blocked':''}`,blocked?'×':'');if(!blocked)cell.style.background=`rgba(0,84,159,${.08+.85*(finite(value)&&max?value/max:0)})`;cell.title=`Query ${row} → key ${col}: ${blocked?'masked future token':`attention ${number(value,5)}`}`;grid.append(cell);}}
  $('attention-range').textContent=`Positions ${start}–${end-1} · shade relative to this window · hover for exact weights`;$('attention-map').replaceChildren(grid);
}

function traceMetric(label,value){const item=make('div');item.append(make('span',null,label),make('strong',null,value));return item;}
// FORWARD_START_VIEW_START
function forwardStartView(before,architecture){
  const tokens=before?.tokens,position=before?.position,target=before?.target;
  const validToken=token=>token&&Number.isSafeInteger(token.id)&&token.id>=0&&typeof token.piece==='string';
  if(!Array.isArray(tokens)||!tokens.length||!tokens.every(validToken)||!Number.isSafeInteger(position)||position<0||position>=tokens.length||!validToken(target))return null;
  const indexed=tokens.map((token,index)=>({...token,index})),layers=architecture?.layers;
  return {position,target,visible:indexed.slice(0,position+1),future:indexed.slice(position+1),futurePreview:indexed.slice(position+1,position+5),tokens:indexed,transformerLabel:Number.isSafeInteger(layers)&&layers>0?`Transformer × ${layers}`:'Transformer'};
}
// FORWARD_START_VIEW_END
function renderForwardStart(before,architecture){
  const box=make('section','forward-start'),view=forwardStartView(before,architecture);
  if(!view){box.append(make('p','empty-message','Visible prefix unavailable in this preview.'));return box;}
  const header=make('div','forward-start-heading'),target=make('div','forward-next-target');header.append(make('h4',null,`Visible at this position · positions 0–${view.position}`));target.append(make('span',null,'Next token target'),make('strong',null,piece(view.target.piece)));target.title=`Target ID ${view.target.id}`;header.append(target);box.append(header);
  const visible=make('div','forward-token-row');visible.setAttribute('aria-label',`Visible input prefix through position ${view.position}`);
  for(const token of view.visible){const current=token.index===view.position,item=make('div',`forward-token${current?' current':''}`);item.append(make('small',null,`${token.index}${current?' · current':''}`),make('span',null,piece(token.piece)));item.title=`Input ${token.index} · ID ${token.id}${current?' · current query':''}`;visible.append(item);}box.append(visible);
  if(view.futurePreview.length){const future=make('div','forward-future');future.append(make('span','forward-future-label','Later positions · masked for this query'));const row=make('div','forward-token-row');for(const token of view.futurePreview){const item=make('div','forward-token masked');item.append(make('small',null,String(token.index)),make('span',null,piece(token.piece)),make('small',null,'masked'));item.title=`Input ${token.index} · ID ${token.id} · masked for query ${view.position}`;row.append(item);}if(view.future.length>view.futurePreview.length)row.append(make('span','forward-more',`+${view.future.length-view.futurePreview.length} more`));future.append(row);box.append(future);}
  const flow=make('ol','forward-flow');flow.setAttribute('aria-label','Forward pass overview');for(const label of ['Token IDs','Token + position embedding',view.transformerLabel,'LM head','Softmax'])flow.append(make('li',null,label));box.append(flow);
  const details=make('details','forward-input-details');details.append(make('summary',null,'Full input tokens & IDs'),make('p',null,`Visibility is specific to query ${view.position}. Later positions can be present in the input window, but this query cannot attend to them. The flow above is a process overview, not measured intermediate activations.`));const all=make('div','forward-full-tokens');for(const token of view.tokens){const item=make('span',token.index>view.position?'masked':'',`${token.index} · ${piece(token.piece)} · ID ${token.id}`);all.append(item);}details.append(all);box.append(details,make('h4','forward-probability-label','Next-token probabilities'));return box;
}

// LOSS_MEAN_VIEW_START
function lossMeanView(trace){
  const targets=trace?.before?.targets,validTargets=Array.isArray(targets)&&targets.length>0&&targets.every(token=>token&&Number.isSafeInteger(token.id)&&token.id>=0);
  const counted=validTargets?targets.filter(token=>token.id!==0).length:0,count=counted>0?counted:null;
  let formula=String.raw`\begin{aligned}\mathcal L_{\mathrm{LM}}&=-\frac1T\sum_{t=1}^{T}\log p_\theta(x_{t+1}\mid x_{\le t})\\[10pt]\ell_i&=-\ln p_i\\[5pt]\mathcal L_{\mathrm{LM}}&=\frac{\ell_1+\ell_2+\cdots+\ell_T}{T}`;
  if(count!==null){const terms=count===1?String.raw`\ell_1`:count===2?String.raw`\ell_1+\ell_2`:String.raw`\ell_1+\ell_2+\cdots+\ell_{${count}}`;formula+=String.raw`\\[5pt]\mathcal L_{\mathrm{LM}}&=\frac{${terms}}{${count}}${Number.isFinite(trace?.loss)?String.raw`\approx ${trace.loss.toFixed(3)}`:''}`;}
  return {count,formula:formula+String.raw`\end{aligned}`};
}
// LOSS_MEAN_VIEW_END

// SELECTED_LOSS_EXAMPLE_START
function latexTokenText(value){
  const escaped={92:String.raw`\textbackslash{}`,123:String.raw`\{`,125:String.raw`\}`,36:String.raw`\$`,38:String.raw`\&`,35:String.raw`\#`,37:String.raw`\%`,95:String.raw`\_`,94:String.raw`\textasciicircum{}`,126:String.raw`\textasciitilde{}`};
  return Array.from(String(value),char=>escaped[char.codePointAt(0)]??char).join('');
}
function selectedLossExampleView(before){
  const input=forwardStartView(before),probability=before?.targetProbability,loss=before?.tokenLoss;
  if(!input||!Number.isFinite(probability)||probability<=0||probability>1||!Number.isFinite(loss)||loss<0)return null;
  const target=piece(input.target.piece),prefix=input.visible.map(token=>piece(token.piece)),fullPrefix=prefix.join(' '),short=prefix.length<=12&&fullPrefix.length<=120;
  const preview=short?fullPrefix:prefix.length>8?[...prefix.slice(0,6),'…',...prefix.slice(-2)].join(' '):fullPrefix.slice(0,110)+'…';
  const [mantissa,exponent]=probability.toExponential(8).split('e'),probabilityMath=probability<.001?String.raw`${mantissa}\times10^{${Number(exponent)}}`:Number(probability.toPrecision(9)).toString();
  const targetMath=target.length<=24?String.raw`\text{${latexTokenText(target)}}`:String.raw`y_{\mathrm{target}}`;
  const formula=String.raw`\begin{aligned}\ell_{\mathrm{selected}}&=-\ln p_\theta(${targetMath}\mid x_{\le ${input.position}})\\[4pt]&=-\ln\left(${probabilityMath}\right)\approx ${loss.toFixed(3)}\end{aligned}`;
  return {position:input.position,target,prefix:fullPrefix,preview,truncated:!short,formula};
}
// SELECTED_LOSS_EXAMPLE_END
function renderSelectedLossExample(before){
  const box=$('selected-token-example'),view=selectedLossExampleView(before);box.hidden=false;box.replaceChildren(make('h4',null,'Selected-token example'));
  if(!view){box.append(make('p','small-note','Selected-token example unavailable for logarithmic display.'));return;}
  box.append(make('p','selected-example-target',`Target: ${view.target}`),make('p','selected-example-context',`${view.truncated?'Context preview':'Visible context'} · positions 0–${view.position}: ${view.preview}`));
  if(view.truncated){const full=make('details','selected-example-full');full.append(make('summary',null,'Full visible prefix'),make('p',null,view.prefix));box.append(full);}
  const formula=make('div','formula selected-example-formula');formula.id='selected-token-example-formula';box.append(formula);math('selected-token-example-formula',view.formula);
}

// GRADIENT_CLIPPING_VIEW_START
function gradientClippingView(trace){
  const norm=trace?.gradientNorm,gradient=trace?.weight?.gradient,clippedGradient=trace?.weight?.clippedGradient;
  if(!Number.isFinite(norm)||norm<0||!Number.isFinite(gradient)||!Number.isFinite(clippedGradient))return null;
  const threshold=1,epsilon=1e-6,scale=Math.min(1,threshold/(norm+epsilon));
  return {norm,gradient,clippedGradient,threshold,epsilon,scale,formulaNorm:scale*norm,formulaGradient:scale*gradient};
}
function clippingMathNumber(value){
  if(!Number.isFinite(value))return String.raw`\text{unavailable}`;
  if(value===0)return '0';
  const [mantissa,exponent]=value.toExponential(8).split('e');
  return Math.abs(value)<.001||Math.abs(value)>=10000?String.raw`${mantissa}\times10^{${Number(exponent)}}`:Number(value.toPrecision(9)).toString();
}
function gradientClippingFormula(view){
  let formula=String.raw`\begin{aligned}g_j&=\frac{\partial\mathcal L_{\mathrm{LM}}}{\partial\theta_j},\qquad g=(g_1,g_2,\ldots)\\[7pt]g&=\nabla_\theta\mathcal L,\qquad \widetilde g=g\,\min\!\left(1,\frac{1}{\lVert g\rVert_2+10^{-6}}\right)\\[7pt]G&=\sqrt{\sum_jg_j^2},\qquad C=1\\[5pt]c&=\min\!\left(1,\frac{C}{G+10^{-6}}\right)`;
  if(view)formula+=String.raw`=\min\!\left(1,\frac{1}{${clippingMathNumber(view.norm)}+10^{-6}}\right)\approx ${clippingMathNumber(view.scale)}\\[5pt]&\underbrace{\lVert\widetilde g\rVert_2=cG}_{\text{All parameters: one global size}}\approx ${clippingMathNumber(view.formulaNorm)}`;
  formula+=String.raw`\\[7pt]j^\ast&=\operatorname*{arg\,max}_j |g_j|\\[5pt]&\underbrace{\widetilde g_{j^\ast}=c\,g_{j^\ast}}_{\text{One parameter: selected by max }|g_j|}`;
  if(view)formula+=String.raw`\approx\left(${clippingMathNumber(view.scale)}\right)\left(${clippingMathNumber(view.gradient)}\right)\approx ${clippingMathNumber(view.formulaGradient)}`;
  return formula+String.raw`\end{aligned}`;
}
// GRADIENT_CLIPPING_VIEW_END
function renderBackward(trace,box){
  const view=gradientClippingView(trace),metrics=make('div','trace-metrics backward-metrics');
  box.append(make('p','backward-note','Each parameter has its own partial derivative; together, these components form the gradient vector g.'));
  metrics.append(traceMetric('Global norm G',finite(trace?.gradientNorm)&&trace.gradientNorm>=0?number(trace.gradientNorm):'Unavailable'),traceMetric('Scale c · computed',view?precise(view.scale):'Unavailable'),traceMetric('Clipped norm · display bound',finite(trace?.clippedGradientNorm)&&trace.clippedGradientNorm>=0?`≈ ${number(trace.clippedGradientNorm)}`:'Unavailable'));box.append(metrics);
  if(view){
    const meaning=view.gradient>0?'Positive gradient: increasing this parameter slightly, with other weights fixed, locally increases the whole-window mean loss.':view.gradient<0?'Negative gradient: increasing this parameter slightly, with other weights fixed, locally decreases the whole-window mean loss.':'This coordinate has zero gradient: no first-order change in the whole-window mean loss at this point.';
    box.append(make('p','backward-meaning',meaning));
    const weight=trace.weight;box.append(make('p','weight-name backward-coordinate',`Maximum |gradient| coordinate (before clipping) · ${weight.name??'Name unavailable'} ${Number.isSafeInteger(weight.index)?`· flat index ${weight.index}`:Array.isArray(weight.index)?`· index ${JSON.stringify(weight.index)}`:'· index unavailable'}`));
    const selected=make('div','trace-metrics backward-gradients');selected.append(traceMetric('Selected raw gradient gⱼ*',precise(view.gradient)),traceMetric('After clipping g̃ⱼ* · computed',precise(view.clippedGradient)));box.append(selected);
    box.append(make('p','backward-note',`Components have different values but share the same scale c; signs are preserved, and Adam receives the full clipped vector. ${trace.applied?'This preview’s Adam update has already been applied.':'Weights are unchanged until Apply commits the Adam update.'}`));
  }else box.append(make('p','empty-message','Gradient details unavailable for this preview.'));
  math('update-formula',gradientClippingFormula(view));
}

// ADAM_UPDATE_VIEW_START
function adamUpdateView(weight){
  if(!weight)return null;
  const {mHat,vHat,learningRate,epsilon}=weight;
  if(![mHat,vHat,learningRate,epsilon].every(Number.isFinite)||vHat<0||learningRate<0||epsilon<=0)return null;
  const denominator=Math.sqrt(vHat)+epsilon,formulaDelta=-learningRate*mHat/denominator;
  return Number.isFinite(formulaDelta)?{mHat,vHat,learningRate,epsilon,denominator,formulaDelta}:null;
}
function adamAxisView(weight){
  const before=weight?.before,after=weight?.after;
  if(!Number.isFinite(before)||!Number.isFinite(after))return null;
  const direction=after>before?1:after<before?-1:0;
  return {before,after,direction,beforePosition:direction>0?.2:direction<0?.8:.5,afterPosition:direction>0?.8:direction<0?.2:.5};
}
function adamDisplay(value,scientific=false){return Number.isFinite(value)?value===0?'0':scientific?`${value>0?'+':''}${value.toExponential(7)}`:value.toPrecision(10):'Unavailable';}
function adamMathNumber(value,scientific=false){
  if(!Number.isFinite(value))return String.raw`\text{unavailable}`;
  const raw=value===0?'0':scientific?value.toExponential(7):Number(value.toPrecision(8)).toString();
  if(!raw.includes('e'))return raw;
  const [mantissa,exponent]=raw.split('e');return String.raw`${mantissa}\times10^{${Number(exponent)}}`;
}
function adamUpdateFormula(weight){
  const view=adamUpdateView(weight);let formula=String.raw`\begin{aligned}\theta_t&=\theta_{t-1}-\eta\,\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}`;
  if(view)formula+=String.raw`\\[7pt]\Delta\theta_{\mathrm{formula}}&=-\left(${adamMathNumber(view.learningRate)}\right)\frac{\left(${adamMathNumber(view.mHat)}\right)}{\sqrt{${adamMathNumber(view.vHat)}}+${adamMathNumber(view.epsilon)}}\\[5pt]&\approx ${adamMathNumber(view.formulaDelta,true)}`;
  if(Number.isFinite(weight?.delta))formula+=String.raw`\\[5pt]\Delta\theta_{\mathrm{actual}}&\approx ${adamMathNumber(weight.delta,true)}`;
  return formula+String.raw`\end{aligned}`;
}
function adamMomentFormula(weight){
  const suppliedBetas=weight?.betas,validBetas=Array.isArray(suppliedBetas)&&suppliedBetas.length===2&&suppliedBetas.every(value=>Number.isFinite(value)&&value>=0&&value<1),[beta1,beta2]=validBetas?suppliedBetas:[0.9,0.999];
  return String.raw`\begin{aligned}m_t&=\beta_1 m_{t-1}+(1-\beta_1)\widetilde g_t\\[6pt]v_t&=\beta_2 v_{t-1}+(1-\beta_2)\widetilde g_t^{\,2}\\[8pt]\widehat m_t&=\frac{m_t}{1-\beta_1^t},\quad\widehat v_t=\frac{v_t}{1-\beta_2^t}\\[6pt]\beta_1&=${adamMathNumber(beta1)},\quad\beta_2=${adamMathNumber(beta2)}\end{aligned}`;
}
// ADAM_UPDATE_VIEW_END
let adamFocus={previewId:null,step:0};
function renderAdamAxis(weight){
  const section=make('section','adam-axis'),axis=adamAxisView(weight);section.append(make('h4',null,'Zoomed parameter scale'));
  if(!axis){section.append(make('p','small-note','Parameter scale unavailable.'));return section;}
  const svg=svgNode('svg',{viewBox:'0 0 640 108',role:'img','aria-label':`Zoomed parameter scale: before ${axis.before}, after ${axis.after}. ${axis.direction>0?'Increase':axis.direction<0?'Decrease':'No representable change'}.`}),xBefore=40+560*axis.beforePosition,xAfter=40+560*axis.afterPosition;
  svg.append(svgNode('line',{x1:40,y1:43,x2:600,y2:43,stroke:'#c5d4e0','stroke-width':1.5}));
  const text=(x,y,value,color='#526a7c')=>{const node=svgNode('text',{x,y,'text-anchor':'middle',fill:color});node.textContent=value;return node;};
  if(axis.direction){
    svg.append(svgNode('line',{x1:xBefore,y1:43,x2:xAfter,y2:43,stroke:'#00549f','stroke-width':2.5}),svgNode('polyline',{points:`${xAfter-9*axis.direction},37 ${xAfter},43 ${xAfter-9*axis.direction},49`,fill:'none',stroke:'#00549f','stroke-width':2.5}),svgNode('circle',{cx:xBefore,cy:43,r:5,fill:'#8a96a0'}),svgNode('circle',{cx:xAfter,cy:43,r:5,fill:'#00549f'}),text(xBefore,24,'Before'),text(xAfter,24,'After','#00549f'),text(xBefore,72,adamDisplay(axis.before)),text(xAfter,72,adamDisplay(axis.after),'#00549f'));
  }else svg.append(svgNode('circle',{cx:xBefore,cy:43,r:6,fill:'#00549f'}),text(xBefore,23,'Before = after'),text(xBefore,72,adamDisplay(axis.before)),text(xBefore,96,'No representable parameter change'));
  section.append(svg);return section;
}
function renderAdam(trace,box){
  const weight=trace.weight||{},key=trace.previewId??trace;if(adamFocus.previewId!==key)adamFocus={previewId:key,step:0};
  const coordinate=Number.isSafeInteger(weight.index)?`flat index ${weight.index}`:Array.isArray(weight.index)?`index ${JSON.stringify(weight.index)}`:'index unavailable';
  box.append(make('p','weight-name adam-parameter',`${weight.name??'Parameter unavailable'} · ${coordinate} · largest |raw gradient|`));
  const flow=make('div','adam-flow'),buttons=[];flow.setAttribute('role','group');flow.setAttribute('aria-label','Adam calculation steps; selection highlights only');
  const values=[['Clipped gradient',[['g̃ⱼ*',adamDisplay(weight.clippedGradient)]]],['Adam memory',[['m̂ · signed trend',adamDisplay(weight.mHat)],['v̂ · mean squared gradient',Number.isFinite(weight.vHat)&&weight.vHat>=0?adamDisplay(weight.vHat):'Unavailable']]],['Actual change Δθ',[['after − before',adamDisplay(weight.delta,true)]]],['Weight before / after',[['Before',adamDisplay(weight.before)],['After',adamDisplay(weight.after)]]]];
  const paint=()=>buttons.forEach((button,index)=>{button.classList.toggle('active',index===adamFocus.step);button.setAttribute('aria-pressed',String(index===adamFocus.step));});
  values.forEach(([title,items],index)=>{const button=make('button','adam-step');button.type='button';button.append(make('span','adam-step-heading',`${index+1} · ${title}`));for(const [label,value]of items){const item=make('span','adam-step-value');item.append(make('small',null,label),make('strong',null,value));button.append(item);}button.addEventListener('click',()=>{adamFocus.step=index;paint();});buttons.push(button);flow.append(button);});paint();box.append(flow,renderAdamAxis(weight));
  const details=make('details','adam-details');details.append(make('summary',null,'Full Adam values & bias correction'));const table=make('table','trace-table');
  for(const [label,value]of [['Weight before',weight.before],['Raw gradient',weight.gradient],['Clipped gradient → Adam',weight.clippedGradient],['First moment m (uncorrected)',weight.m],['Second moment v (uncorrected)',weight.v],['Bias-corrected m̂',weight.mHat],['Bias-corrected v̂',weight.vHat],['Actual change Δθ',weight.delta],['Weight after',weight.after],['Learning rate η',weight.learningRate],['Adam step t',weight.adamStep],['Adam epsilon ε',weight.epsilon]]){const row=make('tr');row.append(make('td',null,label),make('td',null,Number.isFinite(value)?String(value):'Unavailable'));table.append(row);}
  const betaRow=make('tr'),betas=weight.betas,reportedBetas=Array.isArray(betas)&&betas.length===2&&betas.every(value=>Number.isFinite(value)&&value>=0&&value<1);betaRow.append(make('td',null,reportedBetas?'β₁, β₂ · preview values':'β₁, β₂ · demo settings'),make('td',null,reportedBetas?betas.join(', '):'0.9, 0.999'));table.append(betaRow);details.append(table,make('p','small-note','m̂ = m / (1 − β₁ᵗ), v̂ = v / (1 − β₂ᵗ). Bias correction uses optimizer step t; previous moment values are not supplied.'),make('p','small-note','Formula Δθ is derived from the displayed trace inputs. Actual Δθ comes from the previewed parameter update and can differ due to FP32 rounding. The axis uses the actual before and after weights.'));box.append(details);
  const update=make('section','adam-formula-block'),moments=make('section','adam-formula-block'),updateMath=make('div'),momentMath=make('div');updateMath.id='adam-update-equation';momentMath.id='adam-moment-equation';
  update.append(make('h4',null,'Parameter update'),updateMath);moments.append(make('h4',null,'Moments & bias correction'),make('p','adam-moment-labels','m: gradient trend · v: mean squared gradient'),momentMath);$('update-formula').replaceChildren(update,moments);math(updateMath.id,adamUpdateFormula(weight));math(momentMath.id,adamMomentFormula(weight));
}

function renderUpdate(){
  const trace=lab.trace,box=$('update-stage-content');
  $('update-formula').classList.toggle('adam-formula',Boolean(trace&&lab.stage===3));
  $('selected-token-example').hidden=true;$('selected-token-example').replaceChildren();
  for(const button of document.querySelectorAll('[data-stage]'))button.classList.toggle('active',Number(button.dataset.stage)===lab.stage);
  if(!trace){$('preview-badge').textContent='No preview';$('stage-title').textContent='Start with a real example.';$('advance-update').textContent='Next stage →';empty('update-stage-content','Choose Preview a real update to show its actual input prefix and forward pass.');$('update-formula').textContent='';controls();return;}
  const titles=['Forward','Loss','Backward','Adam update','Before / after'];
  $('stage-title').textContent=titles[lab.stage];$('update-example-text').textContent=trace.example.text;
  $('preview-badge').textContent=trace.applied?'APPLIED · real weights updated':'PREVIEW · weights restored';$('preview-badge').className=`badge ${trace.applied?'online':'recorded'}`;
  $('preview-state-note').textContent=trace.applied?'The cached update was committed exactly once. Further training can change the live model.':'This is a real computed update; model, optimizer and RNG state were restored after preview.';
  $('advance-update').textContent=['Measure loss →','Inspect gradients →','Review Adam update →',trace.applied?'Show comparison →':'Apply this update →','Preview another update'][lab.stage];
  box.replaceChildren();
  if(lab.stage===0){const before=trace.before;box.append(renderForwardStart(before,lab.meta?.architecture));const probs=make('div','training-probs');if(Array.isArray(before?.probabilities)&&before?.target)probabilities(probs,before.probabilities,before.target.id);else probs.append(make('p','empty-message','Prediction probabilities unavailable in this preview.'));box.append(probs,make('p','stage-note',`Position ${before?.position??'—'} · target probability ${pct(before?.targetProbability)}`));math('update-formula',String.raw`p_\theta(x_{t+1}\mid x_{\leq t})=\big[\operatorname{softmax}(h_tW_E^\top)\big]_{x_{t+1}}`);}
  else if(lab.stage===1){const mean=lossMeanView(trace),metrics=make('div','trace-metrics');metrics.append(traceMetric('Selected-position loss',number(trace.before?.tokenLoss)),traceMetric('Full-example mean loss',number(trace.loss)));box.append(metrics,make('p','loss-prediction-count',mean.count===null?'Prediction count unavailable.':`T = ${mean.count} next-token prediction${mean.count===1?'':'s'} in this input window`));math('update-formula',mean.formula);renderSelectedLossExample(trace.before);}
  else if(lab.stage===2){renderBackward(trace,box);}
  else if(lab.stage===3){renderAdam(trace,box);}
  else{box.append(make('h3','trace-narration','The same prediction, before and after.'));const comparison=make('div','compare-grid');for(const [name,result]of[['Before',trace.before],['After',trace.after]]){const section=make('section'),probs=make('div','training-probs');section.append(make('h3',null,`${name} · target ${pct(result.targetProbability)}`));probabilities(probs,result.probabilities,result.target.id);section.append(probs);comparison.append(section);}const metrics=make('div','trace-metrics');metrics.append(traceMetric('Mean loss before',number(trace.before.batchLoss)),traceMetric('Mean loss after',number(trace.after.batchLoss)));box.append(comparison,metrics);math('update-formula',String.raw`\Delta\mathcal L=${number(trace.after.batchLoss-trace.before.batchLoss)}\quad\text{on the same example}`);}
  controls();
}

async function previewUpdate(){return action(async()=>{if(!lab.trainingExample)throw new Error('No training example is available. Use Retry to load the training split before previewing an update.');pauseRecording();lab.trace=await api('preview',{exampleId:lab.trainingExample.id,learningRate:Number($('update-lr').value)});lab.stage=0;lab.maxStage=0;renderUpdate();message('Preview uses an example from your training split; weights were restored. Apply commits the update.');},'Computing a real forward, backward and Adam preview…');}

async function advanceUpdate(){
  if(!lab.trace||lab.busy)return;if(lab.stage===4){await previewUpdate();return;}
  if(lab.stage===3&&!lab.trace.applied){await action(async()=>{lab.live=await api('commit',{previewId:lab.trace.previewId});lab.trace.applied=true;lab.stage=4;lab.maxStage=4;renderUpdate();renderConvergence();await refreshInspection();message('The exact previewed update was applied once. The live step counter has advanced.');},'Applying the cached update…');}
  else{lab.stage++;lab.maxStage=Math.max(lab.maxStage,lab.stage);renderUpdate();}
}

function measurementInfo(state,transfer=false){
  const valid=value=>Number.isSafeInteger(value)&&value>=0&&(!Number.isSafeInteger(state?.step)||value<=state.step);
  let step=valid(state?.lastMeasuredStep)?state.lastMeasuredStep:null;
  if(step===null){
    const histories=transfer?(state?.branches||[]).map(branch=>branch.history):[state?.history];
    const observed=histories.map(history=>history?.at(-1)?.step);
    if(observed.length&&observed.every(value=>valid(value)&&value===observed[0]))step=observed[0];
  }
  const interval=Number.isSafeInteger(state?.evaluationIntervalSteps)&&state.evaluationIntervalSteps>0?state.evaluationIntervalSteps:null;
  return{step,interval,lagging:step!==null&&state?.step>step};
}
function measurementText(state,transfer=false){
  const {step,interval,lagging}=measurementInfo(state,transfer),kind=transfer?'Metrics':'Loss';
  const sample=step===null?`${kind} measurement step unavailable.`:`${kind} measured at ${transfer?'fine-tuning ':''}step ${step}.`;
  const current=lagging?` Current ${transfer?'pair':'model'}: step ${state.step}.`:'';
  const schedule=interval?` Evaluation every ${interval} optimizer steps during training.`:'';
  return sample+current+schedule;
}
const chartSignatures=new WeakMap();
function svgNode(tag,attributes,text){const node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,value]of Object.entries(attributes||{}))node.setAttribute(key,String(value));if(text!==undefined)node.textContent=text;return node;}
function drawChart(id,series,{accuracy=false,selected=null,emptyText='No measured observations are available yet.',valueUnit='',windowed=false}={}){
  const svg=typeof id==='string'?$(id):id,signature=JSON.stringify([svg.getAttribute('viewBox'),series,accuracy,selected,emptyText,valueUnit,windowed]);if(chartSignatures.get(svg)===signature)return;
  chartSignatures.set(svg,signature);const view=svg.getAttribute('viewBox').split(' ').map(Number),w=view[2],h=view[3],pad={l:53,r:18,t:17,b:42};svg.replaceChildren();
  const cleaned=series.map(item=>({...item,data:item.data.filter(point=>finite(point.step)&&finite(point.value))})),points=cleaned.flatMap(item=>item.data);
  if(!points.length){svg.append(svgNode('text',{x:w/2,y:h/2,'text-anchor':'middle'},emptyText));return;}
  const xMax=Math.max(1,...points.map(point=>point.step)),firstStep=Math.min(...points.map(point=>point.step)),xMin=windowed&&firstStep<xMax?firstStep:0;let min=accuracy?0:Math.min(...points.map(point=>point.value)),max=accuracy?1:Math.max(...points.map(point=>point.value));
  if(!accuracy){const margin=Math.max((max-min)*.1,.03);min=Math.max(0,min-margin);max+=margin;if(max===min)max=min+1;}
  const x=value=>pad.l+(value-xMin)/(xMax-xMin)*(w-pad.l-pad.r),y=value=>h-pad.b-(value-min)/(max-min)*(h-pad.t-pad.b);
  for(let i=0;i<=4;i++){const value=min+(max-min)*i/4,py=y(value);svg.append(svgNode('line',{x1:pad.l,x2:w-pad.r,y1:py,y2:py,class:'chart-grid'}),svgNode('text',{x:pad.l-9,y:py+4,'text-anchor':'end'},accuracy?`${Math.round(value*100)}%`:value.toFixed(2)));}
  const xTicks=xMax-xMin<4?Array.from({length:Math.floor(xMax-xMin)+1},(_,index)=>xMin+index):[...new Set(Array.from({length:5},(_,index)=>Math.round(xMin+(xMax-xMin)*index/4)))];
  for(const step of xTicks)svg.append(svgNode('text',{x:x(step),y:h-pad.b+20,'text-anchor':'middle'},String(step)));
  svg.append(svgNode('text',{x:(w+pad.l-pad.r)/2,y:h-4,'text-anchor':'middle'},'Optimizer step'));
  if(finite(selected))svg.append(svgNode('line',{x1:x(selected),x2:x(selected),y1:pad.t,y2:h-pad.b,class:'selected-step'}));
  for(const item of cleaned){if(!item.data.length)continue;const path=item.data.map((point,index)=>`${index?'L':'M'}${x(point.step)},${y(point.value)}`).join(' ');svg.append(svgNode('path',{d:path,class:'series-path',stroke:item.color,'stroke-dasharray':item.dashed?'5 4':''}));for(const point of item.data){const dot=svgNode('circle',{cx:x(point.step),cy:y(point.value),r:item.data.length>80?1.7:2.6,fill:item.color});dot.append(svgNode('title',{},`${item.label} · step ${point.step} · ${accuracy?pct(point.value):number(point.value,5)}${valueUnit}`));svg.append(dot);}}
}

const performanceSignatures=new WeakMap();
const timingPhaseNames={data:'Data packing + transfer',setup:'Update setup',forward_loss:'Forward + loss',loss_check:'Loss check',backward:'Backward',gradient_check_clip:'Gradient check + clipping',optimizer:'Adam + loss scaling',train_evaluation:'Training-set evaluation',validation:'Validation',token_observation:'Token probability recording'};
function orderedTimingPhases(phases){
  const order=Object.keys(timingPhaseNames),rank=name=>{const parts=name.split('.'),branch=parts.length===2?parts[0]:null,index=order.indexOf(parts.at(-1));return[branch==='pretrained'?1:branch==='scratch'?2:branch?3:0,index<0?order.length:index];};
  return Object.entries(phases||{}).sort(([a],[b])=>{const x=rank(a),y=rank(b);return x[0]-y[0]||x[1]-y[1]||a.localeCompare(b);});
}
function timingPhaseLabel(name){
  const parts=name.split('.'),branch=parts.length===2?parts[0]:null,phase=parts.at(-1),label=timingPhaseNames[phase]||name;
  return branch==='pretrained'?`Copied LM · ${label}`:branch==='scratch'?`Random weights · ${label}`:label;
}
function timingCount(value){return Number.isSafeInteger(value)&&value>=0?value:0;}
function timingStat(value){return value&&timingCount(value.count)>0&&finite(value.mean)&&value.mean>=0&&finite(value.p95)&&value.p95>=0?value:null;}
function timingClock(rows,key,label,supported){
  const column=make('section',`timing-clock ${key==='gpuMs'?'timing-gpu':'timing-cpu'}`);column.append(make('h5',null,label));
  if(!supported){column.append(make('p','small-note','GPU event timing is not supported on this runtime. CPU wall measurements remain available.'));return column;}
  const largest=Math.max(.001,...rows.flatMap(row=>{const stat=timingStat(row.values?.[key]);return stat?[stat.mean,stat.p95]:[];}));
  for(const row of rows){
    const item=make('div',`timing-phase${row.total?' timing-total':''}`),stat=timingStat(row.values?.[key]);item.append(make('strong',null,row.label));
    if(stat){
      item.append(make('span','timing-values',`Mean ${number(stat.mean,3)} ms · p95 ${number(stat.p95,3)} ms · n = ${stat.count}`));
      const track=make('div','timing-track'),mean=make('span','timing-mean'),p95=make('span','timing-p95');track.setAttribute('aria-hidden','true');mean.style.setProperty('width',`${100*stat.mean/largest}%`);p95.style.setProperty('left',`${100*stat.p95/largest}%`);track.append(mean,p95);item.append(track);
    }else item.append(make('span','timing-values',key==='gpuMs'?'No resolved GPU samples':'No valid CPU samples'));
    column.append(item);
  }
  return column;
}
function timingGroup(timing,key){
  const group=timing[key]||{},evaluations=key==='evaluations',section=make('section','timing-group');
  section.append(make('h4',null,evaluations?'Evaluation time':timing.kind==='transfer'?'Paired fine-tuning update time':'Optimizer update time'));
  section.append(make('p','small-note',`${timingCount(group.successfulCount)} successful / ${timingCount(group.count)} retained samples · ${timingCount(group.failedCount)} failed. Mean and p95 use successful samples only; CPU and GPU sample counts can differ.`));
  const rows=[{label:evaluations?'Total evaluation':'Total update · excludes evaluation',values:group.total,total:true},...orderedTimingPhases(group.phases).map(([name,values])=>({label:timingPhaseLabel(name),values}))];
  const clocks=make('div','timing-clock-grid');clocks.append(timingClock(rows,'cpuMs','CPU wall time',true),timingClock(rows,'gpuMs','GPU stream time',timing.gpuSupported===true));section.append(clocks);
  const heading=make('div','timing-trend-heading');heading.append(make('h5',null,`Total ${evaluations?'evaluation':'update'} trend · ms`));
  const legend=make('p','small-note timing-legend');legend.append(make('span','timing-cpu-key','CPU wall'),make('span','timing-gpu-key','GPU stream'));heading.append(legend);section.append(heading);
  const chart=svgNode('svg',{class:'line-chart timing-chart',viewBox:'0 0 760 230',role:'img','aria-label':`${evaluations?'Evaluation':'Sampled update'} CPU wall and GPU stream duration in milliseconds at actual optimizer steps`});
  const trend=Array.isArray(group.trend)?group.trend:[],successful=trend.filter(point=>point.success===true&&Number.isSafeInteger(point.step)&&point.step>=0);
  drawChart(chart,[{label:'CPU wall',color:'#00549f',data:successful.map(point=>({step:point.step,value:point.cpuMs}))},{label:'GPU stream',color:'#b37528',dashed:true,data:timing.gpuSupported===true?successful.map(point=>({step:point.step,value:point.gpuMs})):[]}],{valueUnit:' ms',windowed:true,emptyText:'No successful timing samples are available yet.'});
  section.append(chart,make('p','small-note','Dots are actual successful samples. Lines connect sampled points; intervening updates are not timed. Missing GPU results are not zero.'));
  return section;
}
function renderPerformance(kind,state,{visible=true}={}){
  const id=kind==='lm'?'lm-performance':'transfer-performance',panel=$(id),summary=$(`${id}-summary`),content=$(`${id}-content`);panel.hidden=!visible;if(!visible)return;
  const timing=state?.timing,valid=timing?.version===1&&timing.kind===kind;
  const unavailable=!valid?'Timing is unavailable on the connected runtime.':timing.enabled!==true?'Timing collection is disabled on this runtime.':null;
  const counters=timing?.counters||{};
  const caption=unavailable||`${timingCount(counters.sampledSteps)} updates sampled · ${timingCount(counters.evaluations)} evaluations`;
  if(summary.textContent!==caption)summary.textContent=caption;
  if(!panel.open)return;
  const signature=JSON.stringify([valid?[timing.kind,timing.enabled,timing.sampleEvery,timing.capacity,timing.gpuSupported,timing.steps,timing.evaluations]:null,unavailable,state?.evaluationIntervalSteps]);
  const diagnosticText=`Since timing reset: ${timingCount(counters.observedSteps)} update attempts observed; ${timingCount(counters.sampledSteps)} updates selected; ${timingCount(counters.successfulSamples)} successful and ${timingCount(counters.failedSamples)} failed update/evaluation samples. ${timingCount(timing?.pendingGpuSamples)} GPU samples pending; ${timingCount(counters.gpuDroppedSamples)} GPU samples skipped; ${timingCount(counters.eventErrors)} GPU event errors.${timingCount(counters.clockErrors)?` ${timingCount(counters.clockErrors)} CPU clock errors; invalid CPU timings are unavailable.`:''}`;
  const cached=performanceSignatures.get(panel);
  const updateDiagnostics=node=>{if(node.textContent!==diagnosticText)node.textContent=diagnosticText;node.classList.toggle('timing-warning',Boolean(timingCount(counters.failedSamples)||timingCount(counters.eventErrors)||timingCount(counters.clockErrors)));};
  if(cached?.signature===signature){if(cached.diagnostics)updateDiagnostics(cached.diagnostics);return;}
  if(unavailable){performanceSignatures.set(panel,{signature,diagnostics:null});content.replaceChildren(make('p','small-note',unavailable+' No timing values are inferred from loss curves or recorded runs.'));return;}
  const sampleEvery=timingCount(timing.sampleEvery),capacity=timingCount(timing.capacity),interval=timingCount(state?.evaluationIntervalSteps);
  const cadence=sampleEvery?`The first update after timing reset and every ${sampleEvery}th absolute update are sampled.`:'Only selected optimizer updates are sampled.';
  const evaluation=interval?`Every actual evaluation is timed; normal training evaluates every ${interval} optimizer steps.`:'Every actual evaluation is timed.';
  const scope=make('p','small-note timing-scope',`${cadence} ${evaluation} Each view retains up to ${capacity||'the runtime limit of'} recent records. Mean and p95 use that same retained window, not the full run.`);
  const clocks=make('p','small-note','CPU wall time includes waiting. GPU stream time includes dispatch gaps; it is not GPU busy time. These clocks overlap. Phases may overlap and omit small overheads, so do not add phases or CPU + GPU totals. Bars show means; ticks show p95. Each clock column has its own scale.');
  const diagnostics=make('p','small-note timing-diagnostics');updateDiagnostics(diagnostics);performanceSignatures.set(panel,{signature,diagnostics});
  content.replaceChildren(scope,clocks,diagnostics,timingGroup(timing,'steps'),timingGroup(timing,'evaluations'),make('p','small-note','Timing is diagnostic and runtime-only. Warmup, earlier sessions and recorded experiments are not reconstructed. Pending GPU results may remain after a job finishes; training never waits just to complete this display.'));
}

function getCheckpoints(){return [...(lab.recordings?.pretraining?.checkpoints||[])].sort((a,b)=>a.step-b.step);}
function getRuns(){return lab.recordings?.transfer?.runs||[];}
function renderConvergence(){
  const recorded=lab.mode==='recorded',checkpoints=getCheckpoints(),history=recorded?(lab.recordings?.pretraining?.history||[]):(lab.live?.history||[]);
  lab.checkpoint=Math.max(0,Math.min(lab.checkpoint,checkpoints.length-1));const checkpoint=recorded?checkpoints[lab.checkpoint]:null;
  const observed=recorded?history.find(point=>point.step===checkpoint?.step):lab.live?.metrics;
  $('mode-live').classList.toggle('active',!recorded);$('mode-live').setAttribute('aria-pressed',String(!recorded));$('mode-recorded').classList.toggle('active',recorded);$('mode-recorded').setAttribute('aria-pressed',String(recorded));
  $('live-controls').hidden=recorded;$('recorded-controls').hidden=!recorded;
  $('convergence-mode').textContent=recorded?'RECORDED RUN':'LIVE TRAINING';$('convergence-mode').className=`badge ${recorded?'recorded':'online'}`;
  const measurement=measurementInfo(lab.live);
  $('convergence-description').textContent=recorded?'Actual saved observations. The slider selects stored checkpoints.':`${executionLabel()?`${executionLabel()}. `:''}One background task; progress every 0.1 s. ${measurement.interval?`Loss evaluation every ${measurement.interval} optimizer steps.`:'Plots show actual measured updates.'}`;
  $('metric-step-label').textContent=recorded?'Selected checkpoint step':'Current training step';
  $('metric-step').textContent=recorded?(checkpoint?.step??'—'):(lab.live?.step??'—');$('metric-train').textContent=number(observed?.trainLoss);$('metric-validation').textContent=number(observed?.valLoss);$('metric-source').textContent=recorded?(checkpoint?.label||'Saved checkpoint'):(lab.live?.source||'—');
  $('metric-measurement').textContent=recorded?(checkpoint&&observed?`Saved loss measured at step ${checkpoint.step}.`:'No saved loss measurement is available at this checkpoint.'):measurementText(lab.live);
  $('metric-measurement').classList.toggle('measurement-lagging',!recorded&&measurement.lagging);
  const slider=$('checkpoint-slider');slider.max=Math.max(0,checkpoints.length-1);slider.value=lab.checkpoint;$('checkpoint-label').textContent=checkpoint?`Step ${checkpoint.step} · ${lab.checkpoint+1}/${checkpoints.length}`:'No checkpoints available';
  drawChart('convergence-chart',[{label:'Train',color:'#00549f',data:history.map(point=>({step:point.step,value:point.trainLoss}))},{label:'Validation',color:'#b37528',dashed:true,data:history.map(point=>({step:point.step,value:point.valLoss}))}],{selected:checkpoint?.step});
  const recordedEvaluation=(lab.recordings?.notes||[]).find(note=>note.includes('TinyStories LM evaluation'));
  $('evaluation-note').textContent=recorded?`${recordedEvaluation||lab.live?.metrics?.evalDescription||'See the saved run metadata for the evaluation subset.'} Values are never interpolated between checkpoints.`:`${lab.live?.metrics?.evalDescription||'Evaluation details are provided by the engine with live metrics.'} Points show their actual measurement steps. Explicit one-update demonstrations may add a measurement between scheduled evaluations.`;
  renderPerformance('lm',lab.live,{visible:!recorded});
  const probe=checkpoint?.probe;
  if(recorded&&probe){$('probe-title').textContent=`Stored prediction · step ${checkpoint.step}`;$('probe-context').textContent=probe.tokens.slice(0,probe.position+1).map(token=>piece(token.piece)).join(' ');probabilities('checkpoint-probabilities',probe.probabilities,probe.target.id);$('probe-note').textContent=`Observed target ${piece(probe.target.piece)}: ${pct(probe.targetProbability)}. Selected-position loss ${number(probe.tokenLoss)}.`;}
  else{$('probe-title').textContent='A stored checkpoint prediction';$('probe-context').textContent=recorded?'No saved probe is available at this checkpoint.':'Switch to Recorded run to compare saved predictions.';$('checkpoint-probabilities').replaceChildren();$('probe-note').textContent='';}
  $('recording-note').textContent=lab.recordings?.pretraining?.available?'Only actual saved checkpoints are selectable. Playback speed changes the presentation, not training.':(lab.recordings?.notes||[]).join(' ')||'No pretraining recording is available yet. Use live training or refresh after a run has been recorded.';
  renderLiveSource();renderSaveStatus();controls();
}

async function trainSteps(total){
  if(lab.busy||jobLocked()||!lab.ready)return;
  if(trainingMode(total)==='teaching')return teachSteps('lm',total,Number($('training-lr').value));
  lab.teaching=null;pauseRecording();lab.trace=null;renderUpdate();
  lab.jobDisplayKind='lm';
  await trainingJobs.start({kind:'lm',steps:total,learningRate:Number($('training-lr').value)});
}

async function resetModel(source){const seed=Number($('reset-seed').value);if(!Number.isInteger(seed)||seed<0||seed>2147483647){message('Enter a whole-number seed between 0 and 2147483647.',true);return;}return action(async()=>{pauseRecording();lab.live=await api('reset',{source,seed});const metadataWarning=await refreshExecutionMetadata();lab.trace=null;renderUpdate();renderConvergence();renderLiveTransfer();if(lab.inspection){lab.inspection=await api('inspect',{text:lab.inspection.text,position:lab.inspection.position});lab.inspectionStep=lab.live.step;renderInspection();}message(`Live model reset from ${source}; actual step ${lab.live.step}.${metadataWarning}`);},`Restoring the ${source} model state…`);}

function pauseRecording(){lab.playing=false;if(lab.timer!==null){clearTimeout(lab.timer);lab.timer=null;}if($('play-recording'))controls();}
function scheduleRecording(){if(!lab.playing)return;lab.timer=setTimeout(()=>{lab.timer=null;const checkpoints=getCheckpoints();if(lab.checkpoint>=checkpoints.length-1){pauseRecording();message('Recorded playback finished at the last real saved checkpoint.');return;}lab.checkpoint++;renderConvergence();if(lab.checkpoint>=checkpoints.length-1){pauseRecording();message('Recorded playback finished at the last real saved checkpoint.');}else scheduleRecording();},1000/Number($('recording-speed').value));}
function playRecording(){if(lab.busy||!getCheckpoints().length)return;if(lab.checkpoint>=getCheckpoints().length-1)lab.checkpoint=0;lab.playing=true;renderConvergence();scheduleRecording();message('Playing actual saved checkpoints. This control does not perform training.');}

function validInteger(id,min,max,label){
  const value=Number($(id).value);if(!$(id).value.trim()||!Number.isInteger(value)||value<min||value>max)throw new Error(`${label} must be a whole number from ${min} to ${max}.`);return value;
}
function sourceDescription(source){return `LM step ${source?.step??'—'} · initialization seed ${source?.seed??'—'} · ${source?.source||source?.model||'unknown source'}`;}
function renderLiveSource(){
  $('live-transfer-current').textContent=`Current language model: ${sourceDescription({...lab.live,seed:lab.live?.seed??lab.meta?.seed})}. Starting again replaces the live transfer pair; save first to retain it.`;
}
function renderSaveStatus(){
  const receipt=lab.saveReceipt;
  const text=lab.saving?'Saving the actual language model and live transfer state…':lab.saveError?`Save was not confirmed: ${lab.saveError}`:receipt?`Saved ${receipt.savedAt} · LM step ${receipt.step} · ${receipt.transferStep===null||receipt.transferStep===undefined?'no transfer pair':`transfer step ${receipt.transferStep}`}. Further changes require another save.`:'No save has been confirmed in this page session.';
  for(const id of ['save-live-status','save-transfer-status']){$(id).textContent=text;$(id).classList.toggle('error',Boolean(lab.saveError));}
}
async function saveLive(){
  if(!lab.ready||!lab.liveUseAvailable)return;
  return action(async()=>{lab.saving=true;lab.saveError=null;renderSaveStatus();try{const result=await api('save-live',{});if(result.saved!==true)throw new Error('The backend did not confirm a saved checkpoint.');lab.saveReceipt=result;message(`Session saved: LM step ${result.step}${result.transferStep!==null&&result.transferStep!==undefined?`, transfer step ${result.transferStep}`:''}.`);}catch(error){lab.saveError=error.message;throw error;}finally{lab.saving=false;renderSaveStatus();}},'Saving the current model and any live transfer pair…');
}
const generationRate=()=>$('generation-speed').value==='immediate'?Infinity:Number($('generation-speed').value);
function createGenerationCard(model,text){
  const card=make('section','generation-card'),heading=make('h4',null,model==='live'?'Current trained model':'Random baseline'),source=make('p','small-note','Waiting for source metadata…'),context=make('p','generation-context-mode','Context mode will appear with the stream.'),window=make('p','generation-window-counts','No token has been revealed yet.'),output=make('p','generated-text'),continuation=make('strong','generation-continuation',''),status=make('p','generation-card-status','Waiting for the GPU stream…'),sampling=make('p','small-note',''),details=make('details','generation-token-details'),tokens=make('div','generated-tokens');
  card.dataset.model=model;output.append(make('span','generation-context',text),continuation);details.append(make('summary',null,'Revealed tokens & sampling probabilities'),tokens,make('p','small-note','Each chip is one actual model token. Hover or focus it for its sampling probability. Text follows the tokenizer’s cumulative decoding.'));card.append(heading,source,context,output,status,window,sampling,details);return {card,source,context,window,output,continuation,status,sampling,tokens};
}
function updateGenerationCard(view,state){
  if(state.start){view.source.textContent=sourceDescription(state.start.source);view.sampling.textContent=`Temperature ${state.start.sampling.temperature} · sampling seed ${state.start.sampling.seed} · maximum ${state.start.maxNewTokens} new tokens`;}
  if(state.start){const context=state.start.context;view.context.textContent=context?`${context.mode==='sliding'?'Sliding window':'Fixed context'} · ${context.windowSize} tokens · prompt ${context.promptTokens} tokens including BOS · model context ${context.modelContext}`:'Fixed context · 128 tokens. This older stream does not report prompt or window counts.';}
  view.window.textContent=state.window?`Revealed token ${state.displayed}: predicted from ${state.window.inputTokens} input tokens · ${state.window.droppedTokens} older tokens excluded.`:state.displayed?'This stream does not provide per-token window counts.':'Window counts appear when the first token is revealed.';
  if(view.continuation.textContent!==state.continuation){const previousTop=view.output.scrollTop,follow=view.output.scrollHeight-view.output.clientHeight-previousTop<=32;view.continuation.textContent=state.continuation;view.output.scrollTop=follow?view.output.scrollHeight:previousTop;}
  const count=`${state.displayed} displayed / ${state.received} received`;
  const finishReason={eos:'EOS — model ended generation',length:'Maximum token limit reached',context:'Context limit reached'}[state.result?.stopReason]??state.result?.stopReason;
  view.status.textContent=state.status==='done'?`Complete · ${state.displayed} actual tokens · ${finishReason}`:state.status==='stopped'?`Stopped · ${count}. Visible text is partial; hidden tokens were discarded.`:state.status==='failed'?`Failed · ${count}. ${state.error}`:state.status==='paused'?`Display paused · ${count} · ${state.queued} buffered. GPU output may continue.`:state.status==='revealing'?`GPU generation finished · ${count} · revealing ${state.queued} buffered tokens…`:state.status==='streaming'?`Receiving real tokens · ${count} · ${state.queued} buffered…`:'Waiting for the GPU stream…';
  view.card.dataset.status=state.status;view.status.classList.toggle('error',state.status==='failed');
}
function appendGeneratedToken(view,event){const chip=make('span',null,piece(event.token.piece));chip.tabIndex=0;chip.title=`Token ${event.index+1} · ID ${event.token.id} · sampling probability ${pct(event.token.p)}`;chip.setAttribute('aria-label',chip.title);view.tokens.append(chip);}
function stopGeneration(){
  const run=lab.generation;if(!run?.active||run.stopped)return;run.stopped=true;run.controller.abort();for(const {player}of run.cards.values())player.stop();$('generation-playback-status').textContent='Stopped. Visible partial text is retained; hidden queued tokens were discarded.';message('Generation stopped. The displayed partial result is retained.');controls();
}
function pauseGeneration(){
  const run=lab.generation;if(!run?.active||run.stopped)return;run.paused=!run.paused;for(const {player}of run.cards.values())run.paused?player.pause():player.resume();$('generation-playback-status').textContent=run.paused?'Display paused. GPU generation can continue to buffer; Resume reveals the remaining tokens.':'Revealing genuine model tokens at the selected display speed.';controls();
}
async function generateText(compare=false){
  if(lab.busy||jobLocked()||!lab.ready||!lab.liveUseAvailable)return;
  let settings;
  try{const text=$('generation-prompt').value;if(!text.trim())throw new Error('Enter a story prompt.');const maxNewTokens=validInteger('generation-tokens',1,1000,'Maximum new tokens'),seed=validInteger('generation-seed',0,2147483647,'Sampling seed'),temperature=Number($('generation-temperature').value),slidingWindow=$('generation-sliding').checked,windowSize=slidingWindow?validInteger('generation-window-size',1,128,'Sliding window length'):128;if(!$('generation-temperature').value.trim()||!finite(temperature)||temperature<0||temperature>1.5)throw new Error('Temperature must be between 0 and 1.5.');settings={text,maxNewTokens,temperature,seed,slidingWindow,windowSize};}
  catch(error){message(error.message,true);$('generation-playback-status').textContent=error.message;return;}
  pauseRecording();const run={active:true,stopped:false,paused:false,controller:new AbortController(),cards:new Map()};lab.generation=run;lab.busy=true;
  const isCurrent=()=>lab.generation===run;
  $('generation-results').replaceChildren();$('generation-results').classList.toggle('has-comparison',compare);
  for(const model of compare?['random','live']:['live']){
    const view=createGenerationCard(model,settings.text),player=new TokenPlayer({rate:generationRate(),onChange:state=>{if(isCurrent())updateGenerationCard(view,state);},onToken:event=>{if(isCurrent()&&!run.stopped)appendGeneratedToken(view,event);}});
    run.cards.set(model,{view,player});$('generation-results').append(view.card);
  }
  controls();$('generation-playback-status').textContent='Streaming genuine model tokens. Display speed is independent of GPU generation speed.';message(compare?'Streaming a controlled current/random comparison…':'Streaming from the current live language model…');
  let activePlayer=null;
  try{
    for(const model of compare?['live','random']:['live']){
      if(run.stopped||!isCurrent())break;activePlayer=run.cards.get(model).player;let timedOut=false;
      const timeout=setTimeout(()=>{timedOut=true;run.controller.abort();},180000);
      try{
        const response=await sessionFetch(`${trainingEndpoint}generate-stream`,{method:'POST',headers:{'Content-Type':'application/json',Accept:'application/x-ndjson'},body:JSON.stringify({...settings,model}),cache:'no-store',signal:run.controller.signal});
        const result=await readGenerationStream(response,{signal:run.controller.signal,onEvent:event=>{if(!isCurrent()||run.stopped)return;if(event.type==='start'){if(event.source.model!==model)throw new Error('The stream source does not match the requested model.');if(event.context?(event.context.mode!==(settings.slidingWindow?'sliding':'fixed')||event.context.windowSize!==settings.windowSize):settings.slidingWindow)throw new Error('The runtime did not confirm the requested context window settings.');activePlayer.start(event);}else if(event.type==='token')activePlayer.enqueue(event);}});
        if(isCurrent()&&!run.stopped)activePlayer.networkDone(result);
      }catch(error){if(timedOut)throw new Error('The GPU token stream timed out. The visible text is partial; generation was not retried.');throw error;}
      finally{clearTimeout(timeout);}
      // Begin the next network request after EOF, without waiting for paced reveal.
    }
    if(!run.stopped&&isCurrent())$('generation-playback-status').textContent=run.paused?'GPU generation finished. Resume display to reveal the buffered tokens.':'GPU generation finished. Revealing any buffered tokens at the selected speed…';
    await Promise.all([...run.cards.values()].map(({player})=>player.finished));
    if(isCurrent()&&!run.stopped){$('generation-playback-status').textContent='Complete. All received tokens are visible; the language-model training state is unchanged.';message('Stream complete. All genuine model tokens have been displayed.');}
  }catch(error){
    if(isCurrent()&&!run.stopped){activePlayer?.fail(error);for(const {player}of run.cards.values())if(player!==activePlayer)player.stop('The comparison was interrupted. Visible text is partial.');$('generation-playback-status').textContent=`Streaming failed: ${error.message}`;message(error.message,true);}run.controller.abort();
  }finally{if(isCurrent()){run.active=false;lab.busy=false;controls();}}
}
function clearLivePrediction(){
  $('live-sentiment-probabilities').replaceChildren();$('live-classifier-checkpoint').textContent=lab.liveTransfer?.step>=1?'Ready to classify with the current fine-tuned heads. Your review is used only for prediction.':'Start a pair and fine-tune at least one step. Your review is used only for prediction.';
}
async function readLiveTransfer({optional=false}={}){
  try{const state=await api('transfer-state',undefined,{optional});if(typeof state.available!=='boolean')throw new Error('The live transfer service returned an invalid state.');lab.liveTransfer=state;lab.liveUseAvailable=true;lab.liveUseError=null;}
  catch(error){lab.liveUseAvailable=false;lab.liveUseError=error.message;if(!optional)throw error;}
  finally{renderLiveTransfer();}
}
function renderLiveTransfer(){
  renderLiveSource();const state=lab.liveTransfer,available=lab.liveUseAvailable&&state?.available;
  const transferLabel=lab.liveUseAvailable&&transferExecutionLabel()?`${transferExecutionLabel()}. `:'';
  $('live-use-status').textContent=lab.liveUseAvailable?'Generation and saving use the current live session, including when the chart above shows a recorded run.':`Live model tools are not available yet. Use Refresh live state in Transfer after the runtime is updated.${lab.liveUseError?` ${lab.liveUseError}`:''}`;
  $('live-use-status').classList.toggle('error',!lab.liveUseAvailable&&Boolean(lab.liveUseError));
  const requiresRestart=available&&state.requiresRestart===true;
  $('live-transfer-status').textContent=transferLabel+(!lab.liveUseAvailable?'Live transfer is unavailable on the connected runtime. The recorded experiments remain available below.':requiresRestart?'This transfer pair requires a restart. Use “Reset transfer” to create a new pair before training again.':available?`Live pair ready · ${state.step} fine-tuning steps per branch. Its copied language-model source stays fixed while you train the pair.`:'No live transfer pair yet. Copy the current language model to begin a real comparison.');
  $('live-transfer-status').classList.toggle('error',Boolean(requiresRestart));$('live-transfer-status').setAttribute('role','status');
  $('start-live-transfer').textContent=available?'Reset transfer':'Initialize transfer';
  $('live-transfer-step').textContent=requiresRestart?'Reset required':available?`Step ${state.step??'—'} / branch`:'Not initialized';
  const provenance=$('live-transfer-provenance');provenance.replaceChildren();
  if(available){
    const stats=make('div','live-transfer-stats');for(const [label,value]of[['Copied source LM step',state.source?.step??'—'],['Fine-tuning step / branch',state.step??'—'],['Comparison seed',state.seed??'—']])stats.append(traceMetric(label,String(value)));provenance.append(stats);
    const details=make('details','live-source-details');details.append(make('summary',null,'Exact source & run provenance'),make('p','small-note',sourceDescription(state.source)),make('p','weight-name',`Source model SHA-256: ${state.source?.modelSha256||'Unavailable'}`),make('p','weight-name',`Live run: ${state.runId??'—'} · pretrained auxiliary weight: ${state.auxiliaryWeight??'—'}`));provenance.append(details);
  }else provenance.append(make('p','small-note',lab.liveUseAvailable?'The source LM step is captured only when you start a pair.':'No current live pair has been confirmed.'));
  const runs=available?(state.branches||[]):[],cards=[],measurement=measurementInfo(state,true);
  $('live-transfer-measurement').textContent=available?measurementText(state,true):'No live transfer measurement is available.';
  $('live-transfer-measurement').classList.toggle('measurement-lagging',Boolean(available&&measurement.lagging));
  for(const name of ['scratch','pretrained']){const run=runs.find(item=>item.name===name);if(!run)continue;const branch=branches[name],card=make('section','transfer-card');card.style.setProperty('--branch-color',branch.color);card.append(make('h3',null,name==='scratch'?'Random → sentiment':'Trained LM → sentiment'),make('p','branch-caption',name==='scratch'?'Classification only · λ = 0':`Copied LM step ${state.source?.step??'—'} · λ = ${state.auxiliaryWeight??0}`),make('strong',null,pct(run.metrics?.accuracy)),make('p','small-note','Validation accuracy'));card.append(renderTransferObjective(run,measurement.step,name==='scratch'?0:state.auxiliaryWeight));const measuredDetails=make('details','transfer-card-measurement');measuredDetails.append(make('summary',null,'Accuracy & classification loss details'),make('p','small-note',measurement.step===null?'Validation accuracy · measurement step unavailable':`Validation accuracy · measured at fine-tuning step ${measurement.step}`),make('p','small-note',`Validation class loss ${number(run.metrics?.valLoss)} · train class loss ${number(run.metrics?.trainLoss)}`));card.append(measuredDetails);cards.push(card);}
  $('live-transfer-cards').replaceChildren(...cards);
  drawChart('live-accuracy-chart',runs.map(run=>({label:branches[run.name]?.label||run.name,color:branches[run.name]?.color||'#647584',data:(run.history||[]).map(point=>({step:point.step,value:point.accuracy}))})),{accuracy:true,emptyText:'Start a live pair to measure validation accuracy.'});
  drawChart('live-transfer-loss-chart',runs.map(run=>({label:branches[run.name]?.label||run.name,color:branches[run.name]?.color||'#647584',data:(run.history||[]).map(point=>({step:point.step,value:point.valLoss}))})),{emptyText:'Start a live pair to measure classification loss.'});
  $('live-transfer-evaluation').textContent=available?`${state.evalDescription||'Validation uses the held-out sentiment split.'} Only actual observations are plotted; user reviews and test data are never used for training or evaluation.`:'Only actual observations are plotted. Evaluation details appear after a pair starts.';
  renderPerformance('transfer',state);
  math('live-transfer-formula',String.raw`\mathcal L=\mathcal L_{\mathrm{class}}+\lambda\,\mathcal L_{\mathrm{LM}}`);controls();
}
async function startLiveTransfer(){
  if(lab.busy||jobLocked()||!lab.ready||!lab.liveUseAvailable)return;
  return action(async()=>{const seed=validInteger('live-transfer-seed',0,2147483647,'Comparison seed'),auxiliaryWeight=Number($('live-transfer-auxiliary').value);lab.liveTransfer=await api('transfer-start',{seed,auxiliaryWeight});const metadataWarning=await refreshExecutionMetadata();clearLivePrediction();renderLiveTransfer();message(`Live pair created from LM step ${lab.liveTransfer.source?.step}. Both branches are at fine-tuning step ${lab.liveTransfer.step}; the language model is unchanged.${metadataWarning}`);},'Copying current language-model weights and initializing the paired classifiers…');
}
async function trainLiveTransfer(total){
  if(lab.busy||jobLocked()||!lab.ready||!lab.liveUseAvailable||!lab.liveTransfer?.available||lab.liveTransfer.requiresRestart===true)return;
  if(trainingMode(total)==='teaching')return teachSteps('transfer',total,Number($('live-transfer-lr').value));
  lab.teaching=null;pauseRecording();clearLivePrediction();
  lab.jobDisplayKind='transfer';
  await trainingJobs.start({kind:'transfer',steps:total,learningRate:Number($('live-transfer-lr').value)});
}
async function predictLiveReview(){
  if(!lab.ready||!lab.liveUseAvailable||!(lab.liveTransfer?.step>=1))return;
  return action(async()=>{const result=await api('transfer-predict',{text:$('live-review-text').value,branch:$('live-classifier-branch').value}),labels=result.probabilities.map(item=>({...item,piece:item.label===0?'Negative':item.label===1?'Positive':String(item.label)}));probabilities('live-sentiment-probabilities',labels,undefined,{remaining:false});const prediction=result.prediction===0?'Negative':result.prediction===1?'Positive':String(result.prediction);$('live-classifier-checkpoint').textContent=`Prediction: ${prediction}. Actual ${result.branch} live head · fine-tuning step ${result.step} · copied LM source step ${result.source?.step} · run ${result.runId}.`;message('Prediction computed by the live fine-tuned classification head. This review was not used for training.');},'Classifying your review with the actual live transfer head…');
}
async function refreshLiveTransfer(){return action(async()=>{await readLiveTransfer();const metadataWarning=await refreshExecutionMetadata();clearLivePrediction();renderLiveTransfer();message(`Live transfer state refreshed from the actual running session.${metadataWarning}`);},'Reading the current live transfer state…');}
function renderTransfer(){renderLiveTransfer();renderRecordedTransfer();}
function renderRecordedTransfer(){
  const runs=getRuns(),seeds=[...new Set(runs.map(run=>run.seed))].sort((a,b)=>a-b);if(!seeds.includes(lab.seed))lab.seed=seeds[0]??null;
  const pretraining=lab.recordings?.pretraining,pretrainSteps=Math.max(0,...(pretraining?.history||[]).map(point=>point.step)),budget=lab.recordings?.transfer?.requestedSteps;
  const cost=pretraining?.completed?`The pretrained branches additionally use a common ${pretrainSteps}-step language-model pretraining run.`:pretraining?.available?`Language-model pretraining is additional work: ${pretrainSteps}${pretraining.requestedSteps?`/${pretraining.requestedSteps}`:''} steps have been recorded so far.`:'Language-model pretraining is additional work.';
  $('transfer-description').textContent=`Three branches use the same fine-tuning budget${budget?` of ${budget} steps per branch`:''}, comparing random initialization, pretraining, and pretraining with an auxiliary LM loss. ${cost} Improvement is an experimental result, not a guarantee.`;
  $('transfer-seed').replaceChildren(...(seeds.length?seeds.map(seed=>{const option=make('option',null,`Seed ${seed}`);option.value=seed;return option;}):[make('option',null,'No runs yet')]));if(lab.seed!==null)$('transfer-seed').value=lab.seed;
  const selected=runs.filter(run=>run.seed===lab.seed);const cards=[];
  for(const run of selected){const branch=branches[run.name]||{label:run.name,caption:`λ = ${run.lambda}`,color:'#647584'},last=run.history.at(-1),card=make('section','transfer-card');card.style.setProperty('--branch-color',branch.color);card.append(make('h3',null,branch.label),make('p','branch-caption',branch.caption),make('strong',null,pct(last?.accuracy)),make('p','small-note',`Validation accuracy · step ${last?.step??'—'}`));cards.push(card);}
  if(cards.length)$('transfer-cards').replaceChildren(...cards);else{empty('transfer-cards','No actual transfer runs are available yet. Refresh after recording the fine-tuning experiments.');}
  drawChart('accuracy-chart',selected.map(run=>({label:branches[run.name]?.label||run.name,color:branches[run.name]?.color||'#647584',data:run.history.map(point=>({step:point.step,value:point.accuracy}))})),{accuracy:true});
  drawChart('transfer-loss-chart',selected.map(run=>({label:branches[run.name]?.label||run.name,color:branches[run.name]?.color||'#647584',data:run.history.map(point=>({step:point.step,value:point.valLoss}))})));
  $('transfer-notes').replaceChildren(...(lab.recordings?.notes||[]).map(note=>make('p',null,note)));
  math('transfer-formula',String.raw`\mathcal L=\mathcal L_{\mathrm{class}}+\lambda\,\mathcal L_{\mathrm{LM}}`);controls();
}

async function refreshRecordings(){return action(async()=>{pauseRecording();lab.recordings=await api('recordings');renderConvergence();renderTransfer();message('Recorded results refreshed from the actual saved artifacts.');},'Reading saved pretraining and transfer observations…');}

async function predictReview(){return action(async()=>{const result=await api('predict',{text:$('review-text').value,branch:$('classifier-branch').value});const labels=result.probabilities.map(item=>({...item,piece:item.label===0?'Negative':item.label===1?'Positive':String(item.label)}));probabilities('sentiment-probabilities',labels,undefined,{remaining:false});$('classifier-checkpoint').textContent=`Prediction: ${result.prediction}. Actual ${result.branch} checkpoint · seed ${result.checkpoint.seed} · step ${result.checkpoint.step}. The comparison-seed selector controls the curves above.`;message('Prediction computed by the actual fine-tuned classification head.');},'Running the saved classifier on this review…');}

async function boot(){
  if(lab.busy)return;lab.teaching=null;lab.busy=true;lab.ready=false;controls();$('engine-badge').textContent='Connecting…';$('engine-badge').className='badge';message('Loading model, public data and real experiment state…');
  try{
    await trainingJobs.discover({reset:true});
    lab.meta=await api('meta');renderMetadata();lab.live=await api('state');
    await loadTrainingExample();lab.examples=SHORT_EXAMPLES;renderExamples();
    lab.recordings=await api('recordings');
    if(lab.examples.length&&!jobLocked()){
      try{lab.inspection=await api('inspect',{text:lab.examples[0].text,position:lab.examples[0].position},{optional:true});lab.inspectionStep=lab.live.step;renderInspection();}
      catch(error){if(error.code!=='training_busy')throw error;}
    }
    const sentiment=(await api('examples?kind=sentiment&split=train&limit=1')).examples;if(sentiment.length&&!$('review-text').value)$('review-text').value=sentiment[0].text;
    await readLiveTransfer({optional:true});clearLivePrediction();renderSaveStatus();
    if(jobIsActive(trainingJobs.state.job)){lab.live=trainingJobs.state.job.liveState;lab.liveTransfer=trainingJobs.state.job.transferState;}
    lab.ready=true;$('engine-badge').textContent='Real training engine ready';$('engine-badge').className='badge online';renderUpdate();renderConvergence();renderTransfer();renderJobProgress();
    message(jobIsActive(lab.job)?`Reconnected to background training: ${lab.job.completedSteps}/${lab.job.requestedSteps} steps. The page follows progress without restarting the task.`:lab.trainingExample?'The model is ready. Inspect a teaching prompt or preview an update on a training example.':'The model is ready, but no training example is available for One update.');
  }catch(error){$('engine-badge').textContent='Training service unavailable';$('engine-badge').className='badge offline';message(error.message,true);}
  finally{lab.busy=false;controls();}
}

for(const button of document.querySelectorAll('[data-tab]')){button.addEventListener('click',()=>changeTab(button.dataset.tab));button.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const names=['data','update','convergence','transfer'],index=names.indexOf(lab.tab),next=event.key==='Home'?0:event.key==='End'?3:(index+(event.key==='ArrowRight'?1:3))%4;changeTab(names[next]);$(`tab-${names[next]}`).focus();});}
for(const button of document.querySelectorAll('[data-stage]'))button.addEventListener('click',()=>{const stage=Number(button.dataset.stage);if(lab.trace&&stage<=lab.maxStage&&!lab.busy){lab.stage=stage;renderUpdate();}});
$('story-example').addEventListener('change',()=>{const example=lab.examples.find(item=>item.id===$('story-example').value);if(example){$('inspect-text').value=example.text;inspect(example.text,example.position);}});
$('inspect-button').addEventListener('click',()=>inspect());$('preview-update').addEventListener('click',previewUpdate);$('advance-update').addEventListener('click',advanceUpdate);
$('train-1000').addEventListener('click',()=>trainSteps(1000));$('train-5000').addEventListener('click',()=>trainSteps(5000));$('pause-training').addEventListener('click',pauseTraining);$('job-pause').addEventListener('click',pauseTraining);
$('reset-random').addEventListener('click',()=>resetModel('random'));$('reset-pretrained').addEventListener('click',()=>resetModel('pretrained'));
$('mode-live').addEventListener('click',()=>{pauseRecording();lab.mode='live';renderConvergence();});$('mode-recorded').addEventListener('click',()=>{lab.mode='recorded';renderConvergence();});
$('lm-performance').addEventListener('toggle',()=>renderPerformance('lm',lab.live,{visible:lab.mode!=='recorded'}));$('transfer-performance').addEventListener('toggle',()=>renderPerformance('transfer',lab.liveTransfer));
$('checkpoint-slider').addEventListener('input',()=>{pauseRecording();lab.checkpoint=Number($('checkpoint-slider').value);renderConvergence();});$('play-recording').addEventListener('click',playRecording);$('pause-recording').addEventListener('click',()=>{pauseRecording();message('Recorded playback paused at this saved checkpoint.');});$('recording-speed').addEventListener('change',()=>{if(lab.playing){clearTimeout(lab.timer);lab.timer=null;scheduleRecording();}});
$('refresh-recordings').addEventListener('click',refreshRecordings);$('refresh-transfer').addEventListener('click',refreshRecordings);$('transfer-seed').addEventListener('change',()=>{lab.seed=Number($('transfer-seed').value);renderTransfer();});$('predict-review').addEventListener('click',predictReview);$('review-text').addEventListener('input',controls);$('retry-training').addEventListener('click',boot);
$('generate-current').addEventListener('click',()=>generateText(false));$('generate-compare').addEventListener('click',()=>generateText(true));$('generation-prompt').addEventListener('input',controls);
$('generation-sliding').addEventListener('change',controls);
$('generation-pause').addEventListener('click',pauseGeneration);$('generation-stop').addEventListener('click',stopGeneration);$('generation-speed').addEventListener('change',()=>{const run=lab.generation;if(run?.active&&!run.stopped)for(const {player}of run.cards.values())player.setRate(generationRate());});
$('save-live').addEventListener('click',saveLive);$('save-transfer').addEventListener('click',saveLive);$('start-live-transfer').addEventListener('click',startLiveTransfer);$('refresh-live-transfer').addEventListener('click',refreshLiveTransfer);
$('transfer-train-500').addEventListener('click',()=>trainLiveTransfer(500));$('transfer-train-1000').addEventListener('click',()=>trainLiveTransfer(1000));$('pause-transfer').addEventListener('click',pauseTraining);
$('predict-live-review').addEventListener('click',predictLiveReview);$('live-review-text').addEventListener('input',controls);
window.addEventListener('load',()=>{if(lab.inspection)renderInspection();renderUpdate();math('transfer-formula',String.raw`\mathcal L=\mathcal L_{\mathrm{class}}+\lambda\mathcal L_{\mathrm{LM}}`);math('live-transfer-formula',String.raw`\mathcal L=\mathcal L_{\mathrm{class}}+\lambda\,\mathcal L_{\mathrm{LM}}`);});
// SEMINAR_GUIDE_START
// Presentation navigation owns no model state. Its only application dependency
// is the existing tab navigation function, which never invokes an optimizer.
export const SEMINAR_STOPS=Object.freeze([
  {id:'data',name:'Data & targets',description:'Text provides the next-token targets.',seconds:75,tab:'data',paper:'Pre-training objective',action:'Choose a story and an input token. Reveal the causal mask.',observe:'The next token supplies the label. A prediction cannot read future tokens.'},
  {id:'update',name:'One update',description:'One prediction error produces a weight update.',seconds:150,tab:'update',paper:'Forward, loss, backward, Adam',action:'Preview a real update. Follow the five stages, then apply it once.',observe:'Cross-entropy produces gradients. Apply commits one weight update.'},
  {id:'convergence',name:'Convergence',description:'Repeated updates change training and validation loss.',seconds:105,tab:'convergence',paper:'Repeated language-model training',action:'Inspect the recorded curve, then return to live mode. Run 1000 steps and watch the measured updates.',observe:'Compare training and validation loss. A recorded curve is a separate experiment.'},
  {id:'generation',name:'Use trained weights',description:'Use the learned weights to continue a story.',seconds:90,tab:'convergence',paper:'Inference after pre-training',action:'Compare random and current weights with the same story prompt.',observe:'Watch actual tokens arrive. EOS can stop early. Generation leaves weights unchanged.'},
  {id:'transfer',name:'Transfer',description:'Fine-tune the model to classify sentiment.',seconds:90,tab:'transfer',paper:'Supervised fine-tuning',action:'Inspect the existing pair and its source LM step, then classify a review. Optionally add 500 steps per branch.',observe:'Both branches use the same labeled batches. Compare measured results, including a possible random-baseline win.'},
  {id:'discussion',name:'Audience discussion',description:'Compare audience examples and discuss the model’s limits.',seconds:90,tab:'data',paper:'Model behavior and limitations',action:'Create a temporary invitation. Open a voluntarily submitted conversation.',observe:'The audience uses a frozen snapshot. Discuss what the model learned and where it fails.'}
]);

export function seminarRoute(hash){
  if(hash==='#seminar-panel')return'discussion';
  if(!String(hash).startsWith('#demo-'))return null;
  const requested=String(hash).slice(6);
  return SEMINAR_STOPS.some(stop=>stop.id===requested)?requested:'data';
}

export function createSeminarGuide({document:doc,window:win,navigateTab}){
  const byId=id=>doc.getElementById(id),guide=byId('presentation-guide');
  if(!guide)return null;
  const route=seminarRoute(win.location.hash);
  let enabled=true,stage=route||'data',navigating=false;
  const formatTime=seconds=>`${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')}`;
  const buttons=[...guide.querySelectorAll('[data-demo-stop]')];
  const historyState=()=>({...win.history.state,gptSeminar:{enabled,stage}});
  const currentUrl=()=>win.location.pathname+win.location.search+win.location.hash;
  const openAudience=()=>{
    if(enabled&&stage==='discussion'){
      const details=byId('seminar-panel')?.querySelector('details');
      if(details)details.open=true;
    }
  };
  function paint(){
    const index=SEMINAR_STOPS.findIndex(stop=>stop.id===stage),stop=SEMINAR_STOPS[index],start=SEMINAR_STOPS.slice(0,index).reduce((sum,item)=>sum+item.seconds,0);
    doc.body.classList.toggle('seminar-mode',enabled);doc.body.dataset.demoStage=stage;
    byId('seminar-route').hidden=!enabled;byId('seminar-cue').hidden=!enabled;
    byId('seminar-mode-toggle').textContent=enabled?'Workspace mode':'Presentation mode';
    byId('seminar-mode-toggle').setAttribute('aria-pressed',String(enabled));
    byId('seminar-mode-toggle').setAttribute('aria-label',enabled?'Switch to workspace mode':'Switch to presentation mode');
    byId('seminar-stage-number').textContent=`${String(index+1).padStart(2,'0')} / 06`;
    byId('seminar-stage-name').textContent=stop.name;
    byId('seminar-stage-description').textContent=stop.description;
    byId('seminar-stage-paper').textContent=stop.paper;
    byId('seminar-stage-time').textContent=`${formatTime(start)}–${formatTime(start+stop.seconds)} of 10:00`;
    byId('seminar-stage-action').textContent=stop.action;byId('seminar-stage-observe').textContent=stop.observe;
    for(const button of buttons){const selected=button.dataset.demoStop===stage;button.classList.toggle('active',selected);if(selected)button.setAttribute('aria-current','step');else button.removeAttribute('aria-current');}
    openAudience();
  }
  function select(next,{writeHash=true,scroll=false,navigate=true}={}){
    const nextStage=SEMINAR_STOPS.some(stop=>stop.id===next)?next:'data';
    stage=nextStage;
    if(navigate){navigating=true;try{navigateTab(SEMINAR_STOPS.find(stop=>stop.id===stage).tab);}finally{navigating=false;}}
    paint();
    if(writeHash&&enabled&&win.location.hash!==`#demo-${stage}`)win.history.pushState(historyState(),'',`#demo-${stage}`);
    if(scroll)guide.scrollIntoView({block:'start',behavior:win.matchMedia?.('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});
  }
  function toggle(){
    enabled=!enabled;
    try{win.localStorage.setItem('gpt-seminar-mode',enabled?'presentation':'workspace');}catch{}
    if(enabled)select(stage);else{paint();win.history.pushState(historyState(),'',win.location.pathname+win.location.search);}
  }
  function followTab(tab){
    if(navigating)return;
    const same=SEMINAR_STOPS.find(stop=>stop.id===stage)?.tab===tab;
    if(enabled&&!same)select(tab,{navigate:false});
    else if(!enabled){select(tab,{writeHash:false,navigate:false});win.history.replaceState(historyState(),'',currentUrl());}
  }
  const hashChanged=()=>{
    const next=seminarRoute(win.location.hash);
    if(next!==null){enabled=true;select(next,{writeHash:false});}
    else if(win.location.hash===''){
      const saved=win.history.state?.gptSeminar;
      enabled=typeof saved?.enabled==='boolean'?saved.enabled:true;
      select(saved?.stage||'data',{writeHash:false});
    }
  };
  byId('seminar-mode-toggle').addEventListener('click',toggle);
  for(const button of buttons)button.addEventListener('click',()=>select(button.dataset.demoStop,{scroll:true}));
  win.addEventListener('hashchange',hashChanged);win.addEventListener('popstate',hashChanged);win.addEventListener('load',openAudience);
  if(win.MutationObserver&&byId('seminar-panel'))new win.MutationObserver(openAudience).observe(byId('seminar-panel'),{childList:true});
  select(stage,{writeHash:false});
  // Remember the initial / entry as well as explicit deep links. Browser Back
  // can then restore Data at /, or the workspace view if that entry was created
  // by the mode toggle, without guessing from the last selected stage.
  win.history.replaceState(historyState(),'',currentUrl());
  return{followTab};
}
// SEMINAR_GUIDE_END

tokenHistoryView=createTokenHistoryView();changeTab('data');seminarGuide=createSeminarGuide({document,window,navigateTab:changeTab});controls();boot();
