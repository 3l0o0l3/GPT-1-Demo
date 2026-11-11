const $=id=>document.getElementById(id);
const worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
let serial=0, pending=new Map(), trace=null, stage=0, committed=false, busy=false;
let running=false, currentView='step', history=[], taskHistory=[], metadata=null, latestEval=null, beforeFinetuning=null;
const fmt=(v,n=3)=>Number.isFinite(v)?v.toFixed(n):'—';
const metric=v=>v>0&&v<.001?'<0.001':fmt(v);
const probabilityText=v=>v>0&&v<.0001?v.toExponential(2):fmt(v,4);
const probabilityMath=v=>probabilityText(v).replace(/e([+-]?\d+)/,'\\times 10^{$1}');
const pct=v=>`${(100*v).toFixed(1)}%`;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function request(type,payload={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});worker.postMessage({id,type,payload});});}
worker.onmessage=({data})=>{const p=pending.get(data.id);if(!p)return;pending.delete(data.id);data.type==='error'?p.reject(new Error(data.message)):p.resolve(data.data);};
worker.onerror=e=>{status('The model could not start: '+e.message,true);for(const p of pending.values())p.reject(new Error(e.message));pending.clear();setBusy(false);};
function status(text,error=false){$('app-status').textContent=text;$('app-status').classList.toggle('error',error);}
function setBusy(value){busy=value;document.querySelectorAll('[data-model-action]').forEach(el=>el.disabled=value);$('train-toggle').disabled=value&&!running;$('restore-base').disabled=value||!beforeFinetuning;}
function math(id,latex){const node=$(id);if(globalThis.katex)katex.render(latex,node,{throwOnError:false,displayMode:true});else node.textContent=latex;}
function record(evaluation,task='lm'){
 latestEval=evaluation;
 const arr=task==='lm'?history:taskHistory;
 if(arr.at(-1)?.step===evaluation.step)arr[arr.length-1]=evaluation;else arr.push(evaluation);
 $('step-count').textContent=String(evaluation.step);
 if(task==='lm'){$('train-loss').textContent=metric(evaluation.trainLoss);$('val-loss').textContent=metric(evaluation.valLoss);drawChart();}
 else{$('task-loss').textContent=metric(evaluation.valLoss);$('task-accuracy').textContent=Number.isFinite(evaluation.accuracy)?pct(evaluation.accuracy):'—';}
}
function bars(id,result,{before=null,limit=6}={}){
 let items=[...result.probabilities].sort((a,b)=>b.p-a.p).slice(0,limit);
 if(!items.some(x=>x.token===result.target)){const target=result.probabilities.find(x=>x.token===result.target);if(target)items[items.length-1]=target;}
 const maximum=Math.max(...items.map(x=>x.p),...(before?.probabilities.map(x=>x.p)||[]));
 const scale=id==='task-probs'?1:Math.max(.1,Math.ceil(maximum*10)/10);
 if(id==='step-probs')$('step-scale').textContent=`Bar scale: 0–${Math.round(scale*100)}%`;
 $(id).innerHTML=items.map(({token,p})=>{
  const old=before?.probabilities.find(x=>x.token===token)?.p;
  return `<div class="prob-row ${token===result.target?'is-target':''}"><div class="prob-label">${esc(token)}${token===result.target?'<span class="correct-label"> target</span>':''}</div><div class="prob-track" role="img" aria-label="${esc(token)} ${pct(p)}"><div class="prob-fill" style="width:${p/scale*100}%"></div>${old!==undefined?`<span class="before-tick" style="left:${old/scale*100}%" title="Before: ${pct(old)}"></span>`:''}</div><div class="prob-value">${pct(p)}</div></div>`;
 }).join('')+(id==='probe-probs'?`<span class="small-note">Bar scale: 0–${Math.round(scale*100)}% · top candidates shown</span>`:'');
}
function attention(result){
 const weights=result.attention?.at(-1)||[];
 return `<div class="detail-heading">Where the last token attends</div><div class="attention-row">${result.tokens.map((t,i)=>`<span class="attention-token" style="--strength:${Math.max(.07,weights[i]||0)}">${esc(t)}<small>${pct(weights[i]||0)}</small></span>`).join('')}</div><p class="small-note">First attention head · one query position. Attention describes information mixing; it is not a complete explanation of the prediction.</p>`;
}
function renderStage(){
 if(!trace)return;
 document.querySelectorAll('[data-stage]').forEach(el=>{const n=Number(el.dataset.stage);el.classList.toggle('active',n===stage);el.classList.toggle('complete',n<stage);el.setAttribute('aria-pressed',String(n===stage));el.setAttribute('aria-current',n===stage?'step':'false');el.disabled=n>stage+1||busy;});
 const b=trace.before,a=trace.after,w=trace.weight;
 const view=stage>=3?a:b;
 $('step-sample').textContent=trace.sample.text;
 $('step-tokens').innerHTML=b.tokens.map(t=>`<span class="token">${esc(t)}</span>`).join('')+`<span class="token-arrow">→</span><span class="token target-token">${esc(b.target)}</span>`;
 bars('step-probs',view,{before:stage>=3?b:null});
 const title=['Predict the next word','Measure the prediction error','Send the error backward','Update the model parameters','Compare the same example'][stage];
 const descriptions=[
  `The input ends at “${b.tokens.at(-1)}”. The model assigns “${b.target}” ${pct(b.targetProbability)} probability. The observed next word supplies the training target.`,
  `This example contributes −log(${probabilityText(b.targetProbability)}) = ${fmt(b.loss)} nats. Giving the correct word more probability makes this loss smaller.`,
  `The gradient says how each parameter affects the batch loss. A positive gradient points toward increasing loss; Adam uses gradients and its running moments to choose an update.`,
  `One real Adam update has now been applied to the embeddings, attention and feed-forward weights. Gradients are clipped at norm 1 before Adam. The trace follows one parameter with a large raw gradient.`,
  `For this same example, the target probability changed from ${pct(b.targetProbability)} to ${pct(a.targetProbability)}. A batch update can help some examples and hurt others; longer-term validation matters.`
 ];
 $('step-explanation').innerHTML=`<h2>${title}</h2><p>${esc(descriptions[stage])}</p>`;
 if(stage===0){math('step-formula',String.raw`p=\operatorname{softmax}(h_L W_e^{\mathsf T})`);$('step-detail').innerHTML=attention(b);}
 if(stage===1){math('step-formula',String.raw`J_{\mathrm{LM}}=-\log p_y=-\log(${probabilityMath(b.targetProbability)})=${fmt(b.loss)}`);$('step-detail').innerHTML=`<div class="metric-grid"><div class="metric-item"><span>This example · last token</span><strong>${fmt(b.loss)} <small>nats</small></strong></div><div class="metric-item"><span>Training batch · mean token loss</span><strong>${fmt(trace.lmLoss)} <small>nats</small></strong></div></div><p class="small-note">The optimizer uses the batch mean, not only the displayed token.</p>`;}
 if(stage===2){math('step-formula',String.raw`\frac{\partial J}{\partial z_c}=p_c-\mathbf{1}[c=y]`);$('step-detail').innerHTML=`<div class="metric-grid"><div class="metric-item"><span>Target logit gradient · this token</span><strong>${fmt(b.targetProbability-1)}</strong></div><div class="metric-item"><span>All-parameter gradient norm · batch</span><strong>${fmt(trace.gradientNorm)}</strong></div></div><p class="small-note">Backpropagation applies the chain rule through the complete Transformer.</p>`;}
 if(stage===3){math('step-formula',String.raw`\theta_{t+1}=\theta_t-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}`);$('step-detail').innerHTML=`<div class="weight-label">${esc(w.name)} · actual Adam update</div><div class="weight-update"><span><small>Before</small>${fmt(w.before,5)}</span><span class="update-arrow">→</span><span><small>After</small>${fmt(w.after,5)}</span></div><p class="small-note">Gradient ${fmt(w.gradient,5)} · parameter change ${fmt(w.after-w.before,5)} · batch update norm ${fmt(trace.updateNorm,5)}</p>`;}
 if(stage===4){math('step-formula',String.raw`p_y:\ ${fmt(b.targetProbability,4)}\longrightarrow ${fmt(a.targetProbability,4)}`);$('step-detail').innerHTML=`<div class="metric-grid"><div class="metric-item"><span>Example loss · before</span><strong>${fmt(b.loss)}</strong></div><div class="metric-item"><span>Example loss · after</span><strong>${fmt(a.loss)}</strong></div></div><p class="small-note">Thin markers show probabilities before the update. Before and after share the displayed bar scale.</p>`;}
 $('next-stage').textContent=['Show the loss','Trace the gradient','Apply this update','Compare predictions','Next training example'][stage];
}
async function prepare(){trace=await request('prepare',{task:'lm',learningRate:Number($('lr-select').value)});stage=0;committed=false;renderStage();status('A real training step is ready. No parameter update has been applied yet.');}
async function moveStage(next){
 if(busy||!trace)return;
 setBusy(true);
 try{
  if(next===5){await prepare();return;}
  if(next>=3&&!committed){const r=await request('commit');committed=true;record(r.evaluation);}
  stage=next;renderStage();status(committed?'Update applied. Inspect the same example before and after.':'Model parameters are unchanged until you apply the update.');
 }catch(e){status(e.message,true);}finally{setBusy(false);renderStage();}
}
const probes={'a cat is an':'animal','a rose is a':'plant','a car is a':'vehicle'};
async function refreshProbe(){const text=$('probe-select').value;const r=await request('inspect',{text,target:probes[text],task:'lm'});bars('probe-probs',r);return r;}
async function refreshTask(){
 const r=await request('inspect',{task:'classify',text:$('task-input').value});bars('task-probs',r,{limit:3});return r;
}
async function runTraining(task,count=Infinity){
 if(running){running=false;status('Pausing after the current small batch…');return;}
 if(busy)return;
 running=true;setBusy(true);$('train-toggle').textContent=task==='lm'?'Pause training':'Train continuously';
 let completed=0;
 try{
  if(task==='classify'&&!beforeFinetuning)beforeFinetuning={state:await request('snapshot'),history:[...history],step:latestEval.step};
  if(task==='lm')beforeFinetuning=null;
  while(running&&completed<count){
   const n=Math.min(10,count-completed);
   const r=await request('train',{task,steps:n,learningRate:Number($('lr-select').value),lambda:Number($('lambda-select').value)});
   completed+=n;if(r.lmEvaluation)record(r.lmEvaluation);record(r.evaluation,task);
   if(task==='lm'){
    await refreshProbe();$('learn-status').textContent=`${completed} live updates in this run. Validation sentences are held out from training. The curve may fluctuate.`;
   }else{
    await refreshTask();$('task-status').textContent=`${completed} fine-tuning updates in this run. Accuracy is measured on held-out synthetic sentences.`;
   }
   status(`${task==='lm'?'Language-model training':'Supervised fine-tuning'} · step ${r.evaluation.step}`);
   await new Promise(resolve=>setTimeout(resolve,120));
  }
  trace=null;
 }catch(e){status(e.message,true);}finally{
  running=false;setBusy(false);$('train-toggle').textContent='Train continuously';
  if(currentView==='step')await safePrepare();
 }
}
async function safePrepare(){setBusy(true);try{await prepare();}catch(e){status(e.message,true);}finally{setBusy(false);renderStage();}}
async function switchView(view){
 if(busy)return;
 currentView=view;
 document.querySelectorAll('[data-view]').forEach(el=>{const active=el.dataset.view===view;el.classList.toggle('active',active);el.setAttribute('aria-pressed',String(active));});
 for(const id of ['step','learn','task'])$('view-'+id).hidden=id!==view;
 setBusy(true);
 try{
  if(view==='step')await prepare();
  else if(view==='learn'){record(await request('evaluate',{task:'lm'}));await refreshProbe();if(beforeFinetuning)$('learn-status').textContent=latestEval.step>beforeFinetuning.step?'The marked region follows supervised fine-tuning. These are still language-model losses: task learning can change next-word prediction.':'Restored to the start of fine-tuning. This is the original language-model training history.';requestAnimationFrame(drawChart);}
  else{record(await request('evaluate',{task:'classify'}),'classify');await refreshTask();mathTask();}
 }catch(e){status(e.message,true);}finally{setBusy(false);renderStage();}
}
function mathTask(){if($('task-formula'))math('task-formula',String.raw`J=J_{\mathrm{class}}+\lambda J_{\mathrm{LM}}`);}
function drawChart(){
 const svg=$('loss-chart');if(!svg||$('view-learn').hidden)return;
 const width=Math.max(280,svg.parentElement.clientWidth),height=300;
 const margin={l:52,r:18,t:20,b:48},iw=width-margin.l-margin.r,ih=height-margin.t-margin.b;
 svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
 const maxX=Math.max(10,...history.map(d=>d.step));
 const maxY=Math.max(1,...history.flatMap(d=>[d.trainLoss,d.valLoss]))*1.12;
 const x=v=>margin.l+v/maxX*iw,y=v=>margin.t+ih-v/maxY*ih;
 let markup=`<title>Training and validation negative log-likelihood in nats per token</title><desc>Each point is measured from this model. Lower values indicate a better fit to the observed next tokens.</desc>`;
 for(let i=0;i<=4;i++){
  const val=i*maxY/4,yy=y(val);markup+=`<line class="chart-grid" x1="${margin.l}" y1="${yy}" x2="${width-margin.r}" y2="${yy}"/><text class="chart-label" x="${margin.l-9}" y="${yy+4}" text-anchor="end">${fmt(val,1)}</text>`;
 }
 for(let i=0;i<=4;i++){const val=i*maxX/4;markup+=`<text class="chart-label" x="${x(val)}" y="${height-25}" text-anchor="middle">${Math.round(val)}</text>`;}
 for(const [key,cls] of [['trainLoss','train-series'],['valLoss','val-series']]){
  const points=history.map(d=>`${x(d.step)},${y(d[key])}`).join(' ');
  markup+=`<polyline class="${cls}" points="${points}" fill="none"/>`;
  for(const d of history)markup+=`<circle class="${cls}" cx="${x(d.step)}" cy="${y(d[key])}" r="2.4"><title>Step ${d.step}: ${key==='trainLoss'?'training':'validation'} loss ${fmt(d[key])}</title></circle>`;
 }
 if(beforeFinetuning&&maxX>beforeFinetuning.step){const xx=x(beforeFinetuning.step);markup+=`<line class="phase-marker" x1="${xx}" x2="${xx}" y1="${margin.t}" y2="${margin.t+ih}"/><text class="chart-label" x="${Math.min(xx+7,width-130)}" y="${margin.t+15}">Fine-tuning begins</text>`;}
 markup+=`<text class="chart-label axis-title" x="${margin.l}" y="12">Loss · nats / token</text><text class="chart-label axis-title" x="${margin.l+iw/2}" y="${height-4}" text-anchor="middle">Optimizer updates</text>`;
 svg.innerHTML=markup;
}
document.querySelectorAll('[data-view]').forEach(el=>el.addEventListener('click',()=>switchView(el.dataset.view)));
document.querySelectorAll('[data-stage]').forEach(el=>el.addEventListener('click',()=>moveStage(Number(el.dataset.stage))));
$('next-stage').addEventListener('click',()=>moveStage(stage+1));
$('new-step').addEventListener('click',safePrepare);
$('train-toggle').addEventListener('click',()=>runTraining('lm'));
$('train-100').addEventListener('click',()=>runTraining('lm',100));
$('finetune-100').addEventListener('click',()=>runTraining('classify',800));
$('restore-base').addEventListener('click',async()=>{
 if(busy||!beforeFinetuning)return;setBusy(true);
 try{await request('restore',beforeFinetuning.state);history=[...beforeFinetuning.history];taskHistory=[];trace=null;record(await request('evaluate',{task:'classify'}),'classify');await refreshTask();$('task-status').textContent='Restored the exact model and optimizer state from before fine-tuning. You can compare a different auxiliary loss weight.';status('Model restored to the start of fine-tuning.');}
 catch(e){status(e.message,true);}finally{setBusy(false);}
});
$('probe-select').addEventListener('change',()=>refreshProbe().catch(e=>status(e.message,true)));
$('predict-task').addEventListener('click',()=>refreshTask().catch(e=>{$('task-status').textContent=e.message;status(e.message,true);}));
$('task-input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!busy)$('predict-task').click();});
$('reset-model').addEventListener('click',async()=>{
 if(busy)return;setBusy(true);
 try{const r=await request('reset',{seed:42});history=[];taskHistory=[];beforeFinetuning=null;record(r.evaluation);record(await request('evaluate',{task:'classify'}),'classify');await refreshProbe();await refreshTask();trace=null;stage=0;status('Reset to the same random initialization. Training history cleared.');$('learn-status').textContent='Fresh random weights. Start training to collect a learning curve.';$('task-status').textContent='The classification head has not been fine-tuned.';if(currentView==='step')await prepare();}
 catch(e){status(e.message,true);}finally{setBusy(false);renderStage();}
});
new ResizeObserver(()=>drawChart()).observe($('loss-chart').parentElement);
async function init(){
 document.querySelectorAll('#next-stage,#new-step,#train-toggle,#train-100,#reset-model,#lr-select,#finetune-100,#restore-base,#lambda-select,#predict-task,#task-input,[data-view]').forEach(el=>el.dataset.modelAction='');
 setBusy(true);status('Initializing the small Transformer…');
 try{
  const r=await request('init',{seed:42});metadata=r.metadata;record(r.evaluation);
  $('architecture').textContent=`${typeof metadata.architecture==='string'?metadata.architecture:JSON.stringify(metadata.architecture)} · ${metadata.parameterCount.toLocaleString()} parameters · ${metadata.vocab.length} tokens. ${metadata.corpusDescription} ${metadata.trainExamples} training examples; ${metadata.validationExamples} held-out examples.`;
  const vocab=document.createElement('p');vocab.className='small-note';vocab.textContent='Vocabulary: '+metadata.vocab.join(', ');$('architecture').after(vocab);
  await prepare();await refreshProbe();mathTask();
 }catch(e){status(e.message,true);}finally{setBusy(false);renderStage();}
}
init();
