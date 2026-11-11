// Small, real GPT-style teaching model. No network access or ML dependencies.
// All visualized values come from forward/backward passes through this model.
const VERSION = 1, D = 12, HEADS = 2, HD = D / HEADS, FF = 24, MAX = 12;
const CLASSES = ['animal', 'plant', 'vehicle'];
const ENTITIES = [
  ['cat','animal','sleeps'],['dog','animal','runs'],['bird','animal','flies'],
  ['rose','plant','grows'],['tulip','plant','grows'],['tree','plant','grows'],
  ['car','vehicle','moves'],['bus','vehicle','moves'],['train','vehicle','moves'],
];
const split = text => String(text).toLowerCase().match(/\[[a-z]+\]|[a-z]+|[.!?]/g) || [];
const LM_TRAIN=[], LM_VAL=[], CLS_TRAIN=[], CLS_VAL=[];
for(const [entity,label,verb] of ENTITIES){
  const article=label==='animal'?'an':'a';
  for(const text of [`a ${entity} is ${article} ${label} .`,`the ${entity} is ${article} ${label} .`,`the ${entity} ${verb} .`,`i see the ${entity} .`,`the ${entity} is here .`,`i like a ${entity} .`,`look at this ${entity} .`]) LM_TRAIN.push({text,label});
  LM_VAL.push({text:`this ${entity} is ${article} ${label} .`,label});
  for(const text of [`the ${entity} ${verb} .`,`i see the ${entity} .`,`a ${entity} is here .`]) CLS_TRAIN.push({text,label});
  for(const text of [`this ${entity} is here .`,`i like the ${entity} .`]) CLS_VAL.push({text,label});
}
const VOCAB=['[start]','[end]','[extract]',...new Set([...LM_TRAIN,...LM_VAL,...CLS_TRAIN,...CLS_VAL].flatMap(e=>split(e.text)))];
const INDEX=new Map(VOCAB.map((v,i)=>[v,i]));
const tiny=1e-9;

function rng(seed){let state=seed>>>0;return {next(){state=(Math.imul(1664525,state)+1013904223)>>>0;return state/4294967296;},get(){return state;},set(s){state=s>>>0;}};}
function node(data,rows=1,cols=data.length){return {data:data instanceof Float64Array?data:new Float64Array(data),grad:new Float64Array(data.length),rows,cols};}
function op(tape,data,rows,cols,back){const n=node(data,rows,cols);if(tape)tape.push(()=>back(n.grad));return n;}
function add(tape,a,b){return op(tape,a.data.map((x,i)=>x+b.data[i]),a.rows,a.cols,g=>{for(let i=0;i<g.length;i++){a.grad[i]+=g[i];b.grad[i]+=g[i];}});}
function matmul(tape,a,b){
  const m=a.rows,k=a.cols,n=b.cols,out=new Float64Array(m*n);
  for(let i=0;i<m;i++)for(let q=0;q<k;q++){const v=a.data[i*k+q];for(let j=0;j<n;j++)out[i*n+j]+=v*b.data[q*n+j];}
  return op(tape,out,m,n,g=>{for(let i=0;i<m;i++)for(let q=0;q<k;q++){let s=0;for(let j=0;j<n;j++){const x=g[i*n+j];s+=x*b.data[q*n+j];b.grad[q*n+j]+=a.data[i*k+q]*x;}a.grad[i*k+q]+=s;}});
}
function transpose(tape,a){const out=new Float64Array(a.data.length);for(let i=0;i<a.rows;i++)for(let j=0;j<a.cols;j++)out[j*a.rows+i]=a.data[i*a.cols+j];return op(tape,out,a.cols,a.rows,g=>{for(let i=0;i<a.rows;i++)for(let j=0;j<a.cols;j++)a.grad[i*a.cols+j]+=g[j*a.rows+i];});}
function bias(tape,a,b){return op(tape,a.data.map((x,i)=>x+b.data[i%a.cols]),a.rows,a.cols,g=>{for(let i=0;i<g.length;i++){a.grad[i]+=g[i];b.grad[i%a.cols]+=g[i];}});}
function gelu(tape,a){const c=Math.sqrt(2/Math.PI);return op(tape,a.data.map(x=>0.5*x*(1+Math.tanh(c*(x+0.044715*x*x*x)))),a.rows,a.cols,g=>{for(let i=0;i<g.length;i++){const x=a.data[i],t=Math.tanh(c*(x+0.044715*x*x*x));a.grad[i]+=g[i]*(0.5*(1+t)+0.5*x*(1-t*t)*c*(1+3*0.044715*x*x));}});}
function norm(tape,a,gamma,beta){
  const d=a.cols,T=a.rows,y=new Float64Array(a.data.length),xhat=new Float64Array(y.length),inv=new Float64Array(T);
  for(let t=0;t<T;t++){let mean=0;for(let j=0;j<d;j++)mean+=a.data[t*d+j]/d;let v=0;for(let j=0;j<d;j++)v+=(a.data[t*d+j]-mean)**2/d;inv[t]=1/Math.sqrt(v+1e-5);for(let j=0;j<d;j++){const i=t*d+j;xhat[i]=(a.data[i]-mean)*inv[t];y[i]=gamma.data[j]*xhat[i]+beta.data[j];}}
  return op(tape,y,T,d,g=>{for(let t=0;t<T;t++){let sg=0,sgx=0;for(let j=0;j<d;j++){const i=t*d+j,dx=g[i]*gamma.data[j];sg+=dx;sgx+=dx*xhat[i];gamma.grad[j]+=g[i]*xhat[i];beta.grad[j]+=g[i];}for(let j=0;j<d;j++){const i=t*d+j;a.grad[i]+=inv[t]*(g[i]*gamma.data[j]-(sg+xhat[i]*sgx)/d);}}});
}
function gather(tape,ids,embedding,position){const out=new Float64Array(ids.length*D);for(let t=0;t<ids.length;t++)for(let j=0;j<D;j++)out[t*D+j]=embedding.data[ids[t]*D+j]+position.data[t*D+j];return op(tape,out,ids.length,D,g=>{for(let t=0;t<ids.length;t++)for(let j=0;j<D;j++){embedding.grad[ids[t]*D+j]+=g[t*D+j];position.grad[t*D+j]+=g[t*D+j];}});}
function attention(tape,q,k,v){
 const T=q.rows,out=new Float64Array(T*D),weights=new Float64Array(HEADS*T*T),scale=1/Math.sqrt(HD);
 for(let h=0;h<HEADS;h++)for(let i=0;i<T;i++){
  let max=-Infinity;for(let j=0;j<=i;j++){let s=0;for(let a=0;a<HD;a++)s+=q.data[i*D+h*HD+a]*k.data[j*D+h*HD+a];weights[(h*T+i)*T+j]=s*scale;max=Math.max(max,s*scale);}
  let sum=0;for(let j=0;j<=i;j++){const w=(h*T+i)*T+j;weights[w]=Math.exp(weights[w]-max);sum+=weights[w];}
  for(let j=0;j<=i;j++){const w=(h*T+i)*T+j;weights[w]/=sum;for(let a=0;a<HD;a++)out[i*D+h*HD+a]+=weights[w]*v.data[j*D+h*HD+a];}
 }
 const result=op(tape,out,T,D,g=>{
  for(let h=0;h<HEADS;h++)for(let i=0;i<T;i++){
   const dw=new Float64Array(i+1);let dot=0;
   for(let j=0;j<=i;j++){const w=weights[(h*T+i)*T+j];for(let a=0;a<HD;a++){dw[j]+=g[i*D+h*HD+a]*v.data[j*D+h*HD+a];v.grad[j*D+h*HD+a]+=w*g[i*D+h*HD+a];}dot+=w*dw[j];}
   for(let j=0;j<=i;j++){const ds=weights[(h*T+i)*T+j]*(dw[j]-dot)*scale;for(let a=0;a<HD;a++){q.grad[i*D+h*HD+a]+=ds*k.data[j*D+h*HD+a];k.grad[j*D+h*HD+a]+=ds*q.data[i*D+h*HD+a];}}
  }
 });
 result.attention=Array.from({length:T},(_,i)=>Array.from(weights.slice(i*T,(i+1)*T)));return result;
}
function last(tape,a){const off=(a.rows-1)*a.cols;return op(tape,a.data.slice(off),1,a.cols,g=>{for(let j=0;j<a.cols;j++)a.grad[off+j]+=g[j];});}
function softmax(values){const max=Math.max(...values),p=values.map(v=>Math.exp(v-max)),s=p.reduce((a,b)=>a+b,0);return p.map(v=>v/s);}
function ce(logits,targets,scale=1){let loss=0;for(let t=0;t<targets.length;t++){const z=Array.from(logits.data.slice(t*logits.cols,(t+1)*logits.cols)),p=softmax(z);loss-=Math.log(Math.max(1e-300,p[targets[t]]))/targets.length;for(let j=0;j<logits.cols;j++)logits.grad[t*logits.cols+j]+=scale*(p[j]-(j===targets[t]?1:0))/targets.length;}return loss;}
function encode(text){const tokens=split(text);if(!tokens.length)throw new Error('Enter a sentence using the demo vocabulary.');const unknown=tokens.filter(t=>!INDEX.has(t));if(unknown.length)throw new Error(`Unknown word: ${unknown[0]}. Try cat, dog, bird, rose, tulip, tree, car, bus or train.`);if(tokens.length>MAX-2)throw new Error(`Use at most ${MAX-2} words and punctuation tokens.`);return tokens.map(t=>INDEX.get(t));}
function kind(task){if(task==='lm')return 'lm';if(['classify','classification','task'].includes(task))return 'classify';throw new Error('Task must be lm or classify.');}

export function createLab({seed=42,learningRate=0.01}={}){
 let random,params,steps,adamSteps,initialSeed=seed;
 const init=(nextSeed=initialSeed)=>{
  initialSeed=Number(nextSeed)>>>0;random=rng(initialSeed);params={};steps=0;adamSteps=0;
  const parameter=(name,r,c,mode='random')=>{const a=new Float64Array(r*c);for(let i=0;i<a.length;i++){if(mode==='one')a[i]=1;else if(mode==='random'){const u=Math.max(tiny,random.next()),v=random.next();a[i]=Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)*0.06;}}const p=node(a,r,c);p.m=new Float64Array(a.length);p.v=new Float64Array(a.length);params[name]=p;return p;};
  parameter('tokenEmbedding',VOCAB.length,D);parameter('positionEmbedding',MAX,D);
  for(const s of ['query','key','value','attentionOutput'])parameter(s,D,D);
  parameter('norm1Scale',1,D,'one');parameter('norm1Bias',1,D,'zero');
  parameter('ff1',D,FF);parameter('ff1Bias',1,FF,'zero');parameter('ff2',FF,D);parameter('ff2Bias',1,D,'zero');
  parameter('norm2Scale',1,D,'one');parameter('norm2Bias',1,D,'zero');
  parameter('classHead',D,CLASSES.length);parameter('classBias',1,CLASSES.length,'zero');
 };
 init(seed);
 const body=(ids,tape)=>{const p=params,x=gather(tape,ids,p.tokenEmbedding,p.positionEmbedding),a=attention(tape,matmul(tape,x,p.query),matmul(tape,x,p.key),matmul(tape,x,p.value));const h=norm(tape,add(tape,x,matmul(tape,a,p.attentionOutput)),p.norm1Scale,p.norm1Bias);const f=bias(tape,matmul(tape,gelu(tape,bias(tape,matmul(tape,h,p.ff1),p.ff1Bias)),p.ff2),p.ff2Bias);const y=norm(tape,add(tape,h,f),p.norm2Scale,p.norm2Bias);return {hidden:y,attention:a.attention};};
 const forward=(ids,task,tape)=>{const b=body(ids,tape);const logits=task==='lm'?matmul(tape,b.hidden,transpose(tape,params.tokenEmbedding)):bias(tape,matmul(tape,last(tape,b.hidden),params.classHead),params.classBias);return {...b,logits};};
 const lossPass=(example,task,scale=0)=>{
  const tape=scale===0?null:[],words=encode(example.text);let result,loss;
  if(task==='lm'){const ids=[INDEX.get('[start]'),...words],targets=[...words,INDEX.get('[end]')];result=forward(ids,'lm',tape);loss=ce(result.logits,targets,scale);}
  else {result=forward([INDEX.get('[start]'),...words,INDEX.get('[extract]')],'classify',tape);loss=ce(result.logits,[CLASSES.indexOf(example.label)],scale);}
  if(tape)for(let i=tape.length-1;i>=0;i--)tape[i]();return {loss,result};
 };
 const inspect=({task='lm',text=task==='lm'?'a cat is an':'the cat sleeps .',target}={})=>{
  task=kind(task);const words=encode(text);let ids=[INDEX.get('[start]'),...words];if(task==='classify')ids.push(INDEX.get('[extract]'));
  if(ids.length>MAX)throw new Error('Sequence is longer than 12 tokens.');const f=forward(ids,task,null),logits=Array.from(f.logits.data.slice((f.logits.rows-1)*f.logits.cols)),p=softmax(logits),labels=task==='lm'?VOCAB:CLASSES;
  if(target===undefined)target=task==='lm'?'animal':(ENTITIES.find(e=>split(text).includes(e[0]))?.[1]||'animal');
  if(!labels.includes(target))throw new Error(`Target must be ${task==='lm'?'a vocabulary token':CLASSES.join(', ')}.`);
  const probabilities=labels.map((token,i)=>({token,p:p[i]})).sort((a,b)=>b.p-a.p),targetProbability=p[labels.indexOf(target)];
  return {task,text,tokens:ids.map(i=>VOCAB[i]),target,probabilities,loss:-Math.log(Math.max(1e-300,targetProbability)),logits,attention:f.attention,prediction:probabilities[0].token,targetProbability};
 };
 const evaluate=({task='lm'}={})=>{task=kind(task);const train=task==='lm'?LM_TRAIN:CLS_TRAIN,val=task==='lm'?LM_VAL:CLS_VAL;let correct=0;const score=(data,isVal)=>data.reduce((sum,e)=>{const {loss,result}=lossPass(e,task);if(isVal&&task!=='lm'){const z=Array.from(result.logits.data);if(z.indexOf(Math.max(...z))===CLASSES.indexOf(e.label))correct++;}return sum+loss/data.length;},0);const trainLoss=score(train,false),valLoss=score(val,true);return {trainLoss,valLoss,accuracy:task==='lm'?null:correct/val.length,step:steps};};
 const step=({task='lm',learningRate:rate=learningRate,lambda=0.1}={})=>{
  task=kind(task);rate=Number(rate);lambda=Number(lambda);if(!Number.isFinite(rate)||rate<=0||rate>1)throw new Error('Learning rate must be greater than 0 and at most 1.');if(!Number.isFinite(lambda)||lambda<0||lambda>2)throw new Error('Auxiliary weight must be between 0 and 2.');
  const data=task==='lm'?LM_TRAIN:CLS_TRAIN,batch=Array.from({length:4},()=>data[Math.floor(random.next()*data.length)]),e=batch[0];
  let sample;if(task==='lm'){const words=split(e.text);const cut=Math.max(1,words.length-2);sample={text:words.slice(0,cut).join(' '),target:words[cut]};}else sample={text:e.text,target:e.label};
  const before=inspect({task,...sample});for(const p of Object.values(params))p.grad.fill(0);
  let lmLoss=0,taskLoss=task==='lm'?null:0;
  for(const ex of batch){if(task==='lm')lmLoss+=lossPass(ex,'lm',1/batch.length).loss/batch.length;else {taskLoss+=lossPass(ex,'classify',1/batch.length).loss/batch.length;lmLoss+=lossPass(ex,'lm',lambda/batch.length).loss/batch.length;}}
  let gradSq=0;for(const p of Object.values(params))for(const g of p.grad)gradSq+=g*g;const gradientNorm=Math.sqrt(gradSq),clip=Math.min(1,1/(gradientNorm+tiny));
  // Pick the largest-gradient parameter for a numerically visible trace; no fabricated values.
  let selectedName='',selectedIndex=0,best=-1;for(const [name,p]of Object.entries(params))for(let i=0;i<p.data.length;i++)if(Math.abs(p.grad[i])>best){best=Math.abs(p.grad[i]);selectedName=name;selectedIndex=i;}
  const selected=params[selectedName],weight={name:`${selectedName}[${selectedIndex}]`,before:selected.data[selectedIndex],gradient:selected.grad[selectedIndex],after:0};
  adamSteps++;let updateSq=0;for(const p of Object.values(params))for(let i=0;i<p.data.length;i++){const g=p.grad[i]*clip;p.m[i]=0.9*p.m[i]+0.1*g;p.v[i]=0.999*p.v[i]+0.001*g*g;const delta=rate*(p.m[i]/(1-0.9**adamSteps))/(Math.sqrt(p.v[i]/(1-0.999**adamSteps))+1e-8);p.data[i]-=delta;updateSq+=delta*delta;}
  steps++;weight.after=selected.data[selectedIndex];const after=inspect({task,...sample});
  const optimizer={algorithm:'Adam',learningRate:rate,beta1:0.9,beta2:0.999,epsilon:1e-8,clipScale:clip,appliedGradient:weight.gradient*clip,mHat:selected.m[selectedIndex]/(1-0.9**adamSteps),vHat:selected.v[selectedIndex]/(1-0.999**adamSteps),delta:weight.after-weight.before};
  return {step:steps,task,loss:task==='lm'?lmLoss:taskLoss+lambda*lmLoss,taskLoss,lmLoss,gradientNorm,updateNorm:Math.sqrt(updateSq),weight,optimizer,sample,before,after};
 };
 const snapshot=()=>({version:VERSION,architecture:'gpt-teaching-12-2-1',seed:initialSeed,rng:random.get(),steps,adamSteps,parameters:Object.fromEntries(Object.entries(params).map(([n,p])=>[n,{data:Array.from(p.data),m:Array.from(p.m),v:Array.from(p.v)}]))});
 const restore=s=>{
  if(!s||s.version!==VERSION||s.architecture!=='gpt-teaching-12-2-1'||!Number.isInteger(s.steps)||s.steps<0||!Number.isInteger(s.adamSteps)||s.adamSteps<0||!Number.isInteger(s.seed)||s.seed<0||s.seed>4294967295||!Number.isInteger(s.rng)||s.rng<0||s.rng>4294967295||!s.parameters)throw new Error('Invalid or incompatible checkpoint.');
  for(const [name,p]of Object.entries(params)){const v=s.parameters[name];for(const key of ['data','m','v'])if(!Array.isArray(v?.[key])||v[key].length!==p.data.length||!v[key].every(Number.isFinite)||(key==='v'&&v[key].some(n=>n<0)))throw new Error(`Invalid checkpoint parameter: ${name}`);}
  for(const[name,p]of Object.entries(params))for(const key of ['data','m','v'])p[key].set(s.parameters[name][key]);initialSeed=s.seed;random.set(s.rng);steps=s.steps;adamSteps=s.adamSteps;
 };
 const diagnostics=()=>({
  gradientCheck({task='lm',epsilon=1e-5}={}){task=kind(task);for(const p of Object.values(params))p.grad.fill(0);const ex=task==='lm'?LM_TRAIN[0]:CLS_TRAIN[0];lossPass(ex,task,1);const checks=[];for(const name of ['tokenEmbedding','positionEmbedding','query','key','value','attentionOutput','norm1Scale','ff1','ff2','norm2Scale',...(task==='classify'?['classHead']:[])]){const p=params[name];let index=0;for(let i=1;i<p.grad.length;i++)if(Math.abs(p.grad[i])>Math.abs(p.grad[index]))index=i;const analytic=p.grad[index],old=p.data[index];p.data[index]=old+epsilon;const plus=lossPass(ex,task).loss;p.data[index]=old-epsilon;const minus=lossPass(ex,task).loss;p.data[index]=old;const numerical=(plus-minus)/(2*epsilon);checks.push({name,index,analytic,numerical,error:Math.abs(analytic-numerical)});}return checks;},
  prefixLogits(text){const ids=[INDEX.get('[start]'),...encode(text)],f=forward(ids,'lm',null);return Array.from({length:ids.length},(_,i)=>Array.from(f.logits.data.slice(i*VOCAB.length,(i+1)*VOCAB.length)));},
  data:{lmTrain:LM_TRAIN,lmValidation:LM_VAL,classTrain:CLS_TRAIN,classValidation:CLS_VAL}
 });
 return {metadata:{architecture:'1 decoder block · 12 features · 2 attention heads · learned positions · Post-LN · GELU · tied LM head',vocab:[...VOCAB],tasks:['lm','classify'],trainExamples:LM_TRAIN.length,validationExamples:LM_VAL.length,classificationTrainExamples:CLS_TRAIN.length,classificationValidationExamples:CLS_VAL.length,parameterCount:Object.values(params).reduce((n,p)=>n+p.data.length,0),corpusDescription:'Synthetic English sentences about 9 entities in 3 categories. Validation sentences use held-out combinations of known words. This is a teaching model, not the original GPT-1 checkpoint.'},inspect,step,evaluate,snapshot,restore,reset({seed=42}={}){init(seed);},diagnostics};
}
