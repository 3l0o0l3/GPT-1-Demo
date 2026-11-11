// Teaching requests expose each measured block; background jobs run independently.
export const jobIsActive=job=>Boolean(job&&['running','pause_requested'].includes(job.status));
export const trainingMode=steps=>steps<=100?'teaching':'background';
const integer=value=>Number.isSafeInteger(value)&&value>=0;
const statuses=new Set(['running','pause_requested','paused','completed','failed']);

export function readJob(value){
  if(!value||typeof value!=='object'||!Object.hasOwn(value,'job'))throw new Error('Invalid background-training response.');
  const job=value.job;if(job===null)return null;
  if(!job||typeof job.id!=='string'||!job.id||job.id.length>128||typeof job.requestId!=='string'||!job.requestId||
      !['lm','transfer'].includes(job.kind)||!statuses.has(job.status)||!integer(job.requestedSteps)||job.requestedSteps<1||job.requestedSteps>5000||
      !integer(job.completedSteps)||job.completedSteps>job.requestedSteps||!integer(job.initialStep)||!integer(job.currentStep)||
      job.currentStep!==job.initialStep+job.completedSteps||!Number.isFinite(job.elapsedSeconds)||job.elapsedSeconds<0||
      !Number.isFinite(job.stepsPerSecond)||job.stepsPerSecond<0||!Number.isFinite(Date.parse(job.startedAt))||!Number.isFinite(Date.parse(job.updatedAt))||
      !job.liveState||!integer(job.liveState.step)||!Array.isArray(job.liveState.history)||!job.transferState||
      (job.kind==='lm'?job.liveState.step!==job.currentStep:job.transferState.step!==job.currentStep)||
      (job.status==='completed'&&job.completedSteps!==job.requestedSteps)||
      (job.error!==null&&(!job.error||typeof job.error.code!=='string'||typeof job.error.message!=='string')))
    throw new Error('Invalid background-training progress. Refresh actual model state before continuing.');
  return job;
}

export async function runTeachingSteps({kind,steps,learningRate,initialState,request,onChunk=()=>{},shouldPause=()=>false,
  afterPaint=()=>new Promise(resolve=>requestAnimationFrame(()=>setTimeout(resolve,0)))}){
  if(!['lm','transfer'].includes(kind)||!integer(steps)||steps<1||steps>100||!initialState||!integer(initialState.step))
    throw new Error('Teaching mode accepts 1–100 steps and a confirmed model state.');
  let state=initialState,completedSteps=0;
  while(completedSteps<steps&&!shouldPause()){
    const count=Math.min(10,steps-completedSteps),before=state.step;
    const next=await request(kind==='lm'?'train':'transfer-train',{steps:count,learningRate});
    if(!next||!integer(next.step)||next.step!==before+count||(kind==='transfer'&&next.runId!==initialState.runId))
      throw new Error('The model or transfer pair changed during teaching mode. Stopped; refresh actual state before continuing.');
    state=next;completedSteps+=count;onChunk({state,completedSteps,requestedSteps:steps});
    await afterPaint();
  }
  return{state,completedSteps,paused:completedSteps<steps};
}

export function createTrainingJobs({request,onUpdate=()=>{},onError=()=>{},interval=100,idleInterval=1000,
  setTimer=setTimeout,clearTimer=clearTimeout,now=()=>Date.now(),newId=()=>crypto.randomUUID()}){
  let job=null,timer=null,reading=null,mutating=false,connected=false,uncertain=false,disposed=false,epoch=0,expectedRequest=null,failures=0;
  const version=value=>value?`${value.id}:${value.status}:${value.currentStep}:${value.updatedAt}`:'none';
  const notify=(source,previous=job)=>onUpdate({job,source,changed:version(previous)!==version(job),snapshotChanged:previous?.id!==job?.id||previous?.currentStep!==job?.currentStep,previous,
    pending:mutating,connected,uncertain,active:jobIsActive(job)});
  function nextDelay(){return failures?Math.min(8000,1000*2**Math.min(3,failures-1)):jobIsActive(job)||uncertain?interval:idleInterval;}
  function schedule(delay=nextDelay()){if(disposed)return;if(timer!==null)clearTimer(timer);timer=setTimer(()=>{timer=null;void poll();},Math.max(25,delay));}
  function accept(value,source){
    const next=readJob(value),previous=job;
    if(!next&&jobIsActive(previous))throw new Error('The service no longer reports this task. It may have restarted. Use Retry to read the actual model state; training will not restart automatically.');
    if(next&&previous?.id===next.id&&next.completedSteps<previous.completedSteps)throw new Error('Received older training progress. Waiting for a fresh snapshot.');
    job=next;connected=true;failures=0;
    if(expectedRequest&&next?.requestId===expectedRequest)expectedRequest=null;
    uncertain=Boolean(expectedRequest);notify(source,previous);return job;
  }
  function failure(error,source){connected=false;failures++;onError(error,{source,job,uncertain});notify(source);}
  async function poll(){
    if(disposed)return job;if(reading)return reading;if(mutating){schedule();return job;}
    const started=now(),generation=epoch;
    reading=(async()=>{try{const value=await request('job');if(!disposed&&generation===epoch)return accept(value,'poll');return job;}
      catch(error){if(!disposed&&generation===epoch)failure(error,'poll');return job;}
      finally{reading=null;schedule(failures?nextDelay():nextDelay()-(now()-started));}})();
    return reading;
  }
  async function discover({reset=false}={}){
    if(disposed)return null;
    if(reset){epoch++;if(timer!==null)clearTimer(timer);timer=null;job=null;expectedRequest=null;uncertain=false;}
    if(reading)await reading;
    if(timer!==null)clearTimer(timer);timer=null;
    const generation=epoch;
    reading=(async()=>{try{const value=await request('job');if(disposed||generation!==epoch)return job;return accept(value,'discovery');}
      catch(error){if(!disposed&&generation===epoch)failure(error,'discovery');throw error;}
      finally{reading=null;schedule();}})();
    return reading;
  }
  async function start({kind,steps,learningRate}){
    if(disposed||mutating||uncertain||jobIsActive(job))return false;
    if(!['lm','transfer'].includes(kind)||!integer(steps)||steps<1||steps>5000||!Number.isFinite(learningRate)||learningRate<.00001||learningRate>.01)
      throw new Error('Choose 1–5000 steps and a supported learning rate.');
    const requestId=newId();epoch++;mutating=true;expectedRequest=requestId;
    if(timer!==null)clearTimer(timer);timer=null;notify('start-pending');
    try{const value=await request('job-start',{kind,steps,learningRate,requestId});const next=readJob(value);
      if(!next||next.requestId!==requestId||next.kind!==kind||next.requestedSteps!==steps)throw new Error('The start response did not identify the requested task. Checking status without repeating it.');
      accept(value,'start');return true;
    }catch(error){
      uncertain=!Number.isInteger(error.status)||error.status>=500;
      if(!uncertain)expectedRequest=null;
      failure(error,'start');return false;
    }finally{mutating=false;notify('start-settled');schedule();}
  }
  async function pause(){
    if(disposed||mutating||!jobIsActive(job)||job.status==='pause_requested')return false;
    const id=job.id;epoch++;mutating=true;notify('pause-pending');
    try{const value=await request('job-pause',{jobId:id});const next=readJob(value);
      if(!next||next.id!==id)throw new Error('Pause response identified another task. Checking status.');
      accept(value,'pause');return true;
    }catch(error){failure(error,'pause');return false;}
    finally{mutating=false;notify('pause-settled');schedule();}
  }
  function dispose(){disposed=true;epoch++;if(timer!==null)clearTimer(timer);timer=null;}
  return{start,pause,poll,discover,dispose,get state(){return{job,pending:mutating,connected,uncertain,active:jobIsActive(job)};}};
}
