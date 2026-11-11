/** Present real batches at a teaching pace. No DOM, model state or prompt ownership. */
export function createTokenPlayback({requestBatch,onToken,onState=()=>{},onError=()=>{},rate=2,
  setTimer=(callback,delay)=>setTimeout(callback,delay),clearTimer=timer=>clearTimeout(timer)}){
  let queue=[],active=false,inFlight=false,generation=0,timer=null,terminal=null,failure=null,phase='idle';
  const validRate=value=>Number.isFinite(value)&&value>0&&value<=100;
  if(!validRate(rate))throw new RangeError('Playback rate must be between 0 and 100 tokens per second.');
  const snapshot=()=>({phase,active,queued:queue.length,inFlight,rate,reason:terminal});
  function emit(){onState(snapshot());}
  function cancelTimer(){if(timer!==null){clearTimer(timer);timer=null;}}
  function fail(error){active=false;failure=error;phase='error';cancelTimer();emit();onError(error);}
  function finish(){active=false;phase='complete';cancelTimer();emit();}
  function schedule(){
    if(!active)return;
    if(queue.length){
      phase='playing';if(timer===null)timer=setTimer(reveal,1000/rate);emit();return;
    }
    if(terminal){finish();return;}
    if(inFlight){phase='waiting';emit();return;}
    request();
  }
  function reveal(){
    timer=null;if(!active)return;
    const token=queue.shift();
    if(token===undefined){schedule();return;}
    try{onToken(token,{last:queue.length===0,finished:queue.length===0&&terminal==='eos',reason:queue.length===0?terminal:null});}
    catch(error){fail(error);return;}
    schedule();
  }
  async function request(){
    if(inFlight||!active)return;
    const requestGeneration=generation;inFlight=true;phase='waiting';emit();
    try{
      const result=await requestBatch();
      if(requestGeneration!==generation)return;
      if(!Array.isArray(result.tokens))throw new Error('The batch provider returned no token list.');
      if(!result.tokens.length&&!result.finished&&!result.limit)throw new Error('The model returned no tokens and did not finish.');
      queue.push(...result.tokens);terminal=result.finished?'eos':result.limit?'limit':null;
    }catch(error){if(requestGeneration===generation)fail(error);}
    finally{
      inFlight=false;
      if(requestGeneration!==generation){if(active)schedule();else emit();}
      else if(failure)emit();
      else if(active)schedule();
      else if(terminal&&!queue.length)finish();
      else{phase='paused';emit();}
    }
  }
  return {
    snapshot,
    start(){if(active||phase==='complete')return;failure=null;active=true;schedule();},
    pause(){active=false;cancelTimer();if(phase!=='idle'&&phase!=='complete'&&phase!=='error')phase='paused';emit();},
    reset(){generation++;active=false;queue=[];terminal=null;failure=null;phase='idle';cancelTimer();emit();},
    setRate(value){if(!validRate(value))throw new RangeError('Playback rate must be between 0 and 100 tokens per second.');rate=value;if(timer!==null){cancelTimer();if(active&&queue.length)timer=setTimer(reveal,1000/rate);}emit();}
  };
}
