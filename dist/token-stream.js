// The transport supplies genuine token events; the player only controls reveal timing.
export class GenerationStreamError extends Error {
  constructor(message,code='invalid_stream'){super(message);this.name='GenerationStreamError';this.code=code;}
}
const abortError=()=>new DOMException('Generation was stopped.','AbortError');
const validToken=t=>t&&Number.isInteger(t.id)&&t.id>=0&&typeof t.piece==='string'&&Number.isFinite(t.p)&&t.p>=0&&t.p<=1;
const sameToken=(a,b)=>a.id===b.id&&a.piece===b.piece&&a.p===b.p;
const validContext=c=>c&&['fixed','sliding'].includes(c.mode)&&Number.isInteger(c.windowSize)&&c.windowSize>=1&&c.windowSize<=128&&c.modelContext===128&&Number.isInteger(c.promptTokens)&&c.promptTokens>=1&&(c.mode==='sliding'||(c.windowSize===128&&c.promptTokens<128));
const sameContext=(a,b)=>a===undefined&&b===undefined||a&&b&&['mode','windowSize','modelContext','promptTokens'].every(key=>a[key]===b[key]);
const validWindow=w=>w&&Number.isInteger(w.inputTokens)&&w.inputTokens>=1&&w.inputTokens<=128&&Number.isInteger(w.droppedTokens)&&w.droppedTokens>=0;
const sameWindow=(a,b)=>a==null&&b==null||a&&b&&a.inputTokens===b.inputTokens&&a.droppedTokens===b.droppedTokens;

export async function readGenerationStream(response,{signal,onEvent=()=>{}}={}){
  if(!response.ok){let data;try{data=await response.json();}catch{}throw new GenerationStreamError(data?.error?.message||`Generation request failed (${response.status}).`,data?.error?.code||`http_${response.status}`);}
  if(!/^application\/x-ndjson(?:;|$)/i.test(response.headers.get('content-type')||''))throw new GenerationStreamError('The runtime did not return a token stream. Update the streaming service; completed JSON output is not used as a fallback.');
  if(!response.body)throw new GenerationStreamError('The generation response has no readable stream.');
  const reader=response.body.getReader(),decoder=new TextDecoder('utf-8',{fatal:true});
  let buffer='',bytes=0,lineBytes=0,start=null,terminal=null,continuation='',lastWindow=null;const tokens=[];
  const onAbort=()=>{reader.cancel().catch(()=>{});};signal?.addEventListener('abort',onAbort,{once:true});
  const checkAbort=()=>{if(signal?.aborted)throw abortError();};
  const parseLine=line=>{
    if(!line.trim())return;
    if(terminal)throw new GenerationStreamError('Unexpected data after the terminal stream event.');
    let event;try{event=JSON.parse(line);}catch{throw new GenerationStreamError('The generation stream contains an invalid JSON line.');}
    if(event?.type==='start'){
      if(start||typeof event.text!=='string'||!['live','random'].includes(event.source?.model)||!Number.isInteger(event.maxNewTokens)||event.maxNewTokens<1||event.maxNewTokens>1000||!Number.isFinite(event.sampling?.temperature)||!Number.isInteger(event.sampling?.seed)||(event.context!==undefined&&!validContext(event.context)))throw new GenerationStreamError('The generation stream contains an invalid start event or context.');
      start=event;onEvent(event);return;
    }
    if(!start)throw new GenerationStreamError('The generation stream did not begin with a start event.');
    if(event?.type==='token'){
      if(event.index!==tokens.length||tokens.length>=start.maxNewTokens||!validToken(event.token)||typeof event.continuation!=='string')throw new GenerationStreamError('The generation stream contains an invalid or out-of-order token.');
      if(event.window!==undefined&&!validWindow(event.window))throw new GenerationStreamError('The generation stream contains invalid token window counts.');
      if(start.context){const context=start.context,total=context.promptTokens+event.index,expectedInput=Math.min(context.windowSize,total),expectedDropped=Math.max(0,total-context.windowSize);if(!event.window||event.window.inputTokens!==expectedInput||event.window.droppedTokens!==expectedDropped||(context.mode==='fixed'&&total>=128))throw new GenerationStreamError('Token window counts do not match the declared generation context.');}
      tokens.push(event.token);continuation=event.continuation;lastWindow=event.window??null;onEvent(event);return;
    }
    if(event?.type==='error')throw new GenerationStreamError(event.error?.message||'The model stopped with a streaming error.',event.error?.code||'generation_error');
    if(event?.type==='done'){
      const result=event.result;
      if(!result||result.text!==start.text||result.continuation!==continuation||typeof result.fullText!=='string'||typeof result.stopReason!=='string'||!Array.isArray(result.tokens)||result.tokens.length!==tokens.length||!tokens.every((token,index)=>validToken(result.tokens[index])&&sameToken(token,result.tokens[index]))||!['model','step','seed','source'].every(key=>result.source?.[key]===start.source?.[key])||!['temperature','seed'].every(key=>result.sampling?.[key]===start.sampling?.[key])||!sameContext(start.context,result.context)||!sameWindow(lastWindow,result.window))throw new GenerationStreamError('The final result does not match the streamed tokens, context or provenance.');
      terminal=event;return;
    }
    throw new GenerationStreamError('The generation stream contains an unknown event.');
  };
  let completed=false;
  try{
    checkAbort();
    while(true){const chunk=await reader.read();checkAbort();if(chunk.done)break;bytes+=chunk.value.byteLength;if(bytes>67108864)throw new GenerationStreamError('The generation stream exceeded its 64 MiB size limit.');for(const byte of chunk.value){lineBytes=byte===10?0:lineBytes+1;if(lineBytes>1048576)throw new GenerationStreamError('A generation stream line exceeded its 1 MiB size limit.');}buffer+=decoder.decode(chunk.value,{stream:true});let newline;while((newline=buffer.indexOf('\n'))!==-1){const line=buffer.slice(0,newline);buffer=buffer.slice(newline+1);parseLine(line);checkAbort();}}
    buffer+=decoder.decode();if(buffer.trim())throw new GenerationStreamError('The generation stream ended with an incomplete line.');
    if(!terminal)throw new GenerationStreamError('The generation stream ended before completion. Visible text is only a partial result.','truncated_stream');
    checkAbort();onEvent(terminal);completed=true;return terminal.result;
  }catch(error){if(error instanceof TypeError&&error.message.includes('encoded data'))throw new GenerationStreamError('The generation stream contains invalid UTF-8.');throw error;}
  finally{signal?.removeEventListener('abort',onAbort);if(!completed)await reader.cancel().catch(()=>{});reader.releaseLock();}
}

export class TokenPlayer {
  constructor({rate=8,onChange=()=>{},onToken=()=>{},setTimer=(fn,delay)=>setTimeout(fn,delay),clearTimer=timer=>clearTimeout(timer)}={}){
    this.rate=rate;this.onChange=onChange;this.onToken=onToken;this.setTimer=setTimer;this.clearTimer=clearTimer;this.queue=[];this.visible=[];this.continuation='';this.window=null;this.startEvent=null;this.result=null;this.network='waiting';this.paused=false;this.error=null;this.terminal=null;this.timer=null;this.timerVersion=0;this.received=0;
    this.finished=new Promise(resolve=>{this.resolveFinished=resolve;});this.checkRate(rate);
  }
  checkRate(rate){if(rate!==Infinity&&(!Number.isFinite(rate)||rate<=0))throw new RangeError('Token rate must be positive or Infinity.');}
  snapshot(){return {status:this.terminal||(this.paused?'paused':this.network==='done'?'revealing':this.network),paused:this.paused,network:this.network,received:this.received,displayed:this.visible.length,queued:this.queue.length,continuation:this.continuation,window:this.window,start:this.startEvent,result:this.result,error:this.error,tokens:[...this.visible]};}
  emit(){this.onChange(this.snapshot());}
  cancelTimer(){this.timerVersion++;if(this.timer!==null){this.clearTimer(this.timer);this.timer=null;}}
  start(event){if(this.terminal)return;if(this.startEvent)throw new GenerationStreamError('The token player already has a source.');this.startEvent=event;this.network='streaming';this.emit();}
  enqueue(event){if(this.terminal)return;if(!this.startEvent||this.network!=='streaming'||event.index!==this.received||!validToken(event.token)||typeof event.continuation!=='string')throw new GenerationStreamError('The token player received an invalid event sequence.');this.received++;this.queue.push(event);this.emit();this.schedule();}
  networkDone(result){if(this.terminal)return;this.result=result;this.network='done';this.emit();this.schedule();}
  reveal(){const event=this.queue.shift();if(!event)return;this.visible.push(event.token);this.continuation=event.continuation;this.window=event.window??null;this.onToken(event,this.snapshot());this.emit();}
  schedule(){
    if(this.terminal)return;
    if(this.network==='done'&&!this.queue.length){this.finish('done');return;}
    if(this.paused||!this.queue.length||this.timer!==null)return;
    if(this.rate===Infinity){while(this.queue.length&&!this.paused&&!this.terminal)this.reveal();if(this.network==='done'&&!this.queue.length)this.finish('done');return;}
    const version=this.timerVersion;
    this.timer=this.setTimer(()=>{if(version!==this.timerVersion||this.terminal||this.paused)return;this.timer=null;this.reveal();this.schedule();},1000/this.rate);
  }
  setRate(rate){this.checkRate(rate);if(this.terminal)return;this.rate=rate;this.cancelTimer();this.schedule();}
  pause(){if(this.terminal)return;this.paused=true;this.cancelTimer();this.emit();}
  resume(){if(this.terminal)return;this.paused=false;this.emit();this.schedule();}
  finish(status){if(this.terminal)return;this.cancelTimer();this.terminal=status;this.emit();this.resolveFinished(this.snapshot());}
  stop(reason='Stopped. The visible text is a partial result.'){if(this.terminal)return;this.error=reason;this.queue=[];this.finish('stopped');}
  fail(error){if(this.terminal)return;this.error=error?.message||String(error);this.queue=[];this.finish('failed');}
}
