import { createLab } from './engine.js';
let lab, preview=null;
const invalidate=()=>{preview=null;};
self.onmessage=({data:{id,type,payload={}}})=>{
 try {
  let data;
  if(type==='init') {lab=createLab(payload);invalidate();data={metadata:lab.metadata,evaluation:lab.evaluate(),probe:lab.inspect()};}
  else {
   if(!lab) throw new Error('The model is still initializing.');
   if(type==='prepare') {
    const original=lab.snapshot();
    try {const trace=lab.step(payload);preview={state:lab.snapshot(),trace};data=trace;}
    finally {lab.restore(original);}
   } else if(type==='commit') {
    if(!preview)throw new Error('Prepare a new update first.');
    lab.restore(preview.state);data={trace:preview.trace,evaluation:lab.evaluate()};invalidate();
   } else if(type==='train') {
    invalidate();const steps=Number(payload.steps??10);
    if(!Number.isInteger(steps)||steps<1||steps>20)throw new Error('Training chunks must contain 1–20 steps.');
    let trace;for(let i=0;i<steps;i++)trace=lab.step(payload);
    data={trace,evaluation:lab.evaluate({task:payload.task||'lm'}),lmEvaluation:payload.task==='classify'?lab.evaluate({task:'lm'}):null};
   } else if(type==='step') {invalidate();data=lab.step(payload);}
   else if(type==='inspect') data=lab.inspect(payload);
   else if(type==='evaluate') data=lab.evaluate(payload);
   else if(type==='snapshot') data=lab.snapshot();
   else if(type==='restore') {lab.restore(payload);invalidate();data=lab.evaluate();}
   else if(type==='reset') {lab.reset(payload);invalidate();data={metadata:lab.metadata,evaluation:lab.evaluate(),probe:lab.inspect()};}
   else throw new Error('Unsupported model action.');
  }
  self.postMessage({id,type:'result',data});
 } catch(error) {self.postMessage({id,type:'error',message:error.message});}
};
