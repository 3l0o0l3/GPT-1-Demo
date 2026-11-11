// Loopback-only UI fixture. All requests stay here; mutations are rejected.
import http from 'node:http';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {resolve,extname,sep} from 'node:path';
const root=fileURLToPath(new URL('../dist/',import.meta.url));
const live={step:137,lastMeasuredStep:100,evaluationIntervalSteps:100,source:'random',
  metrics:{trainLoss:4.125,valLoss:4.25,evalDescription:'LOCAL FIXTURE: fixed held-out validation subset.'},
  history:[{step:0,trainLoss:8.1,valLoss:8.3},{step:100,trainLoss:4.125,valLoss:4.25}]};
const transfer={available:true,step:137,lastMeasuredStep:100,evaluationIntervalSteps:100,seed:42,runId:'fixture-pair',
  source:{step:1000,seed:42,source:'pretrained',modelSha256:'fixture-only'},auxiliaryWeight:0,
  branches:['pretrained','scratch'].map(name=>({name,metrics:{accuracy:.75,trainLoss:.45,valLoss:.5},
    history:[{step:0,accuracy:.5,trainLoss:.8,valLoss:.8},{step:100,accuracy:.75,trainLoss:.45,valLoss:.5}]}))};
if(process.env.TIMING_UI_FIXTURE==='1'){
  const stats=(mean,count=7)=>({count,mean,p95:mean*1.35,min:mean*.8,max:mean*1.4});
  const spans=(value,cpuCount=7,gpuCount=6)=>({cpuMs:stats(value,cpuCount),gpuMs:stats(value*.72,gpuCount)});
  const phases={data:spans(.22),setup:spans(.12),forward_loss:spans(1.15),loss_check:spans(.31),backward:spans(1.74),gradient_check_clip:spans(.28),optimizer:spans(.57)};
  const timing=kind=>({version:1,kind,enabled:true,sampleEvery:20,capacity:200,window:'last-capacity-records',summaryScope:'successful-records-only',gpuSupported:true,pendingGpuSamples:1,
    counters:{observedSteps:137,sampledSteps:7,successfulSamples:9,failedSamples:0,evaluations:2,gpuDroppedSamples:0,eventErrors:0,clockErrors:0},
    steps:{count:7,successfulCount:7,failedCount:0,total:spans(kind==='lm'?4.5:8.4),phases:kind==='lm'?phases:{data:phases.data,...Object.fromEntries(['pretrained','scratch'].flatMap(branch=>Object.entries(phases).filter(([name])=>name!=='data').map(([name,value])=>[`${branch}.${name}`,value])))},
      trend:[1,20,40,60,80,100,120].map((step,i)=>({step,success:true,cpuMs:(kind==='lm'?4.2:8)+(i%3)*.3,gpuMs:i===6?null:(kind==='lm'?3:6)+(i%2)*.2}))},
    evaluations:{count:2,successfulCount:2,failedCount:0,total:spans(35,2,2),phases:kind==='lm'?{train_evaluation:spans(16,2,2),validation:spans(18,2,2)}:{'pretrained.train_evaluation':spans(8,2,2),'pretrained.validation':spans(9,2,2),'scratch.train_evaluation':spans(8,2,2),'scratch.validation':spans(9,2,2)},
      trend:[{step:0,success:true,cpuMs:34,gpuMs:27},{step:100,success:true,cpuMs:36,gpuMs:29}]}});
  live.timing=timing('lm');transfer.timing=timing('transfer');
}
const now=new Date().toISOString(),job={id:'fixture-job',requestId:'fixture-request',kind:'lm',status:'running',
  requestedSteps:1000,completedSteps:137,initialStep:0,currentStep:137,learningRate:.0003,startedAt:now,updatedAt:now,
  elapsedSeconds:5,stepsPerSecond:27.4,liveState:live,transferState:transfer,error:null};
const responses={job:{job},state:live,'transfer-state':transfer,examples:{examples:[]},
  meta:{architecture:{layers:4,heads:4,width:128,context:128,trainContext:64,parameters:1858306},data:{counts:{}},device:'cuda:0',seed:42,
    execution:{trainingPrecision:'FP16 AMP',lmBackend:'compiled',compileEnabled:true,compileReady:true,transferPrecision:'FP16 AMP',transferBackend:'eager'}},
  recordings:{pretraining:{history:[],checkpoints:[]},transfer:{runs:[]}}};
const types={'.html':'text/html','.js':'application/javascript','.css':'text/css','.woff2':'font/woff2','.woff':'font/woff','.ttf':'font/ttf','.svg':'image/svg+xml'};
const server=http.createServer(async(req,res)=>{
  const path=new URL(req.url,'http://127.0.0.1').pathname;
  if(req.method!=='GET'){res.writeHead(405,{'Content-Type':'application/json'});res.end(JSON.stringify({error:{message:'Fixture rejects all mutations.'}}));return;}
  if(path.startsWith('/api/')){
    const value=path.startsWith('/api/training/')?responses[path.slice('/api/training/'.length)]:undefined;
    res.writeHead(value?200:503,{'Content-Type':'application/json','Cache-Control':'no-store'});
    res.end(JSON.stringify(value||{error:{message:'Not provided by the local UI fixture.'}}));return;
  }
  if(path==='/presenter-session.js'){
    res.writeHead(200,{'Content-Type':'application/javascript'});
    res.end("export const hosted=false,trainingBase='/api/training/',telemetryUrl='/api/telemetry'; export const sessionFetch=(url,options)=>fetch(url,options);");return;
  }
  try{
    const target=resolve(root,'.'+(path==='/'?'/index.html':path));
    if(!target.startsWith(root.endsWith(sep)?root:root+sep))throw new Error('Outside fixture assets');
    let body=await readFile(target);
    if(extname(target)==='.html')body=Buffer.from(body.toString().replace('<title>','<title>LOCAL UI FIXTURE · ').replace('<body>','<body><div style="padding:8px 14px;background:#fff3cc;color:#4e431f;font:12px sans-serif">LOCAL UI FIXTURE — synthetic progress and metrics; no model is connected.</div>').replaceAll('<span>Performance breakdown</span>','<span>Performance breakdown · SYNTHETIC FIXTURE</span>'));
    res.writeHead(200,{'Content-Type':types[extname(target)]||'application/octet-stream','Cache-Control':'no-store'});res.end(body);
  }catch{res.writeHead(404);res.end('Not found');}
});
server.listen(0,'127.0.0.1',()=>console.log(`http://127.0.0.1:${server.address().port}/#demo-convergence`));
process.stdin.on('data',()=>{server.closeAllConnections();server.close(()=>process.exit(0));});
