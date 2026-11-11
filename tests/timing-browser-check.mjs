// Optional visual check using an already installed Playwright package/browser.
// All data are synthetic and all requests are restricted to the local fixture.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdir,writeFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const packagePath=process.argv[2];
if(!packagePath)throw new Error('Pass the existing Playwright index.mjs path; this check never installs browsers.');
const {chromium}=await import(pathToFileURL(resolve(packagePath))),out=resolve(process.argv[3]||'test-artifacts/timing');
const fixture=spawn(process.execPath,[fileURLToPath(new URL('./evaluation-browser-fixture.mjs',import.meta.url))],{env:{...process.env,TIMING_UI_FIXTURE:'1'},stdio:['pipe','pipe','pipe'],windowsHide:true});
let browser;
try{
  const url=await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('Fixture startup timed out')),10000);fixture.stdout.once('data',chunk=>{clearTimeout(timer);resolve(String(chunk).trim());});fixture.once('error',reject);});
  const origin=new URL(url).origin,errors=[],blocked=[],mutations=[];browser=await chromium.launch({headless:true,...(process.argv[4]?{executablePath:resolve(process.argv[4])}:{})});
  const page=await browser.newPage({viewport:{width:1365,height:950},deviceScaleFactor:1});
  page.on('pageerror',error=>errors.push(error.message));page.on('request',request=>{if(request.method()!=='GET')mutations.push({method:request.method(),url:request.url()});});
  await page.route('**/*',route=>{if(new URL(route.request().url()).origin===origin)return route.continue();blocked.push(route.request().url());return route.abort();});
  await page.goto(url,{waitUntil:'domcontentloaded'});await page.locator('#lm-performance-summary').filter({hasText:'7 updates sampled'}).waitFor();
  await mkdir(out,{recursive:true});
  await page.locator('#lm-performance>summary').click();await page.locator('#lm-performance-content .timing-chart').first().waitFor();
  assert.match(await page.locator('#lm-performance-content').innerText(),/every 20th absolute update/);
  await page.locator('#lm-performance').screenshot({path:join(out,'lm-timing-synthetic.png')});
  await page.goto(origin+'/#demo-transfer',{waitUntil:'domcontentloaded'});await page.locator('#transfer-performance>summary').click();
  await page.locator('#transfer-performance-content .timing-chart').first().waitFor();
  assert.match(await page.locator('#transfer-performance-content').innerText(),/Random weights · Backward/);
  await page.locator('#transfer-performance').screenshot({path:join(out,'transfer-timing-synthetic.png')});
  await page.setViewportSize({width:390,height:844});await page.locator('#transfer-performance .timing-clock').first().evaluate(node=>node.scrollIntoView({block:'start'}));
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true,'Mobile viewport must not overflow horizontally');
  await page.screenshot({path:join(out,'transfer-mobile-synthetic.png')});
  assert.deepEqual(errors,[]);assert.deepEqual(mutations,[]);
  const report={scope:'Loopback-only synthetic fixture; no model or production authentication',browser:'Existing local Chromium via Playwright',viewport:{desktop:[1365,950],mobile:[390,844]},pageErrors:errors,mutations,blockedExternalRequests:blocked,checks:['LM disclosure and actual-sample labels','Transfer branch phase labels','Mobile no horizontal overflow'],screenshots:['lm-timing-synthetic.png','transfer-timing-synthetic.png','transfer-mobile-synthetic.png']};
  await writeFile(join(out,'browser-report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({out,...report}));
}finally{if(browser)await browser.close();fixture.stdin.end('\n');await new Promise(resolve=>{if(fixture.exitCode!==null)return resolve();fixture.once('exit',resolve);});}
