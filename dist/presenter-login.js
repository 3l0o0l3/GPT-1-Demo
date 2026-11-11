export function validCode(value){return typeof value==='string'&&/^\d{8}$/.test(value);}
export function loginMessage(status,value){const error=new Error(value?.error?.message||`Sign-in request failed (${status}).`);error.status=status;error.code=value?.error?.code;return error;}

export function mountLogin({document,location,fetch,now=()=>Date.now(),setInterval,clearInterval}){
  const $=id=>document.getElementById(id);
  const state={csrf:null,challenge:null,busy:false,configured:false,retryAt:0,expiresAt:0};
  function status(message,error=false){$('login-status').textContent=message;$('login-status').classList.toggle('error',error);}
  function controls(){
    const seconds=Math.max(0,Math.ceil((state.retryAt-now())/1000));
    $('login-email').disabled=state.busy;$('request-code').disabled=state.busy||!state.csrf||!state.configured||seconds>0;
    $('request-code').textContent=state.challenge?'Send a new code':'Send sign-in code';
    $('login-code').disabled=state.busy;$('verify-code').disabled=state.busy||!state.challenge||!validCode($('login-code').value)||now()>=state.expiresAt;
    $('change-email').disabled=state.busy;$('resend-note').textContent=seconds?`You can request another code in ${seconds} seconds.`:'';
  }
  async function request(path,body){
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),20000);
    try{
      const response=await fetch('/api/auth/'+path,{method:body===undefined?'GET':'POST',credentials:'same-origin',cache:'no-store',redirect:'error',headers:body===undefined?{}:{'Content-Type':'application/json','X-CSRF-Token':state.csrf},body:body===undefined?undefined:JSON.stringify(body),signal:controller.signal});
      let value;try{value=await response.json();}catch{throw new Error('The sign-in response could not be read. Reload this page before trying again.');}
      if(!response.ok)throw loginMessage(response.status,value);return value;
    }finally{clearTimeout(timer);}
  }
  async function start(){
    const reason=new URLSearchParams(location.search||'').get('reason');
    if(reason==='session-expired'){$('login-notice').hidden=false;$('login-notice').textContent='Your session ended. After signing in, refresh the model state before repeating any interrupted training or saving action; it may already have completed.';}
    else if(reason==='signed-out'){$('login-notice').hidden=false;$('login-notice').textContent='You have signed out. Training and submitted conversations are preserved.';}
    try{
      const value=await request('session');if(value.authenticated===true){location.replace('/');return;}
      if(typeof value.csrfToken!=='string'||!value.csrfToken||typeof value.smtpConfigured!=='boolean')throw new Error('The sign-in session response is incomplete.');
      state.csrf=value.csrfToken;state.configured=value.smtpConfigured;
      status(state.configured?'Use your approved presenter email address.':'Email sign-in is not configured yet. Contact the site owner.',!state.configured);
    }catch(error){status(error.message,true);}finally{controls();}
  }
  $('request-form').addEventListener('submit',async event=>{
    event.preventDefault();if($('request-code').disabled)return;
    const email=$('login-email').value.trim();if(!email||!$('login-email').checkValidity())return;
    state.busy=true;controls();status('Requesting a sign-in code…');
    try{
      const value=await request('request-code',{email});
      if(typeof value.challengeId!=='string'||!value.challengeId||!Number.isInteger(value.expiresIn)||value.expiresIn<1||!Number.isInteger(value.retryAfter)||value.retryAfter<0)throw new Error('The code request could not be confirmed. Reload this page before requesting another code.');
      state.challenge=value.challengeId;state.retryAt=now()+value.retryAfter*1000;state.expiresAt=now()+value.expiresIn*1000;
      $('verify-form').hidden=false;$('login-code').value='';status(typeof value.message==='string'?value.message:'If this address is approved, a code has been sent. Check your inbox and spam folder.');
    }catch(error){status(error.name==='AbortError'?'The request timed out. A code may still arrive; wait a minute before requesting another.':error.message,true);if(error.name==='AbortError'||error.status===429)state.retryAt=now()+60000;}
    finally{state.busy=false;controls();if(state.challenge)$('login-code').focus();}
  });
  $('verify-form').addEventListener('submit',async event=>{
    event.preventDefault();if($('verify-code').disabled)return;
    const code=$('login-code').value;state.busy=true;controls();status('Verifying your code…');
    try{
      const value=await request('verify',{challengeId:state.challenge,code});
      if(value.authenticated!==true)throw new Error('Sign-in could not be confirmed. Please reload this page.');
      $('login-code').value='';state.challenge=null;status('Signed in. Opening your workspace…');location.replace('/');
    }catch(error){$('login-code').value='';status(error.name==='AbortError'?'Verification timed out. Reload this page to check whether you are signed in.':error.message,true);if(error.status===429){state.challenge=null;state.retryAt=now()+60000;}}
    finally{state.busy=false;controls();}
  });
  $('login-code').addEventListener('input',()=>{$('login-code').value=$('login-code').value.replace(/\D/g,'').slice(0,8);controls();});
  $('login-email').addEventListener('input',()=>{if(state.challenge){state.challenge=null;$('verify-form').hidden=true;$('login-code').value='';}controls();});
  $('change-email').addEventListener('click',()=>{state.challenge=null;$('verify-form').hidden=true;$('login-code').value='';$('login-email').focus();status('Enter your email, then request a new code.');controls();});
  const interval=setInterval(()=>{if(state.challenge&&now()>=state.expiresAt){state.challenge=null;$('login-code').value='';status('This code has expired. Request a new sign-in code.',true);}controls();},1000);
  document.addEventListener('visibilitychange',controls);
  start();return {destroy:()=>clearInterval(interval)};
}
if(typeof document!=='undefined')mountLogin({document,location,fetch:(...args)=>fetch(...args),setInterval,clearInterval});
