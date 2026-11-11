// Shared transport: portable localhost uses this origin; hosted mode retains authentication.
export function createSessionTransport({location,fetch,redirect}={}){
  const local=location?.protocol==='http:'&&['127.0.0.1','localhost'].includes(location.hostname);
  const hosted=Boolean(location)&&!local;
  const trainingBase='/api/training/';
  const telemetryUrl='/api/telemetry';
  let session=null,pending=null,redirected=false;
  function expired(){
    session=null;
    if(!redirected){redirected=true;redirect('/login?reason=session-expired');}
    const error=new Error('Your session has ended. Sign in again, then refresh model state before repeating an interrupted action; it may already have completed.');
    error.status=401;error.code='login_required';return error;
  }
  async function getSession(){
    if(!hosted)return null;
    if(session)return session;
    if(!pending)pending=(async()=>{
      const response=await fetch('/api/auth/session',{credentials:'same-origin',cache:'no-store',redirect:'error'});
      if(response.status===401)throw expired();
      if(!response.ok)throw new Error('Could not check your sign-in session. Please refresh before trying again.');
      const value=await response.json();
      if(value.authenticated!==true)throw expired();
      if(typeof value.csrfToken!=='string'||!value.csrfToken)throw new Error('The sign-in session response is incomplete.');
      session=value;return session;
    })().finally(()=>{pending=null;});
    return pending;
  }
  async function sessionFetch(url,options={}){
    if(!hosted)return fetch(url,options);
    const target=new URL(url,location.origin);
    if(target.origin!==location.origin)throw new Error('Presenter requests must stay on this website.');
    const method=(options.method||'GET').toUpperCase(),headers=new Headers(options.headers);
    if(!['GET','HEAD'].includes(method)){
      const current=await getSession();
      if(options.signal?.aborted)throw options.signal.reason||new DOMException('Request cancelled.','AbortError');
      headers.set('X-CSRF-Token',current.csrfToken);
    }
    const response=await fetch(target.href,{...options,headers,credentials:'same-origin',redirect:'error'});
    if(response.status===401)throw expired();
    return response;
  }
  async function logout(){
    const response=await sessionFetch('/api/auth/logout',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}',cache:'no-store'});
    if(!response.ok)throw new Error('Sign-out could not be confirmed. Try again.');
    session=null;redirect('/login?reason=signed-out');
  }
  return {hosted,trainingBase,telemetryUrl,sessionFetch,getSession,logout};
}

const transport=createSessionTransport({location:typeof location==='undefined'?null:location,fetch:(...args)=>fetch(...args),redirect:url=>location.assign(url)});
export const {hosted,trainingBase,telemetryUrl,sessionFetch}=transport;
if(hosted&&typeof document!=='undefined'){
  const nav=document.querySelector('.supplement-links');
  if(nav){
    // The hosted presenter intentionally has no Qwen API route.
    const qwen=nav.querySelector('a[href="qwen.html"]');if(qwen)qwen.hidden=true;
    const button=document.createElement('button');button.type='button';button.className='text-button';button.textContent='Sign out';button.id='presenter-sign-out';nav.append(button);
    button.addEventListener('click',async()=>{button.disabled=true;try{await transport.logout();}catch(error){button.disabled=false;button.textContent='Retry sign out';button.title=error.message;}});
  }
}
