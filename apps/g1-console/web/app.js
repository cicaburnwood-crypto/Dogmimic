const $ = id => document.getElementById(id);
const names = {jump_60cm:'立定跳远 60 cm',funky_chicken:'Funky Chicken'};
let pending = false;
function render(s){
 const connected=s.connected===true, ready=connected&&s.stable_standing===true;
 for(const [skill,id] of [['jump_60cm','jump-model'],['funky_chicken','dance-model']]){
  const m=(s.models||[]).find(m=>m.skill===skill);
  $(id).textContent=m?`${m.label?m.label+' · ':''}${m.iteration.toLocaleString('zh-CN')} 次 · ${m.duration_s} s`:'模型信息不可用';
 }
 $('connection').textContent=connected?'G1 已连接':'未连接 G1';
 $('connection').classList.toggle('online',connected);
 $('ready').textContent=ready?'稳定站立':connected?'未站稳':'等待连接';
 $('ready').className='tag '+(ready?'success':'neutral');
 $('standing').textContent=ready?'站姿稳定，可以开始':connected?'等待稳定站立':'等待设备状态';
 $('pose-label').textContent=ready?'稳定站立':connected?'未处于稳定站立':'姿态未知';
 $('notice-title').textContent=ready?'选择一个动作，点击即可执行':connected?'等待 G1 稳定站立':'页面已就绪，等待接入 G1';
 $('notice-text').textContent=connected?'动作开始条件：稳定站立。':'连接后，稳定站立时可直接执行动作。';
 if(s.standing_reason&&!ready){$('notice-title').textContent=s.standing_reason;$('notice-text').textContent='请先通过原生控制让 G1 稳定站立，再点击动作按钮。';$('standing').textContent=s.standing_reason;}
 if(s.last_error){$('notice-title').textContent='动作执行异常';$('notice-text').textContent=s.last_error;}
 $('freshness').textContent=connected?'状态更新 '+new Date().toLocaleTimeString('zh-CN'):'尚未收到遥测';
 for(const key of ['height','velocity','tilt','battery']) $(key).textContent=connected&&Number.isFinite(s[key])?s[key].toFixed(key==='battery'?0:2):'—';
 $('current-action').textContent=names[s.action]||'—'; $('phase').textContent=s.phase||'—';
 $('progress').hidden=!connected||!Number.isFinite(s.progress);
 $('fill').style.width=Math.max(0,Math.min(1,s.progress||0))*100+'%';
 document.querySelectorAll('[data-action]').forEach(b=>b.disabled=!ready||pending);
 document.querySelectorAll('.button-reason').forEach(e=>e.textContent=pending?'正在提交动作':ready?'点击执行':connected?'等待稳定站立姿态':'等待连接 G1');
 const events=s.events||[]; $('event-list').replaceChildren();
 for(const event of events.length?events.slice().reverse():[{time:null,text:'尚无动作记录'}]){
  const li=document.createElement('li'),time=document.createElement('time'),text=document.createElement('span');
  time.textContent=event.time?new Date(event.time).toLocaleTimeString('zh-CN'):'—';text.textContent=event.text;li.append(time,text);$('event-list').append(li);
 }
}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error();render(await r.json());}catch{render({connected:false});$('notice-title').textContent='控制服务暂时不可达';}}
let toastTimer;
function toast(text){$('toast').textContent=text;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,5000);}
document.querySelectorAll('[data-action]').forEach(button=>button.addEventListener('click',async()=>{
 if(pending)return;pending=true;document.querySelectorAll('[data-action]').forEach(b=>b.disabled=true);
 try{const r=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:button.dataset.action})});const data=await r.json();if(!r.ok)throw Error(data.error);toast('已提交：'+names[button.dataset.action]);}
 catch(error){toast(error.message||'动作提交失败');}finally{pending=false;await refresh();}
}));
$('refresh').addEventListener('click',refresh);refresh();setInterval(refresh,1000);
