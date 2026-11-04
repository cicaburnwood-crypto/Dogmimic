'use strict';
let state=null,lastReceipt=0,timer=null,lastRevision=null;
const $=id=>document.getElementById(id);
const names={native:'原生 locomotion',trained:'自训练 locomotion',hindstand:'后腿站立',attitude:'姿态编排',unowned:'未接管运动控制',unknown:'状态未知'};
const phases={idle:'待机',await_r1:'等待双击 R1',handover:'控制权交接中',handover_failed:'原生交接未完成',classic:'经典模式',hindstand:'动作执行中',attitude:'姿态编排执行中',recovering:'四足恢复中',waiting_input:'等待模型输入恢复',damping:'阻尼模式'};
const format=(v,n=2)=>Number.isFinite(v)?v.toFixed(n):'—';
function toast(message,error=false){const el=$('toast');el.textContent=message;el.classList.toggle('error',error);el.hidden=false;clearTimeout(timer);timer=setTimeout(()=>el.hidden=true,6500);}
function disableUnavailable(){document.querySelectorAll('[data-select],#plan-confirm').forEach(b=>b.disabled=true);}
function lostConnection(){
 if(state)state.connected=false;
 disableUnavailable();$('connection').className='connection';$('connection').replaceChildren(document.createElement('i'),document.createTextNode('控制服务未连接'));
 $('notice').className='notice warning';$('notice-title').textContent='未连接控制服务';$('notice-text').textContent='暂时无法读取当前姿态，连接恢复后自动更新。';$('backend-tag').textContent='未连接';
 $('active-policy').textContent='状态未知';$('phase').textContent='—';$('pose-label').textContent='姿态未知';$('pose-dot').classList.remove('ready');
 $('gate-status').textContent='等待数据';$('gate-status').className='tag';$('switch-condition').textContent='等待姿态数据';$('stunt-condition').textContent='等待姿态数据';
 ['height','velocity','tilt','battery'].forEach(id=>$(id).textContent='—');
 ['telemetry','posture','stationary'].forEach(key=>{$('check-'+key).classList.remove('pass');$('check-'+key).querySelector('.check-icon').textContent='–';});
 $('freshness').textContent='未收到当前遥测';$('remote-dot').classList.remove('live');$('remote-live').textContent='等待遥控数据';$('remote-buttons').textContent='—';
 $('plan-reason').textContent='可以继续编辑编排；执行需当前稳定站姿数据。';
}
function render(s){
 state=s;const g=s.gate||{},online=s.connected;
 const attitude=s.model_info?.models?.attitude;
 $('attitude-model').textContent=attitude?`姿态 · ${attitude.iteration.toLocaleString('zh-CN')} 次`:'姿态模型未加载';
 $('attitude-validation').textContent=attitude?`实验模型 · 基础评测 ${attitude.simulation_strict_passed} / ${attitude.simulation_strict_total}；脚位偏移尚未达标，未完成实机验证。`:'等待姿态模型';
 $('connection').className='connection'+(online?' online':'');$('connection').replaceChildren(document.createElement('i'),document.createTextNode(online?'机器人已连接':'机器人未连接'));
 $('backend-tag').textContent=s.backend==='robot'?'机器人端服务':'本地预览';
 $('notice').className='notice'+(!online||!g.switch_allowed||s.error?' warning':'');
 if(s.error){$('notice-title').textContent='当前操作状态';$('notice-text').textContent=s.error;}
 else if(!online){$('notice-title').textContent='等待连接 B2';$('notice-text').textContent='收到当前本体数据后判断站立姿态。';}
 else if(s.phase==='handover'){$('notice-title').textContent='正在恢复原生遥控';$('notice-text').textContent='正在确认原生经典行走模式的接管结果。';}
 else if(s.phase==='neutralizing'){$('notice-title').textContent='正在回正站稳';$('notice-text').textContent='由当前模型回正，站稳后恢复原生经典模式。趴下、恢复和阻尼按键仍可使用。';}
 else if(s.active==='attitude'){$('notice-title').textContent='正在执行已确认的编排';$('notice-text').textContent='下方显示当前步骤。编辑待执行编排不会改变本次动作。';}
 else if(!g.switch_allowed){$('notice-title').textContent='等待四足站立';$('notice-text').textContent='可提前编排动作，稳定站立后点击确定执行。';}
 else{$('notice-title').textContent=g.attitude_allowed?'站立稳定，可以执行编排':'站立姿态，可以切换策略';$('notice-text').textContent=g.attitude_allowed?'添加方向与角度、调整顺序，点击“确定并执行”开始。':'编排执行需稳定站立；可以先编辑动作。';}
 $('pose-label').textContent=!online?'姿态未知':g.posture?(g.stationary?'四足站立 · 稳定':'四足站立'):'未达站立姿态';$('pose-dot').classList.toggle('ready',Boolean(g.switch_allowed));
 $('height').textContent=online?format(g.height_m):'—';$('velocity').textContent=online?format(g.speed_m_s):'—';$('tilt').textContent=online?format(g.tilt_deg,1):'—';$('battery').textContent=online?format(s.battery,0):'—';
 $('freshness').textContent=online?'数据时龄 '+format(s.telemetry_age_ms,0)+' ms':'未收到当前遥测';
 for(const key of ['telemetry','posture','stationary']){const el=$('check-'+key),pass=online&&g[key];el.classList.toggle('pass',Boolean(pass));el.querySelector('.check-icon').textContent=pass?'✓':'–';}
 $('switch-condition').textContent=g.switch_allowed?'可以切换':'等待站立';$('stunt-condition').textContent=g.hindstand_allowed?'可以执行':'等待稳定站立';
 $('gate-status').textContent=g.switch_allowed?'站立姿态':'等待站立';$('gate-status').className='tag'+(g.switch_allowed?' success':'');
 document.querySelectorAll('[data-select]').forEach(button=>{const selected=button.dataset.select===s.selected;button.classList.toggle('selected',selected);button.setAttribute('aria-pressed',String(selected));button.disabled=!s.can_select;button.querySelector('.selection-label').textContent=selected?'当前选择':'选择策略';});
 $('selection-help').textContent=s.phase==='await_r1'?'已选择'+names[s.selected]+'，双击 R1 进入经典模式。':g.switch_allowed?'网站选择策略，遥控器进入经典模式。':'需先处于四足站立姿态。';
 $('active-policy').textContent=names[s.active]||'状态未知';$('phase').textContent=phases[s.phase]||s.phase;
 $('trained-note').textContent='自训练速度：合成 XY ≤ 2 m/s，侧向 ≤ 0.6 m/s。';
 if(s.tilt_control?.enabled){
  const tc=s.tilt_control,m=s.model_info?.models?.locomotion,angles=tc.requested_degrees;
  $('selection-help').textContent='稳定站立时按 R2 + Y 双向切换；开机默认原生，不需要 PC。';
  $('trained-note').textContent=`v2 · 第 ${m?.iteration?.toLocaleString('zh-CN')||'—'} 次 · 倾角设定 roll ${angles[0].toFixed(1)}° / pitch ${angles[1].toFixed(1)}°。高档前后 ±4 m/s、侧向 ±1 m/s，合成 ≤4 m/s；角速度随速度限制在训练范围内。模型尚未通过全部物理验收。`;
  $('switch-condition').textContent=tc.switch_allowed?'可按 R2 + Y 切换':'等待稳定站立';
  if(s.active==='trained'&&tc.switch_allowed&&!tc.returning&&!s.error){$('notice-title').textContent='自训练模式 · 稳定站立';$('notice-text').textContent='方向键设置倾角；按 R2 + Y 回正并切回原生。';$('pose-label').textContent='四足支撑 · 稳定';}
  document.querySelector('[data-select="trained"] .strategy-bottom .tag').textContent=`${m?.iteration?.toLocaleString('zh-CN')||'—'} 次模型`;
 }
 renderPlanState();
 const neutral=Object.values(s.remote.axes).every(v=>Math.abs(v)<.05);
 $('remote-dot').classList.toggle('live',online);$('remote-live').textContent=online?'遥控数据已接收 · '+(s.speed_level==='high'?'高速':'低速'):'等待遥控器数据';$('remote-buttons').textContent=online?(s.remote.buttons.join(' + ')||'摇杆 '+(neutral?'居中':'有输入')):'—';
 if(lastRevision!==s.revision){$('event-list').replaceChildren(...s.events.slice(0,12).map(e=>{const li=document.createElement('li');li.className=e.kind;const t=document.createElement('time');t.textContent=new Date(e.time*1000).toLocaleTimeString('zh-CN',{hour12:false});const span=document.createElement('span');span.textContent=e.message;li.append(t,span);return li;}));lastRevision=s.revision;}
 $('updated-at').textContent='状态更新于 '+new Date().toLocaleTimeString('zh-CN',{hour12:false});
}
async function poll(){try{const r=await fetch('/api/status',{cache:'no-store',signal:AbortSignal.timeout(1800)});if(!r.ok)throw new Error('状态读取失败');const data=await r.json();lastReceipt=performance.now();render(data);}catch(e){lostConnection();}finally{setTimeout(poll,250);}}
async function command(action,value){
 if(!state)return;
 try{const body={id:crypto.randomUUID(),action};if(value!==undefined)body.value=value;
 const r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json','X-B2-CSRF':state.csrf},body:JSON.stringify(body),signal:AbortSignal.timeout(8000)});
 const result=await r.json();if(!r.ok)throw new Error(result.error||'操作未完成');
 toast(action==='select'?'策略已选择；双击 R1 进入经典模式。':'后腿站立请求已接受。');
 }catch(e){toast(e.name==='TimeoutError'?'请求超时，执行状态尚未确认。':e.message,true);}
}
const directions={left:{axis:'roll',sign:-1,label:'左倾'},right:{axis:'roll',sign:1,label:'右倾'},forward:{axis:'pitch',sign:1,label:'前俯'},backward:{axis:'pitch',sign:-1,label:'后仰'}};
const draftKey='b2-attitude-plan-v1';
let draft={mode:'return_each',actions:[]},sendingPlan=false,pendingPlan=null;
function actionText(a){const direction=a.axis==='roll'?(a.degrees<0?'左倾':'右倾'):(a.degrees<0?'后仰':'前俯');return a.degrees===0?'回正 0°':`${direction} ${Math.abs(a.degrees)}°`;}
try{const saved=JSON.parse(localStorage.getItem(draftKey));if(saved&&['return_each','continuous'].includes(saved.mode)&&Array.isArray(saved.actions)&&saved.actions.every(a=>['roll','pitch'].includes(a.axis)&&Number.isFinite(a.degrees)&&Math.abs(a.degrees)<=30))draft=saved;}catch{}
function renderDraft(){
 $('plan-mode').value=draft.mode;$('plan-count').textContent=`${draft.actions.length} 步`;$('plan-empty').hidden=draft.actions.length>0;
 $('plan-list').replaceChildren(...draft.actions.map((a,i)=>{const li=document.createElement('li'),number=document.createElement('span'),description=document.createElement('span'),buttons=document.createElement('span');number.className='step-number';number.textContent=String(i+1).padStart(2,'0');description.className='step-description';description.textContent=actionText(a);buttons.className='step-tools';
  for(const [label,text,offset] of [['上移','↑',-1],['下移','↓',1],['删除','×',0]]){const b=document.createElement('button');b.type='button';b.textContent=text;b.setAttribute('aria-label',`${label}第 ${i+1} 步`);b.disabled=(offset===-1&&i===0)||(offset===1&&i===draft.actions.length-1);b.addEventListener('click',()=>{if(offset){[draft.actions[i],draft.actions[i+offset]]=[draft.actions[i+offset],draft.actions[i]];}else draft.actions.splice(i,1);draftChanged();});buttons.append(b);}
  li.append(number,description,buttons);return li;}));
 const separator=draft.mode==='return_each'?' → 回正 → ':' → ';
 $('plan-preview').textContent=draft.actions.length?draft.actions.map(actionText).join(separator)+' → 回正 → 原生遥控':'尚未添加动作';
 $('plan-clear').disabled=draft.actions.length===0;renderPlanState();
}
function draftChanged(){pendingPlan=null;try{localStorage.setItem(draftKey,JSON.stringify(draft));}catch{}renderDraft();}
function renderPlanState(){
 const s=state,online=Boolean(s?.connected)&&performance.now()-lastReceipt<2000,ready=Boolean(s?.capabilities?.attitude?.ready);
 $('plan-confirm').disabled=sendingPlan||!online||!s?.can_attitude||!ready||draft.actions.length===0;
 $('plan-confirm').textContent=sendingPlan?'正在提交…':pendingPlan?'重试确认（同一请求）':`确定并执行${draft.actions.length?' · '+draft.actions.length+' 步':''}`;
 $('plan-reason').textContent=!draft.actions.length?'先添加方向和角度，编辑不会执行。':!online?'编排已保存在本机，等待机器人连接。':!ready?'姿态模型尚未加载。':!s.can_attitude?'编排已准备，等待稳定四足站立。':'当前稳定站立；点击确定后执行上方整份编排。';
 const e=s?.execution,show=e?.skill==='attitude';$('plan-execution').hidden=!show;
 if(show){const labels={prepare:'准备站立',execute:`执行第 ${e.step_number} / ${e.step_count} 步`,recover:`第 ${e.step_number} 步后回正`,complete:e.native_restored?'编排完成 · 原生遥控已恢复':'动作已回正 · 原生交接中'};let label=labels[e.plan_phase]||'执行状态';if(s.active!=='attitude'&&!e.complete)label='编排已被遥控操作中断';if(e.native_return?.status==='failed')label='动作已回正 · 原生交接未完成';if(s.phase==='waiting_input')label='等待模型输入恢复';if(s.phase==='damping')label='遥控器已切换至阻尼模式';
  $('plan-execution-title').textContent=label;$('plan-execution-time').textContent=`${format(e.policy_time_s)} s`;
  $('plan-execution-steps').textContent='已确认：'+(e.plan?.actions||[]).map((a,i)=>`${i+1}. ${actionText(a)}`).join(' → ');
  const angles=a=>Array.isArray(a)?a.map(v=>format(v,1)+'°').join(' / '):'—';
  $('attitude-target').textContent=online?angles(e.requested_degrees):'—';$('attitude-actual').textContent=online&&s.active==='attitude'?angles(e.actual_degrees):'—';
  $('plan-execution-detail').textContent=e.native_restored?'已恢复原生经典模式，可直接使用遥控器移动。再次执行编排需要重新点击确定。':e.native_return?.status==='failed'?'原生交接未完成：'+e.native_return.error:e.complete?'正在等待原生经典行走模式确认，尚未报告编排完成。':'编辑上方草稿不会改变已经确认的任务。';
 }
}
$('plan-add-form').addEventListener('submit',e=>{e.preventDefault();const input=$('plan-angle');if(!input.reportValidity())return;const magnitude=Number(input.value),d=directions[$('plan-direction').value];if(!Number.isFinite(magnitude)||magnitude<0||magnitude>30)return;draft.actions.push({axis:d.axis,degrees:magnitude*d.sign});draftChanged();});
$('plan-clear').addEventListener('click',()=>{draft.actions=[];draftChanged();});
$('plan-mode').addEventListener('change',()=>{draft.mode=$('plan-mode').value;draftChanged();});
$('plan-confirm').addEventListener('click',async()=>{
 if(sendingPlan||$('plan-confirm').disabled)return;
 if(!pendingPlan)pendingPlan={id:crypto.randomUUID(),action:'attitude',value:JSON.parse(JSON.stringify(draft))};
 const request=pendingPlan;sendingPlan=true;renderPlanState();
 try{const response=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json','X-B2-CSRF':state.csrf},body:JSON.stringify(request),signal:AbortSignal.timeout(8000)});const result=await response.json();if(!response.ok){pendingPlan=null;throw new Error(result.error||'编排未被接受');}pendingPlan=null;toast(`已确认 ${request.value.actions.length} 步，按顺序执行。`);await pollOnce();}
 catch(error){toast(error.name==='TimeoutError'?'提交超时，请看执行进度；重试会使用同一请求编号。':error.message,true);}
 finally{sendingPlan=false;renderPlanState();}
});
async function pollOnce(){const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw new Error('状态读取失败');lastReceipt=performance.now();render(await r.json());}
renderDraft();
document.querySelectorAll('[data-select]').forEach(b=>b.addEventListener('click',()=>command('select',b.dataset.select)));
$('mapping-toggle').addEventListener('click',()=>{const expanded=$('mapping-details').hidden;$('mapping-details').hidden=!expanded;$('mapping-toggle').setAttribute('aria-expanded',String(expanded));$('mapping-toggle').textContent=expanded?'收起映射 −':'全部映射 +';});
$('refresh').addEventListener('click',async()=>{try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw new Error();render(await r.json());lastReceipt=performance.now();}catch{lostConnection();}});
setInterval(()=>{if(lastReceipt&&performance.now()-lastReceipt>2000)lostConnection();},200);
poll();
