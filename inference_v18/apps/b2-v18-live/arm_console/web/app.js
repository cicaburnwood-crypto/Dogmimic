const $=id=>document.getElementById(id); let token=document.querySelector('[name=arm-token]').content;
const client=crypto.randomUUID(); let live=null, initialized=false, busy=false, polling=false;
for(let i=0;i<6;i++){$('joints').insertAdjacentHTML('beforeend',`<div class="joint-row"><div><b>J${i+1}</b><small id="limit${i}">读取限位…</small></div><span class="actual" id="actual${i}">—</span><input aria-label="J${i+1} 目标角度" id="j${i}" type="number" step="any"><button id="single${i}" class="motion">执行 J${i+1}</button></div>`);}
['X','Y','Z','Roll','Pitch','Yaw'].forEach((label,i)=>{ $(i<3?'position':'orientation').insertAdjacentHTML('beforeend',`<div class="pose-row"><b>${label}</b><span class="actual" id="poseActual${i}">—</span><input id="p${i}" aria-label="${label} 目标" type="number" step="any"></div>`); });
function note(msg,error=false){$('notice').textContent=msg;$('notice').classList.toggle('error',error);}
async function api(body){const r=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json','X-Arm-Token':token},body:JSON.stringify({client,...body})});const d=await r.json();if(d.code==='SESSION_EXPIRED'){const sessionResponse=await fetch('/api/session',{cache:'no-store'});if(sessionResponse.ok)token=(await sessionResponse.json()).token;throw Error('页面会话已更新，请重新点击操作；上一条操作没有执行');}if(!r.ok||d.error)throw Error(d.error||'请求失败');return d;}
function vals(prefix,n){return Array.from({length:n},(_,i)=>{const el=$(prefix+i);if(el.value.trim()===''||!el.checkValidity())throw Error('请输入有效目标，关节目标需在限位内');return Number(el.value);});}
function payload(kind){return kind==='joint'?{kind,joints_deg:vals('j',6)}:{kind,xyz_mm:vals('p',6).slice(0,3),rpy_deg:vals('p',6).slice(3)};}
function fmt(v,n=2){return typeof v==='number'?v.toFixed(n):'—';}
function fillJ(){if(!live?.fresh)return;live.state.reality.pose.forEach((v,i)=>{$('j'+i).value=(v*180/Math.PI).toFixed(4);});}
function fillE(){if(!live?.fresh)return;const a=live.state;const pose=a.pose_tool||a.pose;[...pose.slice(0,3).map(v=>v*1000),...live.rpy_deg].forEach((v,i)=>{$('p'+i).value=v.toFixed(4);});}
function buttons(){const ok=live?.fresh, s=live?.state||{},idle=s.state===0&&!s.task?.exe_flag&&!live?.active,ready=ok&&idle&&s.servo===1&&s.fsm_state==='POSITION'&&!live.error;
for(const id of ['moveJ','moveE',...Array.from({length:6},(_,i)=>'single'+i)])$(id).disabled=!ready||busy;
$('enable').disabled=!ok||!idle||busy||Boolean(live?.error);$('disable').disabled=!live?.connected;$('stop').disabled=!live?.connected;$('emergency').disabled=!live?.connected;
for(const id of ['previewJ','previewE','syncJ','syncE'])$(id).disabled=!ok||busy;
$('reset').disabled=!ok||busy||Boolean(s.task?.exe_flag);}
async function perform(op,extra={}){const priority=['stop','emergency','disable'].includes(op);if(!priority){busy=true;buttons();}try{const d=await api({op,...extra});if(op==='preview'){const el=$(extra.kind==='joint'?'fkResult':'ikResult');el.textContent=`TCP [mm] ${d.xyz_mm.map(x=>fmt(x)).join(' / ')} · RPY [°] ${d.rpy_deg.map(x=>fmt(x)).join(' / ')} ｜ J1–J6 [°] ${d.joints_deg.map(x=>fmt(x)).join(' / ')}`;note('求解成功，仅计算，机械臂未运动。');}else note(d.message||'操作请求已接收');}catch(e){note(e.message,true);}finally{if(!priority)busy=false;buttons();}}
for(const op of ['enable','disable','stop','emergency','reset'])$(op).onclick=()=>perform(op);
$('syncJ').onclick=fillJ;$('syncE').onclick=fillE;
for(const [id,kind,op] of [['previewJ','joint','preview'],['moveJ','joint','move'],['previewE','ee','preview'],['moveE','ee','move']])$(id).onclick=()=>{try{perform(op,{...payload(kind),speed:Number($('speed').value)});}catch(e){note(e.message,true);}};
for(let i=0;i<6;i++)$('single'+i).onclick=()=>{try{perform('move',{...payload('joint'),joint_index:i,speed:Number($('speed').value)});}catch(e){note(e.message,true);}};
async function poll(){if(polling)return;polling=true;try{await api({op:'heartbeat'});const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error('状态读取失败');live=await r.json();const s=live.state;
$('lamp').classList.toggle('good',live.fresh);$('connection').textContent=live.fresh?'实时连接':'连接中断';$('model').textContent=(s.arm_name||'A3')+' · '+(live.fresh?live.age_ms+' ms':live.problem||'状态已过期');
$('servo').textContent=s.servo===1?'已使能':'未使能';$('mode').textContent=s.fsm_state||'—';$('task').textContent=live.active||s.task?.exe_flag?'执行中':s.state===-1?'故障':'空闲';$('tool').textContent='TCP '+(s.tool??'—');$('gripper').textContent=s.eeff?.is_connect?'已连接':'未连接';
(s.reality?.pose||[]).forEach((v,i)=>{$('actual'+i).textContent=fmt(v*180/Math.PI);if(live.limits){let lo=live.limits.limit_lower[i]*180/Math.PI,hi=live.limits.limit_upper[i]*180/Math.PI;$('limit'+i).textContent=fmt(lo,1)+'° … '+fmt(hi,1)+'°';$('j'+i).min=lo;$('j'+i).max=hi;}});
const pose=s.pose_tool||s.pose;if(pose&&live.rpy_deg)[...pose.slice(0,3).map(v=>v*1000),...live.rpy_deg].forEach((v,i)=>{$('poseActual'+i).textContent=fmt(v);});
if(!initialized&&live.fresh&&live.limits){fillJ();fillE();initialized=true;}
$('toolInfo').textContent='当前工具 TCP '+(s.tool??'—')+'，工具相对法兰配置：'+JSON.stringify(live.tool_coordinate)+'（XYZ 单位 m，四元数 XYZW）。';
$('events').replaceChildren(...live.events.filter(e=>e.kind!=='request').slice(-6).reverse().map(e=>{let div=document.createElement('div');div.textContent=e.time+' · '+(typeof e.detail==='string'?e.detail:JSON.stringify(e.detail));return div;}));
if(live.error)note('控制器错误：'+(live.errMsg||live.error),true);
}catch(e){if(live)live.fresh=false;note('通信中断：'+e.message,true);$('connection').textContent='连接中断';$('lamp').classList.remove('good');}finally{polling=false;buttons();}}
poll();setInterval(poll,250);
