import {Keys,editable,SpaceToggle} from './keyboard.mjs?v=gripper1';
import {readManualGoal,canExecutePlan} from './manual-goal.mjs?v=manual1';
const $=s=>document.querySelector(s),token=$('meta[name="token"]').content,client=crypto.randomUUID(),canvas=$('#view'),ctx=canvas.getContext('2d'),keys=new Keys(),space=new SpaceToggle();
let state=null,plan=null,target=[.6,0,.75],targetQuat=null,az=-.8,el=.65,scale=190,drag=null,dirty=true,version=0,pending=false,enabled=false,seq=0,inputPending=false,enableGeneration=0,initialized=false,requestFault=null,standbyPending=false,foldPending=false,foldRequestId=null,gripperPending=false,remoteBlocked=true,remoteGeneration=null,resumeKeyboard=false;
const message=s=>$('#job').textContent=s;
async function action(op,values={},keepalive=false){const r=await fetch('/api/action',{method:'POST',keepalive,headers:{'Content-Type':'application/json','X-V18-Token':token},body:JSON.stringify({op,client,...values})});const d=await r.json();if(!r.ok){const e=Error(d.error||'请求失败');e.detail=d;throw e;}return d;}
function showError(e){requestFault=e.detail||{message:e.message};message(e.message);drawDiagnostics();}
function manualError(e=null){const box=$('#manual-error');box.hidden=!e;box.textContent=e?[e.message,...(e.detail?.issues||[]).map(i=>`${i.joint?i.joint+'：':''}${i.message}`)].filter((v,i,a)=>v&&a.indexOf(v)===i).join('\n'):'';}
function invalidate(){version++;plan=null;$('#execute').disabled=true;$('#execute').textContent='2 · 确认执行此目标';$('#plan').textContent='目标已更新，请重新预览。';$('#plan-detail').hidden=true;manualError();}
function setTarget(t,rpy){target=t.slice();targetQuat=null;['x','y','z'].forEach((id,i)=>$('#'+id).value=(t[i]*100).toFixed(2));if(rpy)['r','p','yaw'].forEach((id,i)=>$('#'+id).value=rpy[i].toFixed(2));invalidate();dirty=true;}
const manualValues=()=>Object.fromEntries(['x','y','z','r','p','yaw'].map(id=>[id,$('#'+id).value]));
for(const id of ['x','y','z','r','p','yaw'])$('#'+id).oninput=()=>{invalidate();target=['x','y','z'].map(id=>$('#'+id).value.trim()?Number($('#'+id).value)/100:NaN);targetQuat=null;dirty=true;};
$('#current').onclick=()=>{if(state?.current)setTarget(state.current.ee,state.current.rpy_deg);};
async function preview(prepare){
 if(pending||remoteBlocked||state?.busy||state?.keyboard?.enabled)return;
 invalidate();const revision=version,started=Date.now();pending=true;requestFault=null;
 try{
  // Read the displayed form, never the last keyboard goal held in target.
  const values=prepare?{}:readManualGoal(manualValues());
  if(!prepare){target=values.xyz;targetQuat=null;dirty=true;}
  $('#plan').textContent='正在核对实时状态并计算首步动作…';message('正在核对实时状态…');drawManual();
  const d=await action(prepare?'preview_prepare':'preview',values);
  if(revision!==version)return;
  plan={...d.plan,expiresAt:started+120000};
  $('#plan').textContent=prepare?`准备关节目标 / °：${plan.home.map(v=>(v*180/Math.PI).toFixed(1)).join(' / ')}\n${plan.notice}`:`目标 X / Y / Z：${plan.xyz.map(v=>(v*100).toFixed(2)).join(' / ')} cm\n朝向 Roll / Pitch / Yaw：${plan.rpy_deg.map(v=>v.toFixed(2)).join(' / ')}°\n预览完成，尚未运动。确认后联合控制狗与机械臂。`;
  if(!prepare){
   const c=plan.first_body_command;
   $('#plan-commands').textContent=`机体平移 X / Y：${c.slice(0,2).map(v=>v.toFixed(3)).join(' / ')} m/s\n转向角速度：${(c[2]*180/Math.PI).toFixed(2)} °/s\n机体 Roll / Pitch：${c.slice(3,5).map(v=>(v*180/Math.PI).toFixed(2)).join(' / ')}° · 高度调整 ${(c[5]*100).toFixed(2)} cm\n机械臂 J1–J6 首步变化：${plan.first_arm_delta_deg.map(v=>v.toFixed(3)).join(' / ')}°\n${plan.arm_projection_allowed?'该首步机械臂动作通过当前几何检查。':'该首步机械臂动作被几何约束限制，将保持原目标；不代表目标可达。'}\n${plan.notice}`;
   $('#plan-detail').hidden=false;
  }
  $('#execute').textContent=prepare?'确认准备机械臂（5%速度）':'2 · 确认执行此目标';
  message('预览完成，尚未执行动作。');
  if(prepare)$('#manual').scrollIntoView({behavior:'smooth',block:'start'});
 }catch(e){$('#plan').textContent='未得到有效预览，请按错误提示修正后重试。';manualError(e);showError(e);}
 finally{pending=false;drawManual();}
}
$('#preview').onclick=()=>preview(false);$('#prepare').onclick=()=>preview(true);
$('#execute').onclick=async()=>{
 if(!canExecutePlan(plan,manualContext()))return;
 const selected=plan;plan=null;pending=true;requestFault=null;manualError();drawManual();
 try{await action('execute',{id:selected.id});$('#plan').textContent='固定目标已提交，正在根据真实反馈执行；可随时用遥控器接管或点击停止。';message('执行请求已提交');}
 catch(e){$('#plan').textContent='执行未启动，请修正后重新预览。';manualError(e);showError(e);}
 finally{pending=false;drawManual();}
};
function manualContext(){return{pending,blocked:remoteBlocked||!!state?.error,busy:!!state?.busy,keyboard:!!state?.keyboard?.enabled,jointReady:!!state?.joint_ready};}
function drawManual(){
 const c=manualContext();
 if(plan&&Date.now()>=plan.expiresAt){invalidate();$('#plan').textContent='预览已超过 2 分钟，请重新预览。';}
 const blocked=c.blocked||c.busy||c.keyboard;
 for(const id of ['current','x','y','z','r','p','yaw'])$('#'+id).disabled=blocked;
 for(const id of ['preview','prepare','reset'])$('#'+id).disabled=blocked||pending;
 $('#execute').disabled=!canExecutePlan(plan,c);
 $('#manual-state').textContent=c.blocked?'遥控优先 / 暂停':c.busy?'执行中':c.keyboard?'键盘控制中':pending?'处理中':plan?'待确认':'待预览';
 $('#manual-state').className=plan&&!blocked?'badge on':'badge';
 $('#manual-hint').textContent=c.blocked?'遥控器操作期间其他控制暂停；完全松开 2 秒后恢复。已取消的目标需要重新预览。':c.busy?'正在执行联合动作；遥控器可随时接管，或使用停止按钮。':c.keyboard?'先关闭键盘控制，再输入数值目标。':plan&&!c.jointReady&&plan.kind!=='prepare'?'预览已完成；请先进入联合待命，待机器人稳定后重新预览再执行。':'一次提交固定目标，执行时联合策略根据真实反馈持续修正；预览不代表整条轨迹已验证可达。修改输入后需要重新预览。';
}
async function toggleGripper(){
 if(gripperPending||remoteBlocked)return;
 const g=state?.gripper;
 if(!g?.ready){$('#gripper-message').textContent=g?.error||'夹爪反馈尚未就绪';return;}
 gripperPending=true;drawGripper();
 const target=g.next_open?'open':'closed';
 try{const d=await action('gripper_set',{target,id:crypto.randomUUID()});state.gripper=d.gripper;$('#gripper-message').textContent=d.message;}
 catch(e){$('#gripper-message').textContent=e.message;}
 finally{gripperPending=false;drawGripper();}
}
$('#gripper-toggle').onclick=toggleGripper;
function drawGripper(){
 const g=state?.gripper,button=$('#gripper-toggle');
 button.disabled=!!(remoteBlocked||gripperPending||g?.pending||!g?.ready||(state?.busy&&['standby','fold'].includes(state?.job?.operation)));
 button.textContent=gripperPending||g?.pending?'等待夹爪应答…':`空格 · ${g?.next_open===false?'闭合夹爪':'张开夹爪'}`;
 const phases={idle:'待命',pending:'等待应答',accepted:'请求已接收',moving:'向目标运动',at_target:'位置已到达（±5 mm）',paused:'遥控器接管，夹爪已保持当前开口',not_reached:'未到目标：可能接触物体或受阻',unconfirmed:'指令结果未确认'};
 $('#gripper-feedback').textContent=g?.ready?`实际开口 ${g.position_mm.toFixed(1)} mm${g.target_mm!==null?` · 目标 ${g.target_mm.toFixed(0)} mm`:''}\n${phases[g.phase]||g.phase}`:g?.error||'等待夹爪反馈';
}
function clearKeys(){keys.clear();paintKeys();}
function paintKeys(){document.querySelectorAll('kbd[data-code]').forEach(k=>k.classList.toggle('pressed',keys.held.has(k.dataset.code)));}
function localDisable(){enabled=false;enableGeneration++;clearKeys();$('#keyboard-state').textContent='未启用';$('#keyboard-state').className='badge';}
async function disable(reason='键盘已关闭'){resumeKeyboard=false;const was=enabled;localDisable();if(was){try{await action('keyboard_disable',{},true);message(reason);}catch(e){showError(e);}}}
async function stop(){resumeKeyboard=false;standbyPending=false;foldPending=false;localDisable();invalidate();try{const d=await action('stop',{},true);message(d.stop_errors?.length?d.stop_errors.join('；'):'停止请求已发送');}catch(e){showError(e);}}
$('#enter-standby').onclick=async()=>{if(pending)return;pending=true;standbyPending=true;requestFault=null;invalidate();const generation=++enableGeneration;$('#enter-standby').disabled=true;try{await action('heartbeat');if(generation!==enableGeneration||document.hidden||!document.hasFocus()){standbyPending=false;return;}await action('enter_standby');message('准备流程已开始：机械臂待命 → P4 零速站立。');if(generation!==enableGeneration||document.hidden||!document.hasFocus())await stop();}catch(e){standbyPending=false;showError(e);}finally{pending=false;}};
$('#enter-p4-standby').onclick=async()=>{if(pending)return;pending=true;standbyPending=true;requestFault=null;invalidate();const generation=++enableGeneration;$('#enter-p4-standby').disabled=true;try{await action('heartbeat');if(generation!==enableGeneration||document.hidden||!document.hasFocus()){standbyPending=false;return;}await action('enter_p4_standby');message('原地准备已开始：保持 P4，机械臂进入预备位。');if(generation!==enableGeneration||document.hidden||!document.hasFocus())await stop();}catch(e){standbyPending=false;showError(e);}finally{pending=false;}};
function stopStandbyOnBlur(){if(standbyPending||foldPending||(state?.busy&&['standby','fold'].includes(state?.job?.operation)))stop();}
$('#fold-arm').onclick=async()=>{
 if(pending||remoteBlocked||state?.busy||state?.keyboard?.enabled)return;
 pending=true;foldPending=true;foldRequestId=null;requestFault=null;resumeKeyboard=false;invalidate();
 const generation=++enableGeneration;drawFold();
 try{
  await action('heartbeat');
  if(generation!==enableGeneration||document.hidden||!document.hasFocus()){foldPending=false;return;}
  const accepted=await action('fold_arm');foldRequestId=accepted.fold_id;message('折叠已提交：检查路径后以 5% 速度执行。');
  if(generation!==enableGeneration||document.hidden||!document.hasFocus())await stop();
 }catch(e){foldPending=false;showError(e);$('#fold-status').textContent=e.message;}
 finally{pending=false;drawFold();}
};
function drawFold(){
 const running=state?.busy&&state?.job?.operation==='fold',finished=!state?.busy&&state?.job?.phase==='folded';
 $('#fold-arm').disabled=!!(pending||foldPending||remoteBlocked||state?.busy||state?.keyboard?.enabled||!state||state.error||state.fold_error);
 $('#fold-arm').textContent=running||foldPending?'正在检查 / 折叠…':'一键折叠机械臂 · 5% 速度';
 $('#fold-state').textContent=running?'折叠中':finished?'已折叠':'待命';
 $('#fold-state').className=finished?'badge on':'badge';
 $('#fold-status').textContent=state?.job?.operation==='fold'?(state.job.message+(state.job.joint_error_deg!==undefined?` · 最大关节误差 ${state.job.joint_error_deg.toFixed(2)}°`:'')):state?.keyboard?.enabled?'请先关闭键盘控制并停止 EE 运动，再点击折叠。':state?.fold_error||'点击即检查路径并执行；不会切换狗的运动模式，也不改变夹爪开合。';
}
$('#stop').onclick=stop;$('#keyboard-disable').onclick=()=>disable();
async function enableKeyboard(){if(pending)return;pending=true;requestFault=null;const generation=++enableGeneration;try{await action('heartbeat');const d=await action('keyboard_enable');if(generation!==enableGeneration||document.hidden||!document.hasFocus()){await action('keyboard_disable',{},true);return;}enabled=true;seq=0;clearKeys();invalidate();target=d.keyboard.xyz;targetQuat=d.keyboard.quaternion_xyzw;message('已启用。按住映射键开始控制 EE，松键停止累加目标。');await sendKeys();}catch(e){showError(e);}finally{pending=false;}};
$('#keyboard-enable').onclick=enableKeyboard;
async function sendKeys(){if(!enabled||inputPending)return;inputPending=true;try{await action('keyboard_input',{seq:seq++,keys:keys.list()});}catch(e){if(enabled){if(e.detail?.code==='REMOTE_PRIORITY')resumeKeyboard=true;localDisable();showError(e);action('keyboard_disable',{},true).catch(()=>{});}}finally{inputPending=false;}}
setInterval(sendKeys,80);
window.addEventListener('keydown',e=>{if(remoteBlocked&&e.code!=='Escape'){if((e.code==='Space'||e.code.startsWith('Arrow')||keys.held.has(e.code))&&!editable(e.target))e.preventDefault();return;}if(e.code==='Space'){if(space.down(e))toggleGripper();return;}if(e.code==='Escape'&&(enabled||state?.busy)&&!editable(e.target)){e.preventDefault();if(!e.repeat)stop();return;}if(keys.down(e,enabled)){e.preventDefault();paintKeys();sendKeys();}});
window.addEventListener('keyup',e=>{space.up(e);if(keys.up(e)){e.preventDefault();paintKeys();sendKeys();}});
window.addEventListener('blur',()=>{space.clear();enableGeneration++;stopStandbyOnBlur();disable('页面失焦，键盘已关闭并停止联合动作');});
document.addEventListener('visibilitychange',()=>{if(document.hidden){space.clear();enableGeneration++;stopStandbyOnBlur();disable('页面隐藏，键盘已关闭并停止联合动作');}});
window.addEventListener('pagehide',()=>{space.clear();stopStandbyOnBlur();disable();});
document.addEventListener('focusin',e=>{if(enabled&&editable(e.target))disable('已进入输入框，键盘运动控制已关闭');});
$('#release').onclick=async()=>{try{await disable();await action('release');invalidate();message('网页控制已释放，遥控器可操作；R2+Y 切换原生。');}catch(e){showError(e);}};
$('#reset').onclick=()=>action('reset').then(()=>{initialized=false;invalidate();message('参考原点已重置。');}).catch(showError);
function project(p){const x=Math.cos(az)*p[0]-Math.sin(az)*p[1],y=Math.sin(az)*p[0]+Math.cos(az)*p[1];return [canvas.clientWidth/2+x*scale,canvas.clientHeight*.73+(y*Math.sin(el)-p[2]*Math.cos(el))*scale];}
function line(a,b,color,width=1){const p=project(a),q=project(b);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();}
function dot(p,color,r){const q=project(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(...q,r,0,2*Math.PI);ctx.fill();}
function pointing(p,q,color){if(!q)return;const[x,y,z,w]=q,axis=[2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],end=p.map((v,i)=>v+.18*axis[i]);line(p,end,color,2.5);const a=project(p),b=project(end),t=Math.atan2(b[1]-a[1],b[0]-a[0]);ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(b[0]-8*Math.cos(t-.4),b[1]-8*Math.sin(t-.4));ctx.lineTo(...b);ctx.lineTo(b[0]-8*Math.cos(t+.4),b[1]-8*Math.sin(t+.4));ctx.stroke();}
function formQuat(){if(['r','p','yaw'].some(id=>!$('#'+id).value.trim()||!Number.isFinite(Number($('#'+id).value))))return null;const[r,p,y]=['r','p','yaw'].map(id=>Number($('#'+id).value)*Math.PI/360),cr=Math.cos(r),sr=Math.sin(r),cp=Math.cos(p),sp=Math.sin(p),cy=Math.cos(y),sy=Math.sin(y);return[sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy];}
function draw(){if(dirty){dirty=false;const w=canvas.clientWidth,h=canvas.clientHeight,dpr=devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);for(let i=-3;i<=3;i+=.5){line([-3,i,0],[3,i,0],'#223744');line([i,-3,0],[i,3,0],'#223744');}line([0,0,0],[1,0,0],'#d5867c',2);line([0,0,0],[0,1,0],'#7ebd98',2);line([0,0,0],[0,0,1.5],'#849fd0',2);for(const [p,t]of [[[1.1,0,0],'X 前'],[[0,1.1,0],'Y 左'],[[0,0,1.55],'Z 上']]){ctx.fillStyle='#b0c7d6';ctx.fillText(t,...project(p));}const c=state?.current;if(c){dot(c.ee,'#75f3ad',6);dot(c.body_xyz,'#a5b9c1',5);line(c.body_xyz,[...c.body_xyz.slice(0,2),0],'#657d8b');line(c.ee,c.body_xyz,'#75f3ad60');pointing(c.ee,c.quaternion_xyzw,'#75f3ad');}if(target.every(Number.isFinite)){dot(target,'#ffbd6e',6);line(target,[target[0],target[1],0],'#a78256');pointing(target,targetQuat||formQuat(),'#ffbd6e');}}requestAnimationFrame(draw);}requestAnimationFrame(draw);
canvas.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,sx:e.clientX,sy:e.clientY};canvas.setPointerCapture(e.pointerId);};canvas.onpointermove=e=>{if(!drag)return;az+=(e.clientX-drag.x)*.008;el=Math.max(.15,Math.min(1.4,el+(e.clientY-drag.y)*.005));drag.x=e.clientX;drag.y=e.clientY;dirty=true;};canvas.onpointerup=e=>{if(!enabled&&!state?.busy&&!state?.keyboard?.enabled&&drag&&Math.hypot(e.clientX-drag.sx,e.clientY-drag.sy)<4){const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left-canvas.clientWidth/2)/scale,y=((e.clientY-r.top-canvas.clientHeight*.73)/scale+target[2]*Math.cos(el))/Math.sin(el);setTarget([Math.cos(az)*x+Math.sin(az)*y,-Math.sin(az)*x+Math.cos(az)*y,target[2]]);$('#manual').scrollIntoView({behavior:'smooth',block:'start'});}drag=null;};canvas.onwheel=e=>{e.preventDefault();scale=Math.max(65,Math.min(700,scale*Math.exp(-e.deltaY*.001)));dirty=true;};window.onresize=()=>dirty=true;
function drawDiagnostics(){const d=$('#diagnostics');d.replaceChildren();const faults=[requestFault,state?.start_fault,state?.job?.phase==='stopped'||state?.job?.phase==='timeout'?state.job:null,state?.keyboard?.notice].filter(Boolean);const issues=[...faults.flatMap(f=>f.issues?.length?f.issues:[f]),...(state?.busy?state?.job?.issues||[]:[])];d.classList.toggle('bad',issues.length>0);const seen=new Set();for(const issue of issues){if(!issue.message||seen.has(issue.message))continue;seen.add(issue.message);const row=document.createElement('div');row.className='issue';const code=document.createElement('code');code.textContent=issue.joint?`${issue.joint} · ${issue.code||''} `:`${issue.code||'STATUS'} `;row.append(code,document.createTextNode(issue.message));d.append(row);}if(!d.children.length)d.textContent=state?.start_error||'当前未检测到关节角度越限；这不代表任意目标均可到达。';}
function drawJoints(c){const box=$('#joints');box.replaceChildren();for(let i=0;i<6;i++){const v=c.arm_deg[i],[lo,hi]=c.joint_limits_deg[i],div=document.createElement('div');div.className='joint'+(v<lo||v>hi?' rangebad':'');const top=document.createElement('div');top.className='jointline';const name=document.createElement('b');name.textContent=`J${i+1}`;const value=document.createElement('span');value.textContent=`${v.toFixed(1)}°`;top.append(name,value);const bar=document.createElement('div');bar.className='jointbar';const mark=document.createElement('i');mark.style.left=`${Math.max(0,Math.min(100,(v-lo)/(hi-lo)*100))}%`;bar.append(mark);const range=document.createElement('small');range.textContent=`${lo.toFixed(1)}° … ${hi.toFixed(1)}°`;div.append(top,bar,range);box.append(div);}}
function drawRemote(){
 const r=state?.remote_priority,generation=r?.generation,changed=remoteGeneration!==null&&generation!==remoteGeneration;
 const blocked=!!(state?.error||!r||r.blocked);
 if(changed||(blocked&&!remoteBlocked)){resumeKeyboard=resumeKeyboard||enabled;localDisable();space.clear();invalidate();}
 remoteGeneration=generation;remoteBlocked=blocked;
 $('#remote-priority').textContent=blocked?(r?.reason||'遥控反馈不可用，其他控制暂停'):'遥控器优先 · 已静默 2 秒后自动开放其他控制';
 $('#remote-priority').className=blocked?'remote-priority bad':'remote-priority';
 if(!blocked&&resumeKeyboard&&!state.busy&&state.joint_ready&&!state.start_error&&document.hasFocus()&&!document.hidden){resumeKeyboard=false;enableKeyboard();}
}
async function poll(){try{const r=await fetch('/api/status');if(!r.ok)throw Error('状态读取失败');state=await r.json();drawRemote();drawGripper();if(state.job.operation==='standby')standbyPending=false;if(foldRequestId&&state.job.fold_id===foldRequestId)foldPending=false;drawFold();if(enabled&&!state.keyboard.enabled)localDisable();$('#connection').textContent=state.error||`B2 ${state.dog_mode==='native'?'原生':'P4'} · A3 真实反馈`;$('#connection').className=state.error?'bad':'';$('#hint').textContent=state.error||state.start_error||'反馈与起始条件已就绪。';const c=state.current;if(c){if(!initialized){setTarget(c.ee,c.rpy_deg);initialized=true;}$('#telemetry').textContent=`EE / cm\n${c.ee.map(v=>(v*100).toFixed(2)).join(' / ')}\n朝向 R/P/Y / °\n${c.rpy_deg.map(v=>v.toFixed(1)).join(' / ')}\n机体高 ${(c.body_xyz[2]*100).toFixed(1)} cm · ${c.speed.toFixed(3)} m/s\n反馈读取 ${state.read_latency_ms} ms`;drawJoints(c);}if(state.keyboard.enabled){target=state.keyboard.xyz;targetQuat=state.keyboard.quaternion_xyzw;}$('#goal-display').textContent=`X / Y / Z · cm\n${target.map(v=>Number.isFinite(v)?(v*100).toFixed(2):'待填写').join(' / ')}${state.job.error_m!==undefined?`\n追踪误差 ${ (state.job.error_m*100).toFixed(2)} cm / ${state.job.orientation_deg.toFixed(1)}°`:''}`;if(state.busy||['complete','stopped','timeout','prepared','standby','folded'].includes(state.job.phase))message(state.job.message+(state.job.error_m!==undefined?` · ${(state.job.error_m*100).toFixed(2)} cm / ${state.job.orientation_deg.toFixed(1)}°`:''));$('#keyboard-state').textContent=enabled?'已启用 · 按住控制':state.keyboard.enabled?'另一页面控制中':'未启用';$('#keyboard-state').className=enabled?'badge on':'badge';$('#keyboard-enable').disabled=!!(remoteBlocked||pending||enabled||state.busy||state.start_error||state.error||state.keyboard.enabled||!state.joint_ready);$('#keyboard-disable').disabled=!enabled;$('#enter-standby').disabled=!!(remoteBlocked||pending||state.busy||state.keyboard.enabled||state.error||state.standby_error);$('#enter-p4-standby').disabled=!!(remoteBlocked||pending||state.busy||state.keyboard.enabled||state.error||state.p4_standby_error);$('#p4-standby-hint').textContent=state.p4_standby_error||'保持 P4 模式，点击后机械臂以 5% 速度进入预备位。';$('#standby-state').textContent=state.busy&&state.job.operation==='standby'?'准备中':state.joint_ready?'联合模式已就绪':'未进入';$('#standby-state').className=state.joint_ready?'badge on':'badge';$('#standby-hint').textContent=state.busy&&state.job.operation==='standby'?state.job.message:state.joint_ready?'P4 已接管。可启用键盘控制 EE；未按运动键时不追踪新目标。':state.standby_error||'可以进入联合待命；将真实移动机械臂并切换狗的运动系统。';drawManual();$('#release').disabled=state.busy;drawDiagnostics();dirty=true;}catch(e){remoteBlocked=true;invalidate();drawManual();drawFold();if(state)state.gripper=null;drawGripper();$('#remote-priority').textContent='反馈连接中断，其他控制暂停';$('#connection').textContent='本机服务连接中断';$('#connection').className='bad';$('#execute').disabled=true;$('#keyboard-enable').disabled=true;$('#enter-standby').disabled=true;$('#enter-p4-standby').disabled=true;if(enabled)disable('反馈连接中断，停止键盘控制');}setTimeout(poll,250);}poll();
async function heartbeat(){try{await action('heartbeat');}catch{}setTimeout(heartbeat,400);}heartbeat();
