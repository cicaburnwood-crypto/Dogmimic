const fields=['x','y','z','r','p','yaw'];
const labels=['X','Y','Z','Roll','Pitch','Yaw'];

export function readManualGoal(values){
 const numbers=fields.map((field,i)=>{
  const raw=String(values[field]??'').trim(),value=Number(raw);
  if(!raw||!Number.isFinite(value))throw Error(`请填写有效的 ${labels[i]} 数值。`);
  return value;
 });
 const xyz=numbers.slice(0,3).map(v=>v/100);
 if(Math.hypot(xyz[0],xyz[1])>3)throw Error('目标水平距离超出当前任务范围：距参考原点不能超过 300 cm。');
 if(xyz[2]<.1||xyz[2]>1.6)throw Error('目标 Z 超出当前任务范围：允许 10–160 cm。');
 return {xyz,rpy_deg:numbers.slice(3)};
}

export function canExecutePlan(plan,{pending=false,blocked=true,busy=false,keyboard=false,jointReady=false,now=Date.now()}={}){
 return !!(plan?.id&&Number.isFinite(plan.expiresAt)&&now<plan.expiresAt&&!pending&&!blocked&&!busy&&!keyboard&&(plan.kind==='prepare'||jointReady));
}
