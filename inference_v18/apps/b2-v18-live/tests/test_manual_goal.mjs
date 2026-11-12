import assert from 'node:assert/strict';
import {readManualGoal,canExecutePlan} from '../web/manual-goal.mjs';

const input={x:'65.25',y:'-30.4',z:'100.12',r:'-45.5',p:'30',yaw:'180'};
assert.deepEqual(readManualGoal(input),{xyz:[.6525,-.304,1.0012],rpy_deg:[-45.5,30,180]});
for(const field of Object.keys(input)){
 for(const bad of ['', ' ',undefined,'NaN','Infinity','no number']){
  assert.throws(()=>readManualGoal({...input,[field]:bad}),/请填写有效/);
 }
}
assert.throws(()=>readManualGoal({...input,x:'300',y:'1'}),/水平距离/);
for(const z of ['9.9','160.01'])assert.throws(()=>readManualGoal({...input,z}),/目标 Z/);
for(const z of ['10','160'])assert.equal(readManualGoal({...input,x:'-300',y:'0',z}).xyz[2],Number(z)/100);

const plan={id:'fixed-goal',expiresAt:121000},ready={now:1001,blocked:false,jointReady:true};
assert.equal(canExecutePlan(null,ready),false);
assert.equal(canExecutePlan(plan,ready),true);
for(const [key,value] of [['pending',true],['blocked',true],['busy',true],['keyboard',true],['jointReady',false],['now',121000]])assert.equal(canExecutePlan(plan,{...ready,[key]:value}),false);
assert.equal(canExecutePlan({...plan,expiresAt:undefined},ready),false);
assert.equal(canExecutePlan({...plan,kind:'prepare'},{...ready,jointReady:false}),true);
console.log('Manual goal: unit conversion, all six required finite fields, workspace bounds, preview expiry and exclusive control gates PASS');
