import assert from 'node:assert/strict';
import {Keys,CODES,editable} from '../web/keyboard.mjs';
const k=new Keys(),event=code=>({code,target:{closest:()=>null}});
for(const code of CODES){assert.equal(k.down(event(code),true),true);assert.deepEqual(k.list(),[code]);assert.equal(k.up(event(code)),true);}
assert.equal(k.down(event('KeyW'),false),false);
for(const modifier of ['ctrlKey','altKey','metaKey'])assert.equal(k.down({...event('KeyW'),[modifier]:true},true),false);
assert.equal(k.down({...event('KeyW'),target:{closest:()=>({})}},true),false);
assert.equal(k.down(event('KeyQ'),true),false);
k.down(event('KeyW'),true);k.down(event('KeyW'),true);assert.deepEqual(k.list(),['KeyW']);k.clear();assert.deepEqual(k.list(),[]);
console.log('Browser keyboard filtering, all 12 keys, keyup and clear: PASS');

const {SpaceToggle}=await import('../web/keyboard.mjs');
const space=new SpaceToggle(),spaceEvent=(extra={})=>({...event('Space'),preventDefault(){this.prevented=true;},...extra});
let e=spaceEvent();assert.equal(space.down(e),true);assert.equal(e.prevented,true);
assert.equal(space.down(spaceEvent()),false);assert.equal(space.down(spaceEvent({repeat:true})),false);
space.up(spaceEvent());assert.equal(space.down(spaceEvent()),true);
space.clear();assert.equal(space.down(spaceEvent({repeat:true})),false);assert.equal(space.held,false);
for(const modifier of ['ctrlKey','altKey','metaKey']){e=spaceEvent({[modifier]:true});assert.equal(space.down(e),false);assert.equal(e.prevented,undefined);}
e=spaceEvent({target:{closest:()=>({})}});assert.equal(space.down(e),false);assert.equal(e.prevented,undefined);
assert.equal(k.down(spaceEvent(),true),false);assert.equal(space.down(spaceEvent({code:'Escape'})),false);
console.log('Space edge trigger, held repeat, blur reset, input/modifier filtering and EE separation: PASS');

k.clear();assert.equal(k.down({...event("KeyW"),repeat:true},true),false);assert.equal(k.down(event("KeyW"),true),true);assert.equal(k.down({...event("KeyW"),repeat:true},true),true);k.clear();console.log("Post-takeover held keys require a new press: PASS");
