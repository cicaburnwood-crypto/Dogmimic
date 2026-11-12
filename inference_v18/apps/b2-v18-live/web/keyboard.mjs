// Pure key state. Joint commands and policy inference never run in the browser.
export const CODES=new Set(['KeyW','KeyA','KeyS','KeyD','KeyR','KeyF','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','KeyO','KeyP']);
export function editable(target){return !!target?.closest?.('input,textarea,select,[contenteditable="true"],[contenteditable=""]');}
// Edge-triggered accessory control, separate from the held EE movement keys.
export class SpaceToggle {
 constructor(){this.held=false;}
 down(e){
  if(e.code!=='Space'||editable(e.target)||e.ctrlKey||e.metaKey||e.altKey)return false;
  e.preventDefault();
  if(this.held||e.repeat)return false;
  this.held=true;return true;
 }
 up(e){if(e.code==='Space'&&this.held){e.preventDefault();this.held=false;}}
 clear(){this.held=false;}
}
export class Keys{
 constructor(){this.held=new Set();}
 down(e,enabled){if((e.repeat&&!this.held.has(e.code))||!enabled||e.ctrlKey||e.metaKey||e.altKey||editable(e.target)||!CODES.has(e.code))return false;this.held.add(e.code);return true;}
 up(e){return this.held.delete(e.code);}
 clear(){this.held.clear();}
 list(){return [...this.held].sort();}
}
