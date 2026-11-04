"""Read-only 10-second device/interface probe. Never acquires control."""
import json,time
from sdk_transport import SDKTransport
transport=SDKTransport.from_environment()
records=[];end=time.monotonic()+10
while time.monotonic()<end:
 try:
  s=transport.snapshot();records.append(s);print(json.dumps(s),flush=True)
 except Exception as e:print(json.dumps({'waiting':str(e)}),flush=True)
 time.sleep(.1)
print(json.dumps({'samples':len(records),'motor_commands_sent':0,'ownership_changes':0}))
