"""Passive full-field DDS recorder. No publishers, RPC calls or motion control."""
import argparse,dataclasses,gzip,importlib,json,os,queue,signal,struct,sys,threading,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'vendor/unitree_sdk2_python-814556d15970dd2ecf1c9984e845ca02ab07e206'))
MAGIC=b'G1LOG1\n'

def registry():
    pairs=[('unitree_hg','LowState_'),('unitree_hg','LowCmd_'),('unitree_hg','IMUState_'),
           ('unitree_hg','BmsState_'),('unitree_hg','MainBoardState_'),('unitree_hg','PressSensorState_'),
           ('unitree_hg','HandState_'),('unitree_hg','HandCmd_'),('unitree_go','SportModeState_'),
           ('unitree_go','WirelessController_'),('unitree_go','Error_'),('nav_msgs','Odometry_'),
           ('std_msgs','String_'),('unitree_api','Request_'),('unitree_api','Response_')]
    result={}
    for package,name in pairs:
        try:
            cls=getattr(importlib.import_module('unitree_sdk2py.idl.'+package+'.msg.dds_'),name)
            result[package+'::msg::dds_::'+name]=cls
        except (ImportError,AttributeError):pass
    return result

TOPICS={
 'rt/lowstate','rt/lf/lowstate','rt/lowcmd','rt/user_lowcmd','rt/arm_sdk',
 'rt/secondary_imu','rt/lf/secondary_imu','rt/odommodestate','rt/lf/odommodestate',
 'rt/sportmodestate','rt/lf/sportmodestate','rt/wirelesscontroller','rt/multiplestate',
 'rt/bmsstate','rt/lf/bmsstate','rt/mainboardstate','rt/lf/mainboardstate',
 'rt/pressuresensorstate','rt/lf/pressuresensorstate','rt/dex3/left/state','rt/dex3/right/state',
 'rt/dex3/left/cmd','rt/dex3/right/cmd','rt/lf/dex3/left/state','rt/lf/dex3/right/state',
 'rt/state_estimator/fusion_odom','rt/state_estimator/fusion_odom_torso',
 'rt/state_estimator/odom_pelvis','rt/state_estimator/odom_torso',
 'rt/lf/emergency_stop','rt/lf/battery_alarm','rt/selftest','rt/servicestate',
 'rt/servicestateactivate','rt/arm/action/state',
 'rt/api/motion_switcher/request','rt/api/motion_switcher/response',
 'rt/api/sport/request','rt/api/sport/response'}


def write_frame(f,header,payload):
    h=json.dumps(header,separators=(',',':')).encode()
    f.write(struct.pack('<II',len(h),len(payload)));f.write(h);f.write(payload)


def frames(path):
    with gzip.open(path,'rb') as f:
        if f.read(len(MAGIC))!=MAGIC:raise ValueError('Not a G1 logger segment')
        while True:
            sizes=f.read(8)
            if not sizes:return
            if len(sizes)!=8:raise ValueError('Truncated frame sizes')
            nh,np=struct.unpack('<II',sizes)
            if nh>1048576 or np>16777216:raise ValueError('Invalid frame length')
            h,p=f.read(nh),f.read(np)
            if len(h)!=nh or len(p)!=np:raise ValueError('Truncated frame payload')
            yield json.loads(h),p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface',default='eth0');parser.add_argument('--out',default=str(BASE/'results/dds_logs'))
    parser.add_argument('--seconds',type=float,default=0);parser.add_argument('--segment-seconds',type=float,default=60)
    args=parser.parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelFactory,ChannelSubscriber
    from cyclonedds.builtin import BuiltinDataReader,BuiltinTopicDcpsPublication
    from cyclonedds.dynamic import get_types_for_typeid
    from cyclonedds.util import duration
    from cyclonedds.sub import DataReader,ddspy_take
    from cyclonedds.topic import Topic
    from cyclonedds.core import Listener
    from cyclonedds.internal import InvalidSample
    from cyclonedds.qos import Qos,Policy
    ChannelFactoryInitialize(0,args.interface)
    participant=ChannelFactory._ChannelFactory__participant
    reader=BuiltinDataReader(participant,BuiltinTopicDcpsPublication)
    kinds=registry();dynamic_kinds={};subs={};discovered={};unsupported={};received={};written={};dropped={};errors=[];max_gaps={};last_ticks={};dds_lost={};read_errors={}
    pending=queue.Queue(maxsize=16384);stop=threading.Event();lock=threading.Lock()
    start=time.monotonic();session=time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
    out=Path(args.out)/session;out.mkdir(parents=True,exist_ok=False)
    statuspath=BASE/'results/logger_status.json';current=[None];segments=[]
    def status(running=True):
        with lock:
            data=dict(running=running,pid=os.getpid(),updated_unix=time.time(),session=str(out),
                elapsed_s=time.monotonic()-start,subscribed=dict((k,v[1]) for k,v in subs.items()),
                discovered=dict(discovered),unsupported=dict(unsupported),received=dict(received),
                written=dict(written),queue_dropped=dict(dropped),queue_depth=pending.qsize(),
                errors=list(errors),dds_sample_lost=dict(dds_lost),reader_errors=dict(read_errors),lowstate_max_positive_tick_gap=dict(max_gaps),current_segment=current[0],
                closed_segments=list(segments),motion_commands_sent=0,control_rpc_calls=0,
                missing_topics=sorted(TOPICS-set(received)),
                format='G1LOG1 framed gzip, raw DDS CDR payload; dynamic types full JSON')
        for path in (statuspath,out/'status.json'):
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
    def callback(topic,typename,dynamic=False):
        def receive(payload,info):
            if stop.is_set():return
            header=dict(topic=topic,type=typename,encoding='json' if dynamic else 'cdr',
                receive_unix_ns=time.time_ns(),receive_monotonic_ns=time.monotonic_ns(),
                dds_source_timestamp_ns=getattr(info,'source_timestamp',None),
                publication_handle=getattr(info,'publication_handle',None))
            with lock:received[topic]=received.get(topic,0)+1
            try:pending.put_nowait((header,payload))
            except queue.Full:
                with lock:dropped[topic]=dropped.get(topic,0)+1
        return receive
    def writer():
        f=None;path=None;opened=0.;index=0
        try:
            while not stop.is_set() or not pending.empty():
                try:h,raw=pending.get(timeout=.2)
                except queue.Empty:continue
                if f is None or time.monotonic()-opened>=args.segment_seconds:
                    if f:
                        f.close();final=path.with_suffix('');path.rename(final)
                        with lock:segments.append(str(final))
                    path=out/('segment_%05d.cdr.gz.part'%index);index+=1
                    f=gzip.open(path,'wb',compresslevel=1);f.write(MAGIC);opened=time.monotonic();current[0]=str(path)
                payload=json.dumps(dataclasses.asdict(dynamic_kinds[h['type']].deserialize(raw)),separators=(',',':')).encode() if h['encoding']=='json' else raw
                write_frame(f,h,payload)
                with lock:written[h['topic']]=written.get(h['topic'],0)+1
        except Exception as exc:
            with lock:errors.append(repr(exc))
            stop.set()
        finally:
            if f:
                f.close();final=path.with_suffix('');path.rename(final)
                with lock:segments.append(str(final));current[0]=None
    worker=threading.Thread(target=writer);worker.start()
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    last_status=0
    try:
        while not stop.is_set() and (args.seconds<=0 or time.monotonic()-start<args.seconds):
            for sample in reader.take(200):
                topic,typename=sample.topic_name,sample.type_name
                discovered[topic]=typename
                if topic not in TOPICS or topic in subs or topic in unsupported:continue
                cls=kinds.get(typename);dynamic=False
                if cls is None:
                    try:
                        cls,_=get_types_for_typeid(participant,sample.type_id,duration(seconds=2));dynamic=True;dynamic_kinds[typename]=cls
                    except Exception as exc:
                        unsupported[topic]=typename+': '+str(exc);continue
                try:
                    receive=callback(topic,typename,dynamic)
                    def available(r,receive=receive,topic=topic):
                        if r is None:return  # CycloneDDS may notify during DataReader construction.
                        try:
                            while not stop.is_set():
                                batch=ddspy_take(r._ref,64)
                                if isinstance(batch,int):raise RuntimeError("DDS take error %s"%batch)
                                if not batch:break
                                for raw,info in batch:
                                    if info.valid_data:receive(raw,info)
                        except Exception as exc:
                            with lock:read_errors[topic]=dict(error=repr(exc),count=read_errors.get(topic,{}).get('count',0)+1)
                    def lost(r,st,topic=topic):
                        with lock:dds_lost[topic]=dds_lost.get(topic,0)+int(st.total_count_change)
                    t=Topic(participant,topic,cls)
                    sub=DataReader(participant,t,Qos(Policy.History.KeepLast(2048),Policy.Reliability.BestEffort),
                                   listener=Listener(on_data_available=available,on_sample_lost=lost))
                    subs[topic]=((t,sub),typename)
                except Exception as exc:unsupported[topic]=typename+': '+str(exc)
            if time.monotonic()-last_status>=2:status();last_status=time.monotonic()
            stop.wait(.1)
    finally:
        stop.set();worker.join();status(False)
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
