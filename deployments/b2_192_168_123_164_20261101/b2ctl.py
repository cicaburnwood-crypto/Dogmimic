"""Convenient entry point for installed B2 inference; motor actuation is unavailable."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['status','selftest','shadow'])
    p.add_argument('--skill',choices=['both','hindstand','locomotion'],default='both')
    p.add_argument('--seconds',type=float,default=20)
    p.add_argument('--interface',default='eth0')
    p.add_argument('--remote-input',action='store_true')
    args=p.parse_args()
    if args.command=='status':
        print((ROOT/'deployment_status.json').read_text())
        return
    logs=Path('/home/unitree/supermimic_b2/logs')
    logs.mkdir(parents=True,exist_ok=True)
    output=logs/(args.command+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
    command=[sys.executable,str(ROOT/(args.command+'.py')),'--out',str(output)]
    if args.command=='shadow':
        command+=['--skill',args.skill,'--seconds',str(args.seconds),'--interface',args.interface]
        if args.remote_input:command.append('--remote-input')
    result=subprocess.run(command,check=False)
    print('Report:',output)
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
