"""Static scientific comparison of the first two seconds, with hardware context."""
import sys
from pathlib import Path
EXTRA=Path('/tmp/b2_c1_plot_lib_20261103')
if EXTRA.exists():sys.path.insert(0,str(EXTRA))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'diagnostics/b2_sim2sim_20261103'
fig,axes=plt.subplots(2,2,figsize=(13,7.5),sharex=True,layout='constrained')
real=np.load(ROOT/'diagnostics/b2_c1_failure_20261103/142243/signals.npz')
colors={'mujoco':'#137baf','mjlab':'#e58b20','bullet':'#3b9a6c'}
for col,kg in enumerate([0,5]):
    for backend,color in colors.items():
        folder=BASE/('bullet_com_corrected' if backend=='bullet' else 'paired_v1')/backend/f'{kg}kg_measured'
        s=np.load(folder/'substeps.npz');sel=s['time_s']<=2.6
        axes[0,col].plot(s['time_s'][sel],abs(s['dq'][sel,0]).max(-1),color=color,label=backend,lw=1.4)
        axes[1,col].plot(s['time_s'][sel],s['power'][sel,0],color=color,lw=1.2)
    r=(real['t']>=0)&(real['t']<=1.8345823288)
    axes[0,col].plot(real['t'][r],abs(real['dq'][r]).max(-1),color='#383838',lw=1.1,label='hardware pre-flag (payload unknown)')
    for ax in axes[:,col]:
        ax.axvline(1.8345823288,color='#888888',linestyle=':',lw=1)
        ax.grid(alpha=.2);ax.set_xlim(0,2.6)
    axes[0,col].axhline(6,color='#b43232',ls='--',lw=1.2,label='experimental speed limit')
    axes[1,col].axhline(1501,color='#b43232',ls='--',lw=1.2)
    axes[0,col].set_title(f'{kg} kg | measured initial pose | fixed trial 0')
    axes[1,col].set_xlabel('Seconds after control handover')
axes[0,0].set_ylabel('Max absolute joint speed (rad/s)')
axes[1,0].set_ylabel('Simulated sum |actuator torque x speed| (W)')
axes[0,0].legend(fontsize=8,loc='upper left')
fig.suptitle('B2 C1 2000: rise transient across simulators\nDotted line: real 0x20 state change; coincidence does not identify its firmware cause',fontsize=13)
fig.savefig(BASE/'rise_transient.png',dpi=160);fig.savefig(BASE/'rise_transient.pdf');plt.close(fig)
