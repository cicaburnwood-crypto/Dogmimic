import torch
from supermimic.replay import ReplayBuffer
from supermimic.task import NJ, flat_to_state

class CommandReplay(ReplayBuffer):
    def __init__(self,*args):
        super().__init__(*args)
        self.cmd=torch.zeros(self.L,self.n,3,device=self.device)
        self.context=torch.zeros(self.L,self.n,17,device=self.device)
        self.aux=torch.zeros(self.L,self.n,32,device=self.device)

    def add(self,s,a,s1,xi,cmd,context,aux,ep):
        self.context[self.ptr]=context;self.aux[self.ptr]=aux
        self.cmd[self.ptr]=cmd
        super().add(s,a,s1,xi,torch.zeros_like(ep),ep)

    def sample_starts(self,batch,history,task):
        env=torch.randint(0,self.n,(batch,),device=self.device)
        age=torch.randint(0,self.size,(batch,),device=self.device)
        t=self._age_index(age)
        s0,xi,cmd,ep0=self.s[t,env],self.xi[t,env],self.cmd[t,env],self.ep[t,env]
        back=torch.arange(history-1,-1,-1,device=self.device)
        ages=(age[:,None]+back).clamp(max=self.size-1)
        idx=self._age_index(ages);e=env[:,None].expand(-1,history)
        same=(self.ep[idx,e]==ep0[:,None])&((age[:,None]+back)<self.size)
        first=same.float().argmax(-1)
        idx=torch.where(same,idx,idx.gather(1,first[:,None]).expand(-1,history))
        prev_idx=(idx-1)%self.L
        prev_ok=(self.ep[prev_idx,e]==ep0[:,None])&(idx!=self._age_index(torch.full_like(idx,self.size-1)))
        prev=torch.where(prev_ok[...,None],self.a[prev_idx,e],torch.zeros_like(self.a[prev_idx,e]))
        bias=xi[:,None,1:1+NJ]
        hist=task.obs_clean(flat_to_state(self.s[idx,e]),prev,bias.expand(-1,history,-1))
        return s0,xi,cmd,hist,xi[:,20:32],self.context[t,env]

    def sample_context_windows(self,batch,k):
        env=torch.randint(0,self.n,(batch,),device=self.device)
        age=torch.randint(k-1,self.size,(batch,),device=self.device)
        idx=self._age_index(age[:,None]-torch.arange(k,device=self.device))
        e=env[:,None].expand(-1,k);ep=self.ep[idx,e]
        valid=(ep==ep[:,:1]).all(-1)&(ep[:,0]>=0)
        return self.s[idx[:,0],env],self.a[idx,e],self.s1[idx,e],self.xi[idx,e],self.aux[idx,e],valid
