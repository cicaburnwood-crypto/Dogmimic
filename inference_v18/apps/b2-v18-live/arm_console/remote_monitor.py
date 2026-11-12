"""Apply the dog's physical-remote priority to every local A3 command source."""
import asyncio
import time
import aiohttp


class RemoteMonitor:
    def __init__(self, arm, url):
        self.arm = arm
        self.url = url
        self.stamp = 0.
        self.value = {'blocked': True, 'reason': '等待狗端遥控器优先状态', 'generation': None}
        self.was_blocked = True
        self.generation = None

    def status(self):
        if time.monotonic() - self.stamp > .2:
            return {**self.value, 'blocked': True, 'reason': '遥控器优先状态连接中断'}
        return dict(self.value)

    def require(self):
        s = self.status()
        if s['blocked']:
            raise ValueError(s.get('reason') or '遥控器优先，其他控制暂时暂停')

    async def update(self, dog):
        value = dog.get('remote_priority')
        if not isinstance(value, dict) or not dog.get('connected') or not value.get('fresh'):
            value = {'blocked': True, 'reason': '狗端遥控器反馈不可用', 'generation': self.generation}
        self.value, self.stamp = value, time.monotonic()
        generation = value.get('generation')
        changed = self.generation is not None and generation != self.generation
        blocked = bool(value.get('blocked', True))
        if changed or (blocked and not self.was_blocked):
            self.arm.epoch += 1  # Cancel pending solve / enable before their next send.
            self.arm.log('remote', value.get('reason') or '遥控器曾输入，取消旧任务')
            # Run independently: holding the gripper must not wait for a MOVJ stop ACK.
            jobs = [self.arm.gripper.interrupt()]
            if self.arm.active:
                jobs.append(self.arm.stop())
            results = await asyncio.gather(*jobs, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):self.arm.log('error', '遥控接管停止未确认：' + str(result))
        self.generation, self.was_blocked = generation, blocked

    async def run(self):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=.15), trust_env=False) as http:
            while True:
                try:
                    async with http.get(self.url, headers={'Host': '127.0.0.1:8765'}) as r:
                        r.raise_for_status()
                        await self.update(await r.json())
                except asyncio.CancelledError:
                    raise
                except Exception:
                    await self.update({})
                await asyncio.sleep(.05)
