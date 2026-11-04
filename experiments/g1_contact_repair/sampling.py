"""Candidate domain sampler. Not installed in the active training run.

25% nominal, 55% full continuous domain, 10% isolated weak actuators,
10% weak actuators crossed with the original continuous domain.
Pose/reset noise sampling remains the responsibility of RobustEnv.
"""
import torch


def sample_domain(k, device, friction):
    x = torch.zeros(k, 9, device=device)
    x[:, 0] = .9 + .35 * torch.rand(k, device=device)
    x[:, 1] = .8 + .4 * torch.rand(k, device=device)
    x[:, 2] = .7 + .6 * torch.rand(k, device=device)
    x[:, 3] = .7 + .3 * torch.rand(k, device=device)
    x[:, 4] = torch.randint(0, 9, (k,), device=device) / 8
    x[:, 5] = torch.rand(k, device=device) * .6 - .3
    x[:, 6:8] = (torch.rand(k, 2, device=device) * 2 - 1) * 25
    x[:, 8] = (torch.rand(k, device=device) * 2 - 1) * 10
    bucket = torch.rand(k, device=device)
    nominal = bucket < .25
    isolated = (bucket >= .80) & (bucket < .90)
    crossed = bucket >= .90
    base = nominal | isolated
    x[base] = 0
    x[base, :4] = 1
    mu = friction.clone()
    mu[base] = .8
    weak = isolated | crossed
    x[weak, 1] = .8
    x[weak, 2] = .7
    x[weak, 3] = .7
    return x, mu, dict(nominal=nominal, continuous=(bucket >= .25) & (bucket < .80), isolated_weak=isolated, crossed_weak=crossed)
