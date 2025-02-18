from torch.optim import lr_scheduler

from utils.config import Config


def get_lr_scheduler(optimizer, opts: Config, cur_ep=-1):
    if opts.policy == 'linear_decay':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opts.step, gamma=opts.gamma)
    elif opts.policy == 'lambda_decay':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.decay_after) / \
                   float(opts.total_epochs - opts.decay_after + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             T_0=opts.T_0,
                                                             T_mult=opts.T_mult,
                                                             eta_min=opts[('min_lr', 0)],
                                                             verbose=opts[('verbose', False)],
                                                             last_epoch=cur_ep)
    else:
        raise NotImplementedError('no such learn rate policy')
    return scheduler