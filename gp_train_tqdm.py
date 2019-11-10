import torch
from tqdm import trange

def train(gpmodule, optimizer=None, loss_fn=None, retain_graph=None, num_steps=1000,tqdm=True):
    """
    A helper to optimize parameters for a GP module.
    :param ~pyro.contrib.gp.models.GPModel gpmodule: A GP module.
    :param ~torch.optim.Optimizer optimizer: A PyTorch optimizer instance.
        By default, we use Adam with ``lr=0.01``.
    :param callable loss_fn: A loss function which takes inputs are
        ``gpmodule.model``, ``gpmodule.guide``, and returns ELBO loss.
        By default, ``loss_fn=TraceMeanField_ELBO().differentiable_loss``.
    :param bool retain_graph: An optional flag of ``torch.autograd.backward``.
    :param int num_steps: Number of steps to run SVI.
    :param bool tqdm: whether to use trange for the training loop to see progress
    :returns: a list of losses during the training procedure
    :rtype: list
    """
    optimizer = (torch.optim.Adam(gpmodule.parameters(), lr=0.01)
                 if optimizer is None else optimizer)
    # TODO: add support for JIT loss
    loss_fn = TraceMeanField_ELBO().differentiable_loss if loss_fn is None else loss_fn

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(gpmodule.model, gpmodule.guide)
        torch_backward(loss, retain_graph)
        return loss

    losses = []
    for i in trange(num_steps):
        loss = optimizer.step(closure)
        losses.append(torch_item(loss))
    return losses
