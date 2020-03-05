import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import torch


def train_binary(gpmodel, X_test, Y_test, nsteps=2000, lr=0.0025,cuda=True):
    opt = torch.optim.Adam(gpmodel.parameters(), lr=0.0025)
    loss_fn = pyro.infer.TraceMeanField_ELBO().differentiable_loss
    nsteps = 2000
    losses = []
    bce = torch.nn.BCELoss(reduction='mean')
    bce_losses = []
    for i in range(nsteps):
            opt.zero_grad()
            loss = loss_fn(gpmodel.model, gpmodel.guide)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            # Show some output
            if (i+1)%100==0:
                with torch.no_grad():
                    f_loc, f_var = gpmodel(X_test.cuda())
                    probs = dist.Normal(f_loc, f_var)().sigmoid()
                    bce_losses.append(torch.nn.BCELoss()(probs.t(),Y_test.cuda()))
                print(f'{i+1}: Accuracy = {test_binary(gpmodel,X_test,Y_test,cuda):0.5f} -- BCELoss = {bce_losses[-1]:0.5f}',end='\r')
            if (i+1)==nsteps:
                print("\n")
    return losses

def plot2dproj(X,labels=None,method=None, legend=False):
    if labels is not None:
        for i in range(np.unique(labels).shape[0]):
            pts = X[labels==i]
            plt.scatter(pts[:,0],pts[:,1], s=3., label=str(i))
            if legend: plt.legend()
    else:
        plt.scatter(X[:,0],X[:,1],s=3)
    if method is not None :
        plt.title( method+" Embedding",fontsize=16)
    plt.show()
    
def test_binary(gpmodel, X_test, Y_test,cuda):
    if isinstance(gpmodel.likelihood, gp.likelihoods.Binary):
        with torch.no_grad():
            if cuda:
                X_test, Y_test = X_test.cuda(), Y_test.cuda()
            f_loc, f_var = gpmodel(X_test)
            pred = gpmodel.likelihood(f_loc,f_var)
            return (100.*pred.eq(Y_test.t()).detach().cpu().sum()/Y_test.nelement()).item()
        
def plot_before_after(Xi, Xf, Xui, Xuf, labels,method=None, legend=False):
    fig, ax = plt.subplots(1,2,figsize=(18,6))
    fig.suptitle(r"Data Points and VSGP Inducing Points",fontsize=16)
    ax[0].set_title(r"Initial",fontsize=16)
    ax[1].set_title(r"Final",fontsize=16)
    for i in range(np.unique(labels).shape[0]):
        pts_i = Xi[labels==i]
        pts_f = Xf[labels==i]
        ax[0].scatter(pts_i[:,0],pts_i[:,1], s=3., label=str(i))
        ax[1].scatter(pts_f[:,0],pts_f[:,1], s=3., label=str(i))
    ax[0].scatter(Xui[:,0],Xui[:,1], marker='x', color='k', label='Inducing')
    ax[1].scatter(Xuf[:,0],Xuf[:,1], marker='x', color='k', label='Inducing')
    if legend: ax[0].legend()
        
def cuda_gp(gpmodel,lvm=True):
    #A little silly but torch's .cuda doesnt put gpmodel.X and gpmodel.y on the GPU so this function takes care of those plu
    gpmodel.cuda()
    gpmodel.y = gpmodel.y.cuda()
    if not lvm: gpmodel.X.cuda()
        
def batch_train(gpmodel, cuda, X_train, Y_train, X_test, Y_test, 
                loss_fn=pyro.infer.TraceMeanField_ELBO().differentiable_loss, optimizer=torch.optim.Adam, lr=0.0025,
                num_epochs=20, num_steps=200, batch_size=500):
    opt = optimizer(gpmodel.parameters(), lr=lr)
    losses = [] 
    for e in range(num_epochs):
        # Set the batch
        batch_idx = torch.randperm(X_train.shape[0])[:batch_size]
        x_batch = X_train[batch_idx]
        y_batch = Y_train[batch_idx]
        if cuda: x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        gpmodel.set_data(x_batch, y_batch.t())
        #Take the steps
        for i in range(num_steps):
            opt.zero_grad()
            loss = loss_fn(gpmodel.model, gpmodel.guide)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        # Show some output
        print(f'Epoch {e+1} -- Accuracy = {test_binary(gpmodel,X_test,Y_test,cuda)} -- Loss = {losses[-1]}',end='\r')
    return losses
