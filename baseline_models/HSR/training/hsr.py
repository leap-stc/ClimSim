import matplotlib.pyplot as plt
import numpy as np
import torch
from tools import progress


"""
Contains the code for the Heteroskedastic Regression Model and its training.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MLP(torch.nn.Module):
    """
    MLP Estimator for mean and log precision
    """
    def __init__(self, in_dims, out_dims, hidden_dims=512, layers=1, dropout=0):
        super().__init__()
        self.linears = []
        for i in range(layers):
            self.linears += [torch.nn.Sequential(
                torch.nn.Linear(in_dims if i == 0 else hidden_dims, hidden_dims),
                torch.nn.LayerNorm(hidden_dims),
                torch.nn.Dropout(p=dropout))
            ]
            self.add_module('linear%d' % i, self.linears[-1])
        self.final_linear = torch.nn.Linear(hidden_dims, out_dims)

    def forward(self, x):
        torch.flatten(x, start_dim=1)
        for linear in self.linears:
            x = torch.nn.functional.relu(linear(x))
        x = self.final_linear(x)
        return x


class HeteroskedasticRegression(torch.nn.Module):
    def __init__(self, in_dims=124, out_dims=128, hidden_dims=512, layers=1, dropout=0):
        """
        Heteroskedastic Regression model, computing MLE estimates of mean and precision via regularized MLPs

        Inputs:
        -------
        in_dims - [int] size of x
        out_dims - [int] size of y
        hidden_dims - [int] size of hidden layers
        layers - [int] number of layers, including hidden layer
        """
        super().__init__()
        # self.feats = MLP(in_dims, hidden_dims, hidden_dims, layers // 2, dropout)
        # self.mean = MLP(hidden_dims, out_dims, hidden_dims, layers - layers // 2, dropout)
        # self.logprec = MLP(hidden_dims, out_dims, hidden_dims, layers - layers // 2, dropout)
        self.mean = MLP(in_dims, out_dims, hidden_dims, layers, dropout)
        self.logprec = MLP(in_dims, out_dims, hidden_dims, layers, dropout)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)

    def forward(self, x):
        torch.flatten(x, start_dim=1)
        # x = self.feats(x)
        mean = self.mean(x)
        logprec = self.logprec(x)
        # logprec = self.logprec(torch.concatenate([x, mean], axis=-1))
        return mean, logprec

    def sample(self, x, random=True):
        """
        Sample from learned data model

        Inputs:
        -------
        x - [BxN array] label
        """
        mu, logprec = self.forward(x)
        if random:
            return mu + self.N.sample(mu.shape) * torch.exp(logprec)**(-0.5)
        else:
            return mu, torch.exp(logprec)**(-0.5)

    def trainer(self, data, epochs=20, save="models/vae.cp", plot=True, loss_type='mle',
                optimizer='adam', lr=0.0001, gamma=0.01, rho=None):
        """
        Train the Heteroskedastic Regression model

        Inputs:
        -------
        data - [DataLoader] - training data
        epochs - [int] number of epochs
        loss_type - [str] type of loss
        optimizer - [str] type of optimizer
        lr - [float] learning rate
        gamma - [float] trade-off between regularization and likelihood maximization
        rho - [float] trade-off between mean regularization and precision regularization
        save - [str] file path to save trained model to after training (and after every 20 minutes)
        plot - [boolean] if plots of loss curves and samples should be produced
        """
        # Training parameters
        # Regularization, reduce to a line search
        gamma = gamma
        rho = rho if rho is not None else 1 - gamma
        # L2 weight decay
        alpha = (1 - rho) / rho * gamma
        beta = (1 - rho) / rho * (1 - gamma)
        print('alpha: %.3f, beta: %.3f' % (alpha, beta))

        pms = [{'params': self.mean.parameters(), 'lr': lr, 'weight_decay': alpha},
               {'params': self.logprec.parameters(), 'lr': lr, 'weight_decay': beta}]
        if optimizer == 'adam':
            opt = torch.optim.Adam(pms)
        elif optimizer == 'sgd':
            opt = torch.optim.SGD(pms)
        else:
            raise ValueError('Unknown optimizer')

        # Train and checkpoint every 20 minutes
        losses = []
        for epoch, batch in progress(range(epochs), inner=data, text='Training',
                                     timed=[(1200, lambda: torch.save(self.state_dict(), save))]):
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            opt.zero_grad()
            mu, logprec = self(x)
            prec = torch.exp(logprec)
            if loss_type == 'mle':
                if epoch < epochs / 3:
                    # first only mean, via MSE
                    loss = ((y - mu) ** 2).mean()
                else:
                    # non-iid gaussians -> maximum likelihood
                    loss = (prec * (y - mu) ** 2 - logprec).mean()
            else:
                raise ValueError('Unknown loss')

            torch.clip(loss, min=-1e5, max=1e5).backward()
            losses += [loss.item()]
            opt.step()
        print('Last-epoch loss: %.2f' % sum(losses[-len(data):-1]))
        print('Finished Training')

        if plot:
            plt.plot(np.array(losses)[:-1])
            plt.savefig('results/tmp_loss.png')
            plt.figure(figsize=(12, 6))
            plt.plot((y[0:500] - mu[0:500]).detach().cpu().numpy().T, c="C0", alpha=1/100)
            plt.plot((-prec**(-0.5)).detach().cpu().numpy().T, c="C1", alpha=1/100)
            plt.plot((prec**(-0.5)).detach().cpu().numpy().T, c="C1", alpha=1/100)
            plt.tight_layout()
            plt.savefig('results/tmp_last_batch.png')
            # plt.show()
            plt.close('all')
