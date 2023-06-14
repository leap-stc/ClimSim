import matplotlib.pyplot as plt
import numpy as np
import torch
from tools import progress


"""
Contains the code for the Conditional Variational Autoencoder and its training.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VariationalEncoder(torch.nn.Module):
    """
    Conditional VAE Encoder with <layers>+1 fully connected layer
    """
    def __init__(self, in_dims, hidden_dims=512, latent_dims=3, layers=1, dropout=0):
        super().__init__()
        self.linears = []
        for i in range(layers):
            self.linears += [torch.nn.Sequential(
                torch.nn.Linear(in_dims if i == 0 else hidden_dims, hidden_dims),
                torch.nn.LayerNorm(hidden_dims),
                torch.nn.Dropout(p=dropout))
                ]
            self.add_module('linear%d' % i, self.linears[-1])
        self.linear_mean = torch.nn.Linear(hidden_dims, latent_dims)
        self.linear_logstd = torch.nn.Linear(hidden_dims, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, y, x, return_latent=False):
        y = torch.cat([y, x], 1)
        y = torch.flatten(y, start_dim=1)
        for linear in self.linears:
            y = torch.nn.functional.relu(linear(y))
        mu = self.linear_mean(y)
        if return_latent:
            return mu
        else:
            sigma = torch.exp(self.linear_logstd(y))
            z = mu + sigma * self.N.sample(mu.shape)
            self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).mean()
            return z


class Decoder(torch.nn.Module):
    """
    Conditional VAE Decoder with <layers>+1 fully connected layer
    """
    def __init__(self, out_dims, hidden_dims=512, latent_dims=3, layers=1, dropout=0):
        super().__init__()
        self.linears = []
        for i in range(layers):
            self.linears += [torch.nn.Sequential(
                torch.nn.Linear(latent_dims if i == 0 else hidden_dims, hidden_dims),
                torch.nn.LayerNorm(hidden_dims),
                torch.nn.Dropout(p=dropout))
                ]
            self.add_module('linear%d' % i, self.linears[-1])
        self.final_linear1 = torch.nn.Linear(hidden_dims, out_dims)
        # self.final_linear2 = torch.nn.Linear(hidden_dims, out_dims)
        self.final_log_std1 = torch.nn.Linear(hidden_dims, out_dims)
        # self.final_log_std2 = torch.nn.Linear(hidden_dims, out_dims)
        # self.final_lambda = torch.nn.Linear(hidden_dims, 1)
        # self.final_prob_zero = torch.nn.Linear(hidden_dims, out_dims)

    def forward(self, z, x):
        z = torch.cat([z, x], 1)
        for linear in self.linears:
            z = torch.nn.functional.relu(linear(z))
        m1 = self.final_linear1(z)
        # m2 = self.final_linear2(z)
        s1 = torch.exp(self.final_log_std1(z))
        # s2 = torch.exp(self.final_std2(z))
        # lam = self.final_lambda
        # p0 = torch.sigmoid(self.final_prob_zero(z))
        return m1, s1


class ConditionalVAE(torch.nn.Module):
    def __init__(self, beta=0.01, data_dims=124, label_dims=128,
                 latent_dims=3, hidden_dims=512, layers=2, dropout=0):
        """
        Conditional VAE
        Encoder: [y x] -> [mu/sigma] -sample-> [z]
        Decoder: [z x] -> [y_hat]

        Inputs:
        -------
        beta - [float] trade-off between KL divergence (latent space structure) and reconstruction loss
        data_dims - [int] size of x
        label_dims - [int] size of y
        latent_dims - [int] size of z
        hidden_dims - [int] size of hidden layers
        layers - [int] number of layers, including hidden layer
        """
        super().__init__()
        self.latent_dims = latent_dims
        self.label_dims = label_dims
        # Encoder and Decoder are conditioned on x of size data_dims
        self.encoder = VariationalEncoder(label_dims + data_dims, hidden_dims, latent_dims, layers, dropout)
        self.decoder = Decoder(label_dims, hidden_dims, latent_dims + data_dims, layers, dropout)
        self.beta = beta

    def forward(self, y, x, return_latent=False, batch_norm=False):
        # Normalize
        if batch_norm:
            x_m, x_s = x.mean(axis=0), x.std(axis=0)
            y_m, y_s = y.mean(axis=0), y.std(axis=0)
            mx, my = x_s != 0, y_s != 0
            x[:, mx] = x[:, mx] / x_s[mx] - x_m[mx]
            y[:, my] = y[:, my] / y_s[my] - y_m[my]
        z = self.encoder(y, x, return_latent)
        if return_latent:
            return z
        else:
            y_hat_mean, y_hat_std = self.decoder(z, x)
            if batch_norm:
                y_hat_mean = (y_hat_mean + y_m) * y_s
            return y_hat_mean, y_hat_std

    def sample(self, x, random=True):
        """
        Sample conditionally on x

        Inputs:
        -------
        x - [BxN array] label
        random - [boolean] if true sample latent variable from prior else use all-zero vector
        """
        if random:
            # Draw from prior
            z = self.encoder.N.sample([x.shape[0], self.latent_dims])
        else:
            # Set to prior mean
            z = torch.zeros([x.shape[0], self.latent_dims]).to(device)
        mean_y, std_y = self.decoder(z, x)
        if random:
            # add output noise
            y = mean_y + self.encoder.N.sample(mean_y.shape) * std_y
            # y = torch.zeros_like(mean_y)
            # nz = torch.rand(y.shape).to(device) > p0
            # y[nz] = mean_y[nz] + self.encoder.N.sample([(nz == 1).sum()]) * std_y[nz]
            return y
        else:
            return mean_y, std_y

    def trainer(self, data, epochs=20, save="models/vae.cp", plot=True, loss_type='mse',
                optimizer='adam', lr=0.0001, weight_decay=0):
        """
        Train the Conditional VAE

        Inputs:
        -------
        data - [DataLoader] - training data
        epochs - [int] number of epochs
        loss_type - [str] type of loss
        optimizer - [str] type of optimizer
        lr - [float] learning rate
        weight_decay - [float] L2 regularization
        save - [str] file path to save trained model to after training (and after every 20 minutes)
        plot - [boolean] if plots of loss curves and samples should be produced
        """
        # Training parameters
        if optimizer == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'sgd':
            opt = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Unknown optimizer')

        # Train and checkpoint every 20 minutes
        losses = []
        for epoch, batch in progress(range(epochs), inner=data, text='Training',
                                     timed=[(1200, lambda: torch.save(self.state_dict(), save))]):
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            opt.zero_grad()
            y_mean, y_std = self(y, x)
            # y_mean, y_std, p0 = self(y, x)
            if loss_type == 'mse':
                # iid gaussians -> mse
                # y_hat = y_mean + self.encoder.N.sample([x.shape[0], 128]) * y_std
                # loss = ((y - y_hat) ** 2).sum() / self.label_dims + self.beta * self.encoder.kl / self.latent_dims

                # means, so beta' = beta * label_dims / latent_dims
                loss = (0.5 * (y - y_mean) ** 2 / y_std + torch.log(y_std)).mean() + self.beta * self.encoder.kl
                # model as p0 * N(0, 1/1000) + (1-p0) * N(mean, std)
                # loss = (p0 * y**2).sum() + ((1 - p0) * ((y - y_mean) ** 2 / y_std + torch.log(y_std))).mean() + self.beta * self.encoder.kl
            else:
                raise ValueError('Unknown loss')

            torch.clip(loss, min=-1e5, max=1e5).backward()
            losses += [loss.item()]
            opt.step()
        print('Last-epoch loss: %.2f' % sum(losses[-len(data):-1]))
        print('Finished Training')

        if plot:
            y_hat = y_mean + self.encoder.N.sample([x.shape[0], 128]) * y_std
            plt.plot(np.array(losses)[:-1])
            plt.savefig('results/tmp_loss.png')
            fig, ax = plt.subplots(4, 1, sharey=True, figsize=(12, 8))
            ax[0].plot((y[0:500] - y_mean[0:500]).detach().cpu().numpy().T, c="C0", alpha=1/100)
            ax[1].plot((y[0:500] - y_hat[0:500]).detach().cpu().numpy().T, c="C0", alpha=1/100)
            ax[2].plot((y[0:500] - self.sample(x[0:500])).detach().cpu().numpy().T, c="C0", alpha=1/100)
            ax[3].plot((y[0:500] - self.sample(x[0:500], random=False)[0]).detach().cpu().numpy().T, c="C0", alpha=1/100)
            ax[0].set_ylabel('y - rec. sample')
            ax[1].set_ylabel('y - rec. mean')
            ax[2].set_ylabel('y - sample')
            ax[3].set_ylabel('y - mean')
            ax[0].set_ylim([-0.5, 0.5])
            plt.tight_layout()
            plt.savefig('results/tmp_last_batch.png')
            # plt.show()
            plt.close('all')
