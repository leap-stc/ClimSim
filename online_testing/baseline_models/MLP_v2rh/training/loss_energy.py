import torch

'''
a loss function that compares the column integrated mse tendencies between the model and the truth
'''

def loss_energy(pred, truth, ps, hyai, hybi, out_scale):
    """
    Compute the energy loss.
    
    Parameters:
    - pred (torch.Tensor): Predictions from the model. Shape: (batch_size, 368).
    - truth (torch.Tensor): Ground truth. Shape: (batch_size, 368).
    - ps (torch.Tensor): Surface pressure. Shape: (batch_size). with original unit of Pa.
    - hyai (torch.Tensor): Coefficients for calculating pressure at layer interfaces for mass. Shape: (61).
    - hybi (torch.Tensor): Coefficients for calculating pressure at layer interfaces for mass. Shape: (61).
    - out_scale (float): Output scaling factor. shape: (368).
    """
    #code for reference
    # state_ps = np.reshape(state_ps, (-1, self.num_latlon))
    # pressure_grid_p1 = np.array(self.grid_info['P0']*self.grid_info['hyai'])[:,np.newaxis,np.newaxis]
    # pressure_grid_p2 = self.grid_info['hybi'].values[:, np.newaxis, np.newaxis] * state_ps[np.newaxis, :, :]
    # self.pressure_grid_train = pressure_grid_p1 + pressure_grid_p2
    # self.dp_train = self.pressure_grid_train[1:61,:,:] - self.pressure_grid_train[0:60,:,:]

    # convert out_scale to torch tensor if not
    if not torch.is_tensor(out_scale):
        out_scale = torch.tensor(out_scale, dtype=torch.float32)
    # convert hybi and hyai to torch tensor if not
    if not torch.is_tensor(hybi):
        hybi = torch.tensor(hybi, dtype=torch.float32)
    if not torch.is_tensor(hyai):
        hyai = torch.tensor(hyai, dtype=torch.float32)

    L_V = 2.501e6   # Latent heat of vaporization
    # L_I = 3.337e5   # Latent heat of freezing
    # L_F = L_I
    # L_S = L_V + L_I # Sublimation
    C_P = 1.00464e3 # Specific heat capacity of air at constant pressure

    dt_pred = pred[:,0:60]/out_scale[0:60]
    dt_truth = truth[:,0:60]/out_scale[0:60]
    dq_pred = pred[:,60:120]/out_scale[60:120]
    dq_truth = truth[:,60:120]/out_scale[60:120]

    # calculate the pressure difference, make ps (batch_size, 1)
    ps = ps.reshape(-1,1)
    pressure_grid_p1 = 1e5 * hyai.reshape(1,-1) # (1, 61)
    pressure_grid_p2 = hybi.reshape(1,-1) * ps # (batch_size, 61)
    pressure_grid = pressure_grid_p1 + pressure_grid_p2 # (batch_size, 61)
    dp = pressure_grid[:,1:] - pressure_grid[:,:-1] # (batch_size, 60)

    # calculate the integrated tendency
    dt_integrated_pred = torch.sum(dt_pred * dp, dim=1) # (batch_size)
    dt_integrated_truth = torch.sum(dt_truth * dp, dim=1) # (batch_size)
    dq_integrated_pred = torch.sum(dq_pred * dp, dim=1) # (batch_size)
    dq_integrated_truth = torch.sum(dq_truth * dp, dim=1) # (batch_size)

    # energy loss, note moist static energy is the sum of dry static energy and latent heat, h = cp*T + gz + Lq
    energy_loss = torch.mean((C_P * dt_integrated_pred + L_V * dq_integrated_pred - C_P * dt_integrated_truth - L_V * dq_integrated_truth)**2)

    return energy_loss

