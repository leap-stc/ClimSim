import re
import torch

import hsr

# Manually change device to CPU to workaround TorchScript issues
hsr.device = "cpu"

from hsr import HeteroskedasticRegression

class MeanWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, x):
        mean, _ = self.module(x)
        # Only return mean, ignore logprec
        return mean

model = HeteroskedasticRegression(hidden_dims=1024, layers=4, dropout=0.)
state_dict = torch.load("../model/final_hsr.cp")

# Translate to new state dict
state_dict_new = {}
for k, v in state_dict.items():
    match = re.search(r"(mean|logprec).linear[0-9]", k)
    if match:
        start, end = match.span()
        k_new = k[:end-8] + ".linears." + k[end-1:]
        state_dict_new[k_new] = v
    else:
        state_dict_new[k] = v

# Load translated state dictionary
model.load_state_dict(state_dict_new)

# Wrap the model to only output the mean
model = MeanWrapper(model)

# JIT compile model using scripting
scripted_model = torch.jit.script(model)

# Convert to eval mode (to avoid dropout at inference time)
scripted_model = scripted_model.eval()

# Save serialized torchscript model out
scripted_model.save("../model/final_hsr_wrapped.pt")
