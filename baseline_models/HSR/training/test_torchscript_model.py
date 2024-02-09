import re
import torch

from hsr import HeteroskedasticRegression

def load_models(device: torch.device):
    # Load regular pytorch model
    model = HeteroskedasticRegression(hidden_dims=1024, layers=4, dropout=0.).to(device)
    state_dict = torch.load("../model/final_hsr.cp", map_location=device)

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

    # Load torchscript model to the right device
    scripted_model = torch.jit.load("../model/final_hsr_new.pt", map_location=device)

    return model, scripted_model

if __name__ == "__main__":
    # Choose GPU if available, else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, jit_model = load_models(device)

    batch_size = 32
    input_dim = 124
    input = torch.randn((batch_size, input_dim), device=device)

    model_mean, model_logprec = model(input)
    jit_model_mean, jit_model_logprec = jit_model(input)

    assert torch.allclose(model_mean, jit_model_mean)
    assert torch.allclose(model_logprec, jit_model_logprec)

    print("TorchScript model output matches original PyTorch model output! :)")
