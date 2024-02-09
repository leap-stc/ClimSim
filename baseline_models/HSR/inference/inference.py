import numpy as np
import torch

def load_model(device: torch.device):
    # Load torchscript model to the right device
    scripted_model = torch.jit.load("../model/final_hsr_wrapped.pt", map_location=device)
    return scripted_model

if __name__ == "__main__":
    # Choose GPU if available, else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_model(device)

    input = torch.from_numpy(np.loadtxt("test_input.txt", dtype=np.float32)).to(device)
    input = torch.unsqueeze(input, 0)

    output_mean = model(input)

    print("Saving python model output to python_hsr_wrapped_output.txt")
    np.savetxt("python_hsr_wrapped_output.txt", (output_mean.detach().cpu().numpy()[0,:],))
