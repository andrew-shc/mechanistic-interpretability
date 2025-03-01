from pathlib import Path
import torch
from torch.utils.data import Dataset
import os


class HiddenStateDataset(Dataset):
    def __init__(self, layer: int, path: Path):
        self.hidden_states = []
        self.layer = layer

        for files in os.listdir(path):
            self.hidden_states.append(torch.load(path+"/"+files))
        # self.hidden_states = [(k, hidden_states[k]) for k in hidden_states]

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        return self.hidden_states[idx]["decoded_output"], self.hidden_states[idx]["hidden_states"][self.layer].to(0)

