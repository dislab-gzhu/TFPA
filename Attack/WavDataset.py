import torch
import whisper
import os
from torch.utils.data import Dataset

class WavDataset(Dataset):
    def __init__(self, wav_dir: str, transform=None):
        super().__init__()
        self.wav_dir = wav_dir
        self.transform = transform
        self.file_list = [
            os.path.join(wav_dir, fname)
            for fname in os.listdir(wav_dir)
            if fname.lower().endswith('.wav')
        ]
        if not self.file_list:
            raise ValueError(f"在目录 {wav_dir} 中未找到任何 .wav 文件。")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        path = self.file_list[idx]
        waveform = whisper.load_audio(path)
        waveform = torch.from_numpy(waveform)
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform
