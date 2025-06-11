import torch
import numpy as np
import torch.nn.functional as F
import torch.optim 
import torchaudio
import librosa
import random
from tqdm import tqdm
sample_rate = 16000  
n_fft       = 400    
win_length  = None   
hop_length  = 160    
n_mels      = 80     
class TFPA:
    def __init__(self, target, corpus, device):
        self.target = target
        self.corpus = corpus
        self.device = device
        self.mfcc_transform = torchaudio.transforms.MFCC(
                                sample_rate=16000,
                                n_mfcc=40,
                                melkwargs={"n_fft":400, "hop_length":160, "n_mels":128}
                            ).to(device)
    def preprocess_waveform(self, sig, target_length):
        """
        Crop/pad and normalize raw waveform to [1, target_length].
        """
        L = sig.shape[1]
        if L > target_length:
            start = torch.randint(0, L - target_length + 1, (1,)).item()
            seg = sig[:, start:start + target_length]
        else:
            seg = F.pad(sig, (0, target_length - L))
        mean = seg.mean(dim=1, keepdim=True)
        std = seg.std(dim=1, keepdim=True, unbiased=False) + 1e-8
        return (seg - mean) / std

    def apply_spec_augment(self, spectrogram,
                        time_mask_param=7,
                        freq_mask_param=4,
                        num_time_masks=1,
                        num_freq_masks=2):
        spectrogram = spectrogram.squeeze(0)
        T, F = spectrogram.shape
        spec_aug = spectrogram.clone()
        # Random time-mask
        for _ in range(num_time_masks):
                t = random.randint(0, time_mask_param)
                t0 = random.randint(0, max(1, T - t))
                spec_aug[t0:t0 + t, :] = 0
        
        # Random frquence-mask
        for _ in range(num_freq_masks):
            f = random.randint(0, freq_mask_param)
            f0 = random.randint(0, F - f)
            spec_aug[:, f0:f0 + f] = 0
        
        return spec_aug.unsqueeze(0)

    def iniPer(self):
        corpus = self.corpus
        per = torch.zeros(1, int(10*16000), device=self.device, requires_grad=True)
        tar = self.deidentify_stft(self.target)
        t_in = self.preprocess_waveform(tar, int(10*16000))
        target_spec = self.compute_mfcc(t_in)
        optimizer = torch.optim.AdamW([per], lr=0.1)
        for i in tqdm(range(60)):
            g = 0
            tl = 0
            with torch.no_grad():
                per.clamp_(-1, 1)
            for ab in corpus:
                grad_sum = 0.0
                loss_sum = 0.0
                a = ab.unsqueeze(0).to(self.device)
                if per.shape[1] < a.shape[1]:
                    adv_audio = a + torch.cat([per, torch.zeros(1, a.shape[1] - per.shape[1]).to(self.device)], dim=1).to(self.device).requires_grad_(True)
                else:
                    adv_audio = a + per[..., :a.shape[1]].requires_grad_(True)
                adv_processed = self.preprocess_waveform(adv_audio, int(10*16000))
                adv_specs = []
                for _ in range(8):
                    spec = self.compute_mfcc(adv_processed)
                    spec = self.apply_spec_augment(spec)
                    adv_specs.append(spec)
                for adv in adv_specs:
                    loss2 = F.cosine_similarity(adv_processed, t_in.to(self.device))
                    loss = self.SCloss(adv, target_spec) + 0.5 * loss2
                    loss_sum += loss.item()
                    grad = torch.autograd.grad(loss, per, retain_graph=True)[0]
                    grad_sum += grad
                g += grad_sum/len(adv_specs)
                tl += loss_sum
            optimizer.zero_grad()
            with torch.no_grad():
                per.grad = g/len(corpus)
            optimizer.step()
            with torch.no_grad():
                per.clamp_(-1, 1)
        return per.squeeze(0).detach()
