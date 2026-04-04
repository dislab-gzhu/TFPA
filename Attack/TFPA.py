from tqdm import tqdm
import torch, torchaudio, random, librosa
import pandas as pd
import numpy as np
import torch.nn.functional as F


class TFPA:
    def __init__(self, target, corpus, device):
        self.target = target
        self.corpus = corpus
        self.device = device
        self.mfcc_transform = torchaudio.transforms.MFCC(
                                sample_rate=16000,
                                n_mfcc=40,
                                melkwargs={"n_fft":400, "hop_length":160, "n_mels":128}
                            ).to(self.device)
    def preprocess_waveform(self, sig, target_length):
        L = sig.shape[1]
        if L > target_length:
            start = torch.randint(0, L - target_length + 1, (1,)).item()
            seg = sig[:, start:start + target_length]
        else:
            seg = F.pad(sig, (0, target_length - L))
        mean = seg.mean(dim=1, keepdim=True)
        std = seg.std(dim=1, keepdim=True, unbiased=False) + 1e-8
        return (seg - mean) / std
    def compute_mfcc(self, waveform):
        if waveform.dim() == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)
        mfcc = self.mfcc_transform(waveform.to(self.device))
        mfcc = (mfcc - mfcc.mean(dim=-1, keepdim=True)) \
               / (mfcc.std(dim=-1, keepdim=True) + 1e-8)
        return mfcc
    def deidentify_stft(self, wav, sr=16000,
                        n_fft=1024, hop_length=256):
        wav = wav.detach().cpu().numpy().astype(np.float64).squeeze()
        S = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
        mag, phase = np.abs(S), np.angle(S)
        log_mag = np.log1p(mag)
        mu_s   = np.mean(log_mag, axis=1, keepdims=True)
        log_mag_norm = log_mag - mu_s
        mag_norm     = np.expm1(log_mag_norm).clip(min=0)
        S_new = mag_norm * np.exp(1j * phase)
        wav_deid = librosa.istft(S_new, hop_length=hop_length, win_length=n_fft,
                             length=wav.shape[-1])
        res = torch.from_numpy(wav_deid.astype(np.float32))
        return res.unsqueeze(0)
    def apply_spec_augment(self, spectrogram,
                        time_mask_param=7,
                        freq_mask_param=4,
                        num_time_masks=1,
                        num_freq_masks=2):
        spectrogram = spectrogram.squeeze(0)
        T, F = spectrogram.shape
        spec_aug = spectrogram.clone()
        for _ in range(num_time_masks):
            t = random.randint(0, min(time_mask_param, T - 1))
            t0 = random.randint(0, max(0, T - t))
            spec_aug[t0:t0 + t, :] = 0

        for _ in range(num_freq_masks):
            f = random.randint(0, min(freq_mask_param, F - 1))
            f0 = random.randint(0, max(0, F - f))
            spec_aug[:, f0:f0 + f] = 0

        return spec_aug.unsqueeze(0)
    def SCloss(self, adv_spec, target_spec, scales=None):
        if scales is None:
            scales = [1, 2, 4, 8]
        loss = 0
        for s in scales:
            pool = torch.nn.AvgPool1d(kernel_size=s, stride=s)
            loss += F.mse_loss(pool(adv_spec), pool(target_spec))
        diff = target_spec - adv_spec
        num = torch.norm(diff, p=2, dim=(1,2))
        den = torch.norm(adv_spec, p=2, dim=(1,2)) + 1e-8
        sc_loss = (num/den).mean()
        lamada = (loss.item() / sc_loss.item())
        loss = lamada * sc_loss +  loss
        return loss
    def iniPer(self):
        corpus = self.corpus
        per = torch.zeros(1, int(10*16000), device=self.device, requires_grad=True)
        tar = self.deidentify_stft(self.target)
        t_in = self.preprocess_waveform(tar, int(10*16000))
        target_spec = self.compute_mfcc(t_in)
        optimizer = torch.optim.AdamW([per], lr=0.01)

        for i in tqdm(range(60)):
            g = torch.zeros_like(per)
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
                    loss = 30 * (0.5 * self.SCloss(adv, target_spec) + 1 * (1-loss2))
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
    
