import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import whisper
from Attack.TFPA import TFPA
from Attack.WavDataset import WavDataset
import random
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_PATH = os.path.join(BASE_DIR, "CommandTarget/Open_the_door.mp3")
MODLE_ID = "openai/whisper-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_LIST = os.path.join(BASE_DIR, "data/train_corpus")
ETA = 0.95
MAX_EPOCHES = 1000
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

def whisper_trans(model, processor, input_f):
    with torch.no_grad():
        fids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
        attention_mask = torch.ones((input_f.shape[0], input_f.shape[-1]),
                                    device=input_f.device, dtype=torch.long)
        pre_ids = model.generate(input_f, attention_mask=attention_mask, forced_decoder_ids=fids)
        trs = processor.batch_decode(pre_ids, skip_special_tokens=True)[0]
    return trs

def main():
    whisper_model = WhisperForConditionalGeneration.from_pretrained(MODLE_ID)
    whisper_processor = WhisperProcessor.from_pretrained(MODLE_ID)
    whisper_model.to(DEVICE)
    dataset = WavDataset(TRAIN_LIST)
    train_set = list(dataset)
    target = torch.from_numpy(whisper.load_audio(TARGET_PATH))
    t_mel = whisper.log_mel_spectrogram(target.to(DEVICE), n_mels=whisper_model.config.num_mel_bins).unsqueeze(0).to(DEVICE)
    target_phrase = whisper_trans(whisper_model, whisper_processor, t_mel)
    target_ids = whisper_processor(text=target_phrase, return_tensors="pt").input_ids.to(DEVICE)
    tf = TFPA(target, train_set, DEVICE)
    per = tf.iniPer().to(DEVICE).requires_grad_(True)
    attacker = TUAP(whisper_model, whisper_processor, train_set, target_phrase, target_ids, ETA, DEVICE, MAX_EPOCHES, per=per)
    delta = attacker.forward()
    torch.save(delta.detach(), os.path.join(BASE_DIR, "result/Delta.pt"))
if __name__ == "__main__":
    main()
