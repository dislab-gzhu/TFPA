import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import whisper
from TFPA import TFPA
from TUAP import TUAP
from WavDataset import WavDataset
from torch.utils.data import Dataset, DataLoader
TARGET_PATH = "TFPA/CommandTarget/Open_the_door.mp3" ## samples
TARGET_COMMAND = "open the door"
MODLE_ID = "TFPA/model/whisper-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_LIST = 'TFPA/data/train_corpus'
ETA = 0.95
MAX_EPOCHES = 1000
    
def main():
    whisper_model = WhisperForConditionalGeneration.from_pretrained(MODLE_ID)
    whisper_processor = WhisperProcessor.from_pretrained(MODLE_ID)
    whisper_model.to(DEVICE)
    dataset = WavDataset(TRAIN_LIST)
    train_set = list(dataset)
    target = torch.from_numpy(whisper.load_audio(TARGET_PATH))
    tf = TFPA(target, train_set, DEVICE)
    per = tf.iniPer().to(DEVICE).requires_grad_(True)
    attacker = TUAP(whisper_model, whisper_processor, train_set, TARGET_COMMAND, ETA, DEVICE, MAX_EPOCHES, per=per)
    delta = attacker.forward()
    torch.save(delta.detach(), "Delta.pt")
if __name__ == "__main__":
    main()
