# TFPA
This is pytorch repository of "TFPA: Enhancing Adversarial Attack on Speech Recognition via Time-Frequency Pre-alignment"

# Process
1. ```
   conda create -n tfpa python=3.10
   conda activate tfpa
   pip install -r requirements.txt
   ```
2. Download the train dataset to the directory "data"
   ```
   mkdir -p data/train_corpus/
   ```
   Download your training dataset (e.g., LibriSpeech or any custom audio dataset).
      Note: All audio files MUST be in .wav format.

      Do not put them inside subfolders.
   Your folder structure should look exactly like this:
      TFPA_code/
      └── data/
         └── train_corpus/
            ├── sample0.wav
            ├── sample1.wav
            └── ...

3. ```
   python main.py
   ```

