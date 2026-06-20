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

## Citation

If you find this work useful, please cite:

```bibtex
@article{ye2026tfpa,
  title={TFPA: Enhancing adversarial attack on speech recognition via Time-Frequency Pre-alignment},
  author={Ye, Xiangyu and Xiao, Yatie and Chen, Kongyang and Guan, Qingxiao and Liu, Zhenbang},
  journal={Knowledge-Based Systems},
  pages={115992},
  year={2026},
  publisher={Elsevier}
}
