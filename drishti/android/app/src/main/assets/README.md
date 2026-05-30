# Sherpa-ONNX TTS Assets

Place the following files in this directory before building:

## Required Files

1. **Piper ONNX Model** — `vits-piper-en_US-libritts_r-medium.onnx`
2. **Tokens file** — `tokens.txt`
3. **eSpeak-ng data** — `espeak-ng-data/` (entire directory)

## Download Instructions

### Option A: From Sherpa-ONNX releases
1. Go to: https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
2. Download `vits-piper-en_US-libritts_r-medium.tar.bz2`
3. Extract and place files here

### Option B: Direct download script
```bash
# Run from this directory
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2
tar xf vits-piper-en_US-libritts_r-medium.tar.bz2
# Move the files into this assets/ directory
```

## Expected structure after setup
```
assets/
├── README.md (this file)
├── vits-piper-en_US-libritts_r-medium.onnx
├── tokens.txt
└── espeak-ng-data/
    ├── ... (phoneme data files)
```
