# AIForAlzheimers_EyeEyeCaptain
OCT-powered Early Detection for AI for Alzheimer’s Hackathon — seeing the future… literally starts with looking into the eye. 👁️✨

# Alzheimer's Disease Classification using Retinal OCT

This project implements a complete pipeline for classifying
Alzheimer's disease vs healthy controls using retinal OCT / fOCT data.

## Structure
- datasets/: data loading
- models/: neural network architectures
- train/: training and evaluation logic
- utils/: aggregation and metrics
- configs/: experiment configuration

## Run
```bash
pip install -r requirements.txt
python main.py --config configs/default.yaml
