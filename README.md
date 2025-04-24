# Facial Expression Recognition System

[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A Convolutional Neural Network (CNN) based facial expression recognition system that classifies emotions into 7 categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) using the FER2013 dataset.

## Features

- 4-layer CNN architecture with 2 fully-connected layers
- Data augmentation for improved generalization
- Training visualization (accuracy/loss curves)
- Confusion matrix analysis
- Model saving/loading capability

## Dataset

The model uses the [FER2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) from Kaggle.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/facial-expression-recognition.git
cd facial-expression-recognition
