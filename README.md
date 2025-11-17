# NOAA Wind Direction Prediction

A deep learning project for predicting wind direction using NOAA meteorological data with LSTM neural networks.

## Overview

This project implements a multi-class classification model to predict wind direction categories using historical weather data. The model processes time-series meteorological measurements to forecast wind patterns.

## Architecture

The model employs a stacked LSTM (Long Short-Term Memory) architecture optimized for sequential weather data:

- **Input Layer**: Time-series sequences with 3 timesteps and 11 meteorological features
- **LSTM Layers**: 
  - First LSTM layer: 128 units with return sequences
  - Second LSTM layer: 64 units with return sequences  
  - Third LSTM layer: 32 units
- **Regularization**: Dropout layers (rate: 0.2) after each LSTM to prevent overfitting
- **Output Layer**: Dense layer with 6 units (6 wind direction classes) using sparse categorical crossentropy

**Total Parameters**: 43,200 trainable parameters

## Model Performance

The model is trained using:
- Optimizer: Adam
- Loss Function: Sparse categorical crossentropy
- Early stopping with patience=3 epochs on validation loss
- Training conducted on GPU for accelerated computation

## Data

The project uses NOAA meteorological data (`MA615_F21_Project_Final.csv`) containing features such as temperature, humidity, pressure, and other weather variables collected over time.

## Requirements

- TensorFlow/Keras
- pandas
- numpy

## Usage

1. Upload your NOAA dataset
2. The notebook preprocesses data into 3-timestep sequences
3. Train the LSTM model with early stopping
4. Model predicts wind direction categories from input weather patterns

## Note

Originally developed in Google Colab with GPU acceleration enabled.
