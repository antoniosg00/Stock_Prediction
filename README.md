# Stock Prediction Project

<img src="Images/Main.jpeg" alt="Imágenes generadas por Dall-E 3" style="width:600px;height:300px;">

> Imágenes generadas por Dall-E 3

## Overview.

This project of more than 1600 lines of code focuses on stock price prediction utilizing a combination of Variational Autoencoder (VAE) and Long Short-Term Memory (LSTM) or GRU models. The workflow encompasses data extraction, preprocessing, optimization of VAE hyperparameters, VAE training, data transformation into a latent space, optimization of LSTM model hyperparameters, LSTM model training, and ultimately, visualization of predictions.

Additionally, a Generative Adversarial Network (GAN) has been programmed with training and optimization results that were less than satisfactory (GANs are challenging models that demand substantial computational power) for prediction purposes.

Moreover, there is potential to augment this project with unexplored facets, such as integrating other predictive models based on different architectures like Convolutional Neural Networks (CNNs), extending predictions to multiple days and handling new data, or implementing a Deep Reinforcement Learning (DRL) algorithm. The DRL algorithm could leverage historical data and predictions to efficiently learn trading strategies, aiming for maximum rewards.

The project has focused on predicting the stock price of Banco Santander, although it could easily be extended to other companies.

All codes are thoroughly commented in Spanish. All the instructions provided in this README are explained in greater detail in the comments of the code.

## Table of contents
- [Prerequisites](#Prerequisites)
- [Features](#Features)
- [Usage](#Usage)
- [Possible next developments](#possible-next-developments)

## Prerequisites
Before running the code, make sure you have the required dependencies installed. You can install them using:
```
pip install matplotlib tensorflow pandas numpy yfinance scikit-learn optuna plotly joblib
```

## Features
### Data Extraction and Visualization (modules ETL.py and PLOT.py):
After obtaining historical financial data about our company (closing prices, opening prices, volume, etc.) using the Yahoo Finance API (yfinance), we calculate a series of technical indicators such as various simple and exponential moving averages or Fourier transforms of different components. The calculation methods can be seen in the methods of the classes in the ETL.py module.

We represent all the data and indicators in interactive Plotly charts through the PLOT.py module.

### VAE (VAE.py and OPT.py)
The VAE I have programmed aims to extract high-level features from our dataset in order to obtain a more comprehensive dataset for training our predictive model.

First, we optimize the hyperparameters through fine-tuning using the Optuna library. This is achieved with the OPT_VAE class in the OPT.py module. After a quick manual testing, we deduce acceptable ranges for hyperparameters, and after configuring the OPT_VAE class, we run the Bayesian optimization algorithm using the train set and the validation set obtained from ETL.py.

Having the best hyperparameters, we train the VAE model and use it with the ENTIRE dataset to obtain a tensor of the latent space 'z'.

Once we have the 'z' tensor, we combine it with the original ETL dataset and split it into train, val, and test sets, and into X and Y, respectively. This is done with the VAE2GANdata() function from VAE.py. The new datasets obtained will be used in subsequent recurrent models for predictions.

### Recurrent Model (LSTM.py and OPT.py)
Again, prior to using the recurrent model (which I have configured to be based on both LSTM and GRU architectures), we can optimize hyperparameters using the Optuna library through the OPT_LSTM class in the OPT.py module.

Once done, we train our recurrent model with the selected hyperparameters using the train set and the validation set, and we plot the training graphs as shown in main.py.
With that, we are ready to make predictions on the entire dataset. We can then visualize the predictions for the three datasets, overlaid on the curve of actual data (closing prices of the stocks).

Regarding these predictions, it's important to take the following observation into account:
A simple LSTM network that takes the Close prices themselves as training data will produce a graph much more consistent with the real data, having a lower error. This does not mean that the model is better. If we zoom in, we would discover that in detail, the model fails, as it essentially 'predicts' by copying the stock value from the previous day (in reality, we know that the market doesn't work that way). By adding many more data and indicators that make stock prediction more possible (more likely because they are the data on which investors base their decisions so we would 'guess' the future price by mimicking what they do rather than a real relationship between inputs and outputs), the training of the model becomes much more complex. It is more difficult for the model to converge correctly in the last epochs, something that is evident when looking at the graphs, and we need that.

### GAN (GAN.py)
This code implements a experimental Generative Adversarial Network (GAN) designed to predict future stock prices based on historical data. The GAN consists of a generator and a discriminator, which compete with each other to improve the quality of predictions. The generator is an LSTM model trained to generate sequences of future stock prices based on historical input data. The discriminator, on the other hand, is a Convolutional Neural Network (CNN) trained to distinguish between real and generator-produced data.

The main goal of this code is to train the GAN so that the generator learns to generate sequences of stock prices that are indistinguishable from real data, while the discriminator tries to discern whether the data is real or fake. This competition process between the generator and the discriminator leads to the improvement of generator predictions.

As training epochs progress, the generator is expected to produce accurate predictions of stock prices for the next day, which can be valuable in financial and investment applications. Tips from https://github.com/soumith/ganhacks are followed in the implementation.

However, training and optimization results that were less than satisfactory (GANs are challenging models that demand substantial computational power) for prediction purposes. Undoubtedly, there is much work and research needed to make such a complex system as a GAN work satisfactorily for our goal. It could be said that this code has been a 'first step' to better understand the functioning of these architectures.

## Usage
The way to use the code provided in this repository is explained in the main.ipynb notebook with comments. Perhaps, to make the most of the utilities that the code offers, it may be convenient to take a look at the methods of the various classes provided in the modules.

## Possible next developments
In addition, there exists significant potential for the expansion and enrichment of this project through the exploration of uncharted territories. One avenue for advancement involves the integration of alternative predictive models rooted in various architectures, such as Convolutional Neural Networks (CNNs). By diversifying the model architectures, we can potentially capture more intricate patterns within the historical data, leading to more nuanced and accurate predictions.

Moreover, extending the forecasting capabilities beyond a single day opens up new possibilities for understanding and predicting market trends over longer time frames. This extension could involve the development of more sophisticated time-series forecasting models, considering factors that unfold over a series of days and capturing the evolving dynamics of financial markets.

Additionally, adapting the system to handle new and diverse data sources can enhance its adaptability and applicability. Integrating real-time data feeds, sentiment analysis from social media, or macroeconomic indicators could provide a more comprehensive input for the predictive models, leading to a more holistic understanding of the factors influencing stock prices.

Another exciting avenue for potential development is the implementation of a Deep Reinforcement Learning (DRL) algorithm. By leveraging historical data and predictions, a DRL algorithm could autonomously learn and adapt trading strategies over time. The objective would be to dynamically optimize decision-making processes to maximize rewards and navigate the complexities of financial markets.

In essence, the potential next developments for this project are multifaceted and span from diversifying model architectures to incorporating longer forecasting horizons, adapting to new data sources, and exploring the realm of Deep Reinforcement Learning for enhanced trading strategy optimization. These extensions aim to elevate the project's sophistication and efficacy in capturing the intricate dynamics of financial markets.
