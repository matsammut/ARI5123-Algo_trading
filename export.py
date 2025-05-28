import pandas as pd
import numpy as np
import yfinance as yf

# Ensure the data is sorted by date
technical_layer = yf.download("^GSPC", start="2008-01-01", end="2013-12-31", interval="1d")
technical_layer.reset_index(inplace=True)
technical_layer.dropna(inplace=True)


# Parameters
lookback = 14  # typical lookback for most of these indicators

# 1. Stochastic %K
low_min = technical_layer['Low'].rolling(window=lookback).min()
high_max = technical_layer['High'].rolling(window=lookback).max()
technical_layer['Stochastic_%K'] = 100 * ((technical_layer['Close'] - low_min) / (high_max - low_min))

# 2. Williams %R
technical_layer["Williams_%R"] = -100 * ((high_max - technical_layer['Close']) / (high_max - low_min))

# 3. Stochastic %D (3-period SMA of %K)
technical_layer['Stochastic_%D'] = technical_layer['Stochastic_%K'].rolling(window=3).mean()

# 4. A/D Oscillator (Accumulation/Distribution Line)
ad = ((technical_layer['Close'] - technical_layer['Low']) - (technical_layer['High'] - technical_layer['Close'])) / (technical_layer['High'] - technical_layer['Low']) * technical_layer['Volume']
technical_layer['AD_Line'] = ad.cumsum()
technical_layer['AD_Oscillator'] = technical_layer['AD_Line'] - technical_layer['AD_Line'].shift(lookback)

# 5. Momentum (Close - Close n periods ago)
technical_layer['Momentum'] = technical_layer['Close'] - technical_layer['Close'].shift(lookback)

# 6. Disparity (Close / Moving Average * 100)
technical_layer['Disparity'] = (technical_layer['Close'] / technical_layer['Close'].rolling(window=lookback).mean()) * 100

# 7. Rate of Change (ROC)
technical_layer['ROC'] = ((technical_layer['Close'] - technical_layer['Close'].shift(lookback)) / technical_layer['Close'].shift(lookback)) * 100

# Display relevant columns
technical_indicators = technical_layer[['Date', 'Stochastic_%K', 'Williams_%R', 'Stochastic_%D','AD_Oscillator', 'Momentum', 'Disparity', 'ROC']]
technical_indicators.dropna(inplace=True)
print(technical_indicators.head())

from gensim.models import Word2Vec
import re

embedding_layer = pd.read_csv('Sorted_Articles_Reduced.csv')  
embedding_layer = embedding_layer.sort_values('Date').reset_index(drop=True)  

def preprocess_title(title):
    title = str(title).lower()
    title = title.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    tokens = re.findall(r"\b[a-zA-Z']+\b", title)
    return tokens

token_list = embedding_layer['Article_title'].apply(preprocess_title)

model = Word2Vec(sentences=token_list, vector_size=100, window=5, min_count=1, workers=4)

def get_sentence_vector(tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

embedding_layer['sentence_vector'] = token_list.apply(get_sentence_vector)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the combined CNN + LSTM model
class NewsTechLSTM(nn.Module):
    def __init__(self, embedding_dim=100, cnn_out_channels=64, news_lstm_hidden_size=128, tech_lstm_hidden_size=64, lstm_layers=1, dropout=0.5, num_classes=2):
        super(NewsCNN_LSTM, self).__init__()

        # CNN Layer
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(kernel_size=2)  # Optional: downsample after CNN

        # LSTM Layer
        self.news_lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=news_lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True, bidirectional=False)
        self.tech_lstm = nn.LSTM(input_size=7, hidden_size=tech_lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True, bidirectional=False)
                     

        # Fully Connected + Softmax
        self.fc = nn.Linear(news_lstm_hidden_size + tech_lstm_hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, news_seq,tech_seq):
        # x shape: (batch_size, seq_len, embedding_dim)
        x = news_seq.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len) for Conv1D
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # Back to (batch_size, seq_len, channels) for LSTM

        # LSTM layer
        news_out, _ = self.news_lstm(x) 
        news_last = news_out[:, -1, :]  
        tech_out, _ = self.tech_lstm(tech_seq)
        tech_last = tech_out[:, -1, :] 

        combined = torch.cat((news_last,tech_last),dim=1)
        out=self.fc(combined)
        return self.softmax(out)

# Example model instantiation
model = NewsCNN_LSTM()

# Example dummy input: batch of 4 days, each with max_seq_len=10, embedding_dim=100
batch_size = 4
max_seq_len = 10
embedding_dim = 100

dummy_input = torch.randn(batch_size, max_seq_len, embedding_dim)

# Forward pass
output = model(dummy_input)

# Example interpretation
for i, prediction in enumerate(output):
    label = [1,0] if prediction.argmax().item() == 0 else [0,1]
    direction = 'up' if label == [1,0] else 'down'
    print(f"Day {i+1} prediction: {label} -> {direction}")