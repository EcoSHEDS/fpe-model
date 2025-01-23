import pandas as pd
from models.ranknet_aux import FlowPhotoDataset, FlowPhotoRankingPairsDataset, load_pairs_from_csv

images_dir = 'images'

# daily

time_col = 'date'

pairs_df = load_pairs_from_csv('runs/04_ranknet+scalar_500_lstm/stations/29/input/pairs.csv', timestep='D')
images_df = pd.read_csv('runs/04_ranknet+scalar_500_lstm/stations/29/input/images.csv')
aux_df = pd.read_csv('runs/04_ranknet+scalar_500_lstm/stations/29/input/aux.csv')
aux_df[time_col] = pd.to_datetime(aux_df[time_col], utc=True)

print(aux_df.iloc[0][time_col])

ds = FlowPhotoDataset(images_df, images_dir, aux_data = aux_df, aux_model = 'lstm', aux_lstm_timestep='D', aux_sequence_length = 4)

image, aux, label = ds[0]
print(images_df.iloc[0][time_col])
print(image.shape)
print(aux)
print(label)

print('PAIRS')

ds = FlowPhotoRankingPairsDataset(pairs_df, images_dir, aux_data = aux_df, aux_model = 'lstm', aux_lstm_timestep='D', aux_sequence_length = 4)

image1, image2, aux1, aux2, label = ds[0]
print(image1.shape)
print(pairs_df.iloc[0][time_col + '_1'])
print(aux1)
print(image2.shape)
print(pairs_df.iloc[0][time_col + '_2'])
print(aux2)
print(label)

# hourly
time_col = 'timestamp'

pairs_df = load_pairs_from_csv('runs/04_ranknet+scalar_500_lstm_H/stations/29/input/pairs.csv', timestep='H')
images_df = pd.read_csv('runs/04_ranknet+scalar_500_lstm_H/stations/29/input/images.csv')
aux_df = pd.read_csv('runs/04_ranknet+scalar_500_lstm_H/stations/29/input/aux.csv')
aux_df[time_col] = pd.to_datetime(aux_df[time_col], utc=True)

ds = FlowPhotoDataset(images_df, images_dir, aux_data = aux_df, aux_model = 'lstm', aux_lstm_timestep='H', aux_sequence_length = 4)

image, aux, label = ds[0]
print(images_df.iloc[0][time_col])
print(image.shape)
print(aux)
print(label)

ds = FlowPhotoRankingPairsDataset(pairs_df, images_dir, aux_data = aux_df, aux_model = 'lstm', aux_lstm_timestep='H', aux_sequence_length = 4)

image1, image2, aux1, aux2, label = ds[0]
print(image1.shape)
print(pairs_df.iloc[0][time_col + '_1'])
print(aux1)
print(image2.shape)
print(pairs_df.iloc[0][time_col + '_2'])
print(aux2)
print(label)
