Using the original RankNet architecture of the current FPE model as a starting point, we developed three alternative model structures designed to integrate historical weather data with the timelapse imagery as a combined set of model inputs.

The original RankNet model is comprised of two elements: 1) a pre-trained ResNet convolution neural network (CNN) model that encodes an image into a set of features, and 2) a series of fully-connected neural network layers that convert those features into a single scalar value, which is the image score that is an estimate measure of relative streamflow.

To integrate the weather data, we added a third element referred to as the auxiliary data encoder. Similar to the ResNet model that encodes an image into a set of features, the auxiliary data encoder encodes a set of auxiliary data (in this case, weather data) into a set of features. The output features from the auxiliary encoder are then merged with the output features from the ResNet model into a combined set of image + auxiliary data features. The full set of image + auxiliary data features are then passed as inputs to the fully-connected neural network layers, which convert those features into the final score.

Using this 3-part architecture (ResNet image encoder, auxiliary data encoder, fully-connected layers), we evaluated three alternatives for the auxiliary data encoder:

1.	Direct: the auxiliary data are not encoded into a set of output features, but are simply merged (concatenated) directly with the output features from the ResNet model.
2.	MLP: the auxiliary data are encoded using a Multi-Layer Perceptron (a series of fully-connected neural network layers) to create a set of output features that are combined with the ResNet output features.
3.	LSTM: the auxiliary data are encoded using a long-short term memory (LSTM) neural network, which also creates a set of features that are combined with the ResNet output features.

These three alternatives represent a gradient from the simplest (direct) to the most sophisticated (LSTM) approaches for integrating auxiliary timeseries data with the image-based ResNet model. The direct approach was designed to show whether the weather data itself, including both the values of each hourly variable as well as the series of rolling statistics, contained sufficient information to improve the model performance. The MLP is the next step whereby a standard artificial neural network (ANN) is used to convert those same set of values into a set of features, which are numeric values that represent the underlying patterns in the data that the model finds most relevant for estimating the flow of each image. Lastly, the LSTM model is the most sophisticated approach of the three, and uses a Long Short-Term Memory (LSTM) neural network to process sequences of historical weather data. The LSTM model is particularly well-suited for this task because:

1. It can learn temporal dependencies in the weather data by processing sequences of observations over time, rather than just looking at individual timesteps in isolation
2. It maintains an internal memory state that allows it to capture both short-term and long-term patterns in the weather data
3. It can handle variable-length input sequences and deal with missing data through zero-padding

The LSTM auxiliary encoder takes as input a sequence of weather observations (e.g., the previous 30 days or hours of data) for each image. Each sequence element contains the full set of weather variables for that timestep. The LSTM processes this sequence and outputs a fixed-length feature vector that represents the relevant temporal patterns in the weather data leading up to when the image was taken. These LSTM-encoded features are then concatenated with the ResNet image features before being passed to the fully-connected layers.

The LSTM architecture in this implementation includes:
- Configurable sequence length (default 30 timesteps)
- Multiple LSTM layers (default 2) for hierarchical feature extraction
- Adjustable hidden state size (default 64 dimensions) 
- Optional dropout between LSTM layers for regularization
- Support for both daily and hourly timesteps

This approach allows the model to learn how patterns in the weather data over time might influence or relate to the streamflow conditions visible in the images. For example, the LSTM could potentially learn patterns like how sequences of rainfall events or temperature trends affect streamflow levels.

While both the MLP and LSTM models transform weather data into learned feature representations, they differ fundamentally in how they handle temporal information. The MLP model treats each timestep's data independently, processing the raw values and derived statistics (like rolling means and maximums) through fully-connected layers to create a feature embedding. This approach can capture the general magnitude and statistical properties of the weather conditions, but may miss important sequential patterns. In contrast, the LSTM model explicitly processes the weather data as a sequence, maintaining an internal state that can track how conditions evolve over time. This allows it to potentially learn more sophisticated temporal patterns, such as the cumulative effect of multiple rain events or the interaction between temperature and precipitation over time. The tradeoff is that the LSTM model is more complex and may require more data to train effectively compared to the simpler MLP approach.
