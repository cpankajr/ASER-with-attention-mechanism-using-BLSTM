# ASER-with-attention-mechanism-using-BLSTM

 audio -> log mel spectrogram (like RGB image, it will have 3 layers , normal spectrogram, its delta nd double delta respectively)-> 2D CNN layers -> BLSTM layers-> Attention layer -> Fully Connected Layer -> Softmax Layer -> Emotion Probability Distributions

**Note: This model does not perform well enough as of now, I am still working on it**

# Libraries used

- Keras
- Tensorflow (not necessarily required)
- librosa
- python_speech_features 
- Numpy

#
For understanding attention model in speech emotion recognition you can see

[Automatic Speech Emotion Recognition Using Recurrent Neural Networks with Local Attention](https://www.youtube.com/watch?v=NItzgTQ9lvw)


[**_emodb.pkl_**](https://drive.google.com/open?id=1DmmMtHPZUcA16tYGWjFId0wgnxgj2cvh)

[**_emodb_test.pkl_**](https://drive.google.com/open?id=1XHea79-2uBFSkl5-wEYpqTE_N8ZoiaxB)
