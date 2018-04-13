# thalbes
A collection and comparison of sentiment analysis models. Done with Python and Tensorflow.
The models vary in the data format used - Bag of Words(BOW) or GloVe(vector representations of words) - and the 
type of neural networks (dense, CNN, RNN). 
These are very script-like programs, so not intended to be reused "as is" but this is a good start if you're looking for ideas.
Possible areas of improvement:
+ The data pipeline sucks; it would be preferable to use dask and/or tf.data for that
+ The models and their training could be wrapped in classes
+ The trainig data should use a more agnostic format like parquet or hdf5
+ The specific dataset used is too shallow to truly demonstrate the power of vector representation models; which made the plain model look too good
