import numpy as np
import tensorflow as tf
from tqdm import tqdm

class DataPostProcessing:
  def __init__(self):
    print("DataPostProcessing: This class is for DataPostProcessing Parts, mostly to process data from Models Output")

  @staticmethod
  def PostProcessBatch(raw_df, prediction_batch, one_hot_mappings):
    '''
    DataPostProcessing.PostProcessBatch() is for Post Processing Multiple data from the output of a model and append it to RAW Data Frame
    raw_df: A Dataframe. Same dataframe in DataPreProcessing.PreProcessBatchWithoutLabels()
    prediciton_batch: Numpy array from tf.keras.Model.predict
    '''
    if prediction_batch.shape[0] == 1:
      print("DataPostProcessing[PostProcessBatch]: prediction_batch only containing an output of 1 labels, consider using DataPostProcessing.PostProcessSingle()".format(one_hot_mappings))

    print("DataPostProcessing[PostProcessBatch]: Post Processing Batch prediction with Labels {}".format(one_hot_mappings))
    list_prediction = []
    for i in tqdm(range(prediction_batch.shape[0])):
      list_prediction.append(one_hot_mappings[np.argmax(prediction_batch[i])])

    raw_df['Prediction'] = list_prediction
    return raw_df

  @staticmethod
  def PostProcessSingle(prediction_output, one_hot_mappings):
    '''
    DataPostProcessing.PostProcessSingle() is for Post Processing Single data from the output of a model
    prediction_output: Numpy array from tf.keras.Model.predict with shape (1,)
    one_hot_mapping: Dictionary containing mappings of the labels and the One Hot Encodings
    '''
    if prediction_output.shape[0] != 1:
      raise ValueError("DataPostProcessing[PostProcessSingle]: prediction output has a shape of {} with the argument need a shape of (1,)".format(prediction_output.shape))

    print("DataPostProcessing[PostProcessSingle]: Post Processing Single prediction with Labels {}".format(one_hot_mappings))
    prediction = one_hot_mappings[np.argmax(prediction_output[0])]

    print("DataPostProcessing[PostProcessSingle]: The prediction is labeled {}. The labeled is returned, so you can assign this to a variable".format(prediction))
    return prediction

  @staticmethod
  def PostProcessBatchToList(prediction_batch, one_hot_mappings):
    '''
    DataPostProcessing.PostProcessBatch() is for Post Processing Multiple data from the output of a model and return a List of Mappings One Hot Encoding of Prediction
    prediciton_batch: Numpy array from tf.keras.Model.predict
    '''
    if prediction_batch.shape[0] == 1:
      print("DataPostProcessing[PostProcessBatch]: prediction_batch only containing an output of 1 labels, consider using DataPostProcessing.PostProcessSingle()".format(one_hot_mappings))

    print("DataPostProcessing[PostProcessBatch]: Post Processing Batch prediction with Labels {}".format(one_hot_mappings))
    list_prediction = []
    for i in tqdm(range(prediction_batch.shape[0])):
      list_prediction.append(one_hot_mappings[np.argmax(prediction_batch[i])])

    return list_prediction