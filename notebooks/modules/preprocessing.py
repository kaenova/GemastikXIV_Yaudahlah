# reference: https://betterprogramming.pub/build-a-natural-language-classifier-with-bert-and-tensorflow-4770d4442d41

import pandas as pd
from transformers import BertTokenizer
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split  
import random  

class DataPreProcessing:

  def __init__(self):
    print("DataPreProcessing: This class is for DataPreProcessing Parts, mostly to get Trainable Data")

  @classmethod
  def PreProcessBatchBERT(cls, dataFrame, bert_model_name:str, max_length:int = 128, split_ratio:float = 0.8):
    '''
    I.S.: Tersedianya dataframe dengan kolom 'tweet' dan 'labels' serta beberapa parameter pendukung
    F.S.: Tersiapkannya sebuah data training untuk suatu model dengan input BERT [input_ids] dan [labels]
    Output: x_train, x_test, y_train, y_test, one_hot_mappings
    '''
    if len(dataFrame) < 3:
      raise ValueError("Number of rows in dataframe is too small")

    df = dataFrame
    
    # One Hot Encodings 'labels'
    labels, one_hot_mappings = cls.__OneHotEncodingLabels__(df)

    # Tokenizing 'tweets'
    input = cls.__Tokenize__(df, max_length, bert_model_name)

    print("DataPreProcessing[PreProcessBatch]: labels shape {}, input shape {}".format(labels.shape, input.shape))
    print("DataPreProcessing[PreProcessBatch]: Creating train and testing set with split ratio {}".format(split_ratio))

    x_train,x_test,y_train,y_test = train_test_split(input,labels,test_size=split_ratio, random_state=50)

    print("DataPreProcessing[PreProcessBatch]: Train and Test data Created with Train data shape {}, use out_mappings for argmax prediction".format(x_train.shape))

    return x_train,x_test,y_train,y_test, one_hot_mappings

  @classmethod
  def PreProcessBatchLSTM(cls, dataFrame, max_length:int = 128, split_ratio:float = 0.8):
    '''
    I.S.: Tersedianya dataframe dengan kolom 'tweet' dan 'labels' serta beberapa parameter pendukung
    F.S.: Tersiapkannya sebuah data training untuk suatu model dengan input LSTM
    Output: x_train, x_test, y_train, y_test, one_hot_mappings
    '''
    if len(dataFrame) < 3:
      raise ValueError("Number of rows in dataframe is too small")

    df = dataFrame
    
    # One Hot Encodings 'labels'
    labels, one_hot_mappings = cls.__OneHotEncodingLabels__(df)

    # Tokenizing 'tweets'
    input = np.array(df['tweet'].to_list())

    print("DataPreProcessing[PreProcessBatch]: labels shape {}, input shape {}".format(labels.shape, input.shape))
    print("DataPreProcessing[PreProcessBatch]: Creating train and testing set with split ratio {}".format(split_ratio))

    x_train,x_test,y_train,y_test = train_test_split(input,labels,test_size=split_ratio, random_state=50)

    print("DataPreProcessing[PreProcessBatch]: Train and Test data Created with Train data shape {}, use out_mappings for argmax prediction".format(x_train.shape))

    return x_train,x_test,y_train,y_test, one_hot_mappings

  @classmethod
  def PreProcessSingleWithoutLabelsBERT(cls, text: str, bert_model_name:str, max_length:int = 128):
    '''
    I.S.: Tersedianya suatu text dan beberapa parameter pendukung
    F.S.: Tersiapkannya sebuah data input untuk model BERT [input_ids, attention_mask]
    Output: X
    '''

    data = [text]
    df = pd.DataFrame(data, columns=['tweet'])

    tokenize_text = cls.__Tokenize__(df, max_length, bert_model_name)

    return tokenize_text

  @classmethod
  def PreProcessSingleWithoutLabelsLSTM(cls, text: str):
    '''
    I.S.: Tersedianya suatu text dan beberapa parameter pendukung
    F.S.: Tersiapkannya sebuah data input untuk model LSTM [input_ids, attention_mask]
    Output: X
    '''
    return np.array([text])

  @classmethod
  def PreProcessBatchWithoutLabelsBERT(cls, df, bert_model_name:str, max_length:int = 128):
    '''
    I.S.: Tersedianya suatu dataframe dan beberapa parameter pendukung
    F.S.: Tersiapkannya sebuah data input untuk model BERT [input_ids, attention_mask]
    Output: X
    '''

    input = cls.__Tokenize__(df, max_length, bert_model_name)

    return input

  @classmethod
  def PreProcessBatchWithoutLabelsLSTM(cls, df):
    '''
    I.S.: Tersedianya suatu dataframe dan beberapa parameter pendukung
    F.S.: Tersiapkannya sebuah data input untuk model LSTM
    Output: X
    '''

    return np.array(df['tweet'].to_list())
  
  @classmethod
  def PreProcessBatchValidationBERT(cls, dataFrame, bert_model_name:str, max_length:int = 128):
    '''
    I.S.: Tersedianya suatu dataframe dan beberapa parameter pendukung
    F.S.: Tersiapkannya sebuah data input untuk model BERT [input_ids, attention_mask] untuk validasi
    Output: X, Y, One_Hot_Mapping(dictionary)
    '''
    
    df = dataFrame.copy()


    labels, one_hot_mappigns = cls.__OneHotEncodingLabels__(df)
    input = cls.__Tokenize__(df, max_length, bert_model_name)
    print("DataPreProcessing[PreProcessBatchValidation]: labels shape {}, input shape {}".format(labels.shape, input.shape))
    
    return input, labels, one_hot_mappigns

  @classmethod
  def PreProcessBatchValidationLSTM(cls, dataFrame):
    '''
    I.S.: Tersedianya suatu dataframe dan beberapa parameter pendukung
    F.S.: Tersiapkannya sebuah data input untuk model LSTM untuk validasi
    Output: X, Y, One_Hot_Mapping(dictionary)
    '''
    
    df = dataFrame.copy()

    labels, one_hot_mappigns = cls.__OneHotEncodingLabels__(df)
    input = np.array(df['tweet'].to_list())
    print("DataPreProcessing[PreProcessBatchValidation]: labels shape {}, input shape {}".format(labels.shape, input.shape))
    
    return input, labels, one_hot_mappigns
    
  
  def __Tokenize__(dataFrame, max_length: int, bert_model_name: str):
    '''
    I.S: Dataframe with column 'tweet'
    F.S: Numpy array of input_ids from 'tweets' label
    '''

    print("DataPreProcessing[PreProcessBatch|Tokenize]: Fetching Tokenizer")
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def tokenize(sentence):
      tokens = tokenizer(sentence, max_length=max_length,
                                    truncation=True, padding='max_length',
                                    return_attention_mask = True,
                                    return_token_type_ids=False, return_tensors='tf')
      return tokens['input_ids'], tokens['attention_mask']

    Xids = np.zeros((len(dataFrame), max_length))
    Xatt = np.zeros((len(dataFrame), max_length))

    for i, sentence in enumerate(dataFrame['tweet']):
      sentence = str(sentence)
      Xids[i, :], Xatt[i, :] = tokenize(sentence)

    zipped = np.stack((Xids, Xatt), axis=1)

    return zipped

  def __OneHotEncodingLabels__(dataFrame):
    # I.S: Dataframe with column 'labels'
    # F.S: One Hot Encodings to an Arrays of column 'labels'

    # Check the labels is number and if there's a number below 0 (i.e. -1, -2, -3). This need to be all positive
    min_bool_int_normal = False
    if dataFrame['labels'].dtype == 'int':
      minimum_temp = min(dataFrame['labels'])
      if minimum_temp < 0:
        min_bool_int_normal = True
        df_copies = dataFrame.copy()
        print("DataPreProcessing[PreProcessBatch|OneHotEncodingLabels]: Below zero detected in labels data")
        df_new = dataFrame['labels'].apply(lambda x: x+minimum_temp)
        dataFrame['labels'] = df_new 

    # Need to check if there's integer labels
    if str(dataFrame['labels'].dtype) != 'category':
      print("DataPreProcessing[PreProcessBatch|OneHotEncodingLabels]: Non-Category datatype detected, converting to Category datatype")
      dataFrame['labels'] = dataFrame['labels'].astype('category')

    if min_bool_int_normal:
      df_copies['labels'] = df_copies['labels'].astype('category')
      one_hot_mappings = dict( enumerate(df_copies['labels'].cat.categories ))
    else:
      one_hot_mappings = dict( enumerate(dataFrame['labels'].cat.categories ))

    dataFrame['labels'] = dataFrame["labels"].cat.codes

    # One Hot Encodings
    arr = dataFrame['labels'].values 
    labels = np.zeros((arr.size, arr.max()+1))
    labels[np.arange(arr.size), arr] = 1
    return labels, one_hot_mappings