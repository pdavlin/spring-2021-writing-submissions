#!/usr/bin/env python
# imports and declarations
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import random

import tensorflow as tf
import tensorflow.keras.datasets.cifar10 as cifar10

# validate that tensorflow works and GPU is properly configured
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def holdout(data_in,training_pct=0.7):
  """
  Split data into training and validation sets based on a given ratio
  Inputs: 
    - `data_in`, CIFAR10 data
    - `training_pct`, fraction of data used for training (default `0.7`)

  Outputs:
    - tuple of `training_data` and `validation_data` after split
  """
  num_records_in_training_set=int(len(data_in) * training_pct)
  np.random.shuffle(data_in)
  training_data = data_in[0:num_records_in_training_set]
  validation_data = data_in[num_records_in_training_set:]
  return training_data, validation_data

def select_k_training_group(k_groups, validation_set_index):
  training_set = np.array([])
  for i in range(len(k_groups)):
    if i == validation_set_index -1:
      validation_set = k_groups[i]
    else:
      if training_set.size == 0:
        training_set = k_groups[i]
      else:
        training_set = np.concatenate((training_set,k_groups[i]))
  return training_set, validation_set

def k_folds(data_in, num_groups, validation_set_index):
  """
  Split data into a specified number of groups and select one as validation data,
  using the rest as training data

  Inputs:
  `data_in` -- CIFAR10 data; 
  `num_groups` -- number of k-groups used; and
  `validation_set_index` -- k-group to use as validation set

  Outputs:
  `k_groups` -- dataset split into specified number of groups
  """
  np.random.shuffle(data_in)
  leftover_records_as_divided = len(data_in) % num_groups
  num_records_in_k_group = math.floor(len(data_in)/num_groups)
  k_groups = []
  previous_k_slice = 0
  for i in range(num_groups):
    k_slice = previous_k_slice + num_records_in_k_group
    if i < leftover_records_as_divided:
      k_slice = k_slice + 1
    k_group = data_in[previous_k_slice:k_slice]
    k_groups.append(k_group)
    previous_k_slice = k_slice

  training_set = np.array([])
  for i in range(len(k_groups)):
    if i == validation_set_index -1:
      validation_set = k_groups[i]
    else:
      if training_set.size == 0:
        training_set = k_groups[i]
      else:
        training_set = np.concatenate((training_set,k_groups[i]))
  # return k_groups
  return training_set, validation_set

def generate_random_numbers(num_randoms, sample_space):
    """
    Generate an array of random numbers.
    """
    randoms = []
    while len(randoms) < num_randoms:
        new_random = random.randint(0,sample_space)
        randoms.append(new_random)
    return randoms

def bootstrap(data_in, batch_size, num_batches):
  """
  Generate n samples of a given batch size, containing random data, with replacement

  Inputs:
  `data_in` -- CIFAR10 data; 
  `num_groups` -- number of k-groups used; and
  `validation_sel` -- which k group to use for testing data

  Outputs:
  `bootstrap_datasets` -- batched datasets of specified size
  """
  bootstrap_datasets=[None] * num_batches
  for i in range(num_batches):
      bootstrap_datasets[i] = []
      np.random.shuffle(data_in)
      random_indices = generate_random_numbers(batch_size, len(data_in))
      for j in random_indices:
          bootstrap_datasets[i].append(data_in[j])
  return bootstrap_datasets

def batch_generator(data_in, batch_size):
  """
  Yields a batch of specified size, wrapping around to start of data input
  if batch size does not divide evenly into data size

  Inputs:
  `data_in` -- CIFAR10 data; and
  `batch_size` -- desired size of the batch

  Yields:
  `batch` -- batched dataset of specified size
  """
  offset = 0
  overflow_data = [] 
  while True:
      np.random.shuffle(data_in)
      offset = 0
      if len(overflow_data) > 0:
          offset = batch_size - len(overflow_data)
          print('data wraparound occurred here; pulling {} records from front'.format(offset))
          yield np.concatenate((overflow_data, data_in[0:offset]))
          overflow_data = []

      for x in range(offset, len(data_in), batch_size):
          output_data = data_in[x : x + batch_size]
          if len(output_data) != batch_size:
            overflow_data = output_data
          else:
            yield output_data

def k_folds_generator(data_in, num_groups):
  """
  Split data into a specified number of groups and select one as validation data,
  using the rest as training data. In each "fold," yield a different group as the
  validation data

  Inputs:
  `data_in` -- CIFAR10 data; and
  `num_groups` -- number of k-groups to fold

  Yields:
  `training_data`, `validation_data` -- tuple of data for training and validation
  """
  np.random.shuffle(data_in)
  leftover_records_as_divided = len(data_in) % num_groups
  num_records_in_k_group = math.floor(len(data_in)/num_groups)
  k_groups = []
  previous_k_slice = 0
  for i in range(num_groups):
    k_slice = previous_k_slice + num_records_in_k_group
    if i < leftover_records_as_divided:
      k_slice = k_slice + 1
    k_group = data_in[previous_k_slice:k_slice]
    k_groups.append(k_group)
    previous_k_slice = k_slice

  for k in range(0,len(k_groups)):
    training_set = np.array([])
    validation_set = []
    for l in range(len(k_groups)):
      if l == k:
         validation_set = k_groups[l]
      else:
        if training_set.size == 0:
          training_set = k_groups[l]
        else:
          training_set = np.concatenate((training_set,k_groups[l]))
    yield training_set, validation_set

def plot_cifar_img(data_in):
  """
  Generic image plotting function
  """
  n = 16
  now = datetime.datetime.now()
  plt.figure(figsize=(20,4))
  for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(data_in[i].reshape(32,32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.savefig('output/{}.png'.format(now.strftime("%Y%m%d-%H%M")))

# Load CIFAR10 data
(cifar10_data, _), (_, _) = cifar10.load_data()
cifar10_data = cifar10_data / 255

# Display sample images
plot_cifar_img(cifar10_data)

# Perform test of holdout method
holdout_training_pct=0.7
holdout_training_data, holdout_validation_data = holdout(cifar10_data,holdout_training_pct)
print('size of holdout training data with {} ratio: {} records'.format(holdout_training_pct, len(holdout_training_data)))
print('size of holdout validation data with {} ratio: {} records'.format(holdout_training_pct, len(holdout_validation_data)))

# Perform test of k-groups method
num_k_groups = random.randint(3,10)
validation_index = random.randint(1, num_k_groups)
print('getting {} groups of data with k-folding and setting group {} as the validation set'.format(num_k_groups, validation_index))
k_train, k_test = k_folds(cifar10_data, num_k_groups, validation_index)
print('training set length: {}, validation set length: {}'.format(len(k_train),len(k_test)))

# Perform test of bootstrap method
bootstrap_batch_size = random.randint(100,1500)
print('sampling five batches of {} CIFAR10 images'.format(bootstrap_batch_size))
bootstrap_out = bootstrap(cifar10_data, bootstrap_batch_size, 5)
for i in range(len(bootstrap_out)):
    print('{} images in batch {}'.format(len(bootstrap_out[i]),i+1))

# Perform test of batch generator function
gen_batch_size = random.randint(10000,20000)
generator_out = batch_generator(cifar10_data, gen_batch_size)

print('getting ten batches of size {} from CIFAR10 data'.format(gen_batch_size))

for i in range(10):
  print('generator produced {} records for iteration {}'.format(len(next(generator_out)),i+1))

# Perform test of k-folds, generator version
k_fold_gen_num_groups = random.randint(2,15)
print('getting {} groups of data with iterative k-folding'.format(k_fold_gen_num_groups))
k_gen_out = k_folds_generator(cifar10_data, k_fold_gen_num_groups)
for i in range(k_fold_gen_num_groups):
  k_gen_train, k_gen_test = next(k_gen_out)
  print('fold {}: training size -- {}, validation size -- {}'.format(i+1,len(k_gen_train),len(k_gen_test)))
