from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def squeeze(value, num_times):
  for i in xrange(num_times):
    value = array_ops.squeeze(value, 0)
  return value

def expand(value, num_times):
  for i in xrange(num_times):
    value = array_ops.expand_dims(value, 0)
  return value

def split4d(**kwargs):
  print(kwargs)
  value = kwargs['value']
  del kwargs['value']
  axis = kwargs['axis']
  del kwargs['axis']
  num_squeezes = 0
  while len(value.shape) != 4:
    value = array_ops.expand_dims(value, 0)
    axis += 1
    num_squeezes += 1

  reversePermutationNeeded = False
  reversePerm = None
  if axis != 3:
    reversePermutationNeeded = True
    """
    axis 0: [3, 2, 1, 0]
    axis 1: [2, 3, 0, 1]
    axis 2: [1, 0, 3, 2]
    axis 3: [0, 1, 2, 3]
    """
    if axis == 0:
      reversePerm = [3, 2, 1, 0]
    elif axis == 1:
      reversePerm = [2, 3, 0, 1]
    elif axis == 2:
      reversePerm = [1, 0, 3, 2]

    # Permute needed if not along channel axis
    value = array_ops.transpose(value, reversePerm)

  value = array_ops.split(value=value, axis=3, **kwargs)
  if reversePermutationNeeded:
    value = [array_ops.transpose(y, reversePerm) for y in value]
  return [squeeze(y, num_squeezes) for y in value]

#Author: A.Polino
def is_power2(num):
  num = int(num)
  return ((num & (num - 1)) == 0)

def calcSplits(num):
  num = int(num)
  count = 1
  while num > 1:
    count += 1
    num /= 2
  return count

def addMatMul(val1, val2):
  print("MATMUL###")
  print(val1.shape)
  print(val1)
  print(val2.shape)
  print(val2)
  print("###")
  return math_ops.mat_mul(val1, val2)

def splittingMatMul(value1, value2):
  # value1 must be 2d
  # value2 must be 1d
  print("splitting mat mul")
  print(value1.shape) # (2, 812)
  print(value1)
  print(value2.shape) # (812, 2048)
  print(value2)
  print("___")
  numExpands = 0
  while len(value1.shape) > 2:
    numExpands += 1
    value1 = array_ops.squeeze(value1, 0)

  numSplits = int(value1.shape[0])
  if numSplits == 1:
    return math_ops.mat_mul(value1, value2)
  assert is_power2(numSplits), 'beam size must be power of 2'
  numSplits = calcSplits(numSplits)
  print("NUMSPLITS", numSplits)
  splits = split4d(value=value1, num_or_size_splits=numSplits, axis=0)
  print(len(splits))
  print(splits[0].shape)
  muls = [expand(addMatMul(split, value2), numExpands) for split in splits]
  return array_ops.concat(muls, 0)