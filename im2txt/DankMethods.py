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

def expand_until(value, shape_len):
  while len(value.shape) < shape_len:
    value = array_ops.expand_dims(value, 0)
  return value

def squeeze_until(value, shape_len):
  while len(value.shape) > shape_len:
    value = array_ops.squeeze(value, 0)
  return value

def split4d_old(**kwargs):
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

  value = array_ops.split(value=value, axis=axis, **kwargs)
  return [squeeze(y, num_squeezes) for y in value]

def make_axis_array(axis, shape):
  shape2 = [int(i) for i in shape]
  shape2[axis] = 1
  return shape2

def split4d(**kwargs):
  value = kwargs['value']
  axis = kwargs['axis']
  numSplits = kwargs['num_or_size_splits']

  if axis == len(value.shape) - 1:
    # Call the old method if the split is along the ultimate axis
    return split4d_old(**kwargs)
  
  # Expand to 4d
  num_squeezes = 0
  while len(value.shape) != 4:
    value = array_ops.expand_dims(value, 0)
    axis += 1
    num_squeezes += 1

  # We need to use numpy style array slicing manually to call strided slice.
  assert value.shape[axis] == numSplits, 'shape must equal axis'

  if axis == 0:
    return [value[i,:,:,:] for i in xrange(value.shape[axis])]
  elif axis == 1:
    return [value[:,i,:,:] for i in xrange(value.shape[axis])]
  elif axis == 2:
    return [value[:,:,i,:] for i in xrange(value.shape[axis])]
  else:
    raise Exception("Shouldn't happen")

  # return array_ops.strided_slice(value, [0 for _ in value.shape], [-1 for _ in value.shape], make_axis_array(axis, value.shape))

def split4d_old2(**kwargs):
  return split4d_old(**kwargs)
  print("SPLIT4D")
  print(kwargs)

  value = kwargs['value']
  axis = kwargs['axis']
  numSplits = kwargs['num_or_size_splits']

  if axis != len(value.shape) - 1:
    # Split by subscript - requires special case
    assert value.shape[axis] == numSplits, 'shape must equal axis'
    for i in xrange(axis):
      assert value.shape[i] == 1, 'must be filled with 1 shape until axis'
    for i in xrange(axis):
      value = value[0]
    return [expand(value[i], axis + 1) for i in xrange(value.shape[0])]

  # Otherwise, return regular split4d

  #   return split4d_old(value=value, num_or_size_splits=numSplits, axis=3)
  # for i in xrange(1, axis):
  #   assert(value.shape[i] == 0)
  
  # for i in xrange(1, axis):
  #   value = value[0]
  # # [value[i] for i in xrange(numSplits)]

  return split4d_old(**kwargs)

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

def concat(values, axis):
  val1 = values[0]
  expands = 0
  while len(val1.shape) < 4:
    values = [expand(val, 1) for val in values]
    expands += 1
    axis += 1
    val1 = values[0]
  values = array_ops.concat(values, axis)
  values = squeeze(values, expands)
  return values
  

def splittingMatMul(value1, value2):
  # value1 must be 2d
  # value2 must be 1d
  print("splitting mat mul")
  print(value1.shape) # (1, 2, 512)
  print(value1)
  print(value2.shape) # (512, 38521)
  print(value2)
  print("___")
  numExpands = 0
  numInternalExpands = 0

  # Make sure both shapes are at least rank 2
  # while len(value1.shape) < 2:
  #   numInternalExpands += 1
  #   value1 = array_ops.expand(value1, 0)
  # while len(value2.shape) < 2:
  #   num
  #   value = array_ops.expand(value2, 0)

  value1 = squeeze_until(value1, 2)
  numSplits = int(value1.shape[0])
  if numSplits == 1:
    return math_ops.mat_mul(value1, value2)
  assert is_power2(numSplits), 'beam size must be power of 2'
  numSplits = calcSplits(numSplits)
  print("NUMSPLITS", numSplits)
  value1 = array_ops.transpose(value1)
  splits = split4d(value=value1, num_or_size_splits=numSplits, axis=1)
  splits = [array_ops.transpose(splits[i]) for i in xrange(0, len(splits))]
  print(len(splits))
  print(splits[0].shape)
  muls = [expand(addMatMul(squeeze_until(split, 2), value2), numExpands) for split in splits]
  print("MULS", muls)
  return concat(muls, 0)