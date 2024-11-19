from einops.layers.tensorflow import Rearrange
import tensorflow as tf
import numpy as np
import math

relative_joint_map=tf.constant([
  [0, 3, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
  [3, 0, 2, 3, 1, 1, 3, 3, 3, 3, 2, 3, 2, 2, 2, 3, 3, 3],
  [2, 2, 0, 1, 1, 3, 3, 3, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3],
  [1, 3, 1, 0, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3],
  [3, 1, 1, 2, 0, 2, 3, 2, 2, 3, 1, 3, 3, 3, 3, 3, 3, 3],
  [3, 1, 3, 3, 2, 0, 3, 3, 3, 3, 3, 3, 1, 1, 1, 3, 3, 3],
  [3, 3, 3, 3, 3, 3, 0, 3, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3],
  [3, 3, 3, 3, 2, 3, 3, 0, 2, 3, 1, 1, 3, 3, 3, 2, 2, 2],
  [3, 3, 1, 2, 2, 3, 2, 2, 0, 1, 1, 3, 3, 3, 3, 3, 3, 3],
  [3, 3, 2, 3, 3, 3, 1, 3, 1, 0, 2, 3, 3, 3, 3, 3, 3, 3],
  [3, 2, 2, 3, 1, 3, 3, 1, 1, 2, 0, 2, 3, 3, 3, 3, 3, 3],
  [3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 2, 0, 3, 3, 3, 1, 1, 1],
  [3, 2, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 0, 1, 2, 3, 3, 3],
  [3, 2, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 0, 2, 3, 3, 3],
  [3, 2, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 2, 2, 0, 3, 3, 3],
  [3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 1, 3, 3, 3, 0, 1, 2],
  [3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 1, 3, 3, 3, 1, 0, 2],
  [3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 1, 3, 3, 3, 2, 2, 0]
])

def create_initializer(initializer_range=0.02):
  return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

def gelu(x):
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


class MLP(tf.keras.layers.Layer):
  """Feedforward layer."""

  def __init__(self, out_dim, hidden_dim, dropout=0., **kwargs):
    super(MLP, self).__init__(**kwargs)
    self.layer1=tf.keras.layers.Dense(
            hidden_dim, activation=gelu)
    self.dropout_ratio = dropout
    if self.dropout_ratio > 0:
      self.dropout = tf.keras.layers.Dropout(self.dropout_ratio)
    self.layer2=tf.keras.layers.Dense(out_dim)

  def call(self, x, training=None):
    x = self.layer1(x)
    if self.dropout_ratio > 0:
      x = self.dropout(x, training=training)
    out = self.layer2(x)
    return out
  
class CFFN(tf.keras.layers.Layer):
  """
  1D convolution layer. Inspired by 'Exploiting Temporal Contexts With Strided Transformer for 3D Human Pose Estimation' and 'Uplift and Upsample_ Efficient 3D Human Pose Estimation With Uplifting Transformers' papers
  """
  def __init__(self, 
               out_dim, 
               hidden_dim,
               dropout=0.,
               kernel_size=3,
               stride=3, 
               **kwargs):
    super(CFFN, self).__init__(**kwargs)
    self.dim = out_dim
    self.hidden_dim = hidden_dim
    self.dropout_ratio = dropout
    if self.dropout_ratio > 0:
      self.dropout = tf.keras.layers.Dropout(self.dropout_ratio)

    self.cnn1d1 = tf.keras.layers.Conv1D(filters=hidden_dim, 
                                        kernel_size=1, 
                                        strides=1, 
                                        activation='relu',
                                        padding='valid')
    self.cnn1d2 = tf.keras.layers.Conv1D(filters=out_dim,
                                         kernel_size=kernel_size,
                                         strides=stride,
                                         activation='relu',
                                         padding='valid')
  def call(self, x, training):
    x = self.cnn1d1(x)
    if self.dropout_ratio > 0:
      x = self.dropout(x)
    x = self.cnn1d2(x)

    return x


class LinearEmbedding(tf.keras.layers.Layer):
  """Linear projection."""

  def __init__(self, dim, act=None, **kwargs):
    super(LinearEmbedding, self).__init__(**kwargs)
    self.net = tf.keras.layers.Dense(dim, activation=act, name=f"{self.name}")

  def call(self, x):
    return self.net(x)

class PositionEmbedding(tf.keras.layers.Layer):
  """Position Embedding layer."""

  def __init__(self, seq_length, dim, **kwargs):
    super(PositionEmbedding, self).__init__(**kwargs)

    pos_initializer = create_initializer(0.02)
    self.pos_embedding = self.add_weight(
        name=f"{self.name}",
        shape=[seq_length, dim],
        initializer=pos_initializer,
        dtype=tf.float32)

  def call(self, x):
    """Call embedding layer."""
    return x + self.pos_embedding

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, dim, num_heads=8, bias=False, dropout=0., relative=False, **kwargs):
    super(MultiHeadAttention, self).__init__(**kwargs)
    # dim = num_heads * depth
    self.dim = dim
    self.heads=num_heads
    self.depth = dim // num_heads
    self.scale = self.depth ** -0.5
    self.dropout_ratio = dropout

    self.wq = tf.keras.layers.Dense(dim, use_bias=bias)
    self.wv = tf.keras.layers.Dense(dim, use_bias=bias)
    self.wk = tf.keras.layers.Dense(dim, use_bias=bias)

    self.relative = relative
    if relative:
      pos_initializer = create_initializer(0.02)
      self.relative_pos_k = self.add_weight(
          name="relative_K"+f"{self.name}",
          shape=[4, int(dim / num_heads)],
          initializer=pos_initializer,
          dtype=tf.float32)
      self.relative_pos_v = self.add_weight(
          name="relative_V"+f"{self.name}",
          shape=[4, int(dim / num_heads)],
          initializer=pos_initializer,
          dtype=tf.float32)

    self.to_out = tf.keras.layers.Dense(dim)
    if self.dropout_ratio > 0:
      self.dropout = tf.keras.layers.Dropout(dropout)

  def retrieve_relative_pos(self):
    r_pos_k = tf.gather(self.relative_pos_k, relative_joint_map, axis=0)
    r_pos_v = tf.gather(self.relative_pos_v, relative_joint_map, axis=0)
    return r_pos_k, r_pos_v

  def call(self, k, v, q, training=None):
    q = self.wq(q)  # (batch, num, dim)
    k = self.wk(k)  # (batch, num, dim)
    v = self.wv(v)  # (batch, num, dim)

    # Split last column of data (dim) to heads and depth 
    # and reshaping to (batch, heads, num, depth)
    q = Rearrange("b n (h d)->b h n d", h=self.heads, d=self.depth)(q)
    k = Rearrange("b n (h d)->b h n d", h=self.heads, d=self.depth)(k)
    v = Rearrange("b n (h d)->b h n d", h=self.heads, d=self.depth)(v)

    # Compute Similarity Matrix
    if self.relative:
      r_pos_k, r_pos_v = self.retrieve_relative_pos()
      dots = (tf.einsum("bhid,bhjd->bhij", q, k) + tf.tile(tf.expand_dims(tf.einsum("bhid, ijd->bij", k, r_pos_k), axis=1), [1, self.heads, 1, 1]))*self.scale
      att_weight = tf.nn.softmax(dots, axis=-1)

      out = tf.einsum("bhij,bhjd->bhid", att_weight, v) + tf.einsum("bhij, ijd->bhid", att_weight, r_pos_v)
    else:
      dots = tf.einsum("bhid,bhjd->bhij", q, k) * self.scale
      att_weight = tf.nn.softmax(dots, axis=-1)

      out = tf.einsum("bhij,bhjd->bhid", att_weight, v)
    out = Rearrange("b h n d->b n (h d)")(out)
    out = self.to_out(out)
    if self.dropout_ratio > 0:
      out = self.dropout(out, training=training)

    return out, att_weight

class MultiHeadAttention4D(tf.keras.layers.Layer):
  def __init__(self, dim, num_heads=8, bias=False, dropout=0., **kwargs):
    super(MultiHeadAttention4D, self).__init__(**kwargs)
    # dim = num_heads * depth
    self.dim = dim
    self.heads=num_heads
    self.depth = dim // num_heads
    self.scale = self.depth ** -0.5
    self.dropout_ratio = dropout

    self.wq = tf.keras.layers.Dense(dim, use_bias=bias)
    self.wv = tf.keras.layers.Dense(dim, use_bias=bias)
    self.wk = tf.keras.layers.Dense(dim, use_bias=bias)

    self.to_out = tf.keras.layers.Dense(dim)
    if self.dropout_ratio > 0:
      self.dropout = tf.keras.layers.Dropout(dropout)
  
  def call(self, k, v, q, training=None):
    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)

    q = Rearrange("b f n (h d)->b f h n d", h=self.heads, d=self.depth)(q)
    k = Rearrange("b f n (h d)->b f h n d", h=self.heads, d=self.depth)(k)
    v = Rearrange("b f n (h d)->b f h n d", h=self.heads, d=self.depth)(v)

    dots = tf.einsum("bfhid,bfhjd->bfhij", q, k) * self.scale
    att_weight = tf.nn.softmax(dots, axis=-1)

    out = tf.einsum("bfhij,bfhjd->bfhid", att_weight, v)
    out = Rearrange("b f h n d->b f n (h d)")(out)
    out = self.to_out(out)
    if self.dropout_ratio > 0:
      out = self.dropout(out, training=training)

    return out, att_weight


class Transformer4d(tf.keras.layers.Layer):
  def __init__(self, 
               dim=768,
               hidden_dim=3072,
               dropout=0.,
               num_heads=8, 
               att_bias=False,
               att_dropout=0.,
               **kwargs):
    super(Transformer4d, self).__init__(**kwargs)
    self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    self.multi_att = MultiHeadAttention4D(dim=dim, num_heads=num_heads, bias=att_bias, dropout=att_dropout)
    
    self.MLP = MLP(out_dim=dim, hidden_dim=hidden_dim, dropout=dropout)
    self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

  def call(self, q, k, v, training=None):
    q = self.norm1(q)
    k = self.norm1(k)
    v = self.norm1(v)

    y, att_weight = self.multi_att(k, v, q, training=training)

    y = y + q

    out = self.MLP(self.norm2(y), training=training)
    out = out + y
    
    return out, att_weight
  
  
class Transformer(tf.keras.layers.Layer):
  def __init__(self, 
               dim=768,
               hidden_dim=3072,
               dropout=0.,
               num_heads=8, 
               att_bias=False,
               att_dropout=0.,
               relative=False,
               **kwargs):
    super(Transformer, self).__init__(**kwargs)
    # self.position_embedding = PositionEmbedding(dim)
    self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    self.multi_att = MultiHeadAttention(dim=dim, num_heads=num_heads, bias=att_bias, dropout=att_dropout, relative=relative)
    
    self.MLP = MLP(out_dim=dim, hidden_dim=hidden_dim, dropout=dropout)
    self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

  def call(self, x, training=None):
    y = self.norm1(x)
    y, att_weight = self.multi_att(y, y, y, training=training)

    y = y + x

    out = self.MLP(self.norm2(y), training=training)
    out = out + y
    
    return out, att_weight
  

class CrossTransformer(tf.keras.layers.Layer):
  def __init__(self,
               dim=768,
               hidden_dim=3072,
               dropout=0.,
               num_heads=8,
               att_bias=False,
               att_dropout=0.,
               **kwargs):
    super(CrossTransformer, self).__init__(**kwargs)
    self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    self.cross_att = MultiHeadAttention(dim=dim, num_heads=num_heads, bias=att_bias, dropout=att_dropout)

    self.MLP = MLP(out_dim=dim, hidden_dim=hidden_dim, dropout=dropout)
    self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

  def call(self, x1, x2, training=None):
    y1 = self.norm1(x1) # query
    y2 = self.norm2(x2) # key and value
    y, att_weight = self.cross_att(y2, y2, y1, training=training)

    y = y + x1

    out = self.MLP(self.norm3(y), training=training)
    out = out + y
    
    return out, att_weight



class StridedTransformer(tf.keras.layers.Layer):
  def __init__(self, 
               dim=768,
               hidden_dim=3072,
               dropout=0.,
               num_heads=8, 
               att_bias=False,
               att_dropout=0.,
               stride=3,
               kernel_size=3,
               **kwargs):
    super(StridedTransformer, self).__init__(**kwargs)
    self.linear_embedding = LinearEmbedding(dim)
    self.multi_att = MultiHeadAttention(dim=dim, num_heads=num_heads, bias=att_bias, dropout=att_dropout)
    self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    self.maxpool = tf.keras.layers.MaxPool1D(pool_size=stride, strides=stride)
    self.cffn = CFFN(out_dim=dim, hidden_dim=hidden_dim, dropout=dropout, kernel_size=kernel_size, stride=stride)

  def call(self, x, training=None):
    y = self.norm1(x)
    y, att_weight = self.multi_att(y, y, y, training=training)
    y = y + x

    z = self.maxpool(y)
    y = self.norm2(y)

    out = self.cffn(y) + z

    return out, att_weight
  
class CNNResidual(tf.keras.layers.Layer):
  def __init__(self, filter, kernel_size, types='2D',**kwargs):
    super(CNNResidual, self).__init__(**kwargs)
    l2 = tf.keras.regularizers.l2(0.0001)
    initializer = tf.keras.initializers.HeNormal()
    if types=='2D':
      self.conv1 = tf.keras.layers.Conv2D(filters=filter,
                                          kernel_size=kernel_size,
                                          strides=1,
                                          padding='same',
                                          activation=tf.keras.layers.LeakyReLU(),
                                          kernel_initializer=initializer,
                                          kernel_regularizer=l2)
    elif types=='1D':
      self.conv1 = tf.keras.layers.Conv1D(filters=filter,
                                          kernel_size=kernel_size,
                                          strides=1,
                                          padding='same',
                                          activation=tf.keras.layers.LeakyReLU(),
                                          kernel_initializer=initializer,
                                          kernel_regularizer=l2)
    self.bn = tf.keras.layers.BatchNormalization()

  def call(self, x, training=False):
    output = self.conv1(x)
    output = self.bn(output, training=training)
    
    return output + x

def Upsampling1D(data, out_frame, interpolation='bilinear'):
  """
  Upsampling with 'rate' ratio by bilinear method
    Args: 
      data: (batch, frame, dim)
      out_frame: the number of upsampled frame
    Returns:
      Upsampled 3D tensor: (batch, out_frame, dim)
  """
  b, f, d = data.shape
  truncate = False

  if out_frame % f > 0:
    ratio = math.ceil(out_frame / f)
    truncate = True
  else:
    ratio = int(out_frame / f)

  ext_data = data[tf.newaxis, :, :, :]

  up_data = tf.keras.layers.UpSampling2D(size=(ratio, 1), data_format="channels_first", interpolation=interpolation)(ext_data)

  if truncate:
    return up_data[0, :, 0:out_frame, :]
  else:
    return up_data[0, :, :, :]
  
  
  