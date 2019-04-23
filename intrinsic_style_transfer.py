#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/reiinakano/neural-painters/blob/master/notebooks/intrinsic_style_transfer.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## Install dependencies

# In[ ]:


# get_ipython().system('pip install ipdb tqdm cloudpickle matplotlib lucid PyDrive')


# ## Download checkpoint files for painters

# In[ ]:


# get_ipython().system('mkdir tf_vae')
# get_ipython().system("wget -O tf_vae/vae-300000.index 'https://docs.google.com/uc?export=download&id=1ulHdDxebH46m_0ZoLa2Wsz_6vStYqJQm'")
# get_ipython().system("wget -O tf_vae/vae-300000.meta 'https://docs.google.com/uc?export=download&id=1nHN_i7Ro9g0lP4y_YQCvIWrOVX1I3CJa'")
# get_ipython().system("wget -O tf_vae/vae-300000.data-00000-of-00001 'https://docs.google.com/uc?export=download&id=18rAJcUJwFJOAcjzsabtqK12udsHMZkVk'")
# get_ipython().system("wget -O tf_vae/checkpoint 'https://docs.google.com/uc?export=download&id=18U4qMNBdyvEk-Y-Mr3MNPEHSHxhcO9hn'")
#
# get_ipython().system('mkdir tf_gan3')
# get_ipython().system("wget -O tf_gan3/gan-571445.meta 'https://docs.google.com/uc?export=download&id=15kEG1Tiu2FUg5SILVt_9yOsSd3QHwVGA'")
# get_ipython().system("wget -O tf_gan3/gan-571445.index 'https://docs.google.com/uc?export=download&id=11uyFbQsRZoWa9Yq52AFXDXPjPQoGF_ER'")
# get_ipython().system("wget -O tf_gan3/gan-571445.data-00000-of-00001 'https://docs.google.com/uc?export=download&id=11cbvz-CH3KvfZEwNQ2OUujfbf6AKNoQa'")
# get_ipython().system("wget -O tf_gan3/checkpoint 'https://docs.google.com/uc?export=download&id=1A539u51t0L31Ab1M2uPUV2SsCFsNDQRo'")


# In[ ]:





# ## Imports

# In[2]:

import sys, os
import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as tcl

from IPython.display import display
import moviepy.editor as mpy
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

import lucid.modelzoo.vision_models as models
from lucid.modelzoo.vision_models import Model
from lucid.misc.io import show, load, save
import lucid.optvis.objectives as objectives
from lucid.optvis.objectives import wrap_objective
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform
from lucid.misc.redirected_relu_grad import redirected_relu_grad, redirected_relu6_grad
from lucid.misc.gradient_override import gradient_override_map

print(tf.__version__)


# In[ ]:





# ## VAE painter

# In[ ]:


class ConvVAE2(object):
  def __init__(self, reuse=False, gpu_mode=True, graph=None):
    self.z_size = 64
    self.reuse = reuse
    if not gpu_mode:
      with tf.device('/cpu:0'):
        tf.logging.info('conv_vae using cpu.')
        self._build_graph(graph)
    else:
      tf.logging.info('conv_vae using gpu.')
      self._build_graph(graph)
    self._init_session()
  
  def build_decoder(self, z, reuse=False):
    with tf.variable_scope('decoder', reuse=reuse):
      h = tf.layers.dense(z, 4*256, name="fc")
      h = tf.reshape(h, [-1, 1, 1, 4*256])
      h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu, name="deconv1")
      h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name="deconv2")
      h = tf.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu, name="deconv3")
      return tf.layers.conv2d_transpose(h, 3, 6, strides=2, activation=tf.nn.sigmoid, name="deconv4")
  
  def build_predictor(self, actions, reuse=False, is_training=False):
    with tf.variable_scope('predictor', reuse=reuse):
      h = tf.layers.dense(actions, 256, activation=tf.nn.leaky_relu, name="fc1")
      h = tf.layers.batch_normalization(h, training=is_training, name="bn1")
      h = tf.layers.dense(h, 64, activation=tf.nn.leaky_relu, name="fc2")
      h = tf.layers.batch_normalization(h, training=is_training, name="bn2")
      h = tf.layers.dense(h, 64, activation=tf.nn.leaky_relu, name="fc3")
      h = tf.layers.batch_normalization(h, training=is_training, name="bn3")
      return tf.layers.dense(h, self.z_size, name='fc4')
  
  def _build_graph(self, graph):
    if graph is None:
      self.g = tf.Graph()
    else:
      self.g = graph
    with self.g.as_default(), tf.variable_scope('conv_vae', reuse=self.reuse):
      
      #### predicting part
      self.actions = tf.placeholder(tf.float32, shape=[None, 12])
      self.predicted_z = self.build_predictor(self.actions, is_training=False)
      self.predicted_y = self.build_decoder(self.predicted_z)
      
      # initialize vars
      self.init = tf.global_variables_initializer()
  
  def generate_stroke_graph(self, actions):
    with tf.variable_scope('conv_vae', reuse=True):
      with self.g.as_default():
        # Encoder?
        z = self.build_predictor(actions, reuse=True, is_training=False)

        # Decoder
        return self.build_decoder(z, reuse=True)

  def _init_session(self):
    """Launch TensorFlow session and initialize variables"""
    self.sess = tf.Session(graph=self.g)
    self.sess.run(self.init)
  def close_sess(self):
    """ Close TensorFlow session """
    self.sess.close()


# In[ ]:





# ## GAN Painter

# In[ ]:


def relu_batch_norm(x):
    return tf.nn.relu(tf.contrib.layers.batch_norm(x, updates_collections=None))

class GeneratorConditional(object):
    def __init__(self, divisor=1, add_noise=False):
        self.x_dim = 64 * 64 * 3
        self.divisor=divisor
        self.name = 'lsun/dcgan/g_net'
        self.add_noise = add_noise

    def __call__(self, conditions, is_training):
        with tf.contrib.framework.arg_scope([tcl.batch_norm], 
                                            is_training=is_training):
          with tf.variable_scope(self.name) as vs:
              bs = tf.shape(conditions)[0]
              if self.add_noise:
                conditions = tf.concat([conditions, tf.random.uniform([bs, 10])], axis=1)
              fc = tcl.fully_connected(conditions, int(4 * 4 * 1024/self.divisor), activation_fn=tf.identity)
              conv1 = tf.reshape(fc, tf.stack([bs, 4, 4, int(1024/self.divisor)]))
              conv1 = relu_batch_norm(conv1)
              conv2 = tcl.conv2d_transpose(
                  conv1, int(512/self.divisor), [4, 4], [2, 2],
                  weights_initializer=tf.random_normal_initializer(stddev=0.02),
                  activation_fn=relu_batch_norm
              )
              conv3 = tcl.conv2d_transpose(
                  conv2, int(256/self.divisor), [4, 4], [2, 2],
                  weights_initializer=tf.random_normal_initializer(stddev=0.02),
                  activation_fn=relu_batch_norm
              )
              conv4 = tcl.conv2d_transpose(
                  conv3, int(128/self.divisor), [4, 4], [2, 2],
                  weights_initializer=tf.random_normal_initializer(stddev=0.02),
                  activation_fn=relu_batch_norm
              )
              conv5 = tcl.conv2d_transpose(
                  conv4, 3, [4, 4], [2, 2],
                  weights_initializer=tf.random_normal_initializer(stddev=0.02),
                  activation_fn=tf.sigmoid)
              return conv5

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


# In[ ]:


class ConvGAN(object):
  def __init__(self, reuse=False, gpu_mode=True, graph=None):
    self.reuse = reuse
    self.g_net = GeneratorConditional(divisor=4, add_noise=False)
    
    if not gpu_mode:
      with tf.device('/cpu:0'):
        tf.logging.info('conv_gan using cpu.')
        self._build_graph(graph)
    else:
      tf.logging.info('conv_gan using gpu.')
      self._build_graph(graph)
    self._init_session()
      
  def _build_graph(self, graph):
    if graph is None:
      self.g = tf.Graph()
    else:
      self.g = graph
      
    with self.g.as_default(), tf.variable_scope('conv_gan', reuse=self.reuse):
      self.actions = tf.placeholder(tf.float32, shape=[None, 12])
      self.y = self.g_net(self.actions, is_training=False)
      self.init = tf.global_variables_initializer()
  
  def generate_stroke_graph(self, actions):
    with tf.variable_scope('conv_gan', reuse=True):
      with self.g.as_default():
        return self.g_net(actions, is_training=False)
      
  def _init_session(self):
    """Launch TensorFlow session and initialize variables"""
    self.sess = tf.Session(graph=self.g)
    self.sess.run(self.init)
  def close_sess(self):
    """ Close TensorFlow session """
    self.sess.close()


# In[ ]:





# ## Construct the Lucid graph

# In[ ]:


# these constants help remember which image is at which batch dimension
TRANSFER_INDEX = 0
CONTENT_INDEX = 1

content_layers = [
  'mixed3b',
]

def mean_L1(a, b):
  return tf.reduce_mean(tf.abs(a-b))

@wrap_objective
def activation_difference(layer_names, activation_loss_f=mean_L1, transform_f=None, difference_to=CONTENT_INDEX):
  def inner(T):
    # first we collect the (constant) activations of image we're computing the difference to
    image_activations = [T(layer_name)[difference_to] for layer_name in layer_names]
    if transform_f is not None:
      image_activations = [transform_f(act) for act in image_activations]
    
    # we also set get the activations of the optimized image which will change during optimization
    optimization_activations = [T(layer)[TRANSFER_INDEX] for layer in layer_names]
    if transform_f is not None:
      optimization_activations = [transform_f(act) for act in optimization_activations]
    
    # we use the supplied loss function to compute the actual losses
    losses = [activation_loss_f(a, b) for a, b in zip(image_activations, optimization_activations)]
    return tf.add_n(losses) 
    
  return inner

def import_model(model, t_image, t_image_raw, scope="import"):

  model.import_graph(t_image, scope=scope, forget_xy_shape=True)

  def T(layer):
    if layer == "input": return t_image_raw
    if layer == "labels": return model.labels
    if ":" in layer:
        return t_image.graph.get_tensor_by_name("%s/%s" % (scope,layer))
    else:
        return t_image.graph.get_tensor_by_name("%s/%s:0" % (scope,layer))

  return T

class LucidGraph(object):
  def __init__(self, content, overlap_px=10, repeat=2, num_strokes=5, painter_type="GAN", connected=True, alternate=True, bw=False, learning_rate=0.1, gpu_mode=True, graph=None):
    self.overlap_px = overlap_px
    self.repeat = repeat
    self.full_size = 64*repeat - overlap_px*(repeat - 1)
    self.unrepeated_num_strokes= num_strokes
    self.num_strokes= num_strokes * self.repeat**2
    self.painter_type = painter_type
    self.connected=connected
    self.alternate=alternate
    self.bw = bw
    print('full_size', self.full_size, 'max_seq_len', self.num_strokes)
    
    self.content=content
    self.inception_v1 = models.InceptionV1()
    self.inception_v1.load_graphdef()
    transforms = [
      transform.pad(12, mode='constant', constant_value=.5),
      transform.jitter(8),
      transform.random_scale([1 + (i-5)/50. for i in range(11)]),
      transform.random_rotate(list(range(-5, 5)) + 5*[0]),
      transform.jitter(4),
    ]
    
    self.transform_f = render.make_transform_f(transforms)
    
    self.optim = render.make_optimizer(tf.train.AdamOptimizer(learning_rate), [])

    self.gpu_mode = gpu_mode
    if not gpu_mode:
      with tf.device('/cpu:0'):
        tf.logging.info('Model using cpu.')
        self._build_graph(graph)
    else:
      #tf.logging.info('Model using gpu.')
      self._build_graph(graph)
    self._init_session()
  
  def _build_graph(self, graph):
    if graph is None:
      self.g = tf.Graph()
    else:
      self.g = graph
    
    # Set up graphs of VAE or GAN
    if self.painter_type == "GAN":
      self.painter = ConvGAN(
              reuse=False,
              gpu_mode=self.gpu_mode,
              graph=self.g)
    elif self.painter_type=="VAE":
      self.painter = ConvVAE2(
              reuse=False,
              gpu_mode=self.gpu_mode,
              graph=self.g)
    self.painter.close_sess()
    
    with self.g.as_default():
      print('GLOBAL VARS', tf.global_variables())
    
    with self.g.as_default():
      batch_size = 1
      self.actions = tf.get_variable("action_vars", [batch_size, self.num_strokes, 12], 
                                     #initializer=tf.initializers.random_normal()
                                     initializer=tf.initializers.random_uniform()
                                    )
      if self.bw:
        actions2 = tf.concat([self.actions[:, :, :6], tf.zeros([1, self.num_strokes, 3]), self.actions[:, :, 9:]], axis=2)
      else:
        actions2 = self.actions
      
      self.actions_assign_ph = tf.placeholder(dtype=tf.float32)
      self.actions_assign_op = tf.assign(self.actions, self.actions_assign_ph)
      
      # Prepare loop vars for rnn loop
      canvas_state = tf.ones(shape=[batch_size, self.full_size, self.full_size, 3], dtype=tf.float32)
      i = tf.constant(0)
      initial_canvas_ta = tf.TensorArray(dtype=tf.float32, size=self.num_strokes)
      loop_vars = (
          canvas_state, 
          initial_canvas_ta, i)
      
      
      # condition for continuation
      def cond(cs, c_ta, i):
        return tf.less(i, self.num_strokes)
      
      # run one state of rnn cell
      def body(cs, c_ta, i):
        
        trimmed_actions = tf.sigmoid(actions2)
        
        print(trimmed_actions.get_shape())
        
        def use_whole_action():
          return trimmed_actions[:, i, :12]
        
        def use_previous_entrypoint():
          # start x and y are previous end x and y
          # start pressure is previous pressure
          return tf.concat([trimmed_actions[:, i, :9], trimmed_actions[:, i-1, 4:6], trimmed_actions[:, i-1, 0:1]], axis=1)
        
        if self.connected:
          inp = tf.cond(tf.equal(i, 0), true_fn=use_whole_action, false_fn=use_previous_entrypoint)
        else:
          inp = use_whole_action()
        inp = tf.reshape(inp, [-1, 12])
        
        print(inp.get_shape())
        
        decoded_stroke = self.painter.generate_stroke_graph(inp)
        
        cases = []
        ctr = 0
        for a in range(self.repeat):
          for b in range(self.repeat):
            print([int(self.repeat**2), ctr])
            print([[0, 0], [(64-self.overlap_px)*a, (64-self.overlap_px)*(self.repeat-1-a)], [(64-self.overlap_px)*b, (64-self.overlap_px)*(self.repeat-1-b)], [0, 0]])
            cases.append(
              (
                  tf.equal(tf.floormod(i, int(self.repeat**2)), ctr) if self.alternate else tf.less(i, self.unrepeated_num_strokes*(ctr+1)),
                  lambda a=a, b=b: tf.pad(decoded_stroke, 
                                 [[0, 0], [(64-self.overlap_px)*a, (64-self.overlap_px)*(self.repeat-1-a)], [(64-self.overlap_px)*b, (64-self.overlap_px)*(self.repeat-1-b)], [0, 0]], 
                                 constant_values=1)
              )
            )
            ctr += 1
        
        print(cases)
        decoded_stroke = tf.case(cases)

        darkness_mask = tf.reduce_mean(decoded_stroke, axis=3)
        darkness_mask = 1 - tf.reshape(darkness_mask, [batch_size, self.full_size, self.full_size, 1])
        darkness_mask = darkness_mask / tf.reduce_max(darkness_mask)
        
        color_action = trimmed_actions[:, i, 6:9]
        color_action = tf.reshape(color_action, [batch_size, 1, 1, 3])
        color_action = tf.tile(color_action, [1, self.full_size, self.full_size, 1])
        stroke_whitespace = tf.equal(decoded_stroke, 1.)
        maxed_stroke = tf.where(stroke_whitespace, decoded_stroke, color_action)
        
        cs = (darkness_mask)*maxed_stroke + (1-darkness_mask)*cs
        c_ta = c_ta.write(i, cs)
                
        i = tf.add(i, 1)
        return (cs, c_ta, i)
      
      final_canvas_state, final_canvas_ta, _ = tf.while_loop(cond, body, loop_vars, swap_memory=True)
      self.intermediate_canvases = final_canvas_ta.stack()
      
      content_input = tf.image.resize_images(np.expand_dims(self.content, 0), [self.full_size, self.full_size])
      final_canvas_state = tf.stack([final_canvas_state[0], content_input[0]])
      print(final_canvas_state.shape)
      self.final_canvas_state = final_canvas_state
      self.resized_final = tf.image.resize_images(final_canvas_state, [256, 256])
      
      #For visualization
      self.content_style_vis = final_canvas_state[1:]

      global_step = tf.train.get_or_create_global_step()
      
      with gradient_override_map({'Relu': redirected_relu_grad,
                                  'Relu6': redirected_relu6_grad}):
        self.T = render.import_model(self.inception_v1, self.transform_f(final_canvas_state), final_canvas_state)
        
      content_obj = 100 * activation_difference(content_layers, difference_to=CONTENT_INDEX)
      content_obj.description = "Content Loss"
            
      self.loss = content_obj(self.T)

      self.vis_op = self.optim.minimize(self.loss, global_step=global_step, var_list=[self.actions])

      # initialize vars
      self.init = tf.global_variables_initializer()
      
      print('TRAINABLE', tf.trainable_variables())
      
  def train(self, thresholds=range(0, 5000, 30)):
    self.images = []
    print(self.sess.run(self.actions))
    vis = self.sess.run(self.final_canvas_state)
    show(np.hstack(vis[:2]))
    try:
      for i in range(max(thresholds)+1):
        content_loss_, _ = self.sess.run([self.loss, self.vis_op])
        if i in thresholds:
          vis = self.sess.run(self.resized_final)
          print(i, content_loss_,)
          show(np.hstack(vis[:2]))
    except KeyboardInterrupt:
      vis = self.sess.run(self.final_canvas_state)
      show(np.hstack(vis[:2]))

  def _init_session(self):
    self.sess = tf.Session(graph=self.g)
    self.sess.run(self.init)
  def close_sess(self):
    self.sess.close()
    
      
  def load_painter_checkpoint(self, checkpoint_path='tf_conv_vae', actual_path=None):
    sess = self.sess
    with self.g.as_default():
      if self.painter_type == "VAE":
        pth = 'conv_vae'
      elif self.painter_type == "GAN":
        pth = 'conv_gan'
      saver = tf.train.Saver(tf.global_variables(pth))
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if actual_path is None:
      actual_path = ckpt.model_checkpoint_path
    print('loading model', actual_path)
    tf.logging.info('Loading model %s.', actual_path)
    saver.restore(sess, actual_path)


# ## Choose parameters

# In[8]:


#@title After running this cell manually, it will auto-run if you change the selected value. { run: "auto", display-mode: "form" }

NUMBER_STROKES = 4 #@param {type:"slider", min:1, max:10, step:1}
#@markdown Number of strokes per section. By default we have 64 sections.
PAINTER_MODE = "GAN" #@param ["GAN", "VAE"]
#@markdown GAN mode results in strokes that actually look like paintbrush strokes.
CONNECTED_STROKES = False #@param {type:"boolean"}
#@markdown If true, strokes begin at the endpoint of the previous stroke. Otherwise, strokes are independent and can start anywhere.
BW = False #@param {type:"boolean"}
#@markdown Black and white
LEARNING_RATE = 0.1 #@param {type: "number"}

print("Number of strokes", NUMBER_STROKES)
print("Using {} painter".format(PAINTER_MODE))
print("Using connected strokes", CONNECTED_STROKES)
print("Grayscale", BW)
print("Learning Rate", LEARNING_RATE)
print('--------------------')


# ## Input your content image here.
# 
# The `load` function takes a link or local filepath. Input images will be forced to squares.

# In[ ]:


# Load from a URL
if len(sys.argv) > 1:
  local_path = sys.argv[1]
else:
  local_path = "./F561f22668fee4.jpg"

CONTENT_IMAGE = load(local_path)[..., :3]  # Remove transparency channel

# Or load from a local path
#CONTENT_IMAGE = load("local_path.jpg")[..., :3]  # Remove transparency channel

show(CONTENT_IMAGE)


# ## Run!

# In[ ]:

# print(558)
lol = LucidGraph(CONTENT_IMAGE, 32, 8, NUMBER_STROKES, painter_type=PAINTER_MODE, gpu_mode=False, connected=CONNECTED_STROKES, alternate=False, bw=BW, learning_rate=LEARNING_RATE)

if PAINTER_MODE == "GAN":
  lol.load_painter_checkpoint('./tf_gan3')
elif PAINTER_MODE == "VAE":
  lol.load_painter_checkpoint('./tf_vae')
lol.train()


# ## Evaluate results

# In[ ]:


np.save('actions', lol.sess.run(lol.actions))
_acs = np.load('actions.npy')


# In[ ]:


# In[ ]:


# lol = LucidGraph(CONTENT_IMAGE, 32, 8, NUMBER_STROKES, painter_type=PAINTER_MODE, gpu_mode=False, connected=CONNECTED_STROKES, bw=BW, alternate=False, learning_rate=LEARNING_RATE)
# lol.load_painter_checkpoint('./tf_gan3')

_int_canvases, _content_style = lol.sess.run([lol.intermediate_canvases, lol.content_style_vis], feed_dict={lol.actions: _acs})


# In[14]:


_SIZE = lol.full_size

_stacked_plots = []
for _target in range(0, 1):

  _intermediate_canvases_to_plot = np.repeat(_int_canvases[::2][:, _target, :, :, :], 1, axis=0)
  _target_images = np.tile(np.concatenate(_content_style, axis=1).reshape(1, _SIZE, _SIZE, 3), 
                           [len(_intermediate_canvases_to_plot), 1, 1, 1])

  print(_intermediate_canvases_to_plot.shape)

  _plot = _intermediate_canvases_to_plot
  _plot = np.concatenate([
      _target_images[:, :, :_SIZE, :], 
      _plot, 
      #_target_images[:, :, _SIZE:, :]
  ], axis=2)
  
  _plot = np.concatenate([_plot, np.tile(_plot[-1:, :, :, :], [50, 1, 1, 1])], axis=0)
  
  _stacked_plots.append(_plot)


#imageio.mimsave('hello2.gif', np.concatenate(_stacked_plots),'GIF', fps=16)


# In[ ]:


from IPython.display import display

import moviepy.editor as mpy
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

mp4_file = os.path.join(os.path.split(local_path)[0], 'tmp.mp4')

def vid(my_frames):
  
  def frame(t):
    t = int(t*10.)
    if t >= len(my_frames):
      t = len(my_frames)-1
    return ((my_frames[t])*255).astype(np.float)

  clip = mpy.VideoClip(frame, duration=len(my_frames)/10)
  clip.write_videofile(mp4_file, fps=10.)
  display(mpy.ipython_display(mp4_file, height=400, max_duration=100.))
vid(np.concatenate(_stacked_plots))


# In[ ]:




