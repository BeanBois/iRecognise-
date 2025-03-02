# one big file bc its too hard to implement it cleanly

import numpy as np
from PIL import Image
import time
import cv2
from skimage.measure import block_reduce
import os

from lava.proc.dense.process import Dense, LearningDense
from lava.proc.lif.process import LIF
from lava.proc.monitor.process import Monitor
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires

from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io.source import RingBuffer
from utils.load_data import load_mask_no_mask_1

raw_data, labels = load_mask_no_mask_1(training=True,num=2000)
pos_ind = labels == 0
print(pos_ind)
raw_data = raw_data[pos_ind]
labels = labels[pos_ind]
print(raw_data.shape)

# loading image
# img = Image.open('tmp/mask-no-mask-1/Training/Training/Images/video_27_02_2022_1 (669).jpg')

# img = np.array(img)

shape = (800,600)
num_bins = 800 * 600
num_exc_modules = 10 // 2
exc_per_module = 200
num_inh_modules = 2 // 2
inh_per_module = 400 
# process image
def _calculate_hilbert_distances(height, width):
    """Calculate Hilbert distances for each pixel."""
    # Simple implementation - could be optimized with a real Hilbert curve
    distances = np.zeros(height * width)
    for y in range(height):
        for x in range(width):
            # Use Bit interleaving for approximating Hilbert distance
            distances[y * width + x] = _interleave_bits(x, y)
    return distances

def _interleave_bits(x, y):
    """Interleave bits of x and y to approximate Hilbert distance."""
    result = 0
    for i in range(16):
        result |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
    return result

def process_image( image):
    """
    Process image and prepare for spike generation.
    """
    from skimage.measure import block_reduce
    
    # Resize image to expected shape
    resized = cv2.resize(image, shape)
    # Convert to grayscale if needed
    if len(resized.shape) > 2:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # Calculate Hilbert distances for each pixel
    height, width = shape
    hilbert_distances = _calculate_hilbert_distances(height, width)

    # Sort pixel intensities by Hilbert distance
    sorted_pixels = gray.flatten()[np.argsort(hilbert_distances)]

    # Bin the sorted pixels
    bin_size = len(sorted_pixels) // num_bins
    binned_pixels = block_reduce(sorted_pixels, (bin_size,), np.mean)

    # Truncate or pad to ensure correct size
    if len(binned_pixels) > num_bins:
        binned_pixels = binned_pixels[:num_bins]
    elif len(binned_pixels) < num_bins:
        binned_pixels = np.pad(binned_pixels,
                            (0, num_bins - len(binned_pixels)),
                            'constant')

    # Normalize binned pixels to range [0, 1]
    if np.max(binned_pixels) > 0:
        binned_pixels = binned_pixels / np.max(binned_pixels)

    # Store for spike generation
    image = gray
    binned_values = binned_pixels
    return binned_values

data = []
for i in range(len(labels)):
    img = (process_image(raw_data[i]) * 255).astype(int)
    data.append(img)

print(len(data))
data = np.asarray(data)
data = data.reshape((num_bins,-1))
print(data.shape)

# binned_values = (process_image(img) * 255).astype(int)
# binned_values = np.array([binned_values,binned_values])
# binned_values = binned_values.reshape((784,-1))


# buffer to pass data in
spike_in = RingBuffer(data=data.astype(int))
print('initialised buffer')
# spike_to_exc_dense_layers = []
# spacing = num_bins / exc_per_module
# for i in range(num_exc_modules):
#     weights = np.random.rand(exc_per_module, num_bins)
#     connection = Dense(weights = weights, name = 'spike-source-to-exc-dense-layer-{i}')
#     spike_to_exc_dense_layers.append(connection)

# image processing layer than sends splits data into 5, and send each segment to each E-neuron

# dense layers to connect buffer to e-Neurons + connect image to itself
dense_layers = []
for i in range(num_exc_modules):
    # Calculate segment size
    # base_segment_size = num_bins // num_exc_modules
    # remainder = num_bins % num_exc_modules
    # segment_size = base_segment_size + (1 if i < remainder else 0)
    
    # Create weights and dense layer
    # weights = np.random.randn(exc_per_module, segment_size) * 0.1
    weights = np.random.rand(exc_per_module, num_bins) * 15
    dense_layer = Dense(weights=weights, name = f'dense_layer_{i}')
    spike_in.s_out.connect(dense_layer.s_in)
    dense_layers.append(dense_layer)
print('initialised dense layer')

# create Excitatory module + connect dense to itself
excitatory_modules = []
for i in range(num_exc_modules):
    module = LIF(
        shape=(exc_per_module,),
        du=0.2,
        dv=0.2,
        vth=0.8,
        name=f'exc_mod_{i}'
        
    )
    excitatory_modules.append(module)
    dense_layers[i].a_out.connect(excitatory_modules[i].a_in)
print('initialised exc layer')
    
# set up inter-mod connection in exc mods
e_intermod_dense_layers = []
for i in range(5):
    for j in range(5):
        if i != j:  # Don't connect to self
            # Connect with probability 0.2
            # Create random weights
            weights = np.random.randn(
                exc_per_module,
                exc_per_module
            ) 
            weights[weights < 0.2] = 0

            # Create connection
            connection = Dense(weights=weights, name = f'e_intermode_dense_layer_{i}')
            excitatory_modules[i].s_out.connect(
                connection.s_in)
            connection.a_out.connect(
                excitatory_modules[j].a_in)
print('initialised exc intermod con')

# create Inhibitory Module
inhibitory_modules = []
for i in range(num_inh_modules):
    module = LIF(
        shape=(inh_per_module,),
        du=0.2,
        dv=0.2,
        vth=0.8,
        name=f'inh_mod_{i}'
    )
    inhibitory_modules.append(module)
print('initialised inh layer')

# set up self connection in inh module
inh_inter_dense_layers = []
for i in range(num_inh_modules):
    connection = Dense(weights = np.random.random(size= (inh_per_module,inh_per_module)))
    inhibitory_modules[i].s_out.connect(connection.s_in)
    connection.a_out.connect(inhibitory_modules[i].a_in)
print('initialised inh self connection')
          
# connect excitatory to inhibitory
e_i_connections = []

# Initialize a list to store weights for each excitatory module
module_weights = [np.zeros((inh_per_module, exc_per_module)) for _ in range(num_exc_modules)]

# Iterate through inhibitory neurons and update the weights matrices
for i in range(inh_per_module):
    # Choose random module from mega-module
    module_idx = np.random.randint(0, num_exc_modules)
    
    # Choose 8 random excitatory neurons from the selected module
    excitatory_idx = np.random.choice(
        exc_per_module,
        size=8,
        replace=False
    )
    
    # Update weights for the specific inhibitory neuron and excitatory neurons
    module_weights[module_idx][i, excitatory_idx] = np.random.randn(8) * 0.1

# Now create Dense layers for each excitatory module
for module_idx in range(num_exc_modules):
    # Create connection with the accumulated weights
    connection = Dense(weights=module_weights[module_idx], name=f'e-i dense module {module_idx}')
    
    # Connect the excitatory module to the Dense layer
    excitatory_modules[module_idx].s_out.connect(connection.s_in)
    
    # Connect the Dense layer output to the inhibitory module
    connection.a_out.connect(inhibitory_modules[0].a_in)
    
    # Append the connection to the list
    e_i_connections.append(connection)

print('initialised e-i connection')


i_e_connections = []
inh_inter_dense_layers = []
for i in range(num_inh_modules):
    connection = Dense(weights = np.random.random(size= (inh_per_module,inh_per_module)))
    inhibitory_modules[i].s_out.connect(connection.s_in)
    connection.a_out.connect(inhibitory_modules[i].a_in)
print('initialised i-e connection')

# set up learnable final layer
stdp = STDPLoihi(learning_rate=1,
                 A_plus=1,
                 A_minus=-1,
                 tau_plus=10,
                 tau_minus=10,
                 t_epoch=4)

learning_dense = LearningDense(
    weights = np.random.random(size = (1000,1)),
    learning_rule=stdp,
    name = 'plastic-connection'
)


spike_in.run(condition = RunSteps(num_steps=50),run_cfg = Loihi2SimCfg())
monitor = Monitor()
monitor.probe(target = learning_dense.a_out, num_steps = 50)

print('done')
for _ in range(30):
    print(monitor.get_data())