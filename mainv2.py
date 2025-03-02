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
from lava.magma.core.run_conditions import RunContinuous


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


############################### BRANCH 1 ##############################
# dense layers to connect buffer to e-Neurons + connect image to itself

dense_layers_1 = []
for i in range(num_exc_modules):
    weights = np.random.rand(exc_per_module, num_bins) 
    dense_layer = Dense(weights=weights, name = f'dense_layer_{i}_1')
    dense_layers_1.append(dense_layer)
print('initialised dense layer 1')

# create Excitatory module + connect dense to itself
excitatory_modules_1 = []
for i in range(num_exc_modules):
    module = LIF(
        shape=(exc_per_module,),
        du=0.2,
        dv=0.2,
        vth=0.8,
        name=f'exc_mod_{i}_1'
        
    )
    excitatory_modules_1.append(module)
    dense_layers_1[i].a_out.connect(excitatory_modules_1[i].a_in)
print('initialised exc layer 1')
    
# set up inter-mod connection in exc mods
e_intermod_dense_layers_1 = []
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
            connection = Dense(weights=weights, name = f'e_intermode_dense_layer_{i}_1')
            excitatory_modules_1[i].s_out.connect(
                connection.s_in)
            connection.a_out.connect(
                excitatory_modules_1[j].a_in)
print('initialised exc intermod con 1')

# create Inhibitory Module
inhibitory_modules_1 = []
for i in range(num_inh_modules):
    module = LIF(
        shape=(inh_per_module,),
        du=0.2,
        dv=0.2,
        vth=0.8,
        name=f'inh_mod_{i}_1'
    )
    inhibitory_modules_1.append(module)
print('initialised inh layer 1')

# set up self connection in inh module
inh_inter_dense_layers_1 = []
for i in range(num_inh_modules):
    connection = Dense(weights = np.random.random(size= (inh_per_module,inh_per_module)))
    inhibitory_modules_1[i].s_out.connect(connection.s_in)
    connection.a_out.connect(inhibitory_modules_1[i].a_in)
print('initialised inh self connection 1')
          
# connect excitatory to inhibitory
e_i_connections_1 = []
# Initialize a list to store weights for each excitatory module
module_weights_1 = [np.zeros((inh_per_module, exc_per_module)) for _ in range(num_exc_modules)]
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
    module_weights_1[module_idx][i, excitatory_idx] = np.random.randn(8) * 0.1

# Now create Dense layers for each excitatory module
for module_idx in range(num_exc_modules):
    # Create connection with the accumulated weights
    connection = Dense(weights=module_weights_1[module_idx], name=f'e-i dense module {module_idx}_1')
    
    # Connect the excitatory module to the Dense layer
    excitatory_modules_1[module_idx].s_out.connect(connection.s_in)
    
    # Connect the Dense layer output to the inhibitory module
    connection.a_out.connect(inhibitory_modules_1[0].a_in)
    
    # Append the connection to the list
    e_i_connections_1.append(connection)
print('initialised e-i connection 1')


i_e_connections_1 = []
inh_inter_dense_layers_1 = []

for i in range(num_inh_modules):
    connection = Dense(weights = np.random.random(size= (inh_per_module,inh_per_module)))
    inhibitory_modules_1[i].s_out.connect(connection.s_in)
    connection.a_out.connect(inhibitory_modules_1[i].a_in)
print('initialised i-e connection')

# set up learnable final layer
# connect e to Learning Dense layers
learning_dense_layers_1 =[]
monitor_1 = []
for i in range(num_exc_modules):
    stdp = STDPLoihi(learning_rate=1,
                     A_plus=1,
                     A_minus=-1,
                     tau_plus=10,
                     tau_minus=10,
                     t_epoch=4)

    learning_dense = LearningDense(
        weights = np.random.random(size = (1,exc_per_module)),
        learning_rule=stdp,
        name = f'plastic-connection-1.{i}'
    )
    excitatory_modules_1[i].s_out.connect(learning_dense.s_in)
    learning_dense_layers_1.append(learning_dense)
    monitor = Monitor()
    monitor_1.append(monitor)
print('intialised exc-learning dense connections')


############################### BRANCH 2 ##############################
dense_layers_2 = []
for i in range(num_exc_modules):
    weights = np.random.rand(exc_per_module, num_bins) * 15
    dense_layer = Dense(weights=weights, name = f'dense_layer_{i}_2')
    dense_layers_2.append(dense_layer)
print('initialised dense layer 2')

# create Excitatory module + connect dense to itself
excitatory_modules_2 = []
for i in range(num_exc_modules):
    module = LIF(
        shape=(exc_per_module,),
        du=0.2,
        dv=0.2,
        vth=0.8,
        name=f'exc_mod_{i}_2'
        
    )
    excitatory_modules_2.append(module)
    dense_layers_2[i].a_out.connect(excitatory_modules_2[i].a_in)
print('initialised exc layer 2')
    
# set up inter-mod connection in exc mods
e_intermod_dense_layers_2 = []
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
            connection = Dense(weights=weights, name = f'e_intermode_dense_layer_{i}_2')
            excitatory_modules_2[i].s_out.connect(
                connection.s_in)
            connection.a_out.connect(
                excitatory_modules_2[j].a_in)
print('initialised exc intermod con 2')

# create Inhibitory Module
inhibitory_modules_2 = []
for i in range(num_inh_modules):
    module = LIF(
        shape=(inh_per_module,),
        du=0.2,
        dv=0.2,
        vth=0.8,
        name=f'inh_mod_{i}_2'
    )
    inhibitory_modules_2.append(module)
print('initialised inh layer 2')

# set up self connection in inh module
inh_inter_dense_layers_2 = []
for i in range(num_inh_modules):
    connection = Dense(weights = np.random.random(size= (inh_per_module,inh_per_module)))
    inhibitory_modules_2[i].s_out.connect(connection.s_in)
    connection.a_out.connect(inhibitory_modules_2[i].a_in)
print('initialised inh self connection 2')
          
# connect excitatory to inhibitory
e_i_connections_2 = []

# Initialize a list to store weights for each excitatory module
module_weights_2 = [np.zeros((inh_per_module, exc_per_module)) for _ in range(num_exc_modules)]

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
    module_weights_2[module_idx][i, excitatory_idx] = np.random.randn(8) * 0.1

# Now create Dense layers for each excitatory module
for module_idx in range(num_exc_modules):
    # Create connection with the accumulated weights
    connection = Dense(weights=module_weights_2[module_idx], name=f'e-i dense module {module_idx} 2')
    
    # Connect the excitatory module to the Dense layer
    excitatory_modules_2[module_idx].s_out.connect(connection.s_in)
    
    # Connect the Dense layer output to the inhibitory module
    connection.a_out.connect(inhibitory_modules_2[0].a_in)
    
    # Append the connection to the list
    e_i_connections_2.append(connection)
print('initialised e-i connection 2')


i_e_connections_2 = []
inh_inter_dense_layers_2 = []
# init i e connections
for i in range(num_inh_modules):
    connection = Dense(weights = np.random.random(size= (inh_per_module,inh_per_module)))
    inhibitory_modules_2[i].s_out.connect(connection.s_in)
    connection.a_out.connect(inhibitory_modules_2[i].a_in)
print('initialised i-e connection 2')


# init learnable layer
learning_dense_layers_2 =[]
monitor_2 = []
for i in range(num_exc_modules):
    stdp = STDPLoihi(learning_rate=1,
                     A_plus=1,
                     A_minus=-1,
                     tau_plus=10,
                     tau_minus=10,
                     t_epoch=4)

    learning_dense = LearningDense(
        weights = np.random.random(size = (1,exc_per_module)),
        learning_rule=stdp,
        name = f'plastic-connection-2.{i}'
    )
    excitatory_modules_2[i].s_out.connect(learning_dense.s_in)
    learning_dense_layers_2.append(learning_dense)
    monitor = Monitor()
    monitor_2.append(monitor)
print('intialised exc-learning dense connections')

############## WTA#######################
# now initialise WTA setup by connecting ihb1 -> exc2 and inh2 -> exc1
wta_1 = []
for ihb in inhibitory_modules_1:
    for i,exc in enumerate(excitatory_modules_2):
        dense = Dense(
            weights=np.random.random(size=(exc_per_module, inh_per_module)),
            name = 'wta-ihb1-exc2.'
        )
        ihb.s_out.connect(dense.s_in)
        dense.a_out.connect(exc.a_in)
        wta_1.append(dense)
print('initialised ihb1 -> exc2 connections')

wta_2 = []
for ihb in inhibitory_modules_2:
    for i,exc in enumerate(excitatory_modules_1):
        dense = Dense(
            weights=np.random.random(size=(exc_per_module, inh_per_module)),
            name = 'wta-ihb1-exc2.'
        )
        ihb.s_out.connect(dense.s_in)
        dense.a_out.connect(exc.a_in)
        wta_2.append(dense)
print('initialised ihb2 -> exc2 connections')
    
##################DATA###################

spike_in = RingBuffer(data=np.zeros((480000,2)))  # Initialize with appropriate shape
spike_in.run(condition=RunContinuous(),run_cfg=Loihi2SimCfg())

# connect spike input to first dense layer
for dense in dense_layers_1: 
    spike_in.s_out.connect(dense.s_in)

for dense in dense_layers_2:
   spike_in.s_out.connect(dense.s_in)
   
   
# connect monitors to output
for i in range(num_exc_modules):
    m1 = monitor_1[i]
    m2 = monitor_2[i]
    m1.probe(target = learning_dense_layers_1[i].a_out, num_steps = RunContinuous())
    m2.probe(target = learning_dense_layers_2[i].a_out, num_steps = RunContinuous())
    

print('ready to start')
# reward_buffer = RingBuffer(data=np.zeros((1,2)))

DATA_PATH = 'tmp/mask-no-mask-1'
TRAIN_PATH = DATA_PATH + '/Training/Training'
TRAIN_IMG_PATH = TRAIN_PATH + '/Images/'
TRAIN_LABEL_PATH = TRAIN_PATH + '/Labels/'


TRAIN_IMG_DATAFILES = [f for f in os.listdir(TRAIN_IMG_PATH)]
accs = []
_temp = []
num_prediction_per_min = []
start_time = time.time()
while True:
    idx = np.random.randint(0,len(TRAIN_IMG_DATAFILES))
    img_file_path = TRAIN_IMG_PATH + TRAIN_IMG_DATAFILES[idx]
    label_file_path = TRAIN_LABEL_PATH + TRAIN_IMG_DATAFILES[idx][:-3] + 'txt'
    
    if time.time() - start_time > 60:
        accs.append(np.sum(_temp))
        num_prediction_per_min.append(len(_temp))
        _temp = []
    try:
        img = Image.open(img_file_path)
        img = img.convert("L")
        img = np.array(img)
        data = process_image(img)
        data = np.array([data,data]).T
        
        final_label = 0
        with open(label_file_path, 'r') as f:
            lines = f.readlines()
                
            # Extract first number from each line
            first_numbers = []
            
            for line in lines:
                parts = line.strip().split()
                if parts:  # Check if line has content
                    first_number = int(float(parts[0]))  # Handle both integer and float formats
                    first_numbers.append(first_number)
            
            # Count occurrences of 0
            count_zeros = first_numbers.count(0)
            
            # Determine final label
            if count_zeros > len(first_numbers) / 2:
                final_label = 1
            else:
                final_label = 0
        # buffer to pass data in
        spike_in.data = data.astype(int)
        
        # aggregate predictions
        predictions_1 = []
        for i, monitor in enumerate(monitor_1):
            prediction = monitor.get_data()[f'plastic-connection-1.{i}']['a_out']
            predictions_1.append(prediction)
        predictions_2 = []
        for i, monitor in enumerate(monitor_2):
            prediction = monitor.get_data()[f'plastic-connection-2.{i}']['a_out']
            predictions_2.append(prediction)
        print(f'{np.mean(predictions_1)},  {np.mean(predictions_2)}')
        prediction = 0 if np.mean(predictions_1) > np.mean(predictions_2) else 1
            
        if final_label == 0:
            # encourage
            for ld in learning_dense_layers_1:
                ld._learning_rule.A_plus = 1
                ld._learning_rule.A_minus = -1
            # discourage
            for ld in learning_dense_layers_2:
                ld._learning_rule.A_plus = -1
                ld._learning_rule.A_minus = 1
        else:
            # discourage
            for ld in learning_dense_layers_1:
                ld._learning_rule.A_plus = -1
                ld._learning_rule.A_minus = 1
            # encourage
            for ld in learning_dense_layers_2:
                ld._learning_rule.A_plus = 1
                ld._learning_rule.A_minus = -1
                
        _temp.append(prediction == final_label)
        print(f'predicted : {prediction}, actual : {final_label}')
    except KeyboardInterrupt:
        np.save('accuracies', accs)
        np.save('prediction speed', num_prediction_per_min)
        break
    except Exception as e:
        print(f'error : {e}')
        continue