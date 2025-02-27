import numpy as np
from lava.proc import dense
from lava.proc.lif.process import LIF
from lava.proc.monitor.process import Monitor
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import cv2
from skimage.measure import block_reduce

class ImageProcessor(AbstractProcess):
    """
    Process images for SNN input.
    - Converts to grayscale
    - Sorts by Hilbert distance
    - Bins pixels and generates spikes
    """
    
    def __init__(self, shape=(28, 28), num_bins=100):
        super().__init__()
        
        # Ports
        self.s_out = OutPort(shape=(num_bins,))
        
        # Variables
        self.shape = Var(shape)
        self.num_bins = Var(num_bins)
        self.image = Var(np.zeros(shape))
        self.current_spikes = Var(np.zeros(num_bins))
    
    def process_image(self, image):
        """Process image and prepare spike generation."""
        # Resize image to expected shape
        resized = cv2.resize(image, self.shape.get())
        
        # Convert to grayscale if needed
        if len(resized.shape) > 2:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # Calculate Hilbert distances for each pixel
        height, width = self.shape.get()
        hilbert_distances = self._calculate_hilbert_distances(height, width)
        
        # Sort pixel intensities by Hilbert distance
        sorted_pixels = gray.flatten()[np.argsort(hilbert_distances)]
        
        # Bin the sorted pixels
        bin_size = len(sorted_pixels) // self.num_bins.get()
        binned_pixels = block_reduce(sorted_pixels, (bin_size,), np.mean)
        
        # Truncate or pad to ensure correct size
        if len(binned_pixels) > self.num_bins.get():
            binned_pixels = binned_pixels[:self.num_bins.get()]
        elif len(binned_pixels) < self.num_bins.get():
            binned_pixels = np.pad(binned_pixels, 
                                  (0, self.num_bins.get() - len(binned_pixels)), 
                                  'constant')
        
        # Normalize binned pixels to range [0, 1]
        if np.max(binned_pixels) > 0:
            binned_pixels = binned_pixels / np.max(binned_pixels)
        
        # Store for spike generation
        self.image.set(gray)
        self.binned_values = binned_pixels
    
    def _calculate_hilbert_distances(self, height, width):
        """Calculate Hilbert distances for each pixel."""
        # Simple implementation - could be optimized with a real Hilbert curve
        distances = np.zeros(height * width)
        for y in range(height):
            for x in range(width):
                # Use Bit interleaving for approximating Hilbert distance
                distances[y * width + x] = self._interleave_bits(x, y)
        return distances
    
    def _interleave_bits(self, x, y):
        """Interleave bits of x and y to approximate Hilbert distance."""
        result = 0
        for i in range(16):
            result |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
        return result
    
    def run_timestep(self):
        """Generate Poisson spikes based on binned pixel values."""
        # Generate Poisson spikes based on binned pixel values
        spike_rates = self.binned_values * 100  # Scale to reasonable firing rates
        spikes = np.zeros(self.num_bins.get())
        
        for i in range(self.num_bins.get()):
            spikes[i] = np.random.poisson(spike_rates[i]) > 0
        
        # Send spikes to output port
        self.current_spikes.set(spikes)
        self.s_out.send(spikes)


class RSTDPSynapse(AbstractProcess):
    """
    Implements Reward-modulated Spike-Timing-Dependent Plasticity (R-STDP).
    
    This class manages synaptic weight updates based on pre-post spike timing
    and a global reward signal.
    """
    
    def __init__(self, shape, initial_weights=None, learning_rate=0.01, 
                 a_plus=0.1, a_minus=0.12, tau_plus=20, tau_minus=20):
        super().__init__()
        
        # Shape of weight matrix
        if initial_weights is None:
            initial_weights = np.random.randn(*shape) * 0.1
        
        # Synaptic weights
        self.weights = Var(initial_weights)
        self.shape = Var(shape)
        
        # Learning parameters
        self.learning_rate = Var(learning_rate)
        self.a_plus = Var(a_plus)          # Amplitude of weight potentiation
        self.a_minus = Var(a_minus)        # Amplitude of weight depression
        self.tau_plus = Var(tau_plus)      # Time constant for potentiation
        self.tau_minus = Var(tau_minus)    # Time constant for depression
        
        # Eligibility trace
        self.eligibility_trace = Var(np.zeros(shape))
        
        # Spike traces
        self.pre_trace = Var(np.zeros(shape[1]))
        self.post_trace = Var(np.zeros(shape[0]))
        
        # Reward signal
        self.reward = Var(0.0)
        self.baseline_reward = Var(0.0)    # Running average of reward
        self.reward_decay = Var(0.99)      # Decay factor for baseline reward
        
        # Inputs and outputs
        self.pre_spikes = Var(np.zeros(shape[1]))
        self.post_spikes = Var(np.zeros(shape[0]))
        
        # Ports
        self.s_out = OutPort(shape=(shape[0],))
    
    def update_traces(self):
        """Update pre and post synaptic traces based on spike events."""
        # Decay existing traces
        self.pre_trace.set(self.pre_trace.get() * np.exp(-1/self.tau_plus.get()))
        self.post_trace.set(self.post_trace.get() * np.exp(-1/self.tau_minus.get()))
        
        # Add new spikes to traces
        for i, spike in enumerate(self.pre_spikes.get()):
            if spike > 0:
                self.pre_trace.get()[i] += 1
        
        for i, spike in enumerate(self.post_spikes.get()):
            if spike > 0:
                self.post_trace.get()[i] += 1
    
    def compute_eligibility_trace(self):
        """Compute eligibility trace based on pre and post synaptic activity."""
        pre_expanded = np.expand_dims(self.pre_trace.get(), 0)
        post_expanded = np.expand_dims(self.post_trace.get(), 1)
        
        # STDP update rule
        # If post neuron fires after pre neuron: potentiation (pre -> post)
        potentiation = self.a_plus.get() * np.outer(self.post_spikes.get(), self.pre_trace.get())
        
        # If pre neuron fires after post neuron: depression (post -> pre)
        depression = self.a_minus.get() * np.outer(self.post_trace.get(), self.pre_spikes.get())
        
        # Update eligibility trace
        delta_e = potentiation - depression
        self.eligibility_trace.set(self.eligibility_trace.get() * 0.9 + delta_e)
    
    def set_reward(self, reward):
        """Set the current reward value."""
        self.reward.set(reward)
        
        # Update baseline reward (running average)
        current_baseline = self.baseline_reward.get()
        new_baseline = self.reward_decay.get() * current_baseline + (1 - self.reward_decay.get()) * reward
        self.baseline_reward.set(new_baseline)
    
    def update_weights(self):
        """Update weights based on eligibility trace and reward."""
        # Calculate reward prediction error (RPE)
        rpe = self.reward.get() - self.baseline_reward.get()
        
        # Update weights using R-STDP rule
        delta_w = self.learning_rate.get() * rpe * self.eligibility_trace.get()
        
        # Apply weight changes
        new_weights = self.weights.get() + delta_w
        
        # Apply weight constraints (optional)
        new_weights = np.clip(new_weights, -1.0, 1.0)
        
        self.weights.set(new_weights)
    
    def run_timestep(self, pre_spikes, post_spikes=None, reward=None):
        """Process a single timestep of the R-STDP learning."""
        # Set inputs
        self.pre_spikes.set(pre_spikes)
        
        if post_spikes is not None:
            self.post_spikes.set(post_spikes)
        
        # Set reward if provided
        if reward is not None:
            self.set_reward(reward)
        
        # Update traces and compute weight updates
        self.update_traces()
        self.compute_eligibility_trace()
        self.update_weights()
        
        # Forward pass (weighted sum of pre-synaptic spikes)
        output = np.dot(pre_spikes, self.weights.get().T)
        
        # Send output
        self.s_out.send(output)
        
        return output


class SNNImageProcessor(AbstractProcess):
    """
    SNN-based image processor using Lava with R-STDP learning at the output layer.
    
    Architecture:
    - Input Layer: Receives image via listening port
    - Preprocessing: Converts to grayscale, sorts by Hilbert distance, bins pixels
    - SNN: Two-layer network with excitatory and inhibitory neurons
    - Output Layer: Makes binary classification with R-STDP learning
    """
    
    def __init__(self, input_shape=(28, 28), listen_port=8080):
        super().__init__()
        
        # Ports
        self.input_port = InPort(shape=input_shape)
        self.output_port = OutPort(shape=(2,))  # Binary classification
        
        # Variables
        self.listen_port = Var(listen_port)
        self.input_shape = Var(input_shape)
        
        # Network configuration
        self.num_bins = Var(100)
        self.num_excitatory_modules = Var(10)
        self.excitatory_per_module = Var(200)
        self.num_inhibitory_modules = Var(2)
        self.inhibitory_per_module = Var(400)
        
        # R-STDP specific parameters
        self.learning_rate = Var(0.01)
        self.reward_history = Var([])
        self.performance_history = Var([])
        
        # Build network
        self._build_network()
    
    def _build_network(self):
        # Create network components
        self._setup_input_layer()
        self._setup_dense_layers()
        self._setup_snn_layers()
        self._setup_output_layer_with_rstdp()
        self._connect_all_layers()
    
    def _setup_input_layer(self):
        """Setup input layer for image processing and spike generation."""
        self.input_layer = ImageProcessor(
            shape=self.input_shape.get(),
            num_bins=self.num_bins.get()
        )
    
    def _setup_dense_layers(self):
        """Setup two parallel dense layers."""
        excitatory_total = self.num_excitatory_modules.get() * self.excitatory_per_module.get()
        
        # Two parallel dense layers - each connected to half of the excitatory modules
        self.dense_layer1 = dense.Dense(
            weights=np.random.randn(self.num_bins.get(), excitatory_total // 2) * 0.1
        )
        
        self.dense_layer2 = dense.Dense(
            weights=np.random.randn(self.num_bins.get(), excitatory_total // 2) * 0.1
        )
    
    def _setup_snn_layers(self):
        """Setup SNN with excitatory and inhibitory layers."""
        # Layer 1: Excitatory neurons (10 modules of 200 neurons)
        self.excitatory_modules = []
        for i in range(self.num_excitatory_modules.get()):
            module = LIF(
                shape=(self.excitatory_per_module.get(),),
                du=0.2,
                dv=0.2,
                vth=0.8
            )
            self.excitatory_modules.append(module)
        
        # Layer 2: Inhibitory neurons (2 modules of 400 neurons)
        self.inhibitory_modules = []
        for i in range(self.num_inhibitory_modules.get()):
            module = LIF(
                shape=(self.inhibitory_per_module.get(),),
                du=0.2,
                dv=0.2,
                vth=0.8
            )
            self.inhibitory_modules.append(module)
        
        # Create monitors for recording activity
        self.excitatory_monitors = [Monitor() for _ in range(self.num_excitatory_modules.get())]
        self.inhibitory_monitors = [Monitor() for _ in range(self.num_inhibitory_modules.get())]
    
    def _setup_output_layer_with_rstdp(self):
        """Setup output layer connected to inhibitory modules using R-STDP."""
        # Calculate the total number of inhibitory neurons
        inh_total = self.num_inhibitory_modules.get() * self.inhibitory_per_module.get()
        
        # Create R-STDP connection for inhibitory to output
        self.output_rstdp = RSTDPSynapse(
            shape=(2, inh_total),  # Binary classification
            learning_rate=self.learning_rate.get()
        )
        
        # Create a monitor for output activity
        self.output_monitor = Monitor()
    
    def _connect_all_layers(self):
        """Connect all layers to form the complete network."""
        # Connect input to dense layers
        self.input_layer.s_out.connect(self.dense_layer1.a_in)
        self.input_layer.s_out.connect(self.dense_layer2.a_in)
        
        # Connect dense layers to excitatory modules
        # First dense layer connects to first mega-module (modules 0-4)
        for i in range(5):
            self.dense_layer1.s_out.connect(self.excitatory_modules[i].a_in)
            self.excitatory_modules[i].s_out.connect(self.excitatory_monitors[i].a_in)
        
        # Second dense layer connects to second mega-module (modules 5-9)
        for i in range(5, 10):
            self.dense_layer2.s_out.connect(self.excitatory_modules[i].a_in)
            self.excitatory_modules[i].s_out.connect(self.excitatory_monitors[i].a_in)
        
        # Connect excitatory modules within mega-modules with probability 0.2
        self._connect_inter_modules()
        
        # Connect excitatory to inhibitory modules
        self._connect_excitatory_to_inhibitory()
        
        # Connect inhibitory modules for winner-take-all
        self._setup_winner_take_all()
        
        # Connect inhibitory to output via R-STDP
        self._connect_inhibitory_to_output_with_rstdp()
    
    def _connect_inter_modules(self):
        """Connect modules within mega-modules with probability 0.2"""
        # First mega-module (modules 0-4)
        for i in range(5):
            for j in range(5):
                if i != j:  # Don't connect to self
                    # Connect with probability 0.2
                    if np.random.rand() < 0.2:
                        # Create random weights
                        weights = np.random.randn(
                            self.excitatory_per_module.get(),
                            self.excitatory_per_module.get()
                        ) * 0.1
                        
                        # Create connection
                        connection = dense.Dense(weights=weights)
                        self.excitatory_modules[i].s_out.connect(connection.a_in)
                        connection.s_out.connect(self.excitatory_modules[j].a_in)
        
        # Second mega-module (modules 5-9)
        for i in range(5, 10):
            for j in range(5, 10):
                if i != j:  # Don't connect to self
                    # Connect with probability 0.2
                    if np.random.rand() < 0.2:
                        # Create random weights
                        weights = np.random.randn(
                            self.excitatory_per_module.get(),
                            self.excitatory_per_module.get()
                        ) * 0.1
                        
                        # Create connection
                        connection = dense.Dense(weights=weights)
                        self.excitatory_modules[i].s_out.connect(connection.a_in)
                        connection.s_out.connect(self.excitatory_modules[j].a_in)
        
        # Create 2000 random connections within each module
        for i in range(self.num_excitatory_modules.get()):
            module = self.excitatory_modules[i]
            size = self.excitatory_per_module.get()
            
            # Create indices for random connections
            source_idx = np.random.randint(0, size, 2000)
            target_idx = np.random.randint(0, size, 2000)
            
            # Create sparse connectivity matrix
            weights = np.zeros((size, size))
            weights[source_idx, target_idx] = np.random.randn(2000) * 0.1
            
            # Create connection
            connection = dense.Dense(weights=weights)
            module.s_out.connect(connection.a_in)
            connection.s_out.connect(module.a_in)
    
    def _connect_excitatory_to_inhibitory(self):
        """Connect excitatory modules to inhibitory modules."""
        # First inhibitory module connects to first mega-module (modules 0-4)
        for i in range(self.inhibitory_per_module.get()):
            # Choose random module from mega-module
            module_idx = np.random.randint(0, 5)
            
            # Choose 8 random excitatory neurons from the selected module
            excitatory_idx = np.random.choice(
                self.excitatory_per_module.get(), 
                size=8, 
                replace=False
            )
            
            # Create weights (one-to-one connection)
            weights = np.zeros((self.excitatory_per_module.get(), 1))
            weights[excitatory_idx, 0] = np.random.randn(8) * 0.1
            
            # Create connection
            connection = dense.Dense(weights=weights)
            self.excitatory_modules[module_idx].s_out.connect(connection.a_in)
            connection.s_out.connect(self.inhibitory_modules[0].a_in[i:i+1])
        
        # Second inhibitory module connects to second mega-module (modules 5-9)
        for i in range(self.inhibitory_per_module.get()):
            # Choose random module from mega-module
            module_idx = np.random.randint(5, 10)
            
            # Choose 8 random excitatory neurons from the selected module
            excitatory_idx = np.random.choice(
                self.excitatory_per_module.get(), 
                size=8, 
                replace=False
            )
            
            # Create weights (one-to-one connection)
            weights = np.zeros((self.excitatory_per_module.get(), 1))
            weights[excitatory_idx, 0] = np.random.randn(8) * 0.1
            
            # Create connection
            connection = dense.Dense(weights=weights)
            self.excitatory_modules[module_idx].s_out.connect(connection.a_in)
            connection.s_out.connect(self.inhibitory_modules[1].a_in[i:i+1])
    
    def _setup_winner_take_all(self):
        """Setup winner-take-all inhibitory connections."""
        # First inhibitory module inhibits itself and second mega-module
        inh_size = self.inhibitory_per_module.get()
        exc_size = self.excitatory_per_module.get()
        
        # Self-inhibition for first inhibitory module
        weights = -np.ones((inh_size, inh_size)) * 0.5
        np.fill_diagonal(weights, 0)  # No self-inhibition
        self_inh1 = dense.Dense(weights=weights)
        self.inhibitory_modules[0].s_out.connect(self_inh1.a_in)
        self_inh1.s_out.connect(self.inhibitory_modules[0].a_in)
        
        # First inhibitory module inhibits second mega-module (modules 5-9)
        for i in range(5, 10):
            weights = -np.ones((inh_size, exc_size)) * 0.5
            inh_conn = dense.Dense(weights=weights)
            self.inhibitory_modules[0].s_out.connect(inh_conn.a_in)
            inh_conn.s_out.connect(self.excitatory_modules[i].a_in)
        
        # Self-inhibition for second inhibitory module
        weights = -np.ones((inh_size, inh_size)) * 0.5
        np.fill_diagonal(weights, 0)  # No self-inhibition
        self_inh2 = dense.Dense(weights=weights)
        self.inhibitory_modules[1].s_out.connect(self_inh2.a_in)
        self_inh2.s_out.connect(self.inhibitory_modules[1].a_in)
        
        # Second inhibitory module inhibits first mega-module (modules 0-4)
        for i in range(5):
            weights = -np.ones((inh_size, exc_size)) * 0.5
            inh_conn = dense.Dense(weights=weights)
            self.inhibitory_modules[1].s_out.connect(inh_conn.a_in)
            inh_conn.s_out.connect(self.excitatory_modules[i].a_in)
    
    def _connect_inhibitory_to_output_with_rstdp(self):
        """Connect inhibitory modules to output layer with R-STDP."""
        # Function to record inhibitory spikes for learning
        def record_inhibitory_spikes(spikes, module_idx, start_idx, end_idx, combined_buffer):
            # Store spikes in the combined buffer
            combined_buffer[start_idx:end_idx] = spikes
            return spikes
        
        # Function to forward combined inhibitory spikes to R-STDP
        def forward_to_rstdp(combined_spikes):
            # Process through R-STDP
            output = self.output_rstdp.run_timestep(combined_spikes)
            # Forward to output
            self.output_port.send(output)
            # Also send to monitor
            self.output_monitor.a_in.send(output)
            return output
        
        # Create a combined buffer for all inhibitory spikes
        inh_total = self.num_inhibitory_modules.get() * self.inhibitory_per_module.get()
        self.combined_inh_buffer = np.zeros(inh_total)
        
        # Connect inhibitory modules to the combined buffer
        for i in range(self.num_inhibitory_modules.get()):
            # Record activity for monitoring
            self.inhibitory_modules[i].s_out.connect(self.inhibitory_monitors[i].a_in)
            
            # Calculate indices in the combined buffer
            start_idx = i * self.inhibitory_per_module.get()
            end_idx = (i + 1) * self.inhibitory_per_module.get()
            
            # Create a closure to capture the correct module index and buffer indices
            def create_record_func(module_idx, start, end, buffer):
                return lambda spikes: record_inhibitory_spikes(spikes, module_idx, start, end, buffer)
            
            # Connect to record function
            record_func = create_record_func(i, start_idx, end_idx, self.combined_inh_buffer)
            self.inhibitory_modules[i].s_out.connect(record_func)
        
        # Forward combined buffer to R-STDP
        # In a real implementation, this would need to be triggered after all inhibitory
        # modules have reported their spikes for the current timestep
        # For simplicity, we'll connect from the last inhibitory module
        self.inhibitory_modules[-1].s_out.connect(lambda _: forward_to_rstdp(self.combined_inh_buffer))
    
    def train(self, image, label, num_timesteps=100):
        """Train the network on a single example, applying R-STDP learning to the output layer."""
        # Process image
        self.input_layer.process_image(image)
        
        # Output spikes for each timestep
        output_spikes = []
        
        # Run simulation for multiple timesteps
        for t in range(num_timesteps):
            # Process input
            self.input_layer.run_timestep()
            
            # Process dense layers
            self.dense_layer1.run_timestep()
            self.dense_layer2.run_timestep()
            
            # Process excitatory modules
            for module in self.excitatory_modules:
                module.run_timestep()
            
            # Process inhibitory modules
            for module in self.inhibitory_modules:
                module.run_timestep()
            
            # Get output from monitor
            if hasattr(self.output_monitor, 'data'):
                output_spikes.append(self.output_monitor.data)
        
        # Determine prediction from accumulated spikes
        if len(output_spikes) > 0:
            # Sum spikes across all timesteps
            total_spikes = np.sum(np.array(output_spikes), axis=0)
            prediction = np.argmax(total_spikes)
        else:
            # No spikes recorded, make a random prediction
            prediction = np.random.randint(0, 2)
        
        # Calculate reward based on prediction accuracy
        reward = 1.0 if prediction == label else -0.1
        
        # Apply reward to R-STDP output layer for learning
        self.output_rstdp.set_reward(reward)
        
        # Update history
        self.reward_history.get().append(reward)
        self.performance_history.get().append(prediction == label)
        
        return prediction
    
    def run(self, image):
        """Run the network on an input image and return the prediction (without learning)."""
        # Process image
        self.input_layer.process_image(image)
        
        # Output spikes for each timestep
        output_spikes = []
        
        # Run simulation for multiple timesteps
        num_timesteps = 100
        for t in range(num_timesteps):
            # Process input
            self.input_layer.run_timestep()
            
            # Process dense layers
            self.dense_layer1.run_timestep()
            self.dense_layer2.run_timestep()
            
            # Process excitatory modules
            for module in self.excitatory_modules:
                module.run_timestep()
            
            # Process inhibitory modules
            for module in self.inhibitory_modules:
                module.run_timestep()
            
            # Get output from monitor
            if hasattr(self.output_monitor, 'data'):
                output_spikes.append(self.output_monitor.data)
        
        # Determine prediction from accumulated spikes
        if len(output_spikes) > 0:
            # Sum spikes across all timesteps
            total_spikes = np.sum(np.array(output_spikes), axis=0)
            prediction = np.argmax(total_spikes)
        else:
            # No spikes recorded, make a random prediction
            prediction = np.random.randint(0, 2)
        
        return prediction
    
    def evaluate(self, test_images, test_labels):
        """Evaluate network performance on test data."""
        correct = 0
        for img, label in zip(test_images, test_labels):
            prediction = self.run(img)
            if prediction == label:
                correct += 1
        
        accuracy = correct / len(test_labels)
        return accuracy
    
    def get_learning_curves(self):
        """Return learning curves from training history."""
        # Calculate moving average of rewards and performance
        window_size = min(100, len(self.reward_history.get()))
        if window_size > 0:
            reward_ma = np.convolve(
                self.reward_history.get(), 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            
            perf_ma = np.convolve(
                self.performance_history.get(), 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
        else:
            reward_ma = []
            perf_ma = []
        
        return {
            'rewards': self.reward_history.get(),
            'performance': self.performance_history.get(),
            'reward_ma': reward_ma,
            'performance_ma': perf_ma
        }


# Example usage
def train_rstdp_network(train_images, train_labels, test_images, test_labels, epochs=5):
    """
    Train the R-STDP SNN on a dataset.
    
    Parameters:
    - train_images: Training images dataset
    - train_labels: Training labels
    - test_images: Test images dataset
    - test_labels: Test labels
    - epochs: Number of training epochs
    
    Returns:
    - Trained network and performance metrics
    """
    # Initialize network
    network = SNNImageProcessor()
    
    # Track performance
    train_accuracy = []
    test_accuracy = []
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Shuffle training data
        indices = np.random.permutation(len(train_images))
        train_images_shuffled = [train_images[i] for i in indices]
        train_labels_shuffled = [train_labels[i] for i in indices]
        
        # Train on all examples
        correct = 0
        for i, (img, label) in enumerate(zip(train_images_shuffled, train_labels_shuffled)):
            prediction = network.train(img, label)
            correct += (prediction == label)
            
            # Print progress
            if (i+1) % 100 == 0:
                print(f"  Processed {i+1}/{len(train_images)} examples. "
                      f"Accuracy: {correct/(i+1):.4f}")
        
        # Calculate training accuracy
        train_acc = correct / len(train_images)
        train_accuracy.append(train_acc)
        
        # Evaluate on test data
        test_acc = network.evaluate(test_images, test_labels)
        test_accuracy.append(test_acc)
        
        print(f"  Training accuracy: {train_acc:.4f}")
        print(f"  Test accuracy: {test_acc:.4f}")
    
    return network, train_accuracy, test_accuracy


# Example of how to use this with a dataset like MNIST for binary classification
def example_mnist_binary_classification():
    """
    Example usage with MNIST dataset for binary classification (digits 0 and 1).
    This function demonstrates how to:
    1. Load and preprocess the MNIST dataset
    2. Create a binary classification problem by selecting two classes
    3. Train the SNN with R-STDP output layer
    4. Visualize the learning results
    """
    try:
        # Try to import necessary libraries
        import numpy as np
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
        import matplotlib.pyplot as plt
        
        # Load MNIST dataset
        print("Loading MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1)
        X = mnist.data.astype('float32')
        y = mnist.target.astype('int')
        
        # Preprocess data
        print("Preprocessing data...")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_images = X_scaled.reshape(-1, 28, 28)
        
        # Binary classification subset (digits 0 and 1)
        mask = (y == 0) | (y == 1)
        X_binary = X_images[mask]
        y_binary = y[mask]
        
        # Convert labels to binary (0 or 1)
        y_binary = (y_binary == 1).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_binary, y_binary, test_size=0.2, random_state=42
        )
        
        # Limit training samples for faster demonstration if needed
        max_train_samples = 1000  # Adjust based on available computing resources
        X_train = X_train[:max_train_samples]
        y_train = y_train[:max_train_samples]
        
        max_test_samples = 200
        X_test = X_test[:max_test_samples]
        y_test = y_test[:max_test_samples]
        
        print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
        
        # Train network
        print("Training network...")
        network, train_accuracy, test_accuracy = train_rstdp_network(
            X_train, y_train, X_test, y_test, epochs=3
        )
        
        # Get learning curves
        learning_curves = network.get_learning_curves()
        
        # Plot results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_accuracy, '-o', label='Train')
        plt.plot(test_accuracy, '-s', label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Classification Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(learning_curves['reward_ma'], label='Reward (MA)')
        plt.plot(learning_curves['performance_ma'], label='Performance (MA)')
        plt.xlabel('Training Step')
        plt.ylabel('Moving Average')
        plt.title('Learning Progress')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("Example completed successfully!")
        return network
        
    except ImportError as e:
        print(f"Error importing required libraries: {e}")
        print("This example requires: numpy, scikit-learn, matplotlib")
        return None


if __name__ == "__main__":
    example_mnist_binary_classification()