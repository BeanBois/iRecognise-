import numpy as np
from lava.proc import dense
from lava.proc.lif.process import LIF
from lava.proc.monitor import Monitor
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import cv2
from skimage.measure import block_reduce

class SNNImageProcessor(AbstractProcess):
    """
    SNN-based image processor using Lava.
    
    Architecture:
    - Input Layer: Receives image via listening port
    - Preprocessing: Converts to grayscale, sorts by Hilbert distance, bins pixels
    - SNN: Two-layer network with excitatory and inhibitory neurons
    - Output Layer: Makes binary classification
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
        
        # SNN setup
        self._build_network()
    
    def _build_network(self):
        # Create network components
        self._setup_input_layer()
        self._setup_dense_layers()
        self._setup_snn_layers()
        self._setup_output_layer()
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
        
        # Create monitors
        self.excitatory_monitors = [Monitor() for _ in range(self.num_excitatory_modules.get())]
        self.inhibitory_monitors = [Monitor() for _ in range(self.num_inhibitory_modules.get())]
    
    def _setup_output_layer(self):
        """Setup output layer connected to inhibitory modules."""
        self.output_dense = dense.Dense(
            weights=np.random.randn(
                self.num_inhibitory_modules.get() * self.inhibitory_per_module.get(), 
                2
            ) * 0.1
        )
    
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
        
        # Connect inhibitory to output
        for i in range(self.num_inhibitory_modules.get()):
            self.inhibitory_modules[i].s_out.connect(self.output_dense.a_in)
            self.inhibitory_modules[i].s_out.connect(self.inhibitory_monitors[i].a_in)
        
        # Connect output
        self.output_dense.s_out.connect(self.output_port)
    
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
    
    def run(self, image):
        """Process an image through the network and return prediction."""
        # Send image to input layer
        self.input_layer.process_image(image)
        
        # Run network
        timesteps = 100
        for _ in range(timesteps):
            self.input_layer.run_timestep()
            self.dense_layer1.run_timestep()
            self.dense_layer2.run_timestep()
            
            for module in self.excitatory_modules:
                module.run_timestep()
            
            for module in self.inhibitory_modules:
                module.run_timestep()
            
            self.output_dense.run_timestep()
        
        # Get output prediction
        output_spikes = self.output_dense.s_out.data
        prediction = np.argmax(np.sum(output_spikes, axis=0))
        
        return prediction


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