import socket
import struct
import json
import threading
import time
import numpy as np
import random
import uuid
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

# Assuming model.py contains SNNImageProcessor with R-STDP output layer
from model import SNNImageProcessor

# For multicast communication
MULTICAST_GROUP = "239.255.1.1"  # Reserved multicast address
DEFAULT_FEDERATED_PORT = 8082


class ReservoirComputing(AbstractProcess):
    """
    Reservoir Computing module for deciding whether to accept weight updates from other devices.
    Uses Echo State Networks principles with R-STDP learning for output connections.
    """
    
    def __init__(self, reservoir_size=200, input_size=16, output_size=1, 
                 spectral_radius=0.95, sparsity=0.1, noise_level=0.01):
        super().__init__()
        
        # Ports
        self.a_in = InPort(shape=(input_size,))
        self.s_out = OutPort(shape=(output_size,))
        
        # Variables
        self.reservoir_size = Var(reservoir_size)
        self.input_size = Var(input_size)
        self.output_size = Var(output_size)
        self.spectral_radius = Var(spectral_radius)
        self.sparsity = Var(sparsity)
        self.noise_level = Var(noise_level)
        
        # Initialize reservoir weights (internal connections)
        reservoir = self._initialize_reservoir(reservoir_size, sparsity, spectral_radius)
        self.reservoir_weights = Var(reservoir)
        
        # Initialize input weights (input to reservoir)
        input_weights = np.random.randn(reservoir_size, input_size) * 0.1
        self.input_weights = Var(input_weights)
        
        # R-STDP parameters for output connections
        self.learning_rate = Var(0.01)
        self.a_plus = Var(0.1)          # LTP amplitude
        self.a_minus = Var(0.12)        # LTD amplitude
        self.tau_plus = Var(20)         # Pre-post time constant
        self.tau_minus = Var(20)        # Post-pre time constant
        
        # Initialize output weights (reservoir to output) with R-STDP
        output_weights = np.random.randn(output_size, reservoir_size) * 0.1
        self.output_weights = Var(output_weights)
        
        # Eligibility trace for R-STDP
        self.eligibility_trace = Var(np.zeros((output_size, reservoir_size)))
        
        # Spike traces for R-STDP
        self.reservoir_trace = Var(np.zeros(reservoir_size))  # Pre-synaptic trace
        self.output_trace = Var(np.zeros(output_size))        # Post-synaptic trace
        
        # Reservoir state
        self.reservoir_state = Var(np.zeros(reservoir_size))
        self.output_spikes = Var(np.zeros(output_size))
        
        # Reward signal
        self.reward = Var(0.0)
        self.baseline_reward = Var(0.0)    # Running average of reward
        self.reward_decay = Var(0.99)      # Decay factor for baseline reward
        
        # Training history for delayed rewards
        self.history = Var([])  # Will store history entries with all relevant data
        
        # Accuracy tracking for delayed rewards
        self.accuracy_history = Var([])
        
        # Performance tracking
        self.total_decisions = Var(0)
        self.good_decisions = Var(0)  # Decisions that improved performance
    
    def _initialize_reservoir(self, size, sparsity, spectral_radius):
        """Initialize the reservoir with the given sparsity and spectral radius."""
        # Create sparse random matrix
        reservoir = np.random.randn(size, size) * (np.random.rand(size, size) < sparsity)
        
        # Scale to desired spectral radius
        max_eigenvalue = np.max(np.abs(np.linalg.eigvals(reservoir)))
        if max_eigenvalue > 0:
            reservoir *= spectral_radius / max_eigenvalue
        
        return reservoir
    
    def reset_state(self):
        """Reset the reservoir state."""
        self.reservoir_state.set(np.zeros(self.reservoir_size.get()))
        self.reservoir_trace.set(np.zeros(self.reservoir_size.get()))
        self.output_trace.set(np.zeros(self.output_size.get()))
        self.output_spikes.set(np.zeros(self.output_size.get()))
    
    def _update_traces(self):
        """Update pre and post synaptic traces based on activity."""
        # Get current traces
        res_trace = self.reservoir_trace.get()
        out_trace = self.output_trace.get()
        
        # Decay existing traces
        res_trace = res_trace * np.exp(-1/self.tau_plus.get())
        out_trace = out_trace * np.exp(-1/self.tau_minus.get())
        
        # Add new activity to traces
        reservoir_state = self.reservoir_state.get()
        output_spikes = self.output_spikes.get()
        
        # For reservoir neurons, we use activation level as proxy for "spiking"
        # Map continuous activation to discrete "spike contribution"
        res_activity = (np.tanh(np.abs(reservoir_state)) > 0.5).astype(float)
        res_trace += res_activity
        
        # For output, we have binary spikes
        out_trace += output_spikes
        
        # Update traces
        self.reservoir_trace.set(res_trace)
        self.output_trace.set(out_trace)
    
    def _compute_eligibility_trace(self):
        """Compute eligibility trace based on pre and post synaptic activity."""
        # Get current traces and spikes
        res_trace = self.reservoir_trace.get()
        out_trace = self.output_trace.get()
        res_activity = (np.tanh(np.abs(self.reservoir_state.get())) > 0.5).astype(float)
        out_spikes = self.output_spikes.get()
        
        # Get eligibility trace
        eligibility = self.eligibility_trace.get()
        
        # STDP update rule components
        # Potentiation: post-neuron fires after pre-neuron
        potentiation = np.outer(out_spikes, res_trace) * self.a_plus.get()
        
        # Depression: pre-neuron fires after post-neuron
        depression = np.outer(out_trace, res_activity) * self.a_minus.get()
        
        # Update eligibility trace
        delta_e = potentiation - depression
        eligibility = eligibility * 0.9 + delta_e
        
        # Update eligibility trace
        self.eligibility_trace.set(eligibility)
    
    def set_reward(self, reward):
        """Set the current reward value."""
        self.reward.set(reward)
        
        # Update baseline reward (running average)
        current_baseline = self.baseline_reward.get()
        new_baseline = self.reward_decay.get() * current_baseline + (1 - self.reward_decay.get()) * reward
        self.baseline_reward.set(new_baseline)
    
    def _update_weights(self):
        """Update weights based on eligibility trace and reward."""
        # Calculate reward prediction error (RPE)
        rpe = self.reward.get() - self.baseline_reward.get()
        
        # Update weights using R-STDP rule
        eligibility = self.eligibility_trace.get()
        delta_w = self.learning_rate.get() * rpe * eligibility
        
        # Apply weight changes
        output_weights = self.output_weights.get()
        output_weights += delta_w
        
        # Apply weight constraints (optional)
        output_weights = np.clip(output_weights, -1.0, 1.0)
        
        self.output_weights.set(output_weights)
    
    def process_input(self, input_spikes):
        """
        Process input spikes through the reservoir with R-STDP learning.
        
        Parameters:
        - input_spikes: Binary input vector (e.g., from device ID)
        
        Returns:
        - Decision: Whether to accept the update (1) or reject it (0)
        """
        # Update reservoir state
        prev_state = self.reservoir_state.get()
        
        # Compute new state: previous state * reservoir weights + input * input weights
        new_state = np.tanh(
            np.dot(self.reservoir_weights.get(), prev_state) + 
            np.dot(self.input_weights.get(), input_spikes) + 
            np.random.randn(self.reservoir_size.get()) * self.noise_level.get()
        )
        
        # Update state
        self.reservoir_state.set(new_state)
        
        # Compute output (decision) using output weights
        output = np.dot(self.output_weights.get(), new_state)
        
        # Convert to binary decision spike
        decision = int(output[0] > 0)
        output_spike = np.array([decision])
        self.output_spikes.set(output_spike)
        
        # Update traces for R-STDP
        self._update_traces()
        self._compute_eligibility_trace()
        
        # Store history for delayed rewards
        history_entry = {
            'input': input_spikes.copy(),
            'reservoir_state': new_state.copy(),
            'output': output[0],
            'decision': decision,
            'eligibility_trace': self.eligibility_trace.get().copy(),
            'timestamp': time.time(),
            'processed': False
        }
        history = self.history.get()
        history.append(history_entry)
        
        # Keep history bounded (last 1000 entries)
        if len(history) > 1000:
            history = history[-1000:]
        self.history.set(history)
        
        # Update decision counter
        self.total_decisions.set(self.total_decisions.get() + 1)
        
        # Send decision
        self.s_out.send(np.array([decision]))
        
        return decision
    
    def apply_reward(self, reward, history_idx=-1):
        """
        Apply a reward to update the output weights using R-STDP.
        
        Parameters:
        - reward: Reward value (positive for good decisions, negative for bad)
        - history_idx: Index in history to update (default: latest)
        """
        # Set the reward
        self.set_reward(reward)
        
        # Get history entry to use its eligibility trace
        history = self.history.get()
        if not history:
            return False
        
        # Get the historical entry
        if history_idx < 0:
            history_idx = len(history) + history_idx
        
        if 0 <= history_idx < len(history):
            entry = history[history_idx]
            
            # Load the historical eligibility trace
            self.eligibility_trace.set(entry['eligibility_trace'])
            
            # Update weights using R-STDP
            self._update_weights()
            
            # Mark as processed
            entry['processed'] = True
            entry['reward'] = reward
            history[history_idx] = entry
            self.history.set(history)
            
            # Update good decision counter if reward is positive
            if reward > 0:
                self.good_decisions.set(self.good_decisions.get() + 1)
            
            return True
        
        return False
    
    def update_accuracy(self, accuracy):
        """
        Update the accuracy history for delayed reward calculation.
        
        Parameters:
        - accuracy: Current accuracy value
        """
        history = self.accuracy_history.get()
        history.append({
            'accuracy': accuracy,
            'timestamp': time.time()
        })
        
        # Keep history bounded (last 200 entries)
        if len(history) > 200:
            history = history[-200:]
        
        self.accuracy_history.set(history)
    
    def evaluate_delayed_rewards(self):
        """
        Evaluate delayed rewards by comparing accuracy before and after decisions.
        Updates weights based on the performance impact of past decisions.
        """
        accuracy_history = self.accuracy_history.get()
        decision_history = self.history.get()
        
        if len(accuracy_history) < 150 or len(decision_history) < 10:
            return  # Not enough history
        
        # Go through decision history (from oldest to newest)
        for i, entry in enumerate(decision_history[:-10]):  # Skip very recent decisions
            # Skip already processed entries or decisions that didn't accept updates
            if entry.get('processed', False):
                continue
            
            # Calculate time index of the decision
            decision_time = entry['timestamp']
            
            # Find accuracy history indices closest to decision time
            accuracy_times = [h['timestamp'] for h in accuracy_history]
            before_indices = [j for j, t in enumerate(accuracy_times) if t < decision_time]
            after_indices = [j for j, t in enumerate(accuracy_times) if t > decision_time]
            
            if len(before_indices) >= 50 and len(after_indices) >= 100:
                # Calculate accuracy before decision (last 50 entries before)
                before_accuracy = np.mean([accuracy_history[j]['accuracy'] 
                                         for j in before_indices[-50:]])
                
                # Calculate accuracy after decision (first 100 entries after)
                after_accuracy = np.mean([accuracy_history[j]['accuracy'] 
                                        for j in after_indices[:100]])
                
                # Determine if decision was good (improved accuracy)
                improvement = after_accuracy - before_accuracy
                
                # Add random noise to the reward signal
                noise = np.random.normal(0, 0.05)  # Gaussian noise
                noisy_improvement = improvement + noise
                
                # Apply simple reward: +1 for good update, -1 for bad update
                reward = 1.0 if noisy_improvement > 0 else -1.0
                
                # Apply reward using R-STDP
                self.apply_reward(reward, i)
                
                # Mark as processed and store reward info
                entry['processed'] = True
                entry['improvement'] = improvement
                entry['noisy_improvement'] = noisy_improvement
                entry['reward'] = reward
        
        # Update history
        self.history.set(decision_history)


class WeightUpdateBroadcaster(AbstractProcess):
    """
    Process that broadcasts weight updates to other devices in the federated learning network.
    Uses reliable multicast with collision avoidance.
    """
    
    def __init__(self, broadcast_port=DEFAULT_FEDERATED_PORT, device_id=None, 
                 multicast_group=MULTICAST_GROUP, peer_list=None):
        super().__init__()
        
        # Generate random device ID if not provided
        if device_id is None:
            device_id = random.randint(0, 2**16 - 1)
        
        # Ports
        self.a_in = InPort(shape=(1,))  # Trigger port
        
        # Variables
        self.device_id = Var(device_id)
        self.multicast_group = Var(multicast_group)
        self.port = Var(broadcast_port)
        self.running = Var(True)
        self.weight_changes = Var({})  # Accumulated weight changes
        self.training_count = Var(0)  # Counter for training iterations
        self.broadcast_threshold = Var(100)  # Broadcast after this many training iterations
        
        # For handling peer devices
        self.peer_list = Var(peer_list or [])
        self.known_updates = Var({})  # Track updates we've seen by their UUID
        
        # For collision avoidance
        self.is_broadcasting = Var(False)
        self.backoff_time = Var(random.uniform(0.1, 2.0))  # Random initial backoff
        
        # Socket
        self.socket = None
        
        # Start the broadcaster thread
        self.broadcaster_thread = threading.Thread(target=self._maintain_socket)
        self.broadcaster_thread.daemon = True
        self.broadcaster_thread.start()
    
    def _maintain_socket(self):
        """Maintain the multicast socket."""
        print(f"Initializing weight update broadcaster for device {self.device_id.get()}")
        
        try:
            # Create UDP socket for multicasting
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Set Time-to-Live (TTL)
            ttl = struct.pack('b', 32)  # Reasonable TTL for local network
            self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
            
            print(f"Broadcaster initialized and ready to send updates")
            
            while self.running.get():
                time.sleep(0.1)  # Just keep the socket alive
        
        except Exception as e:
            print(f"Error in broadcaster: {e}")
        
        finally:
            if self.socket:
                self.socket.close()
                self.socket = None
    
    def add_peer(self, peer_address):
        """Add a peer device to the peer list."""
        peers = self.peer_list.get()
        if peer_address not in peers:
            peers.append(peer_address)
            self.peer_list.set(peers)
            print(f"Added peer: {peer_address}")
    
    def accumulate_weight_changes(self, layer_name, weight_changes):
        """
        Accumulate weight changes for a layer.
        
        Parameters:
        - layer_name: Name of the layer (e.g., 'output_rstdp')
        - weight_changes: Dictionary of weight changes (key: weight index, value: change)
        """
        accumulated = self.weight_changes.get()
        
        if layer_name not in accumulated:
            accumulated[layer_name] = weight_changes
        else:
            for idx, change in weight_changes.items():
                if idx in accumulated[layer_name]:
                    accumulated[layer_name][idx] += change
                else:
                    accumulated[layer_name][idx] = change
        
        self.weight_changes.set(accumulated)
        
        # Increment training count
        count = self.training_count.get() + 1
        self.training_count.set(count)
        
        # Broadcast if threshold reached
        if count >= self.broadcast_threshold.get():
            # Don't broadcast immediately - use backoff to avoid collisions
            threading.Thread(target=self._schedule_broadcast).start()
            
            # Reset counters
            self.weight_changes.set({})
            self.training_count.set(0)
    
    def _schedule_broadcast(self):
        """Schedule broadcast with random backoff to avoid collisions."""
        # Check if already broadcasting
        if self.is_broadcasting.get():
            print("Already broadcasting, skipping this update")
            return
        
        # Set broadcasting flag
        self.is_broadcasting.set(True)
        
        # Wait for random backoff time
        backoff = self.backoff_time.get()
        print(f"Scheduling broadcast with {backoff:.2f}s backoff")
        time.sleep(backoff)
        
        # Broadcast
        self._broadcast_updates()
        
        # Reset broadcasting flag and adjust backoff for next time
        self.is_broadcasting.set(False)
        self.backoff_time.set(random.uniform(0.1, 2.0))  # New random backoff for next time
    
    def _broadcast_updates(self):
        """Broadcast accumulated weight changes to other devices using multicast."""
        if not self.socket:
            print("Socket not initialized, can't broadcast")
            return
        
        try:
            # Prepare update message
            accumulated = self.weight_changes.get()
            
            if not accumulated:
                print("No weight changes to broadcast")
                return
            
            # Calculate average weight changes
            count = self.training_count.get()
            averaged = {}
            
            for layer_name, changes in accumulated.items():
                averaged[layer_name] = {idx: change/count for idx, change in changes.items()}
            
            # Create unique ID for this update
            update_id = str(uuid.uuid4())
            
            # Create message
            message = {
                'device_id': self.device_id.get(),
                'update_id': update_id,
                'weight_changes': averaged,
                'timestamp': time.time()
            }
            
            # Add to known updates
            known = self.known_updates.get()
            known[update_id] = message
            
            # Keep known updates bounded
            if len(known) > 100:
                # Remove oldest items
                oldest = sorted(known.keys(), key=lambda k: known[k]['timestamp'])[:50]
                for key in oldest:
                    del known[key]
            
            self.known_updates.set(known)
            
            # Serialize message
            json_data = json.dumps(message).encode('utf-8')
            
            # Send via multicast
            self.socket.sendto(json_data, (self.multicast_group.get(), self.port.get()))
            
            print(f"Broadcasted weight updates (ID: {update_id[:8]}...): {len(json_data)} bytes")
            
            # Also direct-send to all known peers (for redundancy)
            for peer in self.peer_list.get():
                try:
                    self.socket.sendto(json_data, peer)
                except Exception as e:
                    print(f"Error sending to peer {peer}: {e}")
        
        except Exception as e:
            print(f"Error broadcasting updates: {e}")
    
    def process_update_ack(self, update_id, peer_address):
        """Process acknowledgment from a peer that they received our update."""
        # Add peer to our list if not already there
        self.add_peer(peer_address)
        
        print(f"Received ACK for update {update_id[:8]}... from {peer_address}")
    
    def stop(self):
        """Stop the broadcaster."""
        self.running.set(False)
        if self.socket:
            self.socket.close()
            self.socket = None


class WeightUpdateListener(AbstractProcess):
    """
    Process that listens for weight updates from other devices in the federated learning network.
    Uses multicast with acknowledgments to ensure reliable delivery.
    """
    
    def __init__(self, listen_port=DEFAULT_FEDERATED_PORT, device_id=None,
                 multicast_group=MULTICAST_GROUP):
        super().__init__()
        
        # Generate device ID if not provided
        if device_id is None:
            device_id = random.randint(0, 2**16 - 1)
        
        # Ports
        self.s_out = OutPort(shape=(2,))  # Output port for (device_id, weight_changes)
        
        # Variables
        self.device_id = Var(device_id)
        self.multicast_group = Var(multicast_group)
        self.port = Var(listen_port)
        self.running = Var(True)
        self.buffer_size = Var(65536)  # UDP buffer size
        
        # For deduplication
        self.received_updates = Var({})  # Track updates we've seen by their UUID
        
        # For sender tracking
        self.known_senders = Var({})  # Track senders we've seen
        
        # Start the listener thread
        self.listener_thread = threading.Thread(target=self._start_listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()
    
    def _start_listener(self):
        """Start listening for weight updates from other devices using multicast."""
        print(f"Starting weight update listener on port {self.port.get()}")
        
        try:
            # Create UDP socket for receiving multicasts
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to the port
            sock.bind(('', self.port.get()))
            
            # Join multicast group
            group = socket.inet_aton(self.multicast_group.get())
            mreq = struct.pack('4sL', group, socket.INADDR_ANY)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            
            print(f"Joined multicast group {self.multicast_group.get()}")
            
            while self.running.get():
                try:
                    # Receive data
                    data, addr = sock.recvfrom(self.buffer_size.get())
                    
                    # Parse message
                    message = json.loads(data.decode('utf-8'))
                    device_id = message.get('device_id', -1)
                    update_id = message.get('update_id', '')
                    weight_changes = message.get('weight_changes', {})
                    
                    # Skip our own messages
                    if device_id == self.device_id.get():
                        continue
                    
                    # Track sender
                    senders = self.known_senders.get()
                    if device_id not in senders:
                        senders[device_id] = addr
                        self.known_senders.set(senders)
                    
                    # Check if we've already seen this update
                    received = self.received_updates.get()
                    if update_id in received:
                        print(f"Ignoring duplicate update {update_id[:8]}... from device {device_id}")
                        continue
                    
                    # Mark as received
                    received[update_id] = time.time()
                    
                    # Keep received updates bounded
                    if len(received) > 200:
                        # Remove oldest items
                        oldest = sorted(received.items(), key=lambda x: x[1])[:100]
                        for key, _ in oldest:
                            del received[key]
                    
                    self.received_updates.set(received)
                    
                    print(f"Received weight update {update_id[:8]}... from device {device_id}")
                    
                    # Send acknowledgment back to sender
                    self._send_ack(update_id, addr)
                    
                    # Forward to processing
                    self.s_out.send((device_id, weight_changes))
                
                except socket.error as e:
                    print(f"Socket error: {e}")
                    time.sleep(1)
                
                except Exception as e:
                    print(f"Error in listener: {e}")
                    time.sleep(1)
        
        except Exception as e:
            print(f"Error starting listener: {e}")
        
        finally:
            if 'sock' in locals():
                sock.close()
            print("Weight update listener stopped")
    
    def _send_ack(self, update_id, sender_addr):
        """Send acknowledgment for received update."""
        try:
            # Create ACK socket (different from listener socket)
            ack_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # Create ACK message
            ack_message = {
                'type': 'ACK',
                'device_id': self.device_id.get(),
                'update_id': update_id,
                'timestamp': time.time()
            }
            
            # Serialize and send
            ack_data = json.dumps(ack_message).encode('utf-8')
            ack_socket.sendto(ack_data, sender_addr)
            
            # Close socket
            ack_socket.close()
            
        except Exception as e:
            print(f"Error sending ACK: {e}")
    
    def get_known_senders(self):
        """Get the list of known senders (for peer discovery)."""
        return list(self.known_senders.get().values())
    
    def stop(self):
        """Stop the listener."""
        self.running.set(False)


class FederatedSNNMetaProcess(AbstractProcess):
    """
    Meta-process that integrates the SNN classifier with federated learning capabilities.
    Uses reliable multicast for communication between devices.
    """
    
    def __init__(self, default_input_shape=(28, 28), 
                 listener_port=8080, broadcaster_port=8081, 
                 federated_port=DEFAULT_FEDERATED_PORT,
                 device_id=None):
        super().__init__()
        
        # Generate random device ID if not provided
        if device_id is None:
            device_id = random.randint(0, 2**16 - 1)
        
        # Variables
        self.device_id = Var(device_id)
        self.current_image_shape = Var(default_input_shape)
        
        # Create sub-processes
        self.image_listener = ImageListener(port=listener_port)
        self.snn_classifier = SNNImageProcessor(input_shape=default_input_shape)
        self.broadcaster = Broadcaster(port=broadcaster_port)
        
        # Federated learning components
        self.reservoir = ReservoirComputing(input_size=16)  # 16 bits for device ID
        self.weight_broadcaster = WeightUpdateBroadcaster(
            broadcast_port=federated_port,
            device_id=device_id
        )
        self.weight_listener = WeightUpdateListener(
            listen_port=federated_port,
            device_id=device_id
        )
        
        # Connect the processes with shape adaptation
        def adapt_and_forward_image(image):
            # Check if we need to handle a new image shape
            if image.shape != self.current_image_shape.get():
                print(f"Received new image shape: {image.shape}, adapting classifier")
                # Store the new shape
                self.current_image_shape.set(image.shape)
                
                # Recreate the classifier with the new shape
                self.snn_classifier = SNNImageProcessor(input_shape=image.shape)
                
                # Reconnect the classifier output
                self.snn_classifier.output_port.connect(process_snn_output)
            
            # Forward the image to the classifier
            self.snn_classifier.input_port.send(image)
            return image
        
        # Connect listener to adapter
        self.image_listener.s_out.connect(adapt_and_forward_image)
        
        # Function to process SNN output
        def process_snn_output(spikes):
            # Determine prediction from spikes
            prediction = np.argmax(np.sum(spikes, axis=0))
            
            # Send to broadcaster and get reward
            reward = self.broadcaster.process_prediction(prediction)
            
            # If we received a reward, use it for learning
            if reward is not None:
                # Apply reward to the classifier for R-STDP learning
                self.snn_classifier.output_rstdp.set_reward(reward)
                
                # Update performance metrics
                self.processed_images.set(self.processed_images.get() + 1)
                if reward > 0:
                    self.correct_predictions.set(self.correct_predictions.get() + 1)
                self.total_reward.set(self.total_reward.get() + reward)
                
                # Calculate current accuracy
                accuracy = self.correct_predictions.get() / max(1, self.processed_images.get())
                
                # Update reservoir's accuracy history
                self.reservoir.update_accuracy(accuracy)
                
                # Every N iterations, evaluate delayed rewards
                if self.processed_images.get() % 10 == 0:
                    self.reservoir.evaluate_delayed_rewards()
                
                # Get weight changes from R-STDP layer
                weight_changes = self.snn_classifier.output_rstdp.get_weight_changes()
                
                # Accumulate weight changes for federated learning
                self.weight_broadcaster.accumulate_weight_changes('output_rstdp', weight_changes)
            
            return prediction
        
        # Connect the SNN output to our processing function
        self.snn_classifier.output_port.connect(process_snn_output)
        
        # Function to process weight updates from other devices
        def process_weight_updates(data):
            device_id, weight_changes = data
            
            # Convert device ID to binary spikes
            device_id_binary = self._device_id_to_binary(device_id)
            
            # Process through reservoir to decide whether to accept update
            decision = self.reservoir.process_input(device_id_binary)
            
            if decision == 1:  # Accept the update
                print(f"Accepting weight update from device {device_id}")
                
                # Apply weight changes to the classifier
                for layer_name, changes in weight_changes.items():
                    if layer_name == 'output_rstdp':
                        self.snn_classifier.output_rstdp.apply_weight_changes(changes)
            else:
                print(f"Rejecting weight update from device {device_id}")
            
            return decision
        
        # Connect weight listener to process function
        self.weight_listener.s_out.connect(process_weight_updates)
        
        # Periodically sync peers between broadcaster and listener
        self.peer_sync_thread = threading.Thread(target=self._sync_peers_periodically)
        self.peer_sync_thread.daemon = True
        self.peer_sync_thread.start()
        
        # Performance tracking
        self.processed_images = Var(0)
        self.correct_predictions = Var(0)
        self.total_reward = Var(0.0)
    
    def _device_id_to_binary(self, device_id):
        """Convert device ID to binary representation for reservoir input."""
        # Convert to 16-bit binary
        binary = [(device_id >> i) & 1 for i in range(16)]
        return np.array(binary, dtype=np.float32)
    
    def _sync_peers_periodically(self):
        """Periodically sync peer lists between broadcaster and listener."""
        while True:
            try:
                # Get known senders from listener
                peers = self.weight_listener.get_known_senders()
                
                # Update peer list in broadcaster
                for peer in peers:
                    self.weight_broadcaster.add_peer(peer)
                
                # Wait before next sync
                time.sleep(60)  # Sync every minute
            
            except Exception as e:
                print(f"Error syncing peers: {e}")
                time.sleep(10)  # Retry after error
    
    def run(self, duration=None):
        """
        Run the meta-process for a specified duration.
        
        Parameters:
        - duration: Duration in seconds, or None to run indefinitely
        """
        start_time = time.time()
        running = True
        
        print(f"Federated SNN Meta-process running for device {self.device_id.get()}"
              f"{f' for {duration} seconds' if duration else ''}")
        
        try:
            while running:
                time.sleep(0.1)  # Main thread just monitors
                
                # Check if duration has elapsed
                if duration and (time.time() - start_time) >= duration:
                    running = False
                
                # Report performance occasionally
                if self.processed_images.get() % 100 == 0 and self.processed_images.get() > 0:
                    accuracy = self.correct_predictions.get() / self.processed_images.get()
                    avg_reward = self.total_reward.get() / self.processed_images.get()
                    
                    # Get reservoir performance
                    rc_decisions = self.reservoir.total_decisions.get()
                    rc_good = self.reservoir.good_decisions.get()
                    rc_accuracy = rc_good / max(1, rc_decisions)
                    
                    print(f"Device {self.device_id.get()} - "
                          f"Processed {self.processed_images.get()} images. "
                          f"Accuracy: {accuracy:.4f}, Avg Reward: {avg_reward:.4f}, "
                          f"RC Accuracy: {rc_accuracy:.4f} ({rc_good}/{rc_decisions})")
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            # Clean up
            self.stop()
            print("Meta-process stopped")
    
    def stop(self):
        """Stop all sub-processes."""
        self.image_listener.stop()
        self.broadcaster.stop()
        self.weight_broadcaster.stop()
        self.weight_listener.stop()


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated SNN Meta-Process")
    parser.add_argument("--device-id", type=int, default=None, 
                        help="Device ID (random if not specified)")
    parser.add_argument("--image-port", type=int, default=8080, 
                        help="Port for receiving images")
    parser.add_argument("--prediction-port", type=int, default=8081, 
                        help="Port for sending predictions")
    parser.add_argument("--federated-port", type=int, default=DEFAULT_FEDERATED_PORT, 
                        help="Port for federated learning communication")
    parser.add_argument("--duration", type=int, default=None, 
                        help="Duration to run in seconds (indefinite if not specified)")
    
    args = parser.parse_args()
    
    # Create and run the meta-process
    meta_process = FederatedSNNMetaProcess(
        default_input_shape=(28, 28),
        listener_port=args.image_port,
        broadcaster_port=args.prediction_port,
        federated_port=args.federated_port,
        device_id=args.device_id
    )
    
    # Run for specified duration or indefinitely
    meta_process.run(duration=args.duration)