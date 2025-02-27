import socket
import struct
import pickle
import json
import threading
import time
import numpy as np
import argparse
import os
from datetime import datetime
from PIL import Image


class SNNTrainingServer:
    """
    Server for sending images to an SNN processor and receiving predictions.
    The server provides rewards based on the accuracy of predictions.
    """
    
    def __init__(self, image_dir, labels_file, image_port=8080, prediction_port=8081):
        """
        Initialize the server.
        
        Parameters:
        - image_dir: Directory containing image files to send
        - labels_file: File containing labels for the images
        - image_port: Port for sending images
        - prediction_port: Port for receiving predictions
        """
        self.image_dir = image_dir
        self.labels_file = labels_file
        self.image_port = image_port
        self.prediction_port = prediction_port
        
        # Load image filenames and labels
        self.images, self.labels = self._load_dataset()
        
        # Current image index
        self.current_idx = 0
        
        # Server status
        self.running = False
        
        # Connections
        self.image_socket = None
        self.image_client = None
        self.prediction_socket = None
        self.prediction_client = None
        
        # Threads
        self.image_thread = None
        self.prediction_thread = None
        
        # Performance tracking
        self.num_images_sent = 0
        self.num_correct_predictions = 0
        self.performance_log = []
        
        # Create log directory
        os.makedirs("logs", exist_ok=True)
    
    def _load_dataset(self):
        """Load image filenames and labels."""
        print(f"Loading dataset from {self.image_dir} with labels from {self.labels_file}")
        
        # Load labels from file
        labels = {}
        with open(self.labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    label = int(parts[1])
                    labels[filename] = label
        
        # Get image files
        image_files = []
        for filename in os.listdir(self.image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(filename)
        
        # Match images with labels
        matched_images = []
        matched_labels = []
        
        for filename in image_files:
            if filename in labels:
                matched_images.append(os.path.join(self.image_dir, filename))
                matched_labels.append(labels[filename])
        
        print(f"Loaded {len(matched_images)} images with labels")
        return matched_images, matched_labels
    
    def start(self):
        """Start the server."""
        if self.running:
            print("Server is already running")
            return
        
        self.running = True
        
        # Start image sender thread
        self.image_thread = threading.Thread(target=self._run_image_sender)
        self.image_thread.daemon = True
        self.image_thread.start()
        
        # Start prediction receiver thread
        self.prediction_thread = threading.Thread(target=self._run_prediction_receiver)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
        
        print("Server started")
    
    def _run_image_sender(self):
        """Run the image sender service."""
        print(f"Starting image sender on port {self.image_port}")
        
        try:
            # Create socket
            self.image_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.image_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.image_socket.bind(('0.0.0.0', self.image_port))
            self.image_socket.listen(1)
            
            print("Waiting for image client connection...")
            
            while self.running:
                # Accept connection
                self.image_client, addr = self.image_socket.accept()
                print(f"Image client connected from {addr}")
                
                try:
                    # Send images continuously
                    while self.running:
                        if self.current_idx >= len(self.images):
                            self.current_idx = 0  # Restart from beginning
                            print("Dataset completed, restarting from beginning")
                        
                        # Get the next image and label
                        image_path = self.images[self.current_idx]
                        label = self.labels[self.current_idx]
                        
                        # Load and preprocess the image
                        try:
                            image = self._load_and_preprocess_image(image_path)
                            
                            # Create metadata
                            metadata = {
                                'filename': os.path.basename(image_path),
                                'shape': list(image.shape),
                                'label': label,
                                'timestamp': time.time()
                            }
                            
                            # Send metadata
                            meta_json = json.dumps(metadata).encode('utf-8')
                            meta_size = struct.pack('!I', len(meta_json))
                            self.image_client.sendall(meta_size)
                            self.image_client.sendall(meta_json)
                            
                            # Send image data
                            image_data = pickle.dumps(image)
                            img_size = struct.pack('!I', len(image_data))
                            self.image_client.sendall(img_size)
                            self.image_client.sendall(image_data)
                            
                            # Update counters
                            self.num_images_sent += 1
                            self.current_idx += 1
                            
                            print(f"Sent image {self.num_images_sent}: {os.path.basename(image_path)}, label: {label}")
                            
                            # Wait a bit before sending the next image
                            time.sleep(0.5)  # Adjust as needed
                        
                        except Exception as e:
                            print(f"Error processing image {image_path}: {e}")
                            self.current_idx += 1  # Skip this image
                
                except socket.error as e:
                    print(f"Image client socket error: {e}")
                    if self.image_client:
                        self.image_client.close()
                        self.image_client = None
        
        except Exception as e:
            print(f"Error in image sender: {e}")
        
        finally:
            # Clean up
            if self.image_client:
                self.image_client.close()
            if self.image_socket:
                self.image_socket.close()
            print("Image sender stopped")
    
    def _run_prediction_receiver(self):
        """Run the prediction receiver service."""
        print(f"Starting prediction receiver on port {self.prediction_port}")
        
        try:
            # Create socket
            self.prediction_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.prediction_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.prediction_socket.bind(('0.0.0.0', self.prediction_port))
            self.prediction_socket.listen(1)
            
            print("Waiting for prediction client connection...")
            
            while self.running:
                # Accept connection
                self.prediction_client, addr = self.prediction_socket.accept()
                print(f"Prediction client connected from {addr}")
                
                try:
                    # Receive predictions continuously
                    while self.running:
                        # Receive prediction size
                        size_data = self.prediction_client.recv(4)
                        if not size_data:
                            break
                        
                        size = struct.unpack('!I', size_data)[0]
                        
                        # Receive prediction data
                        data = b''
                        while len(data) < size:
                            chunk = self.prediction_client.recv(min(1024, size - len(data)))
                            if not chunk:
                                break
                            data += chunk
                        
                        if len(data) == size:
                            # Parse prediction
                            pred_data = json.loads(data.decode('utf-8'))
                            prediction = pred_data.get('prediction', -1)
                            
                            # Determine the correct label for the most recently sent image
                            # Note: This assumes the prediction is for the most recently sent image
                            correct_label = self.labels[(self.current_idx - 1) % len(self.labels)]
                            
                            # Calculate reward
                            is_correct = prediction == correct_label
                            reward = 1.0 if is_correct else -0.1
                            
                            # Update performance
                            if is_correct:
                                self.num_correct_predictions += 1
                            
                            # Log performance
                            accuracy = self.num_correct_predictions / max(1, self.num_images_sent)
                            log_entry = {
                                'timestamp': time.time(),
                                'image_idx': (self.current_idx - 1) % len(self.images),
                                'true_label': correct_label,
                                'prediction': prediction,
                                'correct': is_correct,
                                'reward': reward,
                                'accuracy': accuracy
                            }
                            self.performance_log.append(log_entry)
                            
                            # Print status
                            print(f"Prediction: {prediction}, True: {correct_label}, "
                                  f"Correct: {is_correct}, Reward: {reward}, "
                                  f"Accuracy: {accuracy:.4f}")
                            
                            # Send reward
                            response = {
                                'reward': reward,
                                'correct': is_correct,
                                'true_label': correct_label
                            }
                            resp_json = json.dumps(response).encode('utf-8')
                            resp_size = struct.pack('!I', len(resp_json))
                            self.prediction_client.sendall(resp_size)
                            self.prediction_client.sendall(resp_json)
                            
                            # Save log periodically
                            if len(self.performance_log) % 100 == 0:
                                self._save_performance_log()
                
                except socket.error as e:
                    print(f"Prediction client socket error: {e}")
                    if self.prediction_client:
                        self.prediction_client.close()
                        self.prediction_client = None
        
        except Exception as e:
            print(f"Error in prediction receiver: {e}")
        
        finally:
            # Clean up
            if self.prediction_client:
                self.prediction_client.close()
            if self.prediction_socket:
                self.prediction_socket.close()
            print("Prediction receiver stopped")
    
    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess an image."""
        # Load image
        image = Image.open(image_path)
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 28x28 (common for MNIST-like tasks)
        image = image.resize((28, 28))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    
    def _save_performance_log(self):
        """Save the performance log to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/performance_{timestamp}.json"
        
        with open(log_file, 'w') as f:
            json.dump(self.performance_log, f, indent=2)
        
        print(f"Saved performance log to {log_file}")
    
    def stop(self):
        """Stop the server."""
        if not self.running:
            print("Server is not running")
            return
        
        self.running = False
        
        # Close connections
        if self.image_client:
            self.image_client.close()
        if self.image_socket:
            self.image_socket.close()
        
        if self.prediction_client:
            self.prediction_client.close()
        if self.prediction_socket:
            self.prediction_socket.close()
        
        # Wait for threads to finish
        if self.image_thread and self.image_thread.is_alive():
            self.image_thread.join(timeout=2)
        if self.prediction_thread and self.prediction_thread.is_alive():
            self.prediction_thread.join(timeout=2)
        
        # Save final performance log
        self._save_performance_log()
        
        print("Server stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Training Server")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--labels-file", type=str, required=True, help="File containing image labels")
    parser.add_argument("--image-port", type=int, default=8080, help="Port for sending images")
    parser.add_argument("--prediction-port", type=int, default=8081, help="Port for receiving predictions")
    args = parser.parse_args()
    
    server = SNNTrainingServer(
        image_dir=args.image_dir,
        labels_file=args.labels_file,
        image_port=args.image_port,
        prediction_port=args.prediction_port
    )
    
    try:
        server.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        server.stop()