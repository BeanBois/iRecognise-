from utils import *

# Import parent classes for ProcessModels
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.model import PyLoihiProcessModel

# Import ProcessModel ports, data-types
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

# Import execution protocol and hardware resources
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU

# Import decorators
from lava.magma.core.decorator import implements, requires

# Import MNIST dataset
from lava.utils.dataloader.mnist import MnistDataset
np.set_printoptions(linewidth=np.inf)

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense 

@implements(proc=SpikeInput, protocol=LoihiProtocol)
@requires(CPU)
class PySpikeInputModel(PyLoihiProcessModel):
    num_images: int = LavaPyType(int, int, precision=32)
    spikes_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    label_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32,
                                      precision=32)
    num_steps_per_image: int = LavaPyType(int, int, precision=32)
    input_img: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    ground_truth_label: int = LavaPyType(int, int, precision=32)
    v: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    vth: int = LavaPyType(int, int, precision=32)
    
    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.mnist_dataset = MnistDataset()
        self.curr_img_id = 0

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        if self.time_step % self.num_steps_per_image == 1:
            return True
        return False

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above 
        returns True.
        """
        img = self.mnist_dataset.images[self.curr_img_id]
        self.ground_truth_label = self.mnist_dataset.labels[self.curr_img_id]
        self.input_img = img.astype(np.int32) - 127
        self.v = np.zeros(self.v.shape)
        self.label_out.send(np.array([self.ground_truth_label]))
        self.curr_img_id += 1

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
        self.v[:] = self.v + self.input_img
        s_out = self.v > self.vth
        self.v[s_out] = 0  # reset voltage to 0 after a spike
        self.spikes_out.send(s_out)
        
           

@implements(ImageClassifier)
@requires(CPU)
class PyImageClassifierModel(AbstractSubProcessModel):
    def __init__(self, proc):
        self.dense0 = Dense(weights=proc.w_dense0.init)
        self.lif1 = LIF(shape=(64,), bias_mant=proc.b_lif1.init, vth=400,
                        dv=0, du=4095)
        self.dense1 = Dense(weights=proc.w_dense1.init)
        self.lif2 = LIF(shape=(64,), bias_mant=proc.b_lif2.init, vth=350,
                        dv=0, du=4095)
        self.dense2 = Dense(weights=proc.w_dense2.init)
        self.output_lif = LIF(shape=(10,), bias_mant=proc.b_output_lif.init,
                              vth=1, dv=0, du=4095)

        proc.spikes_in.connect(self.dense0.s_in)
        self.dense0.a_out.connect(self.lif1.a_in)
        self.lif1.s_out.connect(self.dense1.s_in)
        self.dense1.a_out.connect(self.lif2.a_in)
        self.lif2.s_out.connect(self.dense2.s_in)
        self.dense2.a_out.connect(self.output_lif.a_in)
        self.output_lif.s_out.connect(proc.spikes_out)
        
        # Create aliases of SubProcess variables
        proc.lif1_u.alias(self.lif1.u)
        proc.lif1_v.alias(self.lif1.v)
        proc.lif2_u.alias(self.lif2.u)
        proc.lif2_v.alias(self.lif2.v)
        proc.oplif_u.alias(self.output_lif.u)
        proc.oplif_v.alias(self.output_lif.v)
        
@implements(proc=OutputProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputProcessModel(PyLoihiProcessModel):
    label_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)
    spikes_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    num_images: int = LavaPyType(int, int, precision=32)
    spikes_accum: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)
    num_steps_per_image: int = LavaPyType(int, int, precision=32)
    pred_labels: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    gt_labels: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
        
    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.current_img_id = 0

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        if self.time_step % self.num_steps_per_image == 0 and \
                self.time_step > 1:
            return True
        return False

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above 
        returns True.
        """
        gt_label = self.label_in.recv()
        pred_label = np.argmax(self.spikes_accum)
        self.gt_labels[self.current_img_id] = gt_label
        self.pred_labels[self.current_img_id] = pred_label
        self.current_img_id += 1
        self.spikes_accum = np.zeros_like(self.spikes_accum)

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
        spk_in = self.spikes_in.recv()
        self.spikes_accum = self.spikes_accum + spk_in
        
if __name__ == "__main__":
    num_images = 25
    num_steps_per_image = 128

    # Create Process instances
    spike_input = SpikeInput(vth=1,
                            num_images=num_images,
                            num_steps_per_image=num_steps_per_image)
    mnist_clf = ImageClassifier(
        trained_weights_path=os.path.join('.', 'mnist_pretrained.npy'))
    output_proc = OutputProcess(num_images=num_images)

    # Connect Processes
    spike_input.spikes_out.connect(mnist_clf.spikes_in)
    mnist_clf.spikes_out.connect(output_proc.spikes_in)
    
    # Connect Input directly to Output for ground truth labels
    spike_input.label_out.connect(output_proc.label_in)
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi1SimCfg

    # Loop over all images
    for img_id in range(num_images):
        print(f"\rCurrent image: {img_id+1}", end="")
        
        # Run each image-inference for fixed number of steps
        mnist_clf.run(
            condition=RunSteps(num_steps=num_steps_per_image),
            run_cfg=Loihi1SimCfg(select_sub_proc_model=True,
                                select_tag='fixed_pt'))
        
        # Reset internal neural state of LIF neurons
        mnist_clf.lif1_u.set(np.zeros((64,), dtype=np.int32))
        mnist_clf.lif1_v.set(np.zeros((64,), dtype=np.int32))
        mnist_clf.lif2_u.set(np.zeros((64,), dtype=np.int32))
        mnist_clf.lif2_v.set(np.zeros((64,), dtype=np.int32))
        mnist_clf.oplif_u.set(np.zeros((10,), dtype=np.int32))
        mnist_clf.oplif_v.set(np.zeros((10,), dtype=np.int32))

    # Gather ground truth and predictions before stopping exec
    ground_truth = output_proc.gt_labels.get().astype(np.int32)
    predictions = output_proc.pred_labels.get().astype(np.int32)

    # Stop the execution
    mnist_clf.stop()

    accuracy = np.sum(ground_truth==predictions)/ground_truth.size * 100

    print(f"\nGround truth: {ground_truth}\n"
        f"Predictions : {predictions}\n"
        f"Accuracy    : {accuracy}")