import os
import numpy as np
import typing as ty

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort



class SpikeInput(AbstractProcess):
    """Reads image data from the MNIST dataset and converts it to spikes.
    The resulting spike rate is proportional to the pixel value."""

    def __init__(self,
                 vth: int,
                 num_images: ty.Optional[int] = 25,
                 num_steps_per_image: ty.Optional[int] = 128):
        super().__init__()
        shape = (784,)
        self.spikes_out = OutPort(shape=shape)  # Input spikes to the classifier
        self.label_out = OutPort(shape=(1,))  # Ground truth labels to OutputProc
        self.num_images = Var(shape=(1,), init=num_images)
        self.num_steps_per_image = Var(shape=(1,), init=num_steps_per_image)
        self.input_img = Var(shape=shape)
        self.ground_truth_label = Var(shape=(1,))
        self.v = Var(shape=shape, init=0)
        self.vth = Var(shape=(1,), init=vth)
        
        
class ImageClassifier(AbstractProcess):
    """A 3 layer feed-forward network with LIF and Dense Processes."""

    def __init__(self, trained_weights_path: str):
        super().__init__()
        
        # Using pre-trained weights and biases
        real_path_trained_wgts = os.path.realpath(trained_weights_path)

        wb_list = np.load(real_path_trained_wgts, encoding='latin1', allow_pickle=True)
        w0 = wb_list[0].transpose().astype(np.int32)
        w1 = wb_list[2].transpose().astype(np.int32)
        w2 = wb_list[4].transpose().astype(np.int32)
        b1 = wb_list[1].astype(np.int32)
        b2 = wb_list[3].astype(np.int32)
        b3 = wb_list[5].astype(np.int32)

        self.spikes_in = InPort(shape=(w0.shape[1],))
        self.spikes_out = OutPort(shape=(w2.shape[0],))
        self.w_dense0 = Var(shape=w0.shape, init=w0)
        self.b_lif1 = Var(shape=(w0.shape[0],), init=b1)
        self.w_dense1 = Var(shape=w1.shape, init=w1)
        self.b_lif2 = Var(shape=(w1.shape[0],), init=b2)
        self.w_dense2 = Var(shape=w2.shape, init=w2)
        self.b_output_lif = Var(shape=(w2.shape[0],), init=b3)
        
        # Up-level currents and voltages of LIF Processes
        # for resetting (see at the end of the tutorial)
        self.lif1_u = Var(shape=(w0.shape[0],), init=0)
        self.lif1_v = Var(shape=(w0.shape[0],), init=0)
        self.lif2_u = Var(shape=(w1.shape[0],), init=0)
        self.lif2_v = Var(shape=(w1.shape[0],), init=0)
        self.oplif_u = Var(shape=(w2.shape[0],), init=0)
        self.oplif_v = Var(shape=(w2.shape[0],), init=0)
        
        
class OutputProcess(AbstractProcess):
    """Process to gather spikes from 10 output LIF neurons and interpret the
    highest spiking rate as the classifier output"""

    def __init__(self, **kwargs):
        super().__init__()
        shape = (10,)
        n_img = kwargs.pop('num_images', 25)
        self.num_images = Var(shape=(1,), init=n_img)
        self.spikes_in = InPort(shape=shape)
        self.label_in = InPort(shape=(1,))
        self.spikes_accum = Var(shape=shape)  # Accumulated spikes for classification
        self.num_steps_per_image = Var(shape=(1,), init=128)
        self.pred_labels = Var(shape=(n_img,))
        self.gt_labels = Var(shape=(n_img,))