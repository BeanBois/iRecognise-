from lava.magma.core.run_conditions import LoihiRunConditions
from lava.proc.io import SpikeGenerator

# Compile and run
run_condition = LoihiRunConditions(num_steps=100)
network.compile(SpikeGenerator(data), run_condition)
network.run()