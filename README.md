# PySMOT- Python Slime Mold Optimization Tool

PySMOT (Python Slime Mold Optimization Tool) is a simulation tool inspired by the behavior of slime mold for solving optimization problems. It simulates a collection of agents that deposit pheromones and move based on the local pheromone concentration and attraction to specific destinations. The tool can be used to simulate how slime mold can be used to optimize networks or paths, such as public transit systems, through agent-based models.

## Key Features
- Agent-based Simulation: Simulate the behavior of multiple agents (slime mold) interacting with their environment.
- Pheromone Deposition: Agents leave pheromone trails that influence other agents' movements.
- Destination Attraction: Agents are attracted to specific destinations, which can be used to simulate optimization problems.
- Rendering: Visualize the simulation with options to display agents, pheromones, and network paths.
- Video Output: Render the simulation as a video.
- Modular Components: Customize the simulation with additional components.

## Installation
Ensure you have Python 3.x and the required dependencies installed. Install the necessary packages using pip:

```bash
pip install numpy opencv-python matplotlib pillow
```

## Usage
To run the simulation, you can modify the configuration options in the SimulationConfig and RenderingConfig classes and execute the script. Here is a simple example to get you started:

```python
from PySMOT import SimulationConfig, RenderingConfig, AgentSystem
import numpy as np

# Set up the simulation configuration
sim_config = SimulationConfig()

# Define the rendering options
render_config = RenderingConfig(render_network=True)

# Define the network and agent weights
network = np.array([
    [sim_config.size_y // 2, sim_config.size_x // 2],
    [sim_config.size_y // 2 + 50, sim_config.size_x // 2 + 30],
    [sim_config.size_y // 2 - 50, sim_config.size_x // 2 - 30]
], dtype=np.int32)

weights = np.array([0.5, 0.4, 0.4])

# Initialize the system
system = AgentSystem(sim_config, render_config, network=network, weights=weights)

# Render the simulation video
system.render_video(r"simulation_output", video_duration=5, fps=30)
```

## Configuration Parameters

### `SimulationConfig`: Controls the simulation environment

|Parameter|Description|Range|Default|
|---------|-----------|-----|-------|
|`size_x`, `size_y`|Grid dimensions||1920x1080|
|`num_agents`|Number of agents to simulate||10,000|
|`spawn_radius`|Radius from the center to spawn agents|<`size_x,size_y`|300|
|`speed`|Agent speed (in pixels)||3|
|`lookahead`|Agent lookahead distance||5|
|`pheromone_strength`|Strength of pheromone influence|(0,1]|0.1|
|`destination_strength`|Strength of destination attraction|(0,1]|0.3|
|`pheromone_decay`|How quickly pheromones decay|(0,1]|0.02|
|`diffusion_strength`|The diffusion of pheromones|(0,1]|0.8|

### `RenderingConfig`: Controls the visualization of the simulation.

|Parameter|Description|Expected Input|Default|
|---------|-----------|-----|-------|
|`render_agents`, `render_pheromones`, `render_network`|Controls rendering of agents, pheromones, and network|Boolean|`True` for agents and pheromones, `False` for network|
|`agent_color`, `network_color`|Colors for agents and network|```(int, int int)```|```(255, 255, 255)``` (White)|
|`pheromone_color_map`|Colormap for pheromones|matplotlib cmap|```'plasma'```|

## Methods
- `render_frame(make_img=False)`: Renders a single frame of the simulation
- `render_video(file_output, video_duration, fps=30)`: Renders the simulation as a video and saves it to a specified file
- `update()`: Updates the simulation state by applying pheromone influence, destination attraction, and agent movements

## Example Component
To add custom behavior to the simulation, create a new SimulationComponent subclass and attach it to the system. Here is a simple example:

```python
class ExampleComponent(SimulationComponent):
    def __init__(self, system):
        super().__init__(system)

    def initialize(self):
        pass

    def update(self):
        pass

    def finalize(self):
        pass
```

## Final Notes
PySMOT is designed to be extensible and customizable. You can easily integrate your own logic and tweak the simulation parameters to fit your optimization problem.