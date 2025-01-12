from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from dataclasses import dataclass
from typing import Optional, Tuple
import os

# Constants
EPSILON = 1e-2  # Small value to prevent division by zero

@dataclass
class SimulationConfig:
    size_x: int = 1920
    size_y: int = 1080

    num_agents: int = 10_000
    spawn_radius: int = 300

    speed: float = 3
    lookahead: float = 5
    lookahead_angle: float = np.pi / 4

    agent_deposit_intensity: float = 0.8
    pheromone_strength: float = 0.3
    destination_strength: float = 0.1
    destination_tolerance: float = 2.5
    pheromone_decay: float = 0.02
    diffusion_strength: float = 0.8

    def validate(self):
        if self.size_x <= 0 or self.size_y <= 0:
            raise ValueError("Grid size must be positive.")
        if self.num_agents <= 0:
            raise ValueError("Number of agents must be positive.")
        if self.speed <= 0:
            raise ValueError("Agent speed must be positive.")
        if not (0 < self.diffusion_strength <= 1):
            raise ValueError("Diffusion strength must be between 0 and 1.")
        if not (0 <= self.pheromone_decay <= 1):
            raise ValueError("Pheromone decay must be between 0 and 1.")

@dataclass
class RenderingConfig:
    render_agents: bool = True
    render_pheromones: bool = True
    render_network: bool = False

    agent_color: Tuple[int, int, int] = (255, 255, 255)
    network_color: Tuple[int, int, int] = (255, 255, 255)
    pheromone_color_map: str = 'plasma'

class AgentSystem:
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: Optional[RenderingConfig] = None,
        network: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None
    ):
        sim_config.validate()
        self.sim_config = sim_config
        self.render_config = render_config or RenderingConfig()
        self.network = network if network is not None else np.array([])

        if weights is not None:
            if len(weights) != len(self.network):
                raise ValueError("Weights must match the size of the network.")
            self.weights = weights / np.sum(weights)
        else:
            self.weights = np.ones(len(self.network)) / len(self.network) if self.network.size else np.array([])

        self.agents = self._initialize_agents()
        self.pheromone_map = np.zeros((self.sim_config.size_y, self.sim_config.size_x), dtype=np.float32)
        self.components = []

    def add_component(self, component):
        component.initialize()
        self.components.append(component)

    def _initialize_agents(self) -> np.ndarray:
        agents = np.empty((self.sim_config.num_agents, 6), dtype=np.float32)
        spawn_x = np.random.uniform(
            self.sim_config.size_x / 2 - self.sim_config.spawn_radius,
            self.sim_config.size_x / 2 + self.sim_config.spawn_radius,
            self.sim_config.num_agents
        )
        spawn_y = np.random.uniform(
            self.sim_config.size_y / 2 - self.sim_config.spawn_radius,
            self.sim_config.size_y / 2 + self.sim_config.spawn_radius,
            self.sim_config.num_agents
        )

        orientations = np.random.uniform(0, 2 * np.pi, self.sim_config.num_agents)
        velocities = np.column_stack((
            np.cos(orientations) * self.sim_config.speed,
            np.sin(orientations) * self.sim_config.speed
        ))

        destinations = (
            self.network[np.random.choice(len(self.network), size=self.sim_config.num_agents, p=self.weights)]
            if self.network.size != 0
            else np.full((self.sim_config.num_agents, 2), np.nan)
        )

        agents[:, 0] = spawn_y
        agents[:, 1] = spawn_x
        agents[:, 2:4] = velocities
        agents[:, 4:6] = destinations

        return agents

    def update(self):
        self._apply_pheromone_influence()
        self._apply_destination_attraction()
        self._update_positions()
        self._update_pheromone_map()
        self._update_destinations()

        for component in self.components:
            component.update()

    def _apply_pheromone_influence(self):
        forward_vector = self.agents[:, 2:4] * self.sim_config.lookahead
        right_vector, left_vector = self._compute_sensory_vectors(forward_vector)

        forward_strength = self._sample_pheromone_strength(self.agents[:, :2] + forward_vector)
        right_strength = self._sample_pheromone_strength(self.agents[:, :2] + right_vector)
        left_strength = self._sample_pheromone_strength(self.agents[:, :2] + left_vector)

        turn_influence = (
            forward_strength[:, np.newaxis] * forward_vector +
            right_strength[:, np.newaxis] * right_vector +
            left_strength[:, np.newaxis] * left_vector
        )

        direction_adjustment = turn_influence * self.sim_config.pheromone_strength
        self._update_velocities(direction_adjustment)

    def _compute_sensory_vectors(self, forward_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cos_angle, sin_angle = np.cos(self.sim_config.lookahead_angle), np.sin(self.sim_config.lookahead_angle)
        rotation_matrix_right = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        rotation_matrix_left = np.array([[cos_angle, sin_angle], [-sin_angle, cos_angle]])

        return forward_vector @ rotation_matrix_right.T, forward_vector @ rotation_matrix_left.T

    def _sample_pheromone_strength(self, positions: np.ndarray) -> np.ndarray:
        valid_mask = self._in_bounds(positions)
        valid_positions = positions[valid_mask].astype(int)

        strengths = np.zeros(len(positions), dtype=np.float32)
        strengths[valid_mask] = self.pheromone_map[valid_positions[:, 0], valid_positions[:, 1]]
        
        return strengths

    def _apply_destination_attraction(self):
        if self.network.size == 0:
            return

        direction_to_destination = self.agents[:, 4:6] - self.agents[:, :2]
        distances = np.linalg.norm(direction_to_destination, axis=1, keepdims=True)
        normalized_direction = direction_to_destination / np.maximum(distances, EPSILON)
        destination_attraction = normalized_direction * self.sim_config.destination_strength
        self._update_velocities(destination_attraction)

    def _update_destinations(self):
        if self.network.size == 0:
            return

        distances = np.linalg.norm(self.agents[:, 4:6] - self.agents[:, :2], axis=1)
        arrived_agents = distances < self.sim_config.destination_tolerance

        if np.any(arrived_agents):
            new_destinations = self.network[
                np.random.choice(len(self.network), size=np.sum(arrived_agents), p=self.weights)
            ]
            self.agents[arrived_agents, 4:6] = new_destinations

    def _in_bounds(self, positions: np.ndarray) -> np.ndarray:
        return (
            (0 <= positions[:, 0]) & (positions[:, 0] < self.sim_config.size_y) &
            (0 <= positions[:, 1]) & (positions[:, 1] < self.sim_config.size_x)
        )

    def _update_velocities(self, direction_adjustment: np.ndarray):
        new_velocity = self.agents[:, 2:4] + direction_adjustment
        speed_magnitude = np.linalg.norm(new_velocity, axis=1, keepdims=True)
        self.agents[:, 2:4] = (new_velocity / speed_magnitude) * self.sim_config.speed

    def _update_positions(self):
        self.agents[:, :2] += self.agents[:, 2:4]
        
        out_of_bounds_x = (self.agents[:, 0] < 0) | (self.agents[:, 0] >= self.sim_config.size_y)
        out_of_bounds_y = (self.agents[:, 1] < 0) | (self.agents[:, 1] >= self.sim_config.size_x)
        
        self.agents[out_of_bounds_x, 2] *= -1
        self.agents[out_of_bounds_y, 3] *= -1

    def _update_pheromone_map(self):
        self.pheromone_map *= (1 - self.sim_config.pheromone_decay)
        kernel = self._create_diffusion_kernel(self.sim_config.diffusion_strength)
        self.pheromone_map = cv2.filter2D(self.pheromone_map, -1, kernel)
        self.pheromone_map = np.clip(self.pheromone_map, 0, 1)
        self._deposit_agent_pheromones()

    @staticmethod
    def _create_diffusion_kernel(diff_strength: float) -> np.ndarray:
        outer_weight = (1 - diff_strength) / 8
        center_weight = diff_strength
        return np.array([
            [outer_weight, outer_weight, outer_weight],
            [outer_weight, center_weight, outer_weight],
            [outer_weight, outer_weight, outer_weight]
        ])

    def _deposit_agent_pheromones(self):
        x0 = np.clip(self.agents[:, 0].astype(int), 0, self.sim_config.size_y - 1)
        y0 = np.clip(self.agents[:, 1].astype(int), 0, self.sim_config.size_x - 1)
        dx, dy = self.agents[:, 0] - x0, self.agents[:, 1] - y0

        x1, y1 = np.clip(x0 + 1, 0, self.sim_config.size_y - 1), np.clip(y0 + 1, 0, self.sim_config.size_x - 1)

        np.maximum.at(self.pheromone_map, (x0, y0), (1 - dx) * (1 - dy) * self.sim_config.agent_deposit_intensity)
        np.maximum.at(self.pheromone_map, (x1, y0), dx * (1 - dy) * self.sim_config.agent_deposit_intensity)
        np.maximum.at(self.pheromone_map, (x0, y1), (1 - dx) * dy * self.sim_config.agent_deposit_intensity)
        np.maximum.at(self.pheromone_map, (x1, y1), dx * dy * self.sim_config.agent_deposit_intensity)

    def render_frame(self, make_img: bool = False):
        img = np.zeros((self.sim_config.size_y, self.sim_config.size_x, 3), dtype=np.uint8)
        norm = Normalize(vmin=0, vmax=1)

        if self.render_config.render_pheromones:
            pheromone_colormap = plt.get_cmap(self.render_config.pheromone_color_map)
            pheromone_colors = (pheromone_colormap(norm(self.pheromone_map))[:, :, :3] * 255).astype(np.uint8)
            mask = self.pheromone_map > EPSILON
            img[mask] = pheromone_colors[mask]

        if self.render_config.render_network and self.network.size > 0:
            for i in network:
                cv2.circle(img, (i[1], i[0]), 5, self.render_config.network_color, -1)

        if self.render_config.render_agents:
            agent_positions = self.agents[:, :2].astype(np.int32)
            valid_positions = self._in_bounds(agent_positions)
            img[agent_positions[valid_positions, 0], agent_positions[valid_positions, 1]] = self.render_config.agent_color

        return Image.fromarray(img) if make_img else img

    def render_video(self, file_output: str, video_duration: int, fps: int = 30):
        if not file_output:
            raise ValueError("Output file path cannot be empty.")
        if video_duration <= 0 or fps <= 0:
            raise ValueError("Video duration and FPS must be positive integers.")
        
        output_dir = os.path.dirname(file_output)
        if output_dir and not os.path.exists(output_dir):
            raise ValueError(f"Output directory does not exist: {output_dir}")

        total_frames = fps * video_duration
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(f"{file_output}.mp4", fourcc, fps, (self.sim_config.size_x, self.sim_config.size_y))

        print("Rendering...")
        for frame_num in range(total_frames):
            frame = self.render_frame()
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.update()
            print(f"\rProgress: {((frame_num + 1) / total_frames):.2%}", end="")
        print("\nVideo creation complete!")
        video.release()
    
    def finalize(self):
        for component in self.components:
            component.finalize()

class SimulationComponent:
    def __init__(self, system: AgentSystem):
        self.system = system

    def initialize(self):
        """Set up any required data structures."""
        pass

    def update(self):
        """Update the component at each simulation step."""
        pass

    def finalize(self):
        """Perform any cleanup or save results."""
        pass

# ==========================
# Example Component
# ==========================
class ExampleComponent(SimulationComponent):
    def __init__(self, system: SimulationComponent):
        super().__init__(system)

# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    sim_config = SimulationConfig()
    render_config = RenderingConfig(render_network=True)

    network = np.array([
        [sim_config.size_y // 2, sim_config.size_x // 2],
        [sim_config.size_y // 2 + 50, sim_config.size_x // 2 + 30],
        [sim_config.size_y // 2 - 50, sim_config.size_x // 2 - 30]
    ], dtype=np.int32)

    weights = np.array([0.5, 0.4, 0.4])

    system = AgentSystem(sim_config, render_config, network=network, weights=weights)
    system.render_video(r"test", video_duration=2, fps=30)