from collections import defaultdict
from typing import List

import numpy as np
import torch

from src.scene.grid_map import GridMap
from src.type import Character
from src.utils import misc, smpl
from src.utils.transform import quaternion_apply

from ..states import State
from ..utils import spline, spring
from .priority_queue import PriorityQueue


class PathPlanner:
    def __init__(self, grid_map: GridMap, cfg) -> None:
        self.cfg = cfg
        self.grid_map = grid_map
        self.reserved_paths = []

        self.device = cfg.device
        self.framerate = cfg.framerate
        self.dt = 1 / self.framerate
        self.min_planning_timesteps = cfg.path_planner.min_planning_timesteps
        self.max_planning_timesteps = cfg.path_planner.max_planning_timesteps
        self.oracle_timesteps = cfg.path_planner.oracle_timesteps
        self.non_moving_cost = cfg.path_planner.non_moving_cost
        self.collision_soft_threshold = cfg.path_planner.collision_soft_threshold
        self.occupied_neighbor_radius = cfg.path_planner.get(
            "occupied_neighbor_radius", 1
        )  # Radius occupied by surrounding cells
        self.collision_hard_threshold = cfg.path_planner.collision_hard_threshold
        self.collision_cost_scaler = cfg.path_planner.collision_cost_scaler
        self.soft_collision_cost = cfg.path_planner.soft_collision_cost
        self.hard_collision_cost = cfg.path_planner.hard_collision_cost
        self.future_offsets = torch.tensor(
            [int(offset * self.framerate) for offset in cfg.motion_matching.future_offsets],
            device=self.device,
        )
        self.far_threshold = cfg.state_manager.far_threshold
        self.x_axis = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device).reshape(1, 3)

    def get_occupied_positions_with_neighbors(self, center_position: tuple, radius: int = 1) -> List[tuple]:
        """
        Obtain the central position and surrounding cell positions

        Args:
            center_position: Central cell position (x, y)
            radius: Radius occupied (number of cells)

        Returns:
            List of positions including the central and surrounding locations
        """
        occupied_positions = []
        cx, cy = center_position

        # Obtain the size of the grid map
        grid_x_size = len(self.grid_map.grid_points)
        grid_y_size = len(self.grid_map.grid_points[0]) if grid_x_size > 0 else 0

        # Traverse the surrounding cells
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = cx + dx, cy + dy
                # Check if within the grid range
                if 0 <= nx < grid_x_size and 0 <= ny < grid_y_size:
                    occupied_positions.append((nx, ny))

        return occupied_positions

    def update_character_path(self, character: Character, characters: List[Character]):

        existing_paths = []
        occupied_positions = []
        collision_soft_threshold = self.collision_soft_threshold
        collision_hard_threshold = self.collision_hard_threshold

        # Maintain the original collision threshold (do not reduce the collision cost during deadlock avoidance)

        for other in characters:
            if character == other:
                continue

            # Normal mode: Consider the paths of other characters
            if other == character.interactee or other in character.parallel_group_members:
                # hand_target_position = character.target_hand_position
                # if hand_target_position is not None:
                #    hand_occupied_position = self.grid_map.get_closest_grid_position(
                #        hand_target_position, exclude_occupied=False
                #    )
                #    occupied_positions.append([hand_occupied_position] * 150)

                if other.is_resting_posture():
                    continue  # don't consider stationary interactee's path
                elif other.state == State.IDLE:
                    occupied_position = other.position
                    occupied_position = self.grid_map.get_closest_grid_position(
                        occupied_position, exclude_occupied=False
                    )

                    # Occupy the central position and surrounding cells
                    occupied_positions_with_neighbors = self.get_occupied_positions_with_neighbors(
                        occupied_position, radius=self.occupied_neighbor_radius
                    )
                    for pos in occupied_positions_with_neighbors:
                        existing_paths.append([pos] * 150)

                    existing_paths.append([occupied_position] * 150)

                    collision_soft_threshold = 0.5
                else:
                    existing_paths.append(other.grid_path)  # do not push away interactee
            else:
                if (
                    other.state in [State.INTERACTING, State.TRANSITION_IN, State.TRANSITION_OUT, State.IDLE]
                    or other.is_resting_posture()
                ):
                    occupied_position = other.position
                    occupied_position = self.grid_map.get_closest_grid_position(
                        occupied_position, exclude_occupied=False
                    )

                    # Occupy the central position and surrounding cells
                    # occupied_positions_with_neighbors = self.get_occupied_positions_with_neighbors(
                    #     occupied_position, radius=self.occupied_neighbor_radius
                    # )
                    # for pos in occupied_positions_with_neighbors:
                    #     existing_paths.append([pos] * 150)

                    existing_paths.append([occupied_position] * 150)
                # elif other.state == State.IDLE:
                #     continue
                else:
                    existing_paths.append(other.grid_path)

        # update the reserved path (grid level)
        if character.is_resting_posture():
            character.grid_path = [
                self.grid_map.get_closest_grid_position(character.position, exclude_occupied=False)
            ] * self.min_planning_timesteps
        else:
            start = self.grid_map.get_closest_grid_position(character.position)
            goal = self.grid_map.get_closest_grid_position(character.target_position)

            # Start and end points are the same (e.g., a valid target is(0,0))Directly construct static paths to avoid false searches that lead to empty paths
            if start == goal:
                character.grid_path = [start] * self.min_planning_timesteps
                return

            # Perform normal searches; if the search fails, repeat the current position to prevent empty paths
            try:
                character.grid_path = self.search_path(
                    start=start,
                    goal=goal,
                    min_timesteps=self.min_planning_timesteps,
                    max_timesteps=self.max_planning_timesteps,
                    oracle_timesteps=self.oracle_timesteps,
                    occupied_positions=occupied_positions,
                    existing_paths=existing_paths,
                    collision_soft_threshold=collision_soft_threshold,
                    collision_hard_threshold=collision_hard_threshold,
                )
            except Exception:
                character.grid_path = [start] * self.min_planning_timesteps

    def search_path(
        self,
        start,
        goal,
        start_time=0,
        min_timesteps=0,
        max_timesteps=100,
        existing_paths=None,
        occupied_positions=None,
        oracle_timesteps=20,
        collision_soft_threshold=None,
        collision_hard_threshold=None,
    ):
        priority_queue = PriorityQueue((start_time, start), priority=0)
        came_from = {(start_time, start): (-1, None)}
        lowest_cost = defaultdict(lambda: float("inf"))
        lowest_cost[start_time, start] = 0

        while not priority_queue.is_empty():
            current_time, current = priority_queue.pop()
            if current == goal and current_time >= start_time + min_timesteps:
                return self._reconstruct_path(came_from, goal, current_time)

            neighbors = self._get_neighbors(current)
            if current_time >= start_time + max_timesteps:
                neighbors.append(goal)  # make a jump to the goal

            for next in neighbors:
                next_time = current_time + 1
                if not self._is_within_grid_map(next):
                    continue

                max_consider_time = start_time + oracle_timesteps
                new_cost, collision_cost = self._calculate_new_cost(
                    lowest_cost,
                    current,
                    next,
                    next_time,
                    existing_paths,
                    occupied_positions,
                    max_consider_time,
                    collision_soft_threshold,
                    collision_hard_threshold,
                )

                if new_cost < lowest_cost[next_time, next]:
                    came_from[next_time, next] = (current_time, current)
                    lowest_cost[next_time, next] = new_cost

                    priority = new_cost + self._heuristic(next, goal)
                    entry = (next_time, next)
                    if entry in priority_queue:
                        priority_queue.update(entry, priority=priority)
                    else:
                        priority_queue.push(entry, priority=priority)
        raise RuntimeError("There is no path that can reach the goal")

    def _get_neighbors(self, current):
        dxs = (0, 0, -1, -1, -1, 1, 1, 1, 0)
        dys = (1, -1, -1, 0, 1, -1, 0, 1, 0)

        neighbors = []
        for dx, dy in zip(dxs, dys):
            neighbors.append((current[0] + dx, current[1] + dy))
        return neighbors

    def _is_within_grid_map(self, position):
        return (0 <= position[0] < self.grid_map.num_x) and (0 <= position[1] < self.grid_map.num_y)

    def _calculate_new_cost(
        self,
        lowest_cost,
        current,
        next,
        next_time,
        existing_paths,
        occupied_positions,
        max_consider_time,
        collision_soft_threshold=None,
        collision_hard_threshold=None,
    ):
        current_time = next_time - 1
        result = lowest_cost[current_time, current] + self._get_moving_cost(current, next)
        if self.grid_map.occupied[next]:
            result += 1e5

        collision_cost = 0
        if existing_paths is not None and next_time <= max_consider_time:
            collision_cost = self._get_collision_cost(
                next, next_time, existing_paths, occupied_positions, collision_soft_threshold, collision_hard_threshold
            )
            result += collision_cost

        return result, collision_cost

    def _get_moving_cost(self, current, next):
        dx = next[0] - current[0]
        dy = next[1] - current[1]
        if dx == 0 and dy == 0:
            return self.non_moving_cost
        return 1 if dx == 0 or dy == 0 else (2**0.5)

    def _get_collision_cost(
        self,
        position,
        t,
        existing_paths,
        occupied_positions,
        collision_soft_threshold=None,
        collision_hard_threshold=None,
    ):
        cost = 0
        for path in existing_paths:
            if not path or t >= len(path):
                continue

            distance = np.linalg.norm(self.grid_map[position] - self.grid_map[path[t]])
            cost += self._collision_cost(distance, collision_soft_threshold, collision_hard_threshold)

        for occupied_position in occupied_positions:
            distance = np.linalg.norm(self.grid_map[position] - self.grid_map[occupied_position])
            cost += self._collision_cost(distance, collision_soft_threshold, collision_hard_threshold)

        return cost

    def _collision_cost(self, distance, collision_soft_threshold=None, collision_hard_threshold=None):
        if collision_soft_threshold is None:
            collision_soft_threshold = self.collision_soft_threshold
        if collision_hard_threshold is None:
            collision_hard_threshold = self.collision_hard_threshold

        if distance > collision_soft_threshold:
            return self.soft_collision_cost
        if distance < collision_hard_threshold:
            return self.hard_collision_cost

        scale = self.collision_cost_scaler / (collision_soft_threshold**2)
        return scale * ((collision_soft_threshold - distance) ** 2)

    def _heuristic(self, current, goal, D=1, D2=2**0.5):  # Diagonal Distance
        dx = abs(current[0] - goal[0])
        dy = abs(current[1] - goal[1])
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

    def _reconstruct_path(self, came_from, goal, arrival_time):
        route = []
        visited = set()
        current = goal
        current_time = arrival_time
        while current:
            if (current_time, current) in visited:
                raise RuntimeError("There is a loop in the path")
            route.append(current)
            visited.add((current_time, current))
            current_time, current = came_from[current_time, current]
        route = route[::-1]
        return route

    def update_future_prediction(self, character: Character):
        spline_curve, _ = self.get_tcb_spline_curve(character.grid_path, velocity_mps=character.velocity_mps)
        spline_curve = torch.tensor(spline_curve, device=self.device)
        character.spline_curve = spline_curve[: min(self.future_offsets)]

        ### update future positions
        future_positions = []
        for offset in self.future_offsets:
            offset = min(offset, len(spline_curve) - 1)
            future_positions.append(spline_curve[offset])
        character.future_positions = torch.stack(future_positions, dim=0)

        ### update future facing directions
        distance_to_target = torch.norm(character.position - character.target_position)
        # during approaching
        if distance_to_target >= self.far_threshold:
            target_direction = character.future_positions[-1:] - character.position
            target_direction /= torch.norm(target_direction, dim=-1, keepdim=True)
        # waiting interactee
        elif character.interactee is not None:
            direction_to_interactee = character.interactee.position - character.position
            direction_to_interactee /= torch.norm(direction_to_interactee, dim=-1, keepdim=True)
            if character.is_resting_posture():
                target_direction = misc.calculate_rotated_direction(
                    character.target_facing_direction,
                    direction_to_interactee,
                    threshold_angle=torch.pi / 4,
                )
            else:
                target_direction = direction_to_interactee
        # when target direction is given (e.g. human-scene interaction)
        elif character.target_facing_direction is not None:
            target_direction = character.target_facing_direction
        # default case
        else:
            target_direction = character.facing_direction

        target_rotation = smpl.get_character_rotations(target_direction)
        future_rotations, _ = spring.simple_spring_damper_exact_quaternion(
            character.rotation,
            character.angular_velocity,
            target_rotation,
            character.halflife_rotation,
            (self.future_offsets * self.dt).unsqueeze(-1),  # for broadcasting
        )
        character.future_facing_directions = quaternion_apply(future_rotations, self.x_axis)

    def get_tcb_spline_curve(self, grid_path, velocity_mps=0.9, tension=-1.0, continuity=0.0, bias=0.0):
        velocity_mpf = velocity_mps / self.framerate
        control_points = np.array([self.grid_map[p][:2] for p in grid_path]).reshape(1, -1, 2)
        control_points = np.concatenate([control_points, np.zeros((*control_points.shape[:-1], 1))], axis=-1)
        ks = np.empty((0,), dtype=np.int32)
        ts = np.empty((0,), dtype=np.float32)
        for i, (prev_grid, next_grid) in enumerate(zip(grid_path[:-1], grid_path[1:])):
            prev_pos = self.grid_map[prev_grid]
            next_pos = self.grid_map[next_grid]
            distance = np.linalg.norm(next_pos - prev_pos)
            num_frames = int(distance / velocity_mpf)
            ks = np.append(ks, np.repeat(i, num_frames))
            ts = np.append(ts, np.linspace(0, 1, num_frames, endpoint=False))
        ks = np.append(ks, i)
        ts = np.append(ts, 1.0)

        spline_curve = spline.interpolate_tcb_spline(
            control_points,
            ks,
            ts,
            tension=tension,
            continuity=continuity,
            bias=bias,
        )
        _, tangents = spline.calculate_tangents(spline_curve, tension=tension, continuity=continuity, bias=bias)

        # z-value should be set to 0.0
        return (
            misc.add_z(spline_curve[0, :, :2], z=0.0),
            misc.add_z(tangents[0, :, :2], z=0.0),
        )

    # def _update_future_prediction_spline(self, character: Character):
    #     if character.state == State.IDLE:
    #         self._update_future_prediction_idle(character)
    #         return

    #     spline_curve, tangents = self.get_tcb_spline_curve(character.grid_path, velocity_mps=character.velocity_mps)
    #     spline_curve = torch.tensor(spline_curve, device=self.device)
    #     tangents = torch.tensor(tangents, device=self.device)

    #     future_positions = []
    #     future_facing_directions = []
    #     for offset in self.future_offsets:
    #         offset = min(offset, len(spline_curve) - 1)
    #         future_positions.append(spline_curve[offset])
    #         future_facing_directions.append(F.normalize(tangents[offset], dim=-1))

    #     character.future_positions = torch.stack(future_positions, dim=0)
    #     character.future_facing_directions = torch.stack(future_facing_directions, dim=0)

    # def _update_future_prediction_spring(self, character: Character):
    #     if character.state == State.IDLE:
    #         self._update_future_prediction_idle(character)
    #         return

    #     current_position = torch.tensor(self.grid_map[character.grid_path[0]], device=self.device).reshape(1, 3)
    #     target_position = torch.tensor(self.grid_map[character.grid_path[1]], device=self.device).reshape(1, 3)
    #     target_direction = target_position - current_position
    #     target_direction /= torch.norm(target_direction, dim=-1, keepdim=True)

    #     target_velocity = target_direction * character.velocity_mps
    #     target_rotation = smpl.get_character_rotations(target_direction)

    #     character.future_positions, *_ = spring.spring_character_update(
    #         current_position,
    #         character.velocity,
    #         character.acceleration,
    #         target_velocity,
    #         character.halflife_position,
    #         (self.future_offsets * self.dt).unsqueeze(-1),  # for broadcasting
    #     )

    #     future_rotations, _ = spring.simple_spring_damper_exact_quaternion(
    #         character.rotation,
    #         character.angular_velocity,
    #         target_rotation,
    #         character.halflife_rotation,
    #         (self.future_offsets * self.dt).unsqueeze(-1),  # for broadcasting
    #     )
    #     character.future_facing_directions = quaternion_apply(future_rotations, self.x_axis)

    def _update_deadlock_status(self, character: Character):
        """
        Detect and update the character’s deadlock status

        When a character stays in one place for a long time, it may be in a deadlock situation
        Trigger deadlock avoidance behavior to temporarily reduce collision costs and allow the character to move
        """
        # Initialize the position from the previous frame
        if character.last_position is None:
            character.last_position = character.position.clone()
            return

        # Calculate the movement distance
        movement = torch.norm(character.position - character.last_position).item()

        # If the character has hardly moved, proceed
        if movement < self.movement_threshold:
            character.waiting_frames += 1
        else:
            # If the character is moving, reset the waiting counter
            character.waiting_frames = 0
            character.deadlock_avoidance_active = False

        # Update the position from the previous frame
        character.last_position = character.position.clone()

        # Detect deadlocks: if the waiting time exceeds the threshold and the character is not in an interactive state
        if character.waiting_frames > self.deadlock_threshold:
            if character.state == State.IDLE or character.state == State.APPROACHING:
                # Trigger deadlock avoidance only when there is a target position and the distance to the target is significant
                if character.target_position is not None:
                    distance_to_target = torch.norm(character.position - character.target_position).item()
                    if distance_to_target > self.far_threshold:
                        # Activate deadlock avoidance behavior
                        if not character.deadlock_avoidance_active:
                            character.deadlock_avoidance_active = True
                            character.waiting_frames = 0  # Reset the counter, give us some time to avoid behavior

        # If the lock is activated for some time, it will be closed automatically
        if character.deadlock_avoidance_active and character.waiting_frames > self.deadlock_avoidance_duration:
            character.deadlock_avoidance_active = False
            character.waiting_frames = 0
