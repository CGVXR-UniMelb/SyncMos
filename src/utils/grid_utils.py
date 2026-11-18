"""
Grid coordinate conversion utilities
Grid coordinate conversion tool

Provide a two-way conversion between grid coordinates and world coordinates, supporting multiple resolutions.
"""

import numpy as np
import torch
from typing import Union, Tuple


def grid_to_world(
    grid_x: Union[int, float, np.ndarray, torch.Tensor],
    grid_y: Union[int, float, np.ndarray, torch.Tensor],
    grid_size: float = 0.3,
    bbox_min: Union[Tuple[float, float], np.ndarray, torch.Tensor] = (0.0, 0.0),
    return_center: bool = True,
    z: float = 0.0,
    return_torch: bool = False,
    device: str = "cpu",
) -> Union[Tuple[float, float, float], np.ndarray, torch.Tensor]:
    """
    Convert grid coordinates to world coordinates

    Args:
        grid_x: Grid X coordinates (can be single values or arrays)
        grid_y: Grid Y-coordinate (can be a single value or an array)
        grid_size: Grid size (meters), default 0.3m (30cm)
        bbox_min: Minimum point coordinates of the scene boundary (x_min, y_min)
        return_center: Whether to return the center coordinates of the grid or the lower-left corner of the grid (False)
        z: Z-axis coordinates, default 0.0
        return_torch: Whether to return torch.Tensor
        device: Torch device (only when return_torch is True)=Used when True)

    Returns:
        World coordinates (x, y, z)
        - If the input is a scalar: returns a tuple (x, y, z) Or torch.Tensor
        - If the input is an array: returns the corresponding array type

    Examples:
        >>> # Single grid
        >>> grid_to_world(10, 15, grid_size=0.3, bbox_min=(0.0, 0.0))
        (3.15, 4.65, 0.0)

        >>> # Batch conversion
        >>> grid_to_world([10, 20], [15, 25], grid_size=0.3)
        array([[3.15, 4.65, 0.0],
               [6.15, 7.65, 0.0]])
    """
    # Process bbox_min
    if isinstance(bbox_min, (list, tuple)):
        x_min, y_min = bbox_min
    elif isinstance(bbox_min, np.ndarray):
        x_min, y_min = bbox_min[0], bbox_min[1]
    elif isinstance(bbox_min, torch.Tensor):
        x_min, y_min = bbox_min[0].item(), bbox_min[1].item()
    else:
        x_min, y_min = bbox_min, bbox_min

    # Convert to numpy for calculation
    if isinstance(grid_x, torch.Tensor):
        grid_x = grid_x.cpu().numpy()
    if isinstance(grid_y, torch.Tensor):
        grid_y = grid_y.cpu().numpy()

    grid_x = np.asarray(grid_x)
    grid_y = np.asarray(grid_y)

    # Calculate world coordinates
    if return_center:
        # Return to the center of the grid
        world_x = x_min + (grid_x + 0.5) * grid_size
        world_y = y_min + (grid_y + 0.5) * grid_size
    else:
        # Return the lower-left corner of the grid
        world_x = x_min + grid_x * grid_size
        world_y = y_min + grid_y * grid_size

    # Process the return format
    is_scalar = world_x.ndim == 0

    if is_scalar:
        world_x = float(world_x)
        world_y = float(world_y)

        if return_torch:
            return torch.tensor([world_x, world_y, z], dtype=torch.float32, device=device)
        else:
            return (world_x, world_y, z)
    else:
        # Batch processing
        z_array = np.full_like(world_x, z)
        world_coords = np.stack([world_x, world_y, z_array], axis=-1)

        if return_torch:
            return torch.tensor(world_coords, dtype=torch.float32, device=device)
        else:
            return world_coords


def world_to_grid(
    world_x: Union[float, np.ndarray, torch.Tensor],
    world_y: Union[float, np.ndarray, torch.Tensor],
    grid_size: float = 0.3,
    bbox_min: Union[Tuple[float, float], np.ndarray, torch.Tensor] = (0.0, 0.0),
    round_to_int: bool = True,
) -> Union[Tuple[int, int], Tuple[float, float], np.ndarray]:
    """
    Convert world coordinates to grid coordinates

    Args:
        world_x: World X coordinates (can be a single value or array)
        world_y: World Y coordinates (can be a single value or array)
        grid_size: Grid size (meters), default 0.3m (30cm)
        bbox_min: Minimum point coordinates of the scene boundary (x_min, y_min)
        round_to_int: Whether to round to integer grid indices

    Returns:
        Grid coordinates (grid_x, grid_y)
        - If round_to_int=True: Return integer coordinates
        - If round_to_int=False: Return floating-point coordinates

    Examples:
        >>> # Individual coordinates
        >>> world_to_grid(3.15, 4.65, grid_size=0.3, bbox_min=(0.0, 0.0))
        (10, 15)

        >>> # Batch conversion
        >>> world_to_grid([3.15, 6.15], [4.65, 7.65], grid_size=0.3)
        array([[10, 15],
               [20, 25]])
    """
    # Process bbox_min
    if isinstance(bbox_min, (list, tuple)):
        x_min, y_min = bbox_min
    elif isinstance(bbox_min, np.ndarray):
        x_min, y_min = bbox_min[0], bbox_min[1]
    elif isinstance(bbox_min, torch.Tensor):
        x_min, y_min = bbox_min[0].item(), bbox_min[1].item()
    else:
        x_min, y_min = bbox_min, bbox_min

    # Convert to numpy for calculation
    if isinstance(world_x, torch.Tensor):
        world_x = world_x.cpu().numpy()
    if isinstance(world_y, torch.Tensor):
        world_y = world_y.cpu().numpy()

    world_x = np.asarray(world_x)
    world_y = np.asarray(world_y)

    # Calculate grid coordinates
    # Note: grid_to_world returns the center of the grid, so we use floor here directly
    grid_x = (world_x - x_min) / grid_size
    grid_y = (world_y - y_min) / grid_size

    # Convert to integer grid indices
    if round_to_int:
        # Use floor instead of round, as grid_to_world returns the center of the grid
        grid_x = np.floor(grid_x).astype(int)
        grid_y = np.floor(grid_y).astype(int)

    # Process the return format
    is_scalar = grid_x.ndim == 0

    if is_scalar:
        if round_to_int:
            return (int(grid_x), int(grid_y))
        else:
            return (float(grid_x), float(grid_y))
    else:
        # Batch processing
        return np.stack([grid_x, grid_y], axis=-1)


def validate_grid_coords(
    grid_x: Union[int, np.ndarray],
    grid_y: Union[int, np.ndarray],
    grid_width: int,
    grid_height: int,
    clip: bool = False,
) -> Union[Tuple[int, int], np.ndarray]:
    """
    Verify that grid coordinates are within valid range

    Args:
        grid_x: X coordinate of the grid
        grid_y: Y coordinate of the grid
        grid_width: Width of the grid map
        grid_height: Height of the grid map
        clip: Whether to clip coordinates that exceed the range to the boundaries

    Returns:
        Verification/Clipped grid coordinates

    Raises:
        ValueError: If the coordinates exceed the range, perform clipping=False
    """
    grid_x = np.asarray(grid_x)
    grid_y = np.asarray(grid_y)

    if clip:
        grid_x = np.clip(grid_x, 0, grid_width - 1)
        grid_y = np.clip(grid_y, 0, grid_height - 1)
    else:
        # Check if they are within the range
        if np.any(grid_x < 0) or np.any(grid_x >= grid_width):
            raise ValueError(f"grid_x out of range [0, {grid_width}): {grid_x}")
        if np.any(grid_y < 0) or np.any(grid_y >= grid_height):
            raise ValueError(f"grid_y out of range [0, {grid_height}): {grid_y}")

    is_scalar = grid_x.ndim == 0
    if is_scalar:
        return (int(grid_x), int(grid_y))
    else:
        return np.stack([grid_x, grid_y], axis=-1)


def get_grid_info_from_scene(scene) -> dict:
    """
    Extract grid information from scene objects

    Args:
        scene: Scene Object

    Returns:
        Dictionary containing grid information:
        {
            'grid_size': Grid size (meters),
            'grid_width': Grid width,
            'grid_height': Grid height,
            'bbox_min': (x_min, y_min),
            'bbox_max': (x_max, y_max)
        }
    """
    grid_map = scene.grid_map

    return {
        "grid_size": grid_map.grid_step,
        "grid_width": len(grid_map.grid_points),
        "grid_height": len(grid_map.grid_points[0]) if len(grid_map.grid_points) > 0 else 0,
        "bbox_min": (grid_map.bbox[0][0], grid_map.bbox[0][1]),
        "bbox_max": (grid_map.bbox[1][0], grid_map.bbox[1][1]),
    }


class GridCoordinateConverter:
    """
    Grid coordinate converter class

    Encapsulates grid information and conversion methods for easy reuse in code
    """

    def __init__(
        self,
        grid_size: float = 0.3,
        grid_width: int = None,
        grid_height: int = None,
        bbox_min: Tuple[float, float] = (0.0, 0.0),
        bbox_max: Tuple[float, float] = None,
        device: str = "cpu",
    ):
        """
        Initialize the converter

        Args:
            grid_size: Grid size (meters)
            grid_width: Width of the grid map
            grid_height: Height of the grid map
            bbox_min: Minimum point of the scene boundary
            bbox_max: Maximum point of scene boundary (optional)
            device: Torch device
        """
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.device = device

    @classmethod
    def from_scene(cls, scene, device="cpu"):
        """Create converter from scene object"""
        info = get_grid_info_from_scene(scene)
        return cls(
            grid_size=info["grid_size"],
            grid_width=info["grid_width"],
            grid_height=info["grid_height"],
            bbox_min=info["bbox_min"],
            bbox_max=info["bbox_max"],
            device=device,
        )

    def to_world(self, grid_x, grid_y, return_center=True, z=0.0, return_torch=False):
        """Grid coordinates → World coordinates"""
        return grid_to_world(
            grid_x,
            grid_y,
            grid_size=self.grid_size,
            bbox_min=self.bbox_min,
            return_center=return_center,
            z=z,
            return_torch=return_torch,
            device=self.device,
        )

    def to_grid(self, world_x, world_y, round_to_int=True):
        """World coordinates → Grid coordinates"""
        return world_to_grid(
            world_x, world_y, grid_size=self.grid_size, bbox_min=self.bbox_min, round_to_int=round_to_int
        )

    def validate(self, grid_x, grid_y, clip=False):
        """Verify grid coordinates"""
        if self.grid_width is None or self.grid_height is None:
            raise ValueError("Grid dimensions not set")
        return validate_grid_coords(grid_x, grid_y, self.grid_width, self.grid_height, clip=clip)

    def __repr__(self):
        return (
            f"GridCoordinateConverter(grid_size={self.grid_size}, "
            f"size={self.grid_width}×{self.grid_height}, "
            f"bbox_min={self.bbox_min})"
        )


# Convenient function: multi-resolution support
def create_converter(grid_size: float, scene=None, **kwargs):
    """
    Create a converter with the specified resolution

    Args:
        grid_size: Grid size (meters), such as 0.3 (30cm), 0.2 (20cm), 0.5 (50cm)
        scene: Scene objects (optional)
        **kwargs: Other parameters

    Returns:
        GridCoordinateConverter instance

    Examples:
        >>> # 30cm grids
        >>> converter_30cm = create_converter(0.3, scene=scene)

        >>> # 20cm Grid (Higher precision)
        >>> converter_20cm = create_converter(0.2, scene=scene)

        >>> # 50cm grid (lower precision)
        >>> converter_50cm = create_converter(0.5, scene=scene)
    """
    if scene is not None:
        converter = GridCoordinateConverter.from_scene(scene, **kwargs)
        converter.grid_size = grid_size
        # Recalculate the number of grids
        bbox_range = (converter.bbox_max[0] - converter.bbox_min[0], converter.bbox_max[1] - converter.bbox_min[1])
        converter.grid_width = int(np.ceil(bbox_range[0] / grid_size))
        converter.grid_height = int(np.ceil(bbox_range[1] / grid_size))
        return converter
    else:
        return GridCoordinateConverter(grid_size=grid_size, **kwargs)
