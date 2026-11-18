"""Render the annotated scene grid used by the batch vision planner."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle


def ensure_topview_grid(path: Path) -> None:
    """Create the PNG from local scene metadata on a cache miss."""
    if path.is_file():
        return
    scene_name = path.parent.name
    scene_data, grid_data, objects_data = (
        json.loads((path.parent / name).read_text(encoding="utf-8"))
        for name in ("scene.json", "grid_map.json", "objects.json")
    )
    bbox = scene_data["bbox"]
    x_min, y_min, z_min = bbox[0]
    x_max, y_max, z_max = bbox[1]
    grid_step = grid_data["grid_step"]
    grid_width = int(np.ceil((x_max - x_min) / grid_step))
    grid_height = int(np.ceil((y_max - y_min) / grid_step))
    fig = Figure(figsize=(16, 16 * grid_height / grid_width))
    ax = fig.subplots()
    margin = 0.5
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect("equal")
    for i in range(grid_width + 1):
        x = x_min + i * grid_step
        ax.plot([x, x], [y_min, y_max], color="lightgray", linewidth=0.5, alpha=0.7)
    for j in range(grid_height + 1):
        y = y_min + j * grid_step
        ax.plot([x_min, x_max], [y, y], color="lightgray", linewidth=0.5, alpha=0.7)
    for i in range(0, grid_width + 1, 5):
        x = x_min + i * grid_step
        ax.plot([x, x], [y_min, y_max], color="gray", linewidth=1.2, alpha=0.8)
    for j in range(0, grid_height + 1, 5):
        y = y_min + j * grid_step
        ax.plot([x_min, x_max], [y, y], color="gray", linewidth=1.2, alpha=0.8)
    border = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor="black", facecolor="none")
    ax.add_patch(border)
    object_colors = {
        "table": "#8B4513",
        "chair": "#4169E1",
        "sofa": "#9370DB",
        "bed": "#FF69B4",
        "counter": "#2F4F4F",
        "cabinet": "#CD853F",
        "shelf": "#DAA520",
    }

    def get_object_color(label):
        for key in object_colors:
            if key in label.lower():
                return object_colors[key]
        return "#888888"

    object_labels_to_show = []
    for obj in objects_data:
        if not obj.get("use_this", True):
            continue
        center = obj["center"]
        width = obj["width"]
        depth = obj["depth"]
        label = obj["label"]
        idx = obj["idx"]
        bbox = obj.get("bbox", None)
        obj_x = center[0]
        obj_y = center[1]
        color = get_object_color(label)
        if bbox is not None and len(bbox) >= 4:
            xy_points = [(bbox[i][0], bbox[i][1]) for i in range(4)]
            polygon = Polygon(xy_points, closed=True, linewidth=1, edgecolor="black", facecolor=color, alpha=0.6)
            ax.add_patch(polygon)
        else:
            rect = Rectangle(
                (obj_x - width / 2, obj_y - depth / 2),
                width,
                depth,
                linewidth=1,
                edgecolor="black",
                facecolor=color,
                alpha=0.6,
            )
            ax.add_patch(rect)
        object_labels_to_show.append(
            {"x": obj_x, "y": obj_y, "label": label, "idx": idx, "width": width, "depth": depth}
        )
    for obj_info in object_labels_to_show:
        obj_name = f"{obj_info['label']}_{obj_info['idx']}"
        ax.text(
            obj_info["x"],
            obj_info["y"],
            obj_name,
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="none"),
        )
    for i in range(0, grid_width + 1, 5):
        x = x_min + i * grid_step
        ax.text(x, y_min - 0.3, f"{i}", ha="center", va="top", fontsize=8, color="blue")
    for j in range(0, grid_height + 1, 5):
        y = y_min + j * grid_step
        ax.text(x_min - 0.3, y, f"{j}", ha="right", va="center", fontsize=8, color="blue")
    title = f"{scene_name} Scene - Top View (Grid: {grid_step}m = 30cm)"
    ax.set_xlabel("Grid X", fontsize=12)
    ax.set_ylabel("Grid Y", fontsize=12)
    ax.set_title(title, fontsize=16, weight="bold", pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_xticks([])
    ax.set_yticks([])
    legend_elements = []
    used_labels = set()
    for obj in objects_data[:20]:
        label = obj["label"]
        if label in used_labels or not obj.get("use_this", True):
            continue
        used_labels.add(label)
        color = get_object_color(label)
        legend_elements.append(patches.Patch(facecolor=color, edgecolor="black", alpha=0.6, label=label))
    if legend_elements:
        ax.legend(handles=legend_elements[:10], loc="upper right", fontsize=8, framealpha=0.9)
    info_text = f"Grid Size: {grid_width}×{grid_height}\nCell Size: {grid_step}m (30cm)\nBlue Numbers: Grid Index"
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    with NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        fig.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
