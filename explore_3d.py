
import os
import sys
from pathlib import Path

# Dataset and visualization directories
data_root = Path("/data/synthetic_coronagraphs")
output_root = Path.home() / "cme-sil3d" / "visualization" / "3d"

output_root.mkdir(parents=True, exist_ok=True)

# Import the visualization functions from cme_tools
sys.path.insert(0, str(data_root / "cme_tools"))

from render_refined_cme_review import load_surface, interactive

# Select the same CME used for the 2D images
case = "b080_I1_AR1"
sequence = 20

# Find the corresponding 3D surface
folder = data_root / "cme_3d_shapes" / case

matches = list(
    folder.glob(f"*__seq{sequence:03d}__*__envelope20.npz")
)

if len(matches) != 1:
    raise FileNotFoundError(
        f"Expected one 3D file, found {len(matches)}"
    )

# Load the 3D geometry
surface = load_surface(matches[0])

# Save interactive visualization
output_file = output_root / f"{case}__seq{sequence:03d}.html"

interactive(
    case,
    sequence,
    {"envelope20": surface},
    output_file,
    longitudes=[0, 90, 180, 270],
    default="envelope20",
)

print(f"3D visualization saved to: {output_file}")
