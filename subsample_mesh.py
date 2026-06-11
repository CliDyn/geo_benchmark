"""Subsample a mesh file: keep every k-th land point.

The result is an exact subset of the parent mesh (same point dicts, same
deterministic order), so per-point metadata such as addresses stays valid and
results can be compared point-by-point with full-grid runs.

Usage:
    python subsample_mesh.py meshes/mesh_data_1.0deg.json 10
    -> meshes/mesh_data_1.0deg_sub10.json
"""
import json
import sys
from pathlib import Path


def subsample_land_points(mesh_data, k):
    """Return a new mesh dict containing every k-th land point of the parent mesh."""
    land_points = [p for p in mesh_data['mesh_points'] if p['is_land']]
    selected = land_points[::k]
    mesh_info = dict(mesh_data.get('mesh_info', {}))
    mesh_info['subsample_k'] = k
    mesh_info['n_land_points_parent'] = len(land_points)
    mesh_info['n_points'] = len(selected)
    # lon_mesh/lat_mesh deliberately omitted: load_mesh_data() treats such
    # files like chunk files and benchmarks only iterate land points anyway.
    return {
        'mesh_points': selected,
        'resolution': mesh_data.get('resolution'),
        'mesh_info': mesh_info,
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python subsample_mesh.py <mesh_file.json> <k>")
        sys.exit(1)
    mesh_file = sys.argv[1]
    k = int(sys.argv[2])

    with open(mesh_file, 'r') as f:
        mesh_data = json.load(f)

    sub = subsample_land_points(mesh_data, k)
    sub['mesh_info']['parent_mesh_file'] = mesh_file

    out_file = str(Path(mesh_file).with_suffix('')) + f"_sub{k}.json"
    with open(out_file, 'w') as f:
        json.dump(sub, f, indent=2)

    print(f"Parent land points: {sub['mesh_info']['n_land_points_parent']}")
    print(f"Selected every {k}-th land point -> {len(sub['mesh_points'])} points")
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
