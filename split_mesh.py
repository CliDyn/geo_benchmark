#!/usr/bin/env python3
"""
Split mesh file into multiple chunks with equal distribution of land points.

Usage:
    python split_mesh.py mesh_file.json [num_chunks]
    
Default num_chunks is 20.
"""

import json
import sys
from pathlib import Path
import numpy as np
from typing import List, Dict

def load_mesh_data(mesh_file: str) -> Dict:
    """Load mesh data from JSON file"""
    print(f"Loading mesh data from {mesh_file}...")
    
    with open(mesh_file, 'r') as f:
        mesh_data = json.load(f)
    
    return mesh_data

def split_land_points_equally(land_points: List[Dict], num_chunks: int = 20) -> List[List[Dict]]:
    """Split land points into approximately equal chunks"""
    total_points = len(land_points)
    points_per_chunk = total_points // num_chunks
    remainder = total_points % num_chunks
    
    chunks = []
    start_idx = 0
    
    for i in range(num_chunks):
        # Distribute remainder across first chunks
        chunk_size = points_per_chunk + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        
        chunk_points = land_points[start_idx:end_idx]
        chunks.append(chunk_points)
        
        print(f"Chunk {i+1:2d}: {len(chunk_points):4d} points (indices {start_idx:4d}-{end_idx-1:4d})")
        start_idx = end_idx
    
    return chunks

def create_chunk_mesh(original_mesh: Dict, land_points_chunk: List[Dict], chunk_id: int) -> Dict:
    """Create a new mesh data structure for a chunk"""
    
    # Create new mesh with only the land points in this chunk
    chunk_mesh = {
        'mesh_info': {
            **original_mesh['mesh_info'],
            'chunk_id': chunk_id,
            'total_chunks': None,  # Will be set when saving
            'land_points_in_chunk': len(land_points_chunk)
        },
        'resolution': original_mesh['resolution'],
        'mesh_points': land_points_chunk
    }
    
    return chunk_mesh

def save_chunk_mesh(chunk_mesh: Dict, output_file: str):
    """Save chunk mesh to JSON file"""
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(chunk_mesh, f, indent=2)
    
    print(f"Saved chunk to {output_file}")

def split_mesh_file(mesh_file: str, num_chunks: int = 20):
    """Split mesh file into multiple chunks"""
    
    # Load original mesh
    mesh_data = load_mesh_data(mesh_file)
    
    # Extract land points
    all_points = mesh_data['mesh_points']
    land_points = [point for point in all_points if point.get('is_land', False)]
    
    print(f"Total mesh points: {len(all_points)}")
    print(f"Land points: {len(land_points)}")
    print(f"Splitting into {num_chunks} chunks...\n")
    
    # Split land points into chunks
    point_chunks = split_land_points_equally(land_points, num_chunks)
    
    # Create output file pattern
    input_path = Path(mesh_file)
    base_name = input_path.stem  # filename without extension
    output_dir = input_path.parent / "chunks"
    
    # Save each chunk
    for i, chunk_points in enumerate(point_chunks):
        chunk_mesh = create_chunk_mesh(mesh_data, chunk_points, i + 1)
        chunk_mesh['mesh_info']['total_chunks'] = num_chunks
        
        output_file = output_dir / f"{base_name}_chunk_{i+1:02d}_of_{num_chunks:02d}.json"
        save_chunk_mesh(chunk_mesh, str(output_file))
    
    print(f"\nSplit complete!")
    print(f"Created {num_chunks} chunk files in {output_dir}/")
    print(f"Each chunk contains land points only")
    
    # Print summary
    total_land_points = sum(len(chunk) for chunk in point_chunks)
    min_points = min(len(chunk) for chunk in point_chunks)
    max_points = max(len(chunk) for chunk in point_chunks)
    
    print(f"\nSummary:")
    print(f"Total land points distributed: {total_land_points}")
    print(f"Points per chunk: {min_points}-{max_points}")
    print(f"Average points per chunk: {total_land_points / num_chunks:.1f}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python split_mesh.py mesh_file.json [num_chunks]")
        print("Default num_chunks is 20")
        return
    
    mesh_file = sys.argv[1]
    num_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    # Check if mesh file exists
    if not Path(mesh_file).exists():
        print(f"Error: Mesh file '{mesh_file}' not found.")
        return
    
    try:
        split_mesh_file(mesh_file, num_chunks)
    except Exception as e:
        print(f"Error splitting mesh: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()