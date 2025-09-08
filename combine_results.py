#!/usr/bin/env python3
"""
Combine multiple chunk result files into a single complete result file.

Usage:
    python combine_results.py results_pattern.json output_file.json
    
Example:
    python combine_results.py "results/climate_results_1.0deg_r10_gpt-5-nano_chunk_*_simple.json" results/climate_results_1.0deg_r10_gpt-5-nano_combined_simple.json
"""

import json
import sys
import glob
from pathlib import Path
from typing import List, Dict
import time

def load_chunk_results(chunk_files: List[str]) -> List[Dict]:
    """Load results from multiple chunk files"""
    all_chunk_data = []
    
    print(f"Loading {len(chunk_files)} chunk files...")
    
    for chunk_file in sorted(chunk_files):
        print(f"  Loading {Path(chunk_file).name}...")
        
        with open(chunk_file, 'r') as f:
            chunk_data = json.load(f)
        
        # Extract chunk info
        chunk_id = chunk_data.get('mesh_info', {}).get('chunk_id', 'unknown')
        total_chunks = chunk_data.get('mesh_info', {}).get('total_chunks', 'unknown')
        land_points_in_chunk = len(chunk_data.get('results', []))
        
        print(f"    Chunk {chunk_id} of {total_chunks}: {land_points_in_chunk} points")
        
        all_chunk_data.append(chunk_data)
    
    return all_chunk_data

def combine_chunk_results(chunk_data_list: List[Dict]) -> Dict:
    """Combine chunk results into a single result structure"""
    
    if not chunk_data_list:
        raise ValueError("No chunk data provided")
    
    # Use the first chunk as the base structure
    base_chunk = chunk_data_list[0]
    
    # Initialize combined result structure
    combined_data = {
        'mesh_info': {
            # Remove chunk-specific info and add combined info
            **{k: v for k, v in base_chunk['mesh_info'].items() 
               if k not in ['chunk_id', 'total_chunks', 'land_points_in_chunk']},
            'combined_from_chunks': len(chunk_data_list)
        },
        'resolution': base_chunk['resolution'],
        'total_land_points': 0,  # Will be calculated
        'results': [],           # Will be combined from all chunks
        'metadata': {
            # Keep metadata from first chunk as base
            **base_chunk['metadata'],
            'combined_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'chunk_info': {
                'total_chunks_combined': len(chunk_data_list),
                'chunk_files': []
            }
        }
    }
    
    # Combine all results from all chunks
    all_results = []
    total_points = 0
    
    for i, chunk_data in enumerate(chunk_data_list):
        chunk_results = chunk_data.get('results', [])
        chunk_id = chunk_data.get('mesh_info', {}).get('chunk_id', i+1)
        
        # Add chunk info to metadata
        combined_data['metadata']['chunk_info']['chunk_files'].append({
            'chunk_id': chunk_id,
            'points_in_chunk': len(chunk_results),
            'processing_date': chunk_data.get('metadata', {}).get('processing_date', 'unknown')
        })
        
        # Add results from this chunk
        all_results.extend(chunk_results)
        total_points += len(chunk_results)
        
        print(f"  Combined chunk {chunk_id}: {len(chunk_results)} points")
    
    combined_data['results'] = all_results
    combined_data['total_land_points'] = total_points
    
    # Update metadata with combined totals
    if all_results:
        total_queries_per_chunk = []
        successful_queries_per_chunk = []
        
        for chunk_data in chunk_data_list:
            chunk_results = chunk_data.get('results', [])
            if chunk_results:
                num_repeats = len(chunk_results[0].get('llm_responses', []))
                chunk_total_queries = len(chunk_results) * num_repeats
                chunk_successful_queries = sum(
                    sum(1 for resp in r.get('llm_responses', []) if resp) 
                    for r in chunk_results
                )
                total_queries_per_chunk.append(chunk_total_queries)
                successful_queries_per_chunk.append(chunk_successful_queries)
        
        combined_data['metadata']['combined_statistics'] = {
            'total_queries_all_chunks': sum(total_queries_per_chunk),
            'successful_queries_all_chunks': sum(successful_queries_per_chunk),
            'success_rate_percent': (sum(successful_queries_per_chunk) / sum(total_queries_per_chunk) * 100) if sum(total_queries_per_chunk) > 0 else 0,
            'points_with_successful_responses': sum(1 for r in all_results if any(resp for resp in r.get('llm_responses', []) if resp))
        }
    
    return combined_data

def save_combined_results(combined_data: Dict, output_file: str):
    """Save combined results to JSON file"""
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving combined results to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2, default=str)
    
    print(f"Combined results saved to {output_file}")

def combine_chunk_files(pattern: str, output_file: str):
    """Combine chunk files matching pattern into single output file"""
    
    # Find all matching chunk files
    chunk_files = glob.glob(pattern)
    
    if not chunk_files:
        print(f"Error: No files found matching pattern '{pattern}'")
        return
    
    print(f"Found {len(chunk_files)} chunk files matching pattern:")
    for chunk_file in sorted(chunk_files):
        print(f"  {chunk_file}")
    
    # Load all chunk data
    chunk_data_list = load_chunk_results(chunk_files)
    
    # Validate chunk consistency
    resolutions = set(chunk['resolution'] for chunk in chunk_data_list)
    if len(resolutions) > 1:
        print(f"Warning: Multiple resolutions found: {resolutions}")
    
    models = set(chunk['metadata']['model_used'] for chunk in chunk_data_list)
    if len(models) > 1:
        print(f"Warning: Multiple models found: {models}")
    
    modes = set(chunk['metadata']['simple_mode'] for chunk in chunk_data_list)
    if len(modes) > 1:
        print(f"Warning: Multiple modes found: {modes}")
    
    # Combine chunk results
    combined_data = combine_chunk_results(chunk_data_list)
    
    # Save combined results
    save_combined_results(combined_data, output_file)
    
    # Print summary
    total_points = combined_data['total_land_points']
    stats = combined_data['metadata'].get('combined_statistics', {})
    
    print(f"\nCombining completed!")
    print(f"Total chunks combined: {len(chunk_files)}")
    print(f"Total land points: {total_points}")
    
    if stats:
        print(f"Total queries: {stats.get('total_queries_all_chunks', 'unknown')}")
        print(f"Successful queries: {stats.get('successful_queries_all_chunks', 'unknown')}")
        print(f"Success rate: {stats.get('success_rate_percent', 0):.1f}%")
        print(f"Points with successful responses: {stats.get('points_with_successful_responses', 'unknown')}")

def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python combine_results.py results_pattern.json output_file.json")
        print()
        print("Example:")
        print('  python combine_results.py "results/climate_results_1.0deg_r10_gpt-5-nano_chunk_*_simple.json" results/climate_results_1.0deg_r10_gpt-5-nano_combined_simple.json')
        return
    
    pattern = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        combine_chunk_files(pattern, output_file)
    except Exception as e:
        print(f"Error combining results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()