import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from geo_mesh_processor import load_mesh_data

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def initialize_llm(model_name="gpt-5-nano", temperature=0.1):
    """Initialize OpenAI LLM with LangChain"""
    
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=2000,
        timeout=30
    )
    return llm

def create_climate_prompt():
    """Create prompt template for climate data requests"""
    
    prompt_template = """You are a climate data expert. Given the location coordinates and address information below, provide climatological mean values for temperature and precipitation for the period 1991-2020.

Location Information:
- Longitude: {longitude}
- Latitude: {latitude}
- Country: {country}
- State/Region: {state}
- City: {city}

Please provide the following climate data for this location:
1. Temperature at 2m above surface (°C) - monthly climatological means for 1991-2020
2. Total precipitation (mm/day) - monthly climatological means for 1991-2020

For each month (January through December), provide:
- mean: average value
- min: minimum typical value 
- max: maximum typical value

Return ONLY a JSON object with this exact structure (no additional text):
{{
  "temperature_2m_celsius": {{
    "january": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "february": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "march": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "april": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "may": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "june": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "july": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "august": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "september": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "october": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "november": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "december": {{"mean": 0.0, "min": 0.0, "max": 0.0}}
  }},
  "precipitation_mm_per_day": {{
    "january": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "february": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "march": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "april": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "may": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "june": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "july": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "august": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "september": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "october": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "november": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "december": {{"mean": 0.0, "min": 0.0, "max": 0.0}}
  }}
}}"""

    return ChatPromptTemplate.from_template(prompt_template)

def validate_and_parse_response(response_text: str) -> Optional[Dict]:
    """Validate and parse LLM JSON response"""
    try:
        # Try to extract JSON from response
        response_text = response_text.strip()
        
        # Remove any markdown formatting
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        # Parse JSON
        data = json.loads(response_text)
        
        # Validate structure
        required_keys = ["temperature_2m_celsius", "precipitation_mm_per_day"]
        months = ["january", "february", "march", "april", "may", "june",
                 "july", "august", "september", "october", "november", "december"]
        
        for key in required_keys:
            if key not in data:
                return None
                
            for month in months:
                if month not in data[key]:
                    return None
                if not all(stat in data[key][month] for stat in ["mean", "min", "max"]):
                    return None
                    
        return data
        
    except json.JSONDecodeError:
        return None
    except Exception:
        return None

def convert_to_numpy_arrays(climate_data: Dict) -> Dict:
    """Convert climate data to numpy arrays for easier analysis"""
    months = ["january", "february", "march", "april", "may", "june",
             "july", "august", "september", "october", "november", "december"]
    
    result = {
        "temperature_2m_celsius": {
            "mean": np.array([climate_data["temperature_2m_celsius"][month]["mean"] for month in months]),
            "min": np.array([climate_data["temperature_2m_celsius"][month]["min"] for month in months]),
            "max": np.array([climate_data["temperature_2m_celsius"][month]["max"] for month in months])
        },
        "precipitation_mm_per_day": {
            "mean": np.array([climate_data["precipitation_mm_per_day"][month]["mean"] for month in months]),
            "min": np.array([climate_data["precipitation_mm_per_day"][month]["min"] for month in months]),
            "max": np.array([climate_data["precipitation_mm_per_day"][month]["max"] for month in months])
        }
    }
    
    return result

def query_climate_data(llm, prompt_template, point_data: Dict, max_retries: int = 3) -> Optional[Dict]:
    """Query LLM for climate data with retry logic"""
    
    # Prepare location info
    longitude = point_data.get('lon', 'N/A')
    latitude = point_data.get('lat', 'N/A')
    country = point_data.get('country', 'N/A') if point_data.get('country') else 'N/A'
    state = point_data.get('state', 'N/A') if point_data.get('state') else 'N/A'
    city = point_data.get('city', 'N/A') if point_data.get('city') else 'N/A'
    
    for attempt in range(max_retries):
        try:
            # Create the prompt
            messages = prompt_template.format_messages(
                longitude=longitude,
                latitude=latitude,
                country=country,
                state=state,
                city=city
            )
            
            # Query the LLM
            response = llm.invoke(messages)
            response_text = response.content
            
            # Validate and parse response
            parsed_data = validate_and_parse_response(response_text)
            
            if parsed_data is not None:
                return {
                    'raw_response': response_text,
                    'parsed_data': parsed_data,
                    'numpy_arrays': convert_to_numpy_arrays(parsed_data),
                    'attempt': attempt + 1
                }
            else:
                print(f"  Attempt {attempt + 1}: Invalid JSON response, retrying...")
                
        except Exception as e:
            print(f"  Attempt {attempt + 1}: Error querying LLM: {e}")
            
    
    print(f"  Failed to get valid response after {max_retries} attempts")
    return None

def process_climate_benchmark(mesh_file: str, num_repeats: int = 1, model_name: str = "gpt-5-nano"):
    """Main function to process climate benchmark"""
    
    print(f"Loading mesh data from {mesh_file}...")
    mesh_data = load_mesh_data(mesh_file)
    mesh_points = mesh_data['mesh_points']
    resolution = mesh_data['resolution']
    
    print(f"Loaded {len(mesh_points)} points with {resolution}° resolution")
    
    # Filter land points
    land_points = [point for point in mesh_points if point['is_land']]
    print(f"Found {len(land_points)} land points")
    
    # Initialize LLM
    print(f"Initializing LLM: {model_name}")
    llm = initialize_llm(model_name)
    prompt_template = create_climate_prompt()
    
    # Process each land point
    results = []
    
    for i, point_data in enumerate(land_points):
        print(f"\nProcessing point {i+1}/{len(land_points)}: ({point_data['lat']:.1f}, {point_data['lon']:.1f})")
        if point_data.get('country'):
            print(f"  Location: {point_data['country']}, {point_data.get('state', 'N/A')}, {point_data.get('city', 'N/A')}")
        
        point_results = {
            'point_info': point_data,
            'llm_responses': []
        }
        
        # Make multiple queries for statistics
        for repeat in range(num_repeats):
            if num_repeats > 1:
                print(f"  Query {repeat + 1}/{num_repeats}")
            
            climate_response = query_climate_data(llm, prompt_template, point_data)
            
            if climate_response:
                point_results['llm_responses'].append(climate_response)
                print(f"  ✓ Successfully got climate data")
            else:
                print(f"  ✗ Failed to get climate data")
                point_results['llm_responses'].append(None)
            
        
        results.append(point_results)
        
        # Save intermediate results every 10 points
        if (i + 1) % 10 == 0:
            save_results(results, mesh_data, f"results/climate_results_intermediate_{i+1}.json", model_name)
    
    return results, mesh_data

def save_results(results: List[Dict], mesh_data: Dict, output_file: str, model_name: str):
    """Save climate benchmark results"""
    # Create results directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving results to {output_file}...")
    
    output_data = {
        'mesh_info': mesh_data['mesh_info'],
        'resolution': mesh_data['resolution'],
        'total_land_points': len(results),
        'results': results,
        'metadata': {
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_used': model_name,
            'num_repeats_per_point': len(results[0]['llm_responses']) if results else 0
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"Results saved to {output_file}")

def main():
    """Main function"""
    import sys
    
    # Parse command line arguments
    mesh_file = 'meshes/mesh_data_10deg.json'
    num_repeats = 1
    model_name = 'gpt-5-nano'
    
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    if len(sys.argv) > 2:
        num_repeats = int(sys.argv[2])
    if len(sys.argv) > 3:
        model_name = sys.argv[3]
    
    print(f"Climate LLM Benchmark")
    print(f"Mesh file: {mesh_file}")
    print(f"Repeats per point: {num_repeats}")
    print(f"Model: {model_name}")
    
    # Check if mesh file exists
    if not Path(mesh_file).exists():
        print(f"Error: Mesh file '{mesh_file}' not found.")
        print("Please provide a valid mesh file or run geo_mesh_processor.py first.")
        return
    
    try:
        # Process the benchmark
        results, mesh_data = process_climate_benchmark(mesh_file, num_repeats, model_name)
        
        # Save final results
        resolution = mesh_data['resolution']
        output_file = f"results/climate_results_{resolution}deg_r{num_repeats}.json"
        save_results(results, mesh_data, output_file, model_name)
        
        # Print summary
        successful_points = sum(1 for r in results if any(resp for resp in r['llm_responses'] if resp))
        total_queries = len(results) * num_repeats
        successful_queries = sum(sum(1 for resp in r['llm_responses'] if resp) for r in results)
        
        print(f"\nBenchmark completed!")
        print(f"Total land points processed: {len(results)}")
        print(f"Points with successful responses: {successful_points}")
        print(f"Total queries made: {total_queries}")
        print(f"Successful queries: {successful_queries}")
        print(f"Success rate: {successful_queries/total_queries*100:.1f}%")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")

if __name__ == "__main__":
    main()