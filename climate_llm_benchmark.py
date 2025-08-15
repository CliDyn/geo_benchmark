import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from geo_mesh_processor import load_mesh_data

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import glob

def configure_langsmith(disable_tracing: bool = False):
    """Configure LangSmith tracing"""
    if disable_tracing:
        try:
            import langsmith as ls
            ls.configure(enabled=False)
            print("LangSmith tracing disabled")
        except ImportError:
            pass  # LangSmith not installed, ignore
    else:
        print("LangSmith tracing enabled (default)")

def find_latest_intermediate_file(resolution: str, simple_mode: bool = False) -> Optional[str]:
    """Find the latest intermediate file for resuming"""
    mode_suffix = "_simple" if simple_mode else ""
    pattern = f"results/climate_results_intermediate_*{mode_suffix}.json"
    
    # Find all matching intermediate files
    intermediate_files = glob.glob(pattern)
    
    if not intermediate_files:
        return None
    
    # Extract numbers and sort to find the latest
    file_numbers = []
    for file_path in intermediate_files:
        try:
            # Extract number from filename like "climate_results_intermediate_1840_simple.json"
            filename = Path(file_path).stem
            if simple_mode:
                number_part = filename.replace("climate_results_intermediate_", "").replace("_simple", "")
            else:
                number_part = filename.replace("climate_results_intermediate_", "")
            
            number = int(number_part)
            file_numbers.append((number, file_path))
        except ValueError:
            continue
    
    if not file_numbers:
        return None
    
    # Sort by number and return the path of the latest file
    file_numbers.sort(key=lambda x: x[0], reverse=True)
    latest_file = file_numbers[0][1]
    
    return latest_file

def load_intermediate_results(intermediate_file: str) -> tuple[List[Dict], Dict, int]:
    """Load intermediate results and return results, mesh_data, and start_index"""
    print(f"Loading intermediate results from {intermediate_file}...")
    
    with open(intermediate_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Reconstruct mesh_data from the saved data
    mesh_data = {
        'mesh_info': data['mesh_info'],
        'resolution': data['resolution'],
        'mesh_points': []  # Will be loaded from the original mesh file
    }
    
    start_index = len(results)
    print(f"Found {start_index} completed points, resuming from point {start_index + 1}")
    
    return results, mesh_data, start_index

def initialize_llm(model_name="gpt-5-nano", temperature=0, simple_mode=False):
    """Initialize OpenAI LLM with LangChain"""
    
    # Configure parameters based on model and mode
    if "gpt-5" in model_name:
        if simple_mode:
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                verbosity="low",
                reasoning_effort="minimal",
                max_tokens=300,
            )
        else:
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                verbosity="low",
                reasoning_effort="minimal",
            )
    else:
        # For other models (gpt-4o, gpt-4o-mini, etc.)
        if simple_mode:
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=300,
            )
        else:
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
            )
    
    return llm

def create_climate_prompt(simple_mode=False, month="July"):
    """Create prompt template for climate data requests"""
    
    if simple_mode:
        prompt_template = f"""You are a climate data expert. Given the location coordinates and address information below, provide the mean {month} temperature for the period 1991-2020.

Location Information:
- Longitude: {{longitude}}
- Latitude: {{latitude}}
- Country: {{country}}
- State/Region: {{state}}
- City: {{city}}

Provide ONLY the mean {month} temperature at 2m above surface (°C) for this location for the climatological period 1991-2020.

IMPORTANT: Return ONLY a single number (float) representing the mean {month} temperature in Celsius. No text, no JSON, just the number.

Example: 25.4"""
    else:
        prompt_template = """You are a climate data expert. Given the location coordinates and address information below, provide climatological mean values for temperature and precipitation for the period 1991-2020.

Location Information:
- Longitude: {longitude}
- Latitude: {latitude}
- Country: {country}
- State/Region: {state}
- City: {city}

Please provide the following climate data for this location:
1. Temperature at 2m above surface (°C) - monthly climatological means, minimums and maximums for 1991-2020
2. Total precipitation (mm/day) - monthly climatological means, minimums and maximums for 1991-2020

For each month (January through December), provide:
- mean: average value
- min: minimum value 
- max: maximum value

IMPORTANT: return only JSON object nothing else!!!!!!!!!!!!!!!!!!!!!!!

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

def validate_and_parse_response(response_text: str, simple_mode: bool = False, month: str = "July") -> Optional[Dict]:
    """Validate and parse LLM response"""
    try:
        response_text = response_text.strip()
        
        if simple_mode:
            # For simple mode, expect just a number
            try:
                temperature = float(response_text)
                # Basic sanity check for temperature range
                if -100 <= temperature <= 60:  # Reasonable temperature range in Celsius
                    return {f'{month.lower()}_temp_mean': temperature}
                else:
                    return None
            except ValueError:
                return None
        
        else:
            # Original JSON validation for full mode
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

def convert_to_numpy_arrays(climate_data: Dict, simple_mode: bool = False) -> Dict:
    """Convert climate data to numpy arrays for easier analysis"""
    if simple_mode:
        # For simple mode, just return the temperature value (any month)
        # Find the key that ends with '_temp_mean'
        temp_key = next((k for k in climate_data.keys() if k.endswith('_temp_mean')), None)
        if temp_key:
            return {temp_key: climate_data.get(temp_key, np.nan)}
        else:
            return {'temp_mean': np.nan}
    
    else:
        # Original conversion for full mode
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

def query_climate_data(llm, prompt_template, point_data: Dict, max_retries: int = 3, simple_mode: bool = False, month: str = "July") -> Optional[Dict]:
    """Query LLM for climate data with retry logic (single request)"""
    
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
            parsed_data = validate_and_parse_response(response_text, simple_mode, month)
            
            if parsed_data is not None:
                return {
                    'raw_response': response_text,
                    'parsed_data': parsed_data,
                    'numpy_arrays': convert_to_numpy_arrays(parsed_data, simple_mode),
                    'attempt': attempt + 1
                }
            else:
                validation_msg = "Invalid number response" if simple_mode else "Invalid JSON response"
                print(f"  Attempt {attempt + 1}: {validation_msg}, retrying...")
                
        except Exception as e:
            print(f"  Attempt {attempt + 1}: Error querying LLM: {e}")
        
    
    print(f"  Failed to get valid response after {max_retries} attempts")
    return None

def query_climate_data_batch(llm, prompt_template, point_data: Dict, num_repeats: int = 10, simple_mode: bool = False, month: str = "July") -> List[Optional[Dict]]:
    """Query LLM for climate data using batch processing"""
    
    # Prepare location info
    longitude = point_data.get('lon', 'N/A')
    latitude = point_data.get('lat', 'N/A')
    country = point_data.get('country', 'N/A') if point_data.get('country') else 'N/A'
    state = point_data.get('state', 'N/A') if point_data.get('state') else 'N/A'
    city = point_data.get('city', 'N/A') if point_data.get('city') else 'N/A'
    
    # Create the same prompt for all repeats
    messages = prompt_template.format_messages(
        longitude=longitude,
        latitude=latitude,
        country=country,
        state=state,
        city=city
    )
    
    # Create batch inputs (same prompt repeated)
    inputs = [messages] * num_repeats
    
    # Run batch query
    print(f"  Running {num_repeats} queries in parallel batch...")
    results = llm.batch(inputs, config={"max_concurrency": num_repeats})
    
    # Process results
    processed_results = []
    successful_responses = 0
    
    for i, response in enumerate(results):
        if response and hasattr(response, 'content'):
            response_text = response.content
            
            # Validate and parse response
            parsed_data = validate_and_parse_response(response_text, simple_mode, month)
            
            if parsed_data is not None:
                processed_results.append({
                    'raw_response': response_text,
                    'parsed_data': parsed_data,
                    'numpy_arrays': convert_to_numpy_arrays(parsed_data, simple_mode),
                    'batch_index': i + 1
                })
                successful_responses += 1
            else:
                processed_results.append(None)
        else:
            processed_results.append(None)
    
    print(f"  ✓ Batch completed: {successful_responses}/{num_repeats} successful responses")
    return processed_results

def process_climate_benchmark(mesh_file: str, num_repeats: int = 10, model_name: str = "gpt-5-nano", simple_mode: bool = False, month: str = "July", use_batch: bool = True, disable_tracing: bool = False, resume: bool = False):
    """Main function to process climate benchmark"""
    
    # Configure LangSmith tracing
    configure_langsmith(disable_tracing)
    
    print(f"Loading mesh data from {mesh_file}...")
    mesh_data = load_mesh_data(mesh_file)
    mesh_points = mesh_data['mesh_points']
    resolution = mesh_data['resolution']
    
    mode_str = f"Simple ({month} temp only)" if simple_mode else "Full (all months)"
    batch_str = "Batch processing" if use_batch else "Individual processing"
    print(f"Loaded {len(mesh_points)} points with {resolution}° resolution")
    print(f"Mode: {mode_str}")
    print(f"Processing: {batch_str}")
    print(f"Repeats per point: {num_repeats}")
    
    # Filter land points
    land_points = [point for point in mesh_points if point['is_land']]
    print(f"Found {len(land_points)} land points")
    
    # Check for resuming from intermediate file
    results = []
    start_index = 0
    
    if resume:
        latest_file = find_latest_intermediate_file(resolution, simple_mode)
        if latest_file:
            results, saved_mesh_data, start_index = load_intermediate_results(latest_file)
            print(f"Resuming from intermediate file: {latest_file}")
            print(f"Will continue from land point {start_index + 1}/{len(land_points)}")
        else:
            print("No intermediate files found, starting from beginning")
    else:
        print("Starting fresh processing")
    
    # Initialize LLM
    print(f"Initializing LLM: {model_name}")
    if simple_mode:
        print("Simple mode: Using max_tokens=10 for faster processing")
    llm = initialize_llm(model_name, simple_mode=simple_mode)
    prompt_template = create_climate_prompt(simple_mode, month)
    
    # Process each land point (starting from start_index if resuming)
    
    for i, point_data in enumerate(land_points[start_index:], start=start_index):
        print(f"\nProcessing land point {i+1}/{len(land_points)}: ({point_data['lat']:.1f}, {point_data['lon']:.1f})")
        if point_data.get('country'):
            print(f"  Location: {point_data['country']}, {point_data.get('state', 'N/A')}, {point_data.get('city', 'N/A')}")
        
        point_results = {
            'point_info': point_data,
            'llm_responses': []
        }
        
        if use_batch and num_repeats > 1:
            # Use batch processing for multiple repeats
            batch_responses = query_climate_data_batch(llm, prompt_template, point_data, num_repeats, simple_mode, month)
            point_results['llm_responses'] = batch_responses
            
        else:
            # Use individual processing (original method)
            for repeat in range(num_repeats):
                if num_repeats > 1:
                    print(f"  Query {repeat + 1}/{num_repeats}")
                
                climate_response = query_climate_data(llm, prompt_template, point_data, max_retries=3, simple_mode=simple_mode, month=month)
                
                if climate_response:
                    point_results['llm_responses'].append(climate_response)
                    print(f"  ✓ Successfully got climate data")
                else:
                    print(f"  ✗ Failed to get climate data")
                    point_results['llm_responses'].append(None)
        
        results.append(point_results)
        
        # Save intermediate results every 10 points
        if (i + 1) % 10 == 0:
            mode_suffix = "_simple" if simple_mode else ""
            intermediate_file = f"results/climate_results_intermediate_{i+1}{mode_suffix}.json"
            # Create results directory if it doesn't exist
            Path(intermediate_file).parent.mkdir(parents=True, exist_ok=True)
            save_results(results, mesh_data, intermediate_file, model_name, simple_mode, month, use_batch)
    
    return results, mesh_data

def save_results(results: List[Dict], mesh_data: Dict, output_file: str, model_name: str, simple_mode: bool = False, month: str = "July", use_batch: bool = True):
    """Save climate benchmark results"""
    print(f"Saving results to {output_file}...")
    
    output_data = {
        'mesh_info': mesh_data['mesh_info'],
        'resolution': mesh_data['resolution'],
        'total_land_points': len(results),
        'results': results,
        'metadata': {
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_used': model_name,
            'num_repeats_per_point': len(results[0]['llm_responses']) if results else 0,
            'simple_mode': simple_mode,
            'query_type': f'{month.lower()}_temp_only' if simple_mode else 'full_climate_data',
            'month': month if simple_mode else None,
            'batch_processing': use_batch
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
    num_repeats = 10
    model_name = 'gpt-5-nano'
    simple_mode = True
    month = "July"
    use_batch = True
    disable_tracing = False
    resume = False
    
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    if len(sys.argv) > 2:
        num_repeats = int(sys.argv[2])
    if len(sys.argv) > 3:
        model_name = sys.argv[3]
    if len(sys.argv) > 4:
        arg4 = sys.argv[4].lower()
        if arg4 in ['false', '0', 'no', 'full']:
            simple_mode = False
        elif arg4 in ['true', '1', 'yes', 'simple']:
            simple_mode = True
    if len(sys.argv) > 5:
        month = sys.argv[5].capitalize()
    if len(sys.argv) > 6:
        arg6 = sys.argv[6].lower()
        if arg6 in ['false', '0', 'no', 'individual']:
            use_batch = False
        elif arg6 in ['true', '1', 'yes', 'batch']:
            use_batch = True
    if len(sys.argv) > 7:
        arg7 = sys.argv[7].lower()
        if arg7 in ['true', '1', 'yes', 'disable', 'no-trace']:
            disable_tracing = True
        elif arg7 in ['false', '0', 'no', 'enable', 'trace']:
            disable_tracing = False
    if len(sys.argv) > 8:
        arg8 = sys.argv[8].lower()
        if arg8 in ['true', '1', 'yes', 'resume']:
            resume = True
        elif arg8 in ['false', '0', 'no', 'fresh']:
            resume = False
    
    print(f"Climate LLM Benchmark")
    print(f"Mesh file: {mesh_file}")
    print(f"Repeats per point: {num_repeats}")
    print(f"Model: {model_name}")
    print(f"Mode: {f'Simple ({month} temp only)' if simple_mode else 'Full (all months)'}")
    print(f"Processing: {'Batch' if use_batch else 'Individual'}")
    print(f"LangSmith tracing: {'Disabled' if disable_tracing else 'Enabled'}")
    print(f"Resume mode: {'Enabled' if resume else 'Disabled'}")
    
    # Check if mesh file exists
    if not Path(mesh_file).exists():
        print(f"Error: Mesh file '{mesh_file}' not found.")
        print("Please provide a valid mesh file or run geo_mesh_processor.py first.")
        return
    
    try:
        # Process the benchmark
        results, mesh_data = process_climate_benchmark(mesh_file, num_repeats, model_name, simple_mode, month, use_batch, disable_tracing, resume)
        
        # Save final results
        resolution = mesh_data['resolution']
        mode_suffix = "_simple" if simple_mode else ""
        output_file = f"results/climate_results_{resolution}deg_r{num_repeats}{mode_suffix}.json"
        # Create results directory if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        save_results(results, mesh_data, output_file, model_name, simple_mode, month, use_batch)
        
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