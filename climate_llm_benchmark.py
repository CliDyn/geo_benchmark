import numpy as np
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import re
from geo_mesh_processor import load_mesh_data

# Core langchain imports
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
import glob
import yaml

# Model provider imports (with optional handling)
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_ollama import ChatOllama
    from langchain_community.llms import Ollama
    #from langchain_community.chat_models import ChatOllama
except ImportError:
    Ollama = None
    ChatOllama = None

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    if not Path(config_path).exists():
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return {}

def get_config_value(config: Dict, key_path: str, default=None):
    """Get nested configuration value using dot notation"""
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

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

def find_latest_intermediate_file(resolution: str, simple_mode: bool = False, chunk_id: Optional[str] = None, with_address: bool = True, periods: bool = False, month: str = "July", monthly: bool = False, sub_suffix: str = "", model_name: Optional[str] = None) -> Optional[str]:
    """Find the latest intermediate file for resuming (keyed on model name to avoid cross-model collisions)"""
    mode_suffix = "_simple" if simple_mode else ""
    month_suffix = build_month_suffix(month, monthly)
    address_suffix = "_noaddress" if not with_address else ""
    periods_suffix = "_climchange" if periods else ""
    chunk_suffix = f"_chunk_{chunk_id}" if chunk_id else ""
    model_token = f"_{model_name.replace(':', '_').replace('/', '_').replace('.', '_')}" if model_name else ""
    pattern = f"results/climate_results_intermediate_*{model_token}{sub_suffix}{chunk_suffix}*{mode_suffix}{month_suffix}{address_suffix}{periods_suffix}.json"
    
    # Find all matching intermediate files
    intermediate_files = glob.glob(pattern)
    
    if not intermediate_files:
        return None
    
    # Extract numbers and sort to find the latest
    file_numbers = []
    for file_path in intermediate_files:
        try:
            # Extract number from filename like "climate_results_intermediate_1840_chunk_01_simple.json"
            filename = Path(file_path).stem
            
            # Remove prefixes and suffixes to extract the number
            temp_name = filename.replace("climate_results_intermediate_", "")
            if chunk_id:
                temp_name = temp_name.replace(f"_chunk_{chunk_id}", "")
            if simple_mode:
                temp_name = temp_name.replace("_simple", "")
            if periods:
                temp_name = temp_name.replace("_climchange", "")
            
            # Extract any remaining model name parts and the number
            parts = temp_name.split("_")
            # The number should be the first part
            number = int(parts[0])
            file_numbers.append((number, file_path))
        except (ValueError, IndexError):
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

def initialize_llm(config: Dict, model_name: str = None, temperature: float = None, simple_mode: bool = None):
    """Initialize LLM based on configuration and provider"""
    
    # Get values from config or use provided values
    provider = get_config_value(config, 'model.provider', 'openai')
    model_name = model_name or get_config_value(config, 'model.name', 'gpt-5-nano')
    temperature = temperature if temperature is not None else get_config_value(config, 'model.temperature', 0)
    max_tokens = get_config_value(config, 'model.max_tokens', 300 if simple_mode else None)
    timeout = get_config_value(config, 'model.timeout', 30)
    
    print(f"Initializing {provider} model: {model_name} (temperature: {'provider default' if temperature is None else temperature})")
    
    if provider == "openai":
        if ChatOpenAI is None:
            raise ImportError("langchain-openai not installed. Install with: pip install langchain-openai")
        
        api_key_env = get_config_value(config, 'providers.openai.api_key_env', 'OPENAI_API_KEY')
        base_url = get_config_value(config, 'providers.openai.base_url')
        organization = get_config_value(config, 'providers.openai.organization')
        
        llm_kwargs = {
            'model': model_name,
            'request_timeout': timeout,
        }
        if temperature is not None:
            llm_kwargs['temperature'] = temperature
        
        if max_tokens:
            llm_kwargs['max_tokens'] = max_tokens
        if base_url:
            llm_kwargs['base_url'] = base_url
        if organization:
            llm_kwargs['organization'] = organization
        
        # Special handling for GPT-5 models
        if "gpt-5" in model_name:
            llm_kwargs.update({
                'verbosity': "low",
                'reasoning_effort': "minimal",
            })
        
        return ChatOpenAI(**llm_kwargs)
    
    elif provider == "anthropic":
        if ChatAnthropic is None:
            raise ImportError("langchain-anthropic not installed. Install with: pip install langchain-anthropic")
        
        api_key_env = get_config_value(config, 'providers.anthropic.api_key_env', 'ANTHROPIC_API_KEY')
        base_url = get_config_value(config, 'providers.anthropic.base_url')
        
        llm_kwargs = {
            'model': model_name,
            'timeout': timeout,
        }
        if temperature is not None:
            llm_kwargs['temperature'] = temperature
        
        if max_tokens:
            llm_kwargs['max_tokens'] = max_tokens
        if base_url:
            llm_kwargs['base_url'] = base_url
        
        return ChatAnthropic(**llm_kwargs)
    
    elif provider == "google":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai not installed. Install with: pip install langchain-google-genai")
        
        # Retrieve API key configuration.
        # We support two patterns for backward compatibility:
        # 1. providers.google.api_key -> contains the actual key (preferred, NOT committed!)
        # 2. providers.google.api_key_env -> contains either the *name* of the env var (e.g. GOOGLE_API_KEY)
        #    or (legacy / current file) the raw key starting with 'AIza'.
        api_key_direct = get_config_value(config, 'providers.google.api_key')
        api_key_config = get_config_value(config, 'providers.google.api_key_env', 'GOOGLE_API_KEY')

        api_key = None
        if api_key_direct:
            api_key = api_key_direct.strip()
        else:
            # If the value looks like an API key (starts with AIza) treat it as the key, otherwise as env var name
            if isinstance(api_key_config, str) and api_key_config.startswith('AIza'):
                api_key = api_key_config.strip()
                print("Warning: Detected a raw Google API key in 'api_key_env'. Consider moving it to an environment variable 'GOOGLE_API_KEY' and setting providers.google.api_key_env: GOOGLE_API_KEY")
            else:
                env_var_name = api_key_config or 'GOOGLE_API_KEY'
                api_key = os.environ.get(env_var_name)
                if not api_key:
                    raise ValueError(f"Google API key not found. Set env var '{env_var_name}' or add 'providers.google.api_key' in config.yaml (do NOT commit the key).")

        llm_kwargs = {
            'model': model_name,
            'api_key': api_key,
        }
        if temperature is not None:
            llm_kwargs['temperature'] = temperature

        if max_tokens:
            llm_kwargs['max_output_tokens'] = max_tokens

        return ChatGoogleGenerativeAI(**llm_kwargs)
    
    elif provider == "ollama":
        if ChatOllama is None:
            raise ImportError("langchain-community not installed. Install with: pip install langchain-community")
        
        base_url = get_config_value(config, 'providers.ollama.base_url', 'http://localhost:11434')
        ollama_timeout = get_config_value(config, 'providers.ollama.timeout', 60)
        
        llm_kwargs = {
            'model': model_name,
            'base_url': base_url,
            'timeout': ollama_timeout,
        }
        if temperature is not None:
            llm_kwargs['temperature'] = temperature
        
        if max_tokens:
            llm_kwargs['num_predict'] = max_tokens
        
        return ChatOllama(**llm_kwargs)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: openai, anthropic, google, ollama")

def create_climate_prompt(simple_mode=False, month="July", with_address=True, periods=False, monthly=False):
    """Create prompt template for climate data requests"""

    # Check for invalid combination
    if periods and not simple_mode:
        raise ValueError("Error: periods=True is only supported when simple_mode=True")
    if monthly and (periods or not simple_mode):
        raise ValueError("Error: monthly=True requires simple_mode=True and periods=False")

    if monthly:
        # Monthly mode - all 12 monthly means in a single query
        if with_address:
            prompt_template = """You are a climate data expert. Given the location coordinates and address information below, provide the mean monthly temperatures for the period 1991-2020.

Location Information:
- Longitude: {longitude}
- Latitude: {latitude}
- Country: {country}
- State/Region: {state}
- City: {city}

Provide ONLY the mean monthly temperatures at 2m above surface (°C) for this location for the climatological period 1991-2020, for all 12 months from January to December.

IMPORTANT: Return ONLY 12 comma-separated numbers (floats) representing the mean monthly temperatures in Celsius, in order from January to December. No text, no JSON, just the numbers.

Example: -3.25, -1.87, 4.12, 9.65, 15.32, 19.78, 22.41, 21.93, 16.55, 10.02, 3.41, -1.96"""
        else:
            prompt_template = """You are a climate data expert. Given the location coordinates below, provide the mean monthly temperatures for the period 1991-2020.

Location Information:
- Longitude: {longitude}
- Latitude: {latitude}

Provide ONLY the mean monthly temperatures at 2m above surface (°C) for this location for the climatological period 1991-2020, for all 12 months from January to December.

IMPORTANT: Return ONLY 12 comma-separated numbers (floats) representing the mean monthly temperatures in Celsius, in order from January to December. No text, no JSON, just the numbers.

Example: -3.25, -1.87, 4.12, 9.65, 15.32, 19.78, 22.41, 21.93, 16.55, 10.02, 3.41, -1.96"""
    elif simple_mode:
        if periods:
            # Climate change mode - temperature difference between periods
            if with_address:
                prompt_template = f"""You are a climate data expert. Given the location coordinates and address information below, calculate the change in mean {month} temperature between the periods 1950–1974 and 2000–2024.

Location Information:
- Longitude: {{longitude}}
- Latitude: {{latitude}}
- Country: {{country}}
- State/Region: {{state}}
- City: {{city}}

Provide ONLY the temperature difference (2000–2024 minus 1950–1974) at 2 m above surface (°C) for this location.

IMPORTANT: Return ONLY a single number (float) representing the temperature difference in Celsius. No text, no JSON, just the number.

Example: 1.8"""
            else:
                prompt_template = f"""You are a climate data expert. Given the location coordinates below, calculate the change in mean {month} temperature between the periods 1950–1974 and 2000–2024.

Location Information:
- Longitude: {{longitude}}
- Latitude: {{latitude}}

Provide ONLY the temperature difference (2000–2024 minus 1950–1974) at 2 m above surface (°C) for this location.

IMPORTANT: Return ONLY a single number (float) representing the temperature difference in Celsius. No text, no JSON, just the number.

Example: 1.8"""
        else:
            # Original climate mode - absolute temperature
            if with_address:
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
                prompt_template = f"""You are a climate data expert. Given the location coordinates below, provide the mean {month} temperature for the period 1991-2020.

Location Information:
- Longitude: {{longitude}}
- Latitude: {{latitude}}

Provide ONLY the mean {month} temperature at 2m above surface (°C) for this location for the climatological period 1991-2020.

IMPORTANT: Return ONLY a single number (float) representing the mean {month} temperature in Celsius. No text, no JSON, just the number.

Example: 25.4"""
    else:
        if with_address:
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
        else:
            prompt_template = """You are a climate data expert. Given the location coordinates below, provide climatological mean values for temperature and precipitation for the period 1991-2020.

Location Information:
- Longitude: {longitude}
- Latitude: {latitude}

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

def extract_first_float(text: str) -> float:
    """Extracts the first float number from a string after removing any <think>...</think> blocks.
       Returns NaN if no numbers are found."""
    # rimuovi blocchi di reasoning (deepseek-r1 / qwen reasoning ecc.)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else float("nan")


def extract_last_float_after_newlines(text: str) -> float:
    """Extracts the last float number that appears after double newlines (\\n\\n).
       Special function for Mistral models that put thinking before \\n\\n and the answer after.
       Returns NaN if no numbers are found."""
    # Find the last occurrence of \n\n
    parts = text.split('\n\n')
    if len(parts) > 1:
        # Get the last part after the final \n\n
        last_part = parts[-1].strip()
        # Extract float from this last part
        m = re.search(r"[-+]?\d+(?:\.\d+)?", last_part)
        return float(m.group(0)) if m else float("nan")
    else:
        return float("nan")


def parse_monthly_response(response_text: str) -> Optional[Dict]:
    """Parse a monthly-mode response: 12 numbers (January..December).
       Tolerates <think> blocks and out-of-range junk such as echoed years;
       requires exactly 12 plausible values (-100..60 C). Returns None otherwise."""
    text = re.sub(r"<think>.*?</think>", "", response_text or "", flags=re.DOTALL | re.IGNORECASE)
    values = [float(m) for m in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]
    if len(values) != 12:
        values = [v for v in values if -100 <= v <= 60]
    if len(values) != 12:
        return None
    if not all(-100 <= v <= 60 for v in values):
        return None
    return {"monthly_temp_mean": values}


def format_progress(start_index: int, done: int, total: int, successful_queries: int) -> str:
    """One-line progress summary for periodic logging during a run."""
    pct = (done / total * 100) if total else 0.0
    return (f"[progress] {done}/{total} points ({pct:.1f}%) | "
            f"this run started at point {start_index + 1} | "
            f"{successful_queries} successful queries so far")


def build_month_suffix(month: str, monthly: bool) -> str:
    """Filename token: '_monthly' for all-months mode, '_<Month>' for a single month
       (every month, including July, is named explicitly)."""
    if monthly:
        return "_monthly"
    return f"_{month}"


def validate_and_parse_response(
    response_text: str,
    simple_mode: bool = False,
    month: str = "July",
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    monthly: bool = False,
) -> Optional[Dict]:
    """Validate and parse LLM response.
       Updated: for provider 'ollama' with models starting with 'qwen' or 'deepseek' (e.g. deepseek-r1), 
       it cleans and extracts the first number even if the model returns additional text or <think> blocks."""
    try:
        raw = response_text or ""
        response_text = raw.strip()

        if monthly:
            return parse_monthly_response(response_text)

        if simple_mode:
            temperature: Optional[float] = None

            # Caso speciale: reasoning / output prolisso (qwen, deepseek, mistral via ollama)
            if provider == "ollama" and model_name:
                lowered = model_name.lower()
                if lowered.startswith("qwen") or lowered.startswith("deepseek"):
                    val = extract_first_float(response_text)
                    if not (val != val):  # check not NaN
                        temperature = val
                elif lowered.startswith("mistral"):
                    try:
                        temperature = float(response_text)
                    except ValueError:
                        val = extract_last_float_after_newlines(response_text)
                        if not (val != val):  # check not NaN
                            temperature = val

            # Fallback: prova conversione diretta
            if temperature is None:
                try:
                    temperature = float(response_text)
                except ValueError:
                    pass  # If not a valid float, temperature remains None

            if temperature is None:
                return None

            # Range plausibile
            if -100 <= temperature <= 60:
                return {f"{month.lower()}_temp_mean": temperature}
            return None

        # Full mode JSON
        # Rimuove blocchi markdown json
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        data = json.loads(response_text)
        required_keys = ["temperature_2m_celsius", "precipitation_mm_per_day"]
        months = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
        for key in required_keys:
            if key not in data:
                return None
            for m in months:
                if m not in data[key]:
                    return None
                if not all(stat in data[key][m] for stat in ["mean", "min", "max"]):
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

def query_climate_data(llm, prompt_template, point_data: Dict, max_retries: int = 3, simple_mode: bool = False, month: str = "July", provider: Optional[str] = None, model_name: Optional[str] = None, with_address: bool = True, periods: bool = False, monthly: bool = False) -> Optional[Dict]:
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
            if with_address:
                messages = prompt_template.format_messages(
                    longitude=longitude,
                    latitude=latitude,
                    country=country,
                    state=state,
                    city=city
                )
            else:
                messages = prompt_template.format_messages(
                    longitude=longitude,
                    latitude=latitude
                )
            
            # Query the LLM
            response = llm.invoke(messages)
            response_text = response.content
            resp_meta = getattr(response, 'response_metadata', None) or {}

            # Validate and parse response
            parsed_data = validate_and_parse_response(response_text, simple_mode, month, provider=provider, model_name=model_name, monthly=monthly)

            if parsed_data is not None:
                return {
                    'raw_response': response_text,
                    'parsed_data': parsed_data,
                    'numpy_arrays': convert_to_numpy_arrays(parsed_data, simple_mode),
                    'model_version': resp_meta.get('model_name') or resp_meta.get('model'),
                    'attempt': attempt + 1
                }
            else:
                validation_msg = "Invalid number response" if simple_mode else "Invalid JSON response"
                print(f"  Attempt {attempt + 1}: {validation_msg}, retrying...")
                
        except Exception as e:
            print(f"  Attempt {attempt + 1}: Error querying LLM: {e}")
        
    
    print(f"  Failed to get valid response after {max_retries} attempts")
    return None

def query_climate_data_batch(llm, prompt_template, point_data: Dict, config: Dict, num_repeats: int = 10, simple_mode: bool = False, month: str = "July", provider: Optional[str] = None, model_name: Optional[str] = None, with_address: bool = True, periods: bool = False, monthly: bool = False) -> List[Optional[Dict]]:
    """Query LLM for climate data using batch processing"""
    
    # Prepare location info
    longitude = point_data.get('lon', 'N/A')
    latitude = point_data.get('lat', 'N/A')
    country = point_data.get('country', 'N/A') if point_data.get('country') else 'N/A'
    state = point_data.get('state', 'N/A') if point_data.get('state') else 'N/A'
    city = point_data.get('city', 'N/A') if point_data.get('city') else 'N/A'
    
    # Create the same prompt for all repeats
    if with_address:
        messages = prompt_template.format_messages(
            longitude=longitude,
            latitude=latitude,
            country=country,
            state=state,
            city=city
        )
    else:
        messages = prompt_template.format_messages(
            longitude=longitude,
            latitude=latitude
        )
    
    # Get max concurrency from config
    max_concurrency = get_config_value(config, 'batch.max_concurrency', num_repeats)
    
    # Check if the LLM supports batch processing
    provider = get_config_value(config, 'model.provider', 'openai')
    
    if hasattr(llm, 'batch') and provider in ['openai', 'anthropic']:
        # Use native batch processing for supported providers
        # Create batch inputs (same prompt repeated)
        inputs = [messages] * num_repeats
        
        # Run batch query
        print(f"  Running {num_repeats} queries in parallel batch...")
        results = llm.batch(inputs, config={"max_concurrency": max_concurrency})
        
        # Process results
        processed_results = []
        successful_responses = 0
        
        for i, response in enumerate(results):
            if response and hasattr(response, 'content'):
                response_text = response.content
                resp_meta = getattr(response, 'response_metadata', None) or {}

                # Validate and parse response
                parsed_data = validate_and_parse_response(response_text, simple_mode, month, provider=provider, model_name=model_name, monthly=monthly)

                if parsed_data is not None:
                    processed_results.append({
                        'raw_response': response_text,
                        'parsed_data': parsed_data,
                        'numpy_arrays': convert_to_numpy_arrays(parsed_data, simple_mode),
                        'model_version': resp_meta.get('model_name') or resp_meta.get('model'),
                        'batch_index': i + 1
                    })
                    successful_responses += 1
                else:
                    processed_results.append(None)
            else:
                processed_results.append(None)
        
        print(f"  ✓ Batch completed: {successful_responses}/{num_repeats} successful responses")
        return processed_results
    
    else:
        # Fallback to individual queries for providers that don't support batch processing
        print(f"  Running {num_repeats} individual queries (batch not supported for {provider})...")
        processed_results = []
        successful_responses = 0
        
        for i in range(num_repeats):
            try:
                response = llm.invoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)
                resp_meta = getattr(response, 'response_metadata', None) or {}

                # Validate and parse response
                parsed_data = validate_and_parse_response(response_text, simple_mode, month, provider=provider, model_name=model_name, monthly=monthly)

                if parsed_data is not None:
                    processed_results.append({
                        'raw_response': response_text,
                        'parsed_data': parsed_data,
                        'numpy_arrays': convert_to_numpy_arrays(parsed_data, simple_mode),
                        'model_version': resp_meta.get('model_name') or resp_meta.get('model'),
                        'batch_index': i + 1
                    })
                    successful_responses += 1
                else:
                    processed_results.append(None)
            
            except Exception as e:
                print(f"    Query {i+1} failed: {e}")
                processed_results.append(None)
        
        print(f"  ✓ Individual queries completed: {successful_responses}/{num_repeats} successful responses")
        return processed_results

def process_climate_benchmark(config: Dict, mesh_file: str = None):
    """Main function to process climate benchmark"""
    
    # Get all values from config
    if mesh_file is None:
        mesh_file = get_config_value(config, 'benchmark.mesh_file', 'meshes/mesh_data_10deg.json')
    num_repeats = get_config_value(config, 'benchmark.num_repeats', 10)
    model_name = get_config_value(config, 'model.name', 'gpt-5-nano')
    simple_mode = get_config_value(config, 'benchmark.simple_mode', True)
    month = get_config_value(config, 'benchmark.month', 'July')
    with_address = get_config_value(config, 'benchmark.with_address', True)
    periods = get_config_value(config, 'benchmark.periods', False)
    monthly = get_config_value(config, 'benchmark.monthly', False)
    gen_temperature = get_config_value(config, 'model.temperature', 0)
    use_batch = get_config_value(config, 'benchmark.use_batch', True)
    disable_tracing = get_config_value(config, 'benchmark.disable_tracing', False)
    resume = get_config_value(config, 'benchmark.resume', False)

    # Configure LangSmith tracing
    configure_langsmith(disable_tracing)
    
    print(f"Loading mesh data from {mesh_file}...")
    mesh_data = load_mesh_data(mesh_file)
    mesh_points = mesh_data['mesh_points']
    resolution = mesh_data['resolution']
    
    mode_str = "Monthly (all 12 months, one query)" if monthly else (f"Simple ({month} temp only)" if simple_mode else "Full (all months)")
    batch_str = "Batch processing" if use_batch else "Individual processing"
    provider = get_config_value(config, 'model.provider', 'openai')
    print(f"Loaded {len(mesh_points)} points with {resolution}° resolution")
    print(f"Provider: {provider}")
    print(f"Mode: {mode_str}")
    print(f"Processing: {batch_str}")
    print(f"Repeats per point: {num_repeats}")
    
    # Filter land points
    land_points = [point for point in mesh_points if point['is_land']]
    print(f"Found {len(land_points)} land points")
    
    # Subsampled mesh -> distinct filename token (protects full-grid results from overwrite)
    sub_k = mesh_data.get('mesh_info', {}).get('subsample_k')
    sub_suffix = f"_sub{sub_k}" if sub_k else ""

    # Extract chunk information from mesh file if it's a chunk
    chunk_id = None
    if 'chunk_id' in mesh_data.get('mesh_info', {}):
        chunk_id = f"{mesh_data['mesh_info']['chunk_id']:02d}"
        total_chunks = mesh_data['mesh_info'].get('total_chunks', 'unknown')
        print(f"Processing chunk {chunk_id} of {total_chunks}")
    
    # Check for resuming from intermediate file
    results = []
    start_index = 0
    
    if resume:
        latest_file = find_latest_intermediate_file(resolution, simple_mode, chunk_id, with_address, periods, month=month, monthly=monthly, sub_suffix=sub_suffix, model_name=model_name)
        if latest_file:
            results, saved_mesh_data, start_index = load_intermediate_results(latest_file)
            print(f"Resuming from intermediate file: {latest_file}")
            print(f"Will continue from land point {start_index + 1}/{len(land_points)}")
        else:
            print("No intermediate files found, starting from beginning")
    else:
        print("Starting fresh processing")
    
    # Initialize LLM
    llm = initialize_llm(config, model_name, simple_mode=simple_mode)
    prompt_template = create_climate_prompt(simple_mode, month, with_address, periods, monthly=monthly)
    
    # Get save interval from config
    save_interval = get_config_value(config, 'batch.save_interval', 10)

    print(f"\n=== Run starts at land point {start_index + 1}/{len(land_points)} "
          f"({len(land_points) - start_index} points to process, {num_repeats} repeats each, "
          f"progress every {save_interval}) ===")

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
            batch_responses = query_climate_data_batch(llm, prompt_template, point_data, config, num_repeats, simple_mode, month, provider=provider, model_name=model_name, with_address=with_address, periods=periods, monthly=monthly)
            point_results['llm_responses'] = batch_responses
            
        else:
            # Use individual processing (original method)
            max_retries = get_config_value(config, 'model.max_retries', 3)
            for repeat in range(num_repeats):
                if num_repeats > 1:
                    print(f"  Query {repeat + 1}/{num_repeats}")
                
                climate_response = query_climate_data(llm, prompt_template, point_data, max_retries=max_retries, simple_mode=simple_mode, month=month, provider=provider, model_name=model_name, with_address=with_address, periods=periods, monthly=monthly)
                
                if climate_response:
                    point_results['llm_responses'].append(climate_response)
                    print(f"  ✓ Successfully got climate data")
                else:
                    print(f"  ✗ Failed to get climate data")
                    point_results['llm_responses'].append(None)
        
        results.append(point_results)
        
        # Save intermediate results at configured interval
        if (i + 1) % save_interval == 0:
            mode_suffix = "_simple" if simple_mode else ""
            month_suffix = build_month_suffix(month, monthly)
            address_suffix = "_noaddress" if not with_address else ""
            periods_suffix = "_climchange" if periods else ""
            chunk_suffix = f"_chunk_{chunk_id}" if chunk_id else ""
            # Clean model name for filename (replace special characters)
            clean_model_name = model_name.replace(":", "_").replace("/", "_").replace(".", "_")
            results_dir = get_config_value(config, 'output.results_dir', 'results')
            intermediate_file = f"{results_dir}/climate_results_intermediate_{i+1}_{clean_model_name}{sub_suffix}{chunk_suffix}{mode_suffix}{month_suffix}{address_suffix}{periods_suffix}.json"
            # Create results directory if it doesn't exist
            Path(intermediate_file).parent.mkdir(parents=True, exist_ok=True)
            save_results(results, mesh_data, intermediate_file, model_name, simple_mode, month, use_batch, periods, monthly=monthly, temperature=gen_temperature)

            successful_queries = sum(1 for r in results for resp in r['llm_responses'] if resp)
            print(format_progress(start_index, i + 1, len(land_points), successful_queries))

    return results, mesh_data

def save_results(results: List[Dict], mesh_data: Dict, output_file: str, model_name: str, simple_mode: bool = False, month: str = "July", use_batch: bool = True, periods: bool = False, monthly: bool = False, temperature=0):
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
            'monthly': monthly,
            'query_type': 'monthly_temps_jan_dec' if monthly else (f'{month.lower()}_temp_only' if simple_mode else 'full_climate_data'),
            'month': None if monthly else (month if simple_mode else None),
            'temperature': 'provider_default' if temperature is None else temperature,
            'batch_processing': use_batch
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"Results saved to {output_file}")

def main():
    """Main function"""
    import sys
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Climate LLM Benchmark')
    parser.add_argument('--config', default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--chunk', type=int, 
                       help='Chunk number to process (enables chunk mode)')
    parser.add_argument('--base_url',
                       help='Override base URL for model provider (e.g., http://localhost:11434)')
    parser.add_argument('--month',
                       help='Override benchmark.month from config (e.g., February) for single-month runs')
    
    # Support legacy positional arguments for backward compatibility
    parser.add_argument('legacy_args', nargs='*', 
                       help='Legacy: [config_file] [chunk_number]')
    
    args = parser.parse_args()
    
    # Handle legacy argument format for backward compatibility
    config_file = args.config
    chunk_number = args.chunk
    base_url_override = args.base_url
    
    if args.legacy_args:
        # Legacy format: python climate_llm_benchmark.py [config_file] [chunk_number]
        if len(args.legacy_args) == 1:
            # Could be config file or chunk number
            arg = args.legacy_args[0]
            try:
                chunk_number = int(arg)
            except ValueError:
                config_file = arg
        elif len(args.legacy_args) == 2:
            config_file = args.legacy_args[0]
            chunk_number = int(args.legacy_args[1])
    
    config = load_config(config_file)

    # Apply month override if provided (handy for looping single-month runs)
    if args.month:
        config.setdefault('benchmark', {})['month'] = args.month
        print(f"Override month: {args.month}")

    # Apply base_url override if provided
    if base_url_override:
        provider = get_config_value(config, 'model.provider', 'openai')
        if provider == 'ollama':
            # Override Ollama base_url
            if 'providers' not in config:
                config['providers'] = {}
            if 'ollama' not in config['providers']:
                config['providers']['ollama'] = {}
            config['providers']['ollama']['base_url'] = base_url_override
            print(f"Override Ollama base_url: {base_url_override}")
        elif provider == 'openai':
            # Override OpenAI base_url
            if 'providers' not in config:
                config['providers'] = {}
            if 'openai' not in config['providers']:
                config['providers']['openai'] = {}
            config['providers']['openai']['base_url'] = base_url_override
            print(f"Override OpenAI base_url: {base_url_override}")
        elif provider == 'anthropic':
            # Override Anthropic base_url
            if 'providers' not in config:
                config['providers'] = {}
            if 'anthropic' not in config['providers']:
                config['providers']['anthropic'] = {}
            config['providers']['anthropic']['base_url'] = base_url_override
            print(f"Override Anthropic base_url: {base_url_override}")
        else:
            print(f"Warning: --base_url override not supported for provider '{provider}', ignoring")
    
    # Get all values from config
    chunk_mode = get_config_value(config, 'benchmark.chunk_mode', False) or (chunk_number is not None)
    chunks_dir = get_config_value(config, 'benchmark.chunks_dir', 'meshes/chunks')
    chunks_pattern = get_config_value(config, 'benchmark.chunks_pattern', 'mesh_data_1.0deg_chunk_{:02d}_of_{:02d}.json')
    base_mesh_file = get_config_value(config, 'benchmark.mesh_file', 'meshes/mesh_data_10deg.json')
    
    # Determine mesh file based on chunk mode
    if chunk_mode and chunk_number is not None:
        # Find total chunks by looking for existing chunk files
        import glob
        chunk_pattern_search = chunks_pattern.replace('{:02d}', '*')
        chunk_files = glob.glob(f"{chunks_dir}/{chunk_pattern_search}")
        if not chunk_files:
            print(f"Error: No chunk files found in {chunks_dir} matching pattern {chunk_pattern_search}")
            return
        
        # Extract total chunks from first file found
        total_chunks = len(chunk_files)
        mesh_file = f"{chunks_dir}/{chunks_pattern.format(chunk_number, total_chunks)}"
        
        if not Path(mesh_file).exists():
            print(f"Error: Chunk file '{mesh_file}' not found.")
            print(f"Available chunks: {sorted([Path(f).name for f in chunk_files])}")
            return
            
        print(f"Chunk mode enabled: Processing chunk {chunk_number} of {total_chunks}")
        
    elif chunk_mode and chunk_number is None:
        print("Error: Chunk mode is enabled but no chunk number specified.")
        print("Usage: python climate_llm_benchmark.py --chunk=N")
        print("   or: python climate_llm_benchmark.py --config=config.yaml --chunk=N")
        print("Legacy: python climate_llm_benchmark.py [config_file] chunk_number")
        return
        
    else:
        mesh_file = base_mesh_file
        if chunk_number is not None:
            print("Warning: Chunk number specified but chunk_mode is disabled in config. Using regular mesh file.")
    
    num_repeats = get_config_value(config, 'benchmark.num_repeats', 10)
    model_name = get_config_value(config, 'model.name', 'gpt-5-nano')
    simple_mode = get_config_value(config, 'benchmark.simple_mode', True)
    month = get_config_value(config, 'benchmark.month', 'July')
    use_batch = get_config_value(config, 'benchmark.use_batch', True)
    disable_tracing = get_config_value(config, 'benchmark.disable_tracing', False)
    resume = get_config_value(config, 'benchmark.resume', False)
    with_address = get_config_value(config, 'benchmark.with_address', True)
    periods = get_config_value(config, 'benchmark.periods', False)
    monthly = get_config_value(config, 'benchmark.monthly', False)
    provider = get_config_value(config, 'model.provider', 'openai')
    
    print(f"Climate LLM Benchmark")
    print(f"Configuration: {config_file}")
    if base_url_override:
        print(f"Base URL override: {base_url_override}")
    print(f"Chunk mode: {'Enabled' if chunk_mode else 'Disabled'}")
    if chunk_mode and chunk_number is not None:
        print(f"Processing chunk: {chunk_number}")
    print(f"Mesh file: {mesh_file}")
    print(f"Repeats per point: {num_repeats}")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Mode: {'Monthly (all 12 months, one query)' if monthly else (f'Simple ({month} temp only)' if simple_mode else 'Full (all months)')}")
    print(f"Address info: {'Included' if with_address else 'Coordinates only'}")
    print(f"Processing: {'Batch' if use_batch else 'Individual'}")
    print(f"LangSmith tracing: {'Disabled' if disable_tracing else 'Enabled'}")
    print(f"Resume mode: {'Enabled' if resume else 'Disabled'}")
    
    # Check if mesh file exists
    if not Path(mesh_file).exists():
        print(f"Error: Mesh file '{mesh_file}' not found.")
        print("Please provide a valid mesh file or run geo_mesh_processor.py first.")
        return
    
    # Check if final result already exists when resume is enabled
    if resume:
        # Pre-calculate expected output filename to check if job is already done
        # Load mesh data temporarily to get resolution and chunk info
        temp_mesh_data = load_mesh_data(mesh_file)
        temp_resolution = temp_mesh_data['resolution']
        temp_mode_suffix = "_simple" if simple_mode else ""
        temp_month_suffix = build_month_suffix(month, monthly)
        temp_address_suffix = "_noaddress" if not with_address else ""
        temp_periods_suffix = "_climchange" if periods else ""
        temp_sub_k = temp_mesh_data.get('mesh_info', {}).get('subsample_k')
        temp_sub_suffix = f"_sub{temp_sub_k}" if temp_sub_k else ""
        
        # Extract chunk info for filename
        temp_chunk_suffix = ""
        if 'chunk_id' in temp_mesh_data.get('mesh_info', {}):
            temp_chunk_id_num = temp_mesh_data['mesh_info']['chunk_id']
            temp_total_chunks = temp_mesh_data['mesh_info']['total_chunks']
            temp_chunk_suffix = f"_chunk_{temp_chunk_id_num:02d}_of_{temp_total_chunks:02d}"
        
        # Clean model name for filename (replace special characters)
        temp_clean_model_name = model_name.replace(":", "_").replace("/", "_").replace(".", "_")
        results_dir = get_config_value(config, 'output.results_dir', 'results')
        expected_output_file = f"{results_dir}/climate_results_{temp_resolution}deg_r{num_repeats}_{temp_clean_model_name}{temp_sub_suffix}{temp_chunk_suffix}{temp_mode_suffix}{temp_month_suffix}{temp_address_suffix}{temp_periods_suffix}.json"
        
        if Path(expected_output_file).exists():
            print(f"✓ Final result file already exists: {expected_output_file}")
            print("Job already completed - nothing to do. Use resume=false to force re-processing.")
            return
    
    try:
        # Process the benchmark
        results, mesh_data = process_climate_benchmark(config, mesh_file)
        
        # Save final results
        resolution = mesh_data['resolution']
        mode_suffix = "_simple" if simple_mode else ""
        month_suffix = build_month_suffix(month, monthly)
        address_suffix = "_noaddress" if not with_address else ""
        periods_suffix = "_climchange" if periods else ""
        sub_k = mesh_data.get('mesh_info', {}).get('subsample_k')
        sub_suffix = f"_sub{sub_k}" if sub_k else ""
        
        # Extract chunk info for filename
        chunk_suffix = ""
        if 'chunk_id' in mesh_data.get('mesh_info', {}):
            chunk_id_num = mesh_data['mesh_info']['chunk_id']
            total_chunks = mesh_data['mesh_info']['total_chunks']
            chunk_suffix = f"_chunk_{chunk_id_num:02d}_of_{total_chunks:02d}"
        
        # Clean model name for filename (replace special characters)
        clean_model_name = model_name.replace(":", "_").replace("/", "_").replace(".", "_")
        results_dir = get_config_value(config, 'output.results_dir', 'results')
        output_file = f"{results_dir}/climate_results_{resolution}deg_r{num_repeats}_{clean_model_name}{sub_suffix}{chunk_suffix}{mode_suffix}{month_suffix}{address_suffix}{periods_suffix}.json"
        # Create results directory if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        save_results(results, mesh_data, output_file, model_name, simple_mode, month, use_batch, periods, monthly=monthly, temperature=get_config_value(config, 'model.temperature', 0))
        
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
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()