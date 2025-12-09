from typing import Dict, Any, cast, Optional
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.location import Location
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from tqdm import tqdm
from .core import log, config
import time

GEOLOCATOR = Nominatim(user_agent="surabaya_opinion_analysis_project")

def get_coordinates(location_name: str, attempt: int = 1, max_attempts: int = 3) -> Dict[str, Any]:
    """
    Retrieve coordinates (latitude and longitude) for the location name.
    Return a dictionary with 'latitude' and 'longitude' (None if unsuccessful).
    """
    coords = {
        'latitude': None,
        'longitude': None,
        'normalized_address': None
    }
    
    try:
        query = f"{location_name}, Surabaya, Indonesia"
        location = cast(Optional[Location], GEOLOCATOR.geocode(query, timeout=10.0)) # type: ignore
        
        if location:
            coords['latitude'] = location.latitude
            coords['longitude'] = location.longitude
            coords['normalized_address'] = location.address
        else:
            query_fallback = f"{location_name}, Indonesia"
            location_fallback = cast(Optional[Location], GEOLOCATOR.geocode(query_fallback, timeout=10.0)) # type: ignore
            if location_fallback:
                coords['latitude'] = location_fallback.latitude
                coords['longitude'] = location_fallback.longitude
                coords['normalized_address'] = location_fallback.address

    except (GeocoderTimedOut, GeocoderServiceError):
        if attempt <= max_attempts:
            print(f"Timeout for {location_name}. Retrying ({attempt}/{max_attempts})...")
            time.sleep(2 * attempt)
            return get_coordinates(location_name, attempt + 1, max_attempts)
        
    except Exception as e:
        log.error(f"Geocoding error for {location_name}: {e}")
        
    return coords

def run_geocoding(df: pd.DataFrame):
    unique_locs = set()
    for entities in df['entities']:
        if entities:
            for ent in entities:
                if ent['entity_group'] == 'LOC':
                    # ignore location that is part of abbreviations since they are big cities or countries
                    if ent['word'] not in config.loc_abbr.values():
                        unique_locs.add(ent['word'])
    
    if not unique_locs:
        log.info("No 'LOC' entities found for geocoding.")
        return

    log.info(f"Geocoding {len(unique_locs)} unique locations...")
    
    loc_map = {}
    for loc_text in tqdm(unique_locs, desc="Geocoding", unit="loc"):
        loc_map[loc_text] = get_coordinates(str(loc_text))
        time.sleep(1.1)

    updated_count = 0
    for entities in df['entities']:
        if entities:
            for ent in entities:
                if ent['entity_group'] == 'LOC' and ent['word'] in loc_map:
                    coords = loc_map[ent['word']]
                    ent['latitude'] = coords['latitude']
                    ent['longitude'] = coords['longitude']
                    ent['normalized_address'] = coords['normalized_address']
                    updated_count += 1
                else:
                    if 'latitude' not in ent:
                        ent['latitude'] = None
                    if 'longitude' not in ent:
                        ent['longitude'] = None
                    if 'normalized_address' not in ent:
                        ent['normalized_address'] = None

    log.info(f"Geocoding completed. enriched {updated_count} entities with coordinates.")