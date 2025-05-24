"""Tokenization of user sessions into model inputs."""

from dataclasses import asdict
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple

from recsys.prep.merge_events import Event

# Default lookup tables (stubbed with example values)
DEFAULT_LOOKUP_TABLES = {
    'device': {
        'mobile': 1,
        'desktop': 2,
        'tablet': 3,
    },
    'locale': {
        'en-US': 1,
        'en-GB': 2,
        'es-ES': 3,
        'fr-FR': 4,
    },
    'daypart': {
        'morning': 1,      # 6am-12pm
        'afternoon': 2,    # 12pm-6pm
        'evening': 3,      # 6pm-12am
        'night': 4,        # 12am-6am
    },
    'event_type': {
        'view': 1,
        'click': 2,
        'add_to_cart': 3,
        'purchase': 4,
    }
}


def get_daypart(timestamp: datetime) -> str:
    """Determine the daypart (morning/afternoon/evening/night) for a timestamp."""
    hour = timestamp.hour
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'


def tokenize_session(
    events: List[Event],
    max_length: int = 500,
    lookup_tables: Optional[Dict[str, Dict[str, int]]] = None,
) -> List[Dict[str, int]]:
    """Convert a list of events into tokenized model inputs.
    
    Args:
        events: List of Event objects, typically from MergeEventsTransformer.
        max_length: Maximum number of tokens to return (truncates older events).
        lookup_tables: Dictionaries mapping categorical values to indices.
            Should contain keys: 'device', 'locale', 'daypart', 'event_type'.
            If None, uses default lookup tables.
    
    Returns:
        List of token dictionaries, each containing:
        - item_id: The item ID
        - event_type: Encoded event type
        - device: Encoded device type
        - locale: Encoded locale
        - daypart: Encoded time of day
        - timestamp: Unix timestamp (seconds since epoch)
    """
    if not events:
        return []
    
    # Use default lookup tables if none provided
    if lookup_tables is None:
        lookup_tables = DEFAULT_LOOKUP_TABLES
    
    # Get the most recent events up to max_length
    events = events[-max_length:]
    
    # Sort events by timestamp to ensure chronological order
    events_sorted = sorted(events, key=lambda e: e.ts)
    
    tokens = []
    for event in events_sorted:
        # Get or default to 0 (unknown) for categorical features
        device_idx = lookup_tables['device'].get('desktop', 0)  # Default to desktop
        locale_idx = lookup_tables['locale'].get('en-US', 0)    # Default to en-US
        
        # Get daypart based on event timestamp
        daypart = get_daypart(event.ts)
        daypart_idx = lookup_tables['daypart'].get(daypart, 0)
        
        # Get event type, default to 'view' if not found
        event_type = event.event_type.lower()
        event_type_idx = lookup_tables['event_type'].get(event_type, 0)
        
        tokens.append({
            'item_id': event.item_id,
            'event_type': event_type_idx,
            'device': device_idx,
            'locale': locale_idx,
            'daypart': daypart_idx,
            'timestamp': int(event.ts.timestamp()),
            'dwell_seconds': int(event.dwell) if event.dwell is not None else 0,
        })
    
    return tokens
