"""Event merging functionality for RecSys GPT."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class Event:
    """Represents a user interaction event with an item.
    
    Attributes:
        row_id: Unique identifier for the event
        user_id: ID of the user who performed the event
        item_id: ID of the item the event was performed on
        event_type: Type of the event (e.g., 'click', 'add_to_cart', 'purchase')
        ts: Timestamp of when the event occurred
        dwell: Optional duration of the interaction in seconds
    """
    row_id: int
    user_id: int
    item_id: int
    event_type: str
    ts: datetime
    dwell: Optional[float] = None


class MergeEventsTransformer:
    """Transforms raw events by merging related events according to business rules."""
    
    # Define event type priorities (higher number = higher priority)
    EVENT_PRIORITIES = {
        'purchase': 3,
        'add_to_cart': 2,
        'click': 1
    }
    
    def __init__(self, max_time_gap_seconds: int = 1800):
        """Initialize the transformer.
        
        Args:
            max_time_gap_seconds: Maximum time gap in seconds between events to be considered contiguous.
        """
        self.max_time_gap_seconds = max_time_gap_seconds
    
    def _get_event_priority(self, event: Event) -> int:
        """Get the priority of an event type."""
        return self.EVENT_PRIORITIES.get(event.event_type, 0)
    
    def _should_merge(self, prev_event: Event, current_event: Event) -> bool:
        """Determine if two events should be merged."""
        if prev_event.user_id != current_event.user_id or prev_event.item_id != current_event.item_id:
            return False
            
        time_diff = (current_event.ts - prev_event.ts).total_seconds()
        return time_diff <= self.max_time_gap_seconds
    
    def _merge_events(self, events: List[Event]) -> Event:
        """Merge a list of events into a single event."""
        if not events:
            raise ValueError("Cannot merge empty list of events")
            
        # Find the highest priority event
        merged_event = max(events, key=self._get_event_priority)
        
        # Update dwell time if needed
        if any(e.dwell is not None for e in events):
            start_time = min(e.ts for e in events)
            end_time = max(e.ts for e in events)
            merged_event.dwell = (end_time - start_time).total_seconds()
        
        return merged_event
    
    def merge(self, events: List[Event]) -> List[Event]:
        """Merge contiguous events from the same user and item.
        
        Args:
            events: List of events to process, sorted by timestamp.
            
        Returns:
            List of merged events.
        """
        if not events:
            return []
            
        # Sort events by timestamp to ensure processing in chronological order
        sorted_events = sorted(events, key=lambda e: e.ts)
        
        merged_events = []
        current_batch = [sorted_events[0]]
        
        for event in sorted_events[1:]:
            if self._should_merge(current_batch[-1], event):
                current_batch.append(event)
            else:
                # Merge the current batch and start a new one
                merged_events.append(self._merge_events(current_batch))
                current_batch = [event]
        
        # Don't forget the last batch
        if current_batch:
            merged_events.append(self._merge_events(current_batch))
        
        return merged_events
