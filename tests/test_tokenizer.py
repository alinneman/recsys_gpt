"""Tests for the session tokenizer."""

from datetime import datetime, timedelta
from typing import List

import pytest
from hypothesis import given, settings, strategies as st

from recsys.prep.merge_events import Event
from recsys.prep.tokenizer import tokenize_session, DEFAULT_LOOKUP_TABLES


# Strategies for generating test data
event_types = st.sampled_from(list(DEFAULT_LOOKUP_TABLES['event_type'].keys()))
devices = st.sampled_from(list(DEFAULT_LOOKUP_TABLES['device'].keys()))
locales = st.sampled_from(list(DEFAULT_LOOKUP_TABLES['locale'].keys()))


def generate_events(
    n_events: int,
    start_time: datetime = datetime(2023, 1, 1, 12, 0),
    time_step: int = 60,  # 1 minute between events
) -> List[Event]:
    """Generate a list of test events with sequential timestamps."""
    events = []
    for i in range(n_events):
        event_time = start_time + timedelta(seconds=i * time_step)
        events.append(
            Event(
                row_id=i,
                user_id=1,
                item_id=100 + (i % 10),  # Cycle through 10 different items
                event_type='click',
                ts=event_time,
                dwell=30.0 if i % 2 == 0 else None,
            )
        )
    return events


class TestTokenizer:
    """Test the session tokenizer."""

    @given(st.integers(min_value=0, max_value=1000))
    @settings(max_examples=10)
    def test_output_length_leq_input(self, n_events: int):
        """Test that output length is less than or equal to input length."""
        events = generate_events(n_events)
        tokens = tokenize_session(events)
        assert len(tokens) <= len(events)
        assert len(tokens) <= 500  # Should respect max_length

    def test_timestamps_non_decreasing(self):
        """Test that tokens are in chronological order."""
        # Create events with non-sequential timestamps
        events = [
            Event(1, 1, 101, 'click', datetime(2023, 1, 1, 12, 30), 30.0),
            Event(2, 1, 102, 'click', datetime(2023, 1, 1, 12, 0), 45.0),  # Earlier timestamp
            Event(3, 1, 103, 'add_to_cart', datetime(2023, 1, 1, 12, 45), 10.0),
        ]
        
        tokens = tokenize_session(events)
        timestamps = [t['timestamp'] for t in tokens]
        assert timestamps == sorted(timestamps), "Timestamps should be in chronological order"

    def test_highest_priority_event_preserved(self):
        """Test that the highest priority event is preserved in the output."""
        events = [
            Event(1, 1, 101, 'view', datetime(2023, 1, 1, 12, 0), 10.0),
            Event(2, 1, 101, 'click', datetime(2023, 1, 1, 12, 1), 20.0),
            Event(3, 1, 101, 'add_to_cart', datetime(2023, 1, 1, 12, 2), 30.0),
            Event(4, 1, 101, 'purchase', datetime(2023, 1, 1, 12, 3), 40.0),  # Highest priority
            Event(5, 1, 101, 'click', datetime(2023, 1, 1, 12, 4), 50.0),
        ]
        
        tokens = tokenize_session(events)
        event_types = [t['event_type'] for t in tokens]
        
        # The highest priority event type is 'purchase' (4)
        assert max(event_types) == DEFAULT_LOOKUP_TABLES['event_type']['purchase']

    def test_truncation(self):
        """Test that long sequences are truncated to max_length."""
        # Generate more events than max_length
        events = generate_events(600)  # 600 events, more than default max_length of 500
        tokens = tokenize_session(events)
        assert len(tokens) == 500
        
        # The oldest events should be dropped
        original_timestamps = [e.ts for e in events]
        token_timestamps = [datetime.fromtimestamp(t['timestamp']) for t in tokens]
        assert min(token_timestamps) > min(original_timestamps)
        assert max(token_timestamps) == max(original_timestamps)

    def test_empty_input(self):
        """Test that empty input returns empty output."""
        assert tokenize_session([]) == []

    def test_default_lookup_tables_used(self):
        """Test that default lookup tables are used when none provided."""
        events = generate_events(1)
        tokens = tokenize_session(events)
        
        # Check that all required fields are present and use default values
        token = tokens[0]
        assert token['device'] == DEFAULT_LOOKUP_TABLES['device']['desktop']
        assert token['locale'] == DEFAULT_LOOKUP_TABLES['locale']['en-US']
        assert token['event_type'] == DEFAULT_LOOKUP_TABLES['event_type']['click']
        assert token['daypart'] in DEFAULT_LOOKUP_TABLES['daypart'].values()
