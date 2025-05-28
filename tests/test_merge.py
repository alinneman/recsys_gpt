"""Tests for event merging functionality."""

from datetime import datetime, timedelta
from typing import List

import pytest

from recsys.prep.merge_events import Event, MergeEventsTransformer


def create_event(
    row_id: int,
    user_id: int,
    item_id: int,
    event_type: str,
    time_offset: int = 0,
    dwell: float = None,
) -> Event:
    """Helper function to create an Event with a timestamp."""
    base_time = datetime(2023, 1, 1, 12, 0, 0)
    return Event(
        row_id=row_id,
        user_id=user_id,
        item_id=item_id,
        event_type=event_type,
        ts=base_time + timedelta(seconds=time_offset * 60),  # Convert minutes to seconds
        dwell=dwell,
    )


# Test cases as a list of tuples: (test_name, input_events, expected_events, test_id)
TEST_CASES = [
    (
        "simple_click_merge",
        [
            create_event(1, 1, 101, "click", 0, 30),
            create_event(2, 1, 101, "click", 1, 45),  # Same user and item within time window
        ],
        [
            Event(
                row_id=1,  # Keeps the first row_id
                user_id=1,
                item_id=101,
                event_type="click",
                ts=datetime(2023, 1, 1, 12, 0),  # Keeps the first timestamp
                dwell=60.0,  # Time between first and last event
            )
        ],
        "simple_click_merge",
    ),
    (
        "click_to_add_to_cart_upgrade",
        [
            create_event(1, 1, 101, "click", 0, 30),
            create_event(2, 1, 101, "add_to_cart", 1),  # Higher priority event
        ],
        [
            Event(
                row_id=2,  # Keeps the row_id of the higher priority event
                user_id=1,
                item_id=101,
                event_type="add_to_cart",  # Keeps the higher priority event type
                ts=datetime(2023, 1, 1, 12, 1),  # Keeps the timestamp of the higher priority event
                dwell=60.0,  # Time between first and last event
            )
        ],
        "click_to_add_to_cart_upgrade",
    ),
    (
        "separate_items_remain_separate",
        [
            create_event(1, 1, 101, "click", 0, 30),
            create_event(2, 1, 102, "click", 1, 45),  # Different item ID
            create_event(3, 2, 101, "click", 2, 20),  # Different user ID
        ],
        [
            Event(1, 1, 101, "click", datetime(2023, 1, 1, 12, 0), 0.0),  # No dwell for unmerged events
            Event(2, 1, 102, "click", datetime(2023, 1, 1, 12, 1), 0.0),
            Event(3, 2, 101, "click", datetime(2023, 1, 1, 12, 2), 0.0),
        ],
        "separate_items_remain_separate",
    ),
]


@pytest.mark.parametrize("test_name,input_events,expected_events,test_id", TEST_CASES, ids=[tc[3] for tc in TEST_CASES])
def test_merge_events(test_name: str, input_events: List[Event], expected_events: List[Event], test_id: str):
    """Test event merging with various scenarios."""
    transformer = MergeEventsTransformer(max_time_gap_seconds=1800)  # 30 minutes
    result = transformer.merge(input_events)
    
    # Compare the results with expected events
    assert len(result) == len(expected_events), "Number of merged events doesn't match"
    
    for res, exp in zip(result, expected_events):
        assert res.row_id == exp.row_id, f"Row ID mismatch in {test_name}"
        assert res.user_id == exp.user_id, f"User ID mismatch in {test_name}"
        assert res.item_id == exp.item_id, f"Item ID mismatch in {test_name}"
        assert res.event_type == exp.event_type, f"Event type mismatch in {test_name}"
        assert res.ts == exp.ts, f"Timestamp mismatch in {test_name}"
        if exp.dwell is not None:
            assert res.dwell == pytest.approx(exp.dwell), f"Dwell time mismatch in {test_name}"
