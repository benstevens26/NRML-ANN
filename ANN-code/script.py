import sys

import numpy as np
from event_processor import event_processor, yield_events


def process_segment(
    segment_id, base_dirs, num_segments, dark_dir, binning, output_csv_prefix
):
    """
    Processes a specific segment of events, based on `segment_id`.

    Parameters:
    - segment_id: The unique ID for the current segment (used for indexing).
    - base_dirs: The directories where images (events) are stored.
    - num_segments: The number of segments to be processed.
    - dark_dir: Directory containing dark images for processing.
    - binning: The binning parameter for processing.
    - output_csv_prefix: Prefix for output CSV file names.
    """

    # Generate all events (images)
    events = yield_events(base_dirs)  # List of all events

    # Calculate the range of events for this segment
    chunk_size = len(events) // num_segments
    start = segment_id * chunk_size
    end = min(
        start + chunk_size, len(events)
    )  # Ensure we don't exceed the total number of events

    # Subset of events for this specific segment
    segment_events = events[start:end]

    # Run the analysis on the segmented events
    output_csv = (
        f"{output_csv_prefix}_{segment_id}.csv"  # Unique output file per segment
    )
    event_processor(
        segment_events,
        chunk_size=len(
            segment_events
        ),  # Use the actual number of events in this segment
        output_csv=output_csv,
        dark_dir=dark_dir,
        binning=binning,
    )


if __name__ == "__main__":
    # Parse command-line arguments
    segment_id = int(sys.argv[1])  # Segment ID passed from HTCondor
    local = False  # Change to reflect if you're running this locally or on the SSH.
    if local:
        base_dirs = ["Data/C", "Data/F"]
        dark_dir = "Data/darks"
    else:
        base_dirs = ["../../../../MIGDAL/sim_ims/C", "../../../../MIGDAL/sim_ims/F"]
        dark_dir = "../../../../MIGDAL/sim_ims/darks"

    num_segments = 20  # CHANGE to number of segments
    binning = 2
    output_csv_prefix = "2x2_binned_features"

    # Call the processing function
    process_segment(
        segment_id, base_dirs, num_segments, dark_dir, binning, output_csv_prefix
    )
