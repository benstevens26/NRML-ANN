import sys
import itertools
import os

import numpy as np
from event_processor import event_processor, yield_events


def process_segment(
    segment_id,
    base_dirs,
    num_segments,
    dark_dir,
    binning,
    output_csv_prefix,
    total_num_events=49572,
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
    chunk_size = total_num_events // num_segments
    start = segment_id * chunk_size
    end = min(
        start + chunk_size, total_num_events
    )  # Ensure we don't exceed the total number of events

    # Subset of events for this specific segment
    segment_events = itertools.islice(events, start, end)

    # Run the analysis on the segmented events
    output_csv = (
        f"{output_csv_prefix}_{segment_id}.csv"  # Unique output file per segment
    )
    event_processor(
        segment_events,
        chunk_size=chunk_size,
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
        base_dirs = ["/vols/lz/MIGDAL/sim_ims/C", "/vols/lz/MIGDAL/sim_ims/F"]
        dark_dir = "/vols/lz/MIGDAL/sim_ims/darks"

    num_segments = 20  # CHANGE to number of segments
    binning = 2
    output_csv_prefix = "2x2_binned_features"
    # DON'T FORGET TO CHANGE THE NUMBER OF EVENTS IF ANALYSING A SUBSET
    # Call the processing function
    # print("Current working directory:", os.getcwd())
    process_segment(
        segment_id, base_dirs, num_segments, dark_dir, binning, output_csv_prefix
    )
