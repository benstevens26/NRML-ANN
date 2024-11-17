from event_processor import event_processor, yield_events

local = True  # Change to reflect if you're running this locally or on the SSH.
if local:
    base_dirs = ["Data/C", "Data/F"]
else:
    base_dirs = ["../../../../MIGDAL/sim_ims/C", "../../../../MIGDAL/sim_ims/F"]

output_csv = "processed_events.csv"
chunk_size = 50


events = yield_events(base_dirs)
event_processor(
    events,
    chunk_size,
    output_csv="more_features_noisy.csv",
    dark_dir="Data/darks",
    binning=1,
)
