from event_processor import yield_events  
from event_processor import event_processor
import numpy as np
  
base_dirs = ["../../../../MIGDAL/sim_ims/C/300-320keV"]
output_csv = "processed_events.csv"  
chunk_size = 50
master = np.load("../../../../MIGDAL/sim_ims/darks/master_dark_1x1.npy")
darks = np.load("../../../../MIGDAL/sim_ims/darks/quest_std_dark_0.npy")



events = yield_events(base_dirs)  
event_processor(events, chunk_size, output_csv="test_features.csv",m_dark=master, example_dark_list=darks)