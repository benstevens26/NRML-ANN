"""
test module
"""
import random
from feature_extraction import *

folder_path = "Data/C/300-320keV"  # Change to whichever data you want to use
events = load_events(folder_path)


# sample some events for analysis
events = random.sample(events, 10)
test_event = events[0]

# for event in events:
#
#     params = extract_axis(event, plot=True, return_extras=True)
#     principle_axis, mean_x, mean_y = params
#
#     pixels = extract_pixels(event.image, principle_axis, mean_x, mean_y, threshold=2)
#
#     plot_deposition(pixels)




# Example usage
image = np.random.rand(100, 100)  # Replace with your actual image
principal_axis = extract_axis(test_event)  # Replace with your actual principal axis (e.g. from SVD)
num_segments = 10  # Number of segments along the principal axis

segment_intensities = segment_along_principal_axis_with_plot(test_event.image, principal_axis, num_segments)

# Plot the segment intensities (energy loss along the track)
plt.plot(segment_intensities)
plt.xlabel('Segment')
plt.ylabel('Summed Pixel Intensity')
plt.title('Energy Loss Along the Track')
plt.show()

