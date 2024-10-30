import numpy as np
from convert_sim_ims import *


def noise_adder(event_list, m_dark, example_dark_list):
    """Adds noise to an array of raw images

    Args:
        event_list (array): An array containing instances of the Event class to have noise added to them. NOTE that the events should be initialised with the raw images.
        m_dark (array): 2D array (image) containing the master dark. This is by default loaded from "Data/darks/master_dark_1x1.npy", but would change with binning.
        example_dark_list (array): An array containing the example darks. This is by default loaded from "Data/darks/quest_std_dark_1.npy" but should really use a random one from that folder each time.

    Returns:
        array: An array of images that have had a random sample of noise added to them.
    """
    return [
        convert_im(
            event.image,
            get_dark_sample(
                m_dark,
                [len(event.image[0]), len(event.image)],
                example_dark_list[np.random.randint(0, len(example_dark_list) - 1)],
            ),
        )
        for event in event_list
    ]


# folder_path = "Data/C/300-320keV"  # Change to whichever data you want to use
# events = load_events(folder_path)


# m_dark_list = [np.load(f) for f in glob.glob("Data/darks/master_dark_*.npy")]
# m_dark = m_dark_list[
#     1
# ]  # This is only for while I don't care about the others. I cba to implement ordering them by the numbers at the end of the file names
# example_dark_list = np.load(
#     "Data/darks/quest_std_dark_1.npy"  # Make this load a random one of the 10
# )  # This is an array of 200 dark images. I believe the other 9 are the same as well
# example_dark = example_dark_list[np.random.randint(0, len(example_dark_list) - 1)]

# dark_sample = get_dark_sample(
#     m_dark, [len(events[0].image[0]), len(events[0].image)], example_dark
# )
