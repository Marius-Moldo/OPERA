import numpy as np
a = np.load('datasets/covidUK/entire_cough_filenames.npy')
# Replace 'entire_spec_npy' with 'audio' in each filename
a_audio = np.array([filename.replace('entire_spec_npy', 'entire_spec_npy_smooth') for filename in a])
print(a_audio)
a = np.save('datasets/covidUK/entire_cough_filenames_smooth.npy', a_audio)
