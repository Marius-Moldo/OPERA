import numpy as np

# Load the original file
a = np.load('datasets/coughvid/entire_spec_filenames.npy')
print("Original paths:")
#print(a)

a_audio = np.array([filename.replace('entire_spec_npy', 'entire_spec_npy_smooth')  for filename in a])

#print("\nModified paths:")
print(a_audio)

# Save to a new file
np.save('datasets/coughvid/entire_cough_filenames_smooth.npy', a_audio)
#print(f"\nSaved {len(a_audio)} modified paths to 'datasets/covidUK/entire_cough_filenames_audio.npy'")