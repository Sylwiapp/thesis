from neo.rawio import BrainVisionRawIO

# Replace 'your_eeg_file.eeg' with the actual filename
eeg_filename = 'sub-RGA798_task_art_watch1_run-01.eeg'

# Read the EEG file
reader = BrainVisionRawIO(filename=eeg_filename)
reader.parse_header()

# Extract the number of samples from the data
num_samples = reader.get_signal_size()

print("Number of Samples:", num_samples)