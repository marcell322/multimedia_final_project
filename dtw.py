from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
import pandas as pd

import time

# Record start time
start_time = time.time()

# Your code or task goes here
print("opening teacher file")
teacher = pd.read_excel('koordinat_faded_roseau.xlsx')
teacher_thumb = []
for thumb in teacher['THUMB_TIP']:
    teacher_thumb.append(thumb.split(', '))
teacher_thumb = np.array(teacher_thumb)

s1_frame = np.array([eval(i) for i in teacher_thumb[:,0]])
s1_x = np.array([eval(i) for i in teacher_thumb[:,1]])
s1_y = np.array([eval(i) for i in teacher_thumb[:,2]])

s1 = np.stack((s1_frame, s1_x, s1_y), axis=1)
indices = np.lexsort((s1[:, 2], s1[:, 1], s1[:, 0]))

# Use the indices to rearrange the array
s1 = s1[indices]

# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

# Record start time
start_time = time.time()

# Your code or task goes here
print("opening student file")
student = pd.read_excel('koordinat_faded_pianella.xlsx')
student_thumb = []
for thumb in student['THUMB_TIP']:
    student_thumb.append(thumb.split(', '))
student_thumb = np.array(student_thumb)

s2_frame = np.array([eval(i) for i in student_thumb[:,0]])
s2_x = np.array([eval(i) for i in student_thumb[:,1]])
s2_y = np.array([eval(i) for i in student_thumb[:,2]])

s2 = np.stack((s2_frame, s2_x, s2_y), axis=1)
indices = np.lexsort((s2[:, 2], s2[:, 1], s2[:, 0]))

# Use the indices to rearrange the array
s2 = s2[indices]

# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

s1 = s1[:, 1:3]
s2 = s2[:, 1:3]

# Record start time
start_time = time.time()

# Your code or task goes here
print("start DTW-ing")
d, paths = dtw.warping_paths(s1, s2)
# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

# Record start time
start_time = time.time()

# Your code or task goes here
print("start pathing")
best_path = dtw.best_path(paths)
# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

# figure, axes = dtwvis.plot_warping(s1, s2, paths)
figure, axes = dtwvis.plot_warpingpaths(s1, s2, paths, best_path, filename = "D:/(0) Thesis/dtw/test-0.png")

# figure.show()


# Record start time
start_time = time.time()

# Your code or task goes here
print("start DTW-ing")
d, paths = dtw.warping_paths(s1, s2, max_step=2)
# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

# Record start time
start_time = time.time()

# Your code or task goes here
print("start pathing")
best_path = dtw.best_path(paths)
# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

# figure, axes = dtwvis.plot_warping(s1, s2, paths)
figure, axes = dtwvis.plot_warpingpaths(s1, s2, paths, best_path, filename = "D:/(0) Thesis/dtw/test-1-s2.png")


# Record start time
start_time = time.time()

# Your code or task goes here
print("start DTW-ing")
d, paths = dtw.warping_paths(s1, s2, window=1)
# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

# Record start time
start_time = time.time()

# Your code or task goes here
print("start pathing")
best_path = dtw.best_path(paths)
# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

# figure, axes = dtwvis.plot_warping(s1, s2, paths)
figure, axes = dtwvis.plot_warpingpaths(s1, s2, paths, best_path, filename = "D:/(0) Thesis/dtw/test-2-w1.png")


# Record start time
start_time = time.time()

# Your code or task goes here
print("start DTW-ing")
d, paths = dtw.warping_paths(s1, s2, window=1, max_step=2)
# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

# Record start time
start_time = time.time()

# Your code or task goes here
print("start pathing")
best_path = dtw.best_path(paths)
# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

# figure, axes = dtwvis.plot_warping(s1, s2, paths)
figure, axes = dtwvis.plot_warpingpaths(s1, s2, paths, best_path, filename = "D:/(0) Thesis/dtw/test-3-w1s2.png")