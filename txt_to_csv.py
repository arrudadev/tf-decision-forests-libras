import utils

landmarks = []

with open('landmarks.txt', 'r') as file:
  lines = file.readlines()

for line in lines:
  landmarks.append(eval(line))

utils.landmarks_to_csv(landmarks)

print('Done!')
