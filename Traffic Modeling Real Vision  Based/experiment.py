import numpy as np



num_lanes = 8
qLengths1 = []
qLengths2 = []
for i in range(num_lanes):
    qLengths1.append(this_state[0][i][0])
    qLengths2.append(this_new_state[0][i][0])

qLengths11 = [x + 1 for x in qLengths1]   # isme bas 8 numbers hone chahiye the...zyaada aare hain usse kaafi zyada
qLengths21 = [x + 1 for x in qLengths2]

print("This state - ", this_state)
print("New State - ", this_new_state)

print("qlengths 1 = ", qLengths1)
print("qlengths 2 = ", qLengths2)

q1 = np.prod(qLengths11)
q2 = np.prod(qLengths21)

# print("Old State with product : ", q1)
#
# print("New State with product : ", q2)
#
#
# if q1 > q2:
#     this_reward = 1
# else:
#     this_reward = -1




this_reward = q1 - q2

print("Q1 = ", q1)
print("Q2 = ", q2)
print("This reward = ", this_reward)