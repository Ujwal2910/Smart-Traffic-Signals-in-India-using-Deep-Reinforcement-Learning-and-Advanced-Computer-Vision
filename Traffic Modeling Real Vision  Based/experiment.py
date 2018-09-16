import numpy as np
newState = []
for _ in range(3):

    state = [1,
         2,
         3,
         4
         ]

    newState.insert(0, state)
# print (state)
newState = np.array(newState)
print(newState.shape)

abc = np.zeros((3,4,5))
print(abc.shape)
newState = np.dstack((newState, abc))
print(newState.shape)
print(newState)
newState = np.expand_dims(newState, axis=0)
print(newState.shape)
print(newState)
'''
test = []
for i in range(4):
    test.append(newState[0][i][0])
print(test)
'''

num_lanes = 4
qLengths1 = []
qLengths2 = []
for i in range(3):
    qLengths1.append(0)
    qLengths2.append(0)
qLengths1.append(1)
qLengths2.append(1)

qLengths11 = [x + 1 for x in qLengths1]
qLengths21 = [x + 1 for x in qLengths2]

print(qLengths11)
print(qLengths21)

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

print(this_reward)

if this_reward > 0:
    this_reward = 1
elif this_reward < 0:
    this_reward = -1
elif q2 > 1:
    this_reward = -1
else:
    this_reward = 0

print(this_reward)