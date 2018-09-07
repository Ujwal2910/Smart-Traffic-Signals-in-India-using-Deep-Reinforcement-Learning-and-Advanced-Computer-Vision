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

test = []
for i in range(4):
    test.append(newState[0][i][0])
print(test)