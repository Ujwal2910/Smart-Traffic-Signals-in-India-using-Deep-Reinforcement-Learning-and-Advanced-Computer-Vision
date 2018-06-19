from __future__ import absolute_import
from __future__ import print_function
from select import select
import termios
import os
import sys
import optparse
import subprocess
import random
import time
import cv2
import curses

import readScreen2
import numpy as np


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

def generate_routefile():
    random.seed(42)  # make tests reproducible
    N = 3600  # number of time steps
    # demand per second from different directions
    pWE = 1. / 10
    pEW = 1. / 11
    pNS = 1. / 30
    with open("data/cross_auto.rou.xml", "w") as routes:
        print("""<routes>
    <vTypeDistribution id="mixed">
        <vType id="car" vClass="passenger" speedDev="0.2" latAlignment="compact" probability="0.3"/>
        <vType id="moped" vClass="moped" speedDev="0.4" latAlignment="compact" probability="0.7"/>
    </vTypeDistribution>

    <route id="r0" edges="51o 1i 2o 52i"/>
    <route id="r1" edges="51o 1i 4o 54i"/>
    <route id="r2" edges="51o 1i 3o 53i"/>
    <route id="r3" edges="54o 4i 3o 53i"/>
    <route id="r4" edges="54o 4i 1o 51i"/>
    <route id="r5" edges="54o 4i 2o 52i"/>
    <route id="r6" edges="52o 2i 1o 51i"/>
    <route id="r7" edges="52o 2i 4o 54i"/>
    <route id="r8" edges="52o 2i 3o 53i"/>
    <route id="r9" edges="53o 3i 4o 54i"/>
    <route id="r10" edges="53o 3i 1o 51i"/>
    <route id="r11" edges="53o 3i 2o 52i"/>

    <flow id="mixed1" begin="0" end="15000" number="500" route="r0" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed2" begin="0" end="15000" number="500" route="r1" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed3" begin="0" end="15000" number="500" route="r2" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed4" begin="0" end="15000" number="500" route="r3" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed5" begin="0" end="15000" number="500" route="r4" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed6" begin="0" end="15000" number="500" route="r5" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed7" begin="0" end="15000" number="500" route="r6" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed8" begin="0" end="15000" number="500" route="r7" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed9" begin="0" end="15000" number="500" route="r8" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed10" begin="0" end="15000" number="500" route="r9" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed11" begin="0" end="15000" number="500" route="r10" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed12" begin="0" end="15000" number="500" route="r11" type="mixed" departLane="random" departPosLat="random"/>
</routes>""", file=routes)
        lastVeh = 0
        vehNr = 0



try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci

options = get_options()

# this script has been called from the command line. It will start sumo as a
# server, then connect and run

if options.nogui:
    sumoBinary = checkBinary('sumo')
else:
    sumoBinary = checkBinary('sumo-gui')

# first, generate the route file for this simulation
generate_routefile()
print("Now starting TraCI")
# this is the normal way of using traci. sumo is started as a
# subprocess and then the python script connects and runs
traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                         "--tripinfo-output", "tripinfo.xml"])

print("TraCI Started")




#State = State_Lengths()
#print(State.get_tails())

#states = State.get_tails



#runner = Runner()
#print(Runner().run)


def getState():

    #print(States_.get_tails())

    state = np.zeros((5,1))

    state[0,0] = readScreen2.getUpperQlength()
    state[1,0] = readScreen2.getLowerQlength()
    state[2,0] = readScreen2.getRightQlength()
    state[3,0] = readScreen2.getLeftQlength()
    phase = traci.trafficlight.getPhase("0")
    state[4,0] = phase
    print (state)

    return state



print("here")
import traci



def makeMove(state,action):
    if action == 1:
        traci.trafficlight.setPhase("0",((traci.trafficlight.getPhase("0") + 1)%8))

    # agent.simulateFrames(SIM_FRAMES)
    traci.simulationStep()
    traci.simulationStep()
    traci.simulationStep()
    traci.simulationStep()
    traci.simulationStep()
    traci.simulationStep()
    traci.simulationStep()
    traci.simulationStep()
    traci.simulationStep()

    newState = getState()

    return newState

def getReward(state):
    qLengths = state[:4]
    reward = (-1) * np.average(qLengths) * np.std(qLengths)

    return reward



from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(5,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

model.add(Dense(150, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(2, kernel_initializer='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
#model.predict(state.reshape(1,5), batch_size=1)

model.compile(loss='mse', optimizer=rms)#reset weights of neural network
epochs = 100
gamma = 0.975
epsilon = 1
batchSize = 40
buffer = 80
replay = []
#stores tuples of (S, A, R, S')
h = 0
for i in range(epochs):

    """execute the TraCI control loop"""


    traci.trafficlight.setPhase("0", 0)
    state = getState()
    counter = 1

    while traci.simulation.getMinExpectedNumber() > 0:

        qval = model.predict(state.reshape(1,5), batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,2)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state = makeMove(state, action)

        #Observe reward
        reward = getReward(new_state)

        #Experience replay storage
        if (len(replay) < buffer): #if buffer not filled, add to it
            replay.append((state, action, reward, new_state))
        else: #if buffer full, overwrite old values
            if (h < (buffer-1)):
                h += 1
            else:
                h = 0
            replay[h] = (state, action, reward, new_state)
            #randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            X_train = []
            y_train = []
            for memory in minibatch:
                #Get max_Q(S',a)
                old_state, action, reward, new_state = memory
                old_qval = model.predict(old_state.reshape(1,5), batch_size=1)
                newQ = model.predict(new_state.reshape(1,5), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,2))
                y[:] = old_qval[:]
                update = (reward + (gamma * maxQ))
                y[0][action] = update
                X_train.append(old_state.reshape(5,))
                y_train.append(y.reshape(2,))

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            print("Frame #: %s" % (i,))
            model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=1)
            state = new_state
        #status += 1
        if epsilon > 0.1: #decrement epsilon over time
            epsilon -= (1/1000)
        counter +=1





        getState()

    #.curses.endwin()

    traci.close()
    sys.stdout.flush()

'''
    state = getState() #using the harder state initialization function
    status = 1
    #while game still in progress
    while(status < 10000):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1,5), batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,2)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state = makeMove(state, action)
        #Observe reward
        reward = getReward(new_state)

        #Experience replay storage
        if (len(replay) < buffer): #if buffer not filled, add to it
            replay.append((state, action, reward, new_state))
        else: #if buffer full, overwrite old values
            if (h < (buffer-1)):
                h += 1
            else:
                h = 0
            replay[h] = (state, action, reward, new_state)
            #randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            X_train = []
            y_train = []
            for memory in minibatch:
                #Get max_Q(S',a)
                old_state, action, reward, new_state = memory
                old_qval = model.predict(old_state.reshape(1,5), batch_size=1)
                newQ = model.predict(new_state.reshape(1,5), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,2))
                y[:] = old_qval[:]
                update = (reward + (gamma * maxQ))
                y[0][action] = update
                X_train.append(old_state.reshape(5,))
                y_train.append(y.reshape(2,))

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            print("Frame #: %s" % (i,))
            model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=1)
            state = new_state
        #status += 1
        clear_output(wait=True)
    if epsilon > 0.1: #decrement epsilon over time
        epsilon -= (1/epochs)

'''
