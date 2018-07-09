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
    <flow id="mixed1" begin="0" end="15000" number="1500" route="r0" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed2" begin="0" end="15000" number="10" route="r1" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed3" begin="0" end="15000" number="10" route="r2" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed4" begin="0" end="15000" number="0" route="r3" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed5" begin="0" end="15000" number="0" route="r4" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed6" begin="0" end="15000" number="0" route="r5" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed7" begin="0" end="15000" number="0" route="r6" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed8" begin="0" end="15000" number="0" route="r7" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed9" begin="0" end="15000" number="1500" route="r8" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed10" begin="0" end="15000" number="0" route="r9" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed11" begin="0" end="15000" number="110" route="r10" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed12" begin="0" end="15000" number="1000" route="r11" type="mixed" departLane="random" departPosLat="random"/>
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

# this is the normal way of using traci. sumo is started as a
# subprocess and then the python script connects and runs


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

    traci.trafficlight.setPhase("0",action)

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
#
# def getReward(state):
#     qLengths = state[:4]
#     sum = np.sum(qLengths)
#     if sum>=0 and sum <65:
#         reward = 100
#     elif sum>64 and sum<129:
#         reward = 50
#     elif sum>128 and sum<193:
#         reward = 0
#     elif sum>192 and sum<257:
#         reward = -10
#     elif sum>256 and sum<321:
#         reward = -20
#     #reward = (-1) * np.average(qLengths) * np.std(qLengths)
#
#     return reward
#


from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop,Adam

# model = Sequential()
# model.add(Dense(12, input_dim=5, init='uniform', activation='relu'))
# model.add(Dense(24, init='uniform', activation='relu'))
# model.add(Dense(24, init='uniform', activation='relu'))
# model.add(Dense(12, init='uniform', activation='relu'))
# model.add(Dense(2, init='uniform', activation='sigmoid'))
# # Compile model
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model = load_model('traffic_model_uj2.h5')
#reset weights of neural network
epochs = 150
gamma = 0.975
epsilon = 1
batchSize = 80
buffer = 160
replay = []
#stores tuples of (S, A, R, S')
h = 0
generate_routefile()
traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                 "--tripinfo-output", "tripinfo.xml"])




print("Now starting TraCI")



traci.trafficlight.setPhase("0", 0)
state = getState()
counter = 10000

while traci.simulation.getMinExpectedNumber() > 0:

    qval = model.predict(state.reshape(1,5), batch_size=1)

    action = (np.argmax(qval))
        #Take action, observe new state S'
    new_state = makeMove(state, action)

        #Observe reward
    #reward = getReward(new_state)


    state = new_state




