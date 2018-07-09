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
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import readScreen2
import numpy as np


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

def generate_routefile():
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
    <flow id="mixed1" begin="0" end="1500" number="800" route="r0" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed2" begin="0" end="1500" number="80" route="r1" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed3" begin="0" end="1500" number="40" route="r2" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed4" begin="0" end="1500" number="0" route="r3" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed5" begin="0" end="1500" number="0" route="r4" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed6" begin="0" end="1500" number="0" route="r5" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed7" begin="0" end="1500" number="0" route="r6" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed8" begin="0" end="1500" number="0" route="r7" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed9" begin="0" end="1500" number="0" route="r8" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed10" begin="0" end="1500" number="0" route="r9" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed11" begin="0" end="1500" number="0" route="r10" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed12" begin="0" end="1500" number="0" route="r11" type="mixed" departLane="random" departPosLat="random"/>
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
    state = [readScreen2.getUpperQlength(),
             readScreen2.getLowerQlength(),
             readScreen2.getRightQlength(),
             readScreen2.getLeftQlength(),
             traci.trafficlight.getPhase("0")]

    print (state)

    return state



print("here")
import traci



def makeMove(state,action):

    if action == 1:
        traci.trafficlight.setPhase("0", (int(state[4])+1) % 4)
    traci.simulationStep()
    traci.simulationStep()
    traci.simulationStep()
    traci.simulationStep()

    new_state = getState()

    return new_state

def getReward(state,new_state):
    qLengths1 = state[:4]
    qLengths2 = new_state[:4]

    q1 = np.average(qLengths1)*np.std(qLengths1)
    q2 = np.average(qLengths2)*np.std(qLengths2)

    if q1 >= q2:
        reward = -1
    elif q1<q2:
        reward = 1

    return reward



def build_model():
    num_hidden_units_lstm = 32
    num_actions = 2
    model = Sequential()
    model.add(LSTM(num_hidden_units_lstm, input_shape=(100, 5)))
    model.add(Dense(num_actions, activation='linear'))
    opt = RMSprop(lr=0.00025)
    model.compile(loss='mse', optimizer=opt)

    return model

num_episode = 400
gamma = 0.99
epsilon = 1
buffer = 100

#model = build_model()
model = load_model('lstm_v5.h5')

generate_routefile()
traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                 "--tripinfo-output", "tripinfo.xml"])

traci.trafficlight.setPhase("0", 0)


traci.trafficlight.setPhase("0", 0)
state = getState()
experience = []
for i in range(100):
    experience.append(state)#-----
while traci.simulation.getMinExpectedNumber() > 0:
    q_val = model.predict((np.array(experience)).reshape((1, 100, 5)))
    action = np.argmax(q_val)
    state = makeMove(state, action)
    experience.pop(0)
    experience.append(state)







