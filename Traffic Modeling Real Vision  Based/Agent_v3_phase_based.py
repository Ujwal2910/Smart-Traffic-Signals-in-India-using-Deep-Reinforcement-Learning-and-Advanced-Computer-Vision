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
from keras.optimizers import RMSprop,Adam
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import readScreen2
import numpy as np
from time import time


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
    <flow id="mixed1" begin="0" end="25000" number="2000" route="r0" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed2" begin="0" end="25000" number="400" route="r1" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed3" begin="0" end="25000" number="400" route="r2" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed4" begin="0" end="25000" number="2000" route="r3" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed5" begin="0" end="25000" number="400" route="r4" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed6" begin="0" end="25000" number="400" route="r5" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed7" begin="0" end="25000" number="2000" route="r6" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed8" begin="0" end="25000" number="400" route="r7" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed9" begin="0" end="25000" number="400" route="r8" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed10" begin="0" end="25000" number="2000" route="r9" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed11" begin="0" end="25000" number="400" route="r10" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed12" begin="0" end="25000" number="400" route="r11" type="mixed" departLane="random" departPosLat="random"/>
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


def getState():#made the order changes
    state = [readScreen2.getLowerQlength() / 80,
             readScreen2.getRightQlength() / 80,
             readScreen2.getUpperQlength() / 80,
             readScreen2.getLeftQlength() / 80,
             traci.trafficlight.getPhase("0")]

    #print (state)

    return state



print("here")
import traci



def makeMove(phase, transition_time):

    traci.trafficlight.setPhase("0", phase)

    for _ in range(transition_time):
        traci.simulationStep()

    # traci.simulationStep()
    # traci.simulationStep()
    # traci.simulationStep()
    # traci.simulationStep()

    newState = getState()

    return newState


def getReward(this_state, this_new_state):
    qLengths1 = this_state[:4]
    qLengths2 = this_new_state[:4]

    q1 = np.prod(qLengths1 + 1)
    q2 = np.prod(qLengths2 + 1)

    print("Old State with product : ", q1)

    print("New State with product : ", q2)


    if q1 > q2:
        this_reward = 1
    else:
        this_reward = -1

    return this_reward



def build_model(history):
    num_hidden_units_lstm = 64
    num_actions = 4
    model = Sequential()
    model.add(LSTM(num_hidden_units_lstm, input_shape=(history, 5)))
    #model.add(LSTM(8))
    model.add(Dense(num_actions, activation='linear'))
    opt = RMSprop(lr=0.00025)
    model.compile(loss='mse', optimizer=opt)

    return model

num_episode = 50
gamma = 0.95
epsilon = 1


num_history = 100
transition_time = 50
model = build_model(num_history)

generate_routefile()
traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                 "--tripinfo-output", "tripinfo.xml"])

traci.trafficlight.setPhase("0", 0)

nA = 4
state = getState()
experience = []
for i in range(num_history):
    experience.append(state)

for episode in range(num_episode):
    traci.trafficlight.setPhase("0", 0)

    print("INITIAL EXPERIENCE : ")
    print(experience)
    counter = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        print("Episode # ", episode)

        print("Inside episode counter",counter)
        counter+=1
        q_val = model.predict((np.array(experience)).reshape((1, num_history, 5)))
        print(q_val)
        # if random.random() < epsilon:
        #     phase = np.random.choice(4)
        #     print("random action chosen",phase)
        # else:
        #     phase = np.argmax(q_val)
        #     print("else action",phase)
        epsilon = 1.0 / (episode+1)
        policy_s = np.ones(nA) * epsilon / nA

        policy_s[np.argmax(q_val)] = 1 - epsilon + (epsilon / nA)

        action = np.random.choice(np.arange(nA), p=policy_s)
        #phase = np.argmax(q_val)
        phase = action
        print("else action", phase)
        if np.argmax(q_val) != action:
            print("RANDOM CHOICE TAKEN")
        else:
            print("POLICY FOLLOWED ")
        new_state = makeMove(phase,transition_time)
        old_experience = experience
        experience.pop(0)
        experience.append(new_state)
        reward = getReward(state, new_state)
        oracle = np.zeros((1, 4))
        oracle[:] = q_val[:]
        print(reward)
        oracle[0][phase] = (reward + gamma*np.max(model.predict((np.array(experience)).reshape((1, num_history, 5)))))
        print(oracle)
        model.fit((np.array(old_experience)).reshape((1, num_history, 5)), oracle, verbose=1)
        state = new_state
    # if epsilon > 0.01:
    #     epsilon -= (1/num_episode)
    model.save('lstm_phase_50ep_25ks_0907.h5')
    traci.load(["--start", "-c", "data/cross.sumocfg",
                "--tripinfo-output", "tripinfo.xml"])



