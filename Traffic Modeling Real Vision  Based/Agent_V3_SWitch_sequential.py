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
from keras.optimizers import RMSprop, Adam
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import readscreen3
import numpy as np
from time import time


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]


def generate_routefile_random(episode_length, total_vehicles):
    N_ROADS = 4
    division = constrained_sum_sample_pos(N_ROADS, total_vehicles)
    traffic = []

    for i in np.arange(len(division)):
        traffic.append(division[i] * 0.6)
        traffic.append(division[i] * 0.2)
        traffic.append(division[i] * 0.2)

    with open("data/cross.rou.xml", "w") as routes:
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
        <route id="r11" edges="53o 3i 2o 52i"/>""", file=routes)

        for i in np.arange(len(traffic)):
            print(
                '<flow id="mixed%i" begin="0" end="%i" number="%i" route="r%i" type="mixed" departLane="random" departPosLat="random"/>' % (
                i, episode_length, traffic[i], i), file = routes)

        print("</routes>", file=routes)

    print('TRAFFIC CONFIGURATION - ')
    for i in np.arange(len(traffic)):
        print('Lane %i - %i' % (i+1, traffic[i]))

# The program looks like this
#    <tlLogic id="0" type="static" programID="0" offset="0">
# the locations of the tls are      NESW
#        <phase duration="31" state="GrGr"/>
#        <phase duration="6"  state="yryr"/>
#        <phase duration="31" state="rGrG"/>
#        <phase duration="6"  state="ryry"/>
#    </tlLogic>

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
    <flow id="mixed1" begin="0" end="5000" number="1000" route="r0" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed2" begin="0" end="5000" number="100" route="r1" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed3" begin="0" end="5000" number="200" route="r2" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed4" begin="0" end="5000" number="1000" route="r3" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed5" begin="0" end="5000" number="100" route="r4" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed6" begin="0" end="5000" number="200" route="r5" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed7" begin="0" end="5000" number="1000" route="r6" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed8" begin="0" end="5000" number="100" route="r7" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed9" begin="0" end="5000" number="200" route="r8" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed10" begin="0" end="5000" number="1000" route="r9" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed11" begin="0" end="5000" number="200" route="r10" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed12" begin="0" end="5000" number="100" route="r11" type="mixed" departLane="random" departPosLat="random"/>
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


# State = State_Lengths()
# print(State.get_tails())

# states = State.get_tails


# runner = Runner()
# print(Runner().run)


def getState():  # made the order changes
    state = [readscreen3.getLowerQlength() / 80,
             readscreen3.getRightQlength() / 80,
             readscreen3.getUpperQlength() / 80,
             readscreen3.getLeftQlength() / 80,
             traci.trafficlight.getPhase("0")]

    # print (state)

    return state


print("here")
import traci


def makeMove(action, transition_time, experience):
    if action == 1:
        traci.trafficlight.setPhase("0", (int(traci.trafficlight.getPhase("0")) + 1) % 4)

    for _ in range(transition_time):

        traci.simulationStep()


    # traci.simulationStep()
    # traci.simulationStep()
    # traci.simulationStep()
    # traci.simulationStep()

    #newState = getState()
    experience.pop(0)
    experience.append(getState())
    return experience


def getReward(this_state, this_new_state):
    qLengths1 = this_state[:4]
    qLengths2 = this_new_state[:4]

    qLengths11 = [x + 1 for x in qLengths1]
    qLengths21 = [x + 1 for x in qLengths2]

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

    return this_reward


def build_model(history):
    num_hidden_units_lstm = 10
    num_actions = 2
    model = Sequential()
    model.add(Dense(output_dim = 12, input_dim=5, init='uniform', activation='relu'))
    model.add(Dense(output_dim =24, init='uniform', activation='relu'))
    # model.add(Dense(24, init='uniform', activation='relu'))
    model.add(Dense(output_dim =12, init='uniform', activation='relu'))
    model.add(Dense(output_dim =2, init='uniform', activation='linear'))
    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model


def getWaitingTime(laneID):
    return traci.lane.getWaitingTime(laneID)


num_episode = 101
gamma = 0.95
epsilon = 1
num_batches = 25
Average_Q_lengths = []
sum_q_lens = 0
AVG_Q_len_perepisode = []
num_history = 100
episode_time = 5000
num_vehicles = 5200
transition_time = 30
model = build_model(num_history)
print(model.summary())

generate_routefile_random(episode_time, num_vehicles)
traci.start([sumoBinary, "-c", "data/cross.sumocfg",
             "--tripinfo-output", "tripinfo.xml"])

traci.trafficlight.setPhase("0", 0)

nA = 2  ###
state = getState()
experience = []
# for i in range(num_history):
#     experience.append(state)
experience.append(state)
print(np.array(experience).shape)

for episode in range(num_episode):
    traci.trafficlight.setPhase("0", 0)

    print("INITIAL EXPERIENCE : ")
    print(experience)
    counter = 0
    stride = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        print("Episode # ", episode)
        # print("Waiting time on lane 1i_0 = ",getWaitingTime("1i_0"))

        print("Inside episode counter", counter)
        print(experience)
        counter += 1
        # batch_experience = experience[:batch_history]
        q_val = model.predict((np.array(experience)).reshape(1,5))
        print(q_val)
        # if random.random() < epsilon:
        #     phase = np.random.choice(4)
        #     print("random action chosen",phase)
        # else:
        #     phase = np.argmax(q_val)
        #     print("else action",phase)
        epsilon = 1.0 / (episode + 1)
        policy_s = np.ones(nA) * epsilon / nA

        policy_s[np.argmax(q_val)] = 1 - epsilon + (epsilon / nA)

        action = np.random.choice(np.arange(nA), p=policy_s)
        old_experience = experience
        if np.argmax(q_val) != action:
            print("RANDOM CHOICE TAKEN")
        else:
            print("POLICY FOLLOWED ")
        experience = makeMove(action, transition_time, experience)
        new_state = getState()
        Average_Q_lengths = new_state[:4]
        sum_q_lens += np.average(Average_Q_lengths)

        reward = getReward(state, new_state)

        oracle = np.zeros((1, nA))
        oracle[:] = q_val[:]
        print(reward)
        oracle[0][action] = (
                    reward + gamma * np.max(model.predict((np.array(experience)).reshape(1,5))))
        print(oracle)
        model.fit((np.array(old_experience)).reshape(1,5), oracle, verbose=1)
        state = new_state

    AVG_Q_len_perepisode.append(sum_q_lens / 100)
    sum_q_lens = 0
    if episode % 25 == 0:
        model.save('sequential_switch_2007_{}.h5'.format(episode))

    generate_routefile_random(episode_time, num_vehicles)
    traci.load(["--start", "-c", "data/cross.sumocfg",
                "--tripinfo-output", "tripinfo.xml"])

print(AVG_Q_len_perepisode)

# import matplotlib.pyplot as plt
#
# plt.plot([x for x in range(num_episode)],[AVG_Q_len_perepisode], 'ro')
# plt.axis([0, num_episode, 0, 10])
# plt.show()
