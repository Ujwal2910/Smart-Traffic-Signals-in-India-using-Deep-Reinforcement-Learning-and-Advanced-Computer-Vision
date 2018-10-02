from __future__ import absolute_import
from __future__ import print_function

import optparse
import os
import random
import sys
import cross_read
import numpy as np
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop


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
                    i, episode_length, traffic[i], i), file=routes)

        print("</routes>", file=routes)

    print('TRAFFIC CONFIGURATION - ')
    for i in np.arange(len(traffic)):
        print('Lane %i - %i' % (i + 1, traffic[i]))


# The program looks like this
#    <tlLogic id="0" type="static" programID="0" offset="0">
# the locations of the tls are      NESW
#        <phase duration="31" state="GrGr"/>
#        <phase duration="6"  state="yryr"/>
#        <phase duration="31" state="rGrG"/>
#        <phase duration="6"  state="ryry"/>
#    </tlLogic>

def generate_routefile():
    with open("data/cross_2intersections.rou.xml", "w") as routes:
        print("""<routes>
    <vTypeDistribution id="mixed">
        <vType id="car" vClass="passenger" speedDev="0.2" latAlignment="compact" probability="0.3"/>
        <vType id="moped" vClass="moped" speedDev="0.4" latAlignment="compact" probability="0.7"/>
    </vTypeDistribution>
    <route id="r10" edges="51o 1i 4o 54i"/>
    <route id="r11" edges="51o 1i 010i 14o 154i"/>
    <route id="r12" edges="51o 1i 010i 2o 52i"/>
    <route id="r13" edges="51o 1i 010i 13o 153i"/>
    <route id="r14" edges="51o 1i 3o 53i"/>

    <route id="r20" edges="52o 2i 14o 154i"/>
    <route id="r21" edges="52o 2i 010o 4o 54i"/>
    <route id="r22" edges="52o 2i 010o 1o 51i"/>
    <route id="r23" edges="52o 2i 010o 3o 53i"/>
    <route id="r24" edges="52o 2i 13o 153i"/>

    <route id="r30" edges="53o 3i 1o 51i"/>
    <route id="r31" edges="53o 3i 4o 54i"/>
    <route id="r32" edges="53o 3i 010i 14o 154i"/>
    <route id="r33" edges="53o 3i 010i 2o 52i"/>
    <route id="r34" edges="53o 3i 010i 13o 153i"/>

    <route id="r40" edges="54o 4i 1o 51i"/>
    <route id="r41" edges="54o 4i 3o 53i"/>
    <route id="r42" edges="54o 4i 010i 13o 153i"/>
    <route id="r43" edges="54o 4i 010i 2o 52i"/>
    <route id="r44" edges="54o 4i 010i 14o 154i"/>

    <route id="r130" edges="153o 13i 2o 52i"/>
    <route id="r131" edges="153o 13i 14o 154i"/>
    <route id="r132" edges="153o 13i 010o 4o 54i"/>
    <route id="r133" edges="153o 13i 010o 1o 51i"/>
    <route id="r134" edges="153o 13i 010o 3o 53i"/>

    <route id="r140" edges="154o 14i 2o 52i"/>
    <route id="r141" edges="154o 14i 13o 153i"/>
    <route id="r142" edges="154o 14i 010o 3o 53i"/>
    <route id="r143" edges="154o 14i 010o 1o 51i"/>
    <route id="r144" edges="154o 14i 010o 4o 54i"/>

    <flow id="mixed1" begin="0" end="350" number="150" route="r12" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed2" begin="0" end="350" number="0" route="r22" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed3" begin="0" end="350" number="0" route="r31" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed4" begin="0" end="350" number="0" route="r41" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed5" begin="0" end="350" number="0" route="r131" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed6" begin="0" end="350" number="0" route="r141" type="mixed" departLane="random" departPosLat="random"/>
</routes>""", file=routes)
        lastVeh = 0
        vehNr = 0


try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa-f
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


def get_floor_number(phase_left, phase_right):
    floor_number = 4 * phase_left + phase_right
    return floor_number


def getPhaseState(transition_time):
    phase_left = traci.trafficlight.getPhase("0")  # left and right do
    phase_right = traci.trafficlight.getPhase("10")
    # calculate floor number
    phase = get_floor_number(phase_left, phase_right)

    phaseState = np.zeros((transition_time, num_lanes, num_phases))
    for i in range(transition_time):
        for j in range(num_lanes):
            phaseState[i][j][phase] = 1

    return phaseState


def getState(transition_time):  # made the order changes
    newState = []
    for _ in range(transition_time):
        traci.simulationStep()

        state = [cross_read.leftgetLowerQlength() / 80,  # issi sequnce main left and right
                 cross_read.leftgetRightQlength() / 80,
                 cross_read.leftgetUpperQlength() / 80,
                 cross_read.leftgetLeftQlength() / 80,

                 cross_read.rightgetLowerQlength() / 80,  # issi sequnce main left and right
                 cross_read.rightgetRightQlength() / 80,
                 cross_read.rightgetUpperQlength() / 80,
                 cross_read.rightgetLeftQlength() / 80, ]

        newState.insert(0, state)
    # print (state)
    newState = np.array(newState)
    phaseState = getPhaseState(transition_time)
    newState = np.dstack((newState, phaseState))
    newState = np.expand_dims(newState, axis=0)  # tensor format conversion
    return newState


print("here")
import traci


def makeMove(action, transition_time):
    # new action for 4 diff actions- 00,01,10,11

    if action == 1:
        traci.trafficlight.setPhase("10", (int(traci.trafficlight.getPhase("10")) + 1) % 4)
    elif action == 2:
        traci.trafficlight.setPhase("0", (int(traci.trafficlight.getPhase("0")) + 1) % 4)
    elif action == 3:
        traci.trafficlight.setPhase("0", (int(traci.trafficlight.getPhase("0")) + 1) % 4)
        traci.trafficlight.setPhase("10", (int(traci.trafficlight.getPhase("10")) + 1) % 4)

    # traci.simulationStep()
    # traci.simulationStep()
    # traci.simulationStep()
    # traci.simulationStep()

    return getState(transition_time)


def getReward(this_state, this_new_state):
    num_lanes = 8
    qLengths1 = []
    qLengths2 = []
    for i in range(num_lanes):
        qLengths1.append(this_state[0][i][0])
        qLengths2.append(this_new_state[0][i][0])

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

    if this_reward > 0:
        this_reward = 1
    elif this_reward < 0:
        this_reward = -1
    elif q2 > 1:
        this_reward = -1
    else:
        this_reward = 0

    return this_reward


def build_model(transition_time):
    num_hidden_units_cnn = 10
    num_actions = 4
    model = Sequential()
    model.add(Conv2D(num_hidden_units_cnn, kernel_size=(transition_time, 1), strides=1, activation='relu',
                     input_shape=(transition_time, num_lanes, num_phases + 1)))
    # model.add(LSTM(8))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    opt = RMSprop(lr=0.00025)
    model.compile(loss='mse', optimizer=opt)

    return model


def getWaitingTime(laneID):
    return traci.lane.getWaitingTime(laneID)


num_lanes = 8
num_phases = 16

num_episode = 241
discount_factor = 0.9
# epsilon = 1
epsilon_start = 1
epsilon_end = 0.01
epsilon_decay_steps = 3000  # 40 mins rn

Average_Q_lengths = []
sum_q_lens = 0
AVG_Q_len_perepisode = []

episode_time = 350  # one min episode rl
num_vehicles = 250
transition_time = 8
target_update_time = 20
q_estimator_model = load_model('cross_model_ishan_30_09_30.h5')
#target_estimator_model = build_model(transition_time)
replay_memory_init_size = 35
replay_memory_size = 800
action_memory_size = 30
batch_size = 32
print(q_estimator_model.summary())
epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

# generate_routefile_random(episode_time, num_vehicles)

action_memory = []

generate_routefile()
# generate_routefile_random(episode_time, num_vehicles)
traci.start([sumoBinary, "-c", "data/cross_2intersections.sumocfg",
             "--tripinfo-output", "tripinfo.xml"])

traci.trafficlight.setPhase("0", 0)
traci.trafficlight.setPhase("10", 0)

nA = 4


total_t = 0
for episode in range(num_episode):


    state = getState(transition_time)
    counter = 0
    stride = 0
    while traci.simulation.getMinExpectedNumber() > 0:


        print("Episode # ", episode)

        print("Inside episode counter", counter)

        counter += 1
        total_t += 1


        q_val = q_estimator_model.predict(state)
        print(q_val)

        action = np.argmax(q_val)

        same_action_count = 0
        for temp in reversed(action_memory):
            if temp == 0:
                same_action_count += 1
            else:
                break
        if same_action_count == 20:
            action = 3
            print("SAME ACTION PENALTY")

        new_state = makeMove(action, transition_time)
        print(new_state)

        if len(action_memory) == action_memory_size:
            action_memory.pop(0)

        action_memory.append(action)

        state = new_state

    traci.load(["--start", "-c", "data/cross_2intersections.sumocfg",
                "--tripinfo-output", "tripinfo.xml"])
    traci.trafficlight.setPhase("0", 0)
    traci.trafficlight.setPhase("10", 0)
print(AVG_Q_len_perepisode)

# import matplotlib.pyplot as plt
#
# plt.plot([x for x in range(num_episode)],[AVG_Q_len_perepisode], 'ro')
# plt.axis([0, num_episode, 0, 10])
# plt.show()
