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
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import TensorBoard
import readscreen3
import numpy as np
import datetime
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
    with open("data/cross_multi.rou.xml", "w") as routes:
        print("""<routes>
    <vTypeDistribution id="mixed">
        <vType id="car" vClass="passenger" speedDev="0.2" latAlignment="compact" probability="0.3"/>
        <vType id="moped" vClass="moped" speedDev="0.4" latAlignment="compact" probability="0.7"/>
    </vTypeDistribution>
    <route id="r10" edges="51o 1i 010u 4o 54i"/>
    <route id="r11" edges="51o 1i 010i 1110u 14o 154i"/>
    <route id="r12" edges="51o 1i 010i 2o 52i"/>
    <route id="r13" edges="51o 1i 010i 13o 153i"/>
    <route id="r14" edges="51o 1i 3o 53i"/>
    
    <route id="r1010" edges="1051o 101i 4o 54i"/>
    <route id="r1011" edges="1051o 101i 0111i 14o 154i"/>
    <route id="r1012" edges="1051o 101i 0111i 102o 1052i"/>
    <route id="r1013" edges="1051o 101i 0111i 1110d 13o 153i"/>
    <route id="r1014" edges="1051o 101i 010d 3o 53i"/>
    
    <route id="r20" edges="52o 2i 1110u 14o 154i"/>
    <route id="r21" edges="52o 2i 010o 010u 4o 54i"/>
    <route id="r22" edges="52o 2i 010o 1o 51i"/>
    <route id="r23" edges="52o 2i 010o 3o 53i"/>
    <route id="r24" edges="52o 2i 13o 153i"/>
    
    <route id="r1020" edges="1052o 102i 14o 154i"/>
    <route id="r1021" edges="1052o 102i 0111o 4o 54i"/>
    <route id="r1022" edges="1052o 102i 0111o 101o 1051i"/>
    <route id="r1023" edges="1052o 102i 0111o 010d 3o 53i"/>
    <route id="r1024" edges="1052o 102i 1110d 13o 153i"/>
    
    
    
    <route id="r30" edges="53o 3i 1o 51i"/>
    <route id="r31" edges="53o 3i 010u 4o 54i"/>
    <route id="r32" edges="53o 3i 010i 1110u 14o 154i"/>
    <route id="r33" edges="53o 3i 010i 2o 52i"/>
    <route id="r34" edges="53o 3i 010i 13o 153i"/>
    
    <route id="r40" edges="54o 4i 010d 1o 51i"/>
    <route id="r41" edges="54o 4i 010d 3o 53i"/>
    <route id="r42" edges="54o 4i 010d 010i 13o 153i"/>
    <route id="r43" edges="54o 4i 010d 010i 2o 52i"/>
    <route id="r44" edges="54o 4i 010d 010i 1110u 14o 154i"/>
    
    <route id="r130" edges="153o 13i 2o 52i"/>
    <route id="r131" edges="153o 13i 1110u 14o 154i"/>
    <route id="r132" edges="153o 13i 010o 010u 4o 54i"/>
    <route id="r133" edges="153o 13i 010o 1o 51i"/>
    <route id="r134" edges="153o 13i 010o 3o 53i"/>
    
    <route id="r140" edges="154o 14i 1110d 2o 52i"/>
    <route id="r141" edges="154o 14i 1110d 13o 153i"/>
    <route id="r142" edges="154o 14i 1110d 010o 3o 53i"/>
    <route id="r143" edges="154o 14i 1110d 010o 1o 51i"/>
    <route id="r144" edges="154o 14i 1110d 010o 010u 4o 54i"/>

    <flow id="mixed1" begin="0" end="350" number="%i" route="r10" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed2" begin="0" end="350" number="%i" route="r11" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed3" begin="0" end="350" number="%i" route="r12" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed4" begin="0" end="350" number="%i" route="r13" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed5" begin="0" end="350" number="%i" route="r14" type="mixed" departLane="random" departPosLat="random"/>

    <flow id="mixed6" begin="0" end="350" number="%i" route="r20" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed7" begin="0" end="350" number="%i" route="r21" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed8" begin="0" end="350" number="%i" route="r22" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed9" begin="0" end="350" number="%i" route="r23" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed10" begin="0" end="350" number="%i" route="r24" type="mixed" departLane="random" departPosLat="random"/>

    <flow id="mixed11" begin="0" end="350" number="%i" route="r30" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed12" begin="0" end="350" number="%i" route="r31" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed13" begin="0" end="350" number="%i" route="r32" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed14" begin="0" end="350" number="%i" route="r33" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed15" begin="0" end="350" number="%i" route="r34" type="mixed" departLane="random" departPosLat="random"/>

    <flow id="mixed16" begin="0" end="350" number="%i" route="r40" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed17" begin="0" end="350" number="%i" route="r41" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed18" begin="0" end="350" number="%i" route="r42" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed19" begin="0" end="350" number="%i" route="r43" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed20" begin="0" end="350" number="%i" route="r44" type="mixed" departLane="random" departPosLat="random"/>

    <flow id="mixed21" begin="0" end="350" number="%i" route="r130" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed22" begin="0" end="350" number="%i" route="r131" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed23" begin="0" end="350" number="%i" route="r132" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed24" begin="0" end="350" number="%i" route="r133" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed25" begin="0" end="350" number="%i" route="r134" type="mixed" departLane="random" departPosLat="random"/>

    <flow id="mixed26" begin="0" end="350" number="%i" route="r140" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed27" begin="0" end="350" number="%i" route="r141" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed28" begin="0" end="350" number="%i" route="r142" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed29" begin="0" end="350" number="%i" route="r143" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed30" begin="0" end="350" number="%i" route="r144" type="mixed" departLane="random" departPosLat="random"/>

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


def getLeftPhaseState(transition_time):
    num_lanes = 4
    num_phases = 4
    phase = traci.trafficlight.getPhase("0")
    phaseState = np.zeros((transition_time, num_lanes, num_phases))
    for i in range(transition_time):
        for j in range(num_lanes):
            phaseState[i][j][phase] = 1
    return phaseState

def getUpperLeftPhaseState(transition_time):
    num_lanes = 4
    num_phases = 4
    phase = traci.trafficlight.getPhase("01")
    phaseState = np.zeros((transition_time, num_lanes, num_phases))
    for i in range(transition_time):
        for j in range(num_lanes):
            phaseState[i][j][phase] = 1
    return phaseState


def getRightPhaseState(transition_time):
    num_lanes = 4
    num_phases = 4
    phase = traci.trafficlight.getPhase("10")
    phaseState = np.zeros((transition_time, num_lanes, num_phases))
    for i in range(transition_time):
        for j in range(num_lanes):
            phaseState[i][j][phase] = 1
    return phaseState

def getUpperRightPhaseState(transition_time):
    num_lanes = 4
    num_phases = 4
    phase = traci.trafficlight.getPhase("11")
    phaseState = np.zeros((transition_time, num_lanes, num_phases))
    for i in range(transition_time):
        for j in range(num_lanes):
            phaseState[i][j][phase] = 1
    return phaseState


def getStates(transition_time):  # made the order changes
    newLeftState = []
    newRightState = []
    newUpperLeftState = []
    newUpperRightState = []

    for _ in range(transition_time):
        traci.simulationStep()

        l_leftcount = 0
        l_rightcount = 0
        l_topcount = 0
        l_bottomcount = 0

        r_leftcount = 0
        r_rightcount = 0
        r_topcount = 0
        r_bottomcount = 0

        ul_leftcount = 0
        ul_rightcount = 0
        ul_topcount = 0
        ul_bottomcount = 0

        ur_leftcount = 0
        ur_rightcount = 0
        ur_topcount = 0
        ur_bottomcount = 0


        vehicleList = traci.vehicle.getIDList()

        for id in vehicleList:
            x, y = traci.vehicle.getPosition(id)

            if x < 500 and x > 450 and y < 520 and y > 510:
                l_leftcount += 1
            else:
                if x < 510 and x > 500 and y < 500 and y > 450:
                    l_bottomcount += 1
                else:
                    if x < 570 and x > 520 and y < 510 and y > 500:
                        l_rightcount += 1
                    else:
                        if x < 520 and x > 510 and y < 570 and y > 520:
                            l_topcount += 1

            if x < 600 and x > 550 and y < 520 and y > 510:
                r_leftcount += 1
            else:
                if x < 610 and x > 600 and y < 500 and y > 450:
                    r_bottomcount += 1
                else:
                    if x < 670 and x > 620 and y < 510 and y > 500:
                        r_rightcount += 1
                    else:
                        if x < 620 and x > 610 and y < 570 and y > 520:
                            r_topcount += 1

            if x < 500 and x > 450 and y < 620 and y > 610:
                ul_leftcount += 1
            else:
                if x < 510 and x > 500 and y < 600 and y > 550:
                    ul_bottomcount += 1
                else:
                    if x < 570 and x > 520 and y < 610 and y > 600:
                        ul_rightcount += 1
                    else:
                        if x < 520 and x > 510 and y < 670 and y > 620:
                            ul_topcount += 1

            if x < 600 and x > 550 and y < 620 and y > 610:
                ur_leftcount += 1
            else:
                if x < 610 and x > 600 and y < 600 and y > 550:
                    ur_bottomcount += 1
                else:
                    if x < 670 and x > 620 and y < 610 and y > 600:
                        ur_rightcount += 1
                    else:
                        if x < 620 and x > 610 and y < 670 and y > 620:
                            ur_topcount += 1

        print("Left Intersection Left Lane : ", l_leftcount)
        print("Left Intersection Right Lane : ", l_rightcount)
        print("Left Intersection Top Lane : ", l_topcount)
        print("Left Intersection Bottom Lane : ", l_bottomcount)

        print("Right Intersection Left Lane : ", r_leftcount)
        print("Right Intersection Right Lane : ", r_rightcount)
        print("Right Intersection Top Lane : ", r_topcount)
        print("Right Intersection Bottom Lane : ", r_bottomcount)

        print("Upper Left Intersection Left Lane : ", ul_leftcount)
        print("Upper Left Intersection Right Lane : ", ul_rightcount)
        print("Upper Left Intersection Top Lane : ", ul_topcount)
        print("Upper Left Intersection Bottom Lane : ", ul_bottomcount)

        print("Upper Right Intersection Left Lane : ", ur_leftcount)
        print("Upper Right Intersection Right Lane : ", ur_rightcount)
        print("Upper Right Intersection Top Lane : ", ur_topcount)
        print("Upper Right Intersection Bottom Lane : ", ur_bottomcount)

        leftState = [l_bottomcount / 40,
                     l_rightcount / 40,
                     l_topcount / 40,
                     l_leftcount / 40]

        rightState = [r_bottomcount / 40,
                      r_rightcount / 40,
                      r_topcount / 40,
                      r_leftcount / 40]

        upperLeftState = [ul_bottomcount / 40,
                     ul_rightcount / 40,
                     ul_topcount / 40,
                     ul_leftcount / 40]

        upperRightState = [ur_bottomcount / 40,
                      ur_rightcount / 40,
                      ur_topcount / 40,
                      ur_leftcount / 40]

        newLeftState.insert(0, leftState)
        newRightState.insert(0, rightState)
        newUpperLeftState.insert(0, upperLeftState)
        newUpperRightState.insert(0, upperRightState)

    newLeftState = np.array(newLeftState)
    leftPhaseState = getLeftPhaseState(transition_time)
    newLeftState = np.dstack((newLeftState, leftPhaseState))
    newLeftState = np.expand_dims(newLeftState, axis=0)

    newUpperLeftState = np.array(newUpperLeftState)
    upperLeftPhaseState = getUpperLeftPhaseState(transition_time)
    newUpperLeftState = np.dstack((newUpperLeftState, upperLeftPhaseState))
    newUpperLeftState = np.expand_dims(newUpperLeftState, axis=0)

    newRightState = np.array(newRightState)
    rightPhaseState = getRightPhaseState(transition_time)
    newRightState = np.dstack((newRightState, rightPhaseState))
    newRightState = np.expand_dims(newRightState, axis=0)

    newUpperRightState = np.array(newUpperRightState)
    upperRightPhaseState = getUpperRightPhaseState(transition_time)
    newUpperRightState = np.dstack((newUpperRightState, upperRightPhaseState))
    newUpperRightState = np.expand_dims(newUpperRightState, axis=0)

    return newLeftState, newRightState, newUpperLeftState, newUpperRightState


print("here")
import traci

def makeMoves(leftAction, rightAction, upperLeftAction, upperRightAction, transition_time):
    if leftAction == 1:
        traci.trafficlight.setPhase("0", (int(traci.trafficlight.getPhase("0")) + 1) % 4)
    if rightAction == 1:
        traci.trafficlight.setPhase("10", (int(traci.trafficlight.getPhase("10")) + 1) % 4)
    if upperLeftAction == 1:
        traci.trafficlight.setPhase("01", (int(traci.trafficlight.getPhase("01")) + 1) % 4)
    if upperRightAction == 1:
        traci.trafficlight.setPhase("11", (int(traci.trafficlight.getPhase("11")) + 1) % 4)

    return getStates(transition_time)


def getReward(this_state, this_new_state):
    num_lanes = 4
    qLengths1 = []
    qLengths2 = []
    for i in range(num_lanes):
        qLengths1.append(this_state[0][0][i][0])
        qLengths2.append(this_new_state[0][0][i][0])

    qLengths11 = [x + 1 for x in qLengths1]
    qLengths21 = [x + 1 for x in qLengths2]

    print(this_new_state)

    q1 = np.prod(qLengths11)
    q2 = np.prod(qLengths21)

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
    num_actions = 2
    model = Sequential()
    model.add(Conv2D(num_hidden_units_cnn, kernel_size=(transition_time, 1), strides=1, activation='relu',
                     input_shape=(transition_time, 4, 5)))
    # model.add(LSTM(8))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    opt = RMSprop(lr=0.00025)
    model.compile(loss='mse', optimizer=opt)

    return model


def getWaitingTime(laneID):
    return traci.lane.getWaitingTime(laneID)


num_episode = 361
discount_factor = 0.9
# epsilon = 1
epsilon_start = 1
epsilon_end = 0.01
epsilon_decay_steps = 25000

Average_Q_lengths = []
sum_q_lens = 0
AVG_Q_len_perepisode = []

episode_time = 350
num_vehicles = 250
transition_time = 8
target_update_time = 20
q_estimator_model_left = build_model(transition_time)
target_estimator_model_left = build_model(transition_time)

replay_memory_init_size = 200
replay_memory_size = 1800
batch_size = 32
print(q_estimator_model_left.summary())
epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

# generate_routefile_random(episode_time, num_vehicles)

#generate_routefile()
# generate_routefile_random(episode_time, num_vehicles)
traci.start([sumoBinary, "-c", "data/cross_multi.sumocfg",
             "--tripinfo-output", "tripinfo.xml"])

traci.trafficlight.setPhase("0", 0)
traci.trafficlight.setPhase("10", 0)
traci.trafficlight.setPhase("01", 0)
traci.trafficlight.setPhase("11", 0)

nA = 2

target_estimator_model_left.set_weights(q_estimator_model_left.get_weights())


left_replay_memory = []


for _ in range(replay_memory_init_size):
    if traci.simulation.getMinExpectedNumber() <= 0:
        generate_routefile(100, 0)
        traci.load(["--start", "-c", "data/cross_multi.sumocfg",
                    "--tripinfo-output", "tripinfo.xml"])
    leftState, rightState, upperLeftState, upperRightState = getStates(transition_time)
    leftAction = np.random.choice(np.arange(nA))
    rightAction = np.random.choice(np.arange(nA))
    upperLeftAction = np.random.choice(np.arange(nA))
    upperRightAction = np.random.choice(np.arange(nA))
    newLeftState, newRightState, newUpperLeftState, newUpperRightState = makeMoves(leftAction, rightAction, upperLeftAction, upperRightAction,transition_time)
    leftReward = getReward(leftState, newLeftState)
    rightReward = getReward(rightState, newRightState)
    upperLeftReward = getReward(upperLeftState, newUpperLeftState)
    upperRightReward = getReward(upperRightState, newUpperRightState)
    left_replay_memory.append([leftState, leftAction, leftReward, newLeftState])
    left_replay_memory.append([rightState, rightAction, rightReward, newRightState])
    left_replay_memory.append([upperLeftState, upperLeftAction, upperLeftReward, newUpperLeftState])
    left_replay_memory.append([upperRightState, upperRightAction, upperRightReward, newUpperRightState])
    print(len(left_replay_memory))

total_t = 0
for episode in range(num_episode):



    #generate_routefile()
    # generate_routefile_random(episode_time, num_vehicles)
    traci.load(["--start", "-c", "data/cross_multi.sumocfg",
                "--tripinfo-output", "tripinfo.xml"])
    traci.trafficlight.setPhase("0", 0)
    traci.trafficlight.setPhase("10", 0)
    traci.trafficlight.setPhase("01", 0)
    traci.trafficlight.setPhase("11", 0)

    leftState, rightState, upperLeftState, upperRightState = getStates(transition_time)
    counter = 0
    stride = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        print("Episode # ", episode)
        # print("Waiting time on lane 1i_0 = ",getWaitingTime("1i_0"))

        print("Inside episode counter", counter)

        counter += 1
        total_t += 1
        # batch_experience = experience[:batch_history]

        if total_t % target_update_time == 0:
            target_estimator_model_left.set_weights(q_estimator_model_left.get_weights())


        q_val_left = q_estimator_model_left.predict(leftState)
        q_val_right = q_estimator_model_left.predict(rightState)
        q_val_upper_left = q_estimator_model_left.predict(upperLeftState)
        q_val_upper_right = q_estimator_model_left.predict(upperRightState)

        print("Left q values : ", q_val_left)
        print("Right q values : ", q_val_right)
        print("Upper Left q values : ", q_val_upper_left)
        print("Upper Right q values : ", q_val_upper_right)


        epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
        print("Epsilon -", epsilon)
        policy_s = np.ones(nA) * epsilon / nA

        leftPolicy = np.copy(policy_s)
        leftPolicy[np.argmax(q_val_left)] = 1 - epsilon + (epsilon / nA)

        rightPolicy = np.copy(policy_s)
        rightPolicy[np.argmax(q_val_right)] = 1 - epsilon + (epsilon / nA)

        upperLeftPolicy = np.copy(policy_s)
        upperLeftPolicy[np.argmax(q_val_upper_left)] = 1 - epsilon + (epsilon / nA)

        upperRightPolicy = np.copy(policy_s)
        upperRightPolicy[np.argmax(q_val_upper_right)] = 1 - epsilon + (epsilon / nA)

        leftAction = np.random.choice(np.arange(nA), p=leftPolicy)
        rightAction = np.random.choice(np.arange(nA), p=rightPolicy)
        upperLeftAction = np.random.choice(np.arange(nA), p=upperLeftPolicy)
        upperRightAction = np.random.choice(np.arange(nA), p=upperRightPolicy)

        '''
        same_left_action_count = 0
        for temp in reversed(left_replay_memory):
            if temp[1] == 0:
                same_left_action_count += 1
            else:
                break
        if same_left_action_count == 20:
            leftAction = 1
            print("SAME LEFT ACTION PENALTY")

        same_right_action_count = 0
        for temp in reversed(right_replay_memory):
            if temp[1] == 0:
                same_right_action_count += 1
            else:
                break
        if same_right_action_count == 20:
            rightAction = 1
            print("SAME RIGHT ACTION PENALTY")
        '''
        if np.argmax(q_val_left) != leftAction:
            print("RANDOM LEFT CHOICE TAKEN")
        else:
            print("LEFT POLICY FOLLOWED ")

        if np.argmax(q_val_right) != rightAction:
            print("RANDOM RIGHT CHOICE TAKEN")
        else:
            print("RIGHT POLICY FOLLOWED ")

        if np.argmax(q_val_upper_left) != upperLeftAction:
            print("RANDOM UPPER LEFT CHOICE TAKEN")
        else:
            print("UPPER LEFT POLICY FOLLOWED ")

        if np.argmax(q_val_upper_right) != upperRightAction:
            print("RANDOM UPPER RIGHT CHOICE TAKEN")
        else:
            print("UPPER RIGHT POLICY FOLLOWED ")

        newLeftState, newRightState, newUpperLeftState, newUpperRightState = makeMoves(leftAction, rightAction, upperLeftAction, upperRightAction, transition_time)
        leftReward = getReward(leftState, newLeftState)
        rightReward = getReward(rightState, newRightState)
        upperLeftReward = getReward(upperLeftState, newUpperLeftState)
        upperRightReward = getReward(upperRightState, newUpperRightState)

        print("Left reward : ", leftReward)
        print("Right reward : ", rightReward)
        print("Upper Left reward : ", upperLeftReward)
        print("Upper Right reward : ", upperRightReward)

        if len(left_replay_memory) == replay_memory_size:
            left_replay_memory.pop(0)

        left_replay_memory.append([leftState, leftAction, leftReward, newLeftState])

        if len(left_replay_memory) == replay_memory_size:
            left_replay_memory.pop(0)

        left_replay_memory.append([rightState, rightAction, rightReward, newRightState])

        if len(left_replay_memory) == replay_memory_size:
            left_replay_memory.pop(0)

        left_replay_memory.append([upperLeftState, upperLeftAction, upperLeftReward, newUpperLeftState])

        if len(left_replay_memory) == replay_memory_size:
            left_replay_memory.pop(0)

        left_replay_memory.append([upperRightState, upperRightAction, upperRightReward, newUpperRightState])

        print("Memory Length :", len(left_replay_memory))

        leftSamples = random.sample(left_replay_memory, batch_size)


        # MODEL FITTING FOR LEFT
        x_batch, y_batch = [], []
        for inst_state, inst_action, inst_reward, inst_next_state in leftSamples:
            y_target = q_estimator_model_left.predict(inst_state)
            q_val_next = target_estimator_model_left.predict(inst_next_state)
            y_target[0][inst_action] = inst_reward + discount_factor * np.amax(
                q_val_next, axis=1
            )
            x_batch.append(inst_state[0])
            y_batch.append(y_target[0])

        q_estimator_model_left.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

        leftState = newLeftState
        rightState = newRightState

    AVG_Q_len_perepisode.append(sum_q_lens / 702)
    sum_q_lens = 0
    if episode % 10 == 0:
        q_estimator_model_left.save('models/sequential/multi_intersection_{}.h5'.format(episode))


print(AVG_Q_len_perepisode)

# import matplotlib.pyplot as plt
#
# plt.plot([x for x in range(num_episode)],[AVG_Q_len_perepisode], 'ro')
# plt.axis([0, num_episode, 0, 10])
# plt.show()