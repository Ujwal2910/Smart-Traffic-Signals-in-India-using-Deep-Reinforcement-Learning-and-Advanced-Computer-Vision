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
#import cv2
import curses

#from awscli.customizations.emr.constants import TRUE
from keras.optimizers import RMSprop, Adam
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import TensorBoard
#import readscreen3
import numpy as np
import pandas as pd
import datetime
from time import time
import matplotlib.pyplot as plt
from operator import add


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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

def generate_routefile(left_qty, up_qty):
    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
    <!--<vTypeDistribution id="mixed">-->
        <!--<vType id="car" vClass="passenger" speedDev="0.2" latAlignment="compact" probability="0.3"/>-->
        <!--<vType id="moped" vClass="moped" speedDev="0.4" latAlignment="compact" probability="0.7"/>-->
    <!--</vTypeDistribution>-->
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
    <vehicle id='motorcycle0' type='motorcycle' route='r0' depart='5'/>
    <vehicle id='motorcycle1' type='motorcycle' route='r1' depart='5'/>
    <vehicle id='motorcycle2' type='motorcycle' route='r2' depart='5'/>
    <vehicle id='motorcycle3' type='motorcycle' route='r3' depart='5'/>
    <vehicle id='motorcycle4' type='motorcycle' route='r4' depart='5'/>
    <vehicle id='motorcycle5' type='motorcycle' route='r5' depart='10'/>
    <vehicle id='motorcycle6' type='motorcycle' route='r6' depart='10'/>
    <vehicle id='motorcycle7' type='motorcycle' route='r7' depart='10'/>
    <vehicle id='motorcycle8' type='motorcycle' route='r8' depart='10'/>
    <vehicle id='motorcycle9' type='motorcycle' route='r9' depart='10'/>
    <vehicle id='passenger10' type='passenger' route='r10' depart='15'/>
    <vehicle id='passenger11' type='passenger' route='r11' depart='15'/>
    <vehicle id='passenger12' type='passenger' route='r0' depart='15'/>
    <vehicle id='passenger13' type='passenger' route='r1' depart='15'/>
    <vehicle id='passenger14' type='passenger' route='r2' depart='15'/>
    <vehicle id='passenger15' type='passenger' route='r3' depart='20'/>
    <vehicle id='passenger16' type='passenger' route='r4' depart='20'/>
    <vehicle id='passenger17' type='passenger' route='r5' depart='20'/>
    <vehicle id='passenger18' type='passenger' route='r6' depart='20'/>
    <vehicle id='passenger19' type='passenger' route='r7' depart='20'/>
    <vehicle id='passenger/van20' type='passenger/van' route='r8' depart='25'/>
    <vehicle id='passenger/van21' type='passenger/van' route='r9' depart='25'/>
    <vehicle id='passenger/van22' type='passenger/van' route='r10' depart='25'/>
    <vehicle id='passenger/van23' type='passenger/van' route='r11' depart='25'/>
    <vehicle id='passenger/van24' type='passenger/van' route='r0' depart='25'/>
    <vehicle id='passenger/van25' type='passenger/van' route='r1' depart='30'/>
    <vehicle id='passenger/van26' type='passenger/van' route='r2' depart='30'/>
    <vehicle id='passenger/van27' type='passenger/van' route='r3' depart='30'/>
    <vehicle id='passenger/van28' type='passenger/van' route='r4' depart='30'/>
    <vehicle id='passenger/van29' type='passenger/van' route='r5' depart='30'/>
    <vehicle id='truck30' type='truck' route='r6' depart='35'/>
    <vehicle id='truck31' type='truck' route='r7' depart='35'/>
    <vehicle id='truck32' type='truck' route='r8' depart='35'/>
    <vehicle id='truck33' type='truck' route='r9' depart='35'/>
    <vehicle id='truck34' type='truck' route='r10' depart='35'/>
    <vehicle id='truck35' type='truck' route='r11' depart='40'/>
    <vehicle id='truck36' type='truck' route='r0' depart='40'/>
    <vehicle id='truck37' type='truck' route='r1' depart='40'/>
    <vehicle id='truck38' type='truck' route='r2' depart='40'/>
    <vehicle id='truck39' type='truck' route='r3' depart='40'/>
    <vehicle id='bus40' type='bus' route='r4' depart='45'/>
    <vehicle id='bus41' type='bus' route='r5' depart='45'/>
    <vehicle id='bus42' type='bus' route='r6' depart='45'/>
    <vehicle id='bus43' type='bus' route='r7' depart='45'/>
    <vehicle id='bus44' type='bus' route='r8' depart='45'/>
    <vehicle id='bus45' type='bus' route='r9' depart='50'/>
    <vehicle id='bus46' type='bus' route='r10' depart='50'/>
    <vehicle id='bus47' type='bus' route='r11' depart='50'/>
    <vehicle id='bus48' type='bus' route='r0' depart='50'/>
    <vehicle id='bus49' type='bus' route='r1' depart='50'/>
    <vehicle id='bicycle50' type='bicycle' route='r2' depart='55'/>
    <vehicle id='bicycle51' type='bicycle' route='r3' depart='55'/>
    <vehicle id='bicycle52' type='bicycle' route='r4' depart='55'/>
    <vehicle id='bicycle53' type='bicycle' route='r5' depart='55'/>
    <vehicle id='bicycle54' type='bicycle' route='r6' depart='55'/>
    <vehicle id='bicycle55' type='bicycle' route='r7' depart='60'/>
    <vehicle id='bicycle56' type='bicycle' route='r8' depart='60'/>
    <vehicle id='bicycle57' type='bicycle' route='r9' depart='60'/>
    <vehicle id='bicycle58' type='bicycle' route='r10' depart='60'/>
    <vehicle id='bicycle59' type='bicycle' route='r11' depart='60'/>
    <vehicle id='motorcycle60' type='motorcycle' route='r0' depart='65'/>
    <vehicle id='motorcycle61' type='motorcycle' route='r1' depart='65'/>
    <vehicle id='motorcycle62' type='motorcycle' route='r2' depart='65'/>
    <vehicle id='motorcycle63' type='motorcycle' route='r3' depart='65'/>
    <vehicle id='motorcycle64' type='motorcycle' route='r4' depart='65'/>
    <vehicle id='motorcycle65' type='motorcycle' route='r5' depart='70'/>
    <vehicle id='motorcycle66' type='motorcycle' route='r6' depart='70'/>
    <vehicle id='motorcycle67' type='motorcycle' route='r7' depart='70'/>
    <vehicle id='motorcycle68' type='motorcycle' route='r8' depart='70'/>
    <vehicle id='motorcycle69' type='motorcycle' route='r9' depart='70'/>
    <vehicle id='passenger70' type='passenger' route='r10' depart='75'/>
    <vehicle id='passenger71' type='passenger' route='r11' depart='75'/>
    <vehicle id='passenger72' type='passenger' route='r0' depart='75'/>
    <vehicle id='passenger73' type='passenger' route='r1' depart='75'/>
    <vehicle id='passenger74' type='passenger' route='r2' depart='75'/>
    <vehicle id='passenger75' type='passenger' route='r3' depart='80'/>
    <vehicle id='passenger76' type='passenger' route='r4' depart='80'/>
    <vehicle id='passenger77' type='passenger' route='r5' depart='80'/>
    <vehicle id='passenger78' type='passenger' route='r6' depart='80'/>
    <vehicle id='passenger79' type='passenger' route='r7' depart='80'/>
    <vehicle id='passenger/van80' type='passenger/van' route='r8' depart='85'/>
    <vehicle id='passenger/van81' type='passenger/van' route='r9' depart='85'/>
    <vehicle id='passenger/van82' type='passenger/van' route='r10' depart='85'/>
    <vehicle id='passenger/van83' type='passenger/van' route='r11' depart='85'/>
    <vehicle id='passenger/van84' type='passenger/van' route='r0' depart='85'/>
    <vehicle id='passenger/van85' type='passenger/van' route='r1' depart='90'/>
    <vehicle id='passenger/van86' type='passenger/van' route='r2' depart='90'/>
    <vehicle id='passenger/van87' type='passenger/van' route='r3' depart='90'/>
    <vehicle id='passenger/van88' type='passenger/van' route='r4' depart='90'/>
    <vehicle id='passenger/van89' type='passenger/van' route='r5' depart='90'/>
    <vehicle id='truck90' type='truck' route='r6' depart='95'/>
    <vehicle id='truck91' type='truck' route='r7' depart='95'/>
    <vehicle id='truck92' type='truck' route='r8' depart='95'/>
    <vehicle id='truck93' type='truck' route='r9' depart='95'/>
    <vehicle id='truck94' type='truck' route='r10' depart='95'/>
    <vehicle id='truck95' type='truck' route='r11' depart='100'/>
    <vehicle id='truck96' type='truck' route='r0' depart='100'/>
    <vehicle id='truck97' type='truck' route='r1' depart='100'/>
    <vehicle id='truck98' type='truck' route='r2' depart='100'/>
    <vehicle id='truck99' type='truck' route='r3' depart='100'/>
    <vehicle id='bus100' type='bus' route='r4' depart='105'/>
    <vehicle id='bus101' type='bus' route='r5' depart='105'/>
    <vehicle id='bus102' type='bus' route='r6' depart='105'/>
    <vehicle id='bus103' type='bus' route='r7' depart='105'/>
    <vehicle id='bus104' type='bus' route='r8' depart='105'/>
    <vehicle id='bus105' type='bus' route='r9' depart='110'/>
    <vehicle id='bus106' type='bus' route='r10' depart='110'/>
    <vehicle id='bus107' type='bus' route='r11' depart='110'/>
    <vehicle id='bus108' type='bus' route='r0' depart='110'/>
    <vehicle id='bus109' type='bus' route='r1' depart='110'/>
    <vehicle id='bicycle110' type='bicycle' route='r2' depart='115'/>
    <vehicle id='bicycle111' type='bicycle' route='r3' depart='115'/>
    <vehicle id='bicycle112' type='bicycle' route='r4' depart='115'/>
    <vehicle id='bicycle113' type='bicycle' route='r5' depart='115'/>
    <vehicle id='bicycle114' type='bicycle' route='r6' depart='115'/>
    <vehicle id='bicycle115' type='bicycle' route='r7' depart='120'/>
    <vehicle id='bicycle116' type='bicycle' route='r8' depart='120'/>
    <vehicle id='bicycle117' type='bicycle' route='r9' depart='120'/>
    <vehicle id='bicycle118' type='bicycle' route='r10' depart='120'/>
    <vehicle id='bicycle119' type='bicycle' route='r11' depart='120'/>
    <vehicle id='motorcycle120' type='motorcycle' route='r0' depart='125'/>
    <vehicle id='motorcycle121' type='motorcycle' route='r1' depart='125'/>
    <vehicle id='motorcycle122' type='motorcycle' route='r2' depart='125'/>
    <vehicle id='motorcycle123' type='motorcycle' route='r3' depart='125'/>
    <vehicle id='motorcycle124' type='motorcycle' route='r4' depart='125'/>
    <vehicle id='motorcycle125' type='motorcycle' route='r5' depart='130'/>
    <vehicle id='motorcycle126' type='motorcycle' route='r6' depart='130'/>
    <vehicle id='motorcycle127' type='motorcycle' route='r7' depart='130'/>
    <vehicle id='motorcycle128' type='motorcycle' route='r8' depart='130'/>
    <vehicle id='motorcycle129' type='motorcycle' route='r9' depart='130'/>
    <vehicle id='passenger130' type='passenger' route='r10' depart='135'/>
    <vehicle id='passenger131' type='passenger' route='r11' depart='135'/>
    <vehicle id='passenger132' type='passenger' route='r0' depart='135'/>
    <vehicle id='passenger133' type='passenger' route='r1' depart='135'/>
    <vehicle id='passenger134' type='passenger' route='r2' depart='135'/>
    <vehicle id='passenger135' type='passenger' route='r3' depart='140'/>
    <vehicle id='passenger136' type='passenger' route='r4' depart='140'/>
    <vehicle id='passenger137' type='passenger' route='r5' depart='140'/>
    <vehicle id='passenger138' type='passenger' route='r6' depart='140'/>
    <vehicle id='passenger139' type='passenger' route='r7' depart='140'/>
    <vehicle id='passenger/van140' type='passenger/van' route='r8' depart='145'/>
    <vehicle id='passenger/van141' type='passenger/van' route='r9' depart='145'/>
    <vehicle id='passenger/van142' type='passenger/van' route='r10' depart='145'/>
    <vehicle id='passenger/van143' type='passenger/van' route='r11' depart='145'/>
    <vehicle id='passenger/van144' type='passenger/van' route='r0' depart='145'/>
    <vehicle id='passenger/van145' type='passenger/van' route='r1' depart='150'/>
    <vehicle id='passenger/van146' type='passenger/van' route='r2' depart='150'/>
    <vehicle id='passenger/van147' type='passenger/van' route='r3' depart='150'/>
    <vehicle id='passenger/van148' type='passenger/van' route='r4' depart='150'/>
    <vehicle id='passenger/van149' type='passenger/van' route='r5' depart='150'/>
    <vehicle id='truck150' type='truck' route='r6' depart='155'/>
    <vehicle id='truck151' type='truck' route='r7' depart='155'/>
    <vehicle id='truck152' type='truck' route='r8' depart='155'/>
    <vehicle id='truck153' type='truck' route='r9' depart='155'/>
    <vehicle id='truck154' type='truck' route='r10' depart='155'/>
    <vehicle id='truck155' type='truck' route='r11' depart='160'/>
    <vehicle id='truck156' type='truck' route='r0' depart='160'/>
    <vehicle id='truck157' type='truck' route='r1' depart='160'/>
    <vehicle id='truck158' type='truck' route='r2' depart='160'/>
    <vehicle id='truck159' type='truck' route='r3' depart='160'/>
    <vehicle id='bus160' type='bus' route='r4' depart='165'/>
    <vehicle id='bus161' type='bus' route='r5' depart='165'/>
    <vehicle id='bus162' type='bus' route='r6' depart='165'/>
    <vehicle id='bus163' type='bus' route='r7' depart='165'/>
    <vehicle id='bus164' type='bus' route='r8' depart='165'/>
    <vehicle id='bus165' type='bus' route='r9' depart='170'/>
    <vehicle id='bus166' type='bus' route='r10' depart='170'/>
    <vehicle id='bus167' type='bus' route='r11' depart='170'/>
    <vehicle id='bus168' type='bus' route='r0' depart='170'/>
    <vehicle id='bus169' type='bus' route='r1' depart='170'/>
    <vehicle id='bicycle170' type='bicycle' route='r2' depart='175'/>
    <vehicle id='bicycle171' type='bicycle' route='r3' depart='175'/>
    <vehicle id='bicycle172' type='bicycle' route='r4' depart='175'/>
    <vehicle id='bicycle173' type='bicycle' route='r5' depart='175'/>
    <vehicle id='bicycle174' type='bicycle' route='r6' depart='175'/>
    <vehicle id='bicycle175' type='bicycle' route='r7' depart='180'/>
    <vehicle id='bicycle176' type='bicycle' route='r8' depart='180'/>
    <vehicle id='bicycle177' type='bicycle' route='r9' depart='180'/>
    <vehicle id='bicycle178' type='bicycle' route='r10' depart='180'/>
    <vehicle id='bicycle179' type='bicycle' route='r11' depart='180'/>
    <vehicle id='motorcycle180' type='motorcycle' route='r0' depart='185'/>
    <vehicle id='motorcycle181' type='motorcycle' route='r1' depart='185'/>
    <vehicle id='motorcycle182' type='motorcycle' route='r2' depart='185'/>
    <vehicle id='motorcycle183' type='motorcycle' route='r3' depart='185'/>
    <vehicle id='motorcycle184' type='motorcycle' route='r4' depart='185'/>
    <vehicle id='motorcycle185' type='motorcycle' route='r5' depart='190'/>
    <vehicle id='motorcycle186' type='motorcycle' route='r6' depart='190'/>
    <vehicle id='motorcycle187' type='motorcycle' route='r7' depart='190'/>
    <vehicle id='motorcycle188' type='motorcycle' route='r8' depart='190'/>
    <vehicle id='motorcycle189' type='motorcycle' route='r9' depart='190'/>
    <vehicle id='passenger190' type='passenger' route='r10' depart='195'/>
    <vehicle id='passenger191' type='passenger' route='r11' depart='195'/>
    <vehicle id='passenger192' type='passenger' route='r0' depart='195'/>
    <vehicle id='passenger193' type='passenger' route='r1' depart='195'/>
    <vehicle id='passenger194' type='passenger' route='r2' depart='195'/>
    <vehicle id='passenger195' type='passenger' route='r3' depart='200'/>
    <vehicle id='passenger196' type='passenger' route='r4' depart='200'/>
    <vehicle id='passenger197' type='passenger' route='r5' depart='200'/>
    <vehicle id='passenger198' type='passenger' route='r6' depart='200'/>
    <vehicle id='passenger199' type='passenger' route='r7' depart='200'/>
</routes>


""", file=routes)
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


def getPhaseState(transition_time):
    num_lanes = 4
    num_phases = 4
    phase = traci.trafficlight.getPhase("0")
    phaseState = np.zeros((transition_time,num_lanes,num_phases))
    for i in range(transition_time):
        for j in range(num_lanes):
            phaseState[i][j][phase] = 1
    return phaseState


def getState(transition_time):  # made the order changes
    newState = []
    avg_qlength = 0
    # transition_time_step_leftcount = 0
    # transition_time_step_rightcount = 0
    # transition_time_step_topcount = 0
    # transition_time_step_bottomcount = 0
    avg_leftcount = 0
    avg_rightcount = 0
    avg_bottomcount = 0
    avg_topcount = 0
    for _ in range(transition_time):
        traci.simulationStep()




        leftcount = 0
        rightcount = 0
        topcount = 0
        bottomcount = 0
        vehicleList = traci.vehicle.getIDList()

        print("Traffic : ")

        for id in vehicleList:
            x, y = traci.vehicle.getPosition(id)

            if x<110 and x>60 and y<130 and y>120:
                leftcount+=1
            else :
                if x<120 and x>110 and y<110 and y>600:
                    bottomcount+=1
                else :
                    if x<180 and x>130 and y<120 and y>110:
                        rightcount+=1
                    else :
                        if x<130 and x>120 and y<180 and y>130:
                            topcount+=1

        print("Left : ", leftcount)
        print("Right : ", rightcount)
        print("Top : ", topcount)
        print("Bottom : ", bottomcount)

        avg_topcount += topcount
        avg_bottomcount += bottomcount
        avg_leftcount += leftcount
        avg_rightcount += rightcount

        # transition_time_step_bottomcount+= bottomcount
        # transition_time_step_leftcount+= leftcount
        # transition_time_step_rightcount+= rightcount
        # transition_time_step_topcount+= topcount

        state = [bottomcount / 40,
                 rightcount / 40,
                 topcount / 40,
                 leftcount / 40
                 ]

        avg_qlength += ((bottomcount + rightcount + topcount + leftcount)/4)
        newState.insert(0, state)
    # print (state)

    # df = pd.DataFrame([[, 2]], columns=['a', 'b'])
    # params_dict =
    avg_qlength /= transition_time
    avg_leftcount /= transition_time
    avg_topcount /= transition_time
    avg_rightcount /= transition_time
    avg_bottomcount /= transition_time

    avg_lane_qlength = [avg_leftcount, avg_topcount, avg_rightcount, avg_bottomcount]
    newState = np.array(newState)
    phaseState = getPhaseState(transition_time)
    newState = np.dstack((newState, phaseState))
    newState = np.expand_dims(newState, axis=0)
    return newState, avg_qlength, avg_lane_qlength


print("here")
import traci


def makeMove(action, transition_time):
    if action == 1:
        traci.trafficlight.setPhase("0", (int(traci.trafficlight.getPhase("0")) + 1) % 4)




    # traci.simulationStep()
    # traci.simulationStep()
    # traci.simulationStep()
    # traci.simulationStep()

    return getState(transition_time)


def getReward(this_state, this_new_state):
    num_lanes = 4
    qLengths1 = []
    qLengths2 = []
    for i in range(num_lanes):
        qLengths1.append(this_state[0][0][i][0])
        qLengths2.append(this_new_state[0][0][i][0])

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

def getRewardAbsolute(this_state, this_new_state):
    num_lanes = 4
    qLengths1 = []
    qLengths2 = []
    for i in range(num_lanes):
        qLengths1.append(this_state[0][0][i][0])
        qLengths2.append(this_new_state[0][0][i][0])

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


def build_model(transition_time):
    num_hidden_units_cnn = 10
    num_actions = 2
    model = Sequential()
    model.add(Conv2D(num_hidden_units_cnn, kernel_size=(transition_time, 1), strides=1, activation='relu', input_shape=(transition_time, 4,5)))
    # model.add(LSTM(8))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    opt = RMSprop(lr=0.00025)
    model.compile(loss='mse', optimizer=opt)

    return model


def getWaitingTime(laneID):
    return traci.lane.getWaitingTime(laneID)


num_episode = 1
discount_factor = 0.9
#epsilon = 1
epsilon_start = 1
epsilon_end = 0.01
epsilon_decay_steps = 3000

Average_Q_lengths = []

params_dict = [] #for graph writing
sum_q_lens = 0
AVG_Q_len_perepisode = []

transition_time = 8
target_update_time = 20
q_estimator_model = load_model("models/single intersection models/tradeoff_models_absreward/model_15.h5")
replay_memory_init_size = 150
replay_memory_size = 8000
batch_size = 32
print(q_estimator_model.summary())
epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

#generate_routefile_random(episode_time, num_vehicles)
#generate_routefile(290,10)
traci.start([sumoBinary, "-c", "data/cross.sumocfg",
             "--tripinfo-output", "tripinfo.xml"])

traci.trafficlight.setPhase("0", 0)

nA = 2






total_t = 0
for episode in range(num_episode):

    traci.load(["--start", "-c", "data/cross.sumocfg",
                "--tripinfo-output", "tripinfo.xml"])
    traci.trafficlight.setPhase("0", 0)

    state, _, _ = getState(transition_time)
    counter = 0
    stride = 0

    length_data_avg = []
    count_data = []
    delay_data_avg = []
    delay_data_min = []
    delay_data_max = []
    delay_data_time = []
    current_left_time = 0
    current_top_time = 0
    current_bottom_time = 0
    current_right_time = 0
    overall_lane_qlength = [0, 0, 0, 0]
    num_cycles = 0
    num_qlength_instances = 0


    while traci.simulation.getMinExpectedNumber() > 0:
        print("Episode # ", episode)
        # print("Waiting time on lane 1i_0 = ",getWaitingTime("1i_0"))

        print("Inside episode counter", counter)

        counter += 1
        total_t += 1
        # batch_experience = experience[:batch_history]
        prev_phase = traci.trafficlight.getPhase("0")

        action = np.argmax(q_estimator_model.predict(state))

        new_state, qlength, avg_lane_qlength = makeMove(action, transition_time)
        new_phase = traci.trafficlight.getPhase("0")
        print("Previous phase = ", prev_phase)
        print("New phase = ", new_phase)
        vehicleList = traci.vehicle.getIDList()
        num_vehicles = len(vehicleList)
        print("Number of cycles = ", num_cycles)
        if num_vehicles:
            avg = 0
            max = 0
            mini = 100
            for id in vehicleList:
                time = traci.vehicle.getAccumulatedWaitingTime(id)
                if time > max:
                    max = time

                if time < mini:
                    mini = time

                avg += time
            avg /= num_vehicles
            delay_data_avg.append(avg)
            delay_data_max.append(max)
            delay_data_min.append(mini)
            length_data_avg.append(qlength)
            count_data.append(num_vehicles)
            delay_data_time.append(traci.simulation.getCurrentTime() / 1000)

            if traci.simulation.getCurrentTime() / 1000 < 2100:
                overall_lane_qlength = list(map(add, overall_lane_qlength, avg_lane_qlength))
                num_qlength_instances += 1
                if prev_phase == 3 and new_phase == 0:
                    num_cycles += 1
                if prev_phase == 0:
                    current_bottom_time += transition_time
                if prev_phase == 1:
                    current_right_time += transition_time
                if prev_phase == 2:
                    current_top_time += transition_time
                if prev_phase == 3:
                    current_left_time += transition_time



        state = new_state

    overall_lane_qlength[:] = [x / num_qlength_instances for x in overall_lane_qlength]
    current_right_time /= num_cycles
    current_top_time /= num_cycles
    current_left_time /= num_cycles
    current_bottom_time /= num_cycles
    avg_free_time = [current_left_time, current_top_time, current_right_time, current_bottom_time]

    plt.plot(delay_data_time, delay_data_avg, 'b-', label='avg')
    #plt.plot(delay_data_time, delay_data_min, 'g-', label='min')
    #plt.plot(delay_data_time, delay_data_max,'r-', label='max')
    plt.legend(loc='upper left')
    plt.ylabel('Waiting time per minute')
    plt.xlabel('Time in simulation (in s)')

    plt.figure()
    plt.plot(delay_data_time, length_data_avg, 'b-', label='avg')
    plt.legend(loc='upper left')
    plt.ylabel('Average Queue Length')
    plt.xlabel('Time in simulation (in s)')

    plt.figure()
    plt.plot(delay_data_time, count_data, 'b-', label='avg')
    plt.legend(loc='upper left')
    plt.ylabel('Average Number of Vehicles in Map')
    plt.xlabel('Time in simulation (in s)')

    plt.figure()
    label = ['Obstacle Lane', 'Top Lane w/ traffic', 'Right lane', 'Bottom lane']
    index = np.arange(len(label))
    plt.bar(index, avg_free_time, color=['red', 'green', 'blue', 'blue'])
    plt.xlabel('Lane')
    plt.ylabel('Average Green Time per Cycle')
    plt.xticks(index, label)

    plt.figure()
    label = ['Obstacle Lane', 'Top Lane w/ traffic', 'Right lane', 'Bottom lane']
    index = np.arange(len(label))
    plt.bar(index, overall_lane_qlength, color=['red', 'green', 'blue', 'blue'])
    plt.xlabel('Lane')
    plt.ylabel('Average Q-length every 8 seconds')
    plt.xticks(index, label)
    plt.show()

    AVG_Q_len_perepisode.append(sum_q_lens / 702)
    sum_q_lens = 0





