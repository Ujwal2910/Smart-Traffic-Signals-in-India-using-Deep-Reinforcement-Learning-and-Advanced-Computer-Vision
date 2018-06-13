#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2017 German Aerospace Center (DLR) and others.
# This program and the accompanying materials
# are made available under the terms of the Eclipse Public License v2.0
# which accompanies this distribution, and is available at
# http://www.eclipse.org/legal/epl-v20.html

# @file    runner.py
# @author  Lena Kalleske
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @date    2009-03-26
# @version $Id$

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

stdscr = curses.initscr()
curses.cbreak()
stdscr.keypad(1)
stdscr.refresh()
key = ''

# we need to import python modules from the $SUMO_HOME/tools directory
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
    
    <flow id="mixed1" begin="0" end="1500" number="100" route="r0" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed2" begin="0" end="1500" number="100" route="r1" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed3" begin="0" end="1500" number="100" route="r2" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed4" begin="0" end="1500" number="100" route="r3" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed5" begin="0" end="1500" number="100" route="r4" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed6" begin="0" end="1500" number="100" route="r5" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed7" begin="0" end="1500" number="100" route="r6" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed8" begin="0" end="1500" number="100" route="r7" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed9" begin="0" end="1500" number="100" route="r8" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed10" begin="0" end="1500" number="100" route="r9" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed11" begin="0" end="1500" number="100" route="r10" type="mixed" departLane="random" departPosLat="random"/>
    <flow id="mixed12" begin="0" end="1500" number="100" route="r11" type="mixed" departLane="random" departPosLat="random"/>
</routes>""", file=routes)
        lastVeh = 0
        vehNr = 0
    ''' for i in range(N):
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
                lastVeh = i
        print("</routes>", file=routes)'''
    

# The program looks like this
#    <tlLogic id="0" type="static" programID="0" offset="0">
# the locations of the tls are      NESW
#        <phase duration="31" state="GrGr"/>
#        <phase duration="6"  state="yryr"/>
#        <phase duration="31" state="rGrG"/>
#        <phase duration="6"  state="ryry"/>
#    </tlLogic>


def run():
    """execute the TraCI control loop"""
    step = 0
    
    traci.trafficlight.setPhase("0", 0)
    while traci.simulation.getMinExpectedNumber() > 0:
        
        traci.simulationStep()
        # timeout = 0.2
        # print("Time right now - ",step)
        # rlist, wlist, xlist = select([sys.stdin],[],[],timeout)

        # if rlist:
        #     print("Key pressed - ")
        #     print(rlist)
        #     traci.vehicle.addFull(vehID='left_'+str(step),routeID='r0',typeID='car',depart='triggered',departLane='random',departPos='random')
        #     termios.tcflush(sys.stdin,termios.TCIFLUSH)

        key = stdscr.getch()
        stdscr.addch(20,25,key)
        stdscr.refresh()

        if key == curses.KEY_RIGHT: 
            stdscr.addstr(2, 20, "Up")
            traci.vehicle.addFull(vehID='right_'+str(step),routeID='r0',typeID='car',depart='triggered',departLane='random',departPos='random')

        elif key == curses.KEY_DOWN: 
            stdscr.addstr(3, 20, "Down")
            traci.vehicle.addFull(vehID='down_'+str(step),routeID='r3',typeID='car',depart='triggered',departLane='random',departPos='random')

        elif key == curses.KEY_LEFT: 
            stdscr.addstr(4, 20, "Left")
            traci.vehicle.addFull(vehID='left_'+str(step),routeID='r6',typeID='car',depart='triggered',departLane='random',departPos='random')

        elif key == curses.KEY_UP: 
            stdscr.addstr(5, 20, "Up")
            traci.vehicle.addFull(vehID='up_'+str(step),routeID='r9',typeID='car',depart='triggered',departLane='random',departPos='random')

        step += 1
    curses.endwin()
    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

def take_action():
    traci.trafficlight.setPhase("0", traci.trafficlight.getPhase("0") + 1)


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    generate_routefile()

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
    run()
