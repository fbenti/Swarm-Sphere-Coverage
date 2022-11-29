#!/usr/bin/env python

# External
from sympy import false
import numpy as np
from math import cos,sin,radians,asin,tanh
import copy
import time
import sys
import rospy
import os
import os.path


# Internal
from pycrazyswarm import Crazyswarm
from crazyswarm.msg import GenericLogData
from SwarmControl.log import log, plotLog



""" Flight parameters """
HOVER_HEIGHT = 0.5  
LAND_HEIGHT = 0.02
TAKEOFF_SLEEP = 1
LAND_SLEEP = 2

# from cs_helper import *


""" Control parameters """
# rot, scaling_x, scaling_y, t_x, t_y
ETA_DES = np.array([radians(45), 1, 1, 2, 2])
COMM_RANGE = 3
EPS = 0.5

class SwarmControl():
    def __init__(self, swarm, timeHelper, record_log = True):

        # Crazyswarm class
        self.swarm = swarm
        self.timeHelper = timeHelper
        self.record_log = record_log

        # log 
        if self.record_log:
            self._logFolder()
            self.logs = {}
            self.init_pose_log = self.create_log('init_pose_log',folder=self.folder)
            self.pose_log = self.create_log('pose_log',folder=self.folder)
            self.des_pose_log = self.create_log('des_pose_log',folder=self.folder)
            self.des_eta_log = self.create_log('des_eta_log',folder=self.folder)
            self.goal_log = self.create_log('goal_log',folder=self.folder)
            self.eta_log = self.create_log('eta_log',folder=self.folder)
            self.deta_log = self.create_log('deta_log',folder=self.folder)
            self.gamma_log = self.create_log('gamma_log',folder=self.folder)
            self.dgamma_log = self.create_log('dgamma_log',folder=self.folder)

        # Battery chack
        # self._batteryCheck()

        # Swarm Configuration
        self.cx, self.cy = np.meshgrid(np.arange(-1,2),np.arange(-1,2))
        self.configDict, self.c = self._getInitConfig()
        self.currPos = copy.copy(self.configDict)

        # SwarmControl Parameters
        self.M = self.c.shape[1]
        self._initialiseParam()
        
        # Subscribe to posEstimete topic, for each copter
        self._updateAllPos()

        # Neighbors
        self.A = np.array([
            [0,1,0],
            [1,0,1],
            [0,1,0]])
        # self.A = np.array([
        #     [0,1],
        #     [1,0]]
        #     )

    def _logFolder(self,test_type="FlowDeck"):
        root = './SwarmControl/log_' + test_type + '/'
        if not os.path.isdir(root):
            os.mkdir(root)

        root = root + 'log'
        i = 0
        folder = root + str(i) + "/"
        while os.path.isdir(folder):
            i += 1
            folder = root + str(i) + "/"
        os.mkdir(folder)
        
        self.folder = folder


    def _batteryCheck(self):
        """ Chack that level of battery is above 3.7 """
        for cf in self.swarm.crazyflies:
            topic_name = cf.prefix + '/battStatus'
            msg = rospy.wait_for_message(topic_name, GenericLogData, timeout=2)
            voltage = msg.values[0]
            # print("Battery-{}: {}".format(cf.id, voltage))
            if voltage < 3.7:
                print("[ERR] Copter-{} cannot fly -> Exiting".format(cf.id))
                sys.exit()
        time.sleep(2)


    def _getInitConfig(self):
        """ Generate dict and list containing intial configuration """
        crazyfliesDict = dict()
        crazyfliesArr = []

        for cf in self.swarm.crazyflies:
            # append yaw: set at 0
            arr = np.concatenate((cf.initialPosition,[0]),axis=0)
            crazyfliesDict[cf.id] = arr
            crazyfliesArr.append(cf.initialPosition[:2])
        
        if self.record_log:
            goal = []
            for xy in crazyfliesArr:
                for p in xy:
                    goal.append(p)
            self.init_pose_log.log(goal)
        return crazyfliesDict, np.array(crazyfliesArr).T


    def _updateAllPos(self):
        """ Create subscriber to posEst topic for each copter """
        for j,cf in enumerate(self.swarm.crazyflies):
            topic_name = cf.prefix + '/posEst'
            try:
                msg = rospy.wait_for_message(topic_name, GenericLogData, timeout=0.2)
                self.currPos[cf.id] = self.configDict[cf.id] + np.array(msg.values)
            except Exception as e:
                print("[ERR]-line141:", e)
                continue
    

    def _updateSinglePos(self, id:int, idx:int):
        """ Create subscriber to posEst topic for single copter """
        cf = self.swarm.crazyfliesById[id]
        topic_name = cf.prefix + '/posEst'
        msg = None
        try:
            msg = rospy.wait_for_message(topic_name, GenericLogData, timeout=0.2)
            self.currPos[cf.id] = self.configDict[cf.id] + np.array(msg.values)
        except Exception as e:
            print("[ERR]-line54:", e)
            return


    def _initialiseParam(self):
        """
        Initialise parameter:
            - phi : desired rotation
            - s : desired scaling
            - t : desired translation
            - R : rotation matrix
            - s : scaling matrix
            - p_des : homogeneous transformation matrix 
        """
        global ETA_DES,COMM_RANGE,EPS
        self.eta_des = ETA_DES
        self.phi = self.eta_des[0]
        self.s = self.eta_des[1:3]
        self.t = self.eta_des[3:].reshape((2,1))
        self.R = np.array([[cos(self.phi), -sin(self.phi)],[sin(self.phi),cos(self.phi)]])
        self.S = np.diag(self.s)
        self.sigma = 0.0
        self.p_des = self.R @ self.S @ self.c + self.t

        print("config", self.configDict)
        print("DEST:",self.p_des)

        # print("INIT", self.currPos)
        # print("DEST", self.p_des)

        self.eta = np.zeros((5,self.M)) # parameter
        self.deta = np.zeros((5,self.M)) # parameter derivate
        self.gamma = np.zeros(self.M) # penalty multiplier
        self.dgamma = np.zeros(self.M) # penalty multiplier derivative
        self.eta[:,:] = np.array([0, 1, 1, 0, 0]).reshape((5,1))

        self.J = np.zeros((2,5,self.M)) # Jacobian
        self.p = np.zeros((2,self.M)) # robot position
        
        self.cd = np.linalg.norm([COMM_RANGE,COMM_RANGE],2) # Communication range TODO to set correctly
        self.epsilon = EPS # collision clearance distance
        self.rmax = self.cd
        phi = asin(self.epsilon/self.rmax)
        self.delta = self.rmax*cos(phi)
        
        if self.record_log:
            p_des = []
            for i in range(self.p_des.shape[1]):
                p_des.append(self.p_des[0,i])
                p_des.append(self.p_des[1,i])
            self.des_pose_log.log(p_des)
            
            a,b,c,d,e = ETA_DES
            eta_des = []
            eta_des.append(a)
            eta_des.append(b)
            eta_des.append(c)
            eta_des.append(d)
            eta_des.append(e)
            self.des_eta_log.log(eta_des)

              
    def allErrBelowThresh(self, thresh=0.2):
        """
        check that all the sqrt err are below thresh 
        """
        for j,id in enumerate(self.currPos.keys()):
            # USE: position feedback instead of self.p
            currPos = self.currPos[id]
            err = np.square(np.linalg.norm(self.p_des[:,j] - currPos[:2], ord=2, axis=0))
            yawErr = self.phi - currPos[id][3] 
            # TODO change
            if err < thresh or abs(yawErr) < 0.1:
                return True
        return False


    def singleErrBelowThresh(self, idx, id : int, thresh=0.2):
        """
        Check if the sqrt err is below thresh 
        """
        self._updateSinglePos(id=id, idx=idx)
        currPos = self.currPos[id]
        err = np.square(np.linalg.norm(self.p_des[:,idx] - currPos[:2], ord=2, axis=0))
        yawErr = self.phi - currPos[3]
        if err < thresh and abs(yawErr) < 0.1:
            return True
        return False


    def _goalReached(self, goal, id, yaw, thresh=0.2):
        self._updateSinglePos(id=id, idx=0)
        currPos = self.currPos[id]
        err = np.square(np.linalg.norm(goal[:2] - currPos[:2], ord=2, axis=0))
        print("Dist2goal:",err)
        if err < thresh:
            return True
        return False
        # yawErr = self.phi - currPos[3]

    def Jack(self,p,c):
        phi = p[0]
        s = p[1:3]
        # t = p[3:]
        J = np.array([
            [-sin(phi)*s[0]*c[0] - cos(phi)*s[1]*c[1], cos(phi)*c[0], -sin(phi)*c[1], 1, 0],
            [cos(phi)*s[0]*c[0] - sin(phi)*s[1]*c[1], sin(phi)*c[0],  cos(phi)*c[1], 0, 1]])
        return J


    def goToInitPos(self):
        # timer = time.time()
        # while timer + 5 > time.time():
        #     for j,id in enumerate(self.configDict.keys()):
        #         dest = self.c[:,j].tolist()
        #         dest.append(HOVER_HEIGHT)
        #         self.swarm.crazyfliesById[id]
                # self.swarm.crazyfliesById[id].cmdPosition(pos=dest,yaw=0)
            # self.timeHelper.sleepForRate(30)
        for j,id in enumerate(self.configDict.keys()):
            self.swarm.crazyfliesById[id].goTo(goal=np.array([0,0,HOVER_HEIGHT]),yaw=0,duration=4)
        self.timeHelper.sleep(5)     

    def controlLoop(self):
        loop_t = time.time()
        # while self.allErrAboveThresh(thresh=0.3) and start + 5 > time.time():
        while loop_t + 20 > time.time():
            for j,id in enumerate(self.configDict.keys()):
                start = time.time() # for integral calculation

                # Jacobian
                J = self.Jack(self.eta[:,j],self.c[:,j])
                
                # update robot position
                phi = self.eta[0,j]
                s = self.eta[1:3,j]
                t = self.eta[3:,j].reshape(2)
                R = np.array([[cos(phi),-sin(phi)],[sin(phi),cos(phi)]])
                S = np.diag(s)
                self.p[:,j] = (R @ S @ (self.c[:,j]) + t.T).reshape(2)

                # # TODO: send pos/vel command
                if self.singleErrBelowThresh(idx=j, id=id):
                    print("<< {} reached ".format(id, self.currPos[id]))
                    self.swarm.crazyfliesById[id].cmdVelocityWorld([0,0,0],0)
                else:
                    dest = self.p[:,j].tolist()
                    dest.append(HOVER_HEIGHT)
                    print(">> sending {} to {} {}".format(id, dest, phi))
                    self.swarm.crazyfliesById[id].cmdPosition(pos=dest,yaw=phi)


                # ERROR CORRECTION STEP
                correction = (self.p_des[:,j] - self.currPos[id][:2]) #self.p[:,j]) #self.currPos[id][:2]) # TODO: or use p[:,j]
                print("correction {} : {}".format(id,correction))
                dp_des = 5 * tanh(np.linalg.norm(correction,2))*correction/(np.linalg.norm(correction,2)+0.1)
                err1 = np.linalg.pinv(J) @ dp_des

                
                # CONSENSUS STEP
                err2 = np.zeros(5)
                for i in range(self.A.shape[1]):
                    if self.A[j,i] == 1:
                        # err2 = np.sum(np.multiply(Nj, self.eta - eta[:,i,j].reshape(5,1)),axis=2)
                        # print(self.eta[:,i])
                        eta_i = np.array(self.eta[:,i]).reshape(5)
                        err2 += eta_i - self.eta[:,j]

                
                # CONSTRAINT SATISFACTION STEP
                cond = [s[0] > self.epsilon, s[1] > self.epsilon, np.linalg.norm(s,2) > self.rmax]
                if cond == [False,False,False]:
                    s_proj = [self.epsilon, self.epsilon] # A
                elif cond == [True,False,False]:
                    s_proj = [s[0], self.epsilon] # B
                elif cond == [False,True,False]:
                    s_proj = [self.epsilon ,s[1]] # C
                elif cond == [True,True,False]:
                    s_proj = s # D
                elif cond == [True,False,True]:
                    s_proj = [self.delta, self.epsilon] # E
                elif cond == [False,True,True]:
                    s_proj = [self.epsilon, self.delta] # F
                elif cond == [True,True,True]:
                    s_proj = self.rmax*s /np.linalg.norm(s,2) # % G

                eta_proj = np.array([phi, s_proj[0], s_proj[1], t[0], t[1]])
                err3 = self.gamma[j] * (eta_proj - self.eta[:,j]).reshape(5)

                # complete step
                self.deta[:,j] = err1 + err2# + err3
                self.dgamma[j] = np.linalg.norm(self.eta[:,j] - eta_proj,2)

            # Update parameters 
            ts = time.time() - start
            for j,id in enumerate(self.configDict.keys()):
                # Euler integration
                self.eta[:,j] = self.eta[:,j] + ts*self.deta[:,j]
                self.gamma[j] = self.gamma[j] + ts*self.dgamma[j]
                print("eta-{}: {}".format(id, self.eta[:,j]))

            if self.record_log:
                pose = []
                goal = []
                eta = []
                deta = []
                gamma = []
                dgamma = []
                for j,id in enumerate(self.configDict.keys()):
                    # Pose
                    x,y,z,yaw = self.currPos[id]
                    pose.append(x)
                    pose.append(y)
                    pose.append(z)
                    pose.append(yaw)
                    # Goal
                    x,y = self.p[:,j]
                    goal.append(x)
                    goal.append(y)
                    # eta 
                    a,b,c,d,e = self.eta[:,j] 
                    eta.append(a)
                    eta.append(b)
                    eta.append(c)
                    eta.append(d)
                    eta.append(e)
                    # deta
                    a,b,c,d,e = self.deta[:,j]
                    deta.append(a)
                    deta.append(b)
                    deta.append(c)
                    deta.append(d)
                    deta.append(e)
                    # gamma
                    gamma.append(self.gamma[j])
                    # dgamma
                    dgamma.append(self.dgamma[j])


                self.pose_log.log(pose)
                self.goal_log.log(goal)
                self.eta_log.log(eta)
                self.deta_log.log(deta)
                self.gamma_log.log(gamma)
                self.dgamma_log.log(dgamma)

            

    def create_log(self, name, folder):
        _log = log(name=name, folder=folder)
        self.logs[name] = _log
        return _log
    
    def write_log(self):
        for _, _log in self.logs.items():
            _log.write()




if __name__ == "__main__":

    _record_log = True
    _plot_log = True
    _save_plot_log = True

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    SControl = SwarmControl(swarm=allcfs,timeHelper=timeHelper,record_log=_record_log)

    allcfs.takeoff(targetHeight=HOVER_HEIGHT, duration=1.0+TAKEOFF_SLEEP)
    timeHelper.sleep(3)

    SControl.goToInitPos()
    SControl.controlLoop()

    # timer = time.time()
    # while timer + 5 > time.time():
    # for cf in allcfs.crazyflies:    
    #     cf.cmdPosition(np.array([0,0,HOVER_HEIGHT]),yaw=np.pi/2)
    # timeHelper.sleepForRate(50)

    # for cf in allcfs.crazyflies:
    #     cf.goTo(goal=np.array([0,0,HOVER_HEIGHT]),yaw=np.pi/2,duration=4)
    # timeHelper.sleep(5)


    allcfs.land(targetHeight=LAND_HEIGHT, duration=1.0+TAKEOFF_SLEEP)
    timeHelper.sleep(3)


    

    if _record_log:
        SControl.write_log()
        # SControl.init_pose_log.write()
        # SControl.pose_log.write()
        # SControl.des_pose_log.write()
        # SControl.des_eta_log.write()
        # SControl.goal_log.write()
        # SControl.eta_log.write()
        # SControl.deta_log.write()
        # SControl.gamma_log.write()
        # SControl.dgamma_log.write()
    
    if _plot_log:
        plotLog(folder_path=SControl.init_pose_log.folder, save_plot_log=_save_plot_log)