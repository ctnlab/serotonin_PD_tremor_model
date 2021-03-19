# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 14:50:22 2018

@author: Alex
"""

import numpy as np

import utils

class Arm:
    
    def __init__(self, shoulder_range, elbow_range, n=2):
        
        self.n = n
        
        self.M1 = 1.59
        self.M2 = 1.44 
        self.L1 = 0.30
        self.L2 = 0.35 # [m]
        self.S1 = 0.18 
        self.S2 = 0.21
        self.I1 = 0.0477 
        self.I2 = 0.0588 # kg/m^2
        self.b1 = 0.0 
        self.b2 = 0.0
        
        self.shoulder_min = shoulder_range[0]
        self.shoulder_max = shoulder_range[1]
        self.elbow_min = elbow_range[0]
        self.elbow_max = elbow_range[1]
        
        self.theta1 = utils.changeRange(0.5, 
                                        0.,
                                        1., 
                                        shoulder_range[0],
                                        shoulder_range[1]
                                        )

        self.theta2 = utils.changeRange(0.5,
                                        0.,
                                        1., 
                                        elbow_range[0],
                                        elbow_range[1]
                                        )
        self.dtheta1   = 0.0       
        self.dtheta2   = 0.0
        self.ddtheta1 = 0.0
        self.ddtheta2 = 0.0
        self.tau1 = 0.0 # shulder torque
        self.tau2 = 0.0 # elbow torque
        
        self.tau           = np.zeros(2)
        self.theta         = np.zeros(2)
        self.dot_theta     = np.zeros(2)
        self.dot_dot_theta = np.zeros(2)
        
        self.g  = 0.0 # 9.81 gravity
        self.dt = 0.01
        
        self.d_x = 0.0
        self.d_y = 0.0
        
        self.xElbow = self.L1*np.cos(self.theta1)
        self.yElbow = self.L1*np.sin(self.theta1)
        
        self.xEndEf = self.L1*np.cos(self.theta1) +\
                      self.L2*np.cos(self.theta1+self.theta2)
        self.yEndEf = self.L1*np.sin(self.theta1) +\
                      self.L2*np.sin(self.theta1+self.theta2)
        
      #  self.getInverseKinematic(self.xEndEf,self.yEndEf)
        
        self.sensors = np.zeros(4)
        self.sensors[0] = self.theta1 
        self.sensors[1] = self.theta2
        self.sensors[2] = self.dtheta1 
        self.sensors[3] = self.dtheta2
                    
        self.check_limits()
                    
#        self.getElbowPose()
 #       self.getEndEffectorPose()




        
    def PD_controller(self, theta_des, Kp1, Kp2, Kd1, Kd2):

        torquePD1 = Kp1*(theta_des[0] - self.theta1) - Kd1*self.dtheta1
        torquePD2 = Kp2*(theta_des[1] - self.theta2) - Kd2*self.dtheta2

        return [torquePD1, torquePD2]
    
    def check_limits(self):
        if(self.theta1<= np.deg2rad(-60)):
            self.tau1 = 0.0
            self.theta1     = np.deg2rad(-60)
            self.dtheta1   = 0.0
            self.ddtheta1 = 0.0
		
        if(self.theta1 >= np.deg2rad(150)):
            self.tau1 = 0.0
            self.theta1     = np.deg2rad(150)
            self.dtheta1   = 0.0
            self.ddtheta1 = 0.0

        if(self.theta2<= 0.0):
            self.tau2 = 0.0
            self.theta2     = 0.0
            self.dtheta2   = 0.0
            self.ddtheta2 = 0.0

        if(self.theta2>= np.deg2rad(180)):
            self.tau2 = 0.0
            self.theta2     = np.deg2rad(180)
            self.dtheta2   = 0.0
            self.ddtheta2 = 0.0
     
    def runge(self, x_dot, x, _dt):

		# Metodo di Runge-Kutta IV ordine
        k1 = x_dot*_dt
		
        k2 = (x_dot+k1/2.0)*self.dt

        k3 = (x_dot+k2/2.0)*self.dt
	
        k4 = (x_dot+k3)*self.dt

        return  x+(1.0/6.0)*(k1 +2.0*k2 +2.0*k3+k4)
    
    def setJoints(self, thetaS, thetaE):
	    self.theta1 = thetaS
	    self.theta2 = thetaE
	    self.sensPose()


    def setOmega(self, omegaS, omegaE):
        self.dtheta1 = omegaS
        self.dtheta2 = omegaE

    def setAccel(self, accS, accE):
        self.ddtheta1 = accS
        self.ddtheta2 = accE


    def SolveDirectDynamics(self, T1, T2):

        self.check_limits()
        self.tau1 = T1
        self.tau2 = T2

        self.calcArmAcceleration()

		
        self.dtheta2 = self.runge(self.ddtheta2, self.dtheta2, self.dt)
        self.dtheta1 = self.runge(self.ddtheta1, self.dtheta1, self.dt)

        self.theta2 = self.runge(self.dtheta2, self.theta2, self.dt)
        self.theta1 = self.runge(self.dtheta1, self.theta1, self.dt)
        

        self.sensPose()
        self.check_limits()

    
    def getElbowX(self):
        return self.L1*np.cos(self.theta1)

    def getElbowY(self):
        return self.L1*np.sin(self.theta1)
    
    def getEndEffectorPose(self):
        self.xEndEf = self.L1*np.cos(self.theta1) + self.L2*np.cos(self.theta1+self.theta2)
        self.yEndEf = self.L1*np.sin(self.theta1) + self.L2*np.sin(self.theta1+self.theta2)


    def getEndEfX(self):
        return self.L1*np.cos(self.theta1) + self.L2*np.cos(self.theta1+self.theta2)

    def getEndEfY(self):
        return self.L1*np.sin(self.theta1) + self.L2*np.sin(self.theta1+self.theta2)

    
    def sensPose(self):
        self.xEndEf = self.getEndEfX()
        self.yEndEf = self.getEndEfY()

        self.xElbow = self.getElbowX()
        self.yElbow = self.getElbowY()


    def getInverseKinematic(self, HANDX, HANDY):

        ARMLENGTH     = self.L1
        FOREARMLENGTH = self.L2

        r2     = 0.0
        r      = 0.0
        Beta   = 0.0

        r2 = pow(HANDX,2) + pow(HANDY,2)
        r = np.sqrt(r2)

        self.theta2 = 3.14159 - np.arccos( (r2 - pow(ARMLENGTH,2) - pow(FOREARMLENGTH,2) ) / ( (-2.0) * ARMLENGTH * FOREARMLENGTH))

        Beta = np.arccos(( pow(FOREARMLENGTH,2) - pow(ARMLENGTH,2) - r2) / (-2.0 * ARMLENGTH * r))
        self.theta1 = np.arctan2(HANDY, HANDX) - Beta
        
        self.check_limits()
        
        self.xElbow = (ARMLENGTH * np.cos(self.theta1))
        self.yElbow = (ARMLENGTH * np.sin(self.theta1))

    def calcArmAcceleration(self):

        cos2   = np.cos(self.theta2)
        sin2   = np.sin(self.theta2)
        sin12  = np.sin(self.theta1 + self.theta2) 
        M2L1S2 = self.M2*self.L1*self.S2

		
		
		
        b0110  = self.I2 + M2L1S2*cos2
        b00    = self.I1 + self.I2 + 2.0*M2L1S2*cos2 + self.M2*self.L1*self.L1

        B  = np.zeros([2,2])
        C  = np.zeros([2,2])
        Fv = np.zeros([2,2])

        B[0,0] = b00
        B[0,1] = b0110
        B[1,0] = b0110
        B[1,1] = self.I2
		
        c00 = -2*M2L1S2*sin2*self.dtheta2
        c01 = -M2L1S2*self.dtheta2
        c10 =  M2L1S2*sin2*self.dtheta1
        c11 = 0.0

        C[0,0] = c00
        C[0,1] = c01
        C[1,0] = c10
        C[1,1] = c11

        Fv[0,0] = self.b1
        Fv[0,1] = 0.0
        Fv[1,0] = 0.0
        Fv[1,1] = self.b2

        gq0 = self.g*(self.M1*self.S1 + self.M2*self.L1)*np.sin(self.theta1) + self.g*self.M1*self.S2*sin12

        gq = np.zeros(2) # the gravity vector is zero for horizontal arm

        gq[0] = gq0

        gq1 = self.g*self.M2*self.S2*sin12

        gq[1] = gq1

		
        self.tau[0] = self.tau1
        self.tau[1] = self.tau2
		
		
        self.dot_theta[0] = self.dtheta1
        self.dot_theta[1] = self.dtheta2

		
        J = np.zeros([2,2])
        J = self.getJacobian()

        self.dot_dot_theta = np.zeros(2)

        #self.dot_dot_theta = inv(B)*(self.tau - dot(C,self.dot_theta) - dot(Fv,self.dot_theta) - gq -J.T*self.getExternalField())
        auxVec = np.zeros(2)
        auxVec = (self.tau - np.dot(C,self.dot_theta) - np.dot(Fv,self.dot_theta) - gq)
        self.dot_dot_theta = np.dot(np.linalg.inv(B), auxVec)

        #print("DEBUGGING ")
        #print(inv(B))
        #print(self.tau)
        #print(C)
        #print(" self.dot_theta dentro DEBUGGING")
        #print(self.dot_dot_theta.shape)
        #print(self.ddtheta1)
        #print(self.ddtheta2)
		
        self.ddtheta1 = self.dot_dot_theta[0]
        self.ddtheta2 = self.dot_dot_theta[1]

        #print(" self._ddtheta 1 e 2 dentro DEBUGGING subito dopo assegnazione ")
        #print(self.ddtheta1)
        #print(self.ddtheta2)


    def getElbowPose(self):
        self.xElbow = self.L1*np.cos(self.theta1)
        self.yElbow = self.L1*np.sin(self.theta1)


    #def getArmLenghts(self):
		
        #l1 = self.L1
        #l2 = self.L2

     #  return l1, l2


		
    def getExternalField(self):
        BField = np.zeros([2,2])

        # Yes force field
        BField[0,0] =  0.0
        BField[0,1] =  13.0 # -0.3
        BField[1,0] = -13.0 # 0.3
        BField[1,1] =  0.0

        self.updateEndEffectorVelocity()

        EndEffVel = np.zeros(2)
        EndEffVel[0] = 0.0#self.d_x
        EndEffVel[1] = 0.0#self.d_x

        Fxy = np.zeros(2)
        Fxy = BField*EndEffVel

        return Fxy

    def updateEndEffectorVelocity(self):
        d_x_y = np.zeros(2)
        d_qs  = np.zeros(2)

        d_qs[0] = self.dtheta1
        d_qs[1] = self.dtheta2

        J = np.zeros([2,2])
        J = self.getJacobian()
		
        d_x_y = J.T*d_qs
        self.d_x = d_x_y[0]
        self.d_y = d_x_y[1]


    def getEndEffectorVelocity(self):
        self.updateEndEffectorVelocity()

    def getJacobian(self):
        J = np.zeros([2,2])

        j00 = -self.L1*np.sin(self.theta1)-self.L2*np.sin(self.theta1+self.theta2)
        j01 = -self.L2*np.sin(self.theta1+self.theta2)
        j10 =  self.L1*np.cos(self.theta1)+self.L2*np.cos(self.theta1+self.theta2)
        j11 =  self.L2*np.cos(self.theta1+self.theta2)

        J[0,0] = j00
        J[0,1] = j01
        J[1,0] = j10 
        J[1,1] = j11

        return J

    def getEndEffectorPos(self):

        self.xEndEf = self.L1*np.cos(self.theta1) + self.L2*np.cos(self.theta1+self.theta2)
        self.yEndEf = self.L1*np.sin(self.theta1) + self.L2*np.sin(self.theta1+self.theta2)

        return [self.xEndEf, self.yEndEf]
    
    def epochReset(self, shoulderRange, elbowRange):
        
        self.tau1 = 0.0 # shulder torque
        self.tau2 = 0.0 # elbow torque
        self.theta1     = utils.changeRange(0.5, 0.,1., shoulderRange[0], shoulderRange[1])
        self.theta2     = utils.changeRange(0.5, 0.,1., elbowRange[0], elbowRange[1])
        self.dtheta1   = 0.0       
        self.dtheta2   = 0.0
        self.ddtheta1 = 0.0
        self.ddtheta2 = 0.0
        
        self.tau           = np.zeros(2)
        self.theta         = np.zeros(2)
        self.dot_theta     = np.zeros(2)
        self.dot_dot_theta = np.zeros(2)
        
        self.g  = 0.0 # 9.81 gravity
        self.dt = 0.01
        
        self.d_x = 0.0
        self.d_y = 0.0
        
        self.xElbow = self.L1*np.cos(self.theta1)
        self.yElbow = self.L1*np.sin(self.theta1)
        self.xEndEf = self.L1*np.cos(self.theta1) + self.L2*np.cos(self.theta1+self.theta2)
        self.yEndEf = self.L1*np.sin(self.theta1) + self.L2*np.sin(self.theta1+self.theta2)
        
   #     self.getInverseKinematic(self.xEndEf,self.yEndEf)
        
        self.sensors = np.zeros(4)
        self.sensors[0] = self.theta1 
        self.sensors[1] = self.theta2
        self.sensors[2] = self.dtheta1 
        self.sensors[3] = self.dtheta2
                    
        self.check_limits()
                    
    #    self.getElbowPose()
    #    self.getEndEffectorPose()
    
    def setEffPosition(self, position, shoulderRange, elbowRange):
        
        self.tau1 = 0.0 # shulder torque
        self.tau2 = 0.0 # elbow torque
        self.theta1     = utils.changeRange(position[0], 0.,1., shoulderRange[0], shoulderRange[1])
        self.theta2     = utils.changeRange(position[1], 0.,1., elbowRange[0], elbowRange[1])
        self.dtheta1   = 0.0       
        self.dtheta2   = 0.0
        self.ddtheta1 = 0.0
        self.ddtheta2 = 0.0
        
        self.tau           = np.zeros(2)
        self.theta         = np.zeros(2)
        self.dot_theta     = np.zeros(2)
        self.dot_dot_theta = np.zeros(2)
        
        self.g  = 0.0 # 9.81 gravity
        self.dt = 0.01
        
        self.d_x = 0.0
        self.d_y = 0.0
        
        self.xElbow = self.L1*np.cos(self.theta1)
        self.yElbow = self.L1*np.sin(self.theta1)
        self.xEndEf = self.L1*np.cos(self.theta1) + self.L2*np.cos(self.theta1+self.theta2)
        self.yEndEf = self.L1*np.sin(self.theta1) + self.L2*np.sin(self.theta1+self.theta2)
        
    #    self.getInverseKinematic(self.xEndEf,self.yEndEf)
        
        self.sensors = np.zeros(4)
        self.sensors[0] = self.theta1 
        self.sensors[1] = self.theta2
        self.sensors[2] = self.dtheta1 
        self.sensors[3] = self.dtheta2
                    
        self.check_limits()
                    
  #      self.getElbowPose()
   #     self.getEndEffectorPose()
        
        
    def stopMove(self):
        
        self.tau1 = 0.0
        self.tau2 = 0.0
        
        self.dtheta1   = 0.0
        self.ddtheta1 = 0.0
        
        self.dtheta2   = 0.0
        self.ddtheta2 = 0.0
        
            
            
            
            
            
            
            
                    