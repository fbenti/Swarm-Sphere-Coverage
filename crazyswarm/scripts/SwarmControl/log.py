# External
import time
import os.path
import os
from math import cos,sin,radians,degrees,asin,tanh
import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt

# Internal


class log():
    def __init__(self, name, folder, no_header=True, log_time=True, split_string=' '):

        # Chechk if folder is already present
        
        # root = './Swarm_Planner/log_' + test_type + '/'
        # if not os.path.isdir(root):
        #     os.mkdir(root)

        # root = root + 'log'
        # i = 0
        # folder = root + str(i) + "/"
        # while os.path.isdir(folder):
        #     i += 1
        #     folder = root + str(i) + "/"
        # os.mkdir(folder)
        
        self.folder = folder
        self.name = self.folder + name + '.log'
        self.no_header = no_header
        self.log_time = log_time
        
        self.headers = []
        self.data = []

        # Flag used to see if data should be appended or overwritten
        self.data_written = False
        
        self.split_string = split_string
        
        self.last_data = []
    
    
    def set_headers(self, *args):
        if self.no_header:
            return
        # Reset the headers
        self.headers = []
        # If the first input is a list, then use that
        if len(args[0]) > 1:
            for arg in args[0]:
                self.headers.append(arg)
        else:
            # Fill the headers
            for arg in args:
                self.headers.append(arg)

    
    def log(self, *args, only_new_data=True):
        if len(self.headers) == 0 and not self.no_header:
            print("No headers set, using default 1,2,...")
            tmp = [i for i in range(len(args))]
            self.set_headers(tmp)


        # Fill up the data
        tmp = []
        for arg in args:
            # tmp.append(arg)
            # MODIFIED BY ME
            for a in arg:
                tmp.append(a)
            
        # Log the time
        if self.log_time:
            tmp.append(time.time())
            # Switch the time to be the first element
            tmp = [tmp[-1]] + tmp[:-1]
        
        # Only add new data
        if only_new_data and tmp == self.last_data:
            return
        
        # Add this measurement to the data
        self.data.append(tmp)
        
        self.last_data = tmp
    
    # Only do in the end of the program
    def write(self):
        if not self.data_written:
            open_symbol = 'w'
        else:
            open_symbol = 'a'
        
        f = open(self.name, open_symbol)
        # Write header only the first time
        if not self.data_written and not self.no_header:
            f.write(self.list_to_string(self.headers))
        # Write the data
        for info in self.data:
            f.write(self.list_to_string(info))
        f.close()
        
        self.data_written = True
        # Erase the data, so it is not written again
        self.data = []
        
    
    def list_to_string(self, _list):
        string = ''
        for data in _list:
            string += str(data) + self.split_string
        
        # Remove fencepost error
        string = string[:-len(self.split_string)]
        # Add new line
        string += '\n'
        return string


def plotLog(folder_path : str, save_plot_log : bool = False):

    # Upload log into variables
    init_pose = np.loadtxt(folder_path+'init_pose_log.log')
    pose = np.loadtxt(folder_path+'pose_log.log')
    des_pose = np.loadtxt(folder_path+'des_pose_log.log')
    des_eta = np.loadtxt(folder_path+'des_eta_log.log')
    goal = np.loadtxt(folder_path+'goal_log.log')
    eta = np.loadtxt(folder_path+'eta_log.log')
    deta = np.loadtxt(folder_path+'deta_log.log')
    gamma_log = np.loadtxt(folder_path+'gamma_log.log')
    dgamma_log = np.loadtxt(folder_path+'dgamma_log.log')

    if save_plot_log:
        os.mkdir(folder_path + "plots/")

    # Log length
    t = len(pose[:,0])

    # Initial Position
    ix,iy = [],[]
    for i in range(1,len(init_pose[1:]),2):
        ix.append(init_pose[i])
        iy.append(init_pose[i+1])

    # Trajectory
    x,y,z,yaw = [],[],[],[]
    for i in range(1,len(pose[0,1:]),4):
        x.append(pose[:,i])
        y.append(pose[:,i+1])
        z.append(pose[:,i+2])
        yaw.append(pose[:,i+3])

    # Number of drones
    M = len(x)

    # Combining position, for plotting purpose
    arr_x = np.zeros((t,1))
    for el in x:
        arr_x = np.concatenate([arr_x,el.reshape(-1,1)],axis=1)
    arr_x = arr_x[:,1:]
    arr_y = np.zeros((t,1))
    for el in y:
        arr_y = np.concatenate([arr_y,el.reshape(-1,1)],axis=1)
    arr_y = arr_y[:,1:]
    p = np.array([
        arr_x,
        arr_y
    ])


    # Destination
    dx,dy = [],[]
    for i in range(1,len(des_pose[1:]),2):
        dx.append(des_pose[i])
        dy.append(des_pose[i+1])
    p_des = np.array([
        dx,dy
    ])

    # goal ???
    gx,gy = [],[]
    for i in range(1,len(goal[0,1:]),2):
        gx.append(goal[:,i])
        gy.append(goal[:,i+1])

    # Parameter: Eta desirede
    des_a,des_b,des_c,des_d,des_e = [],[],[],[],[]
    for i in range(1,len(des_eta[1:]),5):
        des_a.append(des_eta[i])
        des_b.append(des_eta[i+1])
        des_c.append(des_eta[i+2])
        des_d.append(des_eta[i+3])
        des_e.append(des_eta[i+4])

    # Actual eta
    a,b,c,d,e = [],[],[],[],[]
    for i in range(1,len(eta[0,1:]),5):
        a.append(eta[:,i])
        b.append(eta[:,i+1])
        c.append(eta[:,i+2])
        d.append(eta[:,i+3])
        e.append(eta[:,i+4])

    # Deta
    da,db,dc,dd,de = [],[],[],[],[]
    for i in range(1,len(deta[0,1:]),5):
        da.append(deta[:,i])
        db.append(deta[:,i+1])
        dc.append(deta[:,i+2])
        dd.append(deta[:,i+3])
        de.append(deta[:,i+4])

    # Gamma
    gamma = []
    for i in range(1,len(gamma_log[0,1:])+1):
        gamma.append(gamma_log[:,i])

    # Dgamma
    dgamma = []
    for i in range(1,len(dgamma_log[0,1:]),1):
        dgamma.append(gamma_log[:,i])


    # eta plot
    fig, ax = plt.subplots(5)
    for j in range(M):
        ax[0].plot(a[j][:])
        ax[0].hlines(y=des_a,xmin=0,xmax=t,linestyles='dashed',colors='black' )
        ax[0].set_xlim([0, t])

        ax[1].plot(b[j][:])
        ax[1].hlines(y=des_b,xmin=0,xmax=t,linestyles='dashed',colors='black' )
        ax[1].set_xlim([0, t])

        ax[2].plot(c[j][:])
        ax[2].hlines(y=des_c,xmin=0,xmax=t,linestyles='dashed',colors='black' )
        ax[2].set_xlim([0, t])
        
        ax[3].plot(d[j][:])
        ax[3].hlines(y=des_d,xmin=0,xmax=t,linestyles='dashed',colors='black' )
        ax[3].set_xlim([0, t])

        ax[4].plot(e[j][:])
        ax[4].hlines(y=des_e,xmin=0,xmax=t,linestyles='dashed',colors='black' )
        ax[4].set_xlim([0, t])
    if save_plot_log:
        plt.savefig(folder_path + "plots/" +"Eta_params.png")

    # gamma plot
    plt.figure()
    for j in range(0,M):
        plt.plot(gamma[j][:],linewidth=2)
    if save_plot_log:
        plt.savefig(folder_path + "plots/" + "Gamma.png")


    # position plot
    plt.figure()
    for j in range(0,M):
        plt.plot(x[j][:],y[j][:],linewidth=2,linestyle='dashed')
        plt.plot(x[j][-1],y[j][-1],'gx',markersize=10)
        plt.scatter(ix[:],iy[:],s=80,facecolors='none',edgecolors='b')
        plt.scatter(dx[:],dy[:],s=80,facecolors='none',edgecolors='r')
    if save_plot_log:
        plt.savefig(folder_path + "plots/" + "Trajectories.png")


    # err
    err = np.square(np.linalg.norm(np.reshape(p_des,(2,1,M)) - p, ord=2, axis=0)).reshape(t,M)
    err_mean = np.mean(err,1)
    # plot squared error and mean squared error
    plt.figure()
    for j in range(0,M):
        plt.plot(err[:,j],'k',linewidth=2)
    plt.plot(err_mean,'r',linewidth=2)
    plt.yscale('log')
    if save_plot_log:
        plt.savefig(folder_path + "plots/" + "Errors.png")

    plt.show()               