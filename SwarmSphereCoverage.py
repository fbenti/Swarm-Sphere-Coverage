import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation 
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing
from python_tsp.exact import solve_tsp_dynamic_programming




def rotZ(t):
    """
    Rotation around Z-axis.
    """
    return np.array([
        [np.cos(t), -np.sin(t), 0],
        [np.sin(t), np.cos(t), 0],
        [0, 0, 1]
    ],dtype=object)


def rotY(t):
    """
    Rotation around Z-axis.
    """
    return np.array([
        [np.cos(t), 0, np.sin(t)],
        [0, 1, 0],
        [-np.sin(t),0,np.cos(t)]
    ],dtype=object)


def rotX(t):
    """
    Rotation around Z-axis.
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(t), -np.sin(t)],
        [0, np.sin(t), np.cos(t)]
    ],dtype=object)


def generate_points(sphere_radius: float=5, distance_dir: float=2, points_per_m2: float=4 , plot=False):
    """ 
    Generate equidistant points around sphere.     
    """

    AreaSphere = 4 * np.pi * sphere_radius ** 2
    num_points = np.floor(AreaSphere * points_per_m2).astype(int)

    epsilon = 0.33
    goldenRatio = (1 + 5**0.5)/2
    i = np.arange(0, num_points) 
    theta = 2 *math.pi * i / goldenRatio
    phi = np.arccos(1 - 2*(i+epsilon)/(num_points-1+2*epsilon))
    X, Y, Z = sphere_radius * np.cos(theta) * np.sin(phi),  sphere_radius * np.sin(theta) * np.sin(phi), sphere_radius + sphere_radius * np.cos(phi)
    
    if plot:
        # plot roots
        fig = plt.figure()
        fig.suptitle("Points around sphere", fontsize=12)
        ax = fig.add_subplot(111, projection='3d')
        for i in range(num_points):
            ax.scatter(X[i],Y[i],Z[i], s=30,color='r')#color=color_list[i])
        # ax.set_xlim([-sphere_radius, sphere_radius])
        # ax.set_ylim([-sphere_radius, sphere_radius])
        # ax.set_zlim([0, 2*sphere_radius])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # plot sphere
        u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:20j]
        x = sphere_radius * np.cos(u)*np.sin(v)
        y = sphere_radius * np.sin(u)*np.sin(v)
        z = sphere_radius + sphere_radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color="black",linewidth=0.2)
        # plt.show()

    # all points around sphere
    points = np.array([X,Y,Z]).reshape(3,-1)

    #select only point above threshold
    points_thresh = points[:, np.argwhere(points[2,:] > distance_dir)]
    return points_thresh

def cluster_points(points: np.ndarray, sphere_radius: float=5, direction: np.ndarray=np.zeros((3)), num_drones: float=5, plot=False):
    """ 
    Cluster points using K-Means. 
    For each cluster return points and distance_matrix.
    """
    X=np.column_stack((points[0,:], points[1,:], points[2,:]))
    kmeans = KMeans(n_clusters=num_drones, random_state=0).fit(X)
    labels_ = np.unique(kmeans.labels_)
    centers_ = kmeans.cluster_centers_
    clusters = []
    distance_matrices = []
    for lb in labels_:
        cl = X[kmeans.labels_ == lb]
        cl = np.flip(cl,axis=0)
        R = Rotation.from_rotvec(direction)
        cl = R.apply(cl) 
        clusters.append(cl)
        distance_matrices.append(distance_matrix(cl,cl))

    if plot:
        fig = plt.figure()
        fig.suptitle("Partitions", fontsize=12)
        ax = fig.add_subplot(111, projection='3d')
        R = Rotation.from_rotvec(direction)
        u, v, w = R.apply([0,0,1])
        for i,cluster in enumerate(clusters):
            clusters[i] = cluster
            ax.scatter(cluster[:,0],cluster[:,1],cluster[:,2], s=10)
            centers_[i,:] = R.apply(centers_[i,:]) 
            ax.scatter(centers_[i,0],centers_[i,1],centers_[i,2], marker='x', s=40, color='black')

        ax.quiver(0, 0, 0, u,v,w, length=2, normalize=True)
        # ax.set_xlim([m, M])
        # ax.set_ylim([m, M])
        # ax.set_zlim([0, 2*sphere_radius])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # plt.show()
    
    return clusters, distance_matrices


def find_trajectories(clusters : list, distance_matrices : list, direction: np.ndarray=np.zeros((3)),  num_drones: int = 4, plot=True):
    """
    Given a list of clusters and their distance matrices, find a path through all the points using a tsp solver.
    """
    trajectories  = []
    distances = []
    permutations = []
    for dm in distance_matrices:
        # permutation, distance = solve_tsp_dynamic_programming(dm,1) # global
        permutation, distance = solve_tsp_local_search(dm) # local
        # permutation, distance = solve_tsp_simulated_annealing(dm) # local
        permutations.append(permutation)
        distances.append(distance)
        # print(permutation)
        print(distance)
    for cluster,permutation in zip(clusters,permutations):
        trajectory = []
        for i,index in enumerate(permutation):
            trajectory.append(cluster[index,:])
        trajectories.append(np.array(trajectory).reshape(-1,3))
    
    if plot:
        import time
        first = True
        max_lenght_path = 0
        for perm in permutations:
            length = len(perm)
            if length > max_lenght_path:
                max_lenght_path = len(perm)

        fig = plt.figure()
        fig.suptitle("3D Trajectories", fontsize=12)
        ax = fig.add_subplot(111, projection='3d')
        alpha = 0.1 # to simulate evanescence
        dalpha = (1 - alpha) / max_lenght_path

        from matplotlib.pyplot import cm
        color = cm.rainbow(np.linspace(0, 1, num_drones))
        # ax.set_xlim([-sphere_radius, sphere_radius])
        # ax.set_ylim([-sphere_radius, sphere_radius])
        # ax.set_zlim([0, 2*sphere_radius])
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        R = Rotation.from_rotvec(direction)
        u, v, w = R.apply([0,0,1])
        ax.quiver(0, 0, 0, u,v,w, length=2, normalize=True)
        for t in range(max_lenght_path):
            for j,traj in enumerate(trajectories):
                if t < len(traj):
                    ax.scatter(traj[t,0], traj[t,1], traj[t,2], alpha=alpha, color=color[j])
                    if t > 0:
                        ax.plot(traj[0:t+1,0], traj[0:t+1,1], traj[0:t+1,2],color=color[j],linewidth=1,linestyle='dashed',alpha=alpha)
            alpha += dalpha

            plt.pause(.3)
            if first:
                first = False
                time.sleep(10)
                

    plt.show()


SPHERE_RADIUS = 10
DIRECTION = np.array([np.pi/6,np.pi/6,0])
DISTANCE_DIR = 15 # distance from antenna to plane that cuts the sphere
POINTS_PER_M2 = 0.5 # points per squared meter around the sphere (density).
NUM_DRONES = 4


if __name__ == "__main__":
    # Generate points around sphere.
    points = generate_points(sphere_radius=SPHERE_RADIUS, distance_dir=DISTANCE_DIR, points_per_m2=POINTS_PER_M2, plot=True)
    # Cluster points based on number of drones.
    clusters, distance_matrices = cluster_points(points=points, sphere_radius=SPHERE_RADIUS, direction=DIRECTION, num_drones=NUM_DRONES, plot=True)
    # Find trajectories that satisfy complete coverage.
    trajectories = find_trajectories(clusters=clusters, distance_matrices=distance_matrices, direction=DIRECTION, num_drones=NUM_DRONES, plot=True)