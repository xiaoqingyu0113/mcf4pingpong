import numpy as np
from collections import deque
import gtsam
from gtsam.symbol_shorthand import L,V,W,X
from mcf4pingpong.factors import *
from mcf4pingpong.dynamics.dynamics import *


# class FactorGraph(gtsam.NonlinearFactorGraph):
#     '''
#     tracking the added keys in a set
#     '''
#     def __init__(self):
#         super().__init__()
#         self.X = set()
#         self.L = set()
#         self.V = set()
#         self.W = set()
#     def push_back(self,factor):
#         super().push_back(factor)
#         for k in factor.keys():
#             if k < V(0):
#                 self.L.add(k - L(0))
#             elif k < W(0):
#                 self.V.add(k - V(0))
#             elif k < X(0):
#                 self.W.add(k - W(0))
#             else:
#                 self.X.add(k - X(0))
    


class IsamSolver:
    def __init__(self,camera_param_list,
                    Cd = 0.55,
                    Cm=1.5,
                    ez=0.89,
                    mu=0.20, 
                    graph_minimum_size=10,
                    ground_z0=0.100,
                    spin_prior = np.zeros(3), 
                    verbose = True):

        self.camera_param_list = camera_param_list
        self.aero_param = [Cd, Cm]
        self.bounce_param = [mu,ez]

        self.graph_minimum_size = graph_minimum_size
        self.ground_z0 = ground_z0
        self.verbose = verbose
        
        # settings:
        self.bounce_freeze_frames = 10

        # priors
        self.spin_prior = spin_prior
        self.pos_prior = np.zeros(3)
        self.vel_prior = np.zeros(3)

        # noise models
    
        ## priors    
        self.pos_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*30) # large uncertainty 
        self.vel_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*30) # large uncertainty
        self.spin_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*1) # small uncertainty
        
        ## cameras
        self.uv_noise = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)  # 2 pixels error
        self.camera_calibration_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*1e-3)

        ## aerodynamics
        self.pos_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*1e-3)
        self.vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*1e-4)
        self.spin_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*1e-4)

        ## bounces
        self.bounce_vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*1)
        self.bounce_spin_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3)*1)

        # graph reset
        self.reset()

    def reset(self):
        # time reset 
        self.t_max = -np.inf
        self.dt = None

        # graph reset
        parameters = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(parameters)
        self.initial_estimate = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()

        # index reset
        self.curr_node_idx = -1
        self.spin_idx = 0
        self.prev_bounce_idx = -self.bounce_freeze_frames

        # result reset
        self.current_estimate = None

        if self.verbose:
            print('Solver has been reset.')


    def estimate(self,data, pos_prior=None):
        if pos_prior is not None:
            self.pos_prior = pos_prior
        t, camera_id, u,v = data
        data = [float(t), int(camera_id), float(u),float(v)]
        if t < self.t_max:
            return None
        self.dt = t - self.t_max 
        self.add_node(data)
        
        if self.curr_node_idx > 2:
            try:
                self.optimize()
            except:
                self.reset()

        self.t_max = float(t) # keep at bottom
        if self.current_estimate is None:
            return None
        else:
            return self.current_estimate.atVector(L(self.curr_node_idx))


     
    def add_node(self,data):
        '''
        add nodes to the subgraph. 
        guess for nodes X, W will be added here. L and V will be added later.
        '''
        # accumulate index
        self.curr_node_idx += 1
        j = self.curr_node_idx # for convenience

        # ensure data types
        t, camera_id, u,v = data
        t = float(t); camera_id = int(camera_id);u = float(u);v = float(v)
        u,v = self.camera_param_list[camera_id].undistort_pixel(np.array([u,v]))

        
        # add node X
        K_gtsam, pose_gtsam = self.camera_param_list[camera_id].to_gtsam()
        self.graph.push_back(gtsam.GenericProjectionFactorCal3_S2(np.array([u,v]), self.uv_noise, X(j), L(j), K_gtsam)) # add noise
        self.graph.push_back(gtsam.PriorFactorPose3(X(j), pose_gtsam, self.camera_calibration_noise)) # add prior
        # -- add guess for X, L will be added later
        self.initial_estimate.insert(X(j),pose_gtsam)

        if self.verbose:
            print(f"add pixel detection X({j}) -> L({j})")
            print(f"add prior X({j})")

        # add priors for the variables
        if j == 0: 
            ## add position prior
            self.graph.push_back(PriorFactor3(self.pos_prior_noise,L(j),self.pos_prior))
            self.initial_estimate.insert(L(j),self.pos_prior) 

            ## add velocity prior
            self.graph.push_back(PriorFactor3(self.vel_prior_noise,V(j),self.vel_prior))

            ## add spin prior and guess
            self.graph.push_back(PriorFactor3(self.spin_prior_noise,W(0),self.spin_prior))
            self.initial_estimate.insert(W(self.spin_idx),self.spin_prior + np.random.rand(3)) # add random to avoid singularity

        else:
            ## position factor L
            self.graph.push_back(PositionFactor(self.pos_noise,L(j-1),V(j-1),L(j),self.t_max,t))
            self.graph.push_back(PriorFactor3(self.pos_prior_noise,L(j),self.pos_prior))
            self.initial_estimate.insert(L(j),self.pos_prior) 

            if self.verbose:
                print(f"add position factor L({j-1}),V({j-1}  -> L({j}))")

            ##  --------- dealing with bounce: velocity and spin factor depend on whether bounce occurs ------------
            ### if no estimation, assume no bounce
            if self.current_estimate is None:
                self.graph.push_back(VelocityFactor(self.vel_noise,V(j-1),W(self.spin_idx),V(j), self.t_max, t, self.aero_param))

                if self.verbose:
                    print(f"add Linear factor V({j-1}), W({self.spin_idx}) -> V({j})")

            ### estimation known, use this to decide whether bounce occurs
            else:
                ### bounce if z < z0 and vz < 0
                z_prev = self.current_estimate.atVector(L(j-1))[2]
                vz_prev = self.current_estimate.atVector(V(j-1))[2]
                if (z_prev < self.ground_z0) and (vz_prev < 0.0):
                    #### True bounce
                    if j > self.prev_bounce_idx + self.bounce_freeze_frames:
                        if self.verbose:
                            print('add bounce')
                            print(f'\t- adding bounce factor (v({j-1}), w({self.spin_idx}) -> v({j}))')
                            print(f'\t- adding bounce factor (v({j-1}), w({self.spin_idx}) -> w({self.spin_idx+1}))')
                        self.graph.push_back(BounceVelocityFactor(self.bounce_vel_noise,V(j-1),W(self.spin_idx),V(j),self.bounce_param))
                        self.graph.push_back(BounceSpinFactor(self.bounce_spin_noise,V(j-1),W(self.spin_idx),W(self.spin_idx+1),self.bounce_param))

                        # --- add guess ---------
                        prev_W = self.current_estimate.atVector(W(self.spin_idx))
                        prev_V = self.current_estimate.atVector(V(j-1))
                        al = compute_alpha(prev_V, prev_W, self.bounce_param)
                        if al < 0.4:
                            self.initial_estimate.insert(W(self.spin_idx+1), bounce_slide_spin_forward(prev_V,prev_W,self.bounce_param))
                            self.initial_estimate.insert(V(j), bounce_slide_velocity_forward(prev_V,prev_W,self.bounce_param))
                        else:
                            self.initial_estimate.insert(W(self.spin_idx+1), bounce_roll_spin_forward(prev_V,prev_W,self.bounce_param))
                            self.initial_estimate.insert(V(j), bounce_roll_velocity_forward(prev_V,prev_W,self.bounce_param))
                        # ----------------------

                        self.spin_idx += 1
                        self.prev_bounce_idx = j

                    #### Ajacent bounce, consider as noise 
                    else:
                        self.graph.push_back(VelocityFactor(self.vel_noise,V(j-1),W(self.spin_idx),V(j),self.t_max,t,self.aero_param))
                        # self.curr_bounce_idx = j
                        # if self.verbose:
                        #     print(f'bounce within the smooth, move on. Current bounce idx move to {self.curr_bounce_idx}')

                ### no bounce, use aerodynamics
                else:
                    self.graph.push_back(VelocityFactor(self.vel_noise,V(j-1),W(self.spin_idx),V(j),self.t_max,t,self.aero_param))
                    if self.verbose:
                        print(f"add Linear factor V({j-1}), W({self.spin_idx}) -> V({j})")


        if self.current_estimate is None:
            if not self.initial_estimate.exists(L(j)):
                self.initial_estimate.insert(L(j),self.pos_prior)
            if not self.initial_estimate.exists(V(j)):
                self.initial_estimate.insert(V(j),np.random.rand(3)-0.5)
        else:
            if not self.initial_estimate.exists(L(j)):
                self.initial_estimate.insert(L(j),self.current_estimate.atVector(L(j-1)))
            # just to avoid the added guess when bouncing
            if not self.initial_estimate.exists(V(j)):
                self.initial_estimate.insert(V(j),self.current_estimate.atVector(V(j-1)))



        

    
    def optimize(self):
        if self.verbose:
            print('\t- optimizing!')

        # incremental update
        self.isam.update(self.graph, self.initial_estimate)
        for _ in range(10):
            self.isam.update()

        # obtain the estimation value
        self.current_estimate = self.isam.calculateEstimate()

        # clear the guess
        self.graph.resize(0)
        self.initial_estimate.clear()
        
        

    