import numpy as np
import pandas as pd
from itertools import product
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, tuplelist
from evaluator_constructor import distance_calculation, verify_feasibility
from nnt import nn_for_juice
import time
from scipy.stats import norm

class FOCE_OPTIMIZE:

    def __init__(self, counterfactual):
        self.percentage = counterfactual.graph.percentage
        self.cluster = counterfactual.cluster
        self.ioi_label = self.cluster.undesired_class
        # self.lagrange = counterfactual.lagrange
        self.alpha, self.beta, self.gamma, self.delta1, self.delta2, self.delta3 = counterfactual.alpha, counterfactual.beta, counterfactual.gamma, counterfactual.delta1, counterfactual.delta2, counterfactual.delta3
        self.graph = counterfactual.graph
        start_time = time.time()
        self.normal_x_cf, self.nodes_solution, self.centroid_nodes_solution, self.model_status, self.obj_val = self.Foce(counterfactual)
        end_time = time.time()
        self.run_time = end_time - start_time

    def find_train_cf(self, data, model, type, extra_search=False):
        """
        Finds the set of training observations belonging to, and predicted as, the counterfactual class
        """
        train_np = data.transformed_train_np
        train_target = data.train_target
        train_pred = model.model.predict(train_np)
        if not extra_search:
            train_cf = train_np[(train_target != self.ioi_label) & (train_pred != self.ioi_label)]
        else:
            train_cf = train_np[train_target != self.ioi_label]
        # train_cf_df = pd.DataFrame(columns = ['centroid','feat','feat_val','train_cfs'])
        sort_train_cf_centroid = []
        for idx in range(len(self.cluster.filtered_centroids_list)):
            c = self.cluster.filtered_centroids_list[idx]
            # feat = c.feat
            # feat_val = c.feat_val
            normal_centroid = c.normal_x
            # sort_train_cf_centroid = []
            for i in range(train_cf.shape[0]):
                if extra_search: 
                    if verify_feasibility(normal_centroid, train_cf[i], data):
                        dist = distance_calculation(train_cf[i], normal_centroid, data, type=type)
                        sort_train_cf_centroid.append((train_cf[i], dist))
                else:
                    dist = distance_calculation(train_cf[i], normal_centroid, data, type=type)
                    sort_train_cf_centroid.append((train_cf[i], dist))
            # if len(sort_train_cf_centroid) > self.t:
            # sort_train_cf_centroid = sort_train_cf_centroid[:self.t]
            # centroid_df_data = pd.DataFrame([[normal_centroid, feat, feat_val, sort_train_cf_centroid]], index=[idx], columns=train_cf_df.columns)
            # train_cf_df = pd.concat((train_cf_df, centroid_df_data), axis=0)
        # return train_cf_df
        sort_train_cf_centroid.sort(key=lambda x: x[1])
        sort_train_cf_centroid = [i[0] for i in sort_train_cf_centroid]
        sort_train_cf_centroid = sort_train_cf_centroid[:int(len(sort_train_cf_centroid)*self.percentage)]
        return sort_train_cf_centroid

    def Foce(self, counterfactual):
        """
        FairJUICE algorithm
        """
        normal_x_cf, nodes_solution, centroid_nodes_solution, model_status, obj_val = self.do_optimize_all(counterfactual)
        return normal_x_cf, nodes_solution, centroid_nodes_solution, model_status, obj_val 

    def do_optimize_all(self, counterfactual):
        """
        Method that finds Foce CF prioritizing likelihood using Gurobi optimization package
        """
        """
        MODEL
        """
        opt_model = gp.Model(name='Foce')
        G = nx.DiGraph()
        G.add_nodes_from(self.graph.rho)

        def unfeasible_case(self):
            """
            Obtains the feasible justified solution when the problem is unfeasible
            """
            sol_x, centroids_solved, nodes_solution, centroid_nodes_solution = {}, [], [], {}
            potential_CF = {}
            for c_idx in range(1, len(self.cluster.filtered_centroids_list) + 1):
                for i in range(1, len(self.graph.all_nodes) + 1):
                    if self.graph.F[c_idx, i]:
                        potential_CF[c_idx, i] = self.graph.C[c_idx, i]
                        if c_idx not in centroids_solved:
                            centroids_solved.append(c_idx)
                        if i not in nodes_solution:
                            nodes_solution.append(i)
            for c_idx in centroids_solved:
                centroids_solved_i = dict([(tup, potential_CF[tup]) for tup in list(potential_CF.keys()) if tup[0] == c_idx])
                _, sol_x_idx = min(centroids_solved_i, key=centroids_solved_i.get)
                sol_x[c_idx] = self.graph.all_nodes[sol_x_idx - 1]
                centroid_nodes_solution[c_idx] = sol_x_idx
                if sol_x_idx not in nodes_solution:
                    nodes_solution.append(sol_x_idx)
            not_centroids_solved = [i for i in range(1, len(self.cluster.filtered_centroids_list) + 1) if i not in centroids_solved]
            for c_idx in not_centroids_solved:
                centroid_normal_x = self.cluster.filtered_centroids_list[c_idx - 1].normal_x
                train_cfs = self.find_train_cf(counterfactual.data, counterfactual.model, counterfactual.type, extra_search=True)
                for train_cf_idx in range(train_cfs.shape[0]):
                    train_cf = train_cfs[train_cf_idx,:]
                    if verify_feasibility(centroid_normal_x, train_cf, counterfactual.data):
                        cf_instance = train_cf
                sol_x[c_idx] = cf_instance
                nodes_solution.append(sol_x_idx)
                centroid_nodes_solution[c_idx] = sol_x_idx
            return sol_x, nodes_solution, centroid_nodes_solution

        """
        SETS
        """
        set_Centroids = range(1, len(self.cluster.filtered_centroids_list) + 1)               
            
        """
        VARIABLES
        """
        for c in set_Centroids:
            cf = opt_model.addVars(set_Centroids, G.nodes, vtype=GRB.BINARY, name='Counterfactual')   # Node chosen as destination
            
        """
        CONSTRAINTS AND OBJECTIVE
        """
        for c in set_Centroids:
            for n in G.nodes:
                opt_model.addConstr(cf[c, n] <= self.graph.F[c, n])
        
        for c in set_Centroids:
            opt_model.addConstr(gp.quicksum(cf[c, i] for i in G.nodes) == 1)

        # for c in set_Centroids:
        #     opt_model.addConstr(gp.quicksum(self.graph.rho[i]*cf[c, i] for i in G.nodes) >= counterfactual.rho_min)
        
        def calculate_s_dist(cf, C, W, CW, centroids_idx, nodes_idx):
            var = 0
            mean_value = cf.prod(CW)
            c_idx_checked = []
            c_dist_all = []
            for c_idx in range(len(centroids_idx)):
                c = centroids_idx[c_idx]
                if c in c_idx_checked:
                    continue
                else:
                    sensitive_group = self.cluster.group_dict[c]
                    c_idx_sensitive_group_list = [key for key,val in self.cluster.group_dict.items() if val == sensitive_group]
                    sensitive_group_weight = 0
                    for c_idx_feat_val in c_idx_sensitive_group_list:
                        sensitive_group_weight += W[c_idx_feat_val]
                    c_dist_sensitive_group = 0 
                    for c_idx_feat_val in c_idx_sensitive_group_list:
                        c_dist_cluster = gp.quicksum(cf[c_idx_feat_val, i]*C[c_idx_feat_val, i] for i in nodes_idx)
                        c_dist_sensitive_group += (W[c_idx_feat_val]/sensitive_group_weight)*c_dist_cluster
                    var += sensitive_group_weight*(c_dist_sensitive_group - mean_value)**2
                    c_idx_checked.extend(c_idx_sensitive_group_list)
            return var

        def calculate_s_like(cf, rho, centroids_idx, nodes_idx):
            c_like = gp.quicksum(cf[c, i]*rho[i] for i in nodes_idx for c in set_Centroids)/len(centroids_idx)
            s_like = gp.quicksum((cf[c, i]*rho[i] - c_like)**2 for i in nodes_idx for c in set_Centroids)
            return s_like

        def calculate_s_eff(cf, eta, centroids_idx, nodes_idx):
            c_eff = gp.quicksum(cf[c, i]*eta[i] for i in nodes_idx for c in set_Centroids)/len(centroids_idx)
            s_eff = gp.quicksum((cf[c, i]*eta[i] - c_eff)**2 for i in nodes_idx for c in set_Centroids)
            return s_eff

        def fairness_objective(cf, C, W, CW, rho, eta, centroids_idx, nodes_idx):
            if self.delta1 == 1:
                s_dist = calculate_s_dist(cf, C, W, CW, centroids_idx, nodes_idx)
            else:
                s_dist = 0
            if self.delta2 == 1:
                s_like = calculate_s_like(cf, rho, centroids_idx, nodes_idx)
            else:
                s_like = 0
            if self.delta3 == 1:
                s_eff = calculate_s_eff(cf, eta, centroids_idx, nodes_idx)
            else:
                s_eff = 0
            return s_dist + s_like + s_eff

        opt_model.setObjective(cf.prod(self.graph.CW)*self.alpha
                               - gp.quicksum(cf[c, i]*self.graph.rho[i] for i in G.nodes for c in set_Centroids)*self.beta
                               - gp.quicksum(cf[c, i]*self.graph.eta[i] for i in G.nodes for c in set_Centroids)*self.gamma
                               + fairness_objective(cf, self.graph.C, self.graph.W, self.graph.CW, self.graph.rho, self.graph.eta, set_Centroids, G.nodes), GRB.MINIMIZE)
            
        """
        OPTIMIZATION AND RESULTS
        """
        opt_model.optimize()
        time.sleep(0.25)
        if opt_model.status == 3 or len(self.graph.all_nodes) == len(self.graph.train_cf):
            sol_x, nodes_solution, centroid_nodes_solution = unfeasible_case(self)
            obj_val = 1000
        else:
            print(f'Optimizer solution status: {opt_model.status}') # 1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED', 12: 'NUMERIC', 13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT'
            print(f'Solution:')
            obj_val = opt_model.ObjVal
            sol_x, nodes_solution, centroid_nodes_solution = {}, [], {}
            for c in set_Centroids:
                time.sleep(0.25)
                for i in G.nodes:
                    if cf[c, i].x > 0.1:
                        sol_x[c] = self.graph.all_nodes[i - 1]
                        centroid_nodes_solution[c] = i
                        if i not in nodes_solution:
                            nodes_solution.append(i)
                        print(f'cf{c, i}: {cf[c, i].x}')
                        print(f'Node {i}: {self.graph.all_nodes[i - 1]}')
                        print(f'Centroid: {self.cluster.filtered_centroids_list[c - 1].normal_x}')
                        print(f'Distance: {np.round(self.graph.C[c, i], 5)}')
        return sol_x, nodes_solution, centroid_nodes_solution, opt_model.status, obj_val