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

class FIJUICE_LIKE_OPTIMIZE:

    def __init__(self, counterfactual):
        self.cluster = counterfactual.cluster
        self.ioi_label = self.cluster.undesired_class
        self.lagrange = counterfactual.lagrange
        self.alpha, self.beta, self.gamma = counterfactual.alpha, counterfactual.beta, counterfactual.gamma
        self.t = counterfactual.t
        self.k = counterfactual.k
        self.graph = counterfactual.graph
        start_time = time.time()
        self.normal_x_cf, self.nodes_solution, self.model_status = self.Fijuice(counterfactual)
        end_time = time.time()
        self.run_time = end_time - start_time

    def find_potential_justifiers(self, counterfactual, ijuice_search=False):
        """
        Finds the set of training observations belonging to, and predicted as, the counterfactual class
        """
        train_np = counterfactual.data.transformed_train_np
        train_target = counterfactual.data.train_target
        train_pred = counterfactual.model.model.predict(train_np)
        if not ijuice_search:
            potential_justifiers = train_np[(train_target != self.ioi_label) & (train_pred != self.ioi_label)]
        else:
            potential_justifiers = train_np[train_target != self.ioi_label]
        potential_justifiers_df = pd.DataFrame(columns = ['centroid','feat','feat_val','justifiers'])
        for idx in range(len(self.cluster.filtered_centroids_list)):
            c = self.cluster.filtered_centroids_list[idx]
            feat = c.feat
            feat_val = c.feat_val
            normal_centroid = c.normal_x
            sort_potential_justifiers_centroid = []
            for i in range(potential_justifiers.shape[0]):
                if ijuice_search: 
                    if verify_feasibility(normal_centroid, potential_justifiers[i], counterfactual.data):
                        dist = distance_calculation(potential_justifiers[i], normal_centroid, counterfactual.data, type=counterfactual.type)
                        sort_potential_justifiers_centroid.append((potential_justifiers[i], dist))
                else:
                    dist = distance_calculation(potential_justifiers[i], normal_centroid, counterfactual.data, type=counterfactual.type)
                    sort_potential_justifiers_centroid.append((potential_justifiers[i], dist))
            sort_potential_justifiers_centroid.sort(key=lambda x: x[1])
            sort_potential_justifiers_centroid = [i[0] for i in sort_potential_justifiers_centroid]
            if len(sort_potential_justifiers_centroid) > self.t:
                sort_potential_justifiers_centroid = sort_potential_justifiers_centroid[:self.t]
            centroid_df_data = pd.DataFrame([[normal_centroid, feat, feat_val, sort_potential_justifiers_centroid]], index=[idx], columns=potential_justifiers_df.columns)
            potential_justifiers_df = pd.concat((potential_justifiers_df, centroid_df_data), axis=0)
        return potential_justifiers_df

    def Fijuice(self, counterfactual):
        """
        FairJUICE algorithm
        """
        normal_x_cf, nodes_solution, model_status = self.do_optimize_all(counterfactual)
        return normal_x_cf, nodes_solution, model_status 

    def do_optimize_all(self, counterfactual):
        """
        Method that finds FiJUICE CF prioritizing likelihood using Gurobi optimization package
        """
        """
        MODEL
        """
        opt_model = gp.Model(name='FiJUICE')
        G = nx.DiGraph()
        G.add_nodes_from(self.graph.rho)

        def unfeasible_case(self):
            """
            Obtains the feasible justified solution when the problem is unfeasible
            """
            sol_x, centroids_solved, nodes_solution = {}, [], []
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
                if sol_x_idx not in nodes_solution:
                    nodes_solution.append(sol_x_idx)
            not_centroids_solved = [i for i in range(1, len(self.cluster.filtered_centroids_list) + 1) if i not in centroids_solved]
            for c_idx in not_centroids_solved:
                pot_justifiers = self.find_potential_justifiers(counterfactual, ijuice_search=True)
                cf_instance = pot_justifiers.loc[pot_justifiers.index == c_idx - 1]['justifiers'].values[0][0]
                sol_x[c_idx] = cf_instance
                nodes_solution.append(sol_x_idx)
            return sol_x, nodes_solution

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

        for c in set_Centroids:
            opt_model.addConstr(gp.quicksum(self.graph.rho[i]*cf[c, i] for i in G.nodes) >= counterfactual.rho_min)
            
        def fairness_objective(cf, C, W, CW, centroids_idx, nodes_idx):
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

        opt_model.setObjective(cf.prod(self.graph.CW)*self.alpha - gp.quicksum(cf[c, i]*self.graph.rho[i] for i in G.nodes for c in set_Centroids)*self.beta + fairness_objective(cf, self.graph.C, self.graph.W, self.graph.CW, set_Centroids, G.nodes)*self.gamma, GRB.MINIMIZE)
            
        """
        OPTIMIZATION AND RESULTS
        """
        opt_model.optimize()
        time.sleep(0.25)
        if opt_model.status == 3 or len(self.graph.all_nodes) == len(self.graph.potential_justifiers):
            sol_x, nodes_solution = unfeasible_case(self)
            obj_val = 1000
        else:
            print(f'Optimizer solution status: {opt_model.status}') # 1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED', 12: 'NUMERIC', 13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT'
            print(f'Solution:')
            obj_val = opt_model.ObjVal
            sol_x, nodes_solution = {}, []
            for c in set_Centroids:
                time.sleep(0.25)
                for i in G.nodes:
                    if cf[c, i].x > 0.1:
                        sol_x[c] = self.graph.all_nodes[i - 1]
                        if i not in nodes_solution:
                            nodes_solution.append(i)
                        print(f'cf{c, i}: {cf[c, i].x}')
                        print(f'Node {i}: {self.graph.all_nodes[i - 1]}')
                        print(f'Centroid: {self.cluster.filtered_centroids_list[c - 1].normal_x}')
                        print(f'Distance: {np.round(self.graph.C[c, i], 5)}')
        return sol_x, nodes_solution, opt_model.status, obj_val