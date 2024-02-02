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
from graph_constructor import Graph
import copy

class BIGRACE:

    def __init__(self, counterfactual):
        self.percentage = counterfactual.percentage
        self.cluster = counterfactual.cluster
        self.ioi_label = self.cluster.undesired_class
        self.alpha, self.beta, self.gamma, self.delta1, self.delta2, self.delta3 = counterfactual.alpha, counterfactual.beta, counterfactual.gamma, counterfactual.delta1, counterfactual.delta2, counterfactual.delta3
        self.normal_x_cf, self.nodes_solution, self.centroid_nodes_solution, self.likelihood_dict, self.effectiveness_dict, self.run_time, self.model_status, self.obj_val = self.solve_problem(counterfactual)
    
    def solve_problem(self, counterfactual):
        """
        Generates the solution according to the BIGRACE algorithms
        """
        normal_x_cf_dict, nodes_solution_list, centroid_nodes_solutions_dict, likelihood_dict, effectiveness_dict, model_status_list, obj_val_list = {}, [], {}, {}, {}, [], []
        centroids_feat_list = list(self.cluster.centroids_dict.keys())
        start_time = time.time()
        for feature in centroids_feat_list:
            graph = Graph(counterfactual.data, counterfactual.model, self.cluster, feature, counterfactual.type, self.percentage)
            normal_x_cf, nodes_solution, centroid_nodes_solution, likelihood, effectiveness, model_status, obj_val = self.Bigrace(counterfactual, graph)
            normal_x_cf_dict.update(normal_x_cf)
            nodes_solution_list.extend(nodes_solution)
            centroid_nodes_solutions_dict.update(centroid_nodes_solution)
            likelihood_dict.update(likelihood)
            effectiveness_dict.update(effectiveness)
            model_status_list.append(model_status)
            obj_val_list.append(obj_val)
        end_time = time.time()
        run_time = end_time - start_time
        return normal_x_cf_dict, nodes_solution_list, centroid_nodes_solutions_dict, likelihood_dict, effectiveness_dict, run_time, model_status_list, obj_val_list

    def Bigrace(self, counterfactual, graph):
        """
        BIGRACE algorithm
        """
        normal_x_cf, nodes_solution, centroid_nodes_solution, likelihood, effectiveness, model_status, obj_val = self.do_optimize_all(counterfactual, graph)
        return normal_x_cf, nodes_solution, centroid_nodes_solution, likelihood, effectiveness, model_status, obj_val 

    def do_optimize_all(self, counterfactual, graph):
        """
        Method that finds Foce CF prioritizing likelihood using Gurobi optimization package
        """
        """
        MODEL
        """
        opt_model = gp.Model(name='BIG-RACE')
        G = nx.DiGraph()
        G.add_nodes_from(graph.rho)

        def unfeasible_case(graph):
            """
            Obtains the feasible justified solution when the problem is unfeasible
            """
            sol_x, centroids_solved, nodes_solution, centroid_nodes_solution, likelihood, effectiveness = {}, [], [], {}, {}, {}
            potential_CF = {}
            for c_idx in range(1, len(graph.feature_centroids) + 1):
                for i in range(1, len(graph.all_nodes) + 1):
                    if graph.F[c_idx, i]:
                        potential_CF[c_idx, i] = graph.C[c_idx, i]
                        if c_idx not in centroids_solved:
                            centroids_solved.append(c_idx)
                        if i not in nodes_solution:
                            nodes_solution.append(i)
            for c_idx in centroids_solved:
                centroid_idx = graph.feature_centroids[c_idx - 1].centroid_idx
                centroids_solved_i = dict([(tup, potential_CF[tup]) for tup in list(potential_CF.keys()) if tup[0] == c_idx])
                _, sol_x_idx = min(centroids_solved_i, key=centroids_solved_i.get)
                sol_x[centroid_idx] = graph.all_nodes[sol_x_idx - 1]
                centroid_nodes_solution[centroid_idx] = sol_x_idx
                likelihood[sol_x_idx] = graph.rho[sol_x_idx]
                effectiveness[sol_x_idx] = graph.eta[sol_x_idx]
                if sol_x_idx not in nodes_solution:
                    nodes_solution.append(sol_x_idx)
            not_centroids_solved = [i for i in range(1, len(graph.feature_centroids) + 1) if i not in centroids_solved]
            for c_idx in not_centroids_solved:
                centroid_normal_x = graph.feature_centroids[c_idx - 1].normal_x
                centroid_idx = graph.feature_centroids[c_idx - 1].centroid_idx
                train_cfs = graph.find_train_cf(counterfactual.data, counterfactual.model, counterfactual.type, extra_search=True)
                for train_cf_idx in range(train_cfs.shape[0]):
                    train_cf = train_cfs[train_cf_idx,:]
                    if verify_feasibility(centroid_normal_x, train_cf, counterfactual.data):
                        cf_instance = train_cf
                sol_x[centroid_idx] = cf_instance
                nodes_solution.append(sol_x_idx)
                centroid_nodes_solution[centroid_idx] = sol_x_idx
                likelihood[sol_x_idx] = graph.rho[sol_x_idx]
                effectiveness[sol_x_idx] = graph.eta[sol_x_idx]
            return sol_x, nodes_solution, centroid_nodes_solution, likelihood, effectiveness

        """
        SETS
        """
        set_Centroids = range(1, len(graph.feature_centroids) + 1)               
            
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
                opt_model.addConstr(cf[c, n] <= graph.F[c, n])
        
        for c in set_Centroids:
            opt_model.addConstr(gp.quicksum(cf[c, i] for i in G.nodes) == 1)
        
        def calculate_s_dist(cf, C, centroids_idx, nodes_idx):
            c_dist = cf.prod(C)/len(centroids_idx)
            s_dist = gp.quicksum((cf[c, i]*C[c, i] - c_dist)**2 for i in nodes_idx for c in centroids_idx)
            return s_dist
            
        # def calculate_s_dist(cf, C, W, CW, centroids_idx, nodes_idx):
        #     var = 0
        #     mean_value = cf.prod(CW)
        #     c_idx_checked = []
        #     c_dist_all = []
        #     for c_idx in centroids_idx:
        #         if c_idx in c_idx_checked:
        #             continue
        #         else:
        #             sensitive_group = graph.feature_groups[c_idx - 1]
        #             c_idx_sensitive_group_list = [i[0] + 1 for i in enumerate(graph.feature_groups) if graph.feature_groups[i[0]] == sensitive_group]
        #             sensitive_group_weight = 0
        #             for c_idx_feat_val in c_idx_sensitive_group_list:
        #                 sensitive_group_weight += W[c_idx_feat_val]
        #             c_dist_sensitive_group = 0 
        #             for c_idx_feat_val in c_idx_sensitive_group_list:
        #                 c_dist_cluster = gp.quicksum(cf[c_idx_feat_val, i]*C[c_idx_feat_val, i] for i in nodes_idx)
        #                 c_dist_sensitive_group += (W[c_idx_feat_val]/sensitive_group_weight)*c_dist_cluster
        #             var += sensitive_group_weight*(c_dist_sensitive_group - mean_value)**2
        #             c_idx_checked.extend(c_idx_sensitive_group_list)
        #     return var

        def calculate_s_like(cf, rho, centroids_idx, nodes_idx):
            c_like = gp.quicksum(cf[c, i]*rho[i] for i in nodes_idx for c in centroids_idx)/len(centroids_idx)
            s_like = gp.quicksum((cf[c, i]*rho[i] - c_like)**2 for i in nodes_idx for c in centroids_idx)
            return s_like

        def calculate_s_eff(cf, eta, centroids_idx, nodes_idx):
            c_eff = gp.quicksum(cf[c, i]*eta[i] for i in nodes_idx for c in centroids_idx)/len(centroids_idx)
            s_eff = gp.quicksum((cf[c, i]*eta[i] - c_eff)**2 for i in nodes_idx for c in centroids_idx)
            return s_eff

        def fairness_objective(cf, C, rho, eta, centroids_idx, nodes_idx):
            s_dist = 0
            s_like = 0
            s_eff = 0
            if self.delta1 == 1:
                s_dist = calculate_s_dist(cf, C, centroids_idx, nodes_idx)
            if self.delta2 == 1:
                s_like = calculate_s_like(cf, rho, centroids_idx, nodes_idx)
            if self.delta3 == 1:
                s_eff = calculate_s_eff(cf, eta, centroids_idx, nodes_idx)
            return s_dist + s_like + s_eff

        opt_model.setObjective(cf.prod(graph.C)*self.alpha
                               - gp.quicksum(cf[c, i]*graph.rho[i] for i in G.nodes for c in set_Centroids)*self.beta
                               - gp.quicksum(cf[c, i]*graph.eta[i] for i in G.nodes for c in set_Centroids)*self.gamma
                               + fairness_objective(cf, graph.C, graph.rho, graph.eta, set_Centroids, G.nodes), GRB.MINIMIZE)
            
        """
        OPTIMIZATION AND RESULTS
        """
        opt_model.optimize()
        time.sleep(0.25)
        if opt_model.status == 3 or len(graph.all_nodes) == len(graph.train_cf):
            sol_x, nodes_solution, centroid_nodes_solution, likelihood, effectiveness = unfeasible_case(self, graph)
            obj_val = 1000
        else:
            print(f'Optimizer solution status: {opt_model.status}') # 1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED', 12: 'NUMERIC', 13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT'
            print(f'Solution:')
            obj_val = opt_model.ObjVal
            sol_x, nodes_solution, centroid_nodes_solution, likelihood, effectiveness = {}, [], {}, {}, {}
            for c in set_Centroids:
                time.sleep(0.25)
                for i in G.nodes:
                    if cf[c, i].x > 0.1:
                        centroid_idx = graph.feature_centroids[c - 1].centroid_idx
                        sol_x[centroid_idx] = graph.all_nodes[i - 1]
                        centroid_nodes_solution[centroid_idx] = i
                        likelihood[i] = graph.rho[i]
                        effectiveness[i] = graph.eta[i]
                        if i not in nodes_solution:
                            nodes_solution.append(i)
                        print(f'cf{c, i}: {cf[c, i].x}')
                        print(f'Node {i}: {graph.all_nodes[i - 1]}')
                        print(f'Centroid: {graph.feature_centroids[c - 1].normal_x}')
                        print(f'Distance: {np.round(graph.C[c, i], 5)}')
        return sol_x, nodes_solution, centroid_nodes_solution, likelihood, effectiveness, opt_model.status, obj_val