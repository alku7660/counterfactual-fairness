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
from graph_constructor_nocluster import Graph
import copy

class BIGRACE:

    def __init__(self, counterfactual):
        self.percentage = counterfactual.percentage
        self.feat_protected = counterfactual.data.feat_protected
        self.false_undesired_test_df = counterfactual.data.false_undesired_test_df
        self.ioi_label = counterfactual.data.undesired_class
        self.sensitive_feat_idx_dict = self.select_instances_by_sensitive_group()
        self.alpha, self.beta, self.gamma, self.delta1, self.delta2, self.delta3 = counterfactual.alpha, counterfactual.beta, counterfactual.gamma, counterfactual.delta1, counterfactual.delta2, counterfactual.delta3
        self.normal_x_cf, self.graph_nodes, self.likelihood_dict, self.effectiveness_dict, self.run_time, self.model_status, self.obj_val, self.sensitive_group_idx_feat_value_dict = self.solve_problem(counterfactual)
    
    def select_instances_by_sensitive_group(self):
        """
        Obtains indices of each sensitive group and stores them in a dict
        """
        sensitive_feat_idx_dict = dict()
        for key in self.feat_protected.keys():
            idx_list_by_sensitive_group_dict = dict()
            value_dict = self.feat_protected[key]
            for value in value_dict.keys():
                sensitive_group_df = self.false_undesired_test_df.loc[self.false_undesired_test_df[key] == value]
                idx_list_sensitive_group = sensitive_group_df.index.to_list()
                idx_list_by_sensitive_group_dict[value] = idx_list_sensitive_group
            sensitive_feat_idx_dict[key] = idx_list_by_sensitive_group_dict
        return sensitive_feat_idx_dict

    def solve_problem(self, counterfactual):
        """
        Generates the solution according to the BIGRACE algorithms
        """
        normal_x_cf_dict, graph_nodes_solutions_dict, likelihood_dict, effectiveness_dict, model_status_dict, obj_val_dict, sensitive_group_idx_feat_value_dict = {}, {}, {}, {}, {}, {}, {}
        start_time = time.time()
        for feature in self.feat_protected.keys():
            value_dict = self.feat_protected[feature]
            sensitive_group_dict = self.sensitive_feat_idx_dict[feature]
            graph = Graph(counterfactual.data, counterfactual.model, feature, value_dict.keys(), sensitive_group_dict, counterfactual.type, self.percentage)
            normal_x_cf, graph_nodes_solution, likelihood, effectiveness, model_status, obj_val = self.Bigrace(counterfactual, graph)
            normal_x_cf, graph_nodes_solution = self.adapt_indices_results(graph, normal_x_cf, graph_nodes_solution)
            normal_x_cf_dict[feature] = normal_x_cf
            graph_nodes_solutions_dict[feature] = graph_nodes_solution
            likelihood_dict[feature] = likelihood
            effectiveness_dict[feature] = effectiveness
            model_status_dict[feature] = model_status
            obj_val_dict[feature] = obj_val
            sensitive_group_idx_feat_value_dict[feature] = graph.sensitive_group_idx_feat_value_dict
        end_time = time.time()
        run_time = end_time - start_time
        return normal_x_cf_dict, graph_nodes_solutions_dict, likelihood_dict, effectiveness_dict, run_time, model_status_dict, obj_val_dict, sensitive_group_idx_feat_value_dict

    def adapt_indices_results(self, graph, normal_x_cf, graph_nodes_solution):
        """
        Adapts the indices to the original index for each of the instances
        """
        modified_normal_x_cf, modified_graph_nodes_solution = {}, {}
        for idx in normal_x_cf.keys():
            x_cf = normal_x_cf[idx]
            graph_node = graph_nodes_solution[idx]
            original_x_idx = graph.instance_idx_to_original_idx_dict[idx]
            modified_normal_x_cf[original_x_idx] = x_cf
            modified_graph_nodes_solution[original_x_idx] = graph_node
        return modified_normal_x_cf, modified_graph_nodes_solution

    def Bigrace(self, counterfactual, graph):
        """
        BIGRACE algorithm
        """
        normal_x_cf, graph_nodes_solution, likelihood, effectiveness, model_status, obj_val = self.do_optimize_all(counterfactual, graph)
        return normal_x_cf, graph_nodes_solution, likelihood, effectiveness, model_status, obj_val 

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

        def unfeasible_case(graph, type):
            """
            Obtains the feasible justified solution when the problem is unfeasible
            """
            sol_x, centroids_solved, nodes_solution, graph_nodes_solution, likelihood, effectiveness = {}, [], [], {}, {}, {}
            data = counterfactual.data
            model = counterfactual.model
            feat_values = data.feat_protected[graph.feat].keys()
            potential_CF = {}
            for instance_idx in range(1, len(graph.sensitive_feature_instances) + 1):
                for i in range(1, len(graph.all_nodes) + 1):
                    if graph.F[instance_idx, i]:
                        potential_CF[instance_idx, i] = graph.C[instance_idx, i]
                        if instance_idx not in centroids_solved:
                            centroids_solved.append(instance_idx)
                        if i not in nodes_solution:
                            nodes_solution.append(i)
            for instance_idx in centroids_solved:
                centroids_solved_i = dict([(tup, potential_CF[tup]) for tup in list(potential_CF.keys()) if tup[0] == instance_idx])
                _, sol_x_idx = min(centroids_solved_i, key=centroids_solved_i.get)
                sol_x[instance_idx] = graph.all_nodes[sol_x_idx - 1]
                graph_nodes_solution[instance_idx] = sol_x_idx
                likelihood[sol_x_idx] = graph.rho[sol_x_idx]
                effectiveness[sol_x_idx] = graph.eta[sol_x_idx]
                if sol_x_idx not in nodes_solution:
                    nodes_solution.append(sol_x_idx)
            not_centroids_solved = [i for i in range(1, len(graph.sensitive_feature_instances) + 1) if i not in centroids_solved]
            for instance_idx in not_centroids_solved:
                instance = graph.sensitive_feature_instances[instance_idx - 1]
                train_cfs = graph.find_train_cf(data, model, counterfactual.type, extra_search=True)
                graph.nearest_neighbor_train_cf(self, data, model, feat_values, type, extra_search=False)
                for train_cf_idx in range(train_cfs.shape[0]):
                    train_cf = train_cfs[train_cf_idx,:]
                    if verify_feasibility(instance, train_cf, data):
                        cf_instance = train_cf
                sol_x[instance_idx] = cf_instance
                nodes_solution.append(sol_x_idx)
                graph_nodes_solution[instance_idx] = sol_x_idx
                likelihood[sol_x_idx] = graph.rho[sol_x_idx]
                effectiveness[sol_x_idx] = graph.eta[sol_x_idx]
            return sol_x, graph_nodes_solution, likelihood, effectiveness

        """
        SETS
        """
        set_Instances = range(1, len(graph.sensitive_feature_instances) + 1)               
            
        """
        VARIABLES
        """
        # for c in set_Instances:
        #     cf = opt_model.addVars(set_Instances, G.nodes, vtype=GRB.BINARY, name='Counterfactual')   # Node chosen as destination
        cf = opt_model.addVars(set_Instances, G.nodes, vtype=GRB.BINARY, name='Counterfactual')
        limiter = opt_model.addVars(G.nodes, vtype=GRB.BINARY, name='Limiter')
            
        """
        CONSTRAINTS AND OBJECTIVE
        """
        for i in set_Instances:
            for n in G.nodes:
                opt_model.addConstr(cf[i, n] <= graph.F[i, n])
        
        for i in set_Instances:
            opt_model.addConstr(gp.quicksum(cf[i, n] for n in G.nodes) == 1)
        
        for i in set_Instances:
            for n in G.nodes:
                opt_model.addConstr(cf[i, n] <= limiter[n])

        # def calculate_s_dist(cf, C, centroids_idx, nodes_idx):
        #     c_dist = cf.prod(C)/len(centroids_idx)
        #     s_dist = gp.quicksum((cf[c, i]*C[c, i] - c_dist)**2 for i in nodes_idx for c in centroids_idx)
        #     return s_dist

        # def calculate_s_like(cf, rho, centroids_idx, nodes_idx):
        #     c_like = gp.quicksum(cf[c, i]*rho[i] for i in nodes_idx for c in centroids_idx)/len(centroids_idx)
        #     s_like = gp.quicksum((cf[c, i]*rho[i] - c_like)**2 for i in nodes_idx for c in centroids_idx)
        #     return s_like

        # def calculate_s_eff(cf, eta, centroids_idx, nodes_idx):
        #     c_eff = gp.quicksum(cf[c, i]*eta[i] for i in nodes_idx for c in centroids_idx)/len(centroids_idx)
        #     s_eff = gp.quicksum((cf[c, i]*eta[i] - c_eff)**2 for i in nodes_idx for c in centroids_idx)
        #     return s_eff

        # def fairness_objective(cf, C, rho, eta, centroids_idx, nodes_idx):
        #     s_dist = 0
        #     s_like = 0
        #     s_eff = 0
        #     if self.delta1 == 1:
        #         s_dist = calculate_s_dist(cf, C, centroids_idx, nodes_idx)
        #     if self.delta2 == 1:
        #         s_like = calculate_s_like(cf, rho, centroids_idx, nodes_idx)
        #     if self.delta3 == 1:
        #         s_eff = calculate_s_eff(cf, eta, centroids_idx, nodes_idx)
        #     return s_dist + s_like + s_eff

        # opt_model.setObjective(cf.prod(graph.C)*self.alpha
        #                        - gp.quicksum(cf[c, i]*graph.rho[i] for i in G.nodes for c in set_Instances)*self.beta
        #                        - gp.quicksum(cf[c, i]*graph.eta[i] for i in G.nodes for c in set_Instances)*self.gamma
        #                        + fairness_objective(cf, graph.C, graph.rho, graph.eta, set_Instances, G.nodes), GRB.MINIMIZE)
        
        opt_model.setObjective(cf.prod(graph.C)*self.alpha + gp.quicksum(limiter[n] for n in G.nodes)*(1 - self.alpha), GRB.MINIMIZE)
            
        """
        OPTIMIZATION AND RESULTS
        """
        opt_model.optimize()
        time.sleep(0.25)
        if opt_model.status == 3 or len(graph.all_nodes) == len(graph.train_cf):
            sol_x, graph_nodes_solution, likelihood, effectiveness = unfeasible_case(graph, counterfactual.type)
            obj_val = 1000
        else:
            print(f'Optimizer solution status: {opt_model.status}') # 1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED', 12: 'NUMERIC', 13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT'
            print(f'Solution:')
            obj_val = opt_model.ObjVal
            sol_x, graph_nodes_solution, likelihood, effectiveness = {}, {}, {}, {}
            for i in set_Instances:
                time.sleep(0.25)
                for n in G.nodes:
                    if cf[i, n].x > 0.1:
                        sol_x[i] = graph.all_nodes[n - 1]
                        graph_nodes_solution[i] = n
                        likelihood[n] = graph.rho[n]
                        effectiveness[n] = graph.eta[n]
                        print(f'cf{i, n}: {cf[i, n].x}')
                        print(f'Node {n}: {graph.all_nodes[n - 1]}')
                        print(f'Instance: {graph.sensitive_feature_instances[i - 1]}')
                        print(f'Distance: {np.round(graph.C[i, n], 4)}')
        return sol_x, graph_nodes_solution, likelihood, effectiveness, opt_model.status, obj_val