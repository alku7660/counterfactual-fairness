"""
Imports
"""
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from evaluator_constructor import verify_feasibility
import time
from graph_constructor_nocluster import Graph

class COUNTERFAIR:

    def __init__(self, counterfactual):
        self.percentage = counterfactual.percentage
        self.feat_protected = counterfactual.data.feat_protected
        self.false_undesired_test_df = counterfactual.data.false_undesired_test_df
        self.continuous_bins = counterfactual.continuous_bins
        self.ioi_label = counterfactual.data.undesired_class
        self.sensitive_feat_idx_dict = self.select_instances_by_sensitive_group()
        self.alpha, self.dev, self.eff = counterfactual.alpha, counterfactual.dev, counterfactual.eff
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
        Generates the solution according to the CounterFair algorithms
        """
        normal_x_cf_dict, graph_nodes_solutions_dict, likelihood_dict, effectiveness_dict, model_status_dict, obj_val_dict, sensitive_group_idx_feat_value_dict = {}, {}, {}, {}, {}, {}, {}
        start_time = time.time()
        for feature in self.feat_protected.keys():
            value_dict = self.feat_protected[feature]
            sensitive_group_dict = self.sensitive_feat_idx_dict[feature]
            graph = Graph(counterfactual.data, counterfactual.model, feature, value_dict.keys(), sensitive_group_dict, counterfactual.type, self.percentage, self.continuous_bins)
            normal_x_cf, graph_nodes_solution, likelihood, effectiveness, model_status, obj_val = self.Counterfair(counterfactual, graph)
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

    def Counterfair(self, counterfactual, graph):
        """
        Counterfair algorithm
        """
        print(f'Solving CounterFair: alpha: {self.alpha}, deviation: {self.dev}, effectiveness: {self.eff}')
        normal_x_cf, graph_nodes_solution, likelihood, effectiveness, model_status, obj_val = self.do_optimize_all(counterfactual, graph)
        return normal_x_cf, graph_nodes_solution, likelihood, effectiveness, model_status, obj_val 

    def do_optimize_all(self, counterfactual, graph):
        """
        Method that finds Foce CF prioritizing likelihood using Gurobi optimization package
        """

        def get_list_set_instances_per_feat_value(set_Instances):
            """
            Obtains a list of indices per feature value
            """
            list_set_instances_per_feat_value = []
            for feat_value in graph.feat_values:
                list_idx_feat_value = []
                for instance_idx in set_Instances:
                    feat_value_instance = graph.sensitive_group_idx_feat_value_dict[graph.instance_idx_to_original_idx_dict[instance_idx]]
                    if feat_value == feat_value_instance:
                        list_idx_feat_value.append(instance_idx)
                list_set_instances_per_feat_value.append(list_idx_feat_value)
            return list_set_instances_per_feat_value

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
                train_cfs, _ = graph.nearest_neighbor_train_cf(data, model, feat_values, type, extra_search=True)
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
        MODEL
        """
        opt_model = gp.Model(name='CounterFair')
        G = nx.DiGraph()
        G.add_nodes_from(graph.rho)

        """
        SETS
        """
        set_Instances = range(1, len(graph.sensitive_feature_instances) + 1)
        list_set_instances_per_feat_value = get_list_set_instances_per_feat_value(set_Instances)               
            
        """
        VARIABLES
        """
        cf = opt_model.addVars(set_Instances, G.nodes, vtype=GRB.BINARY, name='Counterfactual')
        allow = opt_model.addVars(set_Instances, G.nodes, vtype=GRB.BINARY, name='Allowance')
        limiter = opt_model.addVars(G.nodes, vtype=GRB.BINARY, name='Limiter')
        
        # Variables required for deviation minimization
        max_burden = opt_model.addVar(vtype=GRB.CONTINUOUS, name='Max Burden')
        min_burden = opt_model.addVar(vtype=GRB.CONTINUOUS, name='Min Burden')

        """
        CONSTRAINTS AND OBJECTIVE
        """
        for i in set_Instances:
            for n in G.nodes:
                opt_model.addConstr(cf[i, n] <= graph.F[i, n] + allow[i, n])
        
        for i in set_Instances:
            opt_model.addConstr(gp.quicksum(cf[i, n] for n in G.nodes) == 1)
        
        for i in set_Instances:
            for n in G.nodes:
                opt_model.addConstr(cf[i, n] <= limiter[n])
        
        # Constraints required for deviation minimization
        for set_instances_per_feature_value in list_set_instances_per_feat_value:
            opt_model.addConstr(gp.quicksum(cf[i, n]*graph.C[i, n] for i in set_instances_per_feature_value for n in G.nodes) <= max_burden)
            opt_model.addConstr(gp.quicksum(cf[i, n]*graph.C[i, n] for i in set_instances_per_feature_value for n in G.nodes) >= min_burden)
        
        # EXPERIMENT 1: alpha = [1.0, 0.5, 0.1]
        if self.dev == False and self.eff == False: 
            opt_model.setObjective(cf.prod(graph.C)*self.alpha + gp.quicksum(limiter[n] for n in G.nodes)/len(set_Instances)*(1 - self.alpha) + gp.quicksum(allow[i, n] for i in set_Instances for n in G.nodes), GRB.MINIMIZE)
            
        # EXPERIMENT 2: Minimization of burden variance
        elif self.dev == True:
            opt_model.setObjective(max_burden - min_burden + gp.quicksum(allow[i, n] for i in set_Instances for n in G.nodes), GRB.MINIMIZE)

        # EXPERIMENT 3: Maximize effectiveness
        elif self.eff == True:
            opt_model.setObjective(-gp.quicksum(cf[c, i]*graph.eta[i] for i in G.nodes for c in set_Instances) + gp.quicksum(allow[i, n] for i in set_Instances for n in G.nodes), GRB.MINIMIZE)

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
            counter_allowed = 0
            for i in set_Instances:
                time.sleep(0.25)
                for n in G.nodes:
                    if cf[i, n].x > 0.1:
                        sol_x[i] = graph.all_nodes[n - 1]
                        graph_nodes_solution[i] = n
                        likelihood[n] = graph.rho[n]
                        effectiveness[n] = graph.eta[n]
                        if allow[i, n].x > 0.1:
                            counter_allowed += 1
                            print(f'Allowance given to this instance! (Total: {counter_allowed})')
                        print(f'cf{i, n}: {cf[i, n].x}. Distance: {np.round(graph.C[i, n], 3)}')
            print(f'Total allowance: {counter_allowed}')
        return sol_x, graph_nodes_solution, likelihood, effectiveness, opt_model.status, obj_val