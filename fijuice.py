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

class FIJUICE:

    def __init__(self, counterfactual):
        self.cluster = counterfactual.cluster
        self.ioi_label = self.cluster.undesired_class
        self.lagrange = counterfactual.lagrange
        self.t = counterfactual.t
        self.k = counterfactual.k
        self.potential_justifiers = self.find_potential_justifiers(counterfactual)
        self.potential_justifiers = self.nn_list(counterfactual)
        start_time = time.time()
        self.normal_x_cf, self.justifiers, self.justifier_ratio = self.Fijuice(counterfactual)
        end_time = time.time()
        self.run_time = end_time - start_time
        self.justifiers = self.transform_dataframe(counterfactual)

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
        for idx in range(len(self.cluster.centroids)):
            c = self.cluster.centroids[idx]
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

    def nn_list(self, counterfactual):
        """
        Method that gets the list of training observations labeled as cf-label with respect to the cf, ordered based on graph nodes size
        """
        permutations_potential_justifiers_all = []
        for c in range(len(self.cluster.centroids)):
            centroid = self.cluster.centroids[c]
            c_justifiers_list = self.potential_justifiers.iloc[c]['justifiers']
            permutations_potential_justifiers = []
            for i in range(len(c_justifiers_list)):
                possible_feat_values_justifier_i = self.get_feat_possible_values(counterfactual.data, obj=[centroid], points=[c_justifiers_list[i]])[0][0]
                len_permutations = len(list(product(*possible_feat_values_justifier_i)))
                permutations_potential_justifiers.append((c_justifiers_list[i], len_permutations))
                # print(f'Justifier {i+1}: Length permutations: {len_permutations}')
            permutations_potential_justifiers.sort(key=lambda x: x[1])
            permutations_potential_justifiers = [i[0] for i in permutations_potential_justifiers]
            if len(permutations_potential_justifiers) > self.k:
                permutations_potential_justifiers = permutations_potential_justifiers[:self.k]
            permutations_potential_justifiers_all.extend(permutations_potential_justifiers)
        return permutations_potential_justifiers_all

    def Fijuice(self, counterfactual):
        """
        Improved JUICE generation method
        """
        print(f'Obtained all potential justifiers: {len(self.potential_justifiers)}')
        self.pot_justifier_feat_possible_values = self.get_feat_possible_values(counterfactual.data)
        print(f'Obtained all possible feature values from potential justifiers')
        self.graph_nodes = self.get_graph_nodes(counterfactual.model)
        self.all_nodes = self.potential_justifiers + self.graph_nodes
        print(f'Obtained all possible nodes in the graph: {len(self.all_nodes)}')
        self.C = self.get_all_costs(counterfactual.data, counterfactual.type)
        print(f'Obtained all costs in the graph')
        self.F = self.get_all_feasibility(counterfactual.data)
        print(f'Obtained all feasibility in the graph')
        self.A = self.get_all_adjacency(counterfactual.data)
        print(f'Obtained adjacency matrix')
        if len(self.potential_justifiers) > 0:
            normal_x_cf, justifiers, justifier_ratio = self.do_optimize_all(counterfactual)
        else:
            print(f'CF cannot be justified. Returning NN counterfactual')
            normal_x_cf, _ = nn_for_juice(counterfactual)
            justifiers = normal_x_cf
            justifier_ratio = 1/len(self.potential_justifiers)
        return normal_x_cf, justifiers, justifier_ratio 

    def continuous_feat_values(self, i, min_val, max_val, data):
        """
        Method that defines how to discretize the continuous features
        """
        sorted_feat_i = list(np.sort(data.transformed_train_np[:,i][(data.transformed_train_np[:,i] >= min_val) & (data.transformed_train_np[:,i] <= max_val)]))
        value = list(np.unique(sorted_feat_i))
        if len(value) <= 100:
            if min_val not in value:
                value = [min_val] + value
            if max_val not in value:
                value = value + [max_val]
            return value
        else:
            mean_val, std_val = np.mean(data.transformed_train_np[:,i]), np.std(data.transformed_train_np[:,i])
            percentiles_range = list(np.linspace(0, 1, 101))
            value = []
            for perc in percentiles_range:
                value.append(norm.ppf(perc, loc=mean_val, scale=std_val))
            value = [val for val in value if val >= min_val and val <= max_val]
            if min_val not in value:
                value = [min_val] + value
            if max_val not in value:
                value = value + [max_val]
        return value

    def get_feat_possible_values(self, data, obj=None, points=None):
        """
        Method that obtains the features possible values
        """
        if obj is None:
            normal_centroids = self.cluster.centroids
        else:
            normal_centroids = obj
        if points is None:
            points = self.potential_justifiers
        else:
            points = points
        pot_justifier_feat_possible_values_all_centroids = {}
        for c_idx in range(len(normal_centroids)):
            pot_justifier_feat_possible_values = {}
            normal_centroid = normal_centroids[c_idx].normal_x
            for k in range(len(points)):
                potential_justifier_k = points[k]
                v = normal_centroid - potential_justifier_k
                nonzero_index = list(np.nonzero(v)[0])
                feat_checked = []
                feat_possible_values = []
                for i in range(len(normal_centroid)):
                    if i not in feat_checked:
                        feat_i = data.processed_features[i]
                        if feat_i in data.bin_enc_cols:
                            if i in nonzero_index:
                                value = [potential_justifier_k[i], normal_centroid[i]]
                            else:
                                value = [potential_justifier_k[i]]
                            feat_checked.extend([i])
                        elif feat_i in data.cat_enc_cols:
                            idx_cat_i = data.idx_cat_cols_dict[feat_i[:-4]]
                            nn_cat_idx = list(potential_justifier_k[idx_cat_i])
                            if any(item in idx_cat_i for item in nonzero_index):
                                ioi_cat_idx = list(normal_centroid[idx_cat_i])
                                value = [nn_cat_idx, ioi_cat_idx]
                            else:
                                value = [nn_cat_idx]
                            feat_checked.extend(idx_cat_i)
                        elif feat_i in data.ordinal:
                            if i in nonzero_index:
                                values_i = list(data.processed_feat_dist[feat_i].keys())
                                max_val_i, min_val_i = max(normal_centroid[i], potential_justifier_k[i]), min(normal_centroid[i], potential_justifier_k[i])
                                value = [j for j in values_i if j <= max_val_i and j >= min_val_i]
                            else:
                                value = [potential_justifier_k[i]]
                            feat_checked.extend([i])
                        elif feat_i in data.continuous:
                            if i in nonzero_index:
                                max_val_i, min_val_i = max(normal_centroid[i], potential_justifier_k[i]), min(normal_centroid[i], potential_justifier_k[i])
                                value = self.continuous_feat_values(i, min_val_i, max_val_i, data)
                            else:
                                value = [potential_justifier_k[i]]
                            feat_checked.extend([i])
                        feat_possible_values.append(value)
                pot_justifier_feat_possible_values[k] = feat_possible_values
            pot_justifier_feat_possible_values_all_centroids[c_idx] = pot_justifier_feat_possible_values
        return pot_justifier_feat_possible_values_all_centroids

    def make_array(self, i):
        """
        Method that transforms a generator instance into array  
        """
        list_i = list(i)
        new_list = []
        for j in list_i:
            if isinstance(j, list):
                new_list.extend([k for k in j])
            else:
                new_list.extend([j])
        return np.array(new_list)
    
    def get_graph_nodes(self, model):
        """
        Generator that contains all the nodes located in the space between the potential justifiers and the normal_ioi (all possible, CF-labeled nodes)
        """
        graph_nodes = []
        for c_idx in range(len(self.cluster.centroids)):
            # print(f'Analyzing centroid {c_idx} for graph nodes...')
            for k in range(len(self.potential_justifiers)):
                print(f'Analyzing centroid {c_idx} and potential justifier {k} for graph nodes...')
                feat_possible_values_k = self.pot_justifier_feat_possible_values[c_idx][k]
                permutations = product(*feat_possible_values_k)
                for i in permutations:
                    perm_i = self.make_array(i)
                    if model.model.predict(perm_i.reshape(1, -1)) != self.ioi_label and \
                        not any(np.array_equal(perm_i, x) for x in graph_nodes) and \
                        not any(np.array_equal(perm_i, x) for x in self.potential_justifiers):
                        graph_nodes.append(perm_i)
        return graph_nodes

    def get_all_costs(self, data, type):
        """
        Method that outputs the cost parameters required for optimization
        """
        C = {}
        for c_idx in range(1, len(self.cluster.centroids) + 1):
            normal_centroid = self.cluster.centroids[c_idx - 1].normal_x
            for k in range(1, len(self.all_nodes) + 1):
                node_k = self.all_nodes[k-1]
                C[c_idx, k] = distance_calculation(normal_centroid, node_k, data, type)
        return C

    def get_all_feasibility(self, data):
        """
        Outputs the counterfactual feasibility parameter for all graph nodes (including the potential justifiers) 
        """
        F = {}
        for c_idx in range(1, len(self.cluster.centroids) + 1):
            normal_centroid = self.cluster.centroids[c_idx - 1].normal_x
            for k in range(1, len(self.all_nodes) + 1):
                node_k = self.all_nodes[k-1]
                F[c_idx, k] = verify_feasibility(normal_centroid, node_k, data)
        return F

    def get_all_adjacency(self, data):
        """
        Method that outputs the adjacency matrix required for optimization
        """
        toler = 0.00001
        centroids_array = np.array([self.cluster.centroids[i].normal_x for i in range(len(self.cluster.centroids))])
        justifiers_array = np.array(self.potential_justifiers)
        A = tuplelist()
        for i in range(1, len(self.all_nodes) + 1):
            node_i = self.all_nodes[i - 1]
            for j in range(i + 1, len(self.all_nodes) + 1):
                node_j = self.all_nodes[j - 1]
                vector_ij = node_j - node_i
                nonzero_index = list(np.nonzero(vector_ij)[0])
                feat_nonzero = [data.processed_features[l] for l in nonzero_index]
                if len(nonzero_index) > 2:
                    continue
                elif len(nonzero_index) == 2:
                    if any(item in data.cat_enc_cols for item in feat_nonzero):
                        A.append((i,j))
                elif len(nonzero_index) == 1:
                    if any(item in data.ordinal for item in feat_nonzero):
                        if np.isclose(np.abs(vector_ij[nonzero_index]), data.feat_step[feat_nonzero], atol=toler).any():
                            A.append((i,j))
                    elif any(item in data.continuous for item in feat_nonzero):
                        max_val, min_val = float(max(max(centroids_array[:,nonzero_index]), max(justifiers_array[:,nonzero_index]))), float(min(min(centroids_array[:,nonzero_index]), min(justifiers_array[:,nonzero_index])))
                        values = self.continuous_feat_values(nonzero_index, min_val, max_val, data)
                        try:
                            value_node_i_idx = int(np.where(np.isclose(values, node_i[nonzero_index]))[0])
                            if value_node_i_idx > 0:
                                value_node_i_idx_inf = value_node_i_idx - 1
                                value_node_i_idx_sup = value_node_i_idx
                            else:
                                value_node_i_idx_inf = value_node_i_idx
                                value_node_i_idx_sup = value_node_i_idx + 1
                            if value_node_i_idx < len(values) - 1:
                                value_node_i_idx_inf = value_node_i_idx
                                value_node_i_idx_sup = value_node_i_idx + 1
                            else:
                                value_node_i_idx_inf = value_node_i_idx -1
                                value_node_i_idx_sup = value_node_i_idx
                        except:
                            if node_i[nonzero_index] < values[0]:
                                value_node_i_idx_inf, value_node_i_idx_sup = 0, 0
                            elif node_i[nonzero_index] > values[-1]:
                                value_node_i_idx_inf, value_node_i_idx_sup = len(values) - 1, len(values) - 1
                            for k in range(len(values) - 1):
                                if node_i[nonzero_index] <= values[k+1] and node_i[nonzero_index] >= values[k]:
                                    value_node_i_idx_inf, value_node_i_idx_sup = k, k+1  
                        close_node_j_values = [values[value_node_i_idx_inf], values[value_node_i_idx_sup]]
                        if any(np.isclose(node_j[nonzero_index], close_node_j_values)):
                            A.append((i,j))
                    elif any(item in data.binary for item in feat_nonzero):
                        if np.isclose(np.abs(vector_ij[nonzero_index]), [0,1], atol=toler).any():
                            A.append((i,j))
        return A

    def do_optimize_all(self, counterfactual):
        """
        Method that finds FiJUICE CF using Gurobi optimization package
        """

        def output_path(node, cf_node, path=[]):
            """
            Prints the connection paths from a justifier towards the found CF
            """
            path.extend([node])
            if cf_node == node:
                return path
            new_node = [j for j in G.successors(node) if edge[node,j].x >= 0.9][0]
            return output_path(new_node, cf_node, path)

        def unfeasible_case(self):
            """
            Obtains the feasible justified solution when the problem is unfeasible
            """
            sol_x, justifiers, centroids_solved, nodes_solution = {}, {}, [], []
            for c_idx in range(1, len(self.cluster.centroids) + 1):
                potential_CF = {}
                for i in range(1, len(self.potential_justifiers) + 1):
                    if self.F[c_idx, i]:
                        potential_CF[c_idx, i] = self.C[c_idx, i]
                        centroids_solved.append(c_idx)
                        if i not in nodes_solution:
                            nodes_solution.append(i)
            for c_idx in centroids_solved:
                centroids_solved_i = dict([(tup, potential_CF[tup]) for tup in list(potential_CF.keys()) if tup[0] == c_idx])
                _, sol_x_idx = min(centroids_solved_i, key=centroids_solved_i.get)
                sol_x[c_idx, sol_x_idx] = self.all_nodes[sol_x_idx - 1]
                justifiers[c_idx, sol_x_idx] = self.all_nodes[sol_x_idx - 1]
                if sol_x_idx not in nodes_solution:
                    nodes_solution.append(sol_x_idx)
            not_centroids_solved = [i for i in range(1, len(self.cluster.centroids) + 1) if i not in centroids_solved]
            for c_idx in not_centroids_solved:
                pot_justifiers = self.find_potential_justifiers(counterfactual, ijuice_search=True)
                cf_instance = pot_justifiers.loc[pot_justifiers.index == c_idx - 1]['justifiers'].values[0][0]
                sol_x_idx = 'close_train_label'
                sol_x[c_idx, sol_x_idx] = cf_instance
                justifiers[c_idx, sol_x_idx] = cf_instance
                nodes_solution.append(sol_x_idx)
            return sol_x, justifiers, nodes_solution       

        if len(self.A) == 0:
            sol_x, justifiers = unfeasible_case(self)
        else:
            """
            MODEL
            """
            opt_model = gp.Model(name='FiJUICE')
            G = nx.DiGraph()
            G.add_edges_from(self.A)
            
            """
            SETS
            """
            set_Centroids = range(1, len(self.cluster.centroids) + 1)
            len_justifiers = len(self.potential_justifiers)
            set_Sources = range(1, len_justifiers + 1)
               
            """
            VARIABLES
            """
            edge = gp.tupledict()
            for (i,j) in G.edges:
                edge[i,j] = opt_model.addVar(vtype=GRB.INTEGER, name='Path')
            cf = opt_model.addVars(set_Centroids, G.nodes, vtype=GRB.BINARY, name='Counterfactual')   # Node chosen as destination
            source = opt_model.addVars(set_Sources, G.nodes, vtype=GRB.BINARY, name='Justifiers')   # Nodes chosen as sources (justifier points)            
            """
            CONSTRAINTS AND OBJECTIVE
            """
            for c in set_Centroids:
                for n in G.nodes:
                    opt_model.addConstr(cf[c, n] <= self.F[c, n])
            
            for n in G.nodes:
                if n in set_Sources:
                    opt_model.addConstr(gp.quicksum(edge[i, n] for i in G.predecessors(n)) - gp.quicksum(edge[n, j] for j in G.successors(n)) == -gp.quicksum(source[n, k] for k in G.nodes)) # Source contraints. A source may justify more than one CF
            
            for c in set_Centroids:
                for n in G.nodes:
                    if n not in set_Sources:
                        opt_model.addConstr(gp.quicksum(edge[i, n] for i in G.predecessors(n)) - gp.quicksum(edge[n, j] for j in G.successors(n)) == cf[c, n]*gp.quicksum(source[s, n] for s in set_Sources)) # Sink constraints
                        # opt_model.addConstr(source[n] == 0)
            
            for c in set_Centroids:
                opt_model.addConstr(gp.quicksum(cf[c, i] for i in G.nodes) == 1)
            
            # opt_model.addConstr(gp.quicksum(source[s, i] for s in set_Sources for i in G.nodes) >= len(self.cluster.centroids))
            # opt_model.addConstr(source.sum() >= len(self.cluster.centroids))
            
            def fairness_objective(x, C, centroids_idx, nodes_idx):
                var = 0
                for c in centroids_idx:
                    for i in nodes_idx:
                        for e in centroids_idx:
                            for j in nodes_idx:
                                if (c, i) != (e, j):
                                    var += (x[c, i]*C[c, i] - x[e, j]*C[e, j])**2
                return var     

            opt_model.setObjective(cf.prod(self.C)*self.lagrange + fairness_objective(cf, self.C, set_Centroids, G.nodes)*(1-self.lagrange), GRB.MINIMIZE)
            
            # list_excluded_nodes = list(np.setdiff1d(set_N, list(G.nodes)))
            # for v in list_excluded_nodes:
            #     for s in set_Sources:
            #         opt_model.addConstr(source[s, v] == 0)
            #     for c in set_Centroids:
            #         opt_model.addConstr(cf[c, v] == 0)

            """
            OPTIMIZATION AND RESULTS
            """
            opt_model.optimize()
            time.sleep(0.25)
            if opt_model.status == 3 or len(self.all_nodes) == len(self.potential_justifiers):
               sol_x, justifiers, nodes_solution = unfeasible_case(self)
            else:
                print(f'Optimizer solution status: {opt_model.status}') # 1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 6: 'CUTOFF', 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED', 12: 'NUMERIC', 13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT'
                print(f'Solution:')
                sol_x, justifiers, nodes_solution = {}, {}, []
                for c in set_Centroids:
                    for i in G.nodes:
                        if cf[c, i].x > 0.1:
                            sol_x[c] = self.all_nodes[i - 1]
                            if i not in nodes_solution:
                                nodes_solution.append(i)
                            print(f'cf{c, i}: {cf[c, i].x}')
                            print(f'Node {i}: {self.all_nodes[i - 1]}')
                            print(f'Centroid: {self.cluster.centroids[c - 1].normal_x}')
                            print(f'Distance: {np.round(self.C[c, i], 3)}')
                for s in set_Sources:
                    for i in nodes_solution:
                        if source[s, i].x > 0.1:
                            justifiers[s, i] = self.all_nodes[i - 1]
                time.sleep(0.25)
                time.sleep(0.25)
                for s, i in justifiers.keys():
                    path = []
                    print(f'Source {s} Path to CF : {output_path(s, i, path=path)}')
                    time.sleep(0.25)
            justifier_ratio = {}
            for i in nodes_solution:
                list_cf_justifier = np.unique([tup[0] for tup in justifiers.keys() if tup[1] == i])
                justifier_ratio[i] = len(list_cf_justifier)/len(self.potential_justifiers)    
                print(f'Justifier Ratio (%) for node {i}: {np.round(justifier_ratio[i]*100, 2)}')
        return sol_x, justifiers, justifier_ratio

    def transform_dataframe(self, counterfactual):
        """
        Transforms the justifiers into dataframe
        """
        justifiers_original = {}
        for s, i in self.justifiers.keys():
            justifier_instance = self.justifiers[s, i]
            justifier_original = counterfactual.data.inverse(justifier_instance)
            justifier_original_df = pd.DataFrame(data=justifier_original, columns=counterfactual.data.features)
            justifiers_original[s, i] = justifier_original_df
        return justifiers_original