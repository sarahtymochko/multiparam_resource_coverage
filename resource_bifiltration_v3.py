# -*- coding: utf-8 -*-
import gudhi as gd
import numpy as np
import bisect
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import useful_functions
import seaborn as sns
import time
import pandas as pd
import geopandas as gpd

'''
CURRENT STATUS: Seems to work on our park bifiltration. (At least, it runs and seems to make sense.)

Recent update: Fixed the horizontal slice (fixed q) filtration code to correctly handle the case where q = Q_{j, k} = f(B_j, M_{jk}) for some k, where M_{jk} is the kth nearest site to block B_j

If you have G, scores, and D as defined below, use this code snippet:
    
    rb = ResourceBifiltration(scores, D, G)
    
to create a ResourceBifiltration object that stores all the data. 

- The class has (will have) functionality for computing vertical/horizontal slices of the bifiltration and also computing f(B_j, r).
- Designed to be somewhat efficient/organized if you're doing lots of slices or lots of f(B_j, r) calculations
- For example, to compute the ph of a vertical slice, you'd use the snippet:
    
    ph = rb.compute_ph_vertical(r)

- or for a horizontal slice:
    
    ph = rb.compute_ph_horizontal(q)
    
- to calculate f(B_j, r) for given j and r:
    
    rb.f(j, r)
    
- You can make a ResourceBifiltration object with just scores and D if all you want to do is calculate block scores f(B_j, r)
- Input additional arrays NN (this is equivalent to Gill's park_order.npy filed), Q, and/or D_sorted (defined below) if they've already been calculated

'''

class TrackedComponent:
    def __init__(self, init_C, init_r, init_q, rb, rs, qs, len_thresh, thresh_type):
        '''
        init_C : a set of blocks
        init_r : (float) initial r
        init_q : (float) initial q
        rb : (ResourceBifiltration object)
        rs : (1D np.array) discrete r filtration values
        qs : (1D np.array) discrete q filtration values
        len_thresh : max component size (integer) OR percentage of total area (float) ... at which to consider a component "maximal" . We stop tracking at that point.
        thresh_type: "areaperc" is a percentage of the total area and "count" is the max number of blocks 
        '''
        self.init_params = {(init_r, init_q)}
        self.rb = rb
        self.max_rep = set()
        self.C_reps = { (init_r, init_q) : init_C }
        self.term_coords = []
        
        # # assumes that rs is either sorted in ascending or descending order. 
        if rs[-1] > rs[0]:
            self.rorder = 'increasing'
        else:
            self.rorder = 'decreasing'

        if qs[-1] > qs[0]:
            self.qorder = 'increasing'
        else:
            self.qorder = 'decreasing'
        #     rs = rs[::-1]
        self.rs = rs
        self.qs = qs
        self.init_C = init_C
        self.b_reps = [next(iter(init_C))]
        self.thresh_type = thresh_type

        # self.len_thres = len_thresh                
            
        
        C_new = init_C
        r_maxed = False
        maxed_threshold = False
        for r in rs:
            for i, q in enumerate(qs):
                C_prev = C_new
                C_new = rb.get_component(init_C, r, q)

                if self.thresh_type == 'areaperc':
                    if compute_area_frac(C_new, rb.area_dict, rb.total_area) > len_thresh:
                        maxed_threshold = True

                elif self.thresh_type == 'count':
                    if len(C_new) > len_thresh:
                        maxed_threshold = True

                if maxed_threshold:
                    if i == 0: 
                        r_maxed = True
                    else:
                        self.term_coords.append((r, qs[i-1])) # the previous coordinate where C_{r, q-1} wasn't too big yet
                    break
                else:
                    self.C_reps[(r, q)] = C_new
                if q == max(qs):
                    self.term_coords.append((r, q))
            if r_maxed:
                break

        self.area = compute_area(C_new, rb.area_dict)
                
    def plot_components(self, blocks, r_vals, q_vals):
        fig, axs = plt.subplots(len(r_vals), len(q_vals))
        fig.dpi = 500
        for i, r in enumerate(r_vals):
            for j, q in enumerate(q_vals):
                if (r, q) in self.C_reps:
                    C = self.C_reps[(r, q)]
                    axs[i, j].set_title(f"r = {r}, q = {q}\nlength = {len(C)}", fontsize = 3)
                    blocks[blocks.index.isin(C)].plot(ax = axs[i, j])
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
        fig.tight_layout(pad = .005)

    def plot_components_v2(self, blocks, r_vals, q_vals):
        '''
        this version of plotting makes sure component doesnt get smaller as q increases
        by just plotting the union of C_reps([r,q]) across all qvals
        '''

        C_reps_plotting = self.C_reps
        for i, r in enumerate(r_vals):
            minq_ind = -1
            for j, q in enumerate(q_vals): 
                if (r, q) in C_reps_plotting:
                    if minq_ind < 0:
                        minq_ind = j
                    else:
                        lst = [C_reps_plotting[(r,newq)] if (r,newq) in C_reps_plotting.keys() else {} for newq in q_vals[minq_ind:j+1] ]
                        
                        C_reps_plotting[(r,q)] = set.union(*lst)
                    
        
        fig, axs = plt.subplots(len(r_vals), len(q_vals))
        fig.dpi = 500
        # print(r_vals)
        for i, r in enumerate(r_vals):
            firstqfound = False
            for j, q in enumerate(q_vals):                    
                if (r, q) in C_reps_plotting:
                    
                    C = C_reps_plotting[(r, q)]

                    if self.thresh_type == 'count':
                        axs[i, j].set_title(f"r = {r}, q = {q}\nlength = {len(C)}", fontsize = 3) 
                    else:
                        area = round(compute_area_frac(C, self.rb.area_dict, self.rb.total_area),4)
                        axs[i, j].set_title(f"r = {r}, q = {q}\n area(frac) = {area}", fontsize = 3) 
                    blocks[blocks.index.isin(C)].plot(ax = axs[i, j])
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
        fig.tight_layout(pad = .005)
        
    def plot_max_rep(self, blocks):
        '''
        blocks : geopandas dataframe
        '''

        all_sets = list(self.C_reps.values())
        self.max_rep = set.union(*all_sets)
        
        blocks[blocks.index.isin(self.max_rep)].plot()
        
    def plot_range(self):
        A = np.array([[1 if (r, q) in self.C_reps else 0 for q in self.qs ] for r in self.rs])
        sns.heatmap(A, xticklabels = self.qs, yticklabels = self.rs)
    
    def average_function_values(self):
        block_sizes = self.block_range_sizes()
        block_avg_fcn_vals = dict()
        for B in self.max_rep:
            avg_val = 0
            for (r, q) in self.C_reps:
                if B in self.C_reps[(r, q)]:
                    val = self.rb.f(B, r)
                    avg_val += val/(len(self.rs)*len(self.qs))
            avg_val /= block_sizes[B]
            block_avg_fcn_vals[B] = avg_val
        return block_avg_fcn_vals
        
    def block_range_sizes(self):
        block_sizes = dict()
        for B in self.max_rep:
            size = 0
            for (r, q) in self.C_reps:
                if B in self.C_reps[(r, q)]:
                    size += 1/(len(self.rs)*len(self.qs))
            block_sizes[B] = size
        return block_sizes
        
    def total_range_size(self):
        size = 0
        for (r, q) in self.C_reps:
            size += 1/(len(self.rs)*len(self.qs))
        return size


class TrackedComponentCollection:
    def __init__(self, rb, rs, qs, len_thresh, blocks, run_auto = True, get_max_rep = False, thresh_type='count'): #, threshfrac=None):
        self.tracked_comps = []
        self.blocks = blocks
        self.rb = rb
        self.qs = qs
        if thresh_type == 'areaperc':
            if len_thresh > 1:
                print('needs to be leq 1')
                return 
            # else:
            #     areafrac = threshfrac*(blocks['geometry'].to_crs('EPSG:26916').area.sum())
            #     len_thresh = areafrac
        
        self.thresh_type = thresh_type
        self.len_thresh = len_thresh

        
        # assumes that rs is either sorted in ascending or descending order. 
        # Warning to user: make sure rs is sorted properly for the particular application! 
        self.rs = rs

        if rs[-1] > rs[0]:
            self.rorder = 'increasing'
        else:
            self.rorder = 'decreasing'

        if qs[-1] > qs[0]:
            self.qorder = 'increasing'
        else:
            self.qorder = 'decreasing'
            
        
        t0 = time.time()
        
        if run_auto:
            for r in rs:   
                for q in qs:
                    print(f"Getting the subgraph at r = {r} and q = {q}...")
                    H = rb.get_subgraph(r, q)
                    print("Getting the subgraph components...")
                    comps = nx.connected_components(H)
                    
                    if thresh_type == 'count':
                        comps = [C for C in comps if len(C) <= len_thresh]
                    else:
                        comps = [C for C in comps if compute_area_frac(C, rb.area_dict, rb.total_area) <= len_thresh]
                        
                    print(f"Checking {len(comps)} components below threshold...")
                    for i, C in enumerate(comps):
                        result = self._add_component(C, r, q)
                        print(f"Component {i}: ", result)
                    print("")
            T = time.time()
            print(T - t0)

        # if get_max_rep:
        #     self.compute_all_max_rep()

    # def compute_all_max_rep(self):
    #     for TC in TCC.tracked_comps():
    #         TC.max_rep = set.union([TC.C_rep[param] for param in TC.term_coords])
            
    
    def _add_component(self, init_C, init_r, init_q):
        already_tracked = False
        for TC in self.tracked_comps:
            if already_tracked: return "previously tracked"
            for b in TC.b_reps:
                if b in init_C:
                    already_tracked = True
                    break
       
        # If we get to this point, then we make a new tracked component, then add it and check to see if it's a duplicate
        if self.rorder == 'decreasing' and self.qorder == 'increasing': #(self.rs[-1] < self.rs[0]) and (self.qs[-1] > self.qs[0]): # if r is decreasing and q is increasing (parks, pubs good)
            TC = TrackedComponent(init_C, init_r, init_q, self.rb, self.rs[np.where(self.rs <= init_r)], self.qs[np.where(self.qs >= init_q)], self.len_thresh, self.thresh_type)     

        elif self.rorder=='increasing' and self.qorder=='increasing': # if r and q are both increasing (landfills, pubs bad)
            print("in the nuisance case!")
            TC = TrackedComponent(init_C, init_r, init_q, self.rb, self.rs[np.where(self.rs >= init_r)], self.qs[np.where(self.qs >= init_q)], self.len_thresh, self.thresh_type)  

        else:
            return "uh on there's a problem!!!"
            
        self.add_tracked_comp(TC)
        return "just tracked it and checked if it's a duplicate"

    
    def add_tracked_comp(self, TC):
        # check if it's the same as any of the others
        duplicates = []
        for i, TC_other in enumerate(self.tracked_comps):
            if TrackedComponentCollection.is_merge_condition(TC, TC_other):
                duplicates.append(i)
        if len(duplicates) == 0:
            self.tracked_comps.append(TC)
        else:
            print("merging the new tracked component with one(s) we already have")
            i = duplicates[0]
            self.tracked_comps[i] = self.get_merged_tracked_comp(self.tracked_comps[i], TC)
            for j in duplicates[1:]:
                self.tracked_comps[i] = self.get_merged_tracked_comp(self.tracked_comps[i], self.tracked_comps[j])
            # Remove the extra duplicates, going backwards, starting from 1st (not 0th) entry of duplicates
            for j in duplicates[1:][::-1]:
                del self.tracked_comps[j]

    
    def get_merged_tracked_comp(self, TC1, TC2):
        # Merge b_reps
        new_elems = list(set(TC2.b_reps) - set(TC1.b_reps))
        TC1.b_reps = TC1.b_reps + new_elems

        # Merge birth parameters

        TC1.init_params = TC1.init_params.union(TC2.init_params) # This might lead to some redudant "initial" parameters, but I think it's ok for the purposes it will be used for
        
        # Merge max rep
        TC1.max_rep = TC1.max_rep.union(TC2.max_rep)
        
        # Merge component reps
        for r in self.rs:
            for q in self.qs:
                if (r, q) in TC1.C_reps and (r, q) in TC2.C_reps:
                    TC1.C_reps[(r, q)] = TC1.C_reps[(r, q)].union(TC2.C_reps[(r, q)])
                
                # this should never get used when using the strict criteria of ALL terminal components have to match
                # but it is needed in the postprocessing since things can get smaller due to the less strict
                # merge criteria for the post processing
                elif (r, q) in TC2.C_reps:
                    TC1.C_reps[(r, q)] = TC2.C_reps[(r, q)]
                    
        TC1.area = TC1.area+TC2.area
        
        return TC1

    
    ### appears not to be used
    def _is_param_in_domain(par, init_params):
        '''
        par : (r, q) tuple
        init_params : set of (r, q) tuples, birth parameters for some TrackedComponent.
        '''
        for r0, q0 in init_params:
            if par[0] <= r0 and par[1] >= q0:
                return True
        return False

    

    def is_merge_condition(TC1, TC2):
        '''
        TC1, TC2: TrackedComponent objects.
        
        Returns
        -------
        True if TC1[r, q] == TC2[r, q] for ALL terminal coordiantes (r, q)
        '''

        # Merge iff EVERY "terminal component" is the same
        
        # if the coordinates of the terminal components dont match return false
        if TC1.term_coords != TC2.term_coords:
            return False

        # if there are no terminal components return false
        if len(TC1.term_coords) == 0:
            return False

        # check if all terminal coordinates match
        for param in TC1.term_coords:
            TC2_param_rep = next(iter(TC2.C_reps[param])) # grab a block from TC2 component at param
            
            if not TC2_param_rep in TC1.C_reps[param]:
                return False

        return True
            

        
    def plot_block_avg_fcn_values(self):
        cmap = plt.cm.get_cmap('viridis')
        plt.rcParams["figure.dpi"] = 500
        fig, ax = plt.subplots()
        
        blocks = self.blocks
        all_TC_blocks = pd.concat([blocks[blocks.index.isin(TC.max_rep)].copy() for TC in self.tracked_comps])
        block_vals = dict()
        for TC in self.tracked_comps:
            block_vals = {**block_vals, **TC.average_function_values()}
        all_TC_blocks["value"] = [block_vals[i] for i, _ in all_TC_blocks.iterrows()]
        all_TC_blocks.plot(ax = ax, column = "value", legend = True)
        
    def plot_block_range_sizes(self):
        cmap = plt.cm.get_cmap('viridis')
        plt.rcParams["figure.dpi"] = 500
        fig, ax = plt.subplots()
        
        blocks = self.blocks
        all_TC_blocks = pd.concat([blocks[blocks.index.isin(TC.max_rep)].copy() for TC in self.tracked_comps])
        block_sizes = dict()
        for TC in self.tracked_comps:
            block_sizes = {**block_sizes, **TC.block_range_sizes()}
            TC_blocks = blocks[blocks.index.isin(TC.max_rep)].copy()
        all_TC_blocks["size"] = [block_sizes[i] for i, _ in all_TC_blocks.iterrows()]
        all_TC_blocks.plot(ax = ax, column = "size", legend = True)
        
    def plot_total_range_sizes(self, relative_color = True):
        '''
        relative_color means minimum range gets mapped to purple, maximum range gets mapped to yellow
        otherwise, just plot as usual on scale from 0 to 1
        '''
        cmap = plt.cm.get_cmap('viridis')
        blocks = self.blocks
        if relative_color:
            max_range_size = 0
            min_range_size = np.inf
            for TC in self.tracked_comps:
                trs = TC.total_range_size()
                if trs > max_range_size:
                    max_range_size = trs
                if trs < min_range_size:
                    min_range_size = trs
        
        plt.rcParams["figure.dpi"] = 500
        fig, ax = plt.subplots()
        for TC in self.tracked_comps:
            if relative_color:
                blocks[blocks.index.isin(TC.max_rep)].plot(ax = ax, color = cmap((TC.total_range_size()-min_range_size)/(max_range_size- min_range_size)))
            else:
                 blocks[blocks.index.isin(TC.max_rep)].plot(ax = ax, color = cmap(TC.total_range_size()), legend = True)


def compute_areas_from_G(G):
        area_dict = {}
        for node in G.nodes(data=True):
            area_dict[node[0]] = node[1]['geometry'].area

        return area_dict

def compute_area(blocknums, area_dict):
    return sum([area_dict[i] for i in blocknums])

def compute_area_frac(blocknums, area_dict, total_area):
    val = compute_area(blocknums, area_dict)/total_area
    if val > 1:
        print('problem')
    
    return val


def postprocess_merge(all_TCs):
    
    delete_inds = []
    new_TCs = []
    for i, TC1 in enumerate(all_TCs): #enumerate(all_TCs):
        
        all_sets1 = list(TC1.C_reps.values())
        TC1_max_rep = set.union(*all_sets1)
            
        for j, TC2 in enumerate(all_TCs):
            if i!=j:
                all_sets2 = list(TC2.C_reps.values())
                TC2_max_rep = set.union(*all_sets2)
                
                if TC2_max_rep.issubset(TC1_max_rep):
                    delete_inds.append(j)
                    TC1.C_reps.update(TC2.C_reps)

    if len(delete_inds) > len(set(delete_inds)):
        print('multiple subsets occuring')
        print(delete_inds)
    return [all_TCs[i] for i in range(len(all_TCs)) if i not in delete_inds]


class ResourceBifiltration:
    def __init__(self, scores, D, G = None, NN = None, Q = None, D_sorted = None):
        '''
        Parameters
        ----------
        scores : list or 1D np array (or array-like object) that stores the score for each park. the ith score corresponds to the ith park
        D : 2D np array where D[j, i] is the distance from the jth census block to the ith park.
        G : (optional) networkx graph (undirected, unweighted). Required for bifiltration slices, not required for 
            Represents the adjacency graph for the census blocks. Nodes are census blocks, edges connected adjacent blocks
        NN : (optional) 2D np array of ints, same shape as D (blocks x sites)
             NN[j, k] is the index of the kth nearest site to block B_j
        Q : (optional) 2D np array, same shape as D.
            Q[j, k] = f(B_j, d(B_j, M_{jk})) where M_{jk} is the kth nearest park to B_j.
            Q[j, 0], ..., Q[j, k], ... is a monotonically increasing sequence (the image of f(B_j, \cdot) as a function of scale r)
        D_sorted : (optional) 2D np array, same shape as D.
                    (j, k)th element is the distance from jth block to its kth nearest site
        '''
        self.G = G
        self.scores = scores
        self.D = D
        self.num_blocks = np.shape(self.D)[0]
        self.num_sites = np.shape(self.D)[1]
        self.Q = Q
        self.NN = NN
        self.D_sorted = D_sorted
        
        self.st = None # simplex tree
        self.horizontal = None # The type of slice used to create st. (True or False)
        self.fixed_param = None # the fixed r or q used to slice, in creating current st.

        self.area_dict = compute_areas_from_G(G)
        self.total_area = sum(self.area_dict.values())
            
        
    def compute_block_filtration_array(self):
        '''
        Returns
        -------
        Q : 2D np array s.t. Q[j, k] = f(B_j, d(B_j, M_{jk})) where M_{jk} is the kth nearest park to B_j.
            Q[j, 0], ..., Q[j, k], ... is a monotonically increasing sequence (the image of f(B_j, \cdot) as a function of scale r)
        '''
        NN = self.get_NN()
        Q = np.zeros((self.num_blocks, self.num_sites))
        Q[:, 0] = np.array([self.scores[NN[j, 0]] for j in range(self.num_blocks)])
        for k in range(1, self.num_sites):
            Q[:, k] = Q[:, k-1] + np.array([self.scores[NN[j, k]] for j in range(self.num_blocks)])
        return Q
    
    def compute_D_sorted(self):
        '''
        Returns
        -------
        D_sorted : 2D np array
            (j, k)th element is the distance from jth block to its kth nearest site

        '''
        NN = self.get_NN()
        D_sorted = np.zeros(np.shape(self.D))
        for j in range(self.num_blocks):
            Dj = self.D[j, :]
            D_sorted[j, :] = Dj[NN[j, :]]
        return D_sorted

    def compute_graph_data_filtration_horizontal(self, q):
        # FLIPS THE SIGN ON r
        assert q >= 0
        Q = self.get_Q()
        assert q <= Q[0, -1]
        
        D_sorted = self.get_D_sorted()
        
        node_values = {j : -D_sorted[j, bisect.bisect_right(Q[j, :], q)] for j in self.G.nodes()}
        nx.set_node_attributes(self.G, node_values, "score")
        edge_values = {(u, v) : max(node_values[u], node_values[v]) for u, v in self.G.edges()}    
        nx.set_edge_attributes(self.G, edge_values, "score")
        return self.G
        
    def compute_graph_data_filtration_vertical(self, r):
        node_values = { j : self.f(j, r) for j in self.G.nodes()}
        nx.set_node_attributes(self.G, node_values, "score")
        edge_values = {(u, v) : max(node_values[u], node_values[v]) for u, v in self.G.edges()}    
        nx.set_edge_attributes(self.G, edge_values, "score")
        return self.G
        
    def compute_NN_array(self):
        '''
        Returns
        -------
        NN : 2D np array such that NN[j, k] is the index of the kth nearest site to block B_j
        '''
        print("Computing nearest neighbor array")
        NN = np.zeros((self.num_blocks, self.num_sites), dtype =  np.int64)
        for j in range(self.num_blocks):
            NN[j, :] = np.argsort(self.D[j, :])
        return NN
    
    def compute_ph_horizontal(self, q):
        '''
        Parameters
        ----------
        q : float.
        
        Returns
        -------
        ph : list of pairs(dimension, pair(birth, death))- The persistence of a horizontal slice of the bifiltration at fixed q
            Note that r is DECREASING in the filtration, so deaths are smaller than births.
        '''
        assert self.G is not None, "Must add adjacency graph G to ResourceBifiltration object"
        
        self.horizontal = True
        self.fixed_param = q
        
        st = gd.SimplexTree()
        Q = self.get_Q()
        
        # If q >= Q[j, -1], then block B_j is always in the filtration. Note that Q[j, -1] is the same for all j, so if q>= Q[0, -1] then q>= Q[j, -1] for all j. 
        # Therefore K_{q, r} = entire graph for all r.
        if q > Q[0, -1]:
            print("Only returning 0D ph")
            num_components = nx.number_connected_components(self.G)
            ph = [(0, (np.inf, -np.inf)) for i in range(num_components)]

        # If q < 0, then K_rq is empty the entire time because f(B_j, r) >= 0 > q for all r.
        elif q < 0:
            ph = []
            
        else:
            D_sorted = self.get_D_sorted()
            for j in self.G.nodes():
                k = bisect.bisect_right(Q[j, :], q) # min{k s.t. Q_{jk} > q}
                st.insert([j], -D_sorted[j, k])
            for u, v in self.G.edges():
                st.insert([u, v], max(st.filtration([u]), st.filtration([v])))
            '''
            for spx in st.get_simplices():
                if len(spx[0]) == 1:
                    print(spx)
            '''
                    
            ph = st.persistence()
            
            # Flip around the sign of the (birth, death) pairs so that they're positive
            ph = [(x[0], (-x[1][0], -x[1][1])) for x in ph]
            
            self.st = st
            
        return ph
    
    def compute_ph_vertical(self, r):
        '''
        Parameters
        ----------
        r : float
        
        Returns
        -------
        ph : list of pairs(dimension, pair(birth, death))- The persistence of a vertical slice of the bifiltration at fixed r
        '''
        assert self.G is not None, "Must add adjacency graph G to ResourceBifiltration object"
        
        self.horizontal = False
        self.fixed_param = r
        
        st = gd.SimplexTree()
        for j in self.G.nodes():
            st.insert([j], self.f(j, r))
        for u, v in self.G.edges():
            st.insert([u, v], max(st.filtration([u]), st.filtration([v])))
        
        '''
        for spx in st.get_simplices():
            if len(spx[0]) == 1:
                print(spx)
        '''
                
        ph = st.persistence()
        self.st = st
        return ph
    
    def get_component(self, C, r, q):
        '''
        Parameters
        ----------
        C : set of ints. Blocks that are connected in the G_{rq} subgraph.
            Code does NOT check that C is actually connected within G_{rq}.
            Typical usage: C is a connected component is G_{r', q'} for some r' >= r, q' <= q.
        
        Returns
        -------
        C_new : set of ints. The full set of blocks in the connected component of G_{rq} that contains C
        '''
        H = self.get_subgraph(r, q)
        b = next(iter(C)) # take any element of C
        C_new = nx.node_connected_component(H, b)
        return C_new

        
    def get_parks_within_r(self, j, r):
        '''
        Parameters
        ----------
        j : integer. Index of a block.
        r : float. Radius value
        
        Returns
        -------
        parks_within_r : List of ints. Indices of the parks that are within r of block j.
        '''
        NN = self.get_NN()
        D_sorted = self.get_D_sorted()
        k = bisect.bisect_right(D_sorted[j, :], r)
        parks_within_r = NN[j, :k]
        return parks_within_r
    
    def get_D_sorted(self):
        '''
        Returns
        -------
        D_sorted : 2D np array
            (j, k)th element is the distance from jth block to its kth nearest site

        '''
        if self.D_sorted is None:
            self.D_sorted = self.compute_D_sorted()
        return self.D_sorted
        
    def get_NN(self):
        '''
        Returns
        -------
        NN : 2D np array such that NN[j, k] is the index of the kth nearest site to block B_j
        '''
        if self.NN is None:
            self.NN = self.compute_NN_array()
        return self.NN
    
    def get_persistence_pairs_horizontal(self, q):
        '''

        Parameters
        ----------
        q : float

        Returns
        -------
        pairs : list of pairs of integer lists
            The pairs are the (birth, death) simplices. Each simplex is a list of integers (indices of the vertices)
        '''
        if self.st is None or not self.horizontal or abs(self.fixed_param - q) > 10**(-10):
            self.compute_ph_horizontal(q)
        pairs = self.st.persistence_pairs()
        return pairs
            
    def get_persistence_pairs_vertical(self, r):
        '''
        Parameters
        ----------
        r : float

        Returns
        -------
        pairs : list of pairs of integer lists
            The pairs are the (birth, death) simplices. Each simplex is a list of integers (indices of the vertices)
        '''
        if self.st is None or self.horizontal or abs(self.fixed_param - r) > 10**(-10):
            self.compute_ph_vertical(r)
        pairs = self.st.persistence_pairs()
        return pairs
    
    def get_ph_horizontal(self, q):
        if self.st is None or not self.horizontal or abs(self.fixed_param - q) > 10**(-10):
            self.compute_ph_horizontal(q)
        ph = self.st.persistence()
        return ph
    
    def get_ph_vertical(self, r):
        if self.st is None or self.horizontal or abs(self.fixed_param - r) > 10**(-10):
            self.compute_ph_vertical(r)
        ph = self.st.persistence()
        return ph
    
    def get_subgraph(self, r, q):
        '''
        Returns
        -------
        H : subgraph of self.G s.t. f(B, r) <= q for all nodes/blocks B of H
        '''
        subgraph_nodes = [j for j in self.G.nodes() if self.f(j, r) <= q]
        # print(len(subgraph_nodes))
        H = self.G.subgraph(subgraph_nodes)
        return H
        
    def get_Q(self):
        '''
        Returns
        -------
        Q : 2D np array s.t. Q[j, k] = f(B_j, d(B_j, M_{jk})) where M_{jk} is the kth nearest park to B_j.
            Q[j, 0], ..., Q[j, k], ... is a monotonically increasing sequence (the image of f(B_j, \cdot) as a function of scale r)
        '''
        if self.Q is None:
            self.Q = self.compute_block_filtration_array()
        return self.Q
   
    
    def f(self, j, r):
        '''
        Parameters
        ----------
        j : (int) index of a census block
        r : (float)
        
        Returns
        -------
        f(B_j, r) as defined in overleaf (sum of park scores for parks within r of B_j)
        '''
        D_sorted = self.get_D_sorted()
        if D_sorted[j, 0] > r:
            return 0 # there are no sites within r of B_j, so f(B_j, r) = 0
        
        Q = self.get_Q()
        
        # Get k = max{k s.t. d(B_j, M_{jk}) <= r}, where M_{jk} = kth nearest site to B_j
        k = bisect.bisect_right(D_sorted[j, :], r) - 1
        return Q[j, k]
    
    def match_cycles(self, param1, param2, qsublevel = True, plot_dgms_prematching = True, overlap_perc = .9, eps = .001):
        '''
        param1, param2: floats. The filtration parameters
        overlap_perc: float in [0, 1]. How much the representatives have to overlap in order to be considered a potential match.
        eps: How close (b1, d1) and (b2, d2) have to be in order to consider them the "same" point in the PD. (Makes it easier to visualize.)
        '''
        pdg1, block_list1 = ResourceBifiltration.run_union_find(self, param1, qsublevel)
        pdg2, block_list2 = ResourceBifiltration.run_union_find(self, param2, qsublevel)
        if plot_dgms_prematching:
            fig,ax = plt.subplots(1,1, figsize=(5,5), sharey=True)
            rc("text", usetex = True)
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["text.usetex"] = True
            gd.plot_persistence_diagram(pdg1, axes = ax, colormap = 'r', legend = False, alpha = 0.3)
            gd.plot_persistence_diagram(pdg2, axes = ax, colormap = 'b', legend = False, alpha = 0.3)
            plt.show()
            
        matches = ResourceBifiltration.match_cycle_candidates(block_list1, block_list2, overlap_perc)
        
        # Below is directly copy/pasted from Sarah's code, except for 
        # (1) having some wiggle room eps to consider points to be at the same spot.
        # (2) plots ALL points in the persistence diagram, regardless of whether they're matched or not.
        # TO DO: solve assignment problem to get "best" matching
        EDGES = []
        scatter_match = [] # points in the two PDs that are at the same spot (or sufficiently close, within eps)
        scatter_1, scatter_2 = [], [] # all the other points in the two PDs
        lws = [] # linewidth of connection between persistence points based on overlap percentage
        #s1, s2 = [], [] # size of persistence points based on # of blocks in cluster
        
        # loop over keys, then for each match for a given key, add the corresponding point(s) to the lists 
        is_matched = [False for a in block_list2] # matched_indices_2[j] = True if we've found a match for the jth element of blocklist_2
        for i, i_matches in enumerate(matches):
            b1, d1 = pdg1[i][1][0], pdg1[i][1][1] # pd[i] = [0, [b, d]] where 0 = homology dimension
            if len(i_matches) == 0:
                scatter_1.append([b1, d1])
                a = set(block_list1[i])
                #s1.append(len(a))
            for j in i_matches:
                is_matched[j] = True
                b2, d2 = pdg2[j][1][0], pdg2[j][1][1]
        
                # persistence points that are matched where they have same b,d in both PDs
                if abs(b1 - b2) < eps and abs(d1 - d2) < eps:
                    scatter_match.append([b2, d2])
                
                else:
                    # persistence points in PD1 that match with points in PD2 but have dif b,d
                    # and vice versa
                    scatter_1.append([b1, d1])
                    scatter_2.append([b2, d2])
        
                    # list of pairs of matches to plot edges
                    EDGES.append([[b1, d1], [b2, d2]])
        
                    a = set(block_list1[i])
                    b = set(block_list2[j])
        
                    #s1.append(len(a))
                    #s2.append(len(b))
        
                    #lws.append(len(a.intersection(b))) # / len( a.union(b) ) )
                    lws.append(len(set(a).intersection(set(b))) / len(set(a).union(set(b))))
        for j in range(len(is_matched)):
            if is_matched[j]:
                b2, d2 = pdg2[j][1][0], pdg2[j][1][1]
                scatter_2.append([b2, d2])
                b = set(block_list2[j])
                #s2.append(len(b))
            
        scatter_match = np.array(scatter_match)
        scatter_1 = np.array(scatter_1)
        scatter_2 = np.array(scatter_2)
        #s1 = np.array(s1)*20
        #s2 = np.array(s2)*20
        print(lws)
        lws = np.array(lws)
        #ResourceBifiltration.plot_matches(scatter_match, scatter_1, scatter_2, EDGES, s1, s2, lws)
        ResourceBifiltration.plot_matches(scatter_match, scatter_1, scatter_2, EDGES, lws)
        
    def match_cycle_candidates(block_list1, block_list2, overlap_perc = .9):
        '''
        Returns match_candidates: List where matchdict[i] is the list of indices j in 
                            block_list2 such that block_list[i] and block_list[j]
                            "math" according to the criteria that the overlap 
                            percentage of blocks is >= overlap_perc
        '''
        # Returns dictionary where the key i is the block index in block_list1 
        # and matchdict[i] is the list of indices of blocks in block_list2 that 
        # match with i. The matching function is symmetric so it doesn't matter which one plays the role of block_list1
        match_candidates = [[] for a in block_list1]
        
        for i, a in enumerate(block_list1):
            for j, b in enumerate(block_list2):
                
                # if the blocks in a are a subset of blocks in b or vice versa
                # note this should be symmetric so which is block_list1 vs 2 doesnt matter
                if set(a) <= set(b) or set(b) <= set(a):
        
                    # calculate the size of the intersection relative to the union
                    ratio = len(set(a).intersection(set(b))) / len(set(a).union(set(b)))
                    
                    # if they overlap more than overlap_perc%, match
                    if ratio >= overlap_perc:
                        match_candidates[i].append(j) # Add block j to the list of blocks matching with block i
        return match_candidates
        
    def max_f(self):
        Q = self.get_Q()
        # Every block has the same max value because for large enough r, we have access to every park. 0 below is arbitrary
        return Q[0, -1]
    
    def plot_matches(pd_matches, pd1_remaining, pd2_remaining, edges, lws):
        # plot them! (Copied from Sarah's code)
        fig, ax = plt.subplots(1,1, figsize=(7,7))
        
        # exact matches
        if len(pd_matches)>0:
            ax.scatter(pd_matches[:,0], pd_matches[:,1], marker = 'd', c = 'C2', alpha = 0.7, s = 50, label = 'Shared point')
        
        
        #ax.scatter(pd1_remaining[:,0], pd1_remaining[:,1], c = 'C0', alpha = 0.7, s = s1, label = 'PD for smaller radius')
        #ax.scatter(pd2_remaining[:,0], pd2_remaining[:,1], marker = 'v', c = 'C1', alpha = 0.7, s = s2, label = 'PD for larger radius')
        ax.scatter(pd1_remaining[:,0], pd1_remaining[:,1], c = 'C0', alpha = 0.7, label = 'PD for smaller radius')
        ax.scatter(pd2_remaining[:,0], pd2_remaining[:,1], marker = 'v', c = 'C1', alpha = 0.7, label = 'PD for larger radius')
        
        coll = matplotlib.collections.LineCollection(edges, linewidth = lws, alpha = 0.7, 
                                                     linestyle = 'dashed', color = 'k', 
                                                     label = 'matches', 
                                                     zorder = -10)
        ax.add_collection(coll)
        
        ax.grid()
        ax.grid()
        
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        
        ax.legend()

    def run_union_find(self, param, qsublevel = True):
        if qsublevel:
            G = self.compute_graph_data_filtration_vertical(param)
        else:
            G = self.compute_graph_data_filtration_horizontal(param)
        birth, death, _, block_list = useful_functions.union_find(G)
        pd = np.zeros([len(birth), 2])
        pd[:, 0] = birth
        pd[:, 1] = death
        pdg = [[0, p] for p in pd] # gudhi format, 0 represents homology dimension
        return pdg, block_list