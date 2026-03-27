'''
file: useful_functions.py
date edited: july 22, 2024
author: Sarah Tymochko

Contains functions to compute the adjacency graph, compute scores, and compute 1d statistics on persistence diagrams (including some helper functions).

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import math
import networkx as nx
from shapely.strtree import STRtree

# import gudhi.wasserstein as gudhiwass
# import gudhi_wasserstein_distance as gudhim
# from gudhi.representations.vector_methods import BettiCurve
import resource_bifiltration_v3 as RB



def number_shared_terminal_comps(TC1, TC2):
    n = 0
    shared_term_params = set(TC1.term_coords).intersection(set(TC2.term_coords))
    for param in shared_term_params:
        TC2_param_rep = next(iter(TC2.C_reps[param])) # grab a block from TC2 component at param
        if TC2_param_rep in TC1.C_reps[param]:
            n += 1
    return n


def construct_graph(TCC):
    # Returns graph G whose nodes are the TrackedComponent objects. Edge weight is the number of shared terminal components they have
    G = nx.Graph()
    G.add_nodes_from(TCC.tracked_comps)
    nodes = list(G.nodes())
    for i, TC1 in enumerate(nodes):
        for TC2 in nodes[:i]:
            n = number_shared_terminal_comps(TC1, TC2)
            smallertermcoords = min([len(TC1.term_coords), len(TC2.term_coords)])

            if smallertermcoords > 0:

                # add edge only if more than half of the terminal components are shared 
                # (half being based on the one with less terminal coordinates. 
                #    if it has only 3 tcs it has to share at least 2 with the other one)
                if n/smallertermcoords >= 0.5:
                    G.add_edge(TC1, TC2, weight = n)
    return G


def compute_persistence_horizontal(rb,r):
    '''
    Compute persistence diagrams and block lists for r and r+eps
    '''

    ### Compute pd1
    G = rb.compute_graph_data_filtration_vertical(r)
    birth, death, _, block_lists = union_find(G)
    pd = np.zeros([len(birth), 2])
    pd[:, 0] = birth
    pd[:, 1] = death
    
    return pd, block_lists


def compute_persistence_vertical(rb,q):
    '''
    Compute persistence diagrams and block lists for r and r+eps
    '''

    ### Compute pd1
    G = rb.compute_graph_data_filtration_horizontal(q) ##NAMING OF FUNCTIONS IS INCONSISTENT
    birth, death, _, block_lists = union_find(G)
    pd = np.zeros([len(birth), 2])
    pd[:, 0] = birth
    pd[:, 1] = death
    
    return pd, block_lists


def adjacency_graph(blocks, calc_area=False): #, writefile=False, path=''):
    '''
    Input:
    blocks: shapefile that has block geometries as a column

    Output:
    G: Adjacency matrix in the form of a networkx graph
    '''
    
    tree = STRtree(blocks['geometry'])
    
    G = nx.Graph()
    N = len(blocks)

    G.add_nodes_from(range(N))

    # for i in range(N):
        
    nx.set_node_attributes(G, blocks['geometry'], "geometry")

    if calc_area:
        nx.set_node_attributes(G, blocks['geometry'].to_crs('EPSG:26916').area, "area")

    #add edges
    for n in G.nodes(data=True):
        state = n[1]['geometry']
        inds = tree.query(state, predicate = 'intersects')
        for o in inds:
            if o!=n[0]:
                G.add_edge(n[0], o) 
    
    # # WARNING this does NOT save node attributes! 
    # if writefile:
    #     nx.write_adjlist(G, path)
          
    return G

################# Functions to Compute Scores #################

# features we want to count as adding to the quality of the park in some way
features_global = ['archery_ra', 'artificial', 'band_shell', 'baseball_b',
       'baseball_j', 'baseball_s', 'basketba_1', 'basketball', 'beach',
       'boat_lau_1', 'boat_launc', 'bocce_cour', 'bowling_gr',
       'boxing_cen', 'carousel', 'casting_pi', 'climbing_w', 'community_',
       'conservato', 'cricket_fi', 'croquet', 'cultural_c', 'dog_friend',
       'fitness_ce', 'fitness_co', 'football_s', 'gallery', 'game_table',
       'garden', 'golf_cours', 'golf_drivi', 'golf_putti',
       'gymnasium', 'gymnastic_', 'handball_i', 'handball_r', 'harbor',
       'horseshoe_', 'iceskating',  'lagoon',  'minigolf',
       'modeltrain', 'modelyacht', 'mountain_b', 'nature_bir', 'nature_cen',
       'playgrou_1', 'playground', 'pool_indoo', 'pool_outdo', 'rowing_clu',
       'senior_cen', 'shuffleboa', 'skate_park',
       'sled_hill', 'sport_roll', 'spray_feat', 'tennis_cou', 'track',
       'volleyba_1', 'volleyball','water_play', 'water_slid',
       'wetland_ar', 'wheelchr_a','zoo']

def ci_score(park_rating, num_ratings):
    '''
    Input:
    park_rating: (float in [1, 5]). Average number of stars for some park.
    num_ratings: (nonnegative int). Number of ratings for some park.

    Output:
    score: (float) estimate of lower bound on 95% confidence interval for true mean park rating. Assumption: let x_1, ..., x_N be the iid individual ratings, let y_i = x_i - 1. Let p = (avg y_i)/4. Assume y_i is binomial(4, p). Then variance of x_i is 4*p = (park_rating - 1)
    '''
    # TO DO- what do we want to return when the park has no ratings?
    if num_ratings <= 0: return 0
    assert (park_rating >= 0 and park_rating <= 4)

    score = max(0, park_rating - 1.96*math.sqrt(10/num_ratings))
    return score

def ci_scores(parks_df, avg_rating_col = 'avg_review', num_ratings_col = 'tot_stars'):
    '''
    parks_df: (data frame).
    avg_rating_col: (string) Name of column in parks_df that stores averate ratings for each park
    num_ratings_col: (string) Name of column in parks_df that stores total number of ratings for each park
    '''
    scores = np.array([ci_score(row[avg_rating_col], row[num_ratings_col]) for i, row in parks_df.iterrows()])
    return scores

def star_scores(parks_df, ratings_df, avg_rating_col = 'avg_review', num_ratings_col = 'tot_stars'):
    '''
    Combines parks and ratings dataframes and computes total and average stars
    parks_df: (data frame).
    ratings_df: (data frame).
    '''

    # make stars on 0 to 4 scale 
    ratings_df['stars'] = (ratings_df['rating']-1)*ratings_df['user_ratings_total']

    # compute total number of ratings for each park
    num_ratings = ratings_df.groupby('PARK').sum()['user_ratings_total']
    
    # compute total number of stars for each park
    tot_stars = ratings_df.groupby('PARK').sum()['stars']
    
    # add total stars to parks dataframe
    parks_df = parks_df.join(tot_stars, on='park', validate='1:1')
    parks_df = parks_df.rename({'stars':'tot_stars'},axis=1)

    parks_df['total_ratings'] = 0        

    for i in range(len(parks_df)):
        parks_df.at[i,'total_ratings'] = num_ratings.loc[parks_df.at[i,'park']] ## equiv to 'user_ratings_total' col in ratings_df
        
        name = parks_df.at[i,'park']
    
        # if the park has no reviews or wasn't on google maps then set scores to 0
        # otherwise compute the average number of stars and add total stars to dataframe
        if (num_ratings[name]==0 or num_ratings[name]==-1):
            parks_df.at[i,avg_rating_col]= 0
            parks_df.at[i,num_ratings_col]= 0
        else:
            parks_df.at[i,avg_rating_col] = tot_stars[name]/num_ratings[name]
            parks_df.at[i,num_ratings_col] = tot_stars[name]

    return parks_df
    

def parkfeatures_score(parks, a=1/10, b=1/10, features=features_global):
    '''
    parks: (data frame) 
    a: parameter that weights acres
    b: parameter that weights number of features
    '''
    
    return (b*(sum(parks[feature] for feature in features)) + a*parks['acres']).values


##### Functions for working with persistence diagrams

def convert_gudhi_to_ripser(dgms, add_zeros=True):
    
    # Convert gudhi format PDs to ripser format PDs
    ripser_format = [[], []]
    for pt in dgms:
        dim = pt[0] 
        if pt[1][1] == np.inf or pt[1][1] == -np.inf:
            ripser_format[dim].append( [pt[1][0], np.inf] )
        else:
            ripser_format[dim].append( [pt[1][0], pt[1][1]] )
    if add_zeros:
        for i in range(2):
            ripser_format[0].append( [0,0] )
            ripser_format[1].append( [0,0] )
    
    ripser_format[0] = np.array(ripser_format[0])
    ripser_format[1] = np.array(ripser_format[1])
    
    return ripser_format

def max_pers(dgms, subsuper='sub'):
    dgms = convert_gudhi_to_ripser(dgms)
    
    mp = []
    for d in dgms:
        
        d = remove_inf_pts(np.array(d)) 

        if subsuper == 'super':
            d = np.fliplr(d)
        
        mp.append(max(d[:,1] - d[:,0]))
        
    return mp

def total_pers(dgms,subsuper='sub'):
    dgms = convert_gudhi_to_ripser(dgms)
    
    tp = []
    for d in dgms:
        d = remove_inf_pts(np.array(d))
        
        if subsuper == 'super':
            d = np.fliplr(d)
            
        tp.append( sum( d[:,1] - d[:,0]) )
        
    return tp

def remove_inf_pts(d,subsuper='sub'):
    d = d[d[:, 1] != np.inf]
        
    return d


def pers_entropy(dgms, normalize=False):
    dgms = convert_gudhi_to_ripser(dgms, add_zeros=False)
    
    pe = []
    dgm = remove_inf_pts(dgms[0])

    l = dgm[:,1] - dgm[:,0]
    if all(l > 0):
        L = np.sum(l)
        p = l / L
        E = -np.sum(p * np.log(p))
        if normalize == True:
            E = E / np.log(len(l))
        pe.append(E)
    else:
        return np.nan
        
    return pe
    
    
def wass_dist(dgm1, dgm2):
    dgm1 = convert_gudhi_to_ripser(dgm1, add_zeros=False)[0]
    # dgm1 = remove_inf_pts(dgm1)
    
    dgm2 = convert_gudhi_to_ripser(dgm2, add_zeros=False)[0]
    # dgm2 = remove_inf_pts(dgm2)

    try:
        d = wasserstein_distance(dgm1, dgm2)
    except:
        print('error', dgm1,dgm2)
        d = -1
    
    return d


def betti_curves(pds, dim=0, grid=None):
    '''
        Inputs: 
            pds: list of persistence diagrams output from gudhi
            dim: homology dimension
            grid: grid of values for filtration parameter (by default its 200 equally spaced between 0 
                    and the max value in any of the pds)

        Returns:
            num pds x num grid values array of betti numbers
    '''
    if not grid:
        m = 0
        pd_list = []
        for pd in pds:
            pd = convert_gudhi_to_ripser(pd)[dim]
            if pd[1:,:].max() > m:
                m = pd[1:,:].max()
            pd_list.append(pd)

        m_rounded = round(m + 5.1, -1)
        grid = np.linspace(0,m_rounded,201)

    bc = BettiCurve(predefined_grid=grid) 
    return grid, bc.fit_transform(pd_list).T    
    
    
def find(elt,parent):
  while (elt != parent[elt]):
    elt = parent[elt]
  return elt

def get_children(elt,parent):
  indx = []
  for i,mama in enumerate(parent):
    if mama==elt:
      indx.append(i)
  return indx

def find_desc(elt,parent):
  desc = [elt]
  children = get_children(elt,parent)
  for child in children:
    grandbabies = find_desc(child,parent)
    desc = desc+grandbabies
  return desc


def union(G,A,B,parent,birth,death,generators,children,block_list,t):
  rootA = find(A,parent)
  rootB = find(B,parent)
  if(rootA==rootB):
    return birth, death
  else:
    #elder rule: if A is born first, B dies.
    #We add the birth and death times for B here.
    if G.nodes[rootA]['score']< G.nodes[rootB]['score']:
      birth.append(G.nodes[rootB]['score'])
      parent[rootB] = rootA
      children[rootA] = children[rootA]+children[rootB]
      death.append(t)
      generators.append(rootB)
      block_list.append(children[rootB])
    #if B is born first, A dies
    else:
      parent[rootA] = rootB
      children[rootB] = children[rootB] + children[rootA]
      birth.append(G.nodes[rootA]['score'])
      death.append(t)
      generators.append(rootA)
      block_list.append(children[rootA])
  return union(G,A,B,parent,birth,death,generators,children,block_list,t)

def union_find(G):
  parent = list(G.nodes)
  birth = []
  death = []
  generators = []
  block_list = []
  children = [[i] for i in G.nodes]
  sorted_by_score = sorted(G.edges(data=True), key=lambda edge_data: edge_data[2]["score"])
  for edge in sorted_by_score:
    t = edge[2]['score']
    v = edge[0]
    w = edge[1]
#    (v,w) = edge.nodes()
    birth,death = union(G,v,w,parent,birth,death,generators,children,block_list,t)

  bad = []
  #this can be rewritten with np.where more efficiently I think
  for i in range(len(birth)):
    if death[i]-birth[i]==0:
      bad.append(i)

  #remove features with 0 persistence, maybe can be done implicitly
  death = [death[i] for i in range(len(death)) if i not in bad]
  birth = [birth[i] for i in range(len(birth)) if i not in bad]
  generators = [generators[i] for i in range(len(generators)) if i not in bad]
  block_list = [block_list[i] for i in range(len(block_list)) if i not in bad]
  #block_list = [find_desc(i,parent) for i in tqdm(generators)]
  return birth,death,generators,block_list