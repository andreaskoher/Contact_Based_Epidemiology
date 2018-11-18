import numpy as np
import networkx as nx


def integrate_contact_based_model(
    alpha,
    beta,
    edgelist,
    outbreak_origin = None,
    Tmax = None,
    individual_based = True,
    directed = True,
    verbose = True ):
    """
    Integrates the contact-based model for SIR type of spreading on a temporal network.

    The algorithm is based on the dynamic message passing algorithm introduced for static networks in:

        Lokhov et al. "Inferring the origin of an epidemic with a dynamic message-passing algorithm", Phys. Rev. E 90, 1, 012801 (2014)

    The generalization to temporal networks has been introduced in:

        Koher et al. "Contact-based model for epidemic spreading on temporal networks", Arxive 1811.05809, (2018)
    

    Parameters
    ----------
    alpha : float
        infection_probability
    
    beta : float
        recovery probability

    edgelist : numpy.ndarray
        time-stamped edgelist with following dimensions:
        number of (temporal contacts) x 3
        columns denote: time, source, target
        By default contacts are assumed directed
        Value dtype is integer, time starts from 0 and,
        node index runs continuously from 0 to 'number of nodes'-1
   
   outbreak_origin : int
        initially infected node.
        Note: code is can be extended to account for multiple
        outbreak locations

    Tmax : int
        integrate from 0 to Tmax-1
        By default integration ends with the last time step,
        i.e. Tmax = edgelist[:,0].max() + 1
        You can choose a smaller value to restrict the time frame
        or, if Tmax > edgelist[:,0].max() + 1 periodicity in 
        time is assumed.

    individual_based : bool
        Use contact-based framework (True by default).
        Otherwise individual-based framework will be invoked.

    directed : bool
        Interprete edgelist as directed (True by default).
        Otherwise reciprical contacts will be added.

    verbose : bool
        tries to import tqdm_notebook from the tqdm package
        and displays a progress bar, optimized for jupyter
        notebook
    

    Output
    ----------

    susceptible : numpy.ndarray
        probability over time to find a node in the susceptible state 
        'number of nodes' x  'time steps'

    infected : numpy.ndarray
        probability over time to find a node in the infected state 
        'number of nodes' x  'time steps'

    recovered : numpy.ndarray
        probability over time to find a node in the recovered state 
        'number of nodes' x  'time steps'
    """

    edgelist = np.array(edgelist, dtype=int)
    alpha = float(alpha)
    beta = float(beta)
    verbose = bool(verbose)
    individual_based = bool(individual_based)
    directed = bool(directed)
   
    nodes = set( edgelist[:,1] ) | set( edgelist[:,2] )
    Nnodes = len(nodes)
    assert nodes == set(xrange(Nnodes)), "nodes must be named continuously from 0 to 'number of nodes'-1 "
    assert alpha >= 0 and alpha <= 1, "infection probability incorrect"
    assert beta >= 0 and beta <= 1, "recovery probability incorrect"

    if isinstance( outbreak_origin, int ):
    
        assert outbreak_origin >= 0 and outbreak_origin <  Nnodes, "0 <= outbreak origin < 'number of nodes'"
        out = _integrate_fixed_source(alpha, 
                               beta, 
                               outbreak_origin, 
                               edgelist, 
                               Tmax=Tmax, 
                               verbose=verbose, 
                               individual_based=individual_based,
                               directed = directed)
    
    susceptible, infected, recovered = out

    return susceptible, infected, recovered







# ==================================================================================
#                      Trajectory for a given initial conditions
# ==================================================================================
def _integrate_fixed_source(alpha, 
                               beta, 
                               outbreak_origin, 
                               edgelist, 
                               Tmax=None, 
                               verbose=False, 
                               individual_based=True,
                               directed = True):
    
    edgelist = np.array(edgelist, dtype=int)

    if not directed:
        edgelist_directed = np.zeros(( len(edgelist)*2, 3 ))
        for idx in xrange( len(edgelist) ):
            t, u, v = edgelist[idx]
            edgelist_directed[2*idx, :] = t, u, v
            edgelist_directed[2*idx + 1, :] = t, v, u
        edgelist = edgelist_directed.astype(int)

    # list of time indices with at least one active edge
    times = np.unique(edgelist[:,0])
    T = max(times) + 1 # last time step + 1
    if Tmax is None:
        Tmax = T
    else:
        Tmax += 1


    # make time-aggregated network
    G = nx.DiGraph()
    G.add_edges_from([(u,v) for t,u,v in edgelist])
    assert G.number_of_selfloops() == 0, "self loops are not allowed"
    static_out_neighbours = G.succ #for speed
    static_in_neighbours = G.pred #for speed
    Nedges = G.number_of_edges()
    Nnodes = G.number_of_nodes()
    edge_to_idx = {edge: idx for idx, edge in enumerate(G.edges())}
    reciprical_link_exists = {edge_to_idx[ u, v ]: True if G.has_edge(v,u) else False for u,v in G.edges()} #for speed
    active_edges_dct = {t: [] for t in xrange(Tmax)} #for speed
    
    # transform edgelist to a temporal predecessor dict:
    # temporal_in_neighbours:
    #   key: time -> value: dict
    #        key: node -> set of predecessors at time instance
    temporal_in_neighbours = {time: {} for time in times}
    for time in times:
        snapshot = edgelist[ edgelist[:,0] == time ] # contacts at time instance
        temporal_in_neighbours[ time ] = { target: set() for target in np.unique( snapshot[:,2] ) }
        for _, source, target in edgelist[ edgelist[:,0] == time ]:
            temporal_in_neighbours[time][target].add(source)
        active_edges_dct[time] = np.array( [edge_to_idx[ u, v ] for _, u, v in snapshot ] )
    active_targets = {t: set(temporal_in_neighbours[t].keys()) if t in times else set() for t in xrange(T)} #for speed #CHANGE NAME
    
    sus_over_time = np.zeros((Nnodes, Tmax+1)) #probability: node is susceptible at time 0 <= t < Tmax
    inf_over_time = np.zeros((Nnodes, Tmax+1)) #probability: node is infected at time 0 <= t < Tmax
    rec_over_time = np.zeros((Nnodes, Tmax+1)) #probability: node is recovered at time 0 <= t < Tmax

    inf_over_time = np.zeros((Nnodes, Tmax+1))

    inf_over_time = np.zeros((Nnodes, Tmax+1))

    sus     = np.ones(Nedges) # conditional probability: node is susceptible given neighbor is in cavity state
    sus_new = np.ones(Nedges) # temporal buffer
    theta   = np.ones(Nedges) # conditional probability: no disease has been transmitted across edge given neighbor is in cavity state
    phi     = np.zeros(Nedges) # conditional probability: node is infected and has not trasmitted infection to neighbor given neighbor is in cavity state
    
    # mark edges that leave initially infected node 
    init_idx = np.array([edge_to_idx[node, cavity] for (node, cavity) in G.edges() if node == outbreak_origin], dtype=int)
    # initial condition
    sus[init_idx] = 0.
    sus_new[init_idx] = 0.
    phi[init_idx] = 1.
    
    susceptible = np.ones(Nnodes)  # susceptible = sus_over_time[ time ]
    susceptible[outbreak_origin] = 0.
    infected = np.zeros(Nnodes) #infected = inf_over_time[ time ]
    infected[outbreak_origin] = 1.
    recovered = np.zeros(Nnodes) # recovered = rec_over_time[ time ]
    
    sus_over_time[:,0] = susceptible
    inf_over_time[:,0] = infected
    
    if verbose: #time iterator
        try:
            from tqdm import tqdm_notebook as tqdm
        except ImportError:
            print "package 'tqdm' not found"
            print "continue with verbose = False..."
            time_iter = xrange(Tmax)
        else:
            time_iter = tqdm(range(Tmax)) #requires tqdm package installed
    else:
        time_iter = xrange(Tmax) #time starts at 0
    
    time_idx = 0 #time starts at 0
    for time in time_iter:
        #update only nodes that have at least one incident edge at time instance
        for target in active_targets[time_idx]:
            # outbreak origin cannot be infected
            if target != outbreak_origin:
                susceptible[target] = 1.
                # go through all (static) predecessors of target
                for source in static_in_neighbours[ target ]:
                    edge_idx = edge_to_idx[ source, target ]
                    # infection transmission (update theta for active edges only)
                    # do not account for non-backtracking property here
                    if source in temporal_in_neighbours[time_idx][target]:
                        # update according to individual-based model
                        if not individual_based:
                            theta[edge_idx] -= alpha * phi[edge_idx]
                        # update according to contact-based model
                        else:
                            theta[edge_idx] *= 1. - alpha * infected[source]
                    susceptible[target] *= theta[ edge_idx ]

                # account for non-backtracking property
                if not individual_based:
                    for cavity in static_out_neighbours[target]:
                        edge_idx = edge_to_idx[ target, cavity ]
                        sus_new[ edge_idx ] = susceptible[ target ]
                        # discount transmission probability along
                        # backtracking contact: cavity -> target
                        if reciprical_link_exists[ edge_idx ]:
                            reciprical_edge_idx = edge_to_idx[ cavity, target ]
                            sus_new[ edge_idx ] /= theta[ reciprical_edge_idx ]

        # update edge-based quantities only for contact-based model
        if not individual_based:    
            active_edges = np.zeros(Nedges, dtype=bool)
            active_edges[ active_edges_dct[time_idx] ] = True
            phi *= (1. - beta) * (1. - alpha * active_edges)
            phi +=  sus - sus_new
            sus = sus_new.copy()
        
        recovered += infected * beta
        infected = 1. - recovered - susceptible
        
        sus_over_time[:, time + 1] = susceptible
        inf_over_time[:, time + 1] = infected
        rec_over_time[:, time + 1] = recovered

        time_idx = (time_idx + 1) % T
        
    return sus_over_time, inf_over_time, rec_over_time









    # check your system
def test_system():
    
    def compare_versions(a,b):
        a2 = tuple( map(int,a.split('.')) )
        b2 = tuple( map(int,b.split('.')) )
        if a2==b2:
            return '* Same version *'
        elif a2>b2:
            return '*** Your version is more recent ***'
        else:
            return '*** Your version older ***'
    
    dc0 = {}
    dc0['python'] = '2.7.15'
    dc0['numpy'] = '1.15.1'
    dc0['networkx'] = '2.1'
    #dc0['tqdm'] = '4.23.4'
    
    lerr = []
    ferr = True    
    
    from sys import version_info   
    
    try:
        import networkx as nx
    except ImportError:
        lerr.append('networkx')
        ferr = False
    
    try:
        import numpy as np
    except ImportError:
        lerr.append('numpy')
        ferr = False

    spu = 'ERROR! MISSING MODULES: '
    for e in lerr:
        spu += e+', '
    spu = spu.strip().strip(',')+'.'

    assert ferr, spu
    
    dc = {}
    dc['python'] = '%s.%s.%s' % version_info[:3]
    dc['numpy'] = np.__version__
    dc['networkx'] = nx.__version__

    
    print '\n---------'
    print '---------'
    print 'All required modules are present'
    print '---------'
    print '---------'
    print '{:16}{:16}{:16}'.format('MODULE','TESTED FOR','YOURS')
    for x,v0 in dc0.iteritems():
        print '{:16}{:16}{:16} {}'.format(x,v0,dc[x],compare_versions(dc[x],v0))
    print '--------'
    print '--------'
    print '--------'
    print '--------\n'

    
if __name__ == '__main__':
    
    test_system()
