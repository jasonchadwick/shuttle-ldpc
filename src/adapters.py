import stim
import networkx as nx
import numpy as np
import numpy as np
from ldpc.mod2 import reduced_row_echelon, rank
from ldpc.code_util import construct_generator_matrix
from itertools import combinations

def pretty_print_matrix(M):
    print(M.shape)
    simbles = ['  ', '* ']
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            print(simbles[M[i,j]], end='')
        print()

def pretty_print_matrix_by_line(matrix, q = 100, symbols = [' ','*'],linebreaks = None, return_string = False):
    squares =[]
    linebreaks = linebreaks or ['-'*q]*len(matrix)
    for i,row in enumerate(matrix):
        st_sq = '|\n|'.join([''.join([symbols[x] for x in row[i*q:(i+1)*q]]) for i in range(len(row)//q)])
        st_sq = '|'+st_sq+'|'
        squares.append(st_sq)
        print(st_sq)
        print(linebreaks[i])
    return squares if return_string else None

## Helpers
def idx_to_array(idx, dim):
    arr = np.zeros(dim, dtype=int)
    arr[idx] = 1
    return arr

def array_to_idx(arr, indicator = 1):
    return np.where(arr == indicator)[0]

def find_induced_tanner(logical: np.ndarray, checks: np.ndarray):
    # find indices of non-zero logical
    logical_supp = np.where(logical != 0)[0]
    # filter checks to only include those that have non-zero intersection with logical
    check_supp = np.where((checks[:,logical_supp] != 0).sum(axis=1) > 0)[0]
    induced_check = checks[check_supp,:][:,logical_supp]
    return logical_supp,check_supp, induced_check

def add_gauge_qubits(H, H_induced, qb_support):
    if rank(H) == H.shape[0]:
        # no gauge qubits needed
        return H, np.array([], dtype=int)

    GxT = construct_generator_matrix(H_induced.T).toarray()

    assert GxT.shape[1] == len(qb_support), "Wrong qubits to find gauge qubits"

    # implement more sophisticated weight reduction methods later

    Gxt_red, _, _, P = reduced_row_echelon(GxT)
    Gxt_red = Gxt_red@P.T
    
    H_new = np.zeros((H.shape[0]+Gxt_red.shape[0], H.shape[1]), dtype=int)
    H_new[:H.shape[0], :H.shape[1]] = H
    H_new[H.shape[0]:,qb_support] = Gxt_red

    return H_new, np.arange(H.shape[0],H_new.shape[0], dtype=int)

def calculate_boundary_cheeger_constant(H_induced):
    '''
    Calculate the boundary cheeger constant of a given induced check matrix
    '''
    num_qbs = H_induced.shape[1]

    best_beta = num_qbs
    assert num_qbs<=20, 'This is a brute force method and will not work for distances > 20'
    for i in range(1,num_qbs//2+1):
        for comb in combinations(range(num_qbs), i):
            boundary_size = (H_induced[:, comb].sum(axis=1)%2).sum()
            beta = boundary_size/i
            if beta<=best_beta:
                best_beta = beta
    return best_beta
    
def add_ckbb_layer(H_primary, H_anticomm, logical_supp, check_anticomm, H_induced, dual_indicator = -1,  check_indicator = -1):
    assert H_primary.shape[1] == H_anticomm.shape[1], "Must be defined on the same qubits"
    assert (check_indicator == -1 or check_indicator in check_anticomm), "Indicated check not in the given check matrix"
    assert (dual_indicator == -1 or dual_indicator in logical_supp), "Indicated logical not in the logical support"
    
    H_primary_new = np.zeros((H_primary.shape[0]+len(logical_supp), H_primary.shape[1] + len(check_anticomm)), dtype=int)
    H_anticomm_new = np.zeros((H_anticomm.shape[0], H_anticomm.shape[1] + len(check_anticomm)), dtype=int)
    
    H_primary_new[:H_primary.shape[0], :H_primary.shape[1]] = H_primary
    H_anticomm_new[:H_anticomm.shape[0], :H_anticomm.shape[1]] = H_anticomm


    # 1. add primary checks
    # 1.1. identity checks to old logical supp (l0-l1)
    H_primary_new[H_primary.shape[0]:H_primary.shape[0]+len(logical_supp),logical_supp] = np.eye(len(logical_supp),dtype=int)
    # 1.2. induced checks to new anticomm_check qubits (l1)
    H_primary_new[H_primary.shape[0]:H_primary.shape[0]+len(logical_supp),H_primary.shape[1]:H_primary.shape[1]+len(check_anticomm)] = H_induced.T

    # 2. add anticomm check qubits
    # 2.1. identity checks added to old anticomm checks (l0-l1)
    H_anticomm_new[check_anticomm,H_anticomm.shape[1]:H_anticomm.shape[1]+len(check_anticomm)] = np.eye(len(check_anticomm),dtype=int)

    new_qbs_from_check = np.arange(H_primary.shape[1], H_primary_new.shape[1])
    new_check_from_logical = np.arange(H_primary.shape[0], H_primary_new.shape[0])

    # 3. track indicated check/qubit
    new_check_indicator = -1
    new_dual_indicator = -1
    if dual_indicator != -1:
        new_check_indicator = np.where(H_primary_new[H_primary.shape[0]:,dual_indicator] == 1)[0][0] + H_primary.shape[0]
    if check_indicator != -1:
        new_dual_indicator = np.where(H_anticomm_new[check_indicator,H_anticomm.shape[1]:] == 1)[0][0] + H_anticomm.shape[1]


    if  (new_check_indicator, new_dual_indicator) != (-1,-1):
        return H_primary_new, H_anticomm_new, new_qbs_from_check, new_check_from_logical, new_dual_indicator, new_check_indicator

    return H_primary_new, H_anticomm_new, new_qbs_from_check, new_check_from_logical

def find_low_weight_gauge(H_ind_id, H_ind_permuted, indicator = -1, tn = 100, best_so_far = None):
    '''
    Find a low weight gauge qubit that is only connected to one check

    Given two induced matrices A_ind, B_ind, we have an overall induced matrix
                _                         _
                | A_ind |   0   |   S_1   |
      H_ind =   |-------+-------+---------|
                | 0     | B_ind |   S_2   |
                -                         -
    and 
      G_ind = ns(H_ind) s.t.

                 _       _ 
                 | H_ind |
      H_ind' =   |-------|
                 | G_ind |
                 -       -
    has the lowest row/col weight. 

    S_1, and S_2 are rectangular permutation matrices which we generate randomly over tn trials.

    Finally, checks that constraint the gauge qubit, the associated S matrices and the weight are returned
    '''

    if tn <0:
        return best_so_far

    m_inv, n_inv = H_ind_id.shape
    m_permuted, n_permuted = H_ind_permuted.shape
    assert m_inv <= m_permuted, "Must pass smaller induced matrix first"
    num_bridge_qbs = m_inv
    
    # extend to bridge checks
    links_permuted = np.random.choice(m_permuted, num_bridge_qbs, replace=False)
    if indicator != -1:
        while True:
            if indicator in links_permuted:
                break
            links_permuted = np.random.choice(m_permuted, num_bridge_qbs, replace=False)


    added_links = np.zeros((m_inv+m_permuted, num_bridge_qbs), dtype=int)

    added_links[:m_inv, :] = np.eye(m_inv, dtype=int)
    for i, link in enumerate(links_permuted):
        added_links[m_inv+link,i] = 1


    H_extended = np.zeros((m_inv+m_permuted, n_inv+n_permuted+num_bridge_qbs), dtype=int)

    H_extended[:m_inv, :n_inv] = H_ind_id
    H_extended[m_inv:, n_inv:n_inv+n_permuted] = H_ind_permuted

    H_extended[:, n_inv+n_permuted:] = added_links

    null_checks = construct_generator_matrix(H_extended).toarray()
    null_checks = null_checks[np.any(null_checks[:, -num_bridge_qbs:], axis=1)]
    assert null_checks.shape[0] == num_bridge_qbs-1, "DOF mismatch"


    # find the num_bridge_qbs-1 rows of null_checks that have the least weight
    weights = null_checks.sum(axis=1)
    best_rows = np.argsort(weights)[:num_bridge_qbs-1]

    max_check_weight = weights[best_rows].max()
    max_qubit_weight = null_checks.sum(axis=0).max()
    max_weight = max(max_check_weight, max_qubit_weight)
    
    if best_so_far is None or max_weight < best_so_far[-1]:
        best_so_far = (
            null_checks[best_rows][:,:n_inv], # Gauge check on inv qubits
            null_checks[best_rows][:,n_inv:n_inv+n_permuted], # Gauge check on permuted qubits
            null_checks[best_rows][:,n_inv+n_permuted:], # Gauge check on bridge qubits
            added_links[m_inv:, :], # Permuted extended Z check
            max_weight)
    
    return find_low_weight_gauge(H_ind_id, H_ind_permuted, indicator, tn-1, best_so_far)

def find_single_ancilla(
    Hx, 
    Hz, 
    logical, 
    logical_type = 'z', 
    num_layers = -1,
    dual_indicator = -1, 
    return_legends = False, 
    verbose = False,
    verbose_print_matrices = False
):
    '''
    Find a single ancilla qubit that is only connected to one check
    '''
    if verbose:
        print(f"\n[find_single_ancilla] Inputs summary: Hx shape: {Hx.shape}, Hz shape: {Hz.shape}, logical: {logical}, logical_type: {logical_type}, num_layers: {num_layers}, dual_indicator: {dual_indicator}")
        if verbose_print_matrices:
            print(f"\tHx:\n{Hx}")
            print(f"\tHz:\n{Hz}")

    logical_types = ['z', 'x'] 
    if logical_type == 'x':
        logical_types = ['x', 'z']
        Hz, Hx = Hx, Hz

    logical_z_supp, check_x_supp, H_induced = find_induced_tanner(logical, Hx)
    duals_indicated = []
    checks_indicated = []
    dual_indicator = logical_z_supp[-1] if dual_indicator == -1 else dual_indicator

    checks_legend = {
        'Hz_original': np.arange(Hz.shape[0], dtype=int),
        'Hx_original': np.arange(Hx.shape[0], dtype=int),
        'induced': (logical_types[1], check_x_supp),
    }
    qbs_legend = {
        'original': np.arange(Hx.shape[1], dtype=int),
        'logical': (logical_types[0], logical_z_supp),
    }

    # add interface
    if verbose:
        print('\n\tAdding interface, given dual indicator')
        print('\t\tdual_indicator:', dual_indicator, 'logical_z_supp:', logical_z_supp)
    Hz_new, Hx_new, new_qbs_x_type, new_z_checks, _, check_indicator = add_ckbb_layer(
        Hz, Hx, logical_z_supp, check_x_supp, H_induced, dual_indicator=dual_indicator
    )
    checks_indicated.append(check_indicator)
    checks_legend['ckbb_layers'] = [(logical_types[0], new_z_checks)]
    qbs_legend['ckbb_layers'] = [(logical_types[1], new_qbs_x_type)]

    if verbose:
        print(f'\t\tindicated check {dual_indicator} -> {check_indicator}')
        if verbose_print_matrices:
            print('\t\tHz_new:\n', Hz_new)
            print('\t\tHx_new:\n', Hx_new)
            print('\t\tnew_qbs_x_type:', new_qbs_x_type)
            print('\t\tnew_z_checks:', new_z_checks)

    if num_layers == -1:
        cheeger_constant = calculate_boundary_cheeger_constant(H_induced)
        num_layers = int(np.ceil(1 / cheeger_constant) * 2 - 1)

    # add additional layers 
    for layer in range(num_layers // 2):
        if verbose: 
            print(f'\n\tlayer {layer}')
        Hx_new, Hz_new, new_qbs_z_type, new_x_checks, dual_indicator, _ = add_ckbb_layer(
            Hx_new, Hz_new, new_qbs_x_type, new_z_checks, H_induced.T, check_indicator=check_indicator
        )
        duals_indicated.append(dual_indicator)
        if verbose:
            print(f'\t\tindicated dual {check_indicator} -> {dual_indicator}')
        checks_legend['ckbb_layers'].append((logical_types[1], new_x_checks))
        qbs_legend['ckbb_layers'].append((logical_types[0], new_qbs_z_type))
        if verbose and verbose_print_matrices:
            print('\t\tHx_new:\n', Hx_new)
            print('\t\tHz_new:\n', Hz_new)
            print('\t\tnew_qbs_z_type:', new_qbs_z_type)
            print('\t\tnew_x_checks:', new_x_checks)
        Hz_new, Hx_new, new_qbs_x_type, new_z_checks, _, check_indicator = add_ckbb_layer(
            Hz_new, Hx_new, new_qbs_z_type, new_x_checks, H_induced, dual_indicator=dual_indicator
        )
        checks_legend['ckbb_layers'].append((logical_types[0], new_z_checks))
        qbs_legend['ckbb_layers'].append((logical_types[1], new_qbs_x_type))
        checks_indicated.append(check_indicator)
        if verbose:
            print(f'\t\tindicated check {dual_indicator} -> {check_indicator}')
        if verbose and verbose_print_matrices:
            print('\t\tHz_new:\n', Hz_new)
            print('\t\tHx_new:\n', Hx_new)
            print('\t\tnew_qbs_x_type:', new_qbs_x_type)
            print('\t\tnew_z_checks:', new_z_checks)

    checks_legend['Hz_ancilla'] = np.arange(Hz.shape[0], Hz_new.shape[0], dtype=int)
    checks_legend['Hx_ancilla'] = np.arange(Hx.shape[0], Hx_new.shape[0], dtype=int)
    checks_legend['indicated_checks'] = (logical_types[0], checks_indicated)

    qbs_legend['ancilla'] = np.arange(Hx.shape[1], Hx_new.shape[1], dtype=int)
    qbs_legend['dual_logical'] = (logical_types[1], duals_indicated)

    # add gauge qubits
    Hx_new, gauge_checks = add_gauge_qubits(Hx_new, H_induced, new_qbs_x_type)
    checks_legend['gauge_checks'] = (logical_types[1], gauge_checks)

    # validation
    # sum of Z checks gives the Z operator
    combined_z_stab = Hz_new[Hz.shape[0]:].sum(axis=0) % 2
    combined_z_stab = np.where(combined_z_stab != 0)[0]
    
    if verbose:
        print('\n\tThe logical operators are:')
        print('\t\tcombined_z_stab:', combined_z_stab)
        print('\t\tlogical_z_supp:', logical_z_supp)
    assert set(combined_z_stab) == set(logical_z_supp), 'Stabilizer product is not the desired logical operator'

    # check commutation
    assert (Hx_new @ Hz_new.T % 2 == 0).all(), 'New Hx and Hz do not commute'

    # check dof
    dof_original = Hx.shape[1] - rank(Hx) - rank(Hz)
    dof_new = Hx_new.shape[1] - rank(Hx_new) - rank(Hz_new)
    assert dof_new == dof_original - 1, 'Incorrect number of gauge qubits'

    if return_legends:
        return Hx_new, Hz_new, checks_legend, qbs_legend
    else:
        return Hx_new, Hz_new
    
def find_joint_ancilla(
    checks: list[np.ndarray],  # (Hx_1, Hz_1, Hx_2, Hz_2)
    logicals: list[np.ndarray],  # (logical_1, logical_2)
    logical_type: str = 'z',
    num_layers: list[int] = [-1, -1],  # (num_layers_1, num_layers_2)
    dual_indicators: list[int] = [-1, -1],
    return_legends: bool = False,
    verbose: bool = False,
    verbose_print_matrices: bool = False,
    assertion_check: bool = False
):
    '''
    Find a single ancilla qubit that is only connected to one check
    '''
    if verbose:
        print(f"\n[find_joint_ancilla] Inputs summary: checks: {[c.shape for c in checks]}, logicals: {[l.shape if hasattr(l, 'shape') else type(l) for l in logicals]}, logical_type: {logical_type}, num_layers: {num_layers}, dual_indicators: {dual_indicators}")
        print(f"\tlogical_1: {logicals[0]}")
        print(f"\tlogical_2: {logicals[1]}")        
        if verbose_print_matrices:
            print(f"\tchecks[0] (Hx_1):\n{checks[0]}")
            print(f"\tchecks[1] (Hz_1):\n{checks[1]}")
            print(f"\tchecks[2] (Hx_2):\n{checks[2]}")
            print(f"\tchecks[3] (Hz_2):\n{checks[3]}")


    Hx_1, Hz_1, Hx_2, Hz_2 = checks
    logical_types = ['z', 'x']

    if logical_type == 'x':
        Hx_1, Hz_1 = (Hz_1, Hx_1)
        Hx_2, Hz_2 = (Hz_2, Hx_2)
        logical_types = ['x', 'z']

    logical_1, logical_2 = logicals

    if verbose:
        print('\n\tProcessing...')
        if verbose_print_matrices:
            print('\t\tHx_1:\n', Hx_1)
            print('\t\tHz_1:\n', Hz_1)
            print('\t\tlogical_1:', logical_1)
            print('\t\tlogical_2:', logical_2)
            print('\t\tlogical_type:', logical_type)
            print('\t\tnum_layers:', num_layers)
            print('\t\tdual_indicators:', dual_indicators)

    Hx_1_adapted, Hz_1_adapted, checks_legend_1, qbs_legend_1 = find_single_ancilla(
        Hx_1, Hz_1, logical_1, logical_type, dual_indicator=dual_indicators[0], num_layers=num_layers[0],
        verbose=verbose, verbose_print_matrices=verbose_print_matrices, return_legends=True
    )
    Hx_2_adapted, Hz_2_adapted, checks_legend_2, qbs_legend_2 = find_single_ancilla(
        Hx_2, Hz_2, logical_2, logical_type, dual_indicator=dual_indicators[1], num_layers=num_layers[1],
        verbose=verbose, verbose_print_matrices=verbose_print_matrices, return_legends=True
    )


    if verbose and verbose_print_matrices:
        print('\n\tchecks_legend_1:', checks_legend_1)
        print('\tqbs_legend_1:', qbs_legend_1)
        print('\tchecks_legend_2:', checks_legend_2)
        print('\tqbs_legend_2:', qbs_legend_2)

    final_check_layer_1 = checks_legend_1['ckbb_layers'][-1]
    final_check_layer_2 = checks_legend_2['ckbb_layers'][-1]
    final_checks_1 = final_check_layer_1[1] 
    final_checks_2 = final_check_layer_2[1]
    checks_legend_1['ckbb_layers'][-1] = (final_check_layer_1[0], final_checks_1)
    checks_legend_2['ckbb_layers'][-1] = (final_check_layer_2[0], final_checks_2)

    final_qbs_1 = qbs_legend_1['ckbb_layers'][-1][1]
    final_qbs_2 = qbs_legend_2['ckbb_layers'][-1][1]

    num_bridge_qbs = min(sum(logical_1), sum(logical_2))
    num_x_checks_1, num_qbs_1 = Hx_1_adapted.shape
    num_x_checks_2, num_qbs_2 = Hx_2_adapted.shape

    num_z_checks_1 = Hz_1_adapted.shape[0]
    num_z_checks_2 = Hz_2_adapted.shape[0]

    total_x_checks = num_x_checks_1 + num_x_checks_2
    total_z_checks = num_z_checks_1 + num_z_checks_2
    total_qbs = num_qbs_1 + num_qbs_2

    Hx_combined = np.zeros((total_x_checks + num_bridge_qbs - 1, total_qbs + num_bridge_qbs), dtype=int)
    Hz_combined = np.zeros((total_z_checks, total_qbs + num_bridge_qbs), dtype=int)

    Hx_combined[:num_x_checks_1, :num_qbs_1] = Hx_1_adapted
    Hz_combined[:num_z_checks_1, :num_qbs_1] = Hz_1_adapted

    Hx_combined[num_x_checks_1:num_x_checks_1 + num_x_checks_2, num_qbs_1:num_qbs_1 + num_qbs_2] = Hx_2_adapted
    Hz_combined[num_z_checks_1:num_z_checks_1 + num_z_checks_2, num_qbs_1:num_qbs_1 + num_qbs_2] = Hz_2_adapted

    # labeling
    checks_legend_combined = {
        'Left code': checks_legend_1
    }
    right_checks_legend_updated = {
        'Hx_original': checks_legend_2['Hx_original'] + num_x_checks_1,
        'Hz_original': checks_legend_2['Hz_original'] + num_z_checks_1,
        'Hx_ancilla': checks_legend_2['Hx_ancilla'] + num_x_checks_1,
        'Hz_ancilla': checks_legend_2['Hz_ancilla'] + num_z_checks_1,
        'gauge_checks': (checks_legend_2['gauge_checks'][0], [x + num_x_checks_1 for x in checks_legend_2['gauge_checks'][1]]),
        'induced': (checks_legend_2['induced'][0], [x + num_x_checks_1 for x in checks_legend_2['induced'][1]]),
        'ckbb_layers': [(layer[0], [x + num_x_checks_1 for x in layer[1]]) for layer in checks_legend_2['ckbb_layers']],
        'indicated_checks': (checks_legend_2['indicated_checks'][0], [x + num_z_checks_1 for x in checks_legend_2['indicated_checks'][1]]),
    }
    checks_legend_combined['Right code'] = right_checks_legend_updated

    qbs_legend_combined = {
        'Left code': qbs_legend_1
    }
    right_qbs_legend_updated = {
        'original': qbs_legend_2['original'] + num_qbs_1,
        'logical': (qbs_legend_2['logical'][0], [x + num_qbs_1 for x in qbs_legend_2['logical'][1]]),
        'dual_logical': (qbs_legend_2['dual_logical'][0], [x + num_qbs_1 for x in qbs_legend_2['dual_logical'][1]]),
        'ancilla': qbs_legend_2['ancilla'] + num_qbs_1,
        'ckbb_layers': [(layer[0], [x + num_qbs_1 for x in layer[1]]) for layer in qbs_legend_2['ckbb_layers']],
    }
    qbs_legend_combined['Right code'] = right_qbs_legend_updated

    # agregating checks
    checks_legend_combined['Hz'] = np.concatenate(
        (checks_legend_1['Hz_original'],
         right_checks_legend_updated['Hz_original']))

    checks_legend_combined['Hx'] = np.concatenate(
        (checks_legend_1['Hx_original'],
         right_checks_legend_updated['Hx_original']))

    checks_legend_combined['Hz_ancilla'] = np.concatenate(
        (checks_legend_1['Hz_ancilla'],
         right_checks_legend_updated['Hz_ancilla']))

    checks_legend_combined['Hx_ancilla'] = np.concatenate(
        (checks_legend_1['Hx_ancilla'],
         right_checks_legend_updated['Hx_ancilla']))

    gauge_check_type = checks_legend_1["gauge_checks"][0]
    checks_legend_combined['gauge_checks'] = (
        gauge_check_type,
        np.concatenate(
            (checks_legend_1['gauge_checks'][1],
             right_checks_legend_updated['gauge_checks'][1])))

    gauge_ancilla_type = 'H' + gauge_check_type + '_ancilla'
    checks_legend_combined[gauge_ancilla_type] = np.concatenate((
        checks_legend_combined[gauge_ancilla_type],
        checks_legend_combined['gauge_checks'][1]
    ))

    # agregating qubits
    qbs_legend_combined['original'] = np.concatenate(
        (qbs_legend_1['original'],
         right_qbs_legend_updated['original']))

    qbs_legend_combined['ancilla'] = np.concatenate(
        (qbs_legend_1['ancilla'],
         right_qbs_legend_updated['ancilla']))

    qbs_legend_combined['logical'] = (
        qbs_legend_1['logical'][0],
        np.concatenate(
            (qbs_legend_1['logical'][1],
             right_qbs_legend_updated['logical'][1])))

    qbs_legend_combined['dual_logical'] = (
        qbs_legend_1['dual_logical'][0],
        np.concatenate(
            (qbs_legend_1['dual_logical'][1],
             right_qbs_legend_updated['dual_logical'][1])))

    # extensions to gauge checks
    H1_induced = Hz_1_adapted[final_checks_1][:,final_qbs_1]
    H2_induced = Hz_2_adapted[final_checks_2][:,final_qbs_2]

    indicated_check_1 = checks_legend_1['indicated_checks'][1][-1]
    indicated_check_2 = checks_legend_2['indicated_checks'][1][-1]

    indicator_idx_1 = np.where(final_checks_1 == indicated_check_1)[0][0]
    indicator_idx_2 = np.where(final_checks_2 == indicated_check_2)[0][0]


    inv_check_id = np.argmin((len(final_checks_1), len(final_checks_2)))
    H_ind_inv = [H1_induced, H2_induced][inv_check_id]
    H_ind_permuted = [H2_induced, H1_induced][inv_check_id]
    indicator_check_idx_inv = [indicator_idx_1, indicator_idx_2][inv_check_id]
    indicator_check_idx_permuted = [indicator_idx_2, indicator_idx_1][inv_check_id]

    null_check_inv, null_check_permuted, null_check_gauge, links_permuted, weight = find_low_weight_gauge(H_ind_inv, H_ind_permuted, indicator_check_idx_permuted, tn= 100)

    if verbose:
        print('\n\tAdding final bridges')
        print('\t\tfinal_checks_1:', final_checks_1)
        print('\t\tfinal_qbs_1:', final_qbs_1)
        print('\t\tfinal_checks_2:', final_checks_2)
        print('\t\tfinal_qbs_2:', final_qbs_2)

    links_inv = np.eye(len(H_ind_inv), dtype=int)
    links_1 = [links_inv, links_permuted][inv_check_id]
    links_2 = [links_permuted, links_inv][inv_check_id]
    null_check_1 = [null_check_inv, null_check_permuted][inv_check_id]
    null_check_2 = [null_check_permuted, null_check_inv][inv_check_id]

    # Expand Z checks to gauge qbs
    final_checks_2_updated = [z_check + num_z_checks_1 for z_check in final_checks_2]
    Hz_combined[final_checks_1, total_qbs:total_qbs+num_bridge_qbs] = links_1
    Hz_combined[final_checks_2_updated, total_qbs:total_qbs+num_bridge_qbs] = links_2

    # Add X checks to the gauge dof
    # print(final_qbs_1.shape, final_qbs_2.shape, null_check_1.shape, null_check_2.shape, null_check_gauge.shape)
    final_qbs_2_updated = [q_check + num_qbs_1 for q_check in final_qbs_2]
    Hx_combined[-(num_bridge_qbs-1):, final_qbs_1] = null_check_1
    Hx_combined[-(num_bridge_qbs-1):, final_qbs_2_updated] = null_check_2
    Hx_combined[-(num_bridge_qbs-1):, -num_bridge_qbs:] = null_check_gauge
    
    
    # calculating bridged duals
    indicator_check_inv = [indicated_check_1, indicated_check_2 + num_z_checks_1][inv_check_id]
    indicator_check_permuted = [indicated_check_2 + num_z_checks_1, indicated_check_1][inv_check_id]


    bridge_indicated_perm = array_to_idx(Hz_combined[indicator_check_permuted][-num_bridge_qbs:])[0]
    bridge_indicated_inv = array_to_idx(Hz_combined[indicator_check_inv][-num_bridge_qbs:])[0]


    bridge_duals = qbs_legend_combined['dual_logical'][1]
    final_x_qbs_1 = checks_legend_1['ckbb_layers'][-2][1] if len(checks_legend_1['ckbb_layers']) > 1 else checks_legend_1['induced'][1]
    final_x_qbs_2 = right_checks_legend_updated['ckbb_layers'][-2][1] if len(right_checks_legend_updated['ckbb_layers']) > 1 else right_checks_legend_updated['induced'][1]
    final_qbs_inv = [final_x_qbs_1, final_x_qbs_2][inv_check_id]

    bridge_duals = np.concatenate((bridge_duals, [bridge_indicated_perm + num_qbs_1 + num_qbs_2]))
    bridge_duals = np.concatenate((bridge_duals, final_qbs_inv[bridge_indicated_inv:bridge_indicated_perm]))
    bridge_duals = np.concatenate((bridge_duals, final_qbs_inv[bridge_indicated_perm:bridge_indicated_inv]))


    # bookkeeping
    qbs_legend_combined['bridge'] = np.arange(total_qbs, total_qbs + num_bridge_qbs)
    bridge_checks = np.arange(total_x_checks, total_x_checks + num_bridge_qbs - 1, dtype=int)
    checks_legend_combined['bridge_checks'] = (logical_types[1], np.arange(total_x_checks, total_x_checks + num_bridge_qbs - 1, dtype=int))
    checks_legend_combined[f'H{logical_types[1]}_ancilla'] = np.hstack(
        (checks_legend_combined[f'H{logical_types[1]}_ancilla'], bridge_checks))
    qbs_legend_combined['dual_logical'] = (
        logical_types[1],
        bridge_duals.astype(int)
    )

    if verbose and verbose_print_matrices:
        print('\n\tchecks_legend_combined[Hz]:', checks_legend_combined['Hz'])
        print('\tchecks_legend_combined[Hz_ancilla]:', checks_legend_combined['Hz_ancilla'])
        print('\tHz_1.shape[0]:', Hz_1.shape[0], 'num_z_checks_1:', num_z_checks_1)
        print('\tnum_z_checks_1+Hz_2.shape[0]:', num_z_checks_1 + Hz_2.shape[0], 'Hz_combined.shape[0]:', Hz_combined.shape[0])

    extra_z_stab = Hz_combined[checks_legend_combined['Hz_ancilla'], :]

    non_zero_stabs = np.where((extra_z_stab.sum(axis=0) % 2) != 0)[0]

    if verbose:
        print('\n\textra_z_stab:', non_zero_stabs)

    logical_supp_1 = np.where(logical_1 != 0)[0]
    logical_supp_2 = np.where(logical_2 != 0)[0] + num_qbs_1
    if verbose:
        print('\t\tlogical_1:', logical_supp_1)
        print('\t\tlogical_2:', logical_supp_2)

    if assertion_check:
        assert set(non_zero_stabs) == (set(logical_supp_1) | set(logical_supp_2)), 'Stabilizer product is not the desired logical operator'

    # check commutation
    if assertion_check:
        assert (Hx_combined[:total_x_checks, :total_qbs] @ Hz_combined[:total_z_checks, :total_qbs].T % 2 == 0).all(), 'Stacked Hx and Hz do not commute'
    if verbose:
        print('\t\tAdded stabs commute')

    if verbose and verbose_print_matrices:
        print('\n\tHx_combined:')
        pretty_print_matrix(Hx_combined)
        print('\n\tHz_combined:')
        pretty_print_matrix(Hz_combined)
    if assertion_check:
        assert (Hx_combined @ Hz_combined.T % 2 == 0).all(), f'Bridged Hx and Hz do not commute, {np.where(Hx_combined @ Hz_combined.T % 2 != 0)}'
    if verbose:
        print('\t\tAdded bridge checks commute')

    dual_1 = qbs_legend_1['logical'][1][-1] if dual_indicators[0] == -1 else dual_indicators[0]
    dual_2 = qbs_legend_2['logical'][1][-1] + num_qbs_1 if dual_indicators[1] == -1 else dual_indicators[1] + num_qbs_1

    dual_support = set(qbs_legend_combined['dual_logical'][1])
    dual_support.add(dual_1)
    dual_support.add(dual_2)
    dual_logical = np.array([[i in dual_support for i in range(Hz_combined.shape[1])]])

    if verbose:
        print('\t\tDual logical:', qbs_legend_combined['dual_logical'][1],dual_1, dual_2+num_qbs_1)
    if verbose_print_matrices:
        pretty_print_matrix(dual_logical)
    
    if not ((Hz_combined[checks_legend_combined['Hz_ancilla']] @ dual_logical.T) % 2 == 0).all():
        # rerun everything  
        return find_joint_ancilla(checks, logicals, logical_type, num_layers, dual_indicators, return_legends, verbose, verbose_print_matrices, assertion_check)
        print('!!!Dual logical does not commute with the anillae!!!')
        print('Failed ancillae:', checks_legend_combined['Hz_ancilla'][np.where(Hz_combined[[checks_legend_combined['Hz_ancilla']][0]] @ dual_logical.T % 2 != 0)[0]])
    
    if assertion_check:
        assert ((Hz_combined[checks_legend_combined['Hz_ancilla']] @ dual_logical.T) % 2 == 0).all(), f'Dual logical does not commute with the anillae'

    if verbose:
        print('\t\tExtendted logical dual logical operator commutes with the anillae')
    


    # check dof
    dof_original = Hx_1.shape[1] - rank(Hx_1) - rank(Hz_1) + Hx_2.shape[1] - rank(Hx_2) - rank(Hz_2)
    dof_new = Hx_combined.shape[1] - rank(Hx_combined) - rank(Hz_combined)

    if assertion_check:
        assert dof_new == dof_original - 1, 'Incorrect number of gauge qubits'
    if verbose:
        print('\n\tNumber of logical qubits before:', dof_original)
        print('\tNumber of logical qubits after:', dof_new)

    if return_legends:
        return Hx_combined, Hz_combined, checks_legend_combined, qbs_legend_combined
    else:
        return Hx_combined, Hz_combined

def find_joint_ancilla_with_incr(
    checks: list[np.ndarray],  # (Hx_1, Hz_1, Hx_2, Hz_2)
    logicals: list[np.ndarray],  # (logical_1, logical_2)
    logical_type: str = 'z',
    num_layers: list[int] = [-1, -1],  # (num_layers_1, num_layers_2)
    dual_indicators: list[int] = [-1, -1],
    incr: list[int] = [0,0],
    return_legends: bool = False,
    verbose: bool = False,
    verbose_print_matrices: bool = False,
    assertion_check: bool = False
):
    '''
    Find a single ancilla qubit that is only connected to one check
    '''
    if verbose:
        print(f"\n[find_joint_ancilla] Inputs summary: checks: {[c.shape for c in checks]}, logicals: {[l.shape if hasattr(l, 'shape') else type(l) for l in logicals]}, logical_type: {logical_type}, num_layers: {num_layers}, dual_indicators: {dual_indicators}, incr: {incr}")
        print(f"\tlogical_1: {logicals[0]}")
        print(f"\tlogical_2: {logicals[1]}")        
        if verbose_print_matrices:
            print(f"\tchecks[0] (Hx_1):\n{checks[0]}")
            print(f"\tchecks[1] (Hz_1):\n{checks[1]}")
            print(f"\tchecks[2] (Hx_2):\n{checks[2]}")
            print(f"\tchecks[3] (Hz_2):\n{checks[3]}")


    Hx_1, Hz_1, Hx_2, Hz_2 = checks
    logical_types = ['z', 'x']

    if logical_type == 'x':
        Hx_1, Hz_1 = (Hz_1, Hx_1)
        Hx_2, Hz_2 = (Hz_2, Hx_2)
        logical_types = ['x', 'z']

    logical_1, logical_2 = logicals

    if verbose:
        print('\n\tProcessing...')
        if verbose_print_matrices:
            print('\t\tHx_1:\n', Hx_1)
            print('\t\tHz_1:\n', Hz_1)
            print('\t\tlogical_1:', logical_1)
            print('\t\tlogical_2:', logical_2)
            print('\t\tlogical_type:', logical_type)
            print('\t\tnum_layers:', num_layers)
            print('\t\tdual_indicators:', dual_indicators)

    Hx_1_adapted, Hz_1_adapted, checks_legend_1, qbs_legend_1 = find_single_ancilla(
        Hx_1, Hz_1, logical_1, logical_type, dual_indicator=dual_indicators[0], num_layers=num_layers[0],
        verbose=verbose, verbose_print_matrices=verbose_print_matrices, return_legends=True
    )
    Hx_2_adapted, Hz_2_adapted, checks_legend_2, qbs_legend_2 = find_single_ancilla(
        Hx_2, Hz_2, logical_2, logical_type, dual_indicator=dual_indicators[1], num_layers=num_layers[1],
        verbose=verbose, verbose_print_matrices=verbose_print_matrices, return_legends=True
    )


    if verbose and verbose_print_matrices:
        print('\n\tchecks_legend_1:', checks_legend_1)
        print('\tqbs_legend_1:', qbs_legend_1)
        print('\tchecks_legend_2:', checks_legend_2)
        print('\tqbs_legend_2:', qbs_legend_2)

    final_check_layer_1 = checks_legend_1['ckbb_layers'][-1]
    final_check_layer_2 = checks_legend_2['ckbb_layers'][-1]
    final_checks_1 = final_check_layer_1[1] 
    final_checks_2 = final_check_layer_2[1]
    checks_legend_1['ckbb_layers'][-1] = (final_check_layer_1[0], final_checks_1)
    checks_legend_2['ckbb_layers'][-1] = (final_check_layer_2[0], final_checks_2)

    final_qbs_1 = qbs_legend_1['ckbb_layers'][-1][1]
    final_qbs_2 = qbs_legend_2['ckbb_layers'][-1][1]

    num_bridge_qbs = min(sum(logical_1), sum(logical_2))
    num_x_checks_1, num_qbs_1 = Hx_1_adapted.shape
    num_x_checks_2, num_qbs_2 = Hx_2_adapted.shape

    num_z_checks_1 = Hz_1_adapted.shape[0]
    num_z_checks_2 = Hz_2_adapted.shape[0]

    total_x_checks = num_x_checks_1 + num_x_checks_2
    total_z_checks = num_z_checks_1 + num_z_checks_2
    total_qbs = num_qbs_1 + num_qbs_2

    Hx_combined = np.zeros((total_x_checks + num_bridge_qbs - 1, total_qbs + num_bridge_qbs), dtype=int)
    Hz_combined = np.zeros((total_z_checks, total_qbs + num_bridge_qbs), dtype=int)

    Hx_combined[:num_x_checks_1, :num_qbs_1] = Hx_1_adapted
    Hz_combined[:num_z_checks_1, :num_qbs_1] = Hz_1_adapted

    Hx_combined[num_x_checks_1:num_x_checks_1 + num_x_checks_2, num_qbs_1:num_qbs_1 + num_qbs_2] = Hx_2_adapted
    Hz_combined[num_z_checks_1:num_z_checks_1 + num_z_checks_2, num_qbs_1:num_qbs_1 + num_qbs_2] = Hz_2_adapted

    # labeling
    checks_legend_combined = {
        'Left code': checks_legend_1
    }
    right_checks_legend_updated = {
        'Hx_original': checks_legend_2['Hx_original'] + num_x_checks_1,
        'Hz_original': checks_legend_2['Hz_original'] + num_z_checks_1,
        'Hx_ancilla': checks_legend_2['Hx_ancilla'] + num_x_checks_1,
        'Hz_ancilla': checks_legend_2['Hz_ancilla'] + num_z_checks_1,
        'gauge_checks': (checks_legend_2['gauge_checks'][0], [x + num_x_checks_1 for x in checks_legend_2['gauge_checks'][1]]),
        'induced': (checks_legend_2['induced'][0], [x + num_x_checks_1 for x in checks_legend_2['induced'][1]]),
        'ckbb_layers': [(layer[0], [x + num_x_checks_1 for x in layer[1]]) for layer in checks_legend_2['ckbb_layers']],
        'indicated_checks': (checks_legend_2['indicated_checks'][0], [x + num_z_checks_1 for x in checks_legend_2['indicated_checks'][1]]),
    }
    checks_legend_combined['Right code'] = right_checks_legend_updated

    qbs_legend_combined = {
        'Left code': qbs_legend_1
    }
    right_qbs_legend_updated = {
        'original': qbs_legend_2['original'] + num_qbs_1,
        'logical': (qbs_legend_2['logical'][0], [x + num_qbs_1 for x in qbs_legend_2['logical'][1]]),
        'dual_logical': (qbs_legend_2['dual_logical'][0], [x + num_qbs_1 for x in qbs_legend_2['dual_logical'][1]]),
        'ancilla': qbs_legend_2['ancilla'] + num_qbs_1,
        'ckbb_layers': [(layer[0], [x + num_qbs_1 for x in layer[1]]) for layer in qbs_legend_2['ckbb_layers']],
    }
    qbs_legend_combined['Right code'] = right_qbs_legend_updated

    # agregating checks
    checks_legend_combined['Hz'] = np.concatenate(
        (checks_legend_1['Hz_original'],
         right_checks_legend_updated['Hz_original']))

    checks_legend_combined['Hx'] = np.concatenate(
        (checks_legend_1['Hx_original'],
         right_checks_legend_updated['Hx_original']))

    checks_legend_combined['Hz_ancilla'] = np.concatenate(
        (checks_legend_1['Hz_ancilla'],
         right_checks_legend_updated['Hz_ancilla']))

    checks_legend_combined['Hx_ancilla'] = np.concatenate(
        (checks_legend_1['Hx_ancilla'],
         right_checks_legend_updated['Hx_ancilla']))

    gauge_check_type = checks_legend_1["gauge_checks"][0]
    checks_legend_combined['gauge_checks'] = (
        gauge_check_type,
        np.concatenate(
            (checks_legend_1['gauge_checks'][1],
             right_checks_legend_updated['gauge_checks'][1])))

    gauge_ancilla_type = 'H' + gauge_check_type + '_ancilla'
    checks_legend_combined[gauge_ancilla_type] = np.concatenate((
        checks_legend_combined[gauge_ancilla_type],
        checks_legend_combined['gauge_checks'][1]
    ))

    # agregating qubits
    qbs_legend_combined['original'] = np.concatenate(
        (qbs_legend_1['original'],
         right_qbs_legend_updated['original']))

    qbs_legend_combined['ancilla'] = np.concatenate(
        (qbs_legend_1['ancilla'],
         right_qbs_legend_updated['ancilla']))

    qbs_legend_combined['logical'] = (
        qbs_legend_1['logical'][0],
        np.concatenate(
            (qbs_legend_1['logical'][1],
             right_qbs_legend_updated['logical'][1])))

    qbs_legend_combined['dual_logical'] = (
        qbs_legend_1['dual_logical'][0],
        np.concatenate(
            (qbs_legend_1['dual_logical'][1],
             right_qbs_legend_updated['dual_logical'][1])))


    final_checks_1 = np.roll(final_checks_1, -incr[0])
    final_checks_2 = np.roll(final_checks_2, -incr[1])
    H1_induced = Hz_1_adapted[:, final_qbs_1]
    H2_induced = Hz_2_adapted[:, final_qbs_2]

    if verbose:
        print('\n\tAdding final bridges')
        print('\t\tfinal_checks_1:', final_checks_1)
        print('\t\tfinal_qbs_1:', final_qbs_1)
        print('\t\tfinal_checks_2:', final_checks_2)
        print('\t\tfinal_qbs_2:', final_qbs_2)

    qubit_degree_1 = H1_induced[final_checks_1[:num_bridge_qbs]].sum(axis=0)
    qubit_degree_2 = H2_induced[final_checks_2[:num_bridge_qbs]].sum(axis=0)

    H1_qubits_used = []
    H2_qubits_used = []

    for bridge_check in range(num_bridge_qbs - 1):
        if verbose:
            print(f'\n\tBridge check {bridge_check}')
        final_qubits = [total_qbs + bridge_check, total_qbs + bridge_check + 1]

        # find qubit from block 1
        intersected_1 = np.where((H1_induced[final_checks_1[bridge_check]] + H1_induced[final_checks_1[bridge_check + 1]]) == 2)[0]
        if verbose:
            print('\t\tintersected_1 at', intersected_1)
        for qubit in intersected_1:
            if verbose:
                print(f'\t\t\tchecking qubit {qubit}, degree {qubit_degree_1[qubit]}')
            if final_qbs_1[qubit] in H1_qubits_used:
                if verbose:
                    print('\t\t\tqubit already used')
                continue
            if qubit_degree_1[qubit] == 2:
                H1_qubits_used.append(final_qbs_1[qubit])
                final_qubits.append(final_qbs_1[qubit])
                if verbose:
                    print('\t\t\tadded qubit', qubit, final_qubits)
                break
        if len(final_qubits) != 3:
            return find_joint_ancilla_with_incr(
                checks,
                logicals,
                logical_type,
                num_layers,
                dual_indicators,
                incr,
                return_legends,
                verbose,
                verbose_print_matrices,
                assertion_check
            )

        # find qubit from block 2
        intersected_2 = np.where(H2_induced[final_checks_2[bridge_check]] + H2_induced[final_checks_2[bridge_check + 1]] == 2)[0]
        if verbose:
            print('\t\tintersected_2 at', intersected_2)
        for qubit in intersected_2:
            if verbose:
                print(f'\t\t\tchecking qubit {qubit}, degree {qubit_degree_2[qubit]}')
            if final_qbs_2[qubit] + num_qbs_1 in H2_qubits_used:
                if verbose:
                    print('\t\t\tqubit already used')
                continue
            if qubit_degree_2[qubit] == 2:
                H2_qubits_used.append(final_qbs_2[qubit] + num_qbs_1)
                final_qubits.append(final_qbs_2[qubit] + num_qbs_1)
                if verbose:
                    print('\t\t\tadded qubit', qubit, final_qubits)
                break
        if verbose:
            print('\t\tfinal_qubits:', final_qubits)

        if len(final_qubits) != 4:
            return find_joint_ancilla_with_incr(
                checks,
                logicals,
                logical_type,
                num_layers,
                dual_indicators,
                [incr[0], incr[1] + 1],
                return_legends,
                verbose,
                verbose_print_matrices,
                assertion_check
            )

        Hx_combined[total_x_checks + bridge_check, final_qubits] = 1

    # calculating bridged duals

    indicated_check_1 = checks_legend_1['indicated_checks'][1][-1]
    indicated_check_2 = checks_legend_2['indicated_checks'][1][-1]

    ind_check_pos_1 = np.where(final_checks_1 == indicated_check_1)[0][0]
    ind_check_pos_2 = np.where(final_checks_2 == indicated_check_2)[0][0]

    min_ind_check_pos = min([ind_check_pos_1, ind_check_pos_2])
    max_ind_check_pos = max([ind_check_pos_1, ind_check_pos_2])

    qubits_late = [H1_qubits_used, H2_qubits_used][np.argmax([ind_check_pos_1, ind_check_pos_2])]

    bridged_duals = [min_ind_check_pos + total_qbs] + qubits_late[min_ind_check_pos:max_ind_check_pos]

    # adding identity bridge checks
    Hz_combined[
        final_checks_1[:num_bridge_qbs],  # final Z checks in block 1
        total_qbs:  #  New bridge qubits
    ] = np.eye(num_bridge_qbs, dtype=int)

    Hz_combined[
        num_z_checks_1 + final_checks_2[:num_bridge_qbs],  # final Z checks in block 2
        total_qbs:  #  New bridge qubits
    ] = np.eye(num_bridge_qbs, dtype=int)

    # bookkeeping
    qbs_legend_combined['bridge'] = np.arange(total_qbs, total_qbs + num_bridge_qbs)
    bridge_checks = np.arange(total_x_checks, total_x_checks + num_bridge_qbs - 1, dtype=int)
    checks_legend_combined['bridge_checks'] = (logical_types[1], np.arange(total_x_checks, total_x_checks + num_bridge_qbs - 1, dtype=int))
    checks_legend_combined[f'H{logical_types[1]}_ancilla'] = np.hstack(
        (checks_legend_combined[f'H{logical_types[1]}_ancilla'], bridge_checks))
    qbs_legend_combined['dual_logical'] = (
        logical_types[1],
        np.concatenate((qbs_legend_combined['dual_logical'][1], bridged_duals))
    )

    if verbose and verbose_print_matrices:
        print('\n\tchecks_legend_combined[Hz]:', checks_legend_combined['Hz'])
        print('\tchecks_legend_combined[Hz_ancilla]:', checks_legend_combined['Hz_ancilla'])
        print('\tHz_1.shape[0]:', Hz_1.shape[0], 'num_z_checks_1:', num_z_checks_1)
        print('\tnum_z_checks_1+Hz_2.shape[0]:', num_z_checks_1 + Hz_2.shape[0], 'Hz_combined.shape[0]:', Hz_combined.shape[0])

    extra_z_stab = Hz_combined[checks_legend_combined['Hz_ancilla'], :]
    non_zero_stabs = np.where((extra_z_stab.sum(axis=0) % 2) != 0)[0]

    if verbose:
        print('\n\textra_z_stab:', non_zero_stabs)

    logical_supp_1 = np.where(logical_1 != 0)[0]
    logical_supp_2 = np.where(logical_2 != 0)[0] + num_qbs_1
    if verbose:
        print('\t\tlogical_1:', logical_supp_1)
        print('\t\tlogical_2:', logical_supp_2)

    if assertion_check:
        assert set(non_zero_stabs) == (set(logical_supp_1) | set(logical_supp_2)), 'Stabilizer product is not the desired logical operator'

    # check commutation
    assert (Hx_combined[:total_x_checks, :total_qbs] @ Hz_combined[:total_z_checks, :total_qbs].T % 2 == 0).all(), 'Stacked Hx and Hz do not commute'
    if verbose:
        print('\t\tAdded stabs commute')

    if verbose and verbose_print_matrices:
        print('\n\tHx_combined:')
        pretty_print_matrix(Hx_combined)
        print('\n\tHz_combined:')
        pretty_print_matrix(Hz_combined)
    if assertion_check:
        assert (Hx_combined @ Hz_combined.T % 2 == 0).all(), 'Bridged Hx and Hz do not commute'
    if verbose:
        print('\t\tAdded bridge checks commute')

    dual_1 = qbs_legend_1['logical'][1][-1] if dual_indicators[0] == -1 else dual_indicators[0]
    dual_2 = qbs_legend_2['logical'][1][-1] + num_qbs_1 if dual_indicators[1] == -1 else dual_indicators[1] + num_qbs_1

    dual_support = set(qbs_legend_combined['dual_logical'][1])
    dual_support.add(dual_1)
    dual_support.add(dual_2)
    dual_logical = np.array([[i in dual_support for i in range(Hz_combined.shape[1])]], dtype=int)

    if verbose:
        print('\t\tDual logical:', qbs_legend_combined['dual_logical'][1],dual_1, dual_2+num_qbs_1)
    if verbose_print_matrices:
        pretty_print_matrix(dual_logical)
    
    if not ((Hz_combined[checks_legend_combined['Hz_ancilla']] @ dual_logical.T) % 2 == 0).all():
        print('!!!Dual logical does not commute with the anillae!!!')
        print('Failed ancillae:', checks_legend_combined['Hz_ancilla'][np.where(Hz_combined[[checks_legend_combined['Hz_ancilla']][0]] @ dual_logical.T % 2 != 0)[0]])
    
    if assertion_check:
        assert ((Hz_combined[checks_legend_combined['Hz_ancilla']] @ dual_logical.T) % 2 == 0).all(), f'Dual logical does not commute with the anillae'

    if verbose:
        print('\t\tExtendted logical dual logical operator commutes with the anillae')
    


    # check dof
    dof_original = Hx_1.shape[1] - rank(Hx_1) - rank(Hz_1) + Hx_2.shape[1] - rank(Hx_2) - rank(Hz_2)
    dof_new = Hx_combined.shape[1] - rank(Hx_combined) - rank(Hz_combined)

    if assertion_check:
        assert dof_new == dof_original - 1, 'Incorrect number of gauge qubits'
    if verbose:
        print('\n\tNumber of logical qubits before:', dof_original)
        print('\tNumber of logical qubits after:', dof_new)

    if return_legends:
        return Hx_combined, Hz_combined, checks_legend_combined, qbs_legend_combined
    else:
        return Hx_combined, Hz_combined

def find_joint_ancilla_with_physical(
    Hx, 
    Hz, 
    logical, 
    logical_type = 'z', 
    num_layers = -1,
    dual_indicator = -1, 
    return_legends = False, 
    verbose = False,
    verbose_print_matrices = False
):
    '''
    Find an ancilla to jointly measure with a physical qubit
    '''
    logical_z_supp = np.where(logical == 1)[0]
    indicated_dual = logical_z_supp[dual_indicator]
    logical_types = ['z', 'x'] 
    if logical_type == 'x':
        logical_types = ['x', 'z']
        Hz, Hx = Hx, Hz
    

    Hx_adapted, Hz_adapted, checks_legend, qbs_legend = find_single_ancilla(
        Hx, Hz, logical, logical_type, dual_indicator=dual_indicator, num_layers=num_layers,
        verbose=verbose, verbose_print_matrices=verbose_print_matrices, return_legends=True
    )    
    Hx_new = np.zeros((Hx_adapted.shape[0], Hx_adapted.shape[1] + 1))
    Hz_new = np.zeros((Hz_adapted.shape[0], Hz_adapted.shape[1] + 1))

    indicated_check = checks_legend['indicated_checks'][1][-1]
    Hx_new[:, :-1] = Hx_adapted
    Hz_new[:, :-1] = Hz_adapted
    Hz_new[indicated_check, -1] = 1


    qbs_legend['dual_logical'] = (qbs_legend['dual_logical'][0], [int(x) for x in np.concatenate((qbs_legend['dual_logical'][1], [Hz_new.shape[1] - 1]))])
    qbs_legend['physical'] = [Hz_new.shape[1] - 1]
    
    
    
    # validation
    # sum of Z checks gives the Z operator
    combined_z_stab = Hz_new[Hz.shape[0]:].sum(axis=0) % 2
    combined_z_stab = np.where(combined_z_stab != 0)[0]
    logical_z_supp_comb = np.concatenate((logical_z_supp, [Hz_new.shape[1] - 1]))
    if verbose:
        print('\n\tThe logical operators are:')
        print('\t\tcombined_z_stab:', combined_z_stab)
        print('\t\tlogical_z_supp:', logical_z_supp_comb)
    assert set(combined_z_stab) == set(logical_z_supp_comb), 'Stabilizer product is not the desired logical operator'

    # check commutation
    assert (Hx_new @ Hz_new.T % 2 == 0).all(), 'New Hx and Hz do not commute'

    # check dof
    dof_original = Hx.shape[1] - rank(Hx) - rank(Hz)
    dof_new = Hx_new.shape[1] - rank(Hx_new) - rank(Hz_new)
    assert dof_new == dof_original, 'Incorrect number of gauge qubits'

    if return_legends:
        return Hx_new, Hz_new, checks_legend, qbs_legend
    else:
        return Hx_new, Hz_new

################################################################################
# Constructing Stim circuits for the injection
################################################################################

def get_cx_layers(
        Hx: np.ndarray,
        Hz: np.ndarray,
        X_anc_indices: list[int],
        Z_anc_indices: list[int],
        data_indices: list[int],
        given_layers: list[list[tuple[int, int]]] = [],
    ) -> list[list[tuple[int, int]]]:
    # assign checks into layers
    rx, n = Hx.shape
    rz, _ = Hz.shape
    checks_x = [(X_anc_indices[a], data_indices[d]) for a in range(rx) for d in range(n) if Hx[a,d]]
    checks_z = [(data_indices[d], Z_anc_indices[a]) for a in range(rz) for d in range(n) if Hz[a,d]]

    graph_x = nx.Graph(checks_x)
    coloring_x = nx.coloring.greedy_color(nx.line_graph(graph_x), strategy='smallest_last')
    num_colors_x = max(coloring_x.values()) + 1

    graph_z = nx.Graph(checks_z)
    coloring_z = nx.coloring.greedy_color(nx.line_graph(graph_z), strategy='smallest_last')
    num_colors_z = max(coloring_z.values()) + 1
    
    layers = [[] for _ in range(num_colors_x + num_colors_z)]

    for edge, color in coloring_x.items():
        q0,q1 = edge
        layers[color].append((q0,q1) if edge in checks_x else (q1,q0))
    for edge, color in coloring_z.items():
        q0,q1 = edge
        layers[num_colors_x + color].append((q0,q1) if edge in checks_z else (q1,q0))

    assert set(pair for layer in layers for pair in layer) == set(checks_x) | set(checks_z), "Some checks missing"

    return layers

def syndrome_extraction_circuit(
        cx_layers: list[list[tuple[int, int]]],
        data_indices: list[int],
        X_anc_indices: list[int],
        Z_anc_indices: list[int],
        meas_rec: dict[int, list[int]],
        global_meas_counter: int,
        init_error: float,
        cx_error: float,
        meas_error: float,
        first_round_dets_deterministic: list[int],
    ) -> tuple[stim.Circuit, dict[int, list[int]], int]:
    circ = stim.Circuit()

    X_ancillae = list(sorted(set(qubit for layer in cx_layers for pair in layer for qubit in pair if qubit in X_anc_indices)))
    Z_ancillae = list(sorted(set(qubit for layer in cx_layers for pair in layer for qubit in pair if qubit in Z_anc_indices)))
    if X_ancillae:
        circ.append('RX', X_ancillae, (), tag='"X ancilla qubit init"')
        circ.append('Z_ERROR', X_ancillae, init_error)
    if Z_ancillae:
        circ.append('R', Z_ancillae, (), tag='"Z ancilla qubit init"')
        circ.append('X_ERROR', Z_ancillae, init_error)
    for layer in cx_layers:
        # apply CNOTs
        qubits = [qubit for pair in layer for qubit in pair]
        circ.append('CX', qubits, ())
        circ.append('DEPOLARIZE2', qubits, cx_error*15/16)
        
    # measure ancilla
    if X_ancillae:
        circ.append('Z_ERROR', X_ancillae, meas_error, tag='"X ancilla qubit measure"')
        circ.append('MX', X_ancillae, ())
    if Z_ancillae:
        circ.append('X_ERROR', Z_ancillae, meas_error, tag='"Z ancilla qubit measure"')
        circ.append('M', Z_ancillae, ())
    for a in X_ancillae + Z_ancillae:
        global_meas_counter += 1
        meas_rec.setdefault(a, []).append(global_meas_counter)

    for a in Z_ancillae + X_ancillae:
        if len(meas_rec[a]) == 1 and a in first_round_dets_deterministic:
            circ.append('DETECTOR', [
                stim.target_rec(meas_rec[a][-1] - global_meas_counter - 1)
            ], (a, len(meas_rec[a]) - 1), tag='"deterministic stabilizer"')
        elif len(meas_rec[a]) > 1:
            circ.append('DETECTOR', [
                stim.target_rec(meas_rec[a][-1] - global_meas_counter - 1),
                stim.target_rec(meas_rec[a][-2] - global_meas_counter - 1)
            ], (a, len(meas_rec[a]) - 1), tag='"stabilizer consistency"')
    return circ, meas_rec, global_meas_counter

def generate_injection_stim(
        p: float,
        qLDPC_Hx: np.ndarray,
        qLDPC_Hz: np.ndarray,
        qLDPC_Lx: np.ndarray,
        qLDPC_Lz: np.ndarray,
        qLDPC_idx_to_inject: int,
        qLDPC_cx_layers: list[list[tuple[int, int]]] | None, # TODO use these
        qLDPC_qubit_legends: dict[str, dict[int, int]] | None,
        inj_Hx: np.ndarray,
        inj_Hz: np.ndarray,
        inj_Lx: np.ndarray,
        inj_Lz: np.ndarray,
        inj_cx_layers: list[list[tuple[int, int]]] | None,
        inj_qubit_legends: dict[str, dict[int, int]] | None,
        deformed_Hx: np.ndarray,
        deformed_Hz: np.ndarray,
        check_legend: dict,
        qbs_legend: dict,
        d_t: int,
        qldpc_is_left_code: bool = True,
        obs_basis: str = 'X',
        postselect_inj_checks: bool = False,
        postselect_anc_checks: bool = False,
        postselect_qLDPC_checks: bool = False,
    ) -> tuple[stim.Circuit, dict]:
    """Construct the Stim circuit implementing magic state injection from some
    base code into a qLDPC code, following the construction in
    "Constant-Overhead Magic State Injection into qLDPC Codes with Error
    Independence Guarantees".

    We assume that the injection code is a [[n_inj, 1, d_inj]] code, and the
    qLDPC code is a [[n_qldpc, k_qldpc, d_qldpc]] code.

    Args:
        p: depolarizing error rate to use for all gates.
        qLDPC_Hx: (rx_qldpc, n_qldpc) array, where the 1s in each row indicate
            which data qubits are part of that X stabilizer of the qLDPC code.
        qLDPC_Hz: (rz_qldpc, n_qldpc) array, where the 1s in each row indicate
            which data qubits are part of that Z stabilizer of the qLDPC code.
        qLDPC_Lx: (k_qldpc, n_qldpc) array, where the 1s in each row indicate
            which data qubits are part of that X logical operator of the qLDPC
            code.
        qLDPC_Lz: (k_qldpc, n_qldpc) array, where the 1s in each row indicate
            which data qubits are part of that Z logical operator of the qLDPC
            code.
        qLDPC_idx_to_inject: index of logical qubit of the qLDPC code to inject
            into.
        inj_Hx: (rx_inj, n_inj) array, where the 1s in each row indicate which
            data qubits are part of that X stabilizer of the injection code.
        inj_Hz: (rz_inj, n_inj) array, where the 1s in each row indicate which
            data qubits are part of that Z stabilizer of the injection code.
        inj_Lx: (n_inj,) array, where the 1s indicate which data qubits are part
            of the X logical operator of the injection code.
        inj_Lz: (n_inj,) array, where the 1s indicate which data qubits are part
            of the Z logical operator of the injection code.
        deformed_Hx: (rx_deformed, n_total) array, where the 1s in each row
            indicate which data qubits are part of that X stabilizer of the
            deformed code (the inj and qLDPC combined with an ancilla system for
            measuring joint ZZ operators).
        deformed_Hz: (rz_deformed, n_total) array, where the 1s in each row
            indicate which data qubits are part of that Z stabilizer of the
            deformed code.
        check_legend: dictionary indicating which rows/checks of the deformed
            check matrices correspond to which parts of the construction.
        qbs_legend: dictionary indicating which columns/qubits of the deformed
            check matrices correspond to which parts of the construction.
        d_t: temporal code distance.
        qldpc_is_left_code: if True, the qLDPC code is the "left" code in the
            check_legend and qbs_legend dictionaries. If False, the qLDPC code
            is the "right" code.

    Key size variables (inferred from the shapes of the input arrays):
        - n_qldpc: number of data qubits in the qLDPC code
        - k_qldpc: number of logical qubits in the qLDPC code
        - rx_qldpc: number of X stabilizers in the qLDPC code
        - rz_qldpc: number of Z stabilizers in the qLDPC code
        - n_inj: number of data qubits in the injection code
        - rx_inj: number of X stabilizers in the injection code
        - rz_inj: number of Z stabilizers in the injection code
        - n_ancilla: number of *additional* data qubits in the ZZ measurement
            ancilla system
        - rx_ancilla: number of *additional* X stabilizers in the ZZ measurement
            ancilla system
        - rz_ancilla: number of *additional* Z stabilizers in the ZZ measurement
            ancilla system
    rx_deformed = rx_qldpc + rx_inj + rx_ancilla
    rz_deformed = rz_qldpc + rz_inj + rz_ancilla
    n_total = n_qldpc + n_inj + n_ancilla
    """
    if not qldpc_is_left_code:
        raise NotImplementedError('Currently only support the qLDPC code being the left code.')

    circuit = stim.Circuit()
    meas_rec = {}
    global_meas_counter = 0

    ############################################################################
    # Initializing variables
    ############################################################################
    num_extra_ancilla_system_data = deformed_Hx.shape[1] - (qLDPC_Hx.shape[1] + inj_Hx.shape[1])
    num_extra_ancilla_system_X_checks = deformed_Hx.shape[0] - (qLDPC_Hx.shape[0] + inj_Hx.shape[0])
    num_extra_ancilla_system_Z_checks = deformed_Hz.shape[0] - (qLDPC_Hz.shape[0] + inj_Hz.shape[0])

    # Checking inputs for consistency with expected qubit counts.
    # Some of these would need to be updated if we supported injecting
    # into multiple logical qubits at once.
    assert qLDPC_Hx.shape[1] == qLDPC_Hz.shape[1] == qLDPC_Lx.shape[1] == qLDPC_Lz.shape[1]
    assert inj_Hx.shape[1] == inj_Hz.shape[1] == inj_Lx.shape[0]

    if 'physical' not in qbs_legend:
        assert len(check_legend['Left code']['Hz_original']) == qLDPC_Hz.shape[0]
        assert len(check_legend['Left code']['Hx_original']) == qLDPC_Hx.shape[0]
        assert len(check_legend['Right code']['Hx_original']) == inj_Hx.shape[0]
        assert len(check_legend['Right code']['Hz_original']) == inj_Hz.shape[0]

        assert len(qbs_legend['Left code']['original']) == qLDPC_Hx.shape[1]
        assert set(idx for _,idcs in qbs_legend['Left code']['ckbb_layers'] for idx in idcs) == set(qbs_legend['Left code']['ancilla'])

        assert len(qbs_legend['Right code']['original']) == inj_Hx.shape[1]
        assert set(idx for _,idcs in qbs_legend['Right code']['ckbb_layers'] for idx in idcs) == set(qbs_legend['Right code']['ancilla'])
        assert set(qbs_legend['Left code']['original']) | set(qbs_legend['Right code']['original']) == set(qbs_legend['original'])
        assert set(idx for _,idcs in qbs_legend['Left code']['ckbb_layers'] for idx in idcs) | set(idx for _,idcs in qbs_legend['Right code']['ckbb_layers'] for idx in idcs) == set(qbs_legend['ancilla'])
        assert len(qbs_legend['original']) == qLDPC_Hx.shape[1] + inj_Hx.shape[1]
        assert len(qbs_legend['ancilla']) + len(qbs_legend['bridge']) == num_extra_ancilla_system_data
    else:
        print('physical: ', qbs_legend['physical'])

    # We will assign qubit indices according to deformed_Hx and deformed_Hz.
    # This will also need to be updated if we support injecting into
    # multiple logical qubits at once.
    data_indices_deformed = [i for i in range(deformed_Hx.shape[1])]
    X_anc_indices_deformed = [i+deformed_Hx.shape[1] for i in range(deformed_Hx.shape[0])]
    Z_anc_indices_deformed = [i+deformed_Hx.shape[1]+deformed_Hz.shape[0] for i in range(deformed_Hz.shape[0])]

    if 'physical' not in qbs_legend:
        data_indices_qldpc = [data_indices_deformed[qbs_legend['Left code']['original'][i]] for i in range(qLDPC_Hx.shape[1])]
        X_anc_indices_qldpc = [X_anc_indices_deformed[check_legend['Left code']['Hx_original'][i]] for i in range(qLDPC_Hx.shape[0])]
        Z_anc_indices_qldpc = [Z_anc_indices_deformed[check_legend['Left code']['Hz_original'][i]] for i in range(qLDPC_Hz.shape[0])]

        data_indices_inj = [data_indices_deformed[qbs_legend['Right code']['original'][i]] for i in range(inj_Hx.shape[1])]
        X_anc_indices_inj = [X_anc_indices_deformed[check_legend['Right code']['Hx_original'][i]] for i in range(inj_Hx.shape[0])]
        Z_anc_indices_inj = [Z_anc_indices_deformed[check_legend['Right code']['Hz_original'][i]] for i in range(inj_Hz.shape[0])]
    else:
        data_indices_qldpc = [data_indices_deformed[qbs_legend['original'][i]] for i in range(qLDPC_Hx.shape[1])]
        X_anc_indices_qldpc = [X_anc_indices_deformed[check_legend['Hx_original'][i]] for i in range(qLDPC_Hx.shape[0])]
        Z_anc_indices_qldpc = [Z_anc_indices_deformed[check_legend['Hz_original'][i]] for i in range(qLDPC_Hz.shape[0])]

        data_indices_inj = qbs_legend['physical']
        X_anc_indices_inj = []
        Z_anc_indices_inj = []

    data_indices_ancilla = [q for q in data_indices_deformed if q not in data_indices_qldpc and q not in data_indices_inj]
    X_anc_indices_ancilla = [q for q in X_anc_indices_deformed if q not in X_anc_indices_qldpc and q not in X_anc_indices_inj]
    Z_anc_indices_ancilla = [q for q in Z_anc_indices_deformed if q not in Z_anc_indices_qldpc and q not in Z_anc_indices_inj]

    Z_stabs_for_ZZ = set(Z_anc_indices_deformed) - set(Z_anc_indices_qldpc) - set(Z_anc_indices_inj)

    assert len(data_indices_qldpc) + len(data_indices_inj) == len(data_indices_deformed) - num_extra_ancilla_system_data
    assert len(X_anc_indices_qldpc) + len(X_anc_indices_inj) == len(X_anc_indices_deformed) - num_extra_ancilla_system_X_checks
    assert len(Z_anc_indices_qldpc) + len(Z_anc_indices_inj) == len(Z_anc_indices_deformed) - num_extra_ancilla_system_Z_checks

    # # TEMP: turn off all ancilla-system checks and qubits
    # deformed_Hx = np.delete(deformed_Hx, [i for i,q in enumerate(X_anc_indices_deformed) if q not in X_anc_indices_qldpc and q not in X_anc_indices_inj], axis=0)
    # deformed_Hx = np.delete(deformed_Hx, [i for i,q in enumerate(data_indices_deformed) if q not in data_indices_qldpc and q not in data_indices_inj], axis=1)
    # deformed_Hz = np.delete(deformed_Hz, [i for i,q in enumerate(Z_anc_indices_deformed) if q not in Z_anc_indices_qldpc and q not in Z_anc_indices_inj], axis=0)
    # deformed_Hz = np.delete(deformed_Hz, [i for i,q in enumerate(data_indices_deformed) if q not in data_indices_qldpc and q not in data_indices_inj], axis=1)
    # print(data_indices_deformed)
    # print(X_anc_indices_deformed)
    # print(Z_anc_indices_deformed)
    # data_idx_to_qubit_idx_deformed_copy = []
    # for i,q in enumerate(data_indices_deformed):
    #     if q in data_indices_qldpc or q in data_indices_inj:
    #         data_idx_to_qubit_idx_deformed_copy.append(q)
    # data_indices_deformed = data_idx_to_qubit_idx_deformed_copy
    # X_anc_idx_to_qubit_idx_deformed_copy = []
    # for i,q in enumerate(X_anc_indices_deformed):
    #     if q in X_anc_indices_qldpc or q in X_anc_indices_inj:
    #         X_anc_idx_to_qubit_idx_deformed_copy.append(q)
    # X_anc_indices_deformed = X_anc_idx_to_qubit_idx_deformed_copy
    # Z_anc_idx_to_qubit_idx_deformed_copy = []
    # for q in Z_anc_indices_deformed:
    #     if q in Z_anc_indices_qldpc or q in Z_anc_indices_inj:
    #         Z_anc_idx_to_qubit_idx_deformed_copy.append(q)
    # Z_anc_indices_deformed = Z_anc_idx_to_qubit_idx_deformed_copy
    # # TEMP: turn off bridge checks and qubits
    # deformed_Hx = np.delete(deformed_Hx, check_legend['bridge_checks'][1], axis=0)
    # deformed_Hx = np.delete(deformed_Hx, qbs_legend['bridge'], axis=1)
    # deformed_Hz = np.delete(deformed_Hz, qbs_legend['bridge'], axis=1)
    # print(data_indices_deformed)
    # print(X_anc_indices_deformed)
    # print(Z_anc_indices_deformed)
    # data_idx_to_qubit_idx_deformed_copy = []
    # for i,q in enumerate(data_indices_deformed):
    #     if i not in qbs_legend['bridge']:
    #         data_idx_to_qubit_idx_deformed_copy.append(q)
    # data_indices_deformed = data_idx_to_qubit_idx_deformed_copy
    # X_anc_idx_to_qubit_idx_deformed_copy = []
    # for i,q in enumerate(X_anc_indices_deformed):
    #     if i not in check_legend['bridge_checks'][1]: 
    #         X_anc_idx_to_qubit_idx_deformed_copy.append(q)
    # X_anc_indices_deformed = X_anc_idx_to_qubit_idx_deformed_copy
    print(data_indices_deformed)
    print(X_anc_indices_deformed)
    print(Z_anc_indices_deformed)
    Z_stabs_for_ZZ = set(Z_anc_indices_deformed) - set(Z_anc_indices_qldpc) - set(Z_anc_indices_inj)
    print(Z_stabs_for_ZZ)

    # TODO: if we are doing surface code, special set of CX layers to avoid hook
    # errors
    use_given_qLDPC_layers = qLDPC_cx_layers is not None
    if qLDPC_cx_layers:
        cx_layers_qldpc = []
        for layer in qLDPC_cx_layers:
            new_layer = []
            for (q1,q2) in layer:
                if q1 in qLDPC_qubit_legends['data']:
                    assert q2 in qLDPC_qubit_legends['Z_anc']
                    new_layer.append((data_indices_qldpc[qLDPC_qubit_legends['data'].index(q1)], Z_anc_indices_qldpc[qLDPC_qubit_legends['Z_anc'].index(q2)]))
                else:
                    assert q1 in qLDPC_qubit_legends['X_anc'] and q2 in qLDPC_qubit_legends['data']
                    new_layer.append((X_anc_indices_qldpc[qLDPC_qubit_legends['X_anc'].index(q1)], data_indices_qldpc[qLDPC_qubit_legends['data'].index(q2)]))
            cx_layers_qldpc.append(new_layer)
    else:
        cx_layers_qldpc = get_cx_layers(qLDPC_Hx, qLDPC_Hz, X_anc_indices_qldpc, Z_anc_indices_qldpc, data_indices_qldpc)
    
    use_given_inj_layers = inj_cx_layers is not None
    if inj_cx_layers:
        cx_layers_inj = []
        for layer in inj_cx_layers:
            new_layer = []
            for (q1,q2) in layer:
                if q1 in inj_qubit_legends['data']:
                    assert q2 in inj_qubit_legends['Z_anc']
                    new_layer.append((data_indices_inj[inj_qubit_legends['data'].index(q1)], Z_anc_indices_inj[inj_qubit_legends['Z_anc'].index(q2)]))
                else:
                    assert q1 in inj_qubit_legends['X_anc'] and q2 in inj_qubit_legends['data']
                    new_layer.append((X_anc_indices_inj[inj_qubit_legends['X_anc'].index(q1)], data_indices_inj[inj_qubit_legends['data'].index(q2)]))
            cx_layers_inj.append(new_layer)
    else:
        cx_layers_inj = get_cx_layers(inj_Hx, inj_Hz, X_anc_indices_inj, Z_anc_indices_inj, data_indices_inj)
    
    cx_layers_deformed = get_cx_layers(
        deformed_Hx,
        deformed_Hz,
        X_anc_indices_deformed,
        Z_anc_indices_deformed,
        data_indices_deformed,
        given_layers = (cx_layers_qldpc if use_given_qLDPC_layers else []) + (cx_layers_inj if use_given_inj_layers else []),
    )

    ############################################################################
    # Building circuit
    ############################################################################

    postselection_mask = []

    ## Initialize magic states in +S state.
    # Assume noiseless b/c we are characterizing the noise of the injection, not
    # of the state prep.
    if obs_basis == 'X' or obs_basis == 'S':
        circuit.append('RX', data_indices_inj, (), tag='"inj data init"')
    if obs_basis == 'Z':
        circuit.append('RZ', data_indices_inj, (), tag='"inj data init"')
    if obs_basis == 'S':
        circuit.append('S', data_indices_inj, (), tag='"inj data init"')

    ## Initialize qLDPC data qubits in +
    circuit.append('RX', data_indices_qldpc, (), tag='"qldpc data init"')

    # First correction term (X measurement of each logical qubit) is trivial
    # because we initialized all logical qubits in +
    mu_1_targets = []

    ## ZZ measurement using the ancilla system
    circuit.append('RX', data_indices_ancilla, (), tag='"anc data init"')
    circuit.append('Z_ERROR', data_indices_ancilla, p)

    # Determine which checks should be deterministic in the first round. Those
    # that have support on data qubits initialized in the wrong basis are not
    # deterministic. This is the injection code Z checks that have support on
    # the ancilla system, and the ancilla system X checks that have support on
    # the injection code.

    first_round_dets_deterministic = set(X_anc_indices_qldpc + X_anc_indices_ancilla)
    if obs_basis == 'X':
        first_round_dets_deterministic |= set(X_anc_indices_inj)
    elif obs_basis == 'Z':
        first_round_dets_deterministic |= set(Z_anc_indices_inj)

    # d_t rounds of syndrome extraction
    for round_idx in range(d_t):
        # Measure stabilizers of whole deformed system
        circ, meas_rec, global_meas_counter = syndrome_extraction_circuit(
            cx_layers_deformed,
            data_indices=data_indices_deformed,
            X_anc_indices=X_anc_indices_deformed,
            Z_anc_indices=Z_anc_indices_deformed,
            meas_rec=meas_rec,
            global_meas_counter=global_meas_counter,
            init_error=p,
            cx_error=p,
            meas_error=p,
            first_round_dets_deterministic=list(first_round_dets_deterministic),
        )
        circuit += circ

    # TODO: in lattice surgery, when we measure out the glue data, the non-merge-basis
    # stabilizers on the edges (that go from weight 4 to weight 2) are no
    # longer consistent with their history, but rather the detector should
    # be comparing the new weight-2 measurement to the old weight-4
    # measurement PLUS the measured-out data qubits. This would be relevant here
    # too, if we do extra rounds on the qLDPC code after the ZZ measurement is
    # done. 

    # Second correction term is Zz measurements between each logical qubit and
    # its corresponding injection code. This is a product of Z stabilizers of
    # the deformed code, indicated by Z_stabs_for_ZZ.
    mu_2_targets = [
        (qubit, len(meas_rec[qubit])-1)
        for qubit in Z_stabs_for_ZZ
    ]

    # circuit.append('OBSERVABLE_INCLUDE', mu_2, 0)

    # Measure out all of the data qubits. This is technically three different
    # operations, but they commute so we can do them all at once.
    # 1) Measure out ancilla system data qubits in the X basis to complete the
    #    ZZ measurements.
    # 2) Measure out injection code data qubits in the X basis to complete the
    #    teleportation.
    # 3) Measure out qLDPC data qubits in the X basis to check the injected
    #    states for errors.
    # TODO: am i measuring ancilla system in the right basis?
    # TODO: do we need to account for frame changes here? Measuring ancilla
    # system in the X basis might give us some frame changes that we need to
    # apply to the X observable measurements of one or both of the other codes,
    # just like in lattice surgery.
    circuit.append('Z_ERROR', data_indices_inj, p)
    circuit.append('MX', data_indices_inj, (), tag='"inj data meas"')
    for d in data_indices_inj:
        global_meas_counter += 1
        meas_rec.setdefault(d, []).append(global_meas_counter)
    
    if obs_basis == 'S':
        circuit.append('S_DAG', data_indices_qldpc, (), tag='"qldpc data S† before meas"')
    if obs_basis == 'X' or obs_basis == 'S':
        circuit.append('Z_ERROR', data_indices_qldpc, p)
        circuit.append('MX', data_indices_qldpc, (), tag='"qldpc data meas"')
    elif obs_basis == 'Z':
        circuit.append('X_ERROR', data_indices_qldpc, p)
        circuit.append('MZ', data_indices_qldpc, (), tag='"qldpc data meas"')
    for d in data_indices_qldpc:
        global_meas_counter += 1
        meas_rec.setdefault(d, []).append(global_meas_counter)
    circuit.append('Z_ERROR', data_indices_ancilla, p)
    circuit.append('MX', data_indices_ancilla, (), tag='"anc data meas"')
    for d in data_indices_ancilla:
        global_meas_counter += 1
        meas_rec.setdefault(d, []).append(global_meas_counter)
    
    # Detectors to check that injection code stabilizers and data measurements
    # agree. We skip any stabilizers that have support on data qubits that are
    # measured in the wrong basis relative to the stabilizer.
    for check_idx in range(inj_Hx.shape[0]):
        idx_in_deformed = X_anc_indices_deformed.index(X_anc_indices_inj[check_idx])
        checked_data = [q for data_idx,q in enumerate(data_indices_deformed) if deformed_Hx[idx_in_deformed,data_idx]]
        # Since ancilla sys and inj are both measured in X, all checks are fine.
        # if any(data not in data_indices_inj for data in checked_data):
        #     print(f'Skipping inj check {check_idx} (qubit {X_anc_indices_inj[check_idx]})')
        #     continue
        circuit.append('DETECTOR', [
                stim.target_rec(meas_rec[X_anc_indices_inj[check_idx]][-1] - global_meas_counter - 1),
                *[stim.target_rec(meas_rec[q][-1] - global_meas_counter - 1) for q in checked_data]
            ],
            (X_anc_indices_inj[check_idx], len(meas_rec[X_anc_indices_inj[check_idx]])),
            tag='"inj final check"'
        )
    # Detectors to check that qLDPC stabilizers and data measurements agree
    if obs_basis == 'X':
        for check_idx in range(qLDPC_Hx.shape[0]):
            idx_in_deformed = X_anc_indices_deformed.index(X_anc_indices_qldpc[check_idx])
            checked_data = [q for data_idx,q in enumerate(data_indices_deformed) if deformed_Hx[idx_in_deformed,data_idx]]
            # if any(data in data_indices_ancilla for data in checked_data):
            #     print(f'Skipping qLDPC check {check_idx} (qubit {X_anc_indices_qldpc[check_idx]})')
            #     continue
            circuit.append('DETECTOR', [
                    stim.target_rec(meas_rec[X_anc_indices_qldpc[check_idx]][-1] - global_meas_counter - 1),
                    *[stim.target_rec(meas_rec[q][-1] - global_meas_counter - 1) for q in checked_data]
                ],
                (X_anc_indices_qldpc[check_idx], len(meas_rec[X_anc_indices_qldpc[check_idx]])),
                tag='"qLDPC final check"'
            )
    elif obs_basis == 'Z':
        for check_idx in range(qLDPC_Hz.shape[0]):
            idx_in_deformed = Z_anc_indices_deformed.index(Z_anc_indices_qldpc[check_idx])
            checked_data = [q for data_idx,q in enumerate(data_indices_deformed) if deformed_Hz[idx_in_deformed,data_idx]]
            # if any(data in data_indices_ancilla for data in checked_data):
            #     print(f'Skipping qLDPC check {check_idx} (qubit {Z_anc_indices_qldpc[check_idx]})')
            #     continue
            circuit.append('DETECTOR', [
                    stim.target_rec(meas_rec[Z_anc_indices_qldpc[check_idx]][-1] - global_meas_counter - 1),
                    *[stim.target_rec(meas_rec[q][-1] - global_meas_counter - 1) for q in checked_data]
                ],
                (Z_anc_indices_qldpc[check_idx], len(meas_rec[Z_anc_indices_qldpc[check_idx]])),
                tag='"qLDPC final check"'
            )
    # Detectors to check that ancilla system stabilizers and data measurements agree
    for X_anc_qubit in X_anc_indices_ancilla:
        idx_in_deformed = X_anc_indices_deformed.index(X_anc_qubit)
        checked_data = [q for data_idx,q in enumerate(data_indices_deformed) if deformed_Hx[idx_in_deformed,data_idx]]
        # if any(data in data_indices_qldpc for data in checked_data):
        #     print(f'Skipping ancilla system check (qubit {X_anc_qubit})')
        #     continue
        circuit.append('DETECTOR', [
                stim.target_rec(meas_rec[X_anc_qubit][-1] - global_meas_counter - 1),
                *[stim.target_rec(meas_rec[q][-1] - global_meas_counter - 1) for q in checked_data]
            ],
            (X_anc_qubit, len(meas_rec[X_anc_qubit])),
            tag='"anc final check"'
        )

    # Third correction term is X measurement of each injection code
    mu_3_targets = [
        (data_indices_inj[q], len(meas_rec[data_indices_inj[q]]) - 1)
        for q,indicator in enumerate(inj_Lx) if indicator
    ]

    # Finally, we define the logical observables that we will check for errors.
    # Final correction term on each logical qubit is V = Z^{mu_1 + mu_3} *
    # X^{mu_2}.
    # Since we are currently measuring the final injected qubits in the X basis,
    # we only need to keep track of the Z part of the correction, and it appears
    # as a frame change.
    if obs_basis == 'X':
        observable_qubits = [data_indices_qldpc[i] for i,indicator in enumerate(qLDPC_Lx[qLDPC_idx_to_inject]) if indicator]
        observable_qubits += [data_indices_deformed[q] for q in (qbs_legend['dual_logical'][1])]
        full_obs = [stim.target_rec(meas_rec[q][-1] - global_meas_counter - 1) for q in observable_qubits] \
                 + [stim.target_rec(meas_rec[q][target_idx] - global_meas_counter - 1) for (q,target_idx) in mu_1_targets + mu_3_targets]
    elif obs_basis == 'Z' or obs_basis == 'S':
        observable_qubits = [data_indices_qldpc[i] for i,indicator in enumerate(qLDPC_Lz[qLDPC_idx_to_inject]) if indicator]
        full_obs = [stim.target_rec(meas_rec[q][-1] - global_meas_counter - 1) for q in observable_qubits] \
                 + [stim.target_rec(meas_rec[q][target_idx] - global_meas_counter - 1) for (q,target_idx) in mu_2_targets]
    else:
        raise ValueError(f'obs_basis {obs_basis} not recognized.')
    print('Observable qubits:', observable_qubits)
    print('mu_1:', mu_1_targets)
    print('mu_3:', mu_3_targets)
    
    circuit.append(
        'OBSERVABLE_INCLUDE',
        full_obs,
        1
    )

    # # TEMP
    # circuit.append('OBSERVABLE_INCLUDE', [stim.target_rec(meas_rec[data_indices_inj[i]][-1] - global_meas_counter - 1) for i,indicator in enumerate(inj_Lz) if indicator], 0)
    # circuit.append('OBSERVABLE_INCLUDE', [stim.target_rec(meas_rec[data_indices_qldpc[i]][-1] - global_meas_counter - 1) for i,indicator in enumerate(qLDPC_Lz[qLDPC_idx_to_inject]) if indicator], 1)

    postselection_mask = []
    for det_idx, det_coords in circuit.get_detector_coordinates().items():
        if postselect_inj_checks and int(det_coords[0]) in Z_anc_indices_inj + X_anc_indices_inj:
            postselection_mask.append(det_idx)
        if postselect_qLDPC_checks and int(det_coords[0]) in Z_anc_indices_qldpc + X_anc_indices_qldpc:
            postselection_mask.append(det_idx)
        if postselect_anc_checks and int(det_coords[0]) in Z_anc_indices_ancilla + X_anc_indices_ancilla:
            postselection_mask.append(det_idx)
    print('Postselecting on', len(postselection_mask), 'detectors.')
    print(postselection_mask)
    postselection_mask_onehot = np.zeros(circuit.num_detectors, dtype=bool)
    postselection_mask_onehot[postselection_mask] = True
    postselection_mask_bitpacked = np.packbits(postselection_mask_onehot, bitorder='little')

    qubit_index_info = {
        'data_indices_deformed':data_indices_deformed,
        'X_anc_indices_deformed':X_anc_indices_deformed,
        'Z_anc_indices_deformed':Z_anc_indices_deformed,

        'data_indices_qldpc':data_indices_qldpc,
        'X_anc_indices_qldpc':X_anc_indices_qldpc,
        'Z_anc_indices_qldpc':Z_anc_indices_qldpc,
        'Z_logical_qldpc':[data_indices_qldpc[i] for i,indicator in enumerate(qLDPC_Lz[qLDPC_idx_to_inject]) if indicator],
        'X_logical_qldpc':[data_indices_qldpc[i] for i,indicator in enumerate(qLDPC_Lx[qLDPC_idx_to_inject]) if indicator],

        'data_indices_inj':data_indices_inj,
        'X_anc_indices_inj':X_anc_indices_inj,
        'Z_anc_indices_inj':Z_anc_indices_inj,
        'Z_logical_inj':[data_indices_inj[i] for i,indicator in enumerate(inj_Lz) if indicator],
        'X_logical_inj':[data_indices_inj[i] for i,indicator in enumerate(inj_Lx) if indicator],

        'checks': {
            anc: [data for data in data_indices_deformed if deformed_Hx[X_anc_indices_deformed.index(anc),data_indices_deformed.index(data)]]
            for anc in X_anc_indices_deformed
        } | {
            anc: [data for data in data_indices_deformed if deformed_Hz[Z_anc_indices_deformed.index(anc),data_indices_deformed.index(data)]]
            for anc in Z_anc_indices_deformed
        }
    }
    return circuit, qubit_index_info, postselection_mask_bitpacked