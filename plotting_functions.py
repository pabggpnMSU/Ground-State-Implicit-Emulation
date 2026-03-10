import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas as pd


data_train = pd.read_csv('gpe_train_data.csv')
mean_q = np.average(data_train['q'])
mean_kappa = np.average(data_train['kappa'])
mean_sigma = np.average(data_train['sigma'])
mean_e = np.average(data_train['E'])

def mult(arr, mat):
    """
    Multiplies array with matrices
    """
    new_mats = []
    for i in arr:
        new_mats.append(i*mat)
    return np.array(new_mats)

def H_mat_poly(n_bases, x, params):
    """
    Build matrix equation 
    
    n_bases: number of dimensions of matrices
    x: set of matrix elements

    returns: full matrix
    """
    n_params = len(params)
    H = np.zeros((n_bases,n_bases)) # Total Hamiltonian
    H_next = np.zeros((n_params, n_bases, n_bases))# symmetric matrix components of the Hamiltonian
    tmp = 0
        
    # Creating the symmetric matrix from the given matrix elements
    for k in range(n_params):
        for i in range(n_bases):
            for j in range(i, n_bases):
                H_next[k][i][j] = x[tmp]
                H_next[k][j][i] = H_next[k][i][j]
                tmp += 1
    
    # Adding the symmetric matrices to the final Hamiltonian
    counter = 0
    for i in range(n_params):
        H = H + mult(params[i,:], H_next[counter])
        counter += 1
    return H

def H_mat_exp(n_bases, x, opt_params):
    n_bases = int(np.round(n_bases))
    num_tri = n_bases * (n_bases + 1) // 2
    
    # 1. Construct matrices from the 1D parameter array 'x'
    # First n_bases elements make up the diagonal matrix
    H_diag = np.diag(x[0:n_bases])
    
    # Next block makes up symm1
    symm1 = np.zeros((n_bases, n_bases))
    iu = np.triu_indices(n_bases)
    symm1[iu] = x[n_bases : n_bases + num_tri]
    symm1 = symm1 + symm1.T - np.diag(symm1.diagonal())
    
    # Final block makes up symm2
    symm2 = np.zeros((n_bases, n_bases))
    symm2[iu] = x[n_bases + num_tri : n_bases + 2 * num_tri]
    symm2 = symm2 + symm2.T - np.diag(symm2.diagonal())

    # 2. Extract batch variables (kappa, q, sigma)
    # Assumes opt_params is shape (3, N_samples)
    kappa = opt_params[0, :].reshape(-1, 1, 1)
    q     = opt_params[1, :].reshape(-1, 1, 1)
    sigma = opt_params[2, :].reshape(-1, 1)
    
    # 3. Calculate the matrix power: symm2^sigma
    evals, evecs = np.linalg.eigh(symm2)
    
    # SAFEGUARD: Extract sign, exponentiate absolute value, and re-apply sign.
    # This completely prevents the LinAlgError (NaNs) from fractional exponents.
    D_p = np.sign(evals) * (np.abs(evals) ** sigma)
    
    # Reconstruct the matrix batch (N_samples, n_bases, n_bases)
    symm2_pow = np.einsum('ij,kj,lj->kil', evecs, D_p, evecs)
    
    # 4. Assemble the final Hamiltonian
    # NumPy broadcasting allows us to add the 2D H_diag to the 3D batched matrices seamlessly
    H = H_diag + (kappa * symm1) + (q * symm2_pow)
    
    return H

def find_e(n_bases, x, params, H_type):
    """
    Find lowest eigenvalue
    
    Lecs: Operator LECs
    n_bases: number of dimensions of the parametric matrix model
    x: set of PMM matrix elements
    H_type: polynomial or exponential
    
    returns: all eigenvalues, the lowest is the energy we fit for
    """
    if H_type == 'exponential':
        H = H_mat_exp(n_bases, x, params)
    elif H_type == 'polynomial':
        H = H_mat_poly(n_bases, x, params)
    ev, ef = np.linalg.eigh(H)
    return ev


# Fit cost function
def cost_function(x, params, BE_training, n_bases, n_ev): # rms cost function
    BE_predict = find_e(n_bases, x, params)

    cost = sum((abs(np.array(BE_predict[:,n_ev])-np.array(BE_training))) ** 2)

    return cost

def cost_function(x, params, BE_training, n_bases, n_ev):
    # BE_predict will be shape (80, n_bases)
    BE_predict = find_e(n_bases, x, params)

    # Extract the specific eigenvalue column (n_ev) for all 80 samples
    predictions = BE_predict[:, n_ev]
    
    # Calculate Mean Squared Error
    errors = predictions - BE_training
    cost = np.sum(errors**2)

    return cost

# Fit PMM
def get_pmm(coeffs, solutions, n_bases, n_train, n_ev):
    """
    n_tests: array of 1 through number of test values, last number should be less than length of train_order
    n_lecs: the number of LECs that have matrices associated
    """
    n_params = len(coeffs)
    n_pred = int(n_params*n_bases*(n_bases+1)/2) # number of parameters to find in PMM
    x_temp = np.array([random.randint(-50,50) for i in range(n_pred)]) # initial guess for PMM parameters
    bounds = [(-500, 500)] * len(x_temp)
    res_da = scipy.optimize.dual_annealing(cost_function, bounds, args=(coeffs[:,:n_train], solutions[:n_train], n_bases, n_ev), x0=x_temp)
    return res_da['x'], res_da['fun']

def mean_center(data, mean):
    return data - mean

def get_square_error(df_results, dim, ev, params, e_mean, H_type):
    x_pmm_output = ast.literal_eval(df_results[df_results['matrix_dimensions'] == dim][df_results['eigenvalue_fit'] == ev]['output_parameters'].iloc[0])
    e_pred_interp = find_e(dim, x_pmm_output, params, H_type)[:, ev]
    interp_error = (e_pred_interp - e_mean)**2
    return interp_error

def get_variable_evs(df_results, q, sigma, kappa, dim, ev, H_type):
    params = np.array([mean_center(kappa, mean_kappa), mean_center(q, mean_q), mean_center(sigma, mean_sigma)])
    x_pmm_output = ast.literal_eval(df_results[df_results['matrix_dimensions'] == dim][df_results['eigenvalue_fit'] == ev]['output_parameters'].iloc[0])
    evs = find_e(5, x_pmm_output, params, H_type)
    return evs

def get_contours(q_diff, q, q_fixed, kappa_diff, kappa, kappa_fixed, sigma_diff, sigma, sigma_fixed, df, n_points, n_dim, n_ev, H_type):
    variable_q_static_sigma = []
    for i in range(len(q_diff)):
        variable_q_static_sigma.append(get_variable_evs(df, q_diff[i]*np.ones(n_points), sigma_fixed, kappa, n_dim, n_ev, H_type)) # sigma=1.75

    variable_kappa_static_sigma = []
    for i in range(len(kappa_diff)):
        variable_kappa_static_sigma.append(get_variable_evs(df, q, sigma_fixed, kappa_diff[i]*np.ones(n_points), n_dim, n_ev, H_type)) # sigma=1.75

    variable_sigma_static_q = []
    for i in range(len(kappa_diff)):
        variable_sigma_static_q.append(get_variable_evs(df, q_fixed, sigma, kappa_diff[i]*np.ones(n_points), n_dim, n_ev, H_type)) # q=1

    variable_kappa_static_q = []
    for i in range(len(sigma_diff)):
        variable_kappa_static_q.append(get_variable_evs(df, q_fixed, sigma_diff[i]*np.ones(n_points), kappa, n_dim, n_ev, H_type)) # q=1

    variable_q_static_kappa = []
    for i in range(len(kappa_diff)):
        variable_q_static_kappa.append(get_variable_evs(df, q, sigma_diff[i]*np.ones(n_points), kappa_fixed, n_dim, n_ev, H_type)) # kappa=1.75

    variable_sigma_static_kappa = []
    for i in range(len(q_diff)):
        variable_sigma_static_kappa.append(get_variable_evs(df, q_diff[i]*np.ones(n_points), sigma, kappa_fixed, n_dim, n_ev, H_type)) # kappa=1.75

    return variable_q_static_sigma, variable_kappa_static_sigma, variable_sigma_static_q, variable_kappa_static_q, variable_q_static_kappa, variable_sigma_static_kappa