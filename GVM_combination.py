import numpy as np
import os
from scipy.optimize import minimize
from scipy.stats import gamma, chi2

import matplotlib.pyplot as plt

from iminuit import Minuit
from iminuit.util import make_func_code
import json

class combination:
    
    def __init__(self, input_file_folder, uncertain_systematics=None, comb = 'lhc', regularize = "n"):
        # Load data
        # Initialization code as provided
        self.comb = comb
        self.input_file_folder = input_file_folder
        self.m_t, self.stat, self.syst, self.correlation_matrices = self.load_data(self.comb, regularize)
        self.uncertain_systematics = {} if uncertain_systematics is None else uncertain_systematics
        self.V_Blue_inv, self.C_matrix_inv, self.Gamma_factors = self.compute_likelihood_matrices()
        self.fit_results = self.minimize_chi2()
        
    
    #load central values, stat and syst errors, and correlation matrices
    def load_data(self, comb, regularize):
        m_t = []
        stat = []
        syst = {}
        correlation_matrices = {}
        
        atlas_file_path = os.path.join( self.input_file_folder + "/data_atlas.txt" )
        cms_file_path = os.path.join( self.input_file_folder + "/data_cms.txt" )
        
        with open(atlas_file_path, 'r') as file:
            string_data = [line.strip().split()[0] for line in file]

        uncertainty_names = string_data[2:-1]
        
        data_atlas = np.loadtxt(atlas_file_path, dtype=float, usecols=range(1, 7))
        data_cms = np.loadtxt(cms_file_path, dtype=float, usecols=range(1, 10))
        
        # Load data for atlas combination
        if comb == 'atlas':
            
            m_t = np.array(data_atlas[0, :])
            stat = np.array(data_atlas[1, :])

            for idx, name in enumerate(uncertainty_names):
                syst[name] = data_atlas[idx + 2, :]

            for source in uncertainty_names:
                file_path = os.path.join( self.input_file_folder + source + '.txt' )
                try:
                    corr_matrix = np.loadtxt(file_path)[0:6, 0:6]
                    if regularize == 'y':
                        eigenvalues = np.linalg.eigvalsh(corr_matrix)
                        min_eigenvalue = np.min(eigenvalues)
                        if min_eigenvalue <= 0:
                            shift = abs(min_eigenvalue) + 1e-8  # Adding a small epsilon to ensure positiveness
                            np.fill_diagonal(corr_matrix, corr_matrix.diagonal() + shift)
                    correlation_matrices[source] = corr_matrix
                    
                except FileNotFoundError:
                    print(f"File not found for {source}, skipping.")

        # Load data for cms combination
        elif comb == 'cms':

            m_t = np.array(data_cms[0, :])
            stat = np.array(data_cms[1, :])

            for idx, name in enumerate(uncertainty_names):
                syst[name] = data_cms[idx + 2, :]

            for source in uncertainty_names:
                file_path = os.path.join( self.input_file_folder + source + '.txt' )
                try:
                    corr_matrix = np.loadtxt(file_path)[6:, 6:]
                    if regularize == 'y':
                        eigenvalues = np.linalg.eigvalsh(corr_matrix)
                        min_eigenvalue = np.min(eigenvalues)
                        if min_eigenvalue <= 0:
                            shift = abs(min_eigenvalue) + 1e-8  # Adding a small epsilon to ensure positiveness
                            np.fill_diagonal(corr_matrix, corr_matrix.diagonal() + shift)
                    correlation_matrices[source] = corr_matrix
                except FileNotFoundError:
                    print(f"File not found for {source}, skipping.")
                    
        elif comb == 'lhc':

            data = np.hstack((data_atlas, data_cms))
            m_t = data[0, :]
            stat = data[1, :]

            for idx, name in enumerate(uncertainty_names):
                syst[name] = data[idx + 2, :]  

            for source in uncertainty_names:
                file_path = os.path.join( self.input_file_folder + source + '.txt' )
                try:
                    corr_matrix = np.loadtxt(file_path)
                    if regularize == "y":
                        eigenvalues = np.linalg.eigvalsh(corr_matrix)
                        min_eigenvalue = np.min(eigenvalues)
                        print(min_eigenvalue)
                        if min_eigenvalue <= 0:
                            shift = abs(min_eigenvalue) + 1e-8  # Adding a small epsilon to ensure positiveness
                            np.fill_diagonal(corr_matrix, corr_matrix.diagonal() + shift)
                    correlation_matrices[source] = corr_matrix
                except FileNotFoundError:
                    print(f"File not found for {source}, skipping.")
                    
        return m_t, stat, syst, correlation_matrices
    
    #Contruct Gamma matrix and auxiliary measurements correlation matrix 
    def reduce_and_track_uncertain_systematics(self, rho_matrix):
        n = rho_matrix.shape[0]
        groups = []
        
        visited = set()
        for i in range(n):
            if i not in visited:
                group = [i]
                for j in range(i + 1, n):
                    if abs(rho_matrix[i, j]) == 1:
                        group.append(j)
                        visited.add(j)
                groups.append(group)
                visited.add(i)

        # Build the reduced matrix
        reduced_size = len(groups)
        reduced_matrix = np.zeros((reduced_size, reduced_size))
        Gamma_matrix = np.zeros((n, reduced_size))
        
        for new_i, group in enumerate(groups):
            for j in group:
                sign = 1.
                if rho_matrix[group[0], j] == -1:
                    sign = -1  
                Gamma_matrix[j, new_i] = sign
                
        for new_i, group_i in enumerate(groups):      
            for new_j, group_j in enumerate(groups):
                vector = [rho_matrix[i, j] * Gamma_matrix[i, new_i] * Gamma_matrix[j, new_j] for i in group_i for j in group_j]
                mean_value = np.mean(vector)

                # Assign the mean value to the reduced matrix
                reduced_matrix[new_i, new_j] = mean_value

        # Check if the reduced matrix is positive definite
        eigenvalues = np.linalg.eigvalsh(reduced_matrix)
        min_eigenvalue = np.min(eigenvalues)

        if min_eigenvalue <= 0:
            # Add to the diagonal a constant term sufficient to make the matrix positive definite
            shift = abs(min_eigenvalue) + 0.01 # Adding a small epsilon to ensure positiveness
            np.fill_diagonal(reduced_matrix, reduced_matrix.diagonal() + shift)
        
        return reduced_matrix, Gamma_matrix
    
    #Construct the Blue matrix for non-uncertain systematics, Correlation matrices for auxiliary measurements, and Gamma matrices
    def compute_likelihood_matrices(self):
        n = np.size(self.stat)
        V_stat = np.diag(self.stat**2)

        V_syst = np.zeros((n, n))

        # Build V_syst excluding systematics with epsilon > 0 (to be treated separately)
        for source, rho_matrix in self.correlation_matrices.items():
            if source not in self.uncertain_systematics:
                sigma_syst = self.syst[source]
                for i in range(n):
                    for j in range(n):
                        V_syst[i, j] += sigma_syst[i] * sigma_syst[j] * rho_matrix[i, j]

        # Calculate the inverse of the total covariance matrix (statistical + systematic)
        V_blue = V_stat + V_syst
        V_inv = np.linalg.inv(V_blue)

        # Now process and potentially reduce the separate systematics
        C_matrix_inv = {}
        Gamma_factors = {}
        for source, sigma_syst in self.syst.items():
            if source in self.uncertain_systematics:
                rho_matrix = self.correlation_matrices[source]

                # Reduce the correlation matrix directly
                reduced_rho_matrix, Gamma_matrix = self.reduce_and_track_uncertain_systematics(rho_matrix)

                # Adjust the Gamma_matrix by multiplying non-zero entries by the corresponding sigma_syst values
                for i in range(Gamma_matrix.shape[0]):
                    for j in range(Gamma_matrix.shape[1]):
                        if Gamma_matrix[i, j] != 0:
                            Gamma_matrix[i, j] *= sigma_syst[i]
                
                # Find columns that are all zeros in Gamma_matrix
                zero_columns = np.all(Gamma_matrix == 0, axis=0)

                # Filter out zero columns from Gamma_matrix
                filtered_Gamma_matrix = Gamma_matrix[:, ~zero_columns]

                # Remove the corresponding columns and rows from reduced_rho_matrix
                if np.any(zero_columns):
                    non_zero_columns = ~zero_columns
                    filtered_rho_matrix = reduced_rho_matrix[non_zero_columns][:, non_zero_columns]
                    C_matrix_inv[source] = np.linalg.inv(filtered_rho_matrix)
                else:
                    C_matrix_inv[source] = np.linalg.inv(reduced_rho_matrix)

                # Update Gamma_factors with the filtered Gamma_matrix
                Gamma_factors[source] = filtered_Gamma_matrix
        
        return V_inv, C_matrix_inv, Gamma_factors
    
    #Compute combination chi2
    def chi2(self, mu, *thetas_flat, y=None, u=None, s=None):
            
        # Use self.m_t as default if y is not provided
        if y is None:
            y = self.m_t

        # Segment thetas_flat into a list of arrays, each corresponding to a separate systematic
        thetas = []
        start = 0
        total_length = 0  # Keep track of the total length to initialize u if not provided
        for key in self.C_matrix_inv.keys():
            len_theta_i = self.C_matrix_inv[key].shape[0]  # Assuming square C_inv matrices
            thetas.append(np.array(thetas_flat[start:start + len_theta_i]))
            start += len_theta_i
            total_length += len_theta_i

        # Initialize u to zeros if not provided
        if u is None:
            u = np.zeros(total_length)
            
        if s is None:
            s = np.ones(len(self.C_matrix_inv.keys()))

        # Ensure u is segmented corresponding to the thetas
        u_segmented = []
        start = 0
        for theta in thetas:
            len_u_i = len(theta)  # Length of u_i matches theta_i
            u_segmented.append(u[start:start + len_u_i])
            start += len_u_i

        # Systematic adjustments calculation
        systematic_adjustments = np.sum(
            [self.Gamma_factors[key] @ thetas[i] for i, key in enumerate(self.Gamma_factors.keys())], axis=0
        )

        # Compute the data fit term
        vect = y - mu - systematic_adjustments
        chi2_y = vect @ self.V_Blue_inv @ vect.T

        # Compute the penalty term for each theta_i, including the adjustment for u_i
        chi2_u = 0
        for i, key in enumerate(self.C_matrix_inv.keys()):
            theta_i = thetas[i] - u_segmented[i]  # Adjust theta_i by u_i
            C_inv_i = self.C_matrix_inv[key]
            eps = self.uncertain_systematics[key]

            cols_with_all_zeros = np.sum(np.all(self.Gamma_factors[key] == 0, axis=0))
                
            N_s = len(theta_i)
            if eps > 0:
                chi2_u += (N_s + 1./(2. * eps**2)) * np.log(1. + 2. * eps**2 / s[i] * theta_i @ C_inv_i @ theta_i)
            else:
                chi2_u += 1. / s[i] * theta_i @ C_inv_i @ theta_i
            #print(theta_i @ C_inv_i @ theta_i)

        # Final chi2 value
        chi2_value = chi2_y + chi2_u
        return chi2_value
  
    #Compute Fisher Information Matrix
    def compute_FIM(self, S=None):
        keys = list(self.C_matrix_inv.keys())
        theta_sizes = [self.C_matrix_inv[key].shape[0] for key in keys]
        total_theta_size = sum(theta_sizes)
        FIM = np.zeros((1 + total_theta_size, 1 + total_theta_size))

        # mu mu entry
        FIM[0, 0] = np.sum(self.V_Blue_inv)

        # Calculate indices for theta parameters
        start_indices = np.cumsum([0] + theta_sizes[:-1])
        indices = [np.arange(size) + start + 1 for size, start in zip(theta_sizes, start_indices)]

        # Precompute V_Blue_inv @ Gamma_s for each source
        V_Blue_inv_Gamma = {key: self.V_Blue_inv @ self.Gamma_factors[key] for key in keys}

        # mu theta entries
        for s, key in enumerate(keys):
            idx_range = indices[s]
            Gamma_sum = V_Blue_inv_Gamma[key].sum(axis=0)
            FIM[0, idx_range] = FIM[idx_range, 0] = Gamma_sum

        # thetas thetas entries
        for s, key_s in enumerate(keys):
            idx_s = indices[s]
            Gamma_s = self.Gamma_factors[key_s]
            C_inv_s = self.C_matrix_inv[key_s]
            S_s = 1.0 if S is None else S[s]
            
            # Compute Gamma_s @ V_Blue_inv @ Gamma_s for each source
            for sp, key_sp in enumerate(keys):
                idx_sp = indices[sp]

                Gs_Vinv_Gsp = Gamma_s.T @ V_Blue_inv_Gamma[key_sp]

                # Add C_inv_s term if keys are the same
                if key_s == key_sp:
                    FIM[np.ix_(idx_s, idx_sp)] = Gs_Vinv_Gsp + (1.0 / S_s) * C_inv_s
                else:
                    FIM[np.ix_(idx_s, idx_sp)] = Gs_Vinv_Gsp

        return FIM

    
    def compute_b_lawley(self):
        thetas_flat = self.fit_results['thetas']

        keys = list(self.C_matrix_inv.keys())
        theta_sizes = [self.C_matrix_inv[key].shape[0] for key in keys]
        indices = np.cumsum([0] + theta_sizes)
        thetas = [thetas_flat[indices[i]:indices[i+1]] for i in range(len(keys))]

        # Compute S for each source
        epsilon_s_list = [self.uncertain_systematics[key] for key in keys]
        C_inv_list = [self.C_matrix_inv[key] for key in keys]
        
        N_s_list = theta_sizes
        S = np.array([
            (1 + 2 * eps**2 * theta @ C_inv @ theta) / (1 + 2 * N_s * eps**2)
            for theta, C_inv, eps, N_s in zip(thetas, C_inv_list, epsilon_s_list, N_s_list)
        ])

        # Calculate the Fisher Information Matrix
        #S = np.ones(len(S))
        FIM = self.compute_FIM(S)
        W_full = np.linalg.inv(FIM)[1:, 1:]
        W_theta = np.linalg.inv(FIM[1:, 1:])

        # Compute b terms
        start_indices = indices[:-1]
        b_lik = b_theta = b_chi2 = 0

        for s, (theta_s, C_inv_s, eps_s, N_s, S_s) in enumerate(zip(thetas, C_inv_list, epsilon_s_list, N_s_list, S)):
            start_idx = start_indices[s]
            end_idx = start_idx + N_s
            W_s = W_full[start_idx:end_idx, start_idx:end_idx]
            W_theta_s = W_theta[start_idx:end_idx, start_idx:end_idx]

            # Common computations
            trace_WC = np.trace(W_s @ C_inv_s)
            trace_WC_squared = trace_WC ** 2
            trace_WCWC = np.trace(W_s @ C_inv_s @ W_s @ C_inv_s)

            trace_W_theta_C = np.trace(W_theta_s @ C_inv_s)
            trace_W_theta_C_squared = trace_W_theta_C ** 2
            trace_W_theta_CWC = np.trace(W_theta_s @ C_inv_s @ W_theta_s @ C_inv_s)

            # Coefficients
            coeff1 = 4 * eps_s**2 / S_s
            coeff2 = -2 * eps_s**2 / S_s**2
            coeff3 = eps_s**2 / S_s**2

            # Update b terms
            b_lik += coeff1 * trace_WC + coeff2 * trace_WCWC + coeff3 * trace_WC_squared
            b_theta += coeff1 * trace_W_theta_C + coeff2 * trace_W_theta_CWC + coeff3 * trace_W_theta_C_squared
            b_chi2 += (2 * N_s + N_s**2) * eps_s**2

        b_p_lik = b_lik - b_theta
        b_chi2 -= b_lik

        return sum(theta_sizes) + 1 + b_lik, 1 + b_p_lik, len(self.m_t) - 1 + b_chi2

    
    def compute_bias_lawley(self, key_input):
        thetas_flat = self.fit_results['thetas']

        keys = list(self.C_matrix_inv.keys())
        theta_sizes = [self.C_matrix_inv[key].shape[0] for key in keys]
        indices = np.cumsum([0] + theta_sizes)
        thetas = [thetas_flat[indices[i]:indices[i+1]] for i in range(len(keys))]

        # Compute S for each source
        epsilon_list = [self.uncertain_systematics[key] for key in keys]
        C_inv_list = [self.C_matrix_inv[key] for key in keys]
        S = np.array([
            (1 + 2 * eps**2 * theta @ C_inv @ theta) / (1 + 2 * len(theta) * eps**2)
            for theta, C_inv, eps in zip(thetas, C_inv_list, epsilon_list)
        ])

        # Compute the Fisher Information Matrix with S
        FIM = self.compute_FIM(S)
        W_full = np.linalg.inv(FIM)
        W = W_full[1:, 1:]

        # Find the index and corresponding slice for the specific key_input
        s = keys.index(key_input)
        start_idx = indices[s]
        end_idx = start_idx + theta_sizes[s]
        W_s = W[start_idx:end_idx, start_idx:end_idx]

        # Retrieve C_inv and epsilon for the specific key_input
        C_inv_s = self.C_matrix_inv[key_input]
        epsilon_s = self.uncertain_systematics[key_input]

        # Compute the bias
        trace = np.trace(W_s @ C_inv_s)
        bias = 2 * epsilon_s**2 * trace

        return bias


    #Minimize chi2 with respect to all parameters, of interest and nuisance
    def minimize_chi2(self, method = 'L-BFGS-B', y=None, u=None, s=None):
        
        def objective(params):
            mu = params[0]
            thetas = params[1:]
            return self.chi2(mu, *thetas, y=y, u=u, s=s)

        # Initial guess for mu and thetas
        param_names = ['mu'] + [f'theta_{key}_{j}' for i, key in enumerate(self.C_matrix_inv.keys()) for j in range(self.C_matrix_inv[key].shape[0])]

        initial_guess = [np.mean(self.m_t)]  # Start with mu's guess

        # Append initial guesses for each theta_i and their bounds
        for key in self.C_matrix_inv.keys():
            len_theta_i = self.C_matrix_inv[key].shape[0]  # Assuming square C_inv matrices
            initial_guess += [0.0] * len_theta_i
            
        if method == 'minuit':# Define parameter names

            # Initialize Minuit with the objective function and parameters
            m = Minuit(objective, initial_guess, name=param_names)

            # Perform the minimization
            m.migrad()

            # Extract best-fit values
            best_fit_mu = m.values['mu']
            best_fit_thetas = [m.values[f'theta_{key}_{j}'] for i, key in enumerate(self.C_matrix_inv.keys()) for j in range(self.C_matrix_inv[key].shape[0])]

            # Optionally, calculate and display the parameter errors
            m.hesse()

            # Package the result
            fit_result = {'mu': best_fit_mu, 'thetas': best_fit_thetas}
            
        # Perform the minimization
        else:
            result = minimize(objective, initial_guess, method = method)

            # Extract best-fit values
            best_fit_mu = result.x[0]
            best_fit_thetas = result.x[1:]

        return {'mu': best_fit_mu, 'thetas': best_fit_thetas}

    #Minimize chi2 with respect to nuisance parameters, keeping the parameter of interest fixed
    def minimize_chi2_nuisance_only(self, mu, method = 'L-BFGS-B', y=None, u=None, s=None):
        
        def objective(thetas):
            return self.chi2(mu, *thetas, y=y, u=u, s=s)

        # Initial guess for thetas
        initial_guess = []        
        for key in self.C_matrix_inv.keys():
            len_theta_i = self.C_matrix_inv[key].shape[0]  # Assuming square C_inv matrices
            initial_guess += [0.0] * len_theta_i
        
        if method == 'minuit':# Define parameter names

            # Initialize Minuit with the objective function and parameters
            m = Minuit(objective, initial_guess, name=param_names)

            # Perform the minimization
            m.migrad()

            # Extract best-fit values
            best_fit_thetas = [m.values[f'theta_{key}_{j}'] for i, key in enumerate(self.C_matrix_inv.keys()) for j in range(self.C_matrix_inv[key].shape[0])]
        
        # Perform the minimization
        else:
            result = minimize(objective, initial_guess, method = method)

            # Extract best-fit values
            best_fit_thetas = result.x

        return {'thetas': best_fit_thetas}
    
    def compute_theta_mle(self, mu, y=None, u=None, s=None, thetas0=None):
        keys = list(self.C_matrix_inv.keys())
        theta_sizes = [self.C_matrix_inv[key].shape[0] for key in keys]
        total_theta_size = sum(theta_sizes)
        
        thetas = np.zeros(total_theta_size) if thetas0 is None else thetas0.copy()
        u = np.zeros(total_theta_size) if u is None else u.copy()
        y = self.m_t if y is None else y.copy()
        S =  np.ones(len(self.C_matrix_inv.keys())) if s is None else s.copy()
        
        thetas = np.split(thetas, np.cumsum(theta_sizes)[:-1])
        u = np.split(u, np.cumsum(theta_sizes)[:-1])
        
        epsilon_list = [self.uncertain_systematics[key] for key in keys]
        C_inv_list = [self.C_matrix_inv[key] for key in keys]
        
        if thetas0 is not None: 
            S = np.array([
            (S_s + 2 * eps_s**2 * (u_s - theta_s) @ C_inv_s @ (u_s - theta_s)) / (1 + 2 * len(theta_s) * eps_s**2)
            for u_s, S_s, theta_s, C_inv_s, eps_s in zip(u, S, thetas, C_inv_list, epsilon_list)
        ])
        
        M = np.zeros((total_theta_size, total_theta_size))

        # Calculate start indices and indices for theta parameters
        start_indices = np.cumsum([0] + theta_sizes[:-1])
        indices = [np.arange(size) + start for size, start in zip(theta_sizes, start_indices)]

        # Precompute V_Blue_inv @ Gamma_s for each source
        V_Blue_inv_Gamma = {key: self.V_Blue_inv @ self.Gamma_factors[key] for key in keys}

        # Interaction among thetas
        for s, key_s in enumerate(keys):
            idx_s = indices[s]
            Gamma_s = self.Gamma_factors[key_s]
            C_inv_s = self.C_matrix_inv[key_s]
            S_s = S[s]

            for sp, key_sp in enumerate(keys):
                idx_sp = indices[sp]
                Gs_Vinv_Gsp = Gamma_s.T @ V_Blue_inv_Gamma[key_sp]

                # Add C_inv_s term if keys are the same
                if key_s == key_sp:
                    M[np.ix_(idx_s, idx_sp)] = Gs_Vinv_Gsp + (1.0 / S_s) * C_inv_s
                else:
                    M[np.ix_(idx_s, idx_sp)] = Gs_Vinv_Gsp
         
        systematic_adjustments = np.sum(
            [self.Gamma_factors[key] @ thetas[i] for i, key in enumerate(self.Gamma_factors.keys())], axis=0
        )
        
        systematic_adjustments = np.sum(
            [self.Gamma_factors[key] @ thetas[i] for i, key in enumerate(self.Gamma_factors.keys())], axis=0
        )
        vect = y - mu - systematic_adjustments
        
        shifts = np.zeros(total_theta_size)
        for s, key_s in enumerate(keys):
            idx_s = indices[s]
            Gamma_s = self.Gamma_factors[key_s]
            C_inv_s = self.C_matrix_inv[key_s]
            theta_s = thetas[s]
            u_s = u[s]
            S_s = 1.0 if S is None else S[s]
            
            shifts[np.ix_(idx_s)] = Gamma_s.T @ self.V_Blue_inv @ vect + 1./S_s * C_inv_s @ (u_s - theta_s)
    
        M_inv = np.linalg.inv(M)
        
        theta_mle = M_inv @ shifts
        return np.array(theta_mle)
    
    def compute_theta_mle_at_order(self, mu, max_order=4):
        """
        Compute theta_mle recursively up to a given order using a correction approach.

        Parameters:
        - mu: Scalar, the value of mu.
        - max_order: Integer, the maximum order of correction to compute (default is 4).

        Returns:
        - theta_mle_orders: A list of numpy arrays, where each entry corresponds to the theta_mle estimate
                            at a particular order (from order 0 to max_order).
        """

        # 1. Compute the 0th order MLE (initial guess)
        theta_mle = self.compute_theta_mle(mu)

        # 2. Compute the successive corrections up to max_order
        for order in range(1, max_order + 1):
            # Compute the next correction
            theta_mle_next = theta_mle + self.compute_theta_mle(mu, thetas0=theta_mle)
            theta_mle = theta_mle_next

        return theta_mle
    
    def compute_mu_mle_at_order(self, max_order=4):
        """
        Numerically minimize the chi2 function with respect to mu, updating thetas at each step.

        Parameters:
        - initial_mu_guess: Float, the initial guess for mu.
        - bounds: Tuple, the bounds within which to search for mu (mu_min, mu_max).
        - options: Dict, additional options to pass to the optimizer.
        - max_order: Integer, the maximum order for computing theta_mle (default is 4).

        Returns:
        - result: The result object from the optimizer containing optimized mu and chi2 value.
        """

        # Define the function to minimize: chi2(mu), where thetas are updated for each mu
        def chi2_mu(mu):
            mu = float(mu)  # Ensure mu is a scalar double value

            # Update thetas given mu up to the specified order
            theta_mle = self.compute_theta_mle_at_order(mu, max_order=max_order)

            # Compute chi2 at this mu and thetas
            chi2_value = self.chi2(mu, *theta_mle)
            return chi2_value

        # Initial guess for mu as a scalar
        mu0 = np.mean(self.m_t)

        # Perform the optimization
        result = minimize(
            chi2_mu,
            mu0,
            method='L-BFGS-B',
        )

        return result.x

    
    #Compute the profile likelihood ratio test statistic, use "minuit" to use minuit as minimizer or any of scipy minimizers
    def likelihood_ratio_test(self, mu, method='L-BFGS-B', y=None, u=None, s=None):
        """
        Compute the likelihood ratio statistic for a given mu.

        Parameters:
        - mu: Float, the value of mu to test.
        - method: String, the optimization method to use. If in the form 'order_N', uses
                  compute_mu_mle_at_order and compute_theta_mle_at_order with max_order=N.
        - y, u, s: Optional parameters for chi2 computation.

        Returns:
        - likelihood_ratio_statistic: Float, the difference in chi2 values.
        """
        # Check if method is of the form 'order_N'
        if isinstance(method, str) and method.startswith('order_'):
            # Extract N from 'order_N'
            try:
                N = int(method.split('_')[1])
            except (IndexError, ValueError):
                raise ValueError("Invalid method format. Expected 'order_N' where N is an integer.")

            # Compute mu MLE up to order N
            mu_mle = self.compute_mu_mle_at_order(max_order=N)

            # Compute theta MLE at mu_mle up to order N
            theta_mle = self.compute_theta_mle_at_order(mu_mle, max_order=N)

            # Compute chi2 at best fit
            chi2_best_fit = self.chi2(mu_mle, *theta_mle, y=y, u=u, s=s)

            # Compute theta MLE at provided mu up to order N
            theta_mle_mu = self.compute_theta_mle_at_order(mu, max_order=N)

            # Compute chi2 at mu and corresponding theta_mle
            chi2_mu = self.chi2(mu, *theta_mle_mu, y=y, u=u, s=s)

        else:
            # Use the standard minimization methods
            # Minimize chi2 to find the best fit values for mu and thetas
            best_fit = self.minimize_chi2(method=method, y=y, u=u, s=s)

            # Extract best-fit mu and thetas
            mu_best = best_fit.get('mu', None)
            thetas_best = best_fit.get('thetas', [])
            # Compute chi2 at best fit
            chi2_best_fit = self.chi2(mu_best, *thetas_best, y=y, u=u, s=s)

            # Check if there are nuisance parameters to minimize over
            if len(thetas_best) > 0:
                # Minimize chi2 for nuisance parameters only, with mu fixed at mu0
                best_fit_nuisance_only = self.minimize_chi2_nuisance_only(mu, method=method, y=y, u=u, s=s)
                thetas_mu = best_fit_nuisance_only.get('thetas', [])
                chi2_mu = self.chi2(mu, *thetas_mu, y=y, u=u, s=s)
            else:
                # If there are no nuisance parameters, use chi2 with mu directly
                chi2_mu = self.chi2(mu, y=y, u=u, s=s)

        # Calculate the likelihood ratio statistic
        likelihood_ratio_statistic = chi2_mu - chi2_best_fit

        return likelihood_ratio_statistic

    def goodness_of_fit(self, method='L-BFGS-B', y=None, u=None, s=None):
        # Minimize chi2 to find the best fit values for mu and thetas
        best_fit = self.minimize_chi2(method=method, y=y, u=u, s=s)
        # Adjusted check for the presence of thetas
        chi2_best_fit = self.chi2(best_fit['mu'], *best_fit['thetas'], y=y, u=u, s=s) if len(best_fit['thetas']) > 0 else self.chi2(best_fit['mu'], y=y, u=u, s=s)


        # Calculate the likelihood ratio statistic
        likelihood_ratio_statistic = chi2_best_fit

        return likelihood_ratio_statistic 
    
    #Scatter plot of the combination measurements
    def plot_comb_measurements(self):
        # Calculate the total error for each measurement
        total_errors = []
        for i in range(len(self.m_t)):
            # Sum of squares of all systematic errors for the i-th measurement
            syst_error_square = sum(value[i]**2 for value in self.syst.values())
            # Square root of sum of squares of statistical and systematic errors for the i-th measurement
            total_error = np.sqrt(self.stat[i]**2 + syst_error_square)
            total_errors.append(total_error)

        # X axis values (measurement indices)
        x_values = np.arange(len(self.m_t))

        # Y axis values (central values)
        y_values = np.array(list(self.m_t))

        # Make the figure larger
        plt.figure(figsize=(14, 8))  # Increased figure size

        # Split plotting by ATLAS (first 6) and CMS (the others)
        # Plot ATLAS measurements in red
        plt.scatter(x_values[:], y_values[:], color='blue')
        plt.errorbar(x_values[:], y_values[:], yerr=total_errors[:], fmt='o', ecolor='red', capsize=5)
        # Labeling the plot
        plt.xlabel('Measurement Index')
        plt.ylabel('Central Value')
        plt.title('Central Values and Confidence Intervals')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def find_confidence_interval(self, tol=0.01, step_size=0.01, max_iter=1000, bartlett_precision=0, lik_ratio_filepath=None, method='L-BFGS-B', b=None):
        # Validate input parameters for Bartlett correction
        if b is None:
            if (bartlett_precision != 0 and lik_ratio_filepath is not None):
                raise ValueError("Specify either bartlett_precision or lik_ratio_filepath, not both.")
            if bartlett_precision == 0 and lik_ratio_filepath is None:
                _, b, _ = self.compute_b_lawley() 
            elif bartlett_precision != 0:
                b, _ = self.compute_bartlett_correction(N_toys=bartlett_precision)
            else:  # lik_ratio_filepath is not None
                b = self.compute_bartlett_correction_from_file(lik_ratio_filepath)
        else:
            # Use the provided Bartlett correction
            pass

        print(f"Bartlett correction b = {b}")
        # First, find the best fit mu
        
        best_fit = self.minimize_chi2()
        mu_hat = best_fit['mu']
        
        if isinstance(method, str) and method.startswith('order_'):
            # Extract N from 'order_N'
            try:
                N = int(method.split('_')[1])
                mu_hat = self.compute_mu_mle_at_order(max_order=N)[0]
            except (IndexError, ValueError):
                raise ValueError("Invalid method format. Expected 'order_N' where N is an integer.")

        # Compute the likelihood ratio statistic for the best fit mu
        lr_stat_best_fit = self.likelihood_ratio_test(mu_hat, method=method)

        # Initialize variables to search for the upper and lower bounds
        m_up = mu_hat
        m_down = mu_hat
        lr_stat_up = lr_stat_best_fit
        lr_stat_down = lr_stat_best_fit

        iter_count = 0
        # Search for the upper bound
        while lr_stat_up <= 1 * b and iter_count < max_iter:
            m_up += step_size
            lr_stat_up = self.likelihood_ratio_test(m_up, method=method)
            iter_count += 1

        # Reset iteration count for the lower bound search
        iter_count = 0
        # Search for the lower bound
        while lr_stat_down <= 1 * b and iter_count < max_iter:
            m_down -= step_size
            lr_stat_down = self.likelihood_ratio_test(m_down, method=method)
            iter_count += 1

        # Fine-tuning the search within the tolerance
        # For m_up
        while abs(lr_stat_up - 1 * b) > tol and iter_count < max_iter:
            step_size /= 2  # Reduce step size for finer search
            if lr_stat_up > 1 * b :
                m_up -= step_size
            else:
                m_up += step_size
            lr_stat_up = self.likelihood_ratio_test(m_up, method=method)
            iter_count += 1

        # Reset iteration count for the lower bound fine-tuning
        iter_count = 0
        # For m_down
        while abs(lr_stat_down - 1 * b) > tol and iter_count < max_iter:
            step_size /= 2  # Reduce step size for finer search
            if lr_stat_down > 1 * b:
                m_down += step_size
            else:
                m_down -= step_size
            lr_stat_down = self.likelihood_ratio_test(m_down, method=method)
            iter_count += 1

        return m_down, m_up, abs(m_down - m_up) * 0.5
    
    def add_measurement(self, m_t_new, stat_new, syst_name, syst_value, uncertain_syst_value):
        # Add the new central value (m_t) for the fictitious measurement
        self.m_t = np.append(self.m_t, m_t_new)

        # Add the new statistical uncertainty for the fictitious measurement
        self.stat = np.append(self.stat, stat_new)

        # Initialize or update the systematic error for "NAME"
        if syst_name not in self.syst:
            self.syst[syst_name] = np.zeros(len(self.m_t) - 1)  # Initialize with zeros for existing measurements
        self.syst[syst_name] = np.append(self.syst[syst_name], syst_value)  # Add syst_value for the new measurement

        # Append a zero for each existing systematic for the new measurement
        for key, value in self.syst.items():
            if key != syst_name:  # Avoid duplicating the operation for the new systematic
                self.syst[key] = np.append(value, 0)  # Append zero for the new measurement

        # Update the correlation matrix for "NAME"
        if syst_name not in self.correlation_matrices:
            self.correlation_matrices[syst_name] = np.eye(len(self.m_t))  # Initialize with identity matrix for the new measurement
        else:
            new_size = len(self.m_t)
            new_matrix = np.eye(new_size)
            new_matrix[:-1, :-1] = self.correlation_matrices[syst_name]  # Copy the existing matrix and expand it
            self.correlation_matrices[syst_name] = new_matrix

        # Update existing correlation matrices by adding a new row and column of zeros, with one on the diagonal
        for key, matrix in self.correlation_matrices.items():
            if key != syst_name:  # Avoid duplicating the operation for the new systematic
                new_size = len(self.m_t)
                new_matrix = np.zeros((new_size, new_size))
                new_matrix[:-1, :-1] = matrix  # Copy the existing matrix
                new_matrix[-1, -1] = 1  # Set the diagonal element for the new measurement to 1
                self.correlation_matrices[key] = new_matrix

        # Recompute the likelihood matrices to reflect the updated measurements and uncertainties
        self.uncertain_systematics[syst_name] = uncertain_syst_value

        self.V_Blue_inv, self.C_matrix_inv, self.Gamma_factors = self.compute_likelihood_matrices()
        self.fit_results = self.minimize_chi2()
        
    def analyze_systematics_combination_CV(self, systematics_input, eps_end=0.6, n_points=10, ylim = None, output_path = None, method = 'L-BFGS-B'):

        self.plot_comb_measurements()
        mu_0 = self.minimize_chi2()['mu']

        #mu_low, mu_up, CI = self.find_confidence_interval()

        # Prepare the epsilon range
        epsilons = np.linspace(0.0, eps_end, n_points)
        
        percentage_variations = []
        plt.figure(figsize=(11, 7))
        for syst_group in systematics_input:
            individual_systematics = syst_group.split(', ')
            epsilon_values = []
            mu_values = []
            percentage_variation = []
            mu_base = None  # Ensure mu_base is defined and reset for each systematic group

            for eps in epsilons:
                uncertain_systematics = {syst: eps for syst in individual_systematics}
                # Update current instance instead of creating a new one
                self.uncertain_systematics = uncertain_systematics
                self.V_Blue_inv, self.C_matrix_inv, self.Gamma_factors = self.compute_likelihood_matrices()
                if isinstance(method, str) and method.startswith('order_'):
                    # Extract N from 'order_N'
                    try:
                        N = int(method.split('_')[1])
                    except (IndexError, ValueError):
                        raise ValueError("Invalid method format. Expected 'order_N' where N is an integer.")

                    # Compute mu MLE up to order N
                    mu_mle = self.compute_mu_mle_at_order(max_order=N)[0]
                    
                else:
                    fit_result = self.minimize_chi2(method = method)
                    mu_mle = fit_result['mu']
                    
                
                mu_values.append(mu_mle)
                epsilon_values.append(eps)

                if mu_base is None:  # Set mu_base for the first epsilon value
                    mu_base = mu_mle

                percentage_variation_eps = (mu_mle - mu_base) / mu_base * 100
                percentage_variation.append(percentage_variation_eps)

            percentage_variations.append(percentage_variation)
            label = ' + '.join(individual_systematics) if len(individual_systematics) > 1 else individual_systematics[0]
            label = replace_labels(label)
            plt.plot(epsilon_values, mu_values, '--o', label=f'{label}')
            
        #plt.axhline(y=172.51, color='black', linestyle='-.', label='Original Combination')
        plt.xlabel("$\epsilon_s$", fontsize=24)
        plt.ylim(ylim)
        plt.xlim(0.0, eps_end)
        plt.ylabel('Central Value $m_t$ (GeV)', fontsize=20)
        plt.axhline(y=172.51, color='black', linestyle='dashdot', label='Original Combination')
        plt.legend(fontsize=15)
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()
        # Add the text to the top left corner with a fancy background
        
        
        plt.text(0.02, 0.95, "Includes fictitious measurement:\n$m_{t}^{\mathrm{NEW}} = 174.5 \pm 0.2 \pm 0.25$ GeV", fontsize=16,
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8))
        
        
        if output_path:
            plt.savefig(output_path + "_CV.pdf")
        
        
        
        plt.show()

        plt.figure(figsize=(11, 7))
        for i, syst_group in enumerate(systematics_input):
            label = ' + '.join(syst_group.split(', ')) if ', ' in syst_group else syst_group
            plt.plot(epsilon_values, percentage_variations[i], '--o', label=f'{label}')

        plt.xlabel("$\epsilon_s$", fontsize=20)
        plt.xlim(0.0, eps_end)
        plt.ylabel('Central Value $m_t$ (GeV)', fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path + "_relative_CV.pdf")
        plt.show()
        
        return mu_values
        
    def analyze_systematics_combination_CI(self, systematics_input, eps_end=0.6, n_points=10, ylim=None,
                                       bartlett_precision=0, bartlett_json_path=None, output_path=None, method='L-BFGS-B'):
        # Prepare the epsilon range
        epsilons = np.linspace(0.0, eps_end, n_points)
        percentage_variations = []
        All_CI = []
        # Load the JSON data if provided
        if bartlett_json_path:
            with open(bartlett_json_path, 'r') as f:
                bartlett_data = json.load(f)
        else:
            bartlett_data = None

        # Initialize figures for plotting
        plt.figure(figsize=(11, 7))
        
        for syst_group in systematics_input:
            individual_systematics = syst_group.split(', ')
            epsilon_values = []
            CI_half_sizes = []
            percentage_variation = []
            initial_CI_half_size = None  # To store the initial CI half-size

            for eps in epsilons:
                # Update uncertain_systematics for current epsilon value
                uncertain_systematics = {syst: eps for syst in individual_systematics}
                self.uncertain_systematics = uncertain_systematics
                self.V_Blue_inv, self.C_matrix_inv, self.Gamma_factors = self.compute_likelihood_matrices()
                self.fit_results = self.minimize_chi2()

                if bartlett_json_path:
                    try:
                        b = self.compute_bartlett_correction_from_file(bartlett_data, individual_systematics[0], eps)
                    except ValueError as e:
                        print(e)
                        continue
                    mu_low, mu_up, CI_half_size = self.find_confidence_interval(b=b, method=method)
                else:
                    mu_low, mu_up, CI_half_size = self.find_confidence_interval(bartlett_precision=bartlett_precision, method=method)

                epsilon_values.append(eps)
                CI_half_sizes.append(CI_half_size)

                # Calculate percentage variation from the initial value
                if initial_CI_half_size is None:  # First iteration sets the baseline
                    initial_CI_half_size = CI_half_size
                    percentage_variation_eps = 0  # No variation from itself
                else:
                    percentage_variation_eps = ((CI_half_size - initial_CI_half_size) / initial_CI_half_size) * 100

                percentage_variation.append(percentage_variation_eps)
            
            All_CI.append(CI_half_sizes)
            percentage_variations.append(percentage_variation)

            # Plotting CI half-size as a function of epsilon
            label = ' + '.join(individual_systematics) if len(individual_systematics) > 1 else individual_systematics[0]
            label = replace_labels(label)
            plt.plot(epsilon_values, CI_half_sizes, '--o', label=f'{label}')
            plt.legend(fontsize=12, loc='lower left')
            plt.xlabel('$\epsilon_s$', fontsize=24)
            plt.ylabel('$68.3\%$ Half-Size Confidence Interval (GeV)', fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=16)
            if ylim:
                plt.ylim(ylim)
            plt.xlim(0.0, eps_end)
            plt.grid(True)
            plt.tight_layout()
            
            
            plt.text(0.02, 0.95, "Includes fictitious measurement:\n$m_{t}^{\mathrm{NEW}} = 174.5 \pm 0.2 \pm 0.25$ GeV", fontsize=16,
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8))
         
         
            
            if output_path:
                plt.savefig(output_path + "_CI" + ".pdf")
        
        plt.figure(figsize=(11, 7))
        # Plotting percentage variation of CI half-size
        for i, syst_group in enumerate(systematics_input):
            label = ' + '.join(syst_group.split(', ')) if ', ' in syst_group else syst_group
            plt.plot(epsilon_values, percentage_variations[i], '--o', label=f'{label}')

        plt.xlabel('$\epsilon_s$', fontsize=20)
        plt.ylabel('Percentage Variation (%)', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path + "_relative_CI" + ".pdf")

        plt.show()
        return All_CI
        
    def generate_toy_data(self, N_toys):
        lik_ratios = []  # Initialize an empty list to store the likelihood ratios
        q_values = []
        
        # Pre-compute the inverse of V_Blue_inv outside the loop
        V_Blue = np.linalg.inv(self.V_Blue_inv)
        eigenvalues = np.linalg.eigvalsh(V_Blue)
        
        if not np.all(eigenvalues > 0):
            print(eigenvalues)
        
        # Pre-compute the inverses of each C_matrix_inv outside the loop
        C_matrices = {key: np.linalg.inv(self.C_matrix_inv[key]) for key in self.C_matrix_inv.keys()}

        for _ in range(N_toys):  # Loop to generate N_toys likelihood ratios
            # Use the fit results for mu and thetas_flat
            mu = self.fit_results['mu']
            thetas_flat = self.fit_results['thetas']

            # Check if thetas_flat is zero-dimensional (no auxiliary parameters)
            if len(thetas_flat) == 0:
                systematic_adjustments = 0
            else:
                thetas = []
                start = 0
                for key in self.C_matrix_inv.keys():
                    len_theta_i = self.C_matrix_inv[key].shape[0]  # Assuming square C_inv matrices
                    thetas.append(np.array(thetas_flat[start:start + len_theta_i]))
                    start += len_theta_i

                systematic_adjustments = np.sum(
                    [self.Gamma_factors[key] @ thetas[i] for i, key in enumerate(self.Gamma_factors.keys())], axis=0
                )
                
            if len(thetas_flat) == 0:
                mean_y = np.full(self.m_t.shape, mu)
            else:
                mean_y = mu + systematic_adjustments
                
            y = np.random.multivariate_normal(mean_y, V_Blue)

            u_flat = np.array([])
            if len(thetas_flat) > 0:
                u = []
                for i, key in enumerate(self.C_matrix_inv.keys()):
                    theta_i = thetas[i] 
                    C_inv_i = self.C_matrix_inv[key]
                    C_i = C_matrices[key]
                    
                    eps = self.uncertain_systematics[key]
                    cols_with_all_zeros = np.sum(np.all(self.Gamma_factors[key] == 0, axis=0))
                    N_s = len(theta_i) - cols_with_all_zeros
                    S = 1. # + 2.*eps**2*theta_i@C_inv_i@theta_i) / (1. + 2.*N_s*eps**2)
                                        
                    u_i = np.random.multivariate_normal(theta_i, C_i*S)      
                    u.append(u_i)
                    
                u_flat = np.concatenate(u)

            s = []
            if len(self.uncertain_systematics) > 0:
                for i, key in enumerate(self.C_matrix_inv.keys()):
                    theta_i = thetas[i] 
                    C_inv_i = self.C_matrix_inv[key]
                    eps = self.uncertain_systematics[key]

                    cols_with_all_zeros = np.sum(np.all(self.Gamma_factors[key] == 0, axis=0))
                    N_s = len(theta_i) - cols_with_all_zeros
                    S = 1. #+ 2.*eps**2*theta_i@C_inv_i@theta_i) / (1. + 2.*N_s*eps**2)
                    
                    alpha = 1. / (4 * eps**2)
                    beta = 1. / (4 * S * eps**2)
                    
                    s_i = gamma.rvs(a=alpha, scale=1./beta)
                    
                    s.append(s_i)

            else:
                s = np.array([])

            lik_ratio = self.likelihood_ratio_test(mu, method='L-BFGS-B', y=y, u=u_flat, s=s)
            q = self.goodness_of_fit(method='L-BFGS-B', y=y, u=u_flat, s=s)
            
            lik_ratios.append(lik_ratio)
            q_values.append(q)

        return np.array(lik_ratios), np.array(q_values)
    
    def compute_bartlett_correction(self, N_toys=10000, output_path=None):
        print('Generating toys...')
        lik_ratios, q_values = self.generate_toy_data(N_toys=N_toys)
        b_lik_ratios = np.nanmean(lik_ratios)
        b_q_values = np.nanmean(q_values)

        # Save both lik_ratios and q_values in a .npz file if a path is provided
        if output_path:
            np.savez(output_path, lik_ratios=lik_ratios, q_values=q_values)

        return b_lik_ratios, b_q_values
    
    def compute_bartlett_correction_from_file(self, bartlett_data, systematic, eps):
        eps_str = f"{eps:.2f}"
        try:
            lik_ratio = bartlett_data[systematic][eps_str]['lik_ratio']
            b = lik_ratio
        except KeyError:
            b = 1.
            print(f"Bartlett correction not found for systematic {systematic} and epsilon {eps_str}")
        return b
    
    def analyze_and_save_bartlett_corrections(self, systematics_input, eps_end=0.6, n_points=10, N_toys=10000, output_path='bartlett_results'):

        # Ensure the output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Prepare the epsilon range
        epsilons = np.linspace(0.01, eps_end, n_points)

        # Collect Bartlett corrections for printing
        bartlett_corrections = []

        for syst_group in systematics_input:
            individual_systematics = syst_group.split(', ')
            for eps in epsilons:
                # Update uncertain_systematics for current epsilon value
                uncertain_systematics = {syst: eps for syst in individual_systematics}
                self.uncertain_systematics = uncertain_systematics
                self.V_Blue_inv, self.C_matrix_inv, self.Gamma_factors = self.compute_likelihood_matrices()
                self.fit_results = self.minimize_chi2()

                # Naming convention for files
                filename_suffix = f"{self.comb}_" + '_'.join(individual_systematics) + f'_eps{eps:.2f}'
                filename = f'lik_q_values_{filename_suffix}.npz'
                full_path = os.path.join(output_path, filename)

                # Compute the Bartlett correction and save the values
                b_correction, b_q_value = self.compute_bartlett_correction(N_toys=N_toys, output_path=full_path)
                bartlett_corrections.append((self.comb, individual_systematics, eps, b_correction, b_q_value))

                print(f"Saved values for combination: {self.comb}, systematics: {', '.join(individual_systematics)} with eps={eps:.2f} to {full_path}")

        # Print Bartlett corrections as the final line
        print("\nBartlett corrections:")
        for comb, systs, eps, b_corr, b_q in bartlett_corrections:
            print(f"Combination: {comb} | Systematics: {', '.join(systs)} | Epsilon: {eps:.2f} | Bartlett Correction: {b_corr} | Q-value Correction: {b_q}")
        
label_replacements = {
    'new': "NEW",
    'LHCbJES': "b-JES",
    'btag': "b tagging",
    'ME': "ME generator",
    'LHCJES1': "JES 1",
    'LHCJES2': "JES 2",
    'method': "Method",
    'CMSbHad': "CMS b hadron BR",
    'LHCrad': "QCD radiation",
}

def replace_labels(original_label):
    """
    Function to replace labels based on a predefined dictionary.
    """
    # Splitting the original label to handle multiple systematics combined with '+'
    systematics = original_label.split(' + ')
    new_labels = [label_replacements.get(syst, syst) for syst in systematics]
    return ' + '.join(new_labels)