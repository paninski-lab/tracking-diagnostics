import numpy as np
from collections import defaultdict
from brainbox.behavior.dlc import get_pupil_diameter

def filtering_pass(y, m0, S0, C, R, A, Q, ensemble_vars):
    """Implements Kalman-filter from - https://random-walks.org/content/misc/kalman/kalman.html
    Args:
        y: np.ndarray
            shape (samples, n_keypoints)
        m0: np.ndarray
            shape (n_latents)
        S0: np.ndarray
            shape (n_latents, n_latents)
        C: np.ndarray
            shape (n_keypoints, n_latents)
        R: np.ndarray
            shape (n_keypoints, n_keypoints)
        A: np.ndarray
            shape (n_latents, n_latents)
        Q: np.ndarray
            shape (n_latents, n_latents)
        ensemble_vars: np.ndarray
            shape (samples, n_keypoints)

    Returns:
        mf: np.ndarray
            shape (samples, n_keypoints)
        Vf: np.ndarray
            shape (samples, n_latents, n_latents)
        S: np.ndarray
            shape (samples, n_latents, n_latents)
    """
    #time-varying observation variance
    for i in range(ensemble_vars.shape[1]):
        R[i,i] = ensemble_vars[0][i]
    T = y.shape[0]
    mf = np.zeros(shape=(T, m0.shape[0]))
    Vf = np.zeros(shape=(T, m0.shape[0], m0.shape[0]))
    S = np.zeros(shape=(T, m0.shape[0], m0.shape[0]))
    mf[0] = m0 + kalman_dot(y[0, :] - np.dot(C, m0), S0, C, R)
    Vf[0, :] = S0 - kalman_dot(np.dot(C, S0), S0, C, R)
    S[0] = S0
    
    for i in range(1, T):
        for t in range(ensemble_vars.shape[1]):
            R[t,t] = ensemble_vars[i][t]
        S[i-1] = np.dot(A, np.dot(Vf[i-1, :], A.T)) + Q
        y_minus_CAmf = y[i, :] - np.dot(C, np.dot(A, mf[i-1, :])) 
        
        mf[i, :] = np.dot(A, mf[i-1, :]) + kalman_dot(y_minus_CAmf, S[i-1], C, R)
        Vf[i, :] = S[i-1] - kalman_dot(np.dot(C, S[i-1]), S[i-1], C, R)
        
    return mf, Vf, S


def kalman_dot(array, V, C, R):
    
    R_CVCT = R + np.dot(C, np.dot(V, C.T))
    R_CVCT_inv_array = np.linalg.solve(R_CVCT, array)
    
    K_array = np.dot(V, np.dot(C.T, R_CVCT_inv_array))
    
    return K_array

def smooth_backward(y, mf, Vf, S, A, Q, C):
    """Implements Kalman-smoothing backwards from - https://random-walks.org/content/misc/kalman/kalman.html
    Args:
        y: np.ndarray
            shape (samples, n_keypoints)
        mf: np.ndarray
            shape (samples, n_keypoints)
        Vf: np.ndarray
            shape (samples, n_latents, n_latents)
        S: np.ndarray
            shape (samples, n_latents, n_latents)
        A: np.ndarray
            shape (n_latents, n_latents)
        Q: np.ndarray
            shape (n_latents, n_latents)
        C: np.ndarray
            shape (n_keypoints, n_latents)

    Returns:
        ms: np.ndarray
            shape (samples, n_keypoints)
        Vs: np.ndarray
            shape (samples, n_latents, n_latents)
        CV: np.ndarray
            shape (samples, n_latents, n_latents)
    """
    T = y.shape[0]
    ms = np.zeros(shape=(T, mf.shape[1]))
    Vs = np.zeros(shape=(T, mf.shape[1], mf.shape[1]))
    CV = np.zeros(shape=(T - 1, mf.shape[1], mf.shape[1]))
        
    # Last-time smoothed posterior is equal to last-time filtered posterior
    ms[-1, :] = mf[-1, :]
    Vs[-1, :, :] = Vf[-1, :, :]
        
    # Smoothing steps
    for i in range(T - 2, -1, -1):
        
        J = np.linalg.solve(S[i], np.dot(A, Vf[i])).T
        
        Vs[i] = Vf[i] + np.dot(J, np.dot(Vs[i+1] - S[i], J.T))
        ms[i] = mf[i] + np.dot(J, ms[i+1] - np.dot(A, mf[i]))
        CV[i] = np.dot(Vs[i+1], J.T)
        
    return ms, Vs, CV

def ensemble_median(markers_list, keys):
    """Computes ensemble median and variance of list of DLC marker dataframes
    Args:
        markers_list: list
            List of DLC marker dataframes
        keys: list
            List of keys in each marker dataframe
            
    Returns:
        ensemble_preds: np.ndarray
            shape (samples, n_keypoints)
        ensemble_vars: np.ndarray
            shape (samples, n_keypoints)
        ensemble_stacks: np.ndarray
            shape (n_models, samples, n_keypoints)
        keypoints_median_dict: dict
            keys: marker keypoints, values: shape (samples)
        keypoints_var_dict: dict
            keys: marker keypoints, values: shape (samples)
        keypoints_stack_dict: dict(dict)
            keys: model_ids, keys: marker keypoints, values: shape (samples)
    """
    ensemble_stacks = []
    ensemble_vars = []
    ensemble_preds = []
    keypoints_median_dict = {}
    keypoints_var_dict = {}
    keypoints_stack_dict = defaultdict(dict)
    for key in keys:
        stack = np.zeros((len(markers_list), markers_list[0].shape[0]))
        for k in range(len(markers_list)):
            stack[k] = markers_list[k][key]
        stack = stack.T
        median = np.median(stack, 1)
        var = np.var(stack, 1) / len(markers_list) #variance of the sample mean
        ensemble_preds.append(median)
        ensemble_vars.append(var)
        ensemble_stacks.append(stack)
        keypoints_median_dict[key] = median
        keypoints_var_dict[key] = var
        for i, keypoints in enumerate(stack.T):
            keypoints_stack_dict[i][key] = stack.T[i]
    ensemble_preds = np.asarray(ensemble_preds).T
    ensemble_vars = np.asarray(ensemble_vars).T
    ensemble_stacks = np.asarray(ensemble_stacks).T
    return ensemble_preds, ensemble_vars, ensemble_stacks, keypoints_median_dict, keypoints_var_dict, keypoints_stack_dict