import numpy as np
from collections import defaultdict
from brainbox.behavior.dlc import get_smooth_pupil_diameter, get_pupil_diameter

def filtering_pass(y, m0, S0, C, R, A, Q, ensemble_vars):
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


def get_pupil_kalman_parameters(baseline_ensemble_preds, baseline_ensemble_stacks, pupil_locations, pupil_diameters):
    #compute center of mass
    com_obs = pupil_locations
    mean_x_obs = np.mean(com_obs[:,0])
    mean_y_obs = np.mean(com_obs[:,1])
    x_t_obs, y_t_obs = com_obs[:,0] - mean_x_obs, com_obs[:,1] - mean_y_obs #make the mean zero
    #compute diameter
    d_t_obs = pupil_diameters
    mean_d_t_obs = np.mean(d_t_obs)

    #latent variables (observed)
    z_t_obs = np.vstack((d_t_obs, x_t_obs, y_t_obs)) #latent variables - diameter, com_x, com_y

    ##### Set values for kalman filter #####
    m0 = np.asarray([np.mean(d_t_obs), 0.0, 0.0]) # initial state: mean
    S0 =  np.asarray([[np.var(d_t_obs), 0.0, 0.0], [0.0 , np.var(x_t_obs), 0.0], [0.0, 0.0 , np.var(y_t_obs)]]) # diagonal: var

    A = np.asarray([[.999, 0, 0], [0, .999, 0], [0, 0, .999]]) #state-transition matrix, parameters hand-picked for smoothing purposes
    Q = np.asarray([[np.var(d_t_obs)*(1-(A[0,0]**2)), 0, 0], [0, np.var(x_t_obs)*(1-A[1,1]**2), 0], [0, 0, np.var(y_t_obs)*(1-(A[2,2]**2))]]) #state covariance matrix

    #['pupil_top_r_x', 'pupil_top_r_y', 'pupil_bottom_r_x', 'pupil_bottom_r_y', 'pupil_right_r_x', 'pupil_right_r_y', 'pupil_left_r_x', 'pupil_left_r_y'] 
    C = np.asarray([[0, 1, 0], [-.5, 0, 1], [0, 1, 0], [.5, 0, 1], [.5, 1, 0], [0, 0, 1], [-.5, 1, 0], [0, 0, 1]]) # Measurement function
    R = np.eye(8) # placeholder diagonal matrix for ensemble variance

    baseline_ensemble_preds_copy = baseline_ensemble_preds.copy()
    baseline_ensemble_stacks_copy = baseline_ensemble_stacks.copy()

    #subtract COM means from the ensemble predictions
    for i in range(baseline_ensemble_preds.shape[1]):
        if i % 2 == 0:
            baseline_ensemble_preds_copy[:, i] -= mean_x_obs
        else:
            baseline_ensemble_preds_copy[:, i] -= mean_y_obs

    #subtract COM means from all the predictions
    for i in range(baseline_ensemble_preds.shape[1]):
        if i % 2 == 0:
            baseline_ensemble_stacks_copy[:,:,i] -= mean_x_obs
        else:
            baseline_ensemble_stacks_copy[:,:,i] -= mean_y_obs

    y = baseline_ensemble_preds_copy
    
    return m0, S0, A, Q, C, R, y, mean_x_obs, mean_y_obs