import tensorflow as tf
from einops.layers.tensorflow import Rearrange
import math
from scipy import signal
import numpy as np

link_map = tf.constant([
    [0, 6, 2, 9, 2,  8,  2,  4,  10,  7, 1, 1],
    [3, 9, 3, 8, 8, 10,  4,  10,  7, 11, 4, 5]
])

# link_map = tf.constant([
#     [2,  8,  2,  4],
#     [8, 10,  4,  10]
# ])
# link_map = tf.constant([
#     [ 8,  2],
#     [10,  4]
# ])



def low_pass_filter(data, N, fs, cutoff):
    ''' 
    Discrete Low pass filter
    N: Order
    fs: Data sampling frequency
    cutoff: cutoff frequency
    '''
    output = []
    b, a = signal.butter(N, cutoff, fs=fs, btype='low', analog=False)
    if len(data.shape)==2:
        num_chunk, dim = data.shape
        for i in range(num_chunk):
            input = np.pad(data[i], (50, ), 'symmetric')
            filtered = signal.filtfilt(b, a, input)
            output.append(filtered[50:-50])

        output = np.array(output)
    elif len(data.shape)==3:
        num_chunk, frame, dim = data.shape
        for i in range(num_chunk):
            output.append(np.array([signal.filtfilt(b, a, data[i, :, 0]),
                                    signal.filtfilt(b, a, data[i, :, 1]), 
                                    signal.filtfilt(b, a, data[i, :, 2])]))
        output = np.array(output).transpose(0, 2, 1)
    return output

@tf.function()
def low_pass_filter_tf(data, b, a):
    frame, _, _, _ = data.shape
    output = tf.TensorArray(tf.float32, size=data.shape[0])
    value_0 = tf.multiply(b[0], data[0])
    value_1 = tf.multiply(b[0], data[1]) + tf.multiply(b[1], data[0]) - tf.multiply(a[1], value_0)
    output = output.write(0, value_0)
    output = output.write(1, value_1)
    
    for i in tf.range(2, frame):
        value_dummy = value_1
        value_1 = tf.multiply(b[0], data[i]) + tf.multiply(b[1], data[i - 1]) + tf.multiply(b[2], data[i - 2]) - tf.multiply(a[1], value_1) - tf.multiply(a[2], value_0)
        value_0 = value_dummy
        output = output.write(i, value_1)
    
    output = output.stack()
    return output[50:-50]

@tf.function()
def apply_filter(data, b, a):
    paddings = tf.constant([[0, 0], [50, 50], [0, 0], [0, 0]])
    data = tf.pad(data, paddings, 'SYMMETRIC')
    data = tf.transpose(data, perm=[1, 0, 2, 3])
    b = b/a[0]
    a = a/a[0]
    batch, frame, joint, dim = data.shape
    output = low_pass_filter_tf(data, b, a)

    return tf.transpose(output, perm=[1, 0, 2, 3])

def L2_6D(y, x):
    """ 
    Compute L2 Distance of 6D rotation representation
    Input:
        Rotation matrix (batch, frames, joints, 3, 3)
    Output:
        Average L2 distance
    """
    rot = y - x
    repre = rot[:, :, :, :2, :]
    repre = Rearrange('b f j c d->b f j (c d)')(repre)

    return tf.math.reduce_mean(tf.math.reduce_mean(tf.math.reduce_mean(tf.norm(repre, axis=-1, ord=2)**2, axis=-1), axis=-1), axis=-1)


def geodesic(y, x):
    """ 
    Compute geodesic error
    Loss = cos^{-1}((M''_{00}+M''_{11}+M''_{22}-1)/2), M''=My(Mx)^{-1}
    Input:
        y(My): Ground truth rotation matrix (batch, frame, joint, 3, 3)
        x(Mx): Estimated rotation matrix (batch, frame, joint, 3, 3)
    Output:
        loss
    Reference:
        Paper: Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, Hao Li, "On the Continuity of Rotation Representations in Neural Networks", 2019, CVPR
        Git: https://github.com/papagina/RotationContinuity
    """ 
    epsilon = 1e-3
    batch, frame, joint, _, _ = x.shape
    Mx = tf.transpose(x, [0, 1, 2, 4, 3])
    M = tf.linalg.matmul(y, Mx)
    cos = (M[:, :, :, 0, 0] + M[:, :, :, 1, 1] + M[:, :, :, 2, 2] - 1) / 2
    cos = tf.clip_by_value(cos, -1+epsilon, 1-epsilon)
    # cos = tf.minimum(cos, tf.ones((batch, frame, joint)))
    # cos = tf.maximum(cos, -1 * tf.ones((batch, frame, joint)))
    theta = tf.math.acos(cos)
    return tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(theta, axis=-1), axis=-1), axis=-1)
    # return tf.cast(tf.math.reduce_mean(value), dtype=tf.float64)

def avg_jpe(x):
    return tf.math.reduce_mean(tf.math.reduce_mean(tf.math.reduce_mean((tf.math.reduce_euclidean_norm(x + 1e-8,axis=-1)),axis=-1), axis = -1), axis = -1)


# def kinematic_loss(x, clip=1):
#     """ 
#     Compute joint length consistency across the frame
#     """
#     # batch, frame, joint, 3
#     num_joint = x.shape[2]
#     if num_joint != 18:
#         print("Number of joints is not 18")
#         return 0
    
#     link = tf.norm(tf.gather(x, link_map[0], axis=2) - tf.gather(x, link_map[1], axis=2), axis=-1, ord=2)
#     if clip < 1:
#         link = (link - tf.math.reduce_min(link, axis=-2, keepdims=True))/(tf.math.reduce_max(link, axis=-2, keepdims=True) - tf.math.reduce_min(link, axis=-2, keepdims=True) + 1e-6)    
#     std = tf.math.reduce_std(link, axis=-2)
#     std = tf.clip_by_value(std, clip, 1000)
#     return tf.math.reduce_mean(std)

def kinematic_loss(x, clip=1):
    """ 
    Compute joint length consistency across the frame
    """
    # batch, frame, joint, 3
    num_joint = x.shape[2]
    if num_joint != 18:
        print("Number of joints is not 18")
        return 0
    
    link = tf.norm(tf.gather(x, link_map[0], axis=2) - tf.gather(x, link_map[1], axis=2), axis=-1, ord=2)
    vel = link[:, 1:] - link[:, :-1]
    vel = tf.math.abs(vel)
    return tf.math.reduce_mean(vel)

def window_kinematic_loss(x, window_size=5):
    # batch, frame, joint
    link = tf.norm(tf.gather(x, link_map[0], axis=2) - tf.gather(x, link_map[1], axis=2), axis=-1, ord=2)
    # normed_link = (link - tf.math.reduce_min(link, axis=-2, keepdims=True))/(tf.math.reduce_max(link, axis=-2, keepdims=True) - tf.math.reduce_min(link, axis=-2, keepdims=True) + 1e-6)    
    windowed_link = tf.signal.frame(link, frame_length=window_size, frame_step=window_size, axis=1)
    std = tf.math.reduce_std(windowed_link, axis=2) 

    return tf.math.reduce_mean(std)

def mpjae(y,x, filtered=False, cos=False):
    # y = tf.reshape(tf.cast(y, dtype=tf.float32), [tf.shape(y)[0],tf.shape(y)[1],-1,3])
    # x = tf.reshape(tf.cast(x, dtype=tf.float32), [tf.shape(x)[0],tf.shape(x)[1],-1,3])
    accel_gt = y[:, :-2] - 2 * y[:, 1:-1] + y[:, 2:]
    accel_pred = x[:, :-2] - 2 * x[:, 1:-1] + x[:, 2:]
    if filtered:
        frame = accel_gt.shape[1] / 3
        b, a = signal.butter(2, int(frame/4), fs=frame, btype='low', analog=False)
        b = tf.constant(b, dtype=tf.float32)
        a = tf.constant(a, dtype=tf.float32)
        accel_gt = apply_filter(accel_gt, b, a)
        accel_pred = apply_filter(accel_pred, b, a)

        weights = tf.norm(accel_gt, axis=-1, ord=2)
        norm_value = weights * tf.math.reduce_euclidean_norm(accel_pred - accel_gt + 1e-8,axis=-1)
        return tf.math.reduce_mean(tf.math.reduce_mean(tf.math.reduce_mean(norm_value, axis=-1), axis = -1), axis = -1)
            
    if cos:
        cos_sim = max_cos_sim(accel_gt, accel_pred)
        return cos_sim
    
    diff = accel_pred - accel_gt
    value = tf.reduce_sum(diff ** 2, axis=-1) + 1e-8
    norm = tf.sqrt(value)
    return tf.math.reduce_mean(norm)
    # return tf.math.reduce_mean(tf.math.reduce_mean(tf.math.reduce_mean((tf.math.reduce_euclidean_norm(accel_pred - accel_gt + 1e-8,axis=-1)),axis=-1), axis = -1), axis = -1)

def mpjve(y, x, filtered=False, audio=False, cos=False):
    vel_gt = y[:, 1:] - y[:, :-1]
    if not audio:
        vel_pred = x[:, 1:] - x[:, :-1]
    else:
        vel_pred = x
        vel_gt = tf.concat((vel_gt, vel_gt[:, -1:, :, :]), axis=1)
    if filtered:
        frame = vel_gt.shape[1] / 3
        b, a = signal.butter(2, int(frame/4), fs=frame, btype='low', analog=False)
        b = tf.constant(b, dtype=tf.float32)
        a = tf.constant(a, dtype=tf.float32)
        vel_gt = apply_filter(vel_gt, b, a)
        vel_pred = apply_filter(vel_pred, b, a)

    if cos:
        cos_sim = max_cos_sim(vel_gt, vel_pred)
        return cos_sim

    diff = vel_pred - vel_gt
    norm = norm_stable(diff)
    return tf.math.reduce_mean(norm)
    # return tf.math.reduce_mean(tf.math.reduce_mean(tf.math.reduce_mean((tf.math.reduce_euclidean_norm(vel_pred - vel_gt + 1e-8,axis=-1)),axis=-1), axis = -1), axis = -1)

def mpjpe_jointwise(y,x):
    return tf.math.reduce_mean(tf.math.reduce_mean((tf.math.reduce_euclidean_norm(x - y + 1e-8,axis=-1)),axis=-2), axis = -2)

def weight_gen(y, ord=1):
    if ord==1:
        diff = y[:, 1:] - y[:, :-1] # batch, frame, joint, dim
        diff = tf.norm(diff, axis=-1, ord=2)    # batch, frame, joint
        diff = (diff - tf.reduce_min(diff)) / (tf.reduce_max(diff) - tf.reduce_min(diff))
        weight = tf.where(diff < 0.3, 0.3, diff)
        weight = tf.concat((weight, weight[:, -1:, :]), axis=1)
        return tf.cast(1/weight, dtype=tf.float32)
    elif ord==2:
        weight = tf.norm(y[:, :-2] - 2 * y[:, 1:-1] + y[:, 2:], axis=-1, ord=2)
        weight = tf.where(weight > 10.0, 10.0, weight)
        weight = tf.concat((weight, weight[:, -2:, :]), axis=1)

        return tf.cast(weight, dtype=tf.float32)
    
    elif ord==3:
        v_weight = weight_gen(y, ord=1)
        a_weight = weight_gen(y, ord=2)
        return tf.cast(v_weight * a_weight, dtype=tf.float32)

def normal_sim(y, x):
    """ 
    Compute norm similarity loss
    Input: batch, frame, joint, dim    
    """
    v_y = y[:, 1:] - y[:, :-1]
    v_x = x[:, 1:] - x[:, :-1]
    a_y = y[:, :-2] - 2 * y[:, 1:-1] + y[:, 2:]
    a_x = x[:, :-2] - 2 * x[:, 1:-1] + x[:, 2:]

    B_y = tf.linalg.cross(v_y[:, :-1], a_y)
    B_x = tf.linalg.cross(v_x[:, :-1], a_x)

    B_norm_y = norm_stable(B_y)[:, :, :, tf.newaxis]
    B_norm_x = norm_stable(B_x)[:, :, :, tf.newaxis]

    sim = max_cos_sim(B_y / B_norm_y, B_x / B_norm_x)

    return sim

def norm_stable(x):
    value = tf.reduce_sum(x ** 2, axis=-1) + 1e-8
    norm = tf.sqrt(value)
    return norm

def max_cos_sim(a, b):
    """ 
    Compute max cosine similarity
    Input: batch, frame, joint, dim
    """
    m = tf.maximum(norm_stable(a), norm_stable(b))
    sim = tf.einsum('bfjd,bfjd->bfj', a, b) / m**2
 
    return tf.reduce_mean(1 - sim)

def mpjpe(y, x, D_WEIGHT=False, ord=1):
    # y = tf.reshape(y, [tf.shape(y)[0],tf.shape(y)[1],-1,3])
    # x = tf.reshape(x, [tf.shape(x)[0],tf.shape(x)[1],-1,3])
    error = tf.cast(x - y, dtype=tf.float32)

    # value = tf.reduce_sum(error ** 2, axis=-1) + 1e-8
    # norm = tf.sqrt(value)
    norm = norm_stable(error)

    if D_WEIGHT:
        weight = weight_gen(y, ord=ord)
        return tf.math.reduce_mean(weight * norm)
    return tf.math.reduce_mean(norm)
    # return tf.math.reduce_mean(tf.math.reduce_mean(tf.math.reduce_mean((tf.math.reduce_euclidean_norm(tf.cast(x - y, dtype=tf.float32) + 1e-8,axis=-1)),axis=-1), axis = -1), axis = -1)

def mpjpeL1(y, x):
    return tf.math.reduce_mean(tf.math.reduce_mean(tf.math.reduce_mean((tf.norm(tf.cast(x - y, dtype=tf.float32),axis=-1, ord=1)),axis=-1), axis = -1), axis = -1)

def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrustes problem.
    Ensure that the first argument is the prediction

    Source: https://en.wikipedia.org/wiki/Kabsch_algorithm
    Reference: https://github.com/aymenmir1/3dpw-eval/tree/master
    
    Input:
        param S1: predicted joint positions array J x 3
        param S2: ground truth joint positions array J x 3
    Output:
        S1_hat: the predicted joint positions after apply similarity transform
    '''
    # If all the values in pred3d are zero then procrustes analysis produces nan values
    # Instead we assume the mean of the GT joint positions is the transformed joint value
    J, _ = S1.shape
    if not (np.sum(np.abs(S1)) == 0):
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = np.transpose(S1)
            S2 = np.transpose(S2)
            transposed = True
        assert (S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1 ** 2)

        # 3. The outer product of X1 and X2.
        K = X1.dot(X2.T)

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        # Construct R.
        R = V.dot(Z.dot(U.T))

        # 5. Recover scale.
        scale = np.trace(R.dot(K)) / var1

        # 6. Recover translation.
        t = mu2 - scale * (R.dot(mu1))

        # 7. Error:
        S1_hat = scale * R.dot(S1) + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat
    else:
        S1_hat = np.tile(np.mean(S2, axis=0), (J, 1))
        R = np.identity(3)

        return S1_hat

def align_by_center(joint):
    """ 
    Align all the joints with respect to root joint
    Input: size=(Batch, Frame, J, 3)
        Upper body case: J=8
            Center: neck = (joint_0 + joint_4)/2
        Whole body case: J=17
            Center: mid-pelvis = (joint_2 + joint_8)/2
    Output:
        Centered keypoints
    """
    shape = joint.shape

    if len(shape)==3:
        j = shape[1]
        joint = np.transpose(joint, [1, 0, 2])  # f j 3 -> j f 3
    elif len(shape)==4:
        j = shape[2]
        joint = np.transpose(joint, [2, 0, 1, 3])   # b f j 3 -> j b f 3

    if j==8:
        center = (joint[0] + joint[4])/2
    elif j==17 or j==18:
        center = (joint[2] + joint[8])/2
    
    if len(shape)==3:
        joint = np.transpose(joint, [1, 0, 2])  # j f 3 -> f j 3
        return joint - center[:, tf.newaxis, :]
    elif len(shape)==4:
        joint = np.transpose(joint, [1, 2, 0, 3])  #  j b f 3 -> b f j 3
        return joint - center[:, :, tf.newaxis, :]


def pa_mpjpe(y, x):
    """ 
    Compute PA-MPJPE (Procrustes MPJPE)
    Implementation reference: https://github.com/aymenmir1/3dpw-eval/tree/master
    Input:
        y: Ground truth keypoints shape (b, f, j, 3)
        x: Estimated keypoints shape (b, f, j, 3)
    Output:
        PA-MPJPE
    """
    batch, frame, _, _ = y.shape
    y = align_by_center(y)
    x = align_by_center(x)

    x_hat = []
    for i in range(batch):
        x_hat.append([compute_similarity_transform(x[i, j, :, :], y[i, j, :, :]) for j in range(frame)])
    
    x_hat = np.asarray(x_hat)

    return np.mean(np.linalg.norm(y-x_hat, ord=2, axis=-1))

def rje(y, x):
    num_joints = x.shape[2]
    if num_joints==14:
        # Right arm Joint Error (7: Elbow, 11: Wrist, 13: Hand)
        gt_joint = tf.concat((y[:, :, 7, :][:, :, tf.newaxis, :], y[:, :, 11, :][:, :, tf.newaxis, :], y[:, :, 13, :][:, :, tf.newaxis, :]), axis=2)
        est_joint = tf.concat((x[:, :, 7, :][:, :, tf.newaxis, :], x[:, :, 11, :][:, :, tf.newaxis, :], x[:, :, 13, :][:, :, tf.newaxis, :]), axis=2)
    if num_joints==12:
        # Right arm Joint Error (7: Elbow, 11: Wrist)
        gt_joint = tf.concat((y[:, :, 7, :][:, :, tf.newaxis, :], y[:, :, 11, :][:, :, tf.newaxis, :]), axis=2)
        est_joint = tf.concat((x[:, :, 7, :][:, :, tf.newaxis, :], x[:, :, 11, :][:, :, tf.newaxis, :]), axis=2)
    elif num_joints==8:
        # Right arm Joint Error (5: Elbow, 6: Wrist, 7: Hand)
        gt_joint = tf.concat((y[:, :, 5, :][:, :, tf.newaxis, :], y[:, :, 6, :][:, :, tf.newaxis, :], y[:, :, 7, :][:, :, tf.newaxis, :]), axis=2)
        est_joint = tf.concat((x[:, :, 5, :][:, :, tf.newaxis, :], x[:, :, 6, :][:, :, tf.newaxis, :], x[:, :, 7, :][:, :, tf.newaxis, :]), axis=2)

    return mpjpe(gt_joint, est_joint), mpjve(gt_joint, est_joint), mpjae(gt_joint, est_joint), gt_joint

def combined(y,x):
    return mpjpe(y,x) + mpjae(y,x)

def new_loss(y,x):
    y = tf.reshape(y, [tf.shape(y)[0],tf.shape(y)[1],-1,3])
    x = tf.reshape(x, [tf.shape(x)[0],tf.shape(x)[1],-1,3])
    # jerk_y = y[3:] - 3*y[2:-1] + 3*y[1:-2] - y[:-3]
    # jerk_x= x[3:] - 3*x[2:-1] + 3*x[1:-2] - x[:-3]
    # return tf.math.reduce_mean(tf.math.reduce_mean(tf.math.reduce_mean((tf.math.reduce_euclidean_norm(jerk_x - jerk_y,axis=-1)),axis=-1), axis = -1), axis = -1)
    v_y = y[1:] - y[:-1]
    v_x = x[1:] - x[:-1]
    return tf.math.reduce_mean(tf.math.reduce_mean(tf.math.reduce_mean((tf.math.reduce_euclidean_norm(v_x - v_y,axis=-1)),axis=-1), axis = -1), axis = -1)


