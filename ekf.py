import math 
import numpy as np

class RobotEKF:
    def __init__(
        self,
        dim_x=1, dim_z=1, dim_u=1,
        eval_gux=None, eval_Gt=None, eval_Vt=None,
        eval_hx=None, eval_Ht=None,
    ):
        """
        Initializes the extended Kalman filter creating the necessary matrices
        """
        self.mu = np.zeros((dim_x, 1))  # mean state estimate
        self.Sigma = np.eye(dim_x)  # covariance state estimate
        self.Mt = np.eye(dim_u)  # process noise
        self.Qt = np.eye(dim_z)  # measurement noise

        self.eval_gux = eval_gux
        self.eval_Gt = eval_Gt
        self.eval_Vt = eval_Vt

        self.eval_hx = eval_hx
        self.eval_Ht = eval_Ht

        self._I = np.eye(dim_x)  # identity matrix used for computations
    def predict(self, u, g_extra_args=()):
        x = self.mu[0,0]
        y = self.mu[1,0]
        theta = self.mu[2,0]
        if u.shape == (2,1):
        # Update the state prediction evaluating the motion model
                v = u[0,0]
                w = u[1,0]
                self.mu = self.eval_gux(x,y,theta,v,w, *g_extra_args)
         # Update the covariance matrix of the state prediction, 
        # you need to evaluate the Jacobians Gt and Vt
                Gt = self.eval_Gt(theta, v, w, *g_extra_args)
                Vt = self.eval_Vt(theta, v, w, *g_extra_args)
        else:
                rot1 = u[0,0]
                trasl = u[1,0]
                rot2 = u[2,0]
                self.mu = self.eval_gux(x,y,theta, trasl, rot1, rot2)
        # Update the covariance matrix of the state prediction, 
        # you need to evaluate the Jacobians Gt and Vt
                Gt = self.eval_Gt(theta, trasl, rot1)
                Vt = self.eval_Vt(theta, trasl, rot1)
        self.Sigma = Gt@self.Sigma@Gt.T + Vt@self.Mt@Vt.T

    def update(self, z, lmark, residual=np.subtract):

    # Convert the measurement to a vector if necessary. Needed for the residual computation
        if np.isscalar(z) and self.dim_z == 1:
                z = np.asarray([z], float)

        x = self.mu[0,0]
        y = self.mu[1,0]
        theta = self.mu[2,0]
        mx = lmark[0]
        my = lmark[1]
    # Compute the Kalman gain, you need to evaluate the Jacobian Ht
    # (mx, x, my, y)
        Ht = self.eval_Ht(mx, x, my, y)
        self.S = Ht@self.Sigma@Ht.T + self.Qt
        self.K = self.Sigma@Ht.T@np.inv(self.S)

    # Evaluate the expected measurement and compute the residual, then update the state prediction
    # (mx, x, my, y, theta)
        z_hat = self.eval_hx(mx, x, my, y, theta)
        self.y = residual(z, z_hat)
        self.mu = self.mu + self.K@(z - z_hat)

    # P = (I-KH)P(I-KH)' + KRK' is more numerically stable and works for non-optimal K vs the equation
    # P = (I-KH)P usually seen in the literature. 
    # Note that I is the identity matrix.
        I_KH = self._I - self.K@Ht
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + self.K @ self.Qt @ self.K.T
