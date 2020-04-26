import numpy as np

class OptFunction:
    '''
    The OptFunction object is a simple interface for a test function.
    :param noise: Whether to add gaussian noise to the function output
    :type noise: bool
    :param noise_stddev: Standard deviation of the added gaussian noise
    :type noise_stddev: float
    :param xmin: The lower bound on the cubic parameter space
    :type xmin: float
    :param xmax: The upper bound on the cubic parameter space
    :type xmax: float
    '''
    def __init__(self, noise=False, noise_stddev=1, xmin=0, xmax=1):
        self.config(noise=noise, noise_stddev=noise_stddev, xmin=xmin, xmax=xmax)
        # Function scaling parameters
        self.a = 0
        self.b = 1

    def __call__(self, x):
        # Rescale x
        x_scaled = self.rescale(x)

        # Evaluate
        y = self.function(x_scaled)

        # Add optinal noise
        if self.noise:
            gauss = np.random.normal(0, self.noise_stddev, y.shape)
            return y + gauss
        else:
            return y

    def config(self, 
               noise=None, 
               noise_stddev=None, 
               xmin=None, 
               xmax=None):
        if noise is not None:
            self.noise = noise
        if noise_stddev is not None:
            self.noise_stddev = noise_stddev
        if xmin is not None:
            self.xmin = xmin
        if xmax is not None:
            self.xmax = xmax
        try:
            assert self.xmin < self.xmax
        except TypeError:
            print("Please provide numerical upper and lower limits.")
    
    def rescale(self, x):
        '''
        Implements linear scaling from [xmin,xmax] to [a,b]
        Reimplement this function as needed for your function
        :param x: Array of numbers, any dimensions
        :type x: Numpy array
        :output x: Array of numbers, same dimensions as x
        '''
        return self.a + ((x - self.xmin)*(self.b - self.a))/(self.xmax - self.xmin)

    @staticmethod
    def function(x):
        '''
        Dummy function, reimplement for your function
        :param x: Input array of numbers, any dimension
        :type x: Numpy array
        :output y: First dimension collapsed, but otherwise same size as x
        '''
        return sum(x)

class Rastrigin(OptFunction):
    def __init__(self, noise=False, noise_stddev=1, xmin=0, xmax=1):
        super().__init__(noise, noise_stddev, xmin, xmax)
        self.a = -5.12
        self.b = 5.12

    @staticmethod
    def function(x):
        A = 10
        d = x.shape[0]
        xsq = np.power(x,2)
        wave = np.cos(2*np.pi*x)
        return A*d + np.sum(xsq + A*wave,axis=0)

class Michalewicz(OptFunction):
    def __init__(self, noise=False, noise_stddev=1, xmin=0, xmax=1):
        super().__init__(noise, noise_stddev, xmin, xmax)
        self.a = 0
        self.b = np.pi
    
    @staticmethod
    def function(x):
        m = 10
        d = x.shape[0]
        i = (np.arange(d) + 1)/np.pi
        xsq = np.power(x,2)
        s = (i*xsq.T).T
        return -np.sum(np.sin(x)*(np.power(np.sin(s),2*m)), axis=0)

class Zakharov(OptFunction):
    def __init__(self, noise=False, noise_stddev=1, xmin=0, xmax=1):
        super().__init__(noise, noise_stddev, xmin, xmax)
        self.a = -5
        self.b = 10

    @staticmethod
    def function(x):
        d = x.shape[0]
        i = (np.arange(d) + 1)*0.5
        xsq = np.power(x,2)
        s = np.sum((i*x.T).T, axis=0)
        return np.sum(xsq, axis=0) + np.power(s,2) + np.power(s,4)

class StyblinskiTang(OptFunction):
    def __init__(self, noise=False, noise_stddev=1, xmin=0, xmax=1):
        super().__init__(noise, noise_stddev, xmin, xmax)
        self.a = -5
        self.b = 5

    @staticmethod
    def function(x):
        return 0.5*np.sum(np.power(x, 4) - 16*np.power(x, 2) + 5*x, axis=0)