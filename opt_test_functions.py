import numpy as np

class OptTestFunction:
    '''
    The OptTestFunction object is a simple interface for a test function.
    :param function: The name of the function to use
    :type function: str
    :param noise: Whether to add gaussian noise to the function output
    :type noise: bool
    :param noise_stddev: Standard deviation of the added gaussian noise
    :type noise_stddev: float
    :param xmin: The lower bound on the cubic parameter space
    :type xmin: float
    :param xmax: The upper bound on the cubic parameter space
    :type xmax: float
    '''
    def __init__(self, function=None, noise=False, noise_stddev=1.0, xmin=0, xmax=1):
               
        self.functions = {
            "Rastrigin": {
                "function": self.rastrigin,
                "function_lower": -5.12,
                "function_upper": 5.12
            },
            "Michalewicz": {
                "function": self.michalewicz,
                "function_lower": 0,
                "function_upper": np.pi
            },
            "Zakharov": {
                "function": self.zakharov,
                "function_lower": -5,
                "function_upper": 10
            },
            "Styblinski": {
                "function": self.styblinski_tang,
                "function_lower": -5,
                "function_upper": 5
            }
        }
        
        self.config(function, noise, noise_stddev, xmin, xmax)
        
    def config(self, function=None, noise=False, noise_stddev=1.0, xmin=0, xmax=1):
        try:
            self.function = self.functions[function]
        except KeyError:
            raise AssertionError('Supported functions are ' + " ".join(list(self.functions.keys())) + '. Received "{0}"'.format(function))
        self.noise = noise
        try:
            assert xmin < xmax
        except TypeError:
            print("Please provide numerical upper and lower limits.")
        self.xmin = xmin
        self.xmax = xmax
        self.noise_stddev = noise_stddev
    
    def get(self, x):
        #rescale
        x_scaled = self.rescale(x=x, 
                                xmin=self.xmin, 
                                xmax=self.xmax, 
                                a=self.function["function_lower"], 
                                b=self.function["function_upper"])
        #evaluate
        y = self.function["function"](x_scaled)
        #add noise
        if self.noise:
            gauss = np.random.normal(0, self.noise_stddev, y.shape)
            return y + gauss
        else:
            return y

    @staticmethod
    def rescale(x, xmin, xmax, a, b):
        return a + ((x - xmin)*(b - a))/(xmax - xmin)
    
    @staticmethod
    def rastrigin(x):
        A = 10
        d = x.shape[0]
        xsq = np.power(x,2)
        wave = np.cos(2*np.pi*x)
        return A*d + np.sum(xsq + A*wave,axis=0)
    
    @staticmethod
    def michalewicz(x):
        m = 10
        d = x.shape[0]
        i = (np.arange(d) + 1)/np.pi
        xsq = np.power(x,2)
        s = (i*xsq.T).T
        return -np.sum(np.sin(x)*(np.power(np.sin(s),2*m)), axis=0)

    @staticmethod
    def zakharov(x):
        d = x.shape[0]
        i = (np.arange(d) + 1)*0.5
        xsq = np.power(x,2)
        s = np.sum((i*x.T).T, axis=0)
        return np.sum(xsq, axis=0) + np.power(s,2) + np.power(s,4)

    @staticmethod
    def styblinski_tang(x):
        return 0.5*np.sum(np.power(x, 4) - 16*np.power(x, 2) + 5*x, axis=0)