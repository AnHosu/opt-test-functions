import numpy as np

class OptTestFunction:
    def __init__(self, function=None, noise=False, noise_stddev=1.0, lower_limit=0, upper_limit=1):        
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
            }
        }
        
        self.config(function, noise, noise_stddev, lower_limit, upper_limit)
        
    def config(self, function=None, noise=False, noise_stddev=1.0, lower_limit=0, upper_limit=1):
        try:
            self.function = self.functions[function]
        except KeyError:
            raise AssertionError('Supported functions are ' + " ".join(list(self.functions.keys())) + '. Received "{0}"'.format(function))
        self.noise = noise
        try:
            assert lower_limit < upper_limit
        except TypeError:
            print("Please provide numerical upper and lower limits.")
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.noise_stddev = noise_stddev
    
    def get(self, x):
        #rescale
        x_scaled = self.rescale(x=x, 
                                xmin=self.lower_limit, 
                                xmax=self.upper_limit, 
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