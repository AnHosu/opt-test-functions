import numpy as np

class OptTestFunction:
    def __init__(self, function=None, noise=False, noise_stddev=1.0, lower_limit=0, upper_limit=1):        
        self.functions = {
            "Langermann": {
                "function": self.langermann,
                "function_lower": -5.12,
                "function_upper": 5.12
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
        y = self.function["function"](x)
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
    def langermann(x):
        A = 10
        d = x.shape[0]
        xsq = np.power(x,2)
        wave = np.cos(2*np.pi*x)
        return A*d + np.sum(xsq + A*wave,axis=0)
