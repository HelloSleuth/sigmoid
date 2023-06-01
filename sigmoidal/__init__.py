import math
import numpy as np
import scipy


class Sigmoid:
    def __new__(cls, *args, **kwargs):
        cls.fn_derivative = {
            0: cls._sigmoid,
            1: cls._derivative_1,
            2: cls._derivative_2,
        }
        obj = super().__new__(cls)
        return obj

    def __init__(self, L=1.0, x0=0.0, k=1.0, b=0.0, derivative=0, symbol='x', e_symbol='exp'):
        self.L = L
        self.x0 = x0
        self.k = k
        self.b = b
        self.derivative = derivative
        self.symbol = symbol
        self.e_symbol = e_symbol

    def __call__(self, x):
        return self.__class__.fn_derivative[self.derivative](x, self.L, self.x0, self.k, self.b)

    def __repr__(self):
        L, x0, k, b, derivative, x, e = self.L, self.x0, self.k, self.b, self.derivative, self.symbol, self.e_symbol
        return f'{self.__class__.__name__}({L=}, {x0=}, {k=}, {b=}, {derivative=}, symbol={x}, e_symbol={e})'

    def __str__(self):
        L, x0, k, b, derivative, x, e = self.L, self.x0, self.k, self.b, self.derivative, self.symbol, self.e_symbol
        if derivative == 0:
            return f'{L} / ( 1 + {e}(-{k}*({x}-{x0})) ) + {b}'
        elif derivative == 1:
            return f'( {k}*{L}*{e}(-{k}*({x}-{x0})) ) / ( ( 1+{e}(-{k}*({x}-{x0})) )**2 )'
        elif derivative == 2:
            e_kxx0 = f'{e}({k}*({x}+{x0}))'
            e_kx0 = f'{e}({k}*{x0})'
            e_kx = f'{e}({k}*x)'
            return f'( ({k}**2)*{L}*{e_kxx0} * ({e_kx0}-{e_kx}) ) / ( ({e_kx0} + {e_kx})**3 )'
        raise NotImplementedError("Sigmoid derivatives beyond 2 not implemented.")

    def __eq__(self, other):
        res = (isinstance(other, self.__class__) and
               self.L == other.L and
               self.k == other.k and
               self.x0 == other.x0 and
               self.b == other.b and
               self.derivative == other.derivative and
               self.symbol == other.symbol and
               self.e_symbol == other.e_symbol)
        return res

    def __ne__(self, other):
        return not self.__eq__(other)

    def roots(self):
        if self.derivative == 0:
            r = self._compute_root(self.L, self.x0, self.k, self.b)
            return np.array([r]) if r is not None else np.array([])
        elif self.derivative == 1:
            return np.array([])  # sigmoid has no extrema
        elif self.derivative == 2:
            return np.array([self.x0])  # only inflection point of sigmoid is x0
        raise NotImplementedError("Sigmoid derivatives beyond 2 not implemented.")

    def deriv(self, m=1):
        if self.derivative + m <= 2:
            return Sigmoid(self.L, self.x0, self.k, self.b, self.derivative+m)
        else:
            raise NotImplementedError("Sigmoid derivatives beyond 2 not implemented.")

    def copy(self):
        return self.__class__(self.L, self.x0, self.k, self.b, self.derivative, self.symbol, self.e_symbol)

    def linspace(self, start, stop, num=50):
        return self(np.linspace(start=start, stop=stop, num=num))

    @classmethod
    def fit(cls, x, y, p0=None, method='lm'):
        if p0 is None:
            p0 = [max(y), np.median(x), 1, min(y)]  # initial guess
        popt, _ = scipy.optimize.curve_fit(cls._sigmoid, x, y, p0, method=method)
        return cls(*popt)

    @staticmethod
    def _compute_root(L, x0, k, b):
        try:
            B = -math.log(-(b+L)/b)
            x = complex(x0+(B/k), 1)  # imaginary part is irrelevant but would be -(2*pi*n)/k for any integer n
            return x.real
        except (ZeroDivisionError, ValueError):  # ZeroDivision is obvious, ValueError is from log(0)
            return None  # no root

    @staticmethod
    def _positive_scaled_sigmoid(x, L):
        return L / (1 + np.exp(-x))

    @staticmethod
    def _negative_scaled_sigmoid(x, L):
        e_ = np.exp(x)
        return (L * e_) / (1 + e_)

    @staticmethod
    def _sigmoid(x, L, x0, k, b):
        x_ = k * (x - x0)

        if isinstance(x_, np.ndarray):
            positive_mask = x_ >= 0
            negative_mask = ~positive_mask

            y = np.empty_like(x_, dtype=np.float64)
            y[positive_mask] = Sigmoid._positive_scaled_sigmoid(x_[positive_mask], L)
            y[negative_mask] = Sigmoid._negative_scaled_sigmoid(x_[negative_mask], L)
        else:
            if x_ >= 0:
                y = Sigmoid._positive_scaled_sigmoid(x_, L)
            else:
                y = Sigmoid._negative_scaled_sigmoid(x_, L)
        y = y + b

        # naive implementation
        # y = L / (1 + np.exp(-k*(x-x0))) + b
        return y

    @staticmethod
    def _derivative_1(x, L, x0, k, *args):
        e_ = np.exp(-k*(x-x0))
        y = (k*L*e_)/((1+e_)**2)
        return y

    @staticmethod
    def _derivative_2(x, L, x0, k, *args):
        e_kxx0 = np.exp(k*(x+x0))
        e_kx0 = np.exp(k*x0)
        e_kx = np.exp(k*x)
        y = ((k**2)*L*e_kxx0*(e_kx0-e_kx))/((e_kx0+e_kx)**3)
        return y
