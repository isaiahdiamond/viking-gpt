class Value:
    '''This class stores a single scalar value and tis gradient'''
    
    def __init__(self,data, _childern=(), _op=''):
        self.data = data
        self._prev = set(_childern)
        self._op = _op
        self.e = 2.718281828459045
        self.grad = 0.0
        self._backward = lambda : None
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):   # lets plain sum() work: 0 + Value
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):   # 2 * Value
        return self * other
    
    def __neg__(self): 
        return self * -1.0

    def __sub__(self, other): 
        return self + (-other)
    
    def __truediv__(self,other):
        out = Value(self.data / other.data, (self,other), '/')
        return out 
    
    def __pow__(self, other):    # other is a plain int/float
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (self.e ** (2 * x) - 1) / (self.e ** (2 * x) + 1)
        out = Value(t, (self,),'tanh')
        
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out 

    def backward(self):
        topo = []
        visted = set()
        
        def build_topo(v):
            if v not in visted:
                visted.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

