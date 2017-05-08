from math import inf
from numbers import Real


"""
TODO Efficiency!
"""


EPSILON = 1e-7


class lpm:
    """
    A: list of assertions
    """
    
    def __init__(self):
        self.obj_fn, self.to_min = None, True
        self.A = []
    
    def to_str(self):
        V = set()
        
        # Objective
        S = ['min. ' if self.to_min else 'max. ', str(self.obj_fn), '\n']
        V.update(self.obj_fn.VC)
        
        # Constraints
        if self.A:
            S.extend(['s.t. ', str(self.A[0]), '\n'])
            V.update(self.A[0].VC)
            for a in self.A[1:]:
                S.extend(['     ', str(a), '\n'])
                V.update(a.VC)
        
        for i in sorted(V):
            S.extend([str(variable._ALL[i]), ' '])
        
        return ''.join(S)
    
    def var(self, names=None):
        """
        Names are just decoration. Different variable instances with the same
          name are actually different variables. (TODO Correct approach?)
        
        USAGE
          x = var()
          y = var('y')
          z, w = var('z w')
          u, v = var(['u', 'v'])
          
          # Default bound is [0, inf)
          x.B(-1, 2)
          z.LB(inf)
          w.UB(3)
          
          t = var('t').LB(-4)
        """
        if names is None:
            return variable()
        
        if isinstance(names, tuple) or isinstance(names, list):
            return [variable(n) for n in names]
        
        N = names.split()
        return variable(N[0]) if len(N) == 1 else [variable(n) for n in N]
    
    def min(self, e):
        self.obj_fn, self.to_min = e, True
    
    def max(self, e):
        self.obj_fn, self.to_min = e, False
    
    def st(self, A):
        if isinstance(A, list) or isinstance(A, tuple):
            self.A.extend(A)
        else:
            self.A.append(A)


class variable:
    """
    _id: identifier (should not be altered)
    
    NOTE The reason why using ALL and _id is that this class needs to override
      __eq__() (for supporting something like 'x == 0') and it is not hashable,
      which means it cannot be used as key of VC of expression and assertion.
      In addition, creating order can be kept.
    """
    
    _ALL = []
    
    def __init__(self, name='_'):
        # Keep self in _ALL and set _id.
        self._id = len(variable._ALL)
        variable._ALL.append(self)
        
        self.name, self.lb, self.ub = name, 0, inf
    
    def __repr__(self):
        return f'{self.name}({self.lb},{self.ub})'
    
    def B(self, lb, ub):
        self.lb, self.ub = lb, ub
        return self
    
    def LB(self, lb):
        self.lb = lb
        return self
    
    def UB(self, ub):
        self.ub = ub
        return self
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        return expression(0, [self, -1])
    
    def __mul__(self, other):
        assert isinstance(other, Real)
        return expression(0, [self, other])
    
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        assert isinstance(other, Real)
        return expression(0, [self, 1 / other])
    
    def __add__(self, other):
        return expression(0, [self, 1]) + other
    
    __radd__ = __add__
    
    def __sub__(self, other):
        return expression(0, [self, 1]) - other
    
    def __rsub__(self, other):
        return other - expression(0, [self, 1])
    
    def __le__(self, other):
        return expression(0, [self, 1]) <= other
    
    def __ge__(self, other):
        return expression(0, [self, 1]) >= other
    
    def __eq__(self, other):
        return expression(0, [self, 1]) == other
    
    def __ne__(self, other):
        assert False


class expression:
    """
    VC: dict of {variable id --> coefficient}
    ct: constant term
    
    TODO Efficiency!
    """
    
    def __init__(self, ct, vc=None):
        self.VC = {vc[0]._id: vc[1]} if vc and abs(vc[1]) > EPSILON else {}
        self.ct = ct
    
    def __repr__(self):
        return ' '.join(_format(self.VC, self.ct))
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        # TODO Using update() is better than below?
        self.VC.update((vid, c * -1) for vid, c in self.VC.items())
        self.ct *= -1
        return self
    
    def __mul__(self, other):
        assert isinstance(other, Real)
        
        # TODO Better way?? (others too)
        for vid in list(self.VC):
            self.VC[vid] *= other
            if abs(self.VC[vid]) <= EPSILON:
                del self.VC[vid]  # remove if coeff == 0
        
        self.ct *= other
        
        return self
    
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        return self.__mul__(1 / other)
    
    def add(self, other, sense=1):
        if isinstance(other, Real):
            self.ct += sense * other
        
        elif isinstance(other, variable):
            if other._id in self.VC:
                self.VC[other._id] += sense
                if abs(self.VC[other._id]) <= EPSILON:
                    del self.VC[other._id]  # remove if coeff == 0
            else:
                self.VC[other._id] = sense
        
        elif isinstance(other, expression):
            for ov, oc in other.VC.items():
                if ov in self.VC:
                    self.VC[ov] += sense * oc
                    if abs(self.VC[ov]) <= EPSILON:
                        del self.VC[ov]  # remove if coeff == 0
                else:
                    self.VC[ov] = sense * oc
        
            self.ct += sense * other.ct
        
        else:
            assert False
        
        return self
    
    def __add__(self, other):
        return self.add(other)
    
    __radd__ = __add__
    
    def __sub__(self, other):
        return self.add(other, -1)
    
    def __rsub__(self, other):
        return (-self).add(other)
    
    def relate(self, other, reverse, le):
        e = expression(self.ct)
        e.VC = dict(self.VC)
        if reverse:
            e = -e
        e = e.add(other, -1)
        return assertion(e.VC, le, -e.ct)
    
    def __le__(self, other):
        return self.relate(other, False, True)
    
    def __ge__(self, other):
        return self.relate(other, True, True)
    
    def __eq__(self, other):
        return self.relate(other, False, False)
    
    def __ne__(self, other):
        assert False


class assertion:
    """
    VC
    le: whether less than or equal relation
    rhs: right-hand side
    """
    
    def __init__(self, VC, le, rhs):
        self.VC, self.le, self.rhs = VC, le, rhs
    
    def __repr__(self):
        S = _format(self.VC, 0)
        S.append('\u2264' if self.le else '=')
        S.append(str(self.rhs))
        return ' '.join(S)


def _format(VC, ct):
    _V, S, first = variable._ALL, [], True
    
    for vid, c in VC.items():
        if first:
            first = False
            if c == -1:
                S.append('-' + _V[vid].name)
            else:
                if c != 1:
                    S.append(str(c))
                S.append(_V[vid].name)
        else:
            S.append('-' if c < 0 else '+')
            if c != 1:
                S.append(str(-c if c < 0 else c))
            S.append(_V[vid].name)
    
    if first:
        S.append(str(ct))
    elif ct != 0:
        S.append('- ' + str(-ct) if ct < 0 else '+ ' + str(ct))
    
    return S


def test():
    m = lpm()
    
    x = m.var('x').LB(-inf)
    y = m.var('y').LB(-3)
    
    m.min(
        -x + 4 * y
    )
    
    m.st([
        -3 * x + y <= 6,
        x + 2 * y <= 4,
    ])
    
    print(m.to_str())


if __name__ == '__main__':
    test()
