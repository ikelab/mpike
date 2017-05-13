from math import inf
from numbers import Real


"""
Solvers:
  scipy: scipy.optimize.linprog (Python implementation)
  glpk: cvxopt.solvers.lp with glpk (C implementation)

TODO Efficiency!!!
"""


EPSILON = 1e-7


class lpm:
    """
    obj_fn
    sense: 1, if minimizing; -1, if maximizing
    A: list of assertions
    """
    
    def __init__(self):
        self.obj_fn, self.sense = None, True
        self.A = []
    
    def to_str(self):
        # Objective
        S = ['min. ' if self.sense < 0 else 'max. ', str(self.obj_fn), '\n']
        V = set(self.obj_fn.VC)
        
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
        self.obj_fn = expression(0, [e, 1]) if isinstance(e, variable) else e
        self.sense = 1
    
    def max(self, e):
        self.obj_fn = expression(0, [e, 1]) if isinstance(e, variable) else e
        self.sense = -1
    
    def st(self, A):
        if isinstance(A, list) or isinstance(A, tuple):
            self.A.extend(A)
        else:
            self.A.append(A)
    
    def solve(self, solver='glpk'):
        """
        RETURN
            None: infeasible
        """
        # Collect variables and do numbering.
        V = set(self.obj_fn.VC)
        for a in self.A:
            V.update(a.VC)
        V = {vid: i for i, vid in enumerate(V)}
        
        n = len(V)
        
        c = [0] * n
        for vid, coeff in self.obj_fn.VC.items():
            c[V[vid]] = coeff * self.sense
        
        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        for a in self.A:
            Ai = [0] * n
            for vid, coeff in a.VC.items():
                Ai[V[vid]] = coeff
            
            if a.le:
                A_ub.append(Ai)
                b_ub.append(a.rhs)
            else:
                A_eq.append(Ai)
                b_eq.append(a.rhs)
        
        if solver == 'glpk':
            z0 = solve_glpk(V, c, A_ub, b_ub, A_eq, b_eq)
        elif solver == 'scipy':
            z0 = solve_scipy(V, c, A_ub, b_ub, A_eq, b_eq)
        else:
            assert False
        
        return z0 * self.sense


def solve_glpk(V, c, A_ub, b_ub, A_eq, b_eq):
    """
    A_ub should not be empty.
    """
    from cvxopt import glpk, matrix, solvers  # @UnresolvedImport
    
    # TODO This does not seem to work. C.f. cvxopt/src/C/glpk.c:220.
    glpk.options['msg_lev'] = 'GLP_MSG_OFF'
    
    n, VALL = len(c), variable._ALL
    for vid, i in V.items():
        v = VALL[vid]
        v.sv = None
        
        if v.lb > -inf:
            Ai = [0] * n; Ai[i] = -1
            A_ub.append(Ai)
            b_ub.append(-v.lb)
        if v.ub < inf:
            Ai = [0] * n; Ai[i] = 1
            A_ub.append(Ai)
            b_ub.append(v.ub)
    
    Ab = (matrix(A_eq, tc='d').T, matrix(b_eq, tc='d')) if A_eq else (None, None)
    R = solvers.lp(matrix(c, tc='d'), matrix(A_ub, tc='d').T, matrix(b_ub, tc='d'),
                   *Ab, solver='glpk')
    
    if R['status'] == 'optimal':
        sol = R['x']
        for vid, i in V.items():
            VALL[vid].sv = sol[i]
        return R['primal objective']
    
    elif R['status'] == 'dual infeasible':  # primal unbounded
        return -inf
    
    assert R['status'] == 'primal infeasible'
    return None


def solve_scipy(V, c, A_ub, b_ub, A_eq, b_eq):
    from scipy.optimize._linprog import linprog
    
    bounds, VALL = [None] * len(V), variable._ALL
    
    for vid, i in V.items():
        v = VALL[vid]
        v.sv = None
        
        bounds[i] = (v.lb, v.ub)
    
    if not A_ub:
        A_ub, b_ub = None, None
    if not A_eq:
        A_eq, b_eq = None, None
    
    R = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
    
    if R.status == 0:  # Optimization terminated successfully.
        sol = R.x
        for vid, i in V.items():
            VALL[vid].sv = sol[i]
        return R.fun
    
    elif R.status == 3:  # Problem appears to be unbounded.
        return -inf
    
    elif R.status == 1:  # Iteration limit reached.
        assert False
    
    assert R.status == 2  # Problem appears to be infeasible.
    return None


class variable:
    """
    name
    lb, ub: lower and upper bound
    sv: solution value
    
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
        
        self.name, self.lb, self.ub, self.sv = name, 0, inf, None
    
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
    
    def relate(self, other, reverse, is_LE):
        """
        NOTE If is_LC is False, it is equality constraint.
        """
        e = expression(self.ct)
        e.VC = dict(self.VC)
        e = e.add(other, -1)
        if reverse:
            e = -e
        return assertion(e.VC, is_LE, -e.ct)
    
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
    le: whether less than (True) or equal relation (False)
    rhs: right-hand side
    """
    
    def __init__(self, VC, le, rhs):
        """
        Self takes ownership of VC.
        """
        self.VC, self.le, self.rhs = VC, le, rhs
    
    def __repr__(self):
        S = _format(self.VC, 0)
        S.append('\u2264' if self.le else '=')
        S.append(str(self.rhs))
        return ' '.join(S)


def _format(VC, ct):
    VALL, S, first = variable._ALL, [], True
    
    for vid, c in VC.items():
        if first:
            first = False
            if c == -1:
                S.append('-' + VALL[vid].name)
            else:
                if c != 1:
                    S.append(str(c))
                S.append(VALL[vid].name)
        else:
            S.append('-' if c < 0 else '+')
            if c != 1:
                S.append(str(-c if c < 0 else c))
            S.append(VALL[vid].name)
    
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
    
    c = -3
    
    m.st([
        c * x + y <= 6,
        x + 2 * y <= 4
    ])
    
    print(m.to_str())
    
    m = lpm()
    
    x = m.var('x').LB(-inf)
    y = m.var('y').LB(-3)
    
    m.min(
        -x + 4 * y
    )
    
    c = 5
    
    m.st([
        c * x + y <= 6,
        x + 2 * y <= 4
    ])
    
    print(m.to_str())
    
    z0 = m.solve()
    #z0 = m.solve('scipy')
    
    print('z* =', z0)
    print(f'(x*, y*) = ({x.sv}, {y.sv})')


if __name__ == '__main__':
    test()
