import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction

from .findroot import Fun, ObjectiveFun
from .searchspace import Point, SearchSpace


class Bayesian:
    def __init__(
        self,
        f: Fun,
        domain: SearchSpace,
        objective: ObjectiveFun,
        initial: Point = None,
        seed=None,
        utility=None,
    ) -> None:
        self.f = f
        self.objective = objective
        if isinstance(seed, np.random.SeedSequence):
            seed = int(seed.generate_state(1)[0])
        self.utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0) if not utility else utility

        assert all(isinstance(b, tuple) for b in domain.bounds.values())
        pbounds = {dim: (lb, ub) for dim, (lb, ub, width) in domain.bounds.items()}
        self.opt = BayesianOptimization(f=None, pbounds=pbounds, verbose=2, random_state=seed)
        # f=lambda p: -objective(f(p))

        self.point = self.opt.suggest(self.utility) if not initial else initial
        self.result = self.f(self.point)

    def __iter__(self):
        while True:
            self.old_result = tuple(self.result)
            self.old_point = self.point
            val = self.objective(self.old_result)

            try:
                self.opt.register(self.old_point, -val)  # Negate for minimization instead of max
            except KeyError as e:
                print(e, flush=True)

            self.point = self.opt.suggest(self.utility)

            self.result = self.f(self.point)
            yield val, self.old_point

    def state(self):
        return self.old_result, self.old_point

    def best(self):
        return self.opt.max['params']
