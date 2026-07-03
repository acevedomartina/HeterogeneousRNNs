# Using these classes, we can define more nicely what we want to loop through with instances
# ------------------------------------------------------------
from itertools import product

class SimParam:
    '''Class for simulation parameters'''
    def __init__(self, sim, vrest, gain):
        self.sim = sim
        self.vrest = vrest
        self.gain = gain

    def __repr__(self):
        '''Special dunder that returns a string representation of object'''
        return f"SimParam(sim={self.sim}, vrest={self.vrest}, gain={self.gain})"


class Config:
    '''Class for configurations of pqif and seed numbers on the different simulation parameters defined by SimParam.
    
    Handles optional parameters. For example, if we also want to vary omega and sg, we can give those as key value pairs.'''
    def __init__(self, name, sims, pqif_numbers, seed_numbers, optional_params=None):
        self.name = name
        self.sims = sims
        self.pqif_numbers = pqif_numbers
        self.seed_numbers = seed_numbers

        # {} If nothing was passed
        self.optional_params = optional_params or {}

    def iter_params(self):
        '''Creates a product of the iteration variables'''

        optional_names = list(self.optional_params.keys())
        optional_values = list(self.optional_params.values())

        for sim_param in self.sims:
            for pqif in self.pqif_numbers:
                for seed in self.seed_numbers:
                    if optional_values:
                        for values in product(*optional_values):
                            # * for unpacking, treat them as two independent loops
                            extras = dict(zip(optional_names, values))
                            yield sim_param, pqif, seed, extras
                    else:
                        yield sim_param, pqif, seed, {}

    def __repr__(self):
        '''Special dunder that returns a string representation of object'''
        # sims_str = "\n".join(f"  {sim}" for sim in self.sims)
        # sim_numbers = [sim.sim for sim in self.sims]
        sims_str = ", ".join(
        f"{s.sim} : (vrest={s.vrest}, gain={s.gain})"
        for s in self.sims
    )

        return f"Config name: {self.name}\n  pqif: {self.pqif_numbers}\n  simulations: {sims_str}\n  across {len(self.seed_numbers)} seeds\n  Optional parameters:\n{[(key, value) for (key, value) in self.optional_params.items()]}"
    
    @property
    def gains(self):
        return [s.gain for s in self.sims]
    
    @property
    def vrests(self):
        return [s.vrest for s in self.sims]
    
    @property
    def sim_numbers(self):
        return [s.sim for s in self.sims]
    
