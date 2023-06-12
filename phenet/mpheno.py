import numpy as np
import copy
import random
import sys


class Endo:
    def __init__(self, name, mean=None, var=None):
        self.name = name
        self.mean = mean
        self.var = var


class Trait:
    def __init__(self, name, beta=None, var=None, effect=None, se=None, dichotomous=False, baseline=None):
        self.name = name
        self.beta = beta
        self.var = var
        self.effect = effect
        self.se = se
        self.dichotomous = dichotomous
        self.baseline = baseline


class Model:
    """
    Construct a model from a list of endo phenotypes and observed traits

      >>> A = Model(input, verbose=0)

      :Parameters:
        - endos : list
            A list of endo phenotype objects to model

        - traits: list
            A list of traits to model
    """

    global NONE
    NONE = 0
    global INFO
    INFO = 1
    global DEBUG
    DEBUG = 2
    global TRACE
    TRACE = 3

    def log(self, message, level=INFO):
        if level <= self.debug_level:
            sys.stderr.write("%s\n" % message)

    def bail(message):
        sys.stderr.write("%s\n" % (message))
        sys.exit(1)

    def obs_name(self, trait):
        return "%s_obs" % trait

    def __init__(self, endos, traits, debug_level=0, use_pymc3=False):
        self.use_pymc3 = use_pymc3
        self.node_names = None
        self.debug_level = debug_level

        default_mu = 0
        default_tau = 0.1
        default_beta = 0

        if self.use_pymc3:
            import pymc3
            self.pymc = pymc3
            import scipy
            self.scipy = scipy

            self.endo_names = [x.name for x in endos]
            self.trait_bames = [x.name for x in traits]

            self.M = self.pymc.Model()
            self.trace = None
            with self.M:
                endo_nodes = {}
                for endo in endos:
                    cur_mu = default_mu
                    cur_sd = np.sqrt(1.0 / default_tau)
                    if endo.mean is not None:
                        cur_mu = endo.mean
                    if endo.var is not None:
                        cur_sd = np.sqrt(endo.var)
                    endo_nodes[endo.name] = self.pymc.Normal(endo.name, mu=cur_mu, sd=cur_sd)

                trait_nodes = {}
                mult_endo_nodes = {}
                obs_nodes = {}
                for trait in traits:
                    cur_beta = default_beta
                    cur_sd = np.sqrt(1.0 / default_tau)
                    if trait.beta is not None:
                        cur_beta = trait.beta
                    if trait.var is not None:
                        cur_sd = np.sqrt(trait.var)

                    mult_endo_nodes[trait.name] = 0
                    for endo in endos:
                        # FIXME: when we add multiple endophenotypes need to handle this
                        mult_endo_nodes[trait.name] += endo_nodes[endo.name] * cur_beta

                    if trait.dichotomous or not trait.se == 0:
                        trait_nodes[trait.name] = self.pymc.Normal(trait.name, mu=mult_endo_nodes[trait.name],
                                                                   sd=cur_sd)
                        if trait.dichotomous:
                            # get the intercept
                            b_o = 0
                            if trait.baseline is not None:
                                b_o = np.log(trait.baseline / (1 - trait.baseline))
                            p = 1 / (1 + np.exp(-b_o - trait_nodes[trait.name]))
                            obs_nodes[self.obs_name(trait.name)] = self.pymc.Bernoulli(self.obs_name(trait.name), p=p,
                                                                                       observed=trait.effect)
                        elif trait.effect is not None and trait.se is not None:
                            obs_value = trait.effect
                            if trait.baseline is not None:
                                obs_value = trait.effect - trait.baseline
                            obs_nodes[self.obs_name(trait.name)] = self.pymc.Normal(self.obs_name(trait.name),
                                                                                    mu=trait_nodes[trait.name],
                                                                                    tau=1 / (trait.se) ** 2,
                                                                                    observed=obs_value)
                    else:
                        trait_nodes[trait.name] = self.pymc.Normal(trait.name, mu=mult_endo_nodes[trait.name],
                                                                   sd=cur_sd, observed=trait.effect)

            self.map_value = None
        else:
            import pymc
            self.pymc = pymc

            def make_model(endos=endos, traits=traits):
                endo_nodes = {}
                for endo in endos:
                    cur_mu = default_mu
                    cur_tau = default_tau
                    if endo.mean is not None:
                        cur_mu = endo.mean
                    if endo.var is not None and endo.var > 0:
                        cur_tau = 1 / endo.var
                    endo_nodes[endo.name] = self.pymc.Normal(endo.name, mu=cur_mu, tau=cur_tau)

                trait_nodes = {}
                obs_nodes = {}
                for trait in traits:
                    cur_beta = default_beta
                    cur_tau = default_tau
                    if trait.beta is not None:
                        cur_beta = trait.beta
                    if trait.var is not None:
                        cur_tau = 1 / trait.var

                    @self.pymc.deterministic
                    def mult_endo(endo_nodes=endo_nodes, cur_beta=cur_beta):
                        for endo in endos:
                            # FIXME: when we add multiple endophenotypes need to handle this
                            return endo_nodes[endo.name] * cur_beta

                    if not trait.se == 0:
                        trait_nodes[trait.name] = self.pymc.Normal(trait.name, mu=mult_endo, tau=cur_tau)

                        if trait.dichotomous:
                            # get the intercept
                            b_o = 0
                            if trait.baseline is not None:
                                b_o = np.log(trait.baseline / (1 - trait.baseline))
                            p = 1 / (1 + np.exp(-b_o - trait_nodes[trait.name]))
                            obs_nodes[self.obs_name(trait.name)] = self.pymc.Bernoulli(self.obs_name(trait.name), p=p,
                                                                                       value=trait.effect,
                                                                                       observed=True)
                        elif trait.effect is not None and trait.se is not None:
                            obs_value = trait.effect
                            if trait.baseline is not None:
                                obs_value = trait.effect - trait.baseline
                            obs_nodes[self.obs_name(trait.name)] = self.pymc.Normal(self.obs_name(trait.name),
                                                                                    mu=trait_nodes[trait.name],
                                                                                    tau=1 / (trait.se) ** 2,
                                                                                    value=obs_value, observed=True)
                    else:
                        trait_nodes[trait.name] = self.pymc.Normal(trait.name, mu=mult_endo, tau=cur_tau,
                                                                   value=trait.effect, observed=True)
                    return locals()

            self.make_model = make_model
            self.MCMC = self.pymc.MCMC(self.make_model())
            self.MAP = self.pymc.MAP(self.make_model())

    def draw(self, output_file):
        if self.use_pymc3:
            with self.M:
                graph = self.pymc.model_to_graphviz(self.M)
                graph.render(output_file)
            # bail("To draw model, must specify to not use pymc3")
        else:
            graph = self.pymc.graph.graph(self.MCMC)
            graph.write_png(output_file)

    def summary(self, var_names):
        summary_values = NONE
        if self.use_pymc3:
            with self.M:
                summary_values = self.pymc.summary(self.trace, var_names=var_names)
        return summary_values

    def save_trace(self, output_file):
        if self.use_pymc3:
            with self.M:
                self.pymc.save_trace(self.trace, output_file)

    def sample(self, iter=1000, burn=100, thin=10):
        if self.use_pymc3:
            with self.M:
                self.trace = self.pymc.sample(iter, step=self.pymc.NUTS(), progressbar=False)
        else:
            self.compute_map_estimate()
            self.MCMC.sample(iter=iter + burn, burn=burn, thin=thin, progress_bar=False)

    def plot_trace(self, output_file):
        # import matplotlib.pyplot as plt
        import arviz as az
        if self.use_pymc3:
            with self.M:
                axes = az.plot_trace(self.trace)
                fig = axes.ravel()[0].figure
                fig.savefig(output_file)

    def get_trace(self, name):
        if self.use_pymc3:
            if self.trace is None:
                return []
            return self.trace[name]
        else:
            if name not in self.get_node_names():
                return []
            return self.MCMC.trace(name)[:]

    def get_node_names(self):
        if self.node_names is None:
            if self.use_pymc3:
                self.node_names = set([x.name for x in self.M.basic_RVs])
            else:
                self.node_names = set(
                    self.MAP.endo_nodes.keys() + self.MAP.trait_nodes.keys() + self.MAP.obs_nodes.keys())
        return self.node_names

    def get_map_node_value(self, name):
        if self.use_pymc3:
            if name in self.map_value:
                return self.map_value[name]
            else:
                return None
        else:
            node_dict = None
            if name in self.MAP.endo_nodes:
                node_dict = self.MAP.endo_nodes
            elif name in self.MAP.trait_nodes:
                node_dict = self.MAP.trait_nodes
            elif name in self.MAP.obs_nodes:
                node_dict = self.MAP.obs_nodes
            else:
                return None
            return node_dict[name].value

    def get_normal_like(self, value, mean, var):
        tau = 1 / var
        mu = mean
        if self.use_pymc3:
            return (-tau * (value - mu) ** 2 + np.log(tau / np.pi / 2.)) / 2.
        else:
            return self.pymc.distributions.normal_like(value, mu, tau)

    def compute_map_estimate(self, max_ests=100, times_seen_break=10, opt_fun="fmin_powell"):
        max_logp = -np.Inf
        max_values = {}
        times_seen = 0
        best_map_value = None
        for i in range(max_ests):
            cur_logp = None
            if self.use_pymc3:
                with self.M:
                    start = None
                    if best_map_value is not None:
                        start = copy.copy(best_map_value)
                        for k in start:
                            start[k] = random.random()
                    best_map_value = self.pymc.find_MAP(start=start)
                    cur_logp = self.M.logp(best_map_value)
            else:
                import cStringIO
                class Capturing(list):
                    def __enter__(self):
                        self._stdout = sys.stdout
                        sys.stdout = self._stringio = cStringIO.StringIO()
                        return self

                    def __exit__(self, *args):
                        self.extend(self._stringio.getvalue().splitlines())
                        sys.stdout = self._stdout

                self.MAP = self.pymc.MAP(self.make_model())
                # M = pymc.NormApprox(self.make_model())
                with Capturing() as output:
                    self.MAP.fit(iterlim=100000, tol=0.00001, method=opt_fun, verbose=True)
                for endo in self.MCMC.endo_nodes:
                    self.log("Iteration %s: %s (%s)" % (i + 1, self.MAP.endo_nodes[endo].value, self.MAP.logp), DEBUG)
                for out in output:
                    toss = False
                    self.log(out, TRACE)
                    if "Warning: Maximum" in out:
                        self.log("Tossing iteration %s" % (i + 1), DEBUG)
                        toss = True
                        break
                if toss:
                    continue

                cur_logp = self.MAP.logp

            if cur_logp > max_logp:
                max_logp = cur_logp
                times_seen = 0

                if self.use_pymc3:
                    self.log("New maximum: %s; saw previous %s times" % (best_map_value, times_seen), DEBUG)
                    self.map_value = best_map_value
                else:
                    self.log("New maximum: %s; saw previous %s times" % (self.MAP.endo_nodes[endo].value, times_seen),
                             DEBUG)
                    for node in self.MAP.endo_nodes:
                        self.MCMC.endo_nodes[node].value = float(self.MAP.endo_nodes[node].value)
                    for node in self.MAP.trait_nodes:
                        self.MCMC.trait_nodes[node].value = float(self.MAP.trait_nodes[node].value)
            elif max_logp - cur_logp < 1e-6:
                times_seen += 1
                self.log("Repeated maximum: %s; seen %s times" % (best_map_value, times_seen), TRACE)
                if times_seen >= times_seen_break:
                    if self.use_pymc3:
                        for endo in self.endo_names:
                            self.log(
                                "Early termination at iteration %s for %s: %s" % (i + 1, endo, best_map_value[endo]),
                                DEBUG)
                    else:
                        for endo in self.MAP.endo_nodes:
                            self.log("Early termination at iteration %s for %s: %s" % (
                            i + 1, endo, self.MAP.endo_nodes[endo].value), DEBUG)
                    break
