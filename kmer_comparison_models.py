import tensorflow as tf
import kmer_comparison_distributions as kcd


class KmerMetricModel:  # abstract specification
    def __init__(self, no_models,
                 unmatched_allowed_distributions,
                 matched_allowed_distributions,
                 distributions_info):
        self.probability_dens_func_factory =  {
            "uniform": kcd.UniformDistribution,
            "reverse-exponential": kcd.ReversedExponentialDistribution,
            "gaussian-mixture": kcd.GaussianMixtureDistribution,
            "exponential": kcd.ExponentialDistribution
        }

        # distributions_info : dictionary z : info
        self.no_models = no_models
        self.unmatched_allowed_distributions = unmatched_allowed_distributions  # names
        self.matched_allowed_distributions = matched_allowed_distributions
        self.distributions_info = distributions_info

    def get_allowed_distributions(self, z):
        if z == 0:
            return self.unmatched_allowed_distributions
        return self.matched_allowed_distributions

    def fill_params(self, z, m_observed_values):
        if z == 0:
            distribution = self.unmatched_allowed_distributions[0]
        else:
            distribution = self.matched_allowed_distributions[0]
        params_func = self.probability_dens_func_factory[distribution].estimate_initial_parameters
        params = params_func(m_observed_values, self.distributions_info)
        return [distribution, params]

    def fill_params_distribution(self, distribution, m_observed_values):
        params_func = self.probability_dens_func_factory[distribution].estimate_initial_parameters
        params = params_func(m_observed_values, self.distributions_info)
        return params

    def get_validifying_function(self, distribution):
        func = self.probability_dens_func_factory[distribution].validify_params
        return func

    def calculate_metric_probability(self, z, m_params, m_observed_values):
        return None


class StubMetricModel(KmerMetricModel):
    def __init__(self, no_models, unmatched_allowed_distributions,
                 matched_allowed_distributions, distributions_info):
        super().__init__(no_models,
                         unmatched_allowed_distributions,
                         matched_allowed_distributions,
                         distributions_info)

    def calculate_metric_probability(self, z, m_params, m_observed_values):
        if m_params is None:
            m_z_params = self.fill_params(z, m_observed_values)
        else:
            m_z_params = m_params[z]  # first parameter here is the name
        # m_params is a list of z sets of params

        distr_type = m_z_params[0]
        params = m_z_params[1]
        distribution = self.probability_dens_func_factory[distr_type]
        return tf.cast(distribution.calculate_probability(m_observed_values, params),
                       tf.float64)
