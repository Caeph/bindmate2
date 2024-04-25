import numpy as np
import time
import kmer_comparison_function as kcf
import kmer_comparison_models as kcm

# TODO more metrics
metric_factory = {
    "gc": kcf.GCcontent,
    "pair": kcf.PairContent
}

# todo make access point
# structure of additional_info: dict(model=instance of KmerMetricModel)
default_no_of_models = 2


def create_metric_instance(no_matched_models, metric_name, background_info, additional_info):
    if metric_name not in metric_factory:
        raise NotImplementedError(f"Metric of name {metric_name} is not known.")
    if "model" not in additional_info:
        probability_model = kcm.StubMetricModel(no_matched_models + 1,
                                                unmatched_allowed_distributions=["uniform",
                                                                                 "reverse-exponential"
                                                                                 ],
                                                matched_allowed_distributions=["exponential",
                                                                               "gaussian-mixture"
                                                                               ],
                                                distributions_info=dict(gmm_models_no=default_no_of_models))
    else:
        probability_model = additional_info["model"]
    metric = metric_factory[metric_name](probability_model, background_info, additional_info)
    return metric


def initialize_functions(k, no_matched_models, unique_kmers, metrics, background_info=None,
                         additional_info_on_metrics=None):
    """
    Initializes all necessities for the functions used to compare metrics.
    :param k: k-mer size
    :param unique_kmers: seen unique_kmers
    :param metrics: metrics the user wishes to initialize
    :param background_info: dictionary with background info
            (keyword: value)
    :param additional_info_on_metrics: dictionary with any additional info on the metrics
            (metricname: dict with values)
    :return:
    """
    if background_info is not None:
        background_info["k"] = k
    full_metrics = []
    for m in metrics:
        start = time.time()
        additional_info = dict()  # TODO define and get info
        full = create_metric_instance(no_matched_models, m, background_info, additional_info)
        full.initialize(unique_kmers)
        print(f"Metric {full.name} initialized: {np.round(time.time() - start, decimals=3)}")

        full_metrics.append(full)
    return full_metrics
