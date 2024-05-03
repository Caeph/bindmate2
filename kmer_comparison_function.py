import os

import numpy as np
import tensorflow as tf
from itertools import product
from collections import Counter
import pyProBound
from tqdm import tqdm

import rpy2.robjects as robjects

robjects.r('library(BiocManager)')

from background_normalizer import *

script_dir = os.path.split(os.path.realpath(__file__))[0]


class KmerMetric:
    def __init__(self,
                 name, annot, metric_type,
                 probability_model, additional_info, unique_kmers
                 ):
        self.name = name
        self.annotation = annot
        self.metric_type = metric_type  # distance or similarity

        # probability_functions and stuff for optimization
        self.probability_model = probability_model

        # background
        if "background_info" in additional_info:
            self.normalizer = Normalizer(additional_info["background_info"], additional_info['k'])
        elif "background_object" in additional_info:
            self.normalizer = additional_info["background_object"]
        else:
            self.normalizer = Normalizer(None, additional_info['k'])
            # dummy object that does nothing
        self.unique_kmers = unique_kmers

    def compare_kmers(self, combinations):
        pass

    def get_type(self):
        return self.metric_type

    def get_model_for_optimization(self):
        return self.probability_model


# UNIFIED INITIALIZER: probabilitiy model, background_info, additional_info
class GCcontent(KmerMetric):
    def __init__(self, probability_model, additional_info, unique_kmers):
        super().__init__(name="gc",
                         metric_type="distance",
                         annot="absolute difference is GC content",
                         probability_model=probability_model,
                         additional_info=additional_info,
                         unique_kmers=unique_kmers
                         )
        self.gc = None

    def compare_kmers(self, combinations):
        unique_kmers_tensor = tf.constant(self.unique_kmers)

        def gc_content_diff(kmers_tensor):
            chars_tensor = tf.strings.bytes_split(kmers_tensor)
            # Calculate GC content
            gc_content = tf.cast(
                tf.math.reduce_sum(tf.where((chars_tensor == b'G') | (chars_tensor == b'C'), 1, 0), axis=1), tf.float32)

            hor, ver = tf.meshgrid(gc_content, gc_content, indexing='ij')
            diff2d = tf.math.abs(hor - ver)
            diff1d = tf.reshape(diff2d, [-1])
            return diff1d

        diff = gc_content_diff(unique_kmers_tensor)

        diff = self.normalizer.normalize(gc_content_diff, diff)
        return diff


class PairContent(KmerMetric):
    def __init__(self, probability_model, additional_info, unique_kmers):
        super().__init__(name="pair-gc",
                         metric_type="distance",
                         annot="absolute difference in nucleotide pair content",
                         probability_model=probability_model,
                         additional_info=additional_info,
                         unique_kmers=unique_kmers
                         )
        self.pairs = None
        self.info = ["".join(pair) for pair in product(list('ACGT'), list('ACGT'))]

    def __characterize(self, seq):
        seq_tensor = tf.strings.unicode_split(seq, 'UTF-8')

        # Generate all possible dinucleotide pairs and count occurrences
        def count_pairs(s):
            pairs = tf.strings.join([s[:-1], s[1:]], separator='')
            # Count occurrences of each pair in `self.info`
            counts = [tf.reduce_sum(tf.cast(tf.equal(pairs, pair), tf.int32)) for pair in self.info]
            return tf.stack(counts)

        res = count_pairs(seq_tensor)
        return res

    def compare_kmers(self, combinations):
        kmers_tensor = tf.constant(self.unique_kmers)

        def nucl_pair_diff(unique_kmers_tensor):
            pairs = tf.map_fn(self.__characterize, unique_kmers_tensor, dtype=tf.int32)
            full_difference = tf.zeros((len(pairs), len(pairs)), dtype=tf.float64)
            for i in range(pairs.shape[1]):
                a = pairs[:, i]
                b = pairs[:, i]
                a_rep, b_rep = tf.meshgrid(a, b, indexing='ij')
                diff = tf.cast(a_rep - b_rep, tf.float64)
                full_difference += diff * diff

            diff = tf.reshape(full_difference, [-1])
            return diff

        mse = nucl_pair_diff(kmers_tensor)
        mse = self.normalizer.normalize(nucl_pair_diff, mse)
        return mse


class ProBoundAffinity(KmerMetric):
    def __init__(self, probability_model, additional_info, unique_kmers):
        super().__init__(name="probound",
                         metric_type="distance",
                         annot="difference in affinity to TF selection as modelled by ProBound",
                         probability_model=probability_model,
                         additional_info=additional_info,
                         unique_kmers=unique_kmers
                         )
        # load stuff
        taxa = additional_info["taxa"]
        mc = pyProBound.MotifCentral()
        if taxa is not None:
            mc = mc.filter(taxa=[taxa])
        mc = mc[mc["gene_symbols"].str.len() == 1]
        models, names = mc['model_id'].values.astype(int), mc["gene_symbols"].str[0]

        # if needed -- filter out
        if "selected_motifs" in additional_info:
            mask = names.isin(additional_info["selected_motifs"])
            models, names = models[mask], names[mask]
        self.models = models
        self.names = names

    def compare_kmers(self, combinations):
        unique_kmers_probound = [str(x) for x in self.unique_kmers]
        if self.normalizer.valid:
            background_kmers_probound = [str(x) for x in self.normalizer.bg_sequences.numpy().astype(str)]
        else:
            background_kmers_probound = None

        full_difference = tf.zeros((len(unique_kmers_probound), len(unique_kmers_probound)), dtype=tf.float64)
        for model_id, name in tqdm(list(zip(self.models, self.names))):
            model_file = os.path.join(script_dir, "additional_knowledge", "probound_models", f"{model_id}.json")
            model = pyProBound.ProBoundModel(model_file, fitjson=True)
            model.select_binding_mode(0)  # in the motif central models there is usually only one
            affinities = np.array(model.score_affinity_sum(unique_kmers_probound))

            if background_kmers_probound is not None:
                background_affinities = model.score_affinity_sum(background_kmers_probound)
                loc = np.mean(background_affinities)
                scale = np.std(background_affinities)
                # normalize
                affinities = (affinities - loc) / scale

            a_rep, b_rep = tf.meshgrid(affinities, affinities, indexing='ij')
            diff = tf.cast(a_rep - b_rep, tf.float64)
            full_difference += diff * diff

        diff = tf.reshape(full_difference, [-1])
        return diff


class ShapeDifference(KmerMetric):
    def __init__(self, probability_model, additional_info, unique_kmers):
        super().__init__(name="shape",
                         metric_type="distance",
                         annot="MSE of chosen shape feature",
                         probability_model=probability_model,
                         additional_info=additional_info,
                         unique_kmers=unique_kmers
                         )
        self.shape_parameter = additional_info["shape_feature"]
        aggregating_functions = {
            "mean": np.mean,
            "max": np.max,
            "min": np.min
        }
        if "shape_aggregator" not in additional_info:
            self.shape_aggregator = aggregating_functions["mean"]
        elif additional_info["shape_aggregator"] in aggregating_functions:
            self.shape_aggregator = aggregating_functions[additional_info["shape_aggregator"]]
        else:
            self.shape_aggregator = additional_info["shape_aggregator"]

    def compare_kmers(self, combinations):
        robjects.r('library(DNAshapeR)')

        def calculate_shape(kmers):
            temp_fasta_name = "tmp.fasta"
            with open(temp_fasta_name, mode='w') as temp_fasta:
                # put stuff to tempfile
                for i, item in enumerate(kmers):
                    print(f">{i}\n{item}", file=temp_fasta)

            # Call the getShape function
            result = robjects.r('getShape')(temp_fasta_name)

            def strip_nan(array):
                good = np.where(~np.isnan(array[0, :]))[0]
                return array[:, good]

            pyresult = {name: np.array(val) for name, val in zip(result.names, list(result))}
            shape_values = strip_nan(pyresult[self.shape_parameter])
            os.remove(temp_fasta_name)
            for name in pyresult.keys():
                os.remove(temp_fasta_name + f".{name}")

            # aggregate values on one kmer
            shape_values = np.apply_along_axis(self.shape_aggregator, 1, shape_values)

            return shape_values
        shape_values = calculate_shape(self.unique_kmers)
        if self.normalizer.valid:
            bg_shape_values = calculate_shape(self.normalizer.bg_sequences.numpy())
            loc = np.mean(bg_shape_values)
            scale = np.std(bg_shape_values)
            shape_values = (shape_values - loc) / scale

        shape_values = tf.constant(shape_values)
        hor, ver = tf.meshgrid(shape_values, shape_values, indexing='ij')
        diff2d = tf.math.abs(hor - ver)
        diff2d = diff2d * diff2d

        diff1d = tf.reshape(diff2d, [-1])
        return diff1d
