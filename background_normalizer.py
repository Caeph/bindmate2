import pandas as pd
from pyfastaq.sequences import file_reader as fasta_reader
import numpy as np
import tensorflow as tf
import swifter


def sample_background(infodict):
    size = infodict['size']
    k = infodict['k']
    source_file = infodict['source']  # expected fasta

    def sample_random_kmer(sequence):
        l = len(sequence) - k
        start = np.random.randint(l)
        output = sequence[start:start + k]
        return output

    source_df = pd.DataFrame(
        [entry.seq for entry in fasta_reader(source_file)], columns=["seq"]
    )
    source_df['kmer'] = source_df['seq'].swifter.progress_bar(False).apply(sample_random_kmer).str.upper()
    undefined_seq = "".join(['N' for _ in range(k)])
    source_df = source_df[source_df['kmer'] != undefined_seq]

    if len(source_df) > size:
        source_df = source_df.sample(size)

    return tf.constant(source_df['kmer'].values)


def generate_random_background(infodict):
    size = infodict['size']
    length = infodict['k']
    bases = list("ACGT")
    seqs = tf.constant(["".join(np.random.choice(bases, size=length)) for _ in range(size)])
    return seqs


class Normalizer:
    def __init__(self, background_info_dict, k):
        bg_generator = {
            "sampled": sample_background,
            "random": generate_random_background
        }

        if background_info_dict is None:
            self.valid = False
        else:
            self.valid = True

            background_info_dict['k'] = k
            bgfunc = bg_generator[background_info_dict["type"]]
            self.bg_sequences = bgfunc(background_info_dict)

    def normalize(self, func_to_apply, values_to_normalize):
        if self.valid:
            applied = func_to_apply(self.bg_sequences)
            loc = tf.reduce_mean(applied)
            scale = tf.math.reduce_std(applied)
            return (values_to_normalize - loc) / scale
        else:
            return values_to_normalize