"""
Predict embedding of protein sequences using
large protein models
"""

import argparse
import time


if __name__ == "__main__":
    start_time = time.time()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-ff", "--fasta_file", required=True, help="fasta file containing protein sequences")

    # get argument values
    args = vars(arg_parser.parse_args())
    fasta_file = args["fasta_file"]

    end_time = time.time()
    print("Program finished in %s seconds" % str(end_time - start_time))
