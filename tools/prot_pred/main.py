"""
Predict embedding of protein sequences using
large protein models
"""

import argparse
import time

import torch
from transformers import T5Tokenizer, T5Model, T5EncoderModel
from Bio import SeqIO
import h5py


def process_sequences(seq_file):
    fasta_sequences = SeqIO.parse(open(input_file),'fasta')
    seqs_fasta = list()
    with open(output_file) as out_file:
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            seqs_fasta.append(sequence)
    return seqs_fasta
    

def encode_seqs(seqs_fasta, out_file):
    
    print("mapping to rare amino acids...")
    sequences = [re.sub(r"[UZOJB]", "X", sequence) for sequence in seqs_fasta]
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    print("tokenization...")
    ids = tokenizer.batch_encode_plus(
        sequences, add_special_tokens=True, padding=True
    )
    input_ids = torch.tensor(ids["input_ids"]).to(device)
    attention_mask = torch.tensor(ids["attention_mask"]).to(device)
	
    print("computing embedding...")

    agg_embedding = list()
    s_time = time.time()
	
    batch_size = 1

    for i in range(int(len(sequences) / batch_size)):
        curr = i * batch_size
        nex = i * batch_size + batch_size
        with torch.no_grad():
            embedding = model(input_ids=input_ids[curr:nex,], attention_mask=attention_mask[curr:nex,])
            embedding = embedding.last_hidden_state.cpu().numpy()
            agg_embedding.extend(embedding)
    e_time = time.time()
    print("embedding computed")
	
    #hf = h5py.File('data/save_agg_embedding_{}.h5'.format(str(len(sequences))), 'w')
    hf = h5py.File(out_file, 'w')
    hf.create_dataset('embedding', data=agg_embedding)
    hf.close()


if __name__ == "__main__":
    start_time = time.time()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-ff", "--fasta_file", required=True, help="fasta file containing protein sequences")
    arg_parser.add_argument("-oe", "--output_embed", required=True, help="embedding of protein sequences")

    # get argument values
    args = vars(arg_parser.parse_args())
    fasta_file = args["fasta_file"]
    out_file = args["output_embed"]
    
    print(torch.__version__)
    print(fasta_file, out_file)

    fasta_seqs = process_sequences(fasta_file)

    encode_seqs(fasta_seqs, out_file)

    end_time = time.time()
    print("Program finished in %s seconds" % str(end_time - start_time))
