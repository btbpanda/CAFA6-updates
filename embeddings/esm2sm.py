import argparse
import os
import re

import numpy as np
import tqdm
import pyarrow.feather as feather

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str)
parser.add_argument('-t', '--fasta', type=str)
parser.add_argument('-e', '--embed-path', type=str)
parser.add_argument('-d', '--device', type=str)


def get_embeddings(model, tokenizer, seq):
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]

    ids = tokenizer(sequence_examples, add_special_tokens=True, padding=True, truncation=True, max_length=1024)

    input_ids = torch.tensor(ids['input_ids']).to(DEVICE)
    attention_mask = torch.tensor(ids['attention_mask']).to(DEVICE)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,
                               attention_mask=attention_mask)

    # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7])
    emb_0 = embedding_repr.last_hidden_state[0]
    emb_0_per_protein = emb_0.mean(dim=0)

    return emb_0_per_protein


if __name__ == '__main__':

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    import torch
    from transformers import EsmModel, EsmTokenizer

    MODEL_NAME = 'facebook/esm2_t33_650M_UR50D'
    DEVICE = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    DEVICE_IDS = [0, ]

    fasta_path = Path(args.fasta)
    embed_path = Path(args.embed_path)

    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME, add_cross_attention=False, is_decoder=False).to(DEVICE)
    model.eval()

    output_path = embed_path / 'esm2S1280'
    output_path.mkdir(parents=True, exist_ok=True)

    fn = fasta_path / (args.file + '_seq.feather')
    read_df = feather.read_feather(fn)
    num_sequences = read_df.shape[0]

    ids = []
    embeds = np.zeros((num_sequences, 1280))
    for i in tqdm.tqdm(range(num_sequences)):
        seq_id, seq = read_df['EntryID'].values[i], read_df['seq'].values[i]
        ids.append(seq_id)
        embeds[i] = get_embeddings(model, tokenizer, str(seq)).detach().cpu().numpy()

    np.save(output_path / (args.file + '_embed.npy'), embeds)
    np.save(output_path / (args.file + '_ids.npy'), np.array(ids))