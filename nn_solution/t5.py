import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3,4'
import re

import numpy as np
import torch
import tqdm
import yaml
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
import pyarrow.feather as feather


device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
device_ids=[0,1]

def get_embeddings(model, tokenizer, seq):
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]

    ids = tokenizer.batch_encode_plus(sequence_examples, return_tensors="pt", padding=True, truncation=True, max_length=1000)

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,
                               attention_mask=attention_mask)

    # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7])
    emb_0 = embedding_repr.last_hidden_state[0]
    emb_0_per_protein = emb_0.mean(dim=0)

    torch.cuda.empty_cache()

    return emb_0_per_protein


if __name__ == '__main__':
    config = {
        'base_path': './',
        'embeds_path': './embeds'
    }    

    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    model = T5EncoderModel.from_pretrained(
        'Rostlab/prot_t5_xl_half_uniref50-enc',
        device_map="auto",
        torch_dtype=torch.float16,  
        trust_remote_code=True
    )
    model.eval()

    kaggle_dataset = config['base_path']  # sys.argv[1]
    output_path = os.path.join(config['base_path'], config['embeds_path'], 't5')  # sys.argv[2]
    os.makedirs(output_path, exist_ok=True)

    fn = os.path.join(kaggle_dataset, 'helpers/fasta/old_train_seq.feather')
    read_df = feather.read_feather(fn)
    num_sequences = read_df.shape[0]

    ids = []
    embeds = np.zeros((num_sequences, 1024))
    for i in tqdm.tqdm(range(num_sequences)):
        seq_id, seq = read_df['EntryID'].values[i], read_df['seq'].values[i]
        ids.append(seq_id)
        embeds[i] = get_embeddings(model, tokenizer, str(seq)).detach().cpu().numpy()

    np.save(os.path.join(output_path, 'old_train_embeds.npy'), embeds)
    np.save(os.path.join(output_path, 'old_train_ids.npy'), np.array(ids))
