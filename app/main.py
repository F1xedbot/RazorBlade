import streamlit as st
import torch
from evm_cfg_builder.cfg import CFG
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from GraphNN import GCN

model = Word2Vec.load("../w2vec/word2vec_opcode_16.bin")

badrandomness_check = GCN(hidden_channels=64)
reentrancy_check = GCN(hidden_channels=64)

badrandomness_check.load_state_dict(torch.load('../models/badrandom_gcn_model.pth'))
reentrancy_check.load_state_dict(torch.load('../models/reentrancy_gcn_model.pth'))

badrandomness_check.eval()
reentrancy_check.eval()

def map_opcode_to_embedding(opcode_sequence):
    tokens = word_tokenize(opcode_sequence)
    opcode_sequence = [token.upper() for token in tokens if token.isalnum()]
    embedding_sequence = [model.wv[token] for token in opcode_sequence]
    average_embedding = np.mean(embedding_sequence, axis=0)
    max_embedding = np.max(embedding_sequence, axis=0)
    sum_embedding = np.sum(embedding_sequence, axis=0)
    final_embedding = np.concatenate([average_embedding, max_embedding, sum_embedding])
    norm = np.linalg.norm(final_embedding)
    if norm > 0:
        final_embedding = final_embedding / norm
    final_embedding = np.resize(final_embedding, (48))
    return final_embedding

def process_block(block):
    block_opcode = [info[0] for info in block['info']]
    block_embedding = map_opcode_to_embedding('\n'.join(block_opcode))
    return block_embedding

def process_bytecode(bytecode):
    cfg = CFG(bytecode)
    sorted_data_mapping = {
        str(block): {
            'pos': block.start.pc,
            'info': [(instr.mnemonic, instr.description) for instr in block.instructions],
            'out': [str(out_block) for out_block in block.all_outgoing_basic_blocks],
        }
        for block in cfg.basic_blocks
    }
    sorted_keys = sorted(sorted_data_mapping.keys(), key=lambda key: sorted_data_mapping[key]['pos'])

    graph_embedding = [process_block(sorted_data_mapping[key]) for key in sorted_keys]

    x = torch.tensor(np.vstack(graph_embedding), dtype=torch.float)
    position_mapping = {block: idx for idx, block in enumerate(sorted_keys)}

    edge_index = torch.tensor([
        [position_mapping[key] for key in sorted_keys for _ in sorted_data_mapping[key]['out']],
        [position_mapping[out_block] for key in sorted_keys for out_block in sorted_data_mapping[key]['out']]
    ], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)

def scan_bytecode(bytecode):
    # Placeholder function for bytecode scanning
    graph = process_bytecode(bytecode)
    input_data = [graph]

    badrandom_preds = []
    reentrancy_preds = []

    data_loader = DataLoader(input_data, batch_size=1, shuffle=False)
    for data in data_loader:
        with torch.no_grad():
            badrandom_preds = badrandomness_check(data.x, data.edge_index, data.batch)
            reentrancy_preds = reentrancy_check(data.x, data.edge_index, data.batch)

    final_predictions = []

    if badrandom_preds.argmax(dim=1):
        final_predictions.append("Bad Randomness")
    if reentrancy_preds.argmax(dim=1):
        final_predictions.append("Reentrancy")

    return final_predictions

def scan_file(file):
    # Placeholder function for source code file scanning
    return ["Command Injection"]

st.markdown('# RaZorBlade - A Multilabel Vulnerability Detection Framework')
option = st.radio("Select Input Type:", ("Raw Bytecode", "Source Code File"))

if option == "Raw Bytecode":
    bytecode = st.text_area("Paste Raw Bytecode here")
else:
    file = st.file_uploader("Upload Source Code File (e.g., .sol)", type="sol")

if st.button("Scan"):
    results = []

    if option == "Raw Bytecode":    
        results = scan_bytecode(bytecode)
    else:
        results = scan_file(file)

    st.write("### Vulnerabilities found:")
    
    # Display vulnerabilities in columns
    col1, col2 = st.columns(2)
    for i, label in enumerate(results, start=1):
        if i % 2 == 1:
            col1.write(f"{i}. {label}")
        else:
            col2.write(f"{i}. {label}")

    st.write(f"**Summary:** {len(results)} vulnerabilities found.")
