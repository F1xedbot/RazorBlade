# RazorBlade: Advanced Smart Contract Vulnerability Detection

## Overview

Leveraging the semantic structure provided by Control Flow Graphs (CFG) generated from smart contract bytecode, RazorBlade offers a robust solution for identifying vulnerabilities.

## Model Workflow

1. **CFG Generation:**
   - Utilizes the evm-cfg-builder to generate the CFG from the smart contract bytecode.

2. **Graph Mapping:**
   - Maps the CFG into a graph, where edges represent connections, and nodes correspond to blocks within the CFG.

3. **Content Embedding:**
   - Embeds the content of each node using word2vec.

4. **Graph Convolutional Network (GCN):**
   - Applies a multi-headed Graph Convolutional Network (GCN) for comprehensive training and evaluation.

## Impressive Results

RazorBlade has undergone rigorous testing on prominent datasets:
- Smartbugs dataset on GitHub
- Slither-audited-smart-contracts on Hugging Face

The model has consistently achieved remarkable results in vulnerability detection.

## Repository Contents

Within this repository, you'll discover:
- Multiple notebooks for data processing and model building.
- Generated datasets for each vulnerability, facilitating seamless testing and evaluation.
