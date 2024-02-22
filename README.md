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
  
![workflow image](https://github.com/F1xedbot/RazorBlade/blob/main/RazorBlade_workflow.png?raw=true)

## Impressive Results

RazorBlade has undergone rigorous testing on prominent datasets:
- [Smartbugs](https://github.com/smartbugs/smartbugs-curated) on GitHub
- [Slither-audited-smart-contracts](https://huggingface.co/datasets/mwritescode/slither-audited-smart-contracts) on Hugging Face

The model has consistently achieved remarkable results in vulnerability detection.

Overall results on both datasets

![overall results](https://github.com/F1xedbot/RazorBlade/blob/5402db0b35af1e729f8f99afb45f647b4bacd45a/huggingface-razorblade.png?raw=true)

## Repository Contents

Within this repository, you'll discover:
- Multiple notebooks for data processing and model building.
- Google Drive folder links for generated plain text datasets and vectorized datasets for each vulnerability, for testing and evaluation.

## Datasets

- PlainTextDatasets: [dataset_plain_text](https://drive.google.com/drive/folders/1RcoA56eKNumz03aj0Gt314P9VwwIjAWV?usp=sharing)
- VectorizedDatasets: [dataset_vetor](https://drive.google.com/drive/folders/1VyUy1pcUrOUejJV_obk6MOeDTq-4-xH_?usp=sharing)

## Demo on RazorBladev1

[Watch the Demo](https://drive.google.com/file/d/129RvMGbPNjYWHoo8Go2HZa4UsMkfrVBx/view?usp=sharing)
