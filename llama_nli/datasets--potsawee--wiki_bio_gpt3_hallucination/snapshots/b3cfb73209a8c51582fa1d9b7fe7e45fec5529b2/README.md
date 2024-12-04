---
license: cc-by-sa-3.0
task_categories:
- text-classification
language:
- en
size_categories:
- n<1K
dataset_info:
  features:
  - name: gpt3_text
    dtype: string
  - name: wiki_bio_text
    dtype: string
  - name: gpt3_sentences
    sequence: string
  - name: annotation
    sequence: string
  - name: wiki_bio_test_idx
    dtype: int64
  - name: gpt3_text_samples
    sequence: string
  splits:
  - name: evaluation
    num_bytes: 5042581
    num_examples: 238
  download_size: 2561507
  dataset_size: 5042581
---
# Dataset Card for WikiBio GPT-3 Hallucination Dataset

- GitHub repository: https://github.com/potsawee/selfcheckgpt
- Paper: [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896)

### Dataset Summary

- We generate Wikipedia-like passages using GPT-3 (text-davinci-003) using the prompt: ```This is a Wikipedia passage about {concept}``` where `concept` represents an individual from the WikiBio dataset.
- We split the generated passages into sentences, and we annotate each sentence into one of the 3 options: (1) accurate (2) minor_inaccurate (3) major_inaccurate.
- We report the data statistics, annotation process, and inter-annotator agreement in our paper.
  
## Update
- v3 (5 May 2023): 238 test IDs have been annotated in total.
- v2 (6 April 2023): 142 test IDs have been annotated, GPT-3 sampled passages are now included in this dataset.
- v1 (15 March 2023): 65 test IDs -- here is `wiki_bio_test_idx` of the documents in v1 [[Link]](https://drive.google.com/file/d/1N3_ZQmr9yBbsOP2JCpgiea9oiNIu78Xw/view?usp=sharing)

## Dataset Structure

Each instance consists of:
- `gpt3_text`: GPT-3 generated passage
- `wiki_bio_text`: Actual Wikipedia passage (first paragraph)
- `gpt3_sentences`: `gpt3_text` split into sentences using `spacy`
- `annotation`: human annotation at the sentence level
- `wiki_bio_test_idx`: ID of the concept/individual from the original wikibio dataset (testset)
- `gpt3_text_samples`: list of 20 sampled passages (do_sample = True & temperature = 1.0)

### Citation Information
```
@misc{manakul2023selfcheckgpt,
      title={SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models}, 
      author={Potsawee Manakul and Adian Liusie and Mark J. F. Gales},
      year={2023},
      eprint={2303.08896},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```