from __future__ import annotations

from typing import List, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms  # ADD THIS
from PIL import Image  # ADD THIS

class PathVQADataset(Dataset):
    """
    PyTorch Dataset for PathVQA (CNN Baseline) with LAZY LOADING.
    Images are loaded on-demand to save memory.
    """
    def __init__(
        self,
        metadata: List[Dict],
        hf_dataset,  # Reference to HuggingFace dataset for lazy image loading
        question_vocab: BaselineVocabulary,
        answer_vocab: BaselineVocabulary,
        text_preprocessor: TextPreprocessor,
        transform: transforms.Compose,
        max_question_len: int = 20,
        max_answer_len: int = 10
    ):
        self.metadata = metadata
        self.hf_dataset = hf_dataset  # Keep reference for lazy loading
        self.question_vocab = question_vocab
        self.answer_vocab = answer_vocab
        self.text_preprocessor = text_preprocessor
        self.transform = transform
        self.max_question_len = max_question_len
        self.max_answer_len = max_answer_len

        # Build answer to index mapping for classification
        self.answer2idx = {}
        self.idx2answer = {}
        for item in metadata:
            ans = text_preprocessor.preprocess(item['answer'])
            if ans not in self.answer2idx:
                idx = len(self.answer2idx)
                self.answer2idx[ans] = idx
                self.idx2answer[idx] = ans

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        hf_idx = item['idx']  # Index into HuggingFace dataset

        # LAZY LOAD image from HuggingFace dataset
        image = self.hf_dataset[hf_idx]['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)

        # Process question
        question = self.text_preprocessor.preprocess(item['question'])
        question_tokens = self.question_vocab.encode(question, self.max_question_len)
        question_tensor = torch.tensor(question_tokens, dtype=torch.long)

        # Process answer
        answer = self.text_preprocessor.preprocess(item['answer'])
        answer_tokens = self.answer_vocab.encode(answer, self.max_answer_len)
        answer_tensor = torch.tensor(answer_tokens, dtype=torch.long)

        # Answer class index for classification
        answer_idx = self.answer2idx.get(answer, 0)

        # Question type (0: closed, 1: open)
        question_type = 0 if item['question_type'] == 'closed' else 1

        return {
            'image': image,
            'question': question_tensor,
            'answer': answer_tensor,
            'answer_idx': answer_idx,
            'question_type': question_type,
            'answer_text': answer
        }