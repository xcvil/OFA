import torch
from transformers import AutoTokenizer, BartModel
model = BartModel.from_pretrained("facebook/bart-base", local_files_only=True)