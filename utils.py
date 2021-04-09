import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast



def get_tags(string):
    return ' '.join(re.findall('\(([A-Z]+)', string))


def compute_metrics(pred, labels):
    target_names = ['entailment', 'contradiction', 'neutral']
    preds = pred
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    class_report = classification_report(labels, preds, target_names=target_names, output_dict=True)
    acc = accuracy_score(labels, preds)
    contradiction_recall = class_report['contradiction']['recall']
    return {
        'accuracy': acc,
        'macro_f1': f1,
        'macro_precision': precision,
        'macro_recall': recall,
        'contradiction_recall' : contradiction_recall
    }


def predict_and_print_metrics(model, dataset, print_metrics_and_return_score=False):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=200)
    outputs = torch.Tensor()
    # Не забываем поставить модель в режим оценки
    model.eval()
    for inputs, labels in dataloader:
        with torch.no_grad():
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            tag_ids = inputs['tag_input_ids'][:, 0].to(device)
            tag_attention_mask = inputs['tag_attention_mask'][:, 0].to(device)
            output = model(bert_ids=input_ids, bert_mask=attention_mask, tag_ids=tag_ids, tag_mask=tag_attention_mask)

            output = output.argmax(axis=1).cpu()
            outputs = torch.cat([outputs, output], dim=0)
                
    if print_metrics_and_return_score:
        print(compute_metrics(outputs, dataset.labels), sep='\n')
        return compute_metrics(outputs, dataset.labels)['accuracy']


# Типичный датасет в pytorch
class MultiNLI_dataset(Dataset):
    
    def __init__(self, tokenized_pairs, labels, sentences1, sentences2, max_length):
        # Будем токенизировать тэги во время обучения, чтобы сохранить память
        self.tokenizer = DistilBertTokenizerFast(vocab_file='./vocab.txt', do_lower_case=False)
        self.input_ids = tokenized_pairs['input_ids']
        self.attention_mask = tokenized_pairs['attention_mask']
        self.labels = torch.Tensor(labels).view(-1,).long()
        self.parsed_tags = [(sent1, sent2) for sent1, sent2 in zip(sentences1, sentences2)]
        self.max_length = max_length
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Для удобства пользования датасет будет возвращать словарь
        output_dict = {}
        output_dict['input_ids'] = self.input_ids[idx]
        output_dict['attention_mask'] = self.attention_mask[idx]
        tokenized_tags = self.tokenizer(self.parsed_tags[idx][0], text_pair=self.parsed_tags[idx][1], padding='max_length', max_length=self.max_length, return_tensors='pt')
        output_dict['tag_input_ids'] = tokenized_tags['input_ids']
        output_dict['tag_attention_mask'] = tokenized_tags['attention_mask']
        return output_dict, self.labels[idx]


def preprocess_tokenize_dataset(tokenizer, dataset):
    df = dataset.copy()
    df = df[df['gold_label'] != '-']
    df['gold_label'].replace(to_replace=label_set, inplace=True)
    df['sentence1_tagged'] = df['sentence1_parse'].apply(get_tags)
    df['sentence2_tagged'] = df['sentence2_parse'].apply(get_tags)
    tags_sentence_1 = list(df['sentence1_tagged'].values)
    tags_sentence_2 = list(df['sentence2_tagged'].values)
    sentences_1 = list(df['sentence1'].str.lower().replace(re.compile(punctuation), ' ').values)
    sentences_2 = list(df['sentence2'].str.lower().replace(re.compile(punctuation), ' ').values)
    labels = df['gold_label'].values
    tokenized_pairs = tokenizer(text=sentences_mis1, text_pair=sentences_mis2, padding=True, return_tensors='pt')
    max_len = 0
    text = df['sentence1_tagged'].values.astype(str)
    text_pairs = df['sentence2_tagged'].values.astype(str)

    for seq1, seq2 in zip(text, text_pairs):
        input_ids = tag_tokenizer(seq1, text_pair=seq2, add_special_tokens=True)['input_ids']
        if max_len < len(input_ids):
            max_len = max(max_len, len(input_ids))

    return tokenized_pairs, labels, tags_sentence_1, tags_sentence_2, max_len
