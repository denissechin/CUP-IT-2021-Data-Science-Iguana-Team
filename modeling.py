import torch
import torch.nn as nn
import transformers
import copy


class SemBERT(torch.nn.Module):
    
    def __init__(self):
        super(SemBERT, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        
        # Инициализируем собственную конфигурацию и модель DistilBERT'a для обработки синтаксических тэгов
        
        configuration = transformers.DistilBertConfig(dim=128, hidden_dim=256, n_heads=16)
        self.tagbert = transformers.DistilBertModel(configuration)
        self.tagbert_pooler = nn.Sequential(
                                            nn.Linear(in_features=128, out_features=128, bias=True),
                                            nn.Tanh())
        
        self.fc = nn.Sequential(nn.Dropout(p=0.1, inplace=False),
                                nn.Linear(in_features=768+128, out_features=3, bias=True))
        
        # Инициализируем сразу с замороженными слоями BERT'а, чтобы не забыть
        
        for param in self.bert.parameters():
            param.requires_grad = False

        
    def forward(self, bert_ids, bert_mask, tag_ids, tag_mask):
        
        bert_output = self.bert(input_ids=bert_ids, attention_mask=bert_mask).pooler_output
        
        # Пулингом извлекается только скрытое состояние, соответствующее первому токену последовательности [CLS]
        
        tagbert_output = self.tagbert(input_ids=tag_ids, attention_mask=tag_mask).last_hidden_state[:, 0]
        tagbert_output = self.tagbert_pooler(tagbert_output)
        
        pooled_output = torch.cat([bert_output, tagbert_output], dim=1)
        classified_output = self.fc(pooled_output)
        return classified_output
        

def train_model(model, training_dataloader, optimizer, lr_scheduler, loss_fct, num_epochs=1, load_best_model=True):
    
    print('Starting training...')
    
    best_score = 0
    frozen_flag = True
    
    for epoch in range(num_epochs):
        
        i = 0
        print(f'Epoch № {epoch+1}')
        
        for inputs, labels in training_dataloader:
            
            # Не забываем поставить модель в режим обучения
            model.train()
            
            # Если модель прошла четверть первой эпохи, разморозим слои BERT'a и снимем флаг
            if frozen_flag and i > (len(dataloader) // 4):
                frozen_flag = False
                for param in model.bert.parameters():
                    param.requires_grad = True
                
            optimizer.zero_grad()
            
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            tag_ids = inputs['tag_input_ids'][:, 0].to(device)
            tag_attention_mask = inputs['tag_attention_mask'][:, 0].to(device)

            labels = labels.to(device)
            outputs = model(bert_ids=input_ids, bert_mask=attention_mask, tag_ids=tag_ids, tag_mask=tag_attention_mask)
            
            loss = loss_fct(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            i += 1
            
            # Каждые 0.2 эпохи выводим лосс и считаем accuracy на валидационном датасете, если он лучший - сохраняем состояние модели
            
            if i % (len(dataloader) // 5) == 0:
                print('Loss: ', loss.item())
                print('Eval metrics on mismatched:')
                eval_score = predict_and_print_metrics(model, mis_dataset, print_metrics_and_return_score=True)
                if eval_score > best_score:
                    best_model = copy.deepcopy(model)
                    best_score = eval_score
                
    if load_best_model:
        model = copy.deepcopy(best_model)
            