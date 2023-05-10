import os
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup



def train(args, model, train_inputs, train_labels, validation_inputs, validation_labels,
            batch_size_train, batch_size_validation,
            output_dir,
            save_boundary_accuracy,
            learning_rate=1e-5,
            warmup_steps=50,
            num_training_steps=2000,
            num_epochs=20,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            eval_period=50,
            device='mps'):

    # batch size = 16, step = 200 : 3200 training examples
    # 1 epoch = 4210 steps (4210 * 16 training examples)

    optimizer, scheduler = get_optimizer_and_scheduler(model, learning_rate=learning_rate, warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    train_dataloader = get_dataloader(train_inputs, train_labels, batch_size_train, is_training=True)

    model.to(device)
    model.train()


    global_step = 0
    train_losses = []

    early_stopping = 0
    best_accuracy = -1

    stop_training = False

    print("Start training")
    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader):
            global_step += 1

            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

            if torch.innan(loss).data:
                print("Stop training because loss=%s" % (loss.data))
                stop_training = True
                break

            train_losses.append(loss.detach().cpu())

            loss.backward()


            if global_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                model.zero_grad()

                if scheduler is not None :
                    scheduler.step()
            
            if global_step % eval_period == 0:
                acc = test(args=args, model=model, inputs=validation_inputs, labels=validation_labels,
                            batch_size=batch_size_validation, step=global_step, device=device, version="train")
                
                print("\n[ Validation ] step: %d\tAccuracy: %.1f%%" % (global_step, acc))

                if acc > save_boundary_accuracy:
                    model_state_dict = {k: v.cpu() for (k,v) in model.state_dict().items()}

                    torch.save(model_state_dict, os.path.join(output_dir, "model-{}.pt".format(global_step)))
                    print("Saving model at global_step: %d\t(train loss: %.2f)\t(learning_rate: %f)\t(time: {?})" 
                            % (global_step, np.mean(train_losses), learning_rate))

                    train_losses = []

                    if best_accuracy == -1:
                        best_accuracy = acc
                    elif best_accuracy <= acc:
                        best_accuracy = acc
                        early_stopping = 0
                    else :
                        early_stopping += 1
                    
                    if early_stopping == 10:
                        break

            if global_step == num_training_steps:
                break
        if global_step == num_training_steps:
            break
    print("Finish training")



def test(args, model, inputs, labels, batch_size, step, device="mps", version="test"):
    if version == "train":
        save_dir = "save_accuracy.csv"
        if not os.path.exists(save_dir):
            df = pd.DataFrame(
                    columns=['model_name', 'learning_rate', 'batch_size_train', 'step', 'accuracy']
                )
        else :
            df = pd.read_csv(save_dir, index_col='Unnamed: 0')

    model.to(device)
    model.eval()

    dataloader = get_dataloader(inputs, None, batch_size, is_training=False)
    all_logits = []

    for batch in tqdm(dataloader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        logits = logits.detach().cpu()
        all_logits.append(logits)

    all_logits = torch.cat(all_logits, axis=0)
    predictions = torch.argmax(all_logits, axis=1)
    labels = torch.LongTensor(labels)

    acc = torch.sum(predictions==labels) / labels.shape[0]
    acc = acc * 100


    if version == "train":
        filt = (df['model_name'] == args.model) & \
                (df['learning_rate'] == args.learning_rate) & \
                (df['batch_size_train'] == args.batch_size_train) & \
                (df['step'] == step)
        if df[filt].empty:
            df.loc[len(df)] = [args.model, args.learning_rate, args.batch_size_train, step, acc.item()]
        else :
            df.loc[filt, 'accuracy'] = acc.item()
        
        df.to_csv(save_dir)
    
    return acc
    # np.mean(np.array(predictions) == np.array(labels))



def get_dataloader(inputs, labels, batch_size, is_training):
    if labels is not None:
        labels = torch.LongTensor(labels)
        dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], labels)
    else :
        dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])

    if is_training:
        sampler = RandomSampler(dataset)
    else :
        sampler = SequentialSampler(dataset)
    
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader



def get_optimizer_and_scheduler(model, learning_rate=1e-5, warmup_proportion=0.01, warmup_steps=50, 
                                weight_decay=0.0, adam_epsilon=1e-8, num_training_steps=1000):
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any (nd in n for nd in no_decay)],
        'weight_decay': weight_decay},
        {'params': [p for n,p in model.named_parameters() if any (nd in n for nd in no_decay)],
        'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    return optimizer, scheduler