import os
import time
import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score



def train(model, train_inputs, train_labels, validation_inputs, validation_labels,
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
    
    model_name = '/'.join(output_dir.split('/')[1:-1])
    save_dir = output_dir.split('/')[0] + "_accuracy.csv"

    print(f"\n[ 저장되는 directory : {save_dir} ]")


    if not os.path.exists(save_dir):
        df = pd.DataFrame(
                columns=['model_name', 'learning_rate', 'batch_size_train', 'step', 
                        'train_loss', 'train_accuracy', 'f1_score_train', 'test_loss', 'test_accuracy', 'f1_score_test', 'time']
            )
    else :
        df = pd.read_csv(save_dir, index_col='Unnamed: 0')

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
    start = time.time()
    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader):
            global_step += 1

            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            # loss : when labels is provided, Classification loss.
            # logits : Classification scores before SoftMax.
            output_model = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output_model.loss


            # Check Train Accuracy
            logits = output_model.logits # shape = (16, 2) = (batch_size_train, 2)
            predictions = torch.argmax(logits, axis=1).cpu() # shape = (batch_size_train)
            labels = torch.LongTensor(labels.cpu()) # shape = (batch_size_train)
            train_accuracy = np.mean(np.array(predictions) == np.array(labels))
            f1_score_train = f1_score(labels, predictions, average='binary')


            if torch.isnan(loss).data:
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
                # from IPython import embed; embed()
                acc, test_loss, f1_score_test = test(model=model, inputs=validation_inputs, labels=validation_labels,
                            batch_size=batch_size_validation, device=device)
                test_loss = np.array(test_loss).item()
                
                end = time.time()
                sec = end - start
                time_result = str(datetime.timedelta(seconds=sec)).split('.')[0]

                print("\n[ Validation ] step: %d\tLoss: %.4f\tAccuracy: %.4f%%\tF1-Score: %.4f" % (global_step, test_loss, acc, f1_score_test))



                filt = (df['model_name'] == model_name) & \
                        (df['learning_rate'] == learning_rate) & \
                        (df['batch_size_train'] == batch_size_train) & \
                        (df['step'] == global_step)
                if df[filt].empty:
                    df.loc[len(df)] = [model_name, learning_rate, batch_size_train, global_step, 
                                        round(loss.item(), 4), round(train_accuracy, 4), round(f1_score_train, 4),
                                        round(test_loss, 4), round(acc.item(), 4), round(f1_score_test, 4), 
                                        time_result]
                else :
                    df.loc[filt, ['train_loss', 'train_accuracy', 'f1_score_train', 
                                'test_loss', 'test_accuracy', 'f1_score_test', 
                                'time']] = [round(loss.item(), 4), round(train_accuracy, 4), round(f1_score_train, 4), 
                                            round(test_loss, 4), round(acc.item(), 4), round(f1_score_test, 4), time_result]
                df.to_csv(save_dir)


                if acc > save_boundary_accuracy:
                    model_state_dict = {k: v.cpu() for (k,v) in model.state_dict().items()}

                    torch.save(model_state_dict, os.path.join(output_dir, "model-{}.pt".format(global_step)))
                    print("Saving model at global_step: %d\t(learning_rate: %f)\t(train loss: %.3f)\t(train accuracy: %.2f)\t(f1-score train: %.2f)\t(test loss: %.3f\t(test accuracy: %.2f)\t(f1-score test: %.2f)\t(time: {%s})" 
                            % (global_step, learning_rate, np.mean(train_losses), train_accuracy, f1_score_train, test_loss, acc, f1_score_test, time_result))

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
        if global_step == num_training_steps or stop_training:
            break
    print("Finish training")



def test(model, inputs, labels, batch_size, device="mps"):

    model.to(device)
    model.eval()

    dataloader = get_dataloader(inputs, None, batch_size, is_training=False)
    all_logits = []

    for batch in tqdm(dataloader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)

        # loss : when labels is provided, Classification loss.
        # logits : Classification scores before SoftMax.
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        logits = logits.detach().cpu()
        all_logits.append(logits)


    # all_logits : shape = (500, 4, 2) 
    # = (data_size / batch_size, batch_size, 2)

    all_logits = torch.cat(all_logits, axis=0) # shape = (2000, 2)
    predictions = torch.argmax(all_logits, axis=1) # shape = (2000)
    labels = torch.LongTensor(labels) # shape = (2000)

    acc = torch.sum(predictions==labels) / labels.shape[0]
    acc = acc * 100

    # F1-Score
    f1_score_test = f1_score(labels, predictions, average='binary')


    # [ loss 구하는 방법 ]
    num_categories = len(np.unique(labels))
    num_labels = len(labels)
    labels_one_hot = np.zeros((num_labels, num_categories)) # shape = (2000, 2)
    labels_one_hot[np.arange(num_labels), labels] = 1
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=all_logits))

    return acc, loss, f1_score_test
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