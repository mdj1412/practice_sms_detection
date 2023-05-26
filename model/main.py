import os
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from train import train, test


id2label = {0: "ham", 1: "spam"}
label2id = {"ham": 0, "spam": 1}


def string_to_index_label(string_label):
    """
        'ham' -> 0, 'spam' -> 1
    """
    return 0 if string_label=='ham' else 1


def tokenized_data(tokenizer, inputs, max_length=64):
    return tokenizer.batch_encode_plus(
        inputs,
        return_tensors="pt",
        padding="max_length",
        # max_length=max_length,
        max_length=128,
        truncation=True)

















def load_source_dataset():
    """
        Source Dataset ( huggingface dataset : SetFit/enron_spam )
        A dataset for pre-training
    """

    print("[ Load source dataset ( enron spam dataset ) ]\n")
    enron_spam_dataset = load_dataset('SetFit/enron_spam')
    print(enron_spam_dataset, '\n\n')

    train_source_dataset = pd.DataFrame({'text' : enron_spam_dataset['train']['text'], 
                                        'label' : enron_spam_dataset['train']['label']})
    test_source_dataset = pd.DataFrame({'text' : enron_spam_dataset['test']['text'], 
                                        'label' : enron_spam_dataset['test']['label']})

    # length of text
    train_source_dataset['length'] = train_source_dataset['text'].apply(len)
    test_source_dataset['length'] = test_source_dataset['text'].apply(len)

    print("=================================================\n\n")
    print(f"Num of train source dataset : {len(train_source_dataset)}\n")
    print(train_source_dataset['length'].describe(percentiles=[.25, .5, .75, .95, .99]), '\n\n')

    print(f"Num of ham (train source dataset) : {len(train_source_dataset.loc[train_source_dataset['label']==0])}\n")
    print(train_source_dataset.loc[train_source_dataset['label']==0, 'length'].describe(percentiles=[.25, .5, .75, .95, .99]), '\n\n')

    print(f"Num of spam (train source dataset) : {len(train_source_dataset.loc[train_source_dataset['label']==1])}\n")
    print(train_source_dataset.loc[train_source_dataset['label']==1, 'length'].describe(percentiles=[.25, .5, .75, .95, .99]), '\n\n')

    print("=================================================\n\n")

    print(f"Num of test source dataset : {len(test_source_dataset)}\n")
    print(test_source_dataset['length'].describe(percentiles=[.25, .5, .75, .95, .99]), '\n\n')

    print(f"Num of ham (test source dataset) : {len(test_source_dataset.loc[test_source_dataset['label']==0])}\n")
    print(test_source_dataset.loc[test_source_dataset['label']==0, 'length'].describe(percentiles=[.25, .5, .75, .95, .99]), '\n\n')

    print(f"Num of spam (test source dataset) : {len(test_source_dataset.loc[test_source_dataset['label']==1])}\n")
    print(test_source_dataset.loc[test_source_dataset['label']==1, 'length'].describe(percentiles=[.25, .5, .75, .95, .99]), '\n\n')

    print("=================================================\n\n")
    return train_source_dataset.loc[:, ['text', 'label']], test_source_dataset.loc[:, ['text', 'label']]




def load_target_dataset():
    """
        * huggingface dataset : 
        sms_spam, Ngadou/Spam_SMS

        * kaggle : 
        https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
    """

    print("\n\n\n\n[ Load dataset ]")

    spam_csv_file = 'kaggle_data/spam.csv'
    # Determine if it exists
    if not os.path.exists(spam_csv_file):
        raise NotImplementedError("Not exists spam.csv")
    else :
        spam_csv_df = pd.read_csv(spam_csv_file, encoding='latin1')

        print("\n\n================================== [ Kaggle Dataset ] =======================================\n\n")

        print("\n[ kaggle dataset analysis ]")
        print(spam_csv_df.info())

        print("\n[ v1 (label) print sample 5 ]")
        print(spam_csv_df['v1'].sample(5))
        print("\n[ v2 (sms) print sample 5 ]")
        print(spam_csv_df['v2'].sample(5))
        print("\n[ Unnamed: 2 (sms) print total ]")
        print(spam_csv_df['Unnamed: 2'].dropna())
        print("\n[ Unnamed: 3 (sms) print total ]")
        print(spam_csv_df['Unnamed: 3'].dropna())
        print("\n[ Unnamed: 4 (sms) print total ]")
        print(spam_csv_df['Unnamed: 4'].dropna())

        # ham, spam -> 0, 1
        spam_csv_df['label'] = spam_csv_df['v1'].apply(string_to_index_label)


        # v2 - 5572
        # Unnamed: 2 - 50
        # Unnamed: 3 - 12
        # Unnamed: 4 - 6
        # total sum - 5640
        kaggle_sms_df = pd.DataFrame({'sms': spam_csv_df['v2'].values, 'label': spam_csv_df['label'].values})
        
        # v2 말고 다른 column 에 있는 데이터들 합치기
        unnamed2_idx = spam_csv_df['Unnamed: 2'].dropna().index
        unnamed2_df = spam_csv_df.loc[unnamed2_idx, ['Unnamed: 2', 'label']].rename(columns={'Unnamed: 2' : 'sms'})
        unnamed3_idx = spam_csv_df['Unnamed: 3'].dropna().index
        unnamed3_df = spam_csv_df.loc[unnamed3_idx, ['Unnamed: 3', 'label']].rename(columns={'Unnamed: 3' : 'sms'})
        unnamed4_idx = spam_csv_df['Unnamed: 4'].dropna().index
        unnamed4_df = spam_csv_df.loc[unnamed4_idx, ['Unnamed: 4', 'label']].rename(columns={'Unnamed: 4' : 'sms'})

        kaggle_sms_df = pd.concat([kaggle_sms_df, unnamed2_df, unnamed3_df, unnamed4_df], ignore_index=True) # DataFrame concate

        kaggle_sms_df['length'] = kaggle_sms_df['sms'].apply(len) # sms 문자열 길이

        kaggle_sms_df.info()
        print("\n\n==============================================================================================\n\n")



    # ===================================================================================================
    sms_dataset = load_dataset('sms_spam')
    only_spam_dataset = load_dataset('Ngadou/Spam_SMS')

    # print(sms_dataset)
    # print(only_spam_dataset)

    print("\n\n================================== [ Huggingface Dataset ] =======================================\n\n")


    sms_df = pd.DataFrame({'sms' : sms_dataset['train']['sms'],
                            'label' : sms_dataset['train']['label']})
    sms_df['length'] = sms_df['sms'].apply(len) # sms 문자열 길이

    ngadou_spam_df = pd.DataFrame({'sms' : only_spam_dataset['train']['email'],
                                    'label' : only_spam_dataset['train']['label']})
    ngadou_spam_df = ngadou_spam_df.dropna() # null 한 개 있음
    ngadou_spam_df['length'] = ngadou_spam_df['sms'].apply(len) # sms 문자열 길이

    # 최종 다 합친거 ( kaggle + sms_spam + Ngadou/Spam_SMS )
    df = pd.concat([kaggle_sms_df, sms_df, ngadou_spam_df], ignore_index=True)
    # ham: 0, spam: 1
    # kaggle_sms_df : 0(4886), 1(754)
    # sms_df : 0(4827), 1(747)
    # ngadou_spam_df : 0(2500), 1(499)



    print("[ spam and ham dataset ]")
    print(f"Num of datset : {len(df)}\n")
    print(df['length'].describe(percentiles=[.25, .5, .75, .95, .99]), '\n\n')

    print("[ ham dataset ]")
    print(f"Num of ham datset : {len(df.loc[df['label']==0])}\n")
    print(df.loc[df['label']==0, 'length'].describe(percentiles=[.25, .5, .75, .95, .99]), '\n\n')

    print("[ spam dataset ]")
    print(f"Num of spam datset : {len(df.loc[df['label']==1])}\n")
    print(df.loc[df['label']==1, 'length'].describe(percentiles=[.25, .5, .75, .95, .99]), '\n\n')



    print("\n\n==============================================================================================\n\n")


    # Ham Data, Spam Data (Split)
    ham_df = df[df['label'] == 0]
    spam_df = df[df['label'] == 1]

    # 각각의 Ham Data와 Spam Data를 Train / Validation / Test 분할 (8:1:1)
    train_ham_df, remaining_ham_df = train_test_split(ham_df, test_size=0.2, random_state=42)
    validation_ham_df, test_ham_df = train_test_split(remaining_ham_df, test_size=0.5, random_state=42)
    train_spam_df, remaining_spam_df = train_test_split(spam_df, test_size=0.2, random_state=42)
    validation_spam_df, test_spam_df = train_test_split(remaining_spam_df, test_size=0.5, random_state=42)


    # Train Data 끼리, Test Data 끼리 합친다.
    train_df = pd.concat([train_ham_df, train_spam_df], ignore_index=True)
    validation_df = pd.concat([validation_ham_df, validation_spam_df], ignore_index=True)
    test_df = pd.concat([test_ham_df, test_spam_df], ignore_index=True)


    print("[ Train Dataset ]")
    print(f"Number of Ham Data: {len(train_df[train_df['label']==0])}, \
            Numer of Spam Data: {len(train_df[train_df['label']==1])}")
    print("[ Validation Dataset ]")
    print(f"Number of Ham Data: {len(validation_df[validation_df['label']==0])}, \
            Numer of Spam Data: {len(validation_df[validation_df['label']==1])}")
    print("[ Test Dataset ]")
    print(f"Number of Ham Data: {len(test_df[test_df['label']==0])}, \
            Numer of Spam Data: {len(test_df[test_df['label']==1])}")
    print("\n")
    

    # Train Dataset : 9770 ~= 1600 x 6
    # Spam Data 를 6배 정도 해주면 될 듯
    # Validation Dataset / Test Dataset 유지
    expanded_train_spam_df = train_spam_df.loc[np.repeat(train_spam_df.index.values, 6)].reset_index(drop=True)
    expanded_train_df = pd.concat([train_ham_df, expanded_train_spam_df], ignore_index=True)
    print("[ 변경된 Train Dataset ]")
    print(f"Number of Ham Data: {len(expanded_train_df[expanded_train_df['label']==0])}, \
            Numer of Spam Data: {len(expanded_train_df[expanded_train_df['label']==1])}")
    print("\n")

    return expanded_train_df.loc[:, ['sms', 'label']], validation_df.loc[:, ['sms', 'label']], test_df.loc[:, ['sms', 'label']]

    # =========================================================================
    # 문자열 전체 보기 (show full long string in pandas DataFrame)
    # pd.options.display.max_colwidth = 2000



























# Print Top 10 Accuracy
def show_top_ten(df):
    idx = df['test_accuracy'].argsort()[-10:][::-1]
    idx_unnamed = df.index[idx]
    best_10_accuracy = df.iloc[idx]

    print("[ Best Top 10 Accuracy : {} ]".format(args.model))
    for i in range(len(best_10_accuracy)):
        print("{}: {:.2f}% ( lr = {} / bs = {} / step = {} )"
            .format(i+1, best_10_accuracy['test_accuracy'][idx_unnamed[i]], 
                best_10_accuracy['learning_rate'][idx_unnamed[i]], 
                int(best_10_accuracy['batch_size_train'][idx_unnamed[i]]), 
                int(best_10_accuracy['step'][idx_unnamed[i]])))
    print()

    # best_10_accuracy 에서 등장한 (learning_rate, batch_size_train) 분석
    check_list = np.unique(best_10_accuracy[['learning_rate', 'batch_size_train']].values, axis=0)
    for lr, bs in check_list:
        show_specific_lr_bs(df, lr, int(bs))
        print()


# learning_rate and batch_size_train 기준 Best Top Accuracy
def show_specific_lr_bs(df, lr, bs):
    filt = (df['learning_rate'] == lr) & (df['batch_size_train'] == bs)
    df = df[filt]

    # Accuracy 정렬 이후, 가장 높은 정확도 10개 추출, 마지막으로 내림차순 정렬
    idx = df['test_accuracy'].argsort()[-10:][::-1]
    best_10_accuracy = df.iloc[idx]

    print("[ Best Top 10 Accuracy : ( lr = {} bs = {} ) ]".format(lr, bs))
    print(best_10_accuracy)





















def main(args):
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else :
        device = 'cpu'
    print(device)



    # python model/main.py --graph --model=roberta-base --source_train

    if args.graph:
        save_dir = "save_source_accuracy.csv"

        if not os.path.exists(save_dir):
            raise NotImplementedError("Not exists save_accuracy.csv")
        else:
            df = pd.read_csv(save_dir, index_col='Unnamed: 0')

        
        df.drop_duplicates(['model_name', 'learning_rate', 'batch_size_train', 'step'], ignore_index=True, inplace=True)
        df.to_csv(save_dir)


        filt = (df['model_name'] == args.model)
        df = df.loc[filt, ['learning_rate', 'batch_size_train', 'step', 'test_accuracy', 'f1_score_test']]
        plt.figure(figsize=(6, 9))

        ##################################################################################################################

        minY = df['test_accuracy'].min()
        maxY = df['test_accuracy'].max()

        plt.subplot(2, 1, 1)
        lr_list = list(set(df['learning_rate'].values))
        for lr in lr_list:
            df1 = df.groupby('learning_rate').get_group(lr)
            bs_list = list(set(df1['batch_size_train'].values))
            for bs in bs_list:
                df2 = df1.groupby('batch_size_train').get_group(bs)
                
                plt.plot(df2['step'], df2['test_accuracy'],
                        label='{} / {}'.format(lr, bs),
                        ls='-', marker='o', markersize=2)


        # Best Accuracy
        best_accuracy = df.iloc[np.argmax(df['test_accuracy'])]

        # Max Step Size
        step_size = df.iloc[np.argmax(df['step']), 2]

        # Print Top 10 Accuracy
        show_top_ten(df)
        # learning_rate and batch_size_train 기준 Best Top Accuracy
        # show_specific_lr_bs(df, 7e-5, 32)


        plt.legend(loc='lower right', fontsize=10)
        plt.axis([0, step_size+100, 0, 100])
        plt.ylim(max(0, minY-7), min(100, maxY+7)) # y축 범위 설정
        plt.title('Model Name : {0}\n Best Accuracy : {1:.2f}% ( lr = {2} / bs = {3} / step = {4})'
            .format(args.model, best_accuracy['test_accuracy'], best_accuracy['learning_rate'], 
            int(best_accuracy['batch_size_train']), int(best_accuracy['step'])))
        plt.xlabel('Step')
        plt.ylabel('Accuracy')


        print("\n\n\n###############################################################################")
        print("###############################################################################")
        print("###############################################################################\n\n\n")
        #######################################################################################################

        minY = df['f1_score_test'].min()
        maxY = df['f1_score_test'].max()

        plt.subplot(2, 1, 2)
        lr_list = list(set(df['learning_rate'].values))
        for lr in lr_list:
            df1 = df.groupby('learning_rate').get_group(lr)
            bs_list = list(set(df1['batch_size_train'].values))
            for bs in bs_list:
                df2 = df1.groupby('batch_size_train').get_group(bs)
                
                plt.plot(df2['step'], df2['f1_score_test'],
                        label='{} / {}'.format(lr, bs),
                        ls='-', marker='o', markersize=2)
                
        # Best F1-Score
        best_f1_score = df.iloc[np.argmax(df['f1_score_test'])]


        plt.legend(loc='lower right', fontsize=10)
        plt.axis([0, step_size+100, 0, 100])
        plt.ylim(max(-1, minY-0.05), min(1, maxY+0.05)) # y축 범위 설정
        plt.title('Model Name : {0}\n Best F1-Score : {1:.4f} ( lr = {2} / bs = {3} / step = {4})'
            .format(args.model, best_f1_score['f1_score_test'], best_f1_score['learning_rate'], 
            int(best_f1_score['batch_size_train']), int(best_f1_score['step'])))
        plt.xlabel('Step')
        plt.ylabel('F1-Score')
        plt.tight_layout()
        plt.show()
















        exit()




    # Load Tokenizer
    print("\n[ Load Tokenizer ]")
    tokenizer = AutoTokenizer.from_pretrained(args.model)


    if args.pretrain:
        # Load Source Dataset ( source dataset -> transfer learning )
        train_source_dataset, validation_source_dataset = load_source_dataset()

        # 시작하기 전에 항상 데이터셋을 shuffle
        np.random.seed(100)
        train_source_dataset = np.random.permutation(train_source_dataset)
        validation_source_dataset = np.random.permutation(validation_source_dataset)


        train_inputs = list(train_source_dataset[:, 0])
        train_labels = list(train_source_dataset[:, 1])
        validation_inputs = list(validation_source_dataset[:, 0])
        validation_labels = list(validation_source_dataset[:, 1])

        print("{} examples in train".format(len(train_inputs)))
        print("{} examples in validation".format(len(validation_inputs)))


        output_dir = f"save_source/{ args.model }/lr({ args.learning_rate })bs_train({ args.batch_size_train })"

        # max length 를 train, validation data 길이의 95% 비율로 지정해줌 
        # (결국 이렇게 못함 : 512가 최대 )
        # https://github.com/huggingface/transformers/issues/1215
        train_inputs = tokenized_data(tokenizer, inputs=train_inputs, 
                                    max_length=512) # tokenized about train data
        validation_inputs = tokenized_data(tokenizer, inputs=validation_inputs, 
                                        max_length=512) # tokenized about validation data

        print("\n[ Load pretrained model ]")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2, id2label=id2label, label2id=label2id
        )

        # Determine if it exists
        if not os.path.exists(output_dir):
            output_dir_list = output_dir.split('/')
            for i in range(len(output_dir_list)):
                dir = '/'.join(output_dir_list[:i+1])
                if not os.path.exists(dir):
                    os.mkdir(dir)
        else:
            print ("%s already exists!" % output_dir)


        print("\n[ Start Train ]")
        train(model, train_inputs, train_labels, validation_inputs, validation_labels, 
            batch_size_train=args.batch_size_train, 
            batch_size_validation=args.batch_size_test,
            output_dir=output_dir,
            save_boundary_accuracy=args.save_boundary_accuracy,
            learning_rate=args.learning_rate,
            warmup_steps=50,
            num_training_steps=args.num_training_steps,
            num_epochs=args.epochs,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            eval_period=100,
            device=device)
        

    ########################################################################
    ########################################################################
    ########################################################################
    # 여기서부터 target dataset train, validation, test

    # Load Target Dataset
    train_datset, validation_dataset, test_dataset = load_target_dataset()

    # 시작하기 전에 항상 데이터셋을 shuffle
    np.random.seed(100)
    train_datset = np.random.permutation(train_datset)
    validation_dataset = np.random.permutation(validation_dataset)
    test_dataset = np.random.permutation(test_dataset)

    train_inputs = list(train_datset[:, 0])
    train_labels = list(train_datset[:, 1])
    validation_inputs = list(validation_dataset[:, 0])
    validation_labels = list(validation_dataset[:, 1])
    test_inputs = list(test_dataset[:, 0])
    test_labels = list(test_dataset[:, 1])

    print("{} examples in train".format(len(train_inputs)))
    print("{} examples in validation".format(len(validation_inputs)))
    print("{} examples in test".format(len(test_inputs)))

    # model 이름 / learning rate / batch size train
    output_dir = f"{args.model}/lr({args.learning_rate})bs_train({args.batch_size_train})"




    if args.train or args.transfer_learning:
        # max length 를 train, validation data 길이의 95% 비율로 지정해줌 
        # (결국 이렇게 못함 : 512가 최대 )
        # https://github.com/huggingface/transformers/issues/1215
        train_inputs = tokenized_data(tokenizer, inputs=train_inputs, 
                                    max_length=512) # tokenized about train data
        validation_inputs = tokenized_data(tokenizer, inputs=validation_inputs, 
                                        max_length=512) # tokenized about validation data

        # Huggingface 에 있는 모델을 학습 시킬 경우
        if args.train:
            # Load model
            print("\n[ Load model ]")
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model, num_labels=2, id2label=id2label, label2id=label2id
            )
            output_dir = os.path.join("save_target_train", output_dir)

        # Source Dataset 에서 Pre-train 시켰던 모델을 학습시킬 경우
        elif args.transfer_learning:
            # Load pre-trained model
            print("\n[ Load pre-trained model ]")
            print("[ Model : {} / learning_rate : {} / batch_size_train : {} / steps : {} ]"
                .format(args.model, args.learning_rate, args.batch_size_train, args.steps))
            
            file_name = "model-{}.pt".format(args.steps)
            state_dict = torch.load(os.path.join("save_source", output_dir, file_name))

            model = AutoModelForSequenceClassification.from_pretrained(
                args.model, num_labels=2, id2label=id2label, label2id=label2id,
                state_dict=state_dict
            )
            output_dir = os.path.join("save_target_transfer_learning", output_dir)
        else:
            raise NotImplementedError("Problem about Train or Transfer Learning")

        # Determine if it exists
        if not os.path.exists(output_dir):
            output_dir_list = output_dir.split('/')
            for i in range(len(output_dir_list)):
                dir = '/'.join(output_dir_list[:i+1])
                if not os.path.exists(dir):
                    os.mkdir(dir)
        else:
            print ("%s already exists!" % output_dir)

        print("\n[ Start Train ]")
        train(model, train_inputs, train_labels, validation_inputs, validation_labels, 
            batch_size_train=args.batch_size_train, 
            batch_size_validation=args.batch_size_test,
            output_dir=output_dir,
            save_boundary_accuracy=args.save_boundary_accuracy,
            learning_rate=args.learning_rate,
            warmup_steps=50,
            num_training_steps=args.num_training_steps,
            num_epochs=args.epochs,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            eval_period=100,
            device=device)


    if args.validation or args.test:
        if args.validation: 
            print("\n[ Set the validation dataset ]\n")
            inputs = validation_inputs
            labels = validation_labels
        if args.test:
            print("\n[ Set the test dataset ]\n")
            inputs = test_inputs
            labels = test_labels
        else:
            raise NotImplementedError("Problem about Validation or Test")

        # Determine if it exists
        if not os.path.exists(output_dir):
            raise NotImplementedError("Not exists {} directory", output_dir)
        

        # 입력한게 없으면 해당 디렉토리 모든 .pt 파일 실행
        if args.steps is None:
            steps = [ckpt.split("-")[1].split(".")[0] \
                for ckpt in os.listdir(output_dir) \
                if ckpt.startswith("model-") and ckpt.endswith(".pt")]
        # 입력할 때, 콤마(,)로 구분
        else:
            steps = args.steps.split(",")

        steps = sorted([int(step) for step in steps])
        print("[ Model : {} / learning_rate : {} / batch_size_train : {} / steps : {} / batch_size_test : {} ]"
            .format(args.model, args.learning_rate, args.batch_size_train, args.steps, args.batch_size_test))
        
        for step in steps:
            file_name = "model-{}.pt".format(step)
            state_dict = torch.load(os.path.join(output_dir, file_name))

            model = AutoModelForSequenceClassification.from_pretrained(
                args.model, num_labels=2, id2label=id2label, label2id=label2id,
                state_dict=state_dict
            )
        
        acc, loss, f1_score = test(model, inputs, labels, batch_size=args.batch_size_test, step=step, device=device)

        print("File Name: %s\tAccuracy: %.1f%%\tLoss: %.3f\tF1-Score: %.4f" % (file_name, acc, loss, f1_score))
        print("# of parameters : %d\n" % (np.sum([p.numel() for p in model.parameters()]))) # 11,123,023 == 11M














if __name__ == '__main__':
    # 1. create parser
    parser = argparse.ArgumentParser(description='This is SMS Spam Detection python program.')

    # 2. add arguments to parser
    parser.add_argument('--model', '-m', type=str, default="distilbert-base-uncased", 
        help='model name (Example) : distilbert-base-uncased, distilbert-base-cased, \
            bert-base-uncased, bert-base-cased, roberta-base, etc')

    # 넷 중에 하나 선택
    parser.add_argument('--pretrain', action="store_true", help='Pre-training input model with dataset(SetFit/enron_spam)') # type : boolean
    parser.add_argument('--train', action="store_true", help='First-time train model with dataset(sms_spam, Ngadou/Spam_SMS, kaggle)') # type : boolean
    parser.add_argument('--transfer_learning', action="store_true", help='Learn pre-trained models with dataset(sms_spam, Ngadou/Spam_SMS, kaggle)') # type : boolean
    parser.add_argument('--validation', action="store_true", help='Validation input model with dataset(sms_spam, Ngadou/Spam_SMS, kaggle)') # type : boolean
    parser.add_argument('--test', action="store_true", help='Test input model with dataset(sms_spam, Ngadou/Spam_SMS, kaggle)') # type : boolean
    parser.add_argument('--graph', action="store_true", help='Draw Graph') # type : boolean

    parser.add_argument('--source_train', action="store_true", help='Result of learning with source dataset') # type : boolean
    parser.add_argument('--target_train', action="store_true", help='Result of learning from base model to target dataset') # type : boolean
    parser.add_argument('--target_transfer_learning', action="store_true", help='Result of learning from pre-train model to target dataset') # type : boolean

    parser.add_argument('--batch_size_train', '-bstr', type=int, default=16, help='Size of batch (train)')
    parser.add_argument('--batch_size_test', '-bste', type=int, default=4, help='Size of batch (test)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='Learning Rate to train')

    parser.add_argument('--steps', '-s', type=str, default=None, help='Writing''steps accuracy when validation or testing') # 제대로 활용하기 !!!
    parser.add_argument('--num_training_steps', '-n', type=int, default=2000, help='how many steps when we training')
    parser.add_argument('--save_boundary_accuracy', '-sba', type=float, default=93.0, help='save boundary accuracy in excel file when we training')
    parser.add_argument('--save_boundary_f1_score', '-sbfs', type=float, default=93.0, help='save boundary f1-score in excel file when we training')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epoch to train.')#ignore


    # 3. parse arguments
    args = parser.parse_args()

    assert args.pretrain or args.train or args.transfer_learning or args.validation or args.test or args.graph, 'Choose pretrain or train or validation or test'
    if args.graph:
        assert args.source_train or args.target_train or args.target_transfer_learning, 'Choose source_train or target_train or target_transfer_learning'
    main(args)