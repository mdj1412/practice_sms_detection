import os
import argparse

import numpy as np
import pandas as pd
import torch

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
        max_length=max_length,
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




def load_train_test_dataset():
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
        print("\n\n==========================================================\n\n")



    # =================================
    sms_dataset = load_dataset('sms_spam')
    only_spam_dataset = load_dataset('Ngadou/Spam_SMS')

    print(sms_dataset)
    print(only_spam_dataset)

    print("\n\n==========================================================\n\n")


    sms_df = pd.DataFrame({'sms' : sms_dataset['train']['sms'],
                            'label' : sms_dataset['train']['label']})
    sms_df['length'] = sms_df['sms'].apply(len) # sms 문자열 길이

    ngadou_spam_df = pd.DataFrame({'sms' : only_spam_dataset['train']['email'],
                                    'label' : only_spam_dataset['train']['label']})
    ngadou_spam_df = ngadou_spam_df.dropna() # null 한 개 있음
    ngadou_spam_df['length'] = ngadou_spam_df['sms'].apply(len) # sms 문자열 길이

    # 최종 다 합친거 ( kaggle + sms_spam + Ngadou/Spam_SMS )
    df = pd.concat([kaggle_sms_df, sms_df, ngadou_spam_df], ignore_index=True)


    print("[ spam and ham dataset ]")
    print(f"Num of datset : {len(df)}\n")
    print(df['length'].describe(percentiles=[.25, .5, .75, .95, .99]), '\n\n')


    print("[ ham dataset ]")
    print(f"Num of ham datset : {len(df.loc[df['label']==0])}\n")
    print(df.loc[df['label']==0, 'length'].describe(percentiles=[.25, .5, .75, .95, .99]), '\n\n')


    print("[ spam dataset ]")
    print(f"Num of spam datset : {len(df.loc[df['label']==1])}\n")
    print(df.loc[df['label']==1, 'length'].describe(percentiles=[.25, .5, .75, .95, .99]), '\n\n')



    print("\n\n==========================================================\n\n")
    


    # =========================================================================
    # 문자열 전체 보기 (show full long string in pandas DataFrame)
    # pd.options.display.max_colwidth = 2000








def main(args):
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else :
        device = 'cpu'
    print(device)



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


        output_dir = f"save/{ args.model }/lr({ args.learning_rate })bs_train({ args.batch_size_train })"

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
            # ' save/mrm8488/bert-tiny-finetuned-sms-spam-detection/lr(1e-05)bs_train(16) '
            # save directory 부터 존재하지 않을 때, 생각
            output_dir_list = output_dir.split('/')
            for i in range(len(output_dir_list)):
                dir = '/'.join(output_dir_list[:i+1])
                if not os.path.exists(dir):
                    os.mkdir(dir)
        else:
            print ("%s already exists!" % output_dir)


        print("\n[ Start Train ]")
        train(args, model, train_inputs, train_labels, validation_inputs, validation_labels, 
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




    if args.train:
        # Load Dataset
        load_train_test_dataset()









if __name__ == '__main__':
    # 1. create parser
    parser = argparse.ArgumentParser(description='This is SMS Spam Detection python program.')

    # 2. add arguments to parser
    parser.add_argument('--model', '-m', type=str, default="distilbert-base-uncased", 
        help='model name (Example) : distilbert-base-uncased, distilbert-base-cased, \
            bert-base-uncased, bert-base-cased, roberta-base, etc')

    # 넷 중에 하나 선택
    parser.add_argument('--pretrain', action="store_true", help='Pre-training input model with dataset(SetFit/enron_spam)') # type : boolean
    parser.add_argument('--train', action="store_true", help='Train (Fine-tuning) input model with dataset(sms_spam, Ngadou/Spam_SMS, kaggle)') # type : boolean
    parser.add_argument('--validation', action="store_true", help='Validation input model with dataset(sms_spam, Ngadou/Spam_SMS, kaggle)') # type : boolean
    parser.add_argument('--test', action="store_true", help='Test input model with dataset(sms_spam, Ngadou/Spam_SMS, kaggle)') # type : boolean

    parser.add_argument('--batch_size_train', '-bstr', type=int, default=16, help='Size of batch (train)')
    parser.add_argument('--batch_size_test', '-bste', type=int, default=4, help='Size of batch (test)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='Learning Rate to train')

    parser.add_argument('--steps', '-s', type=str, default=None, help='Writing''steps accuracy when validation or testing') # 제대로 활용하기 !!!
    parser.add_argument('--num_training_steps', '-n', type=int, default=2000, help='how many steps when we training')
    parser.add_argument('--save_boundary_accuracy', '-sba', type=float, default=93.0, help='save boundary accuracy in excel file when we training')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epoch to train.')#ignore
    

    # 3. parse arguments
    args = parser.parse_args()

    assert args.pretrain or args.train or args.validation or args.test, 'Choose pretrain or train or validation or test'
    if args.steps is not None: 
        assert args.validation or args.test, 'Check arguments steps and (validation or test)'

    main(args)