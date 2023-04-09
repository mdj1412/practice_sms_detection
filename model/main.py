import os
import argparse

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import AutoTokenizer


id2label = {0: "ham", 1: "spam"}
label2id = {"ham": 0, "spam": 1}


def string_to_index_label(string_label):
    """
        'ham' -> 0, 'spam' -> 1
    """
    return 0 if string_label=='ham' else 1



def ready_source_dataset():
    """
        source dataset ( huggingface dataset : SetFit/enron_spam )
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


    print(f"Num of train source dataset : {len(train_source_dataset)}\n")
    print(train_source_dataset['length'].describe(), '\n\n')

    print(f"Num of ham (train source dataset) : {len(train_source_dataset.loc[train_source_dataset['label']==0])}\n")
    print(train_source_dataset.loc[train_source_dataset['label']==0, 'length'].describe(), '\n\n')

    print(f"Num of spam (train source dataset) : {len(train_source_dataset.loc[train_source_dataset['label']==1])}\n")
    print(train_source_dataset.loc[train_source_dataset['label']==1, 'length'].describe(), '\n\n')

    print("\n")

    print(f"Num of test source dataset : {len(test_source_dataset)}\n")
    print(test_source_dataset['length'].describe(), '\n\n')

    print(f"Num of ham (test source dataset) : {len(test_source_dataset.loc[test_source_dataset['label']==0])}\n")
    print(test_source_dataset.loc[test_source_dataset['label']==0, 'length'].describe(), '\n\n')

    print(f"Num of spam (test source dataset) : {len(test_source_dataset.loc[test_source_dataset['label']==1])}\n")
    print(test_source_dataset.loc[test_source_dataset['label']==1, 'length'].describe(), '\n\n')

    return train_source_dataset.loc[:, ['text', 'label']], test_source_dataset.loc[:, ['text', 'label']]




def dataset_analysis():
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
    print(df['length'].describe(), '\n\n')


    print("[ ham dataset ]")
    print(f"Num of ham datset : {len(df.loc[df['label']==0])}\n")
    print(df.loc[df['label']==0, 'length'].describe(), '\n\n')


    print("[ spam dataset ]")
    print(f"Num of spam datset : {len(df.loc[df['label']==1])}\n")
    print(df.loc[df['label']==1, 'length'].describe(), '\n\n')



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
        train_source_dataset, validation_source_dataset = ready_source_dataset()

        # 시작하기 전에 항상 데이터셋을 shuffle
        np.random.seed(100)
        train_source_dataset = np.random.permutation(train_source_dataset)
        test_source_dataset = np.random.permutation(test_source_dataset)

        train_inputs = list(train_source_dataset['text'].values)
        train_labels = list(train_source_dataset['label'].values)
        validation_inputs = list(test_source_dataset['text'].values)
        validation_labels = list(test_source_dataset['label'].values)

        print("{} examples in train".format(len(train_inputs)))
        print("{} examples in validation".format(len(validation_inputs)))




    from IPython import embed; embed()

    # Load Dataset
    dataset_analysis()









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
    
    

    # 3. parse arguments
    args = parser.parse_args()

    assert args.pretrain or args.train or args.validation or args.test, 'Choose pretrain or train or validation or test'

    main(args)