import os
import time
import argparse
from tqdm import tqdm
from evaluate import load
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.util import *
# import apex
# from apex import amp

# 다른논문 모델
from model import *
# dataset
from dataset import *  #########keep on change

from pytorchtools import EarlyStopping
Precautions_msg = '(주의사항) ---- \n'
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import corpus_bleu
# import timm
'''
- train.py

모델을 학습하는 전과정을 담은 코드

#### 실행법 ####
Terminal을 이용하는 경우 경로 설정 후 아래 코드를 직접 실행
python train.py --kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 30

pycharm의 경우: 
Run -> Edit Configuration -> train.py 가 선택되었는지 확인 
-> parameters 이동 후 아래를 입력 -> 적용하기 후 실행/디버깅
--kernel-type 5fold_b3_256_30ep --data-folder original_stone/ --enet-type tf_efficientnet_b3_ns --n-epochs 30

*** def parse_args(): 실행 파라미터에 대한 모든 정보가 있다.  
*** def run(): 학습의 모든과정이 담긴 함수. 이곳에 다양한 trick을 적용하여 성능을 높혀보자. 
** def main(): fold로 나뉜 데이터를 run 함수에 분배해서 실행
* def train_epoch(), def val_epoch() : 완벽히 이해 후 수정하도록


Training list
python train.py --kernel-type volo_1 --out-dim 4 --data-folder images/ --enet-type volo_1 --n-epochs 200 --batch-size 32 --k-fold 0 --image-size 256 --CUDA_VISIBLE_DEVICES 5
python train.py --kernel-type cait_1 --out-dim 4 --data-folder images/ --enet-type cait_1 --n-epochs 200 --batch-size 32 --k-fold 0 --image-size 256 --CUDA_VISIBLE_DEVICES 6
python train.py --kernel-type pvt_1 --out-dim 4 --data-folder images/ --enet-type pvt_1 --n-epochs 200 --batch-size 32 --k-fold 0 --image-size 256 --CUDA_VISIBLE_DEVICES 6
python train.py --kernel-type convnext_small --out-dim 4 --data-folder images/ --enet-type convnext_small --n-epochs 200 --batch-size 32 --k-fold 0 --image-size 256 --CUDA_VISIBLE_DEVICES 4
python train.py --kernel-type coat_1 --out-dim 4 --data-folder images/ --enet-type coat_1 --n-epochs 200 --batch-size 32 --k-fold 0 --image-size 256 --CUDA_VISIBLE_DEVICES 5




'''


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel-type', type=str, required=True)
    # kernel_type : 실험 세팅에 대한 전반적인 정보가 담긴 고유 이름

    parser.add_argument('--data-dir', type=str, default='./data/')
    # base 데이터 폴더 ('./data/')

    parser.add_argument('--data-folder', type=str, required=True)
    # 데이터 세부 폴더 예: 'original_stone/'
    # os.path.join(data_dir, data_folder, 'train.csv')

    parser.add_argument('--image-size', type=int, default='256')
    # 입력으로 넣을 이미지 데이터 사이즈

    parser.add_argument('--enet-type', type=str, required=True, default='tf_efficientnet_b0_ns')
    # 학습에 적용할 네트워크 이름
    # {resnest101, seresnext101,
    #  tf_efficientnet_b7_ns,
    #  tf_efficientnet_b6_ns,
    #  tf_efficientnet_b5_ns...}

    parser.add_argument('--use-amp', action='store_true')
    # 'A Pytorch EXtension'(APEX)
    # APEX의 Automatic Mixed Precision (AMP)사용
    # 기능을 사용하면 속도가 증가한다. 성능은 비슷
    # 옵션 00, 01, 02, 03이 있고, 01과 02를 사용하는게 적절
    # LR Scheduler와 동시 사용에 버그가 있음 (고쳐지기전까지 비활성화)
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/2309

    parser.add_argument('--use-meta', action='store_true')
    # meta데이터 (사진 외의 나이, 성별 등)을 사용할지 여부

    parser.add_argument('--n-meta-dim', type=str, default='512,256')
    # meta데이터 사용 시 중간레이어 사이즈

    parser.add_argument('--out-dim', type=int, default=4)
    # 모델 출력 output dimension

    parser.add_argument('--DEBUG', action='store_true')

    # 디버깅용 파라미터 (실험 에포크를 5로 잡음)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    # 학습에 사용할 GPU 번호

    parser.add_argument('--k-fold', type=int, default=5)
    # data cross-validation
    # k-fold의 k 값을 명시

    parser.add_argument('--log-dir', type=str, default='./logs')
    # Evaluation results will be printed out and saved to ./logs/
    # Out-of-folds prediction results will be saved to ./oofs/

    parser.add_argument('--accumulation-step', type=int, default=1)
    # Gradient accumulation step
    # GPU 메모리가 부족할때, 배치를 잘개 쪼개서 처리한 뒤 합치는 기법
    # 배치가 30이면, 60으로 합쳐서 모델 업데이트함

    # parser.add_argument('--model-dir', type=str, default='./total_weights')
    parser.add_argument('--model-dir', type=str, default='./weights')
    # weight 저장 폴더 지정
    # best :

    parser.add_argument('--use-ext', action='store_true')
    # 원본데이터에 추가로 외부 데이터를 사용할지 여부
    parser.add_argument('--patience', type=int, default=30)

    parser.add_argument('--batch-size', type=int, default=32)  # 배치 사이즈
    parser.add_argument('--num-workers', type=int, default=8)  # 데이터 읽어오는 스레드 개수
    parser.add_argument('--init-lr', type=float, default=1e-6)  # 초기 러닝 레이트. pretrained를 쓰면 매우 작은값 # 4e-5
    parser.add_argument('--n-epochs', type=int, default=200)  # epoch 수
    args, _ = parser.parse_known_args()
    return args


def train_epoch(model, loader, optimizer):
    model.train()
    train_loss = []
    y_pred = []
    y = []
    y_factor = []
    bar = tqdm(loader)

    for (input_ids, attention_mask, target) in tqdm(loader):
        input_ids, attention_mask = input_ids.squeeze().to(device), attention_mask.to(device)
        target = target.input_ids.squeeze().to(device)

        logits = model(input_ids, attention_mask, target)
        # T5 and Kobart
        loss = logits.loss

        # koelectra and GPT
        # loss = criterion(logits, target.float())



        loss.backward()

        # # 그라디언트가 너무 크면 값을 0.5로 잘라준다 (max_grad_norm=0.5)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # gradient accumulation (메모리 부족할때)
        # if args.accumulation_step:
        #     if (i + 1) % args.accumulation_step == 0:
        #         optimizer.step()
        #         # optimizer.zero_grad()
        # else:
        optimizer.step()
            # optimizer.zero_grad()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smooth_loss: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    return train_loss


def val_epoch(model, loader, rot_class, tokenizer):
    '''

    Output:
    val_loss, acc,
    auc   : 전체 데이터베이스로 진행한 validation
    auc_no_ext: 외부 데이터베이스를 제외한 validation
    '''

    model.eval()
    val_loss = []
    mse_loss = []
    weight_loss = []
    class_correct = torch.zeros(len(list(rot_class.keys())))
    class_total = torch.zeros(len(list(rot_class.keys())))
    bleu = load("bleu")
    with torch.no_grad():
        for (input_ids, attention_mask, target, target_data) in tqdm(loader):
            input_ids, attention_mask = input_ids.squeeze().to(device), attention_mask.to(device)
            target = target.input_ids.squeeze().to(device)
            logits = model(input_ids, attention_mask, target)

            # koelectra
            # loss = criterion(logits, target.float())

            predictions = tokenizer.batch_decode(logits, skip_special_tokens=True)
            bleu = load("bleu")
            loss = bleu.compute(predictions=predictions, references=target_data)['bleu']

            val_loss.append(np.array(loss))



    val_loss = np.mean(val_loss)

    return val_loss


def run(df, df_val1, transforms_train, transforms_val):
    # fold, df, transforms_train, transforms_val
    '''
    학습 진행 메인 함수
    :param fold: cross-validation에서 valid에 쓰일 분할 번호
    :param df: DataFrame 학습용 전체 데이터 목록
    :param transforms_train, transforms_val: 데이터셋 transform 함수
    '''
    # tokenizer = T5Tokenizer.from_pretrained("j5ng/et5-formal-convertor")
    # # tokenizer = tokenizer.to(device)
    # input_encoding = tokenizer(df.informal.to, return_tensors="pt", max_length=128, truncation=True)
    # target_encoding = tokenizer(df.informal, return_tensors="pt", max_length=128, truncation=True)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    train_loss_list = []
    valid_loss_set1_list = []

    if args.DEBUG:
        args.n_epochs = 5
        df = df.sample(args.batch_size * 3)

    model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold.pth')  # total_weights

    if os.path.isfile(model_file):
        model = ModelClass(
            args.enet_type,
            out_dim=args.out_dim,
            # pretrained=True,
            # im_size = args.image_size
        )
        model.load_state_dict(torch.load(model_file))

        # loaded_state_dict = torch.load(model_file)
        # new_state_dict = OrderedDict()
        # for n, v in loaded_state_dict.items():
        #     name = n.replace("module.","") # .module이 중간에 포함된 형태라면 (".module","")로 치환
        #     new_state_dict[name] = v

        # model.load_state_dict(new_state_dict)
    else:
        model = ModelClass(
            args.enet_type,
            # out_dim = args.out_dim,
        )

    model = model.to(device)
    # print(torchsummary.summary(model, (1, 256, 256)))
    val_loss_max = 99999.
    val_loss_max2 = 99999.

    optimizer = optim.AdamW(model.parameters(), lr=args.init_lr)

    if DP:
        model = nn.DataParallel(model)

    # amp를 사용하면 버그 (use_amp 비활성화)
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    # scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1,
    #                                             after_scheduler=scheduler_cosine)
    scheduler_warmup = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    # 데이터셋 읽어오기  csv, image_size=256, transform=None, mode='all'
    dataset_valid_set1 = Dataset_train(csv=df_val1, image_size=args.image_size)
    valid_loader_set1 = torch.utils.data.DataLoader(dataset_valid_set1, batch_size=args.batch_size,
                                                    num_workers=args.num_workers)

    # dataset_train = Dataset_train(csv=df, image_size=args.image_size)
    # train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
    #                                            num_workers=args.num_workers)

    classes = {int(s * 10): idx for idx, s in enumerate(range(1, 11))}
    tokenizer = AutoTokenizer.from_pretrained("heegyu/kobart-text-style-transfer")
    for epoch in range(1, args.n_epochs + 1):

        print(time.ctime(), f'Epoch {epoch}')
        # train_loss = train_epoch(model, train_loader, optimizer)
        train_loss=1
        if epoch > 0:
            val_loss_set1 = val_epoch(model, valid_loader_set1, classes, tokenizer)  # val_S, weight_loss

        else:
            val_loss_set1 = 1
        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid set1 loss: {(val_loss_set1):.5f}'#, val_l1 loss: {(mse_loss):.5f}'
        print(content)

        # early_stopping
        early_stopping(val_loss_set1, model)

        if early_stopping.early_stop or epoch == args.n_epochs:
            if early_stopping.early_stop:
                print("Early stopping epoch: ", epoch)
            plt.figure(figsize=(10, 40))
            plt.subplot(3, 1, 1)

            train_min = min(train_loss_list)
            train_x = np.argmin(train_loss_list)

            valid_min_set1 = min(valid_loss_set1_list)
            valid_x_set1 = np.argmin(valid_loss_set1_list)


            plt.plot(train_loss_list)
            plt.text(train_x, train_min, round(train_min, 4))
            plt.plot(valid_loss_set1_list)
            plt.text(valid_x_set1, valid_min_set1, round(valid_min_set1, 4))
            # plt.plot(valid_loss_set2_list)
            # plt.text(valid_x_set2, valid_min_set2, round(valid_min_set2, 4))
            # plt.plot(valid_loss_set3_list)
            # plt.text(valid_x_set2, valid_min_set3, round(valid_min_set3, 4))
            plt.legend(['train_loss', 'val_s_loss_set1'])
            plt.ylabel('loss')
            plt.title(f'{args.kernel_type}')
            plt.grid()

            plt.subplot(3, 1, 2)
            plt.plot(train_loss_list)
            plt.text(train_x, train_min, round(train_min, 4))
            plt.legend(['train_loss'])
            plt.grid()

            plt.subplot(3, 1, 3)
            plt.plot(valid_loss_set1_list)
            plt.text(valid_x_set1, valid_min_set1, round(valid_min_set1, 4))
            plt.legend(['val_s_loss_set1'])
            plt.grid()

            # plt.subplot(5, 1, 4)
            # plt.plot(valid_loss_set2_list)
            # plt.text(valid_x_set2, valid_min_set2, round(valid_min_set2, 4))
            # plt.legend(['val_s_loss_set2'])
            # plt.grid()
            # plt.savefig(f'./SR_results/{args.kernel_type}.jpg')
            # plt.show()

            # plt.subplot(5, 1, 5)
            # plt.plot(valid_loss_set3_list)
            # plt.text(valid_x_set3, valid_min_set3, round(valid_min_set3, 4))
            # plt.legend(['val_s_loss_set3'])
            # plt.grid()
            plt.savefig(f'./results/{args.kernel_type}.jpg')
            plt.show()
            break
        train_loss_list.append(train_loss)
        valid_loss_set1_list.append(val_loss_set1)
        # valid_loss_set2_list.append(val_loss_set2)
        # valid_loss_set3_list.append(val_loss_set3)

        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

        scheduler_warmup.step()
        if epoch == 2:
            scheduler_warmup.step()  # bug workaround

        if val_loss_set1 < val_loss_max:
            print('val_loss_max1 ({:.6f} --> {:.6f}). Saving model ...'.format(val_loss_max, val_loss_set1))
            torch.save(model.state_dict(), model_file)  # SR_weights
            val_loss_max = val_loss_set1

    #################################################################


def main():
    # 데이터셋 읽어오기
    # df_train, df_val1 = get_dataframe(args.k_fold, args.data_dir, args.data_folder, args.out_dim)
    df_train = pd.read_csv('./data/smilestyle_dataset.tsv', delimiter='\t')
    df_train = df_train[['formal', 'informal']].dropna(axis=0)
    df_test = df_train[2800:]
    df_train = df_train[:2800]
    # df_val1 = get_dataframe_val(args.data_dir, args.data_folder, args.out_dim)
    # 모델 트랜스폼 가져오기
    transforms_train, transforms_val = get_transforms(args.image_size)

    run(df_train, df_test, transforms_train, transforms_val)
    # a = torch.tensor([[[[1, 1, 1][1, -8, 1], [1, 1, 1]]]]


if __name__ == '__main__':

    print('----------------------------')
    print(Precautions_msg)
    print('----------------------------')

    # argument값 만들기
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    ###################################
    # 네트워크 타입 설정
    if 'T5_model' == args.enet_type:
        ModelClass = T5_model
    elif 'ElectraModel1' == args.enet_type:
        ModelClass = ElectraModel1
    elif 'kobart' == args.enet_type:
        ModelClass = kobart
    elif 'BART' == args.enet_type:
        ModelClass = bart
    elif 'GPT' == args.enet_type:
        ModelClass = GPT
    elif 'BERT' == args.enet_type:
        ModelClass = BERT
    else:
        raise NotImplementedError()

    # GPU가 여러개인 경우 멀티 GPU를 사용함
    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    # 실험 재현을 위한 random seed 부여하기
    set_seed(4922)
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()
    # criterion2 = nn.L1Loss()
    # criterion = nn.MSELoss()
    # criterion = nn.NLLLoss()
    # cr = nn.HuberLoss()

    # 메인 기능 수행
    main()
