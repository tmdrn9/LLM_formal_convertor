import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
from transformers import ElectraModel, ElectraTokenizer, BertLMHeadModel
from transformers import AutoTokenizer, BartForConditionalGeneration, OpenAIGPTLMHeadModel, ElectraForMaskedLM

# https://huggingface.co/docs/transformers/model_doc/t5
class T5_model(nn.Module):
    def __init__(self, enet_type=None, out_dim=2):
        super(T5_model, self).__init__()
        # self.tokenizer = T5Tokenizer.from_pretrained("j5ng/et5-formal-convertor")
        # self.model = T5ForConditionalGeneration.from_pretrained("j5ng/et5-formal-convertor")
        self.model = T5ForConditionalGeneration.from_pretrained("Suppi123/T5-Base-Text-Style-Transfer-Using-Examples")
        # self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        # self.model = T5ForConditionalGeneration.from_pretrained("t5-3b")
        # t5-3b, t5-11b, t5-small
    def forward(self, in_ids, in_attn, target):
        # x = self.tokenizer(x)   , input.attention_mask
        # x = self.model(input_ids = in_ids, attention_mask = in_attn, labels=target)
        x = self.model.generate(input_ids = in_ids,attention_mask = in_attn,do_sample = False)
        return x

class ElectraModel1(nn.Module):
    def __init__(self, enet_type=None, out_dim=2):
        super(ElectraModel1, self).__init__()
        # self.model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-generator")
        self.model = ElectraForMaskedLM.from_pretrained("google/electra-small-generator")
        self.fc = nn.Linear(128,128)
    def forward(self, in_ids, in_attn, target):
        x = self.model(input_ids = in_ids, attention_mask = in_attn , labels=target)#.last_hidden_state
        return x

class kobart(nn.Module):
    def __init__(self, enet_type=None, out_dim=2):
        super(kobart, self).__init__()
        # self.model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-generator")
        self.model = BartForConditionalGeneration.from_pretrained("heegyu/kobart-text-style-transfer")
        # tokenizer = AutoTokenizer.from_pretrained("heegyu/kobart-text-style-transfer")
        self.fc = nn.Linear(128,128)
        # t5-3b, t5-11b, t5-small
    def forward(self, in_ids, in_attn, target):
        # x = self.tokenizer(x)   , input.attention_mask
        x = self.model(input_ids = in_ids, labels=target)
        # x = self.fc(torch.mean(x, 2))
        return x

class bart(nn.Module):
    def __init__(self, enet_type=None, out_dim=2):
        super(bart, self).__init__()
        # self.model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-generator")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        # tokenizer = AutoTokenizer.from_pretrained("heegyu/kobart-text-style-transfer")
        # self.fc = nn.Linear(128,128)
        # t5-3b, t5-11b, t5-small
    def forward(self, in_ids, in_attn, target):
        # x = self.tokenizer(x)   , input.attention_mask
        # x = self.model(input_ids = in_ids, labels=target)
        x = self.model.generate(input_ids = in_ids, labels = target)
        # x = self.fc(torch.mean(x, 2))
        return x

class GPT(nn.Module):
    def __init__(self, enet_type=None, out_dim=2):
        super(GPT, self).__init__()
        self.model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")

    def forward(self, in_ids, in_attn, target):
        # x = self.tokenizer(x)   , input.attention_mask
        # x = self.model(input_ids = in_ids, labels = target)
        x = self.model.generate(input_ids=in_ids, labels = target)

        # x = self.fc(torch.mean(x, 2))
        return x

class BERT(nn.Module):
    def __init__(self, enet_type=None, out_dim=2):
        super(BERT, self).__init__()
        # self.model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-generator")
        self.model = BertLMHeadModel.from_pretrained("Suppi123/Bert-Base-Uncased-Text-Style-Transfer-Using-Examples")
        # tokenizer = AutoTokenizer.from_pretrained("heegyu/kobart-text-style-transfer")
        # self.fc = nn.Linear(128,128)
        # t5-3b, t5-11b, t5-small
    def forward(self, in_ids, in_attn, target):
        # x = self.tokenizer(x)   , input.attention_mask
        # x = self.model(input_ids = in_ids, labels = target)
        x = self.model.generate(input_ids = in_ids, labels = target)
        # x = self.fc(torch.mean(x, 2))
        return x



# OpenAIGPTModel.from_pretrained("openai-gpt")
# heegyu/kobart-text-style-transfer
# model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-generator")
# tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-generator")
# T5 모델 로드
# model = T5ForConditionalGeneration.from_pretrained("j5ng/et5-formal-convertor")
# tokenizer = T5Tokenizer.from_pretrained("j5ng/et5-formal-convertor")
#
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# # device = "mps:0" if torch.cuda.is_available() else "cpu" # for mac m1
#
# model = model.to(device)
#
# # 예시 입력 문장
# input_text = "이게 프로젝트라고 가져온건가요?"
#
# # 입력 문장 인코딩
# input_encoding = tokenizer("존댓말로 바꿔주세요: " + input_text, return_tensors="pt")
# input_encoding = tokenizer("반말로 바꿔주세요: " + input_text, return_tensors="pt")
#
# input_ids = input_encoding.input_ids.to(device)
# attention_mask = input_encoding.attention_mask.to(device)
#
# # T5 모델 출력 생성
# output_encoding = model.generate(
#     input_ids=input_ids,
#     attention_mask=attention_mask,
#     max_length=128,
#     num_beams=5,
#     early_stopping=True,
# )
#
# # 출력 문장 디코딩
# output_text = tokenizer.decode(output_encoding[0], skip_special_tokens=True)
#
# # 결과 출력
# print(output_text) # 저 진짜 화났습니다 지금.
