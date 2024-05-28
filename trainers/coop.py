import os.path as osp

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from copy import deepcopy
from data.vqav2 import VQAv2
import torchvision.transforms as transforms
import PIL
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils.tpt_tools import Summary, ProgressMeter, accuracy, load_model_weight, set_random_seed
from dassl.utils.tpt_tools import AverageMeter as AverageMeter_TPT


from clip import clip
import time
import re
import torch.backends.cudnn as cudnn
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = 1000
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        ctx = nn.Parameter(ctx_vectors)
        # Default is 1, which is compound shallow prompting
        
        # visual_ctx_vectors = torch.empty(n_ctx, 768, dtype=dtype)
        # nn.init.normal_(visual_ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
        # print(' design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        # self.proj = nn.Linear(ctx_dim, 768)
        # self.proj.half()
        
        self.ctx = ctx
        self.n_cls = n_cls
        self.n_ctx = n_ctx

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        # if label is not None:
        #     prefix = prefix[label]
        #     suffix = suffix[label]

        # print(prefix.shape)
        # print(ctx.shape)
        # print(suffix.shape)
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts
    def forward(self,embed,t):
        # ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        # nn.init.normal_(ctx_vectors, std=0.02)
        # self.ctx = nn.Parameter(ctx_vectors)
        
        ctx = self.ctx

        if ctx.dim() == 2:
            if t: 
                ctx = ctx.unsqueeze(0).expand(2, -1, -1)
            else: 

                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, embed[:, :1, :], embed[:, 1+self.n_ctx:, :])


        return prompts
    def set_prompt_init_states(self):
        '''
        Store the initial prompts
        '''
        ctx_vectors = self.ctx.detach().clone()
        self.ctx_init_state = ctx_vectors
    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip = clip_model
        self.vocab={}
        reverse ={}
        vocab_path = '/raid/biplab/hassan/VQA_CLIP/vqa_common_ab.txt'
        i=0
        with open(vocab_path, 'r') as file:
            for line in file:
                line = line.replace('\n','')
                self.vocab[line]=i
                reverse[i]=line
                i+=1
    def generate_yes_no_phrases(self,phrase):
        # print(phrase)
        pattern = r"\b(is|was|has been|are|were)\b"
        match = re.search(pattern, phrase, re.IGNORECASE)
        
        if match:
            verb = match.group(0)
            start, end = match.span(0)
            yes_phrase = "Yes, " + phrase
            no_phrase = phrase[:start] + "is not" + phrase[end:]
            
            return ["No, " + no_phrase,yes_phrase]
        else:
            return ["No, " + phrase,"Yes, " + phrase]
    def forward(self, image,phrase,t):
        # print(image.shape)
        if t:
            phrases = self.generate_yes_no_phrases(phrase[0])
        else:
            pattern = r"(.*?)<answer>(.*?)| "
            # print(ques)
            phrases=[]
            # print(phrase)
            if re.match(pattern,phrase[0]):
                for c in self.vocab.keys():
                    phrases.append("X"+phrase[0].replace("<answer>",c))
        # print(phrases)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in phrases]).to('cuda') # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = self.clip.token_embedding(tokenized_prompts).type(self.dtype)
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner(embedding,t)
        tokenized_prompts = tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
    def reset(self):
        self.prompt_learner.reset()
    def set_prompt_inits(self):
        print("Re-updating prompt initializations to current prompts.")
        self.prompt_learner.set_prompt_init_states()


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
    def build_dataset(self,set_id,transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
        dataset = VQAv2(self.cfg)

        return dataset
    def get_tpt_dataloader(self):

        # print("number of test samples: {}".format(len(val_dataset)))
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])
        data_transform = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.GaussianBlur(3),
                transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
                transforms.ToTensor(),
                normalize,
            ])
        batchsize = 1
        set_id = self.cfg.DATASET.DATA
        val_dataset = self.build_dataset(set_id, data_transform, self.cfg.DATASET.ROOT, mode='test')
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=8, pin_memory=True)
        
        return val_loader
    def build_data_loader(self):
        super().build_data_loader()
        self.tpt_loader = self.get_tpt_dataloader()
    
    def tpt(self):
        """
        Run Test-time prompt Tuning
        """
        self.model.set_prompt_inits()   # Init with current prompts
        for name, param in self.model.named_parameters():
            # if not self.cfg.TPT.COCOOP: # MaPLe and CoOp
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")



        
        trainable_param = self.model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, self.cfg.TPT.LR)
        optim_state = deepcopy(optimizer.state_dict())

        # setup automatic mixed-precision (Amp) loss scaling
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        print('=> Using native Torch AMP. Training in mixed precision.')
        print("number of test samples: {}".format(len(self.tpt_loader.dataset)))

        cudnn.benchmark = True

        results = {}
        set_id = self.cfg.DATASET.TPT
        results[set_id] = self.test_time_adapt_eval(self.tpt_loader, self.model, optimizer, optim_state, scaler, self.cfg.TPT)
        return results
    def generate_yes_no_phrases(self,phrase):
        # print(phrase)
        pattern = r"\b(is|was|has been|are|were)\b"
        match = re.search(pattern, phrase, re.IGNORECASE)
        
        if match:
            verb = match.group(0)
            start, end = match.span(0)
            yes_phrase = "Yes, " + phrase
            no_phrase = phrase[:start] + "is not" + phrase[end:]
            
            return ["No, " + no_phrase, yes_phrase ]
        else:
            return "No, " + phrase,"Yes, " + phrase
    def test_time_adapt_eval(self, val_loader, model, optimizer, optim_state, scaler, args):
        batch_time = AverageMeter_TPT('Time', ':6.3f', Summary.NONE)
        top1 = AverageMeter_TPT('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter_TPT('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: ')
        print("$"*40)

        # reset model and switch to evaluate mode
        model.eval()
        # if not args.COCOOP: # no need to reset cocoop because it's fixed
        with torch.no_grad():
                model.reset()
        end = time.time()
        total = 0
        correct =0
        for i, batch in enumerate(val_loader):
            # images, target = self.parse_batch_test(batch)
            images,phrase,answer,ques = batch["img"],batch["phrase"],batch["answer"],batch["question"]
            pattern = r"(.*?)<answer>(.*?)| "
            # print(ques)
            # phrases=[]
            # print(phrase)
            t=0
            if len(phrase[0].split(" "))>60:
                continue
            if answer[0]==0 or answer[0]==1:
                t=1
            elif not re.match(pattern,phrase[0]):
                continue

            if isinstance(images, list):
                for k in range(len(images)):
                    # images[k] = images[k].cuda(args.gpu, non_blocking=True)
                    images[k] = images[k].to(self.device)
                image = images[0]
            else:
                if len(images.size()) > 4:
                    # when using ImageNet Sampler as the dataset
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                # images = images.cuda(args.gpu, non_blocking=True)
                images = images.to(self.device)
                image = images
            # target = target.cuda(args.gpu, non_blocking=True)
            # target = target.to(self.device)
            if args.RUN:
                images = torch.cat(images, dim=0)

            
            with torch.no_grad():
                model.reset()
            image = images[0].unsqueeze(0)
            images =  images.to('cuda')
            image = image.to('cuda')
            answer =  answer.to('cuda')

            optimizer.load_state_dict(optim_state)
            self.test_time_tuning(model, (images,phrase,t), optimizer, scaler, args,i)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output = model(image,phrase,t)

            # measure accuracy and record loss
            # if output.isnan().any():
            #     print(phrase)
            # else:
            #     print("Got it")
            # print(ques)

            # print(output)
            # print()
            # print("-----")
            # print(torch.argmax(output,dim=1))
            # print(answer)
            if answer[0]!=0 and answer[0]!=1:
                if answer[0]==torch.argmax(output,dim=1):
                    correct+=1
                # print("-"*40)
                # print(answer[0])
                # print(torch.argmax(output,dim=1))
                # print("-"*40)

                total+=1
            # print("-----")
            acc1, acc5 = accuracy(output,answer, topk=(1, 2))
                    
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % 200 == 0:
                progress.display(i)
                print("Correct ",correct," total ",total)

        progress.display_summary()

        return top1.avg
    def select_confident_samples(self, logits, topTPT):
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        idxTPT = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topTPT)]
        return logits[idxTPT]
    def avg_entropy(self, outputs):
        # outputs = outputs - outputs.max()
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
    def test_time_tuning(self, model, inputs, optimizer, scaler, args,c):
        trainable_param = self.model.prompt_learner.parameters()
        # optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=5e-2, momentum=0.9)
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=4e-2)
        for j in range(1):
            with torch.cuda.amp.autocast():
                # print(inputs[0].shape)
                output = model(inputs[0],inputs[1],inputs[2]) 
                # print(model.prompt_learner.proj.weight)
                output= self.select_confident_samples(output,0.2)
                # print(output.shape)
                # print(output)
                # print(torch.argmax(output[:3],dim=1))
                loss = self.avg_entropy(output)
                # print(output[0])
                # loss = nn.CrossEntropyLoss()(output[0],torch.tensor(1).to('cuda'))
                # loss = -(output[0].unsqueeze(0).softmax(1) * output[0].unsqueeze(0).log_softmax(1)).sum(1).sum()
                # print(loss.item())
                # print(output[0][:2])

        optimizer.zero_grad()
        # print("---------------------------------")
        # print(model.prompt_learner.ctx.isnan().any())
        # print(model.prompt_learner.ctx.grad.isnan().any())
        # print(loss.item())
        loss.backward()
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        # Unscales the gradients of optimizer's assigned params in-place
        # scaler.step(optimizer)
        # scaler.update()
        optimizer.step()
        # print(model.prompt_learner.ctx.isnan().any())
        # print("----------------------------------")
                # if model.prompt_learner.ctx.isnan().any():
                #     print(model.prompt_learner.ctx)
                #     print(c)
                    
                # print(model.prompt_learner.vis_ctx)
                # output = model(inputs[0],inputs[1]) 
                # print(output[0])
                # scaler.update()
        
        return

    def build_model(self):
        cfg = self.cfg
        # classnames = self.dm.dataset.classnames
        classnames =[]
        vocab_path = '/raid/biplab/hassan/VQA_CLIP/vqa_common_ab.txt'
        with open(vocab_path, 'r') as file:
                for line in file:
                    line = line.replace('\n','')
                    classnames.append(line)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)