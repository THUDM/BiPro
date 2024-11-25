

# here put the import lib
from functools import partial
import os
import sys
sys.getdefaultencoding()
      
import random
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import argparse
import stat
from functools import partial
from poem_verifier import *

from sat import mpu, get_args, get_tokenizer
from sat.training.model_io import get_checkpoint_iteration,get_checkpoint_name
from sat.arguments import initialize_distributed, set_random_seed
from sat.generation.autoregressive_sampling import get_masks_and_position_ids_default, update_mems
from sat.model import GLMModel
from sat.model.mixins import CachedAutoregressiveMixin
from beam_search_strategy import BeamSearchStrategy
from mtgen import MultiGenStrategy
from sat.generation.sampling_strategies import BaseStrategy
from sat.generation.utils import timed_name, generate_continually
from pynvml import *
from poem_verifier import get_last_piece
from ip_score import compute_ip,compute_p


def get_masks_and_position_ids_glm(seq, mask_position, context_length):
    
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    attention_mask.unsqueeze_(1)

    position_ids = torch.zeros(2, len(seq), device=tokens.device, dtype=torch.long)
    torch.arange(0, context_length, out=position_ids[0, :context_length])
    position_ids[0, context_length:] = mask_position
    torch.arange(1, len(seq) - context_length + 1, out=position_ids[1, context_length:])

    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids

# GLM-10b-chinese ckpt has different name than sat requires
def load_checkpoint(model, args, load_path=None, prefix=''):
    """Load a model checkpoint."""
    if load_path is None:
        load_path = args.load

    # If model-only mode, set necessary args.
    if not hasattr(args, 'mode'):
        from copy import deepcopy
        args = deepcopy(args)
        args.mode = 'inference'

    iteration, release, success = get_checkpoint_iteration(load_path)
    if not success:
        return 0
    
    checkpoint_name = get_checkpoint_name(load_path, iteration, release)
    if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))
    sd = torch.load(checkpoint_name, map_location='cpu')
    # fix name differences
    sd['module']['transformer.word_embeddings.weight'] = sd['module']['word_embeddings.weight']
    del sd['module']['word_embeddings.weight']

    sd['module']['mixins.block_position_embedding.block_position_embeddings.weight'] = sd['module']['transformer.block_position_embeddings.weight']
    del sd['module']['transformer.block_position_embeddings.weight']
    new_sd = {'module':{}}
    for k in sd:
        if k != 'module':
            new_sd[k] = sd[k]
    for k in sd['module']:
        if k.startswith(prefix):
            new_sd['module'][k[len(prefix):]] = sd['module'][k]
    
    sd = new_sd
    
    if hasattr(model, 'module'):
        module = model.module
    else: # inference without deepspeed
        module = model

    # only load module, other hyperparameters are just for recording.
    #print(module['transformer.word_embeddings.weight'])
    missing_keys, unexpected_keys = module.load_state_dict(sd['module'], strict=False)
    if len(unexpected_keys) > 0:
        print_rank_0(
            f'Will continue but found unexpected_keys! Check whether you are loading correct checkpoints: {unexpected_keys}.')
    if len(missing_keys) > 0:
        if args.mode == 'inference':
            if 'force_inference' in args and args.force_inference:
                print(f'Warning: Missing keys for inference: {missing_keys}.')
            else:
                raise ValueError(f'Missing keys for inference: {missing_keys}.\nIf you still want to inference anyway, pass --force_inference to args.')
        else: # new params
            assert all(name.find('mixins')>=0 for name in missing_keys), missing_keys
            assert args.mode == 'finetune'
            # list all mixin names
            mixin_names = []
            for key_name in missing_keys:
                parts = key_name.split('.')
                mixin_name = parts[parts.index('mixins')+1]
                if mixin_name not in mixin_names:
                    mixin_names.append(mixin_name)
            module.reinit(mixin_names) # initialize mixins

   
    module.eval()

    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))
    del sd
    return iteration
def judge_type(numline,totallines):
    line=totallines[numline].replace('•','·')
    if '·' in line:
        lsp=line.strip().split('·')
        return [lsp[1],lsp[0]]
    else:
        title=line.strip()
        line2=totallines[numline+1].replace('，',',').replace('。',',').replace('.',',').split(',')[0].strip()
        #print(totallines[numline+1],line2)
        lline=len(line2)
        lc=0
       
        while (numline<len(totallines)) and (len(totallines[numline].strip())!=0):
            l=totallines[numline+1].replace('，',',').replace('。',',').replace('.',',').replace('\n','').split(',')
            #print(l,lc)
            for case in l:
                if len(case)==lline:
                    lc+=1
            numline+=1
        current=''

        if lline==5: 
            current+='五'
        if lline==7:
            current+='七'
        if lc==4:
            current+='绝'
        if lc==8:
            current+='律'
        return [title,current]
    






                

def main(args):
    args.do_train = False
    initialize_distributed(args)
    print("initialize finished")
    tokenizer = get_tokenizer(args)
    print("tokenizer finished")
    # build model 
    model = GLMModel(args)
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    if args.fp16:
        model = model.half()
    model = model.to(args.device)
    #print(model)
    load_checkpoint(model, args)
    set_random_seed(args.seed)
    model.eval()
    worddict,shengdict,allbu,allsb=pingshui()
 
    raw_end_tokens = [tokenizer.get_command('eop').Id, tokenizer.get_command('eos').Id]
    # define function for each query
    strategy=MultiGenStrategy(args.batch_size, length_penalty=args.length_penalty, end_tokens=raw_end_tokens, verifier=poem_verifier,
        tmp_factor=1.0)
    strategy.set_model(model)
    def reverse_end(endtk):
        import copy
        if '。' in endtk:
            
            new_endtk=[',','，']
            end_tokens=[tokenizer.TokenToId(',')]
        else:
            end_tokens=[tokenizer.TokenToId('。'),tokenizer.TokenToId('.')]
            new_endtk=['.','。']
        strategy.set_end_tokens(copy.copy(end_tokens))
        strategy.set_verifier_param(copy.copy(new_endtk),7)
        return new_endtk,end_tokens
    def poemgen(title,author='杜甫',emo=None,head='',format=''):
        title=title.replace('《','').replace('》','')
        if args.with_id:
            query_id, raw_text = raw_text.split('\t')
        emo_str=''
        if ' ' in title:
            tt=title.split()[0]
            author=title.split()[1]
            title=tt
        if emo is not None:
            emo_str=' 情感：'+emo
        
        if format=='':
            format='格律诗'
        else:
            format=format.replace('五','五言').replace('七','七言').replace('绝','绝句').replace('律','律诗')
        raw_text='诗歌《'+title+'》 '+'作者:李白 体裁:'+format+emo_str+' 标题:'+title+' 正文:[gMASK]'
     
        max_len=7
        min_len=5
        
        if '五' in format:
            max_len=5
        if '七' in format:
            min_len=7
        
        tl=8
        if '绝' in format:
            tl=4      
      
        output_list = []
        end_tokens=raw_end_tokens+[tokenizer.TokenToId('。'),tokenizer.TokenToId('.')]
        endtk=['.','。']
            
        vparam=[tokenizer,worddict,shengdict,2,2,min_len,max_len,endtk,'']
        strategy.set_verifier_params(vparam)
        mems=None
        yayun=''
        rhy=2
        input_sentence=raw_text
        #print(input_sentence)
        for ids in range(tl):
            endtk,end_tokens=reverse_end(endtk)
            #print(endtk)
            if ids%2==0:
                strategy.set_verifier_param('',8)
            else:
                strategy.set_verifier_param(yayun,8)
            if ids>0:
                strategy.set_verifier_param(1-ids%2,4)
            if rhy!=2:
                if ids in [1,2,5,6]:
                    strategy.set_verifier_param(1-rhy,3)
                else:
                    strategy.set_verifier_param(rhy,3)
                #print(input_seq)
            start=None
            if len(head)>ids:
                start=head[ids]

            output = generate_sentence_with_constraint(model, input_sentence,
                        tokenizer,
                        num_generated=10,
                        strategy=strategy,
                        end_tokens=end_tokens,
                        weight=0.45,
                        start=start
                        )
            
            new_sentence=input_sentence.replace('[gMASK]','')+output
            #print("generated sentence:",new_sentence)
            if ids==0:
                input_sentence=new_sentence+'[gMASK]'
            yayun,rhy,length=verify_rhy(new_sentence,ids,shengdict,yayun,rhy)
            # returned rhy from verify_rhy is poem-wise rhy, have to transform in sentence-wise usage.

            if ids==0:
                strategy.set_verifier_param(length,5)
                strategy.set_verifier_param(length,6)
            strategy._init_cache()
              
                # refine 
            
            if ids>0:
                pos=len(input_sentence.replace('[gMASK]',''))-2 
                while not(input_sentence[pos] in [',','，','.','。',':','：']):
                    pos-=1
                
                prev_sentence=input_sentence[pos+1:].replace('[gMASK]','')
                
                new_sentence_shifted=input_sentence[:pos+1]+'[sMASK]'+output
                    # set new verifier params
                w,end_tokens=reverse_end(endtk)
                if (ids%2==1 and ids>1):
                    strategy.set_verifier_param('',8)
                else:
                    strategy.set_verifier_param(yayun,8)
                strategy.set_verifier_param(ids%2,4)
                
                if rhy!=2:
                    if ids-1 in [1,2,5,6]:
                        strategy.set_verifier_param(1-rhy,3)
                    else:
                        strategy.set_verifier_param(rhy,3)
                start=None
                if len(head)>ids-1:
                    start=head[ids-1]
                out = generate_sentence_with_constraint(model, new_sentence_shifted,
                        tokenizer,
                        num_generated=10,
                        strategy=strategy,
                        end_tokens=end_tokens,
                        weight=0.45,
                        start=start,
                        excess_beam=prev_sentence
                        )

                input_sentence=new_sentence_shifted.replace('[sMASK]',out)
                if out!=prev_sentence:
                    q=0
                    #print('refined_sentence:',input_sentence," replaced ",prev_sentence," with ",out)
                yayun,rhy,length=verify_rhy(out,ids-1,shengdict,yayun,rhy)
                input_sentence=input_sentence+'[gMASK]'
                strategy._init_cache()
        #
        if '格律诗' in format:
            # compare 4 and 8 sentence version

            score_title,score_emo,score_cor=compute_ip(model,input_sentence,tokenizer)
            total_score_8=aggregate_scores(score_title,score_emo,0)
            input_sentence_4='。'.join(input_sentence.replace('.','。').split('。')[0:2])+'。'
            score_title,score_emo,score_cor=compute_ip(model,input_sentence_4,tokenizer)
            total_score_4=aggregate_scores(score_title,score_emo,0)
            if total_score_4>total_score_8:
                input_sentence=input_sentence_4

        final_output=input_sentence.split('：')[-1].replace('[gMASK]','')

        return yayun,rhy,length,final_output
    def splitpoem(ids,poem):

        
        pts=0
        num_tks=0
        while num_tks<ids:
            if poem[pts] in [',','，','。','.']:
                num_tks+=1
            pts+=1
        spbf=poem[:pts]
        pp=pts
        while num_tks<=ids:
            if poem[pts] in [',','，','。','.']:
                num_tks+=1
            pts+=1
        spaf=poem[pts:]
        sp=poem[pp:pts]
        return spbf,spaf,sp
    def computesp(poem):
        count=0
        for char in poem:
            if char in [',','，','。','.']:
                count+=1
        return count
    def refine_poem(yayun,rhy,length,poem_full,wt=0,head='',format=None):
        poem=poem_full.split(':')[-1]
        pretext=poem_full[:len(poem_full)-len(poem)]

        endtk=['。']
        
        pms_len=computesp(poem)
        q=0
        for ids in range(pms_len):
            strategy._init_cache()
            
            bf,af,sp=splitpoem(ids,poem)
            endtk,end_tokens=reverse_end(endtk)
            if (ids%2==0 and ids>0):
                strategy.set_verifier_param('',8)
            else:
                strategy.set_verifier_param(yayun,8)

            if ids>0:
                strategy.set_verifier_param(1-ids%2,4)
            else:
                strategy.set_verifier_param(2,4)
            if rhy!=2:
                if ids in [1,2,5,6]:
                    strategy.set_verifier_param(1-rhy,3)
                else:
                    strategy.set_verifier_param(rhy,3)
            
            new_prior=pretext+bf
            
            new_poior='[sMASK]'+af
            if ids==pms_len-1:
                new_poior='[gMASK]'
            
            new_sentence=new_prior+new_poior
                
            
            #pos=strategy.get_pos()
            global pos
            start=None
            if len(head)>ids:
                start=head[ids]
            #print("before refine:",new_sentence)
            out = generate_sentence_with_constraint(model, new_sentence,
                        tokenizer,
                        num_generated=10,
                        strategy=strategy,
                        end_tokens=end_tokens,
                        weight=0.45,
                        start=start,
                        excess_beam=sp
                        )
            get_out=new_sentence+out
            if '[sMASK]' in new_sentence:
                poem_full=new_sentence.replace('[sMASK]',out)
            else:
                poem_full=new_sentence.replace('[gMASK]',out)
            poem=poem_full.split(':')[-1]
            #print("after refine:",poem)
            if out!=sp:
                q+=1
                    #print('refined_sentence:',input_sentence," replaced ",prev_sentence," with ",out)
            yayun,rhy,length=verify_rhy(out,ids,shengdict,yayun,rhy)
            
            #print("current rhy:",rhy)
            strategy._init_cache()
        return poem_full,q
    def generate_from_source(isource):
        p=open(isource,'r')
        q=p.readlines()
        p.close()

        tt_list=[]
        for num in range(len(q)):
            if (num==0 or len(q[num-1].strip())==0):
                if q[num]!='\n':
                    if not('，' in q[num]):
                        tpo=judge_type(num,q)
                        tt_list.append(tpo)
        p=open("supported_list.txt",'r')
        q=p.readlines()
        for num in range(len(q)):
            q[num]=q[num].strip()
        supported_list=q
    
        for item in tt_list:
            print(item,supported_list)
            if item[1] in supported_list:
                process(item[0],format=item[1])
        
        return 0
    def generate_from_titles(isource):
        p=open(isource,'r')
        q=p.readlines()
        p.close()

        tt_list=[]
        for num in range(len(q)):
            if (len(q[num].strip())!=0):
               tt_list.append(q[num].strip())
        import random
        random.shuffle(tt_list)
        for item in tt_list:
            process(item)
        return 0
    
    def process(title,out_dir="李白",author='李白',emo=None,format=''):
        head=''
        if '|' in title:
            head=title.split('|')[1].strip()
            title=title.split('|')[0].strip()
        emo_str=''
        if ' ' in title:
            tt=title.split()[0]
            emo=title.split()[1]
            title=tt
        if emo is not None:
            emo_str=' 情感：'+emo
        generated_list=os.listdir("iprompt_exp2")
        #if title+'.txt' in generated_list:
        #    return 0

        f=open("iprompt_exp2/"+title+'.txt','w',encoding='utf-8')
        print(title)
        print(title,file=f)
        yayun,rhy,length,poem=poemgen(title,author=author,emo=emo,head=head,format=format)
        print("Generation:",poem)

        print("Generation:",poem.split(':')[-1].replace('.','。').replace(',','，'),file=f)
        refine_times=10

        for i in range(refine_times):
            poem,q=refine_poem(yayun,rhy,length,poem,head=head,format=format)
            if q==0:
                break
            print("Refinement",i,':',poem.split(':')[-1].replace('.','。').replace(',','，'))
            print("Refinement",i,':',poem.split(':')[-1].replace('.','。').replace(',','，'),file=f)
        f.close()
        return 0
    #generate_from_source("human.txt")
    generate_from_titles("titles2.txt")
def aggregate_scores(score_title,score_emo,score_cor):
    score=score_title
    if score_emo is not None:
        score=score+0.2*score_emo
    if score_cor is not None:
        score=score+0.15*score_cor
    return score
def generate_sentence_with_constraint(
        model,
        input_word,
        tokenizer,
        num_generated=1,
        max_memory_length=100000,
        strategy=None,
        end_tokens=[],
        verifier_params=None,
        weight=0.75,
        use_ip=0,
        excess_beam=None,
        start=None
        ):
    prefix=""
    #prefix=  "体裁：五言绝句 标题：静夜思 正文：床上明月光，疑是地上霜。 举头望明月，低头思故乡。\n 体裁：五言律诗 标题：赋得古原草送别 正文：离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。远芳侵古道，晴翠接荒城。又送王孙去，萋萋满别情。\n"
    #prefix=prefix+"体裁：七言绝句 标题：望庐山瀑布 正文：日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。\n体裁：七言律诗 标题：黄鹤楼 正文：故人西辞黄鹤楼，烟花三月下扬州。黄鹤一去不复返，白云千载空悠悠。晴川历历汉阳树，芳草萋萋鹦鹉洲。日暮乡关何处是，烟波江上使人愁。\n"
    prefix_len=0
    if len(prefix)>0:
        prefix_len=len(tokenizer.EncodeAsIds(prefix).tokenization)
    
    # building the initial tokens, attention_mask, and position_ids
    fix_logit=None
    mems=None
    # label the positions of title
    import copy
    input_word_initial=copy.deepcopy(input_word)
    input_word=prefix+input_word
    input_word_pre_title=input_word.split("》")[0]+'》'
    len_pre_title=len(tokenizer.EncodeAsIds(input_word_pre_title).tokenization)
    if '[sMASK]' in input_word:
        input_word_pre_mask=input_word.split("[sMASK]")[0]
    else:
        input_word_pre_mask=input_word.split("[gMASK]")[0]
    len_pre_mask=len(tokenizer.EncodeAsIds(input_word_pre_mask).tokenization)

    seq = tokenizer.EncodeAsIds(input_word).tokenization
    print(tokenizer.DecodeIds(seq).split(':')[-1])
    seq = [tokenizer.get_command('ENC').Id] + seq
    

    if '[gMASK]' in input_word:
        input_seq = torch.cuda.LongTensor(
                    seq + [tokenizer.get_command('sop').Id] + [-1] * (args.out_seq_length - len(seq) - 1),
                    device=args.device)
        context_length=len(seq)
    else:
        input_seq = torch.cuda.LongTensor(
                    seq + [tokenizer.get_command('eos').Id,tokenizer.get_command('sop').Id] + [-1] * (args.out_seq_length - len(seq) - 2),
                    device=args.device)
        context_length=len(seq)+1

   
    tokens, attention_mask, position_ids =get_masks_and_position_ids_glm(input_seq,len_pre_mask+1,context_length)
    if len(end_tokens)>0:
        strategy.set_end_tokens(end_tokens)
    
    tokens = tokens[..., :context_length+1]
    attention_mask = attention_mask.type_as(next(model.parameters())) # if fp16
    # initialize generation
    counter = context_length  # Last fixed index is ``counter''
    
    
    index = 0  # Next forward starting index, also the length of cache.
    # step-by-step generation

    ban_end=False
    strategy.st_pos=context_length
    fix_logit=None
    if start is not None:
        start='.'+start
        fix_logit=tokenizer.EncodeAsIds(start).tokenization
        #print(fix_logit)
        fix_logit=fix_logit[-1]
    context_length = 0
    with torch.no_grad():
        while counter < len(input_seq) - 1:
            
            log_attention_weights_part=torch.zeros(counter+1).cuda()   
            
            log_attention_weights_part[prefix_len+1:len_pre_title+1] = weight
            
           
            logits, *output = model(
                tokens[:, index:],
                position_ids[..., index: counter+1],
                attention_mask[..., index: counter+1, :counter+1], # TODO memlen
                mems=mems,
                log_attention_weights=log_attention_weights_part
            )
            mem_kv=[o['mem_kv'] for o in output]
            mems = update_mems( mem_kv,mems, max_memory_length=max_memory_length)
            #print(mems)
            counter += 1
            #print('counter:',counter)
            index = counter
            
            expansion_size=len(logits)
            
            # sampling
           
            if expansion_size>0:
                logits = logits[:, -1].expand(expansion_size, -1) # [batch size, vocab size]
                tokens = tokens.expand(expansion_size, -1)
                tokens, mems = strategy.forward(logits, tokens, mems,fixed_next=fix_logit)
                fix_logit=None
            if strategy.is_done:
                break
    del mems
    del logits
    torch.cuda.empty_cache()
    all_beams=strategy.finalize()
    
    best_score=-100000
    best_output=''
    for sentence in all_beams:
        beam_output= tokenizer.DecodeIds(sentence.cpu().tolist()) 
       
        
        
        score_title,score_emo,score_cor=compute_ip(model,beam_output,tokenizer)
        #print(beam_output,score_title,score_emo,score_cor)
        total_score=aggregate_scores(score_title,score_emo,score_cor)
        #print(total_score)
        if total_score>best_score:
            best_score=total_score
            best_output=beam_output
            print(beam_output.split(':')[-1],total_score)

    if excess_beam is not None:
        if not(isinstance(excess_beam,list)):
            excess_beam=[excess_beam]
        for st in excess_beam:
            beam_output=input_word+'<|startofpiece|>'+st
            score_title,score_emo,score_cor=compute_ip(model,beam_output,tokenizer)
            total_score=aggregate_scores(score_title,score_emo,score_cor)
            if total_score>best_score:
                best_score=total_score
                best_output=beam_output
    best_output=best_output.split('>')[-1]
    return best_output
    
    

    
if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)

    py_parser.add_argument('--sampling-strategy', type=str, default='BaseStrategy', help='type name of sampling strategy')

    # fix sat 3.0 argument issues by recalling previous arguments 
    py_parser.add_argument('--task-mask', type=bool, default=False, help='type name of sampling strategy')
    py_parser.add_argument('--block-mask-prob', type=float, default=0.0, help='type name of sampling strategy')
    py_parser.add_argument('--tokenizer-model-type', type=str, default='glm-10b', help='type name of sampling strategy')
    py_parser.add_argument('--inf-mode', type=str, default='continuous', help='type name of sampling strategy')
    
    known, args_list = py_parser.parse_known_args()
    print(args_list)
    args = get_args(args_list)
    import random
    args.seed=random.randint(0,1000000)
    print('args:',args)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    with torch.no_grad():
        main(args)
