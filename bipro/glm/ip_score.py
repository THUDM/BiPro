# -*- encoding: utf-8 -*-
'''
'''

# here put the import lib
from re import L
import torch
import torch.nn.functional as F
import numpy as np
from ppl import ppl

def get_correspond_sentence(current_sentence):
    #print('input of correspond:',current_sentence)
    if '[gMASK]' in current_sentence:

        wt=current_sentence.replace("[gMASK]",'').replace('<|startofpiece|>','').strip()
        pos=len(wt)-2
            
        while not(wt[pos] in [',','.','。','.','：',':']):
            pos-=1
        pos0=pos
        if wt[pos0] in ['：',':']:
            return None,None
        pos-=1
        while not(wt[pos] in [',','.','。','.','：',':']):
            pos-=1
        pos1=pos
        wt_new=wt[:pos1+1]+"[sMASK]"+wt[pos0+1:]
        target=wt[pos1+1:pos0+1]
        return wt_new,target
    # smask format: '诗歌《'+title+'》 '+'作者:'+author+' 体裁:'+format+emo_str+' 标题:'+title+'正文:  aaa [sMASK] bbb <|endofpiece|><|startofpiece|> 
    if '[sMASK]' in current_sentence:
        smask0,smask1=current_sentence.split('[sMASK]')
        smask1_sp=smask1.replace('<|endoftext|>','').split('<|startofpiece|>')
        wt=smask0+smask1_sp[1]+smask1_sp[0]
        if ('.' in smask1_sp[1]) or ('。' in smask1_sp[1]):
            # query the prior sentence
            pos0=len(smask0)
            pos=pos0-2
            while not(wt[pos] in [',','.','。','.','：',':']):
                pos-=1
            pos1=pos
            wt_new=wt[:pos1+1]+"[sMASK]"+wt[pos0:]
            target_new=wt[pos1+1:pos0]
            return wt_new,target_new
        else:
            # query the posterior sentence
            pos0=len(smask0)+len(smask1_sp[1])
            pos=pos0
            while not(wt[pos] in [',','.','。','.','：',':']) and pos<len(wt):
                pos+=1
            pos1=pos
            if pos1<len(wt)-1:
                wt_new=wt[:pos0]+"[sMASK]"+wt[pos1+1:]
            else:
                wt_new=wt[:pos0]+"[gMASK]"
            target_new=wt[pos0:pos1+1]
            return wt_new,target_new
    return None,None



def compute_ip(model,answer,tokenizer,enhanced_start=None,enhance_end=None,weight=0.5,fix_st=0):
    #compute inverse prompting score. Code written in a plain way.
    
    # due to [sMASK] shall match for all beams, it seems that plain multi-beam is impossible for GLM-10B. 
    cls=tokenizer.get_command('ENC').Id
    smask=tokenizer.get_command('sMASK').Id
    eos=tokenizer.get_command('eos').Id
    sop=tokenizer.get_command('sop').Id
    # format of answers
    
    inv_titles=[]
    target_titles=[]
    inv_emos=[]
    target_emos=[]
    inv_prev=[]
    target_prev=[]
    has_emo=True
    has_prev=True
   
        # format='诗歌《'+title+'》 '+'作者:'+author+' 体裁：'+format+emo_str+' 标题：'+title+'正文：[gMASK]'

    new_answer=answer

    wt=new_answer.split('》')[-1].replace("[gMASK]",'').replace('<|startofpiece|>','')
    target_title=wt.split('标题:')[1].split('正文')[0]
    inv_title=wt.split('标题')[0]+'标题：[sMASK] 正文'+wt.split('正文')[1]
        
    inv_emo=None
    target_emo=None
    if '情感:' in wt:
        inv_emo=wt.split('情感')[0]+'情感：[sMASK] 标题'+wt.split('标题')[1]
        target_emo=wt.split('情感:')[1].split('标题')[0]
            
    inv_prev_sentence,target_prev_sentence=get_correspond_sentence(new_answer)

    inv_title_seq = [cls]+tokenizer.EncodeAsIds(inv_title).tokenization+[eos]
        
    target_title_seq=tokenizer.EncodeAsIds(target_title).tokenization+[eos]
    inv_titles.append(inv_title_seq)
    target_titles.append(target_title_seq)
    if inv_emo!=None:
        inv_emo_seq = [cls]+tokenizer.EncodeAsIds(inv_emo).tokenization+[eos]
        target_emo_seq=tokenizer.EncodeAsIds(target_emo).tokenization+[eos]
        inv_titles.append(inv_emo_seq)
        target_titles.append(target_emo_seq)
    else:
        has_emo=False
    if inv_prev_sentence!=None:
        #print('prompt:',inv_prev_sentence,' target:',target_prev_sentence)
        inv_prev_seq = [cls]+tokenizer.EncodeAsIds(inv_prev_sentence).tokenization
        if "[sMASK]" in inv_prev_sentence:
            inv_prev_seq=inv_prev_seq+[eos]
        target_prev_seq=tokenizer.EncodeAsIds(target_prev_sentence).tokenization+[eos]
        inv_prev.append(inv_prev_seq)
        target_prev.append(target_prev_seq)
    else:
        has_prev=False
    
    title_maskp=inv_titles[0].index(smask)

    ppl_title=ppl(model,inv_titles,target_titles,title_maskp,sop=sop)
    #penalty score for using the same char as title
    nn=0
    for ch in target_title:
        if ch in inv_title:
            nn+=1
    ppl_title-=nn*nn

    ppl_emo=None
    if has_emo:
        emo_maskp=inv_emo[0].index(smask)
        ppl_emo=ppl(model,inv_emo,target_emo,emo_maskp,sop=sop)
    ppl_prev=None
    if has_prev:
        if smask in inv_prev[0]:
            prev_maskp=inv_prev[0].index(smask)
        else:
            prev_maskp=len(inv_prev[0])-1
        ppl_prev=ppl(model,inv_prev,target_prev,prev_maskp,sop=sop)
    
    return ppl_title,ppl_emo,ppl_prev
      

def compute_p(model,sop_pos,beam,tokenizer=None,sop=50006,ini_pos=None,start_pos=None,end_pos=None,weight=0.5):
    new_beams=[]
    targets=[]
    for i  in range(len(beam)):
        bm=beam[i].cpu().tolist()
        new_bm=bm[:sop_pos]
        new_tag=bm[sop_pos+1:]
        new_beams.append(new_bm)
        targets.append(new_tag)

    counter=len(new_beams[0])+len(targets[0])
    log_attention_weights_part=None
    if ini_pos is not None:
        log_attention_weights_part=torch.zeros(counter+1).cuda()
        log_attention_weights_part[start_pos:end_pos] = weight
    #if tokenizer is not None:
    #    print(ini_pos,start_pos,end_pos,new_beams[0],new_beams[0][1:ini_pos],new_beams[0][start_pos:end_pos],tokenizer.DecodeIds(new_beams[0][1:ini_pos]),tokenizer.DecodeIds(new_beams[0][start_pos:end_pos]))
    return ppl(model,new_beams,targets,sop_pos,sop=sop,additional_attention=log_attention_weights_part)

