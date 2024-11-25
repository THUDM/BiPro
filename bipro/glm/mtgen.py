# -*- encoding: utf-8 -*-
'''
'''

# here put the import lib
from re import L
import torch
import torch.nn.functional as F
import numpy as np
from ppl import ppl
import copy

class MultiGenStrategy:
    def __init__(self, num_beams, length_penalty=1.,
                end_tokens=[], invalid_slices=[],verifier=None,verifier_params=None,tmp_factor=1):
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.end_tokens = end_tokens
        self.verifier=verifier
        self.verifier_params=verifier_params
        self.num_samples=num_beams*3
        self.tmp_factor=tmp_factor
        self.max_set=50000
        self.fix_st=0
        self._init_cache()
     

    def set_verifier_params(self,params):
        self.verifier_params=params
        
    def set_verifier_param(self,param_idx,idx):
        self.verifier_params[idx]=param_idx
    
    def set_end_tokens(self,end_tokens):
        self.end_tokens=end_tokens
    
 
    def adjust_start(self,length):
        self.start_pos=length
        
    def set_start(self,start):
        self.iprompt_start_pos=start
    def set_end(self,end):
        self.iprompt_end_pos=end
    def set_gen_pos(self,pos):
        self.gen_pos=pos
    def set_sop_pos(self,pos):
        self.sop_pos=pos
    def set_ini_pos(self,pos):
        self.ini_pos=pos
    def set_model(self,model):
        self.model=model
    def get_pos(self):
        return self.ini_pos,self.iprompt_start_pos,self.iprompt_end_pos
    def _init_cache(self):
        
        self.end_beams = [] # list of LongTensors
        self.is_done = False
        torch.cuda.empty_cache()
        
    
    def _add_end_beams(self, beam):
        
        verify_score=self.verifier(beam,self.verifier_params)
        #print(self.end_beams,verify_score)
        if verify_score>-100: 
           
            self.end_beams.append(beam)  
             
            if len(self.end_beams)>self.num_samples:
                self.is_done=True
            
    def forward(self, logits, tokens, mems,fixed_next=None):
        batch_size, vocab_size = logits.shape
        #print(batch_size)
        seq_len = tokens.shape[-1]
        logits = logits.float()
        
        for i in range(batch_size):
            for w in range(len(tokens)):
                logits[i,tokens[w]]-=self.tmp_factor
        
        next_token_scores = F.log_softmax(logits, dim=-1) # [batch_size, vocab_size]
       
        scores_for_sample=logits*self.tmp_factor
        
        next_token_scores = next_token_scores.view(batch_size * vocab_size)
        
        scores_for_sample=scores_for_sample.view(batch_size*vocab_size)
        #emphasis on last word
        total_added=0
        times=0
        beam_continue = []
        scores_continue = []
        bans_continue = []
        mems_continue = []
        verify_scores=[]
       


        while (total_added<self.num_beams):
            
            probs = F.softmax(scores_for_sample, dim=0)
            if (times>2 and len(self.end_beams)>0):
                self.is_done=True
                break
            if (times>5 and total_added>0):
                break
            if (times>0 and fixed_next is not None):
                break
            next_tokens = torch.multinomial(probs,
                    num_samples=self.num_samples*2) # [2*nb]
            
            if fixed_next is not None:
                next_tokens=torch.LongTensor(np.zeros(batch_size)).cuda()

            
            
            
            if (times==0) and not(fixed_next):
                
                for i in range(batch_size):
                    if 43361 in self.end_tokens:
                        next_tokens[-i*2-1]=vocab_size*i+43361
                    if 43359 in self.end_tokens:
                        next_tokens[-i*2-2]=vocab_size*i+43359
                        
            scores_for_sample[next_tokens]-=1000
            if fixed_next is not None:
                for i in range(batch_size):
                    next_tokens[i]=i*vocab_size+fixed_next
            times+=1
            sampled_scores = next_token_scores[next_tokens]
            

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode='trunc')
            next_tokens = next_tokens % vocab_size

                # select out end beams or continue beams
            if mems.shape[1] < batch_size:
                mems = mems.expand(-1, batch_size, -1, -1)
            #print(tokens,next_tokens)
            for i in range(len(next_tokens)):
                if self.is_done:
                    continue
                if total_added>=self.num_samples:
                    break
                if next_tokens[i]>=self.max_set:
                    continue
                if next_tokens[i]>=self.max_set+9:
                    continue
                beam = torch.cat((tokens[next_indices[i]], next_tokens[i:i+1]))
                #print(beam,vocab_size,logits.shape)
                if self.verifier is not None:
                    verify_score=self.verifier(beam,self.verifier_params)
                else:
                    verify_score=0
                    
                if verify_score<-100:
                    continue
                    
                total_added+=1
                #print("total_added:",total_added)
                
                if (int(next_tokens[i]) in self.end_tokens) or (verify_score>100) :
                    self._add_end_beams(beam)
                elif len(beam_continue) < self.num_samples:
                    beam_continue.append(beam)
                    mems_continue.append(mems[:, next_indices[i]])
                else:
                    break
        #print(total_added,len(beam_continue),self.num_beams)
        
        if len(beam_continue)>0:
            tokens = torch.stack(beam_continue)
            mems = torch.stack(mems_continue, dim=1)
        else:
            self.is_done=True
        del mems_continue
        torch.cuda.empty_cache()
        # TODO is_done
        return tokens, mems

    def finalize(self):
        ret=self.end_beams
        self._init_cache()
        return ret

