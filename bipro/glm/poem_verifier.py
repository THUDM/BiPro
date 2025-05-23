

import jsonlines
def cilin():
    f=open("cilin.txt",'r')
    t=f.readlines()
    bu=0
    nowsb=0
    allbu=[]
    def_chr=['[',']','(',')','\n',' ','，','\u3000','。','《','》']
    allsb=[[],[]]
    worddict={}
    shengdict={}
    for line in t:
        if len(line)<5:
            continue
        if ('第' in line) and ('部' in line):
            bu+=1
            allbu.append([])
        if ('平声' in line):
            nowsb=0
        if ('仄声' in line):
            nowsb=1
        if ('入声' in line):
            nowsb=1
            
        if ('【' in line):
            tks=line.split('】')[1]
            currentst1=0
            currentst2=0
            for num in range(len(tks)):
                char=tks[num]
                if currentst1+currentst2==0:
                    if not(char in def_chr):
                        allbu[-1].append(char)
                        allsb[nowsb].append(char)
                        if char in worddict:
                            if not(bu in worddict[char]):
                                worddict[char].append(bu)
                        else:
                            worddict[char]=[bu]
                        if char in shengdict:
                            if not(nowsb in shengdict[char]):
                                shengdict[char].append(nowsb)
                        else:
                            
                            shengdict[char]=[nowsb]
                            
                if char=='[':
                    currentst1=1
                if char==']':
                    currentst1=0
                if char=='(':
                    currentst2=1
                if char==')':
                    currentst2=0
    
    
    return worddict,shengdict,allbu,allsb
 
def pingshui():
    f=open("pingshui.txt",'r')
    t=f.readlines()
    f.close()
    bu=0
    nowsb=0
    allbu=[]
    def_chr=['[',']','(',')','\n',' ','，','\u3000','。','《','》']
    allsb=[[],[]]
    worddict={}
    shengdict={}
    for line in t:
        if len(line)<5:
            continue
        if "其它僻字" in line:
            continue
        words=line.strip()
        if ('\u3000' in line):
            bu+=1
            allbu.append([])
            if ('平声' in line):
                nowsb=0
            if ('上声' in line):
                nowsb=1
            if ('去声' in line):
                nowsb=1
            if ('入声' in line):
                nowsb=1
            words=words.split('\u3000')[1]

        
        currentst1=0
        currentst2=0
        for num in range(len(words)):
            char=words[num]
            if currentst1+currentst2==0:
                if not(char in def_chr):
                    allbu[-1].append(char)
                    allsb[nowsb].append(char)
                    if char in worddict:
                        if not(bu in worddict[char]):
                            worddict[char].append(bu)
                    else:
                        worddict[char]=[bu]
                    if char in shengdict:
                        if not(nowsb in shengdict[char]):
                            shengdict[char].append(nowsb)
                    else:    
                        shengdict[char]=[nowsb]
                            
            if char=='[':
                currentst1=1
            if char==']':
                currentst1=0
            if char=='(':
                currentst2=1
            if char==')':
                currentst2=0
    
    
    return worddict,shengdict,allbu,allsb

def cmudict():
    f=open("cmudict.txt",'r')
    t=f.readlines()
    f.close()
    for line in t:
        if len(line)>0:
            #
            if (line[0]>='A' and line[0]<='Z'):
                word,pronounce=line.split('  ')[0]
                pro=pronounce.split(' ')
                stresses=[]
                
    return worddict,rhymedict,stressdict
def get_last_piece(sentence):
    i=-2
   
    while (-i<len(sentence)) and not(sentence[i] in [',','，','.','。',':','：','?','？','!','！',';','；','>']):
        i-=1
    #print(sentence,sentence[i+1:])
    return sentence[i+1:]

def eng_verifier(sentence,verifier_params,print_reason=False):
    tokenizer,rhymedict,stressdict,current_rhyme=verifier_params[0]
    try:
        decode_tokens = tokenizer.DecodeIds(sentence.cpu().tolist())
        #print(decode_tokens)
    except:
        try:
            decode_tokens=sentence.replace('[gMASK]','').replace('[sMASK]','').replace('<|startofpiece|>','').replace('<|endofpiece|>','')
        except:
            print(decode_tokens)

    sentences=decode_tokens.split('\n')
    fullst=False
    if sentences[-1]=='\n':
        fullst=True
    # only verify last sentence
    j=len(sentences)-1
    while len(sentences[j].strip())<3:
        j-=1
    
    last_sentence=sentences[j]
    # only verify full words
    
    words=last_sentence.split(' ')
    num_stresses=0
    if not(fullst):
        words=words[:-1]
    
    for item in words:
        if not(item in stressdict):
            if print_reason:
                print(item," not in dict")
            return -1000
        for w in range(len(stressdict[item])):
            num_stresses+=1
            standard_stress=(1-num_stresses)%2 # da dum da dum 010101
            if stressdict[item][w]==2 and standard_stress==0:
                if print_reason:
                    print(item," stress not match")
                return -1000
            if stressdict[item][w]==0 and standard_stress==1:
                if print_reason:
                    print(item," stress not match")
                return -1000
            if num_stresses>10:
                if print_reason:
                    print(item," too many stress")
                return -1000
    if fullst:
        if num_stresses<10:
            if print_reason:
                print(item," too less stress")
            return -1000
        if rhymedict[item]!=rhymedict[current_rhyme]:
            if print_reason:
                print(item," rhyme not match")
            return -1000
    return 0
    
def code_verifier(sentence,verifier_params):
    tokenizer=verifier_params[0]
    decode_tokens = tokenizer.DecodeIds(sentence.cpu().tolist())
    if '<|end' in decode_tokens:
        return -1,decode_tokens.split('<|end')[0]
    lines=decode_tokens.split('\n')
    remaining_st=[]
    count=0
    for i in range(len(lines)):
        line=lines[i]
        
        if len(line)>0:
            if line[0]!=' ':
                count+=1
                if count>1:
                    return -1,remaining_st
        remaining_st=remaining_st+line+'\n'
        
    return 0,sentence


def poem_verifier(sentence,verifier_params,print_reason=False):
    
    tokenizer,wdic,shengdict,rhy,endrhy,min_length,max_length,end_tokens,yayun=verifier_params
    
    try:
        decode_tokens = tokenizer.DecodeIds(sentence.cpu().tolist())
        #if len(decode_tokens.split('<|')[0].split(':')[-1])<10:
        #    print("Evaluating ",decode_tokens)
    except:
        try:
            decode_tokens=sentence.replace('[gMASK]','').replace('[sMASK]','').replace('<|startofpiece|>','').replace('<|endofpiece|>','')
        except:
            print(decode_tokens)
    #for w in end_tokens:
    #    if w in decode_tokens:
    decode_token=get_last_piece(decode_tokens)
    
    #if "钟声对月" in decode_tokens:
    #   print(decode_tokens)
    #    print_reason=True
    icount=0
    
    if len(decode_token)==0:
        if print_reason:
            print(st,'异常')
        return -1000
        
    prev=decode_tokens[:-len(decode_token)]
    prev=prev.replace('：',':').split(":")[-1]
    st=decode_token
    for i in range(len(decode_token)-2):
        if decode_token[i:i+3] in prev:
            if print_reason:
                print(st,'重复')
            return -1000
   
    
    
    fullst=False
    with_=False
    if (len(st)>0 and (st[-1] in end_tokens)):
        st=st[:-1]
        fullst=True
        with_=True
        if len(st)<min_length:
            if print_reason:
                print(st,'过短')
            return -1000
        if len(st)==6:
            if print_reason:
                print(st,'字数不合')
            return -1000
        if decode_token[len(st)-1:] in prev:
            if print_reason:
                print(st,'重复')
            return -1000
    
    if len(st)>max_length:
        if print_reason:
            print(st,decode_tokens,end_tokens,'过长')
        return -1000
    if len(st)==max_length:
        fullst=True
        if st[-1]+'。' in prev:
            if print_reason:
                print(st,'重复')
            return -1000
        if st[-1]+'，' in prev:
            if print_reason:
                print(st,'重复')
            return -1000
        if st[-1]+'.' in prev:
            if print_reason:
                print(st,'重复')
            return -1000
        if st[-1]+',' in prev:
            if print_reason:
                print(st,'重复')
            return -1000
    for i in range(len(st)):
        if st[i] in prev:
            icount-=3
        if st[i] in st[:i]:
            icount-=3
        if i!=len(st)-1:
            if st[i:i+2] in prev:
                if print_reason:
                    print(st,'重复')
                return -1000
            if st[i:i+2] in st[:i]:
                if print_reason:
                    print(st,'重复')
                return -1000
        if st[i] in ['的','些','么','了']:
            if print_reason:
                print(st,'使用禁字')
            return -1000

                
                
    for i in st:
        if not(i in shengdict):
            if print_reason:
                print(st,i)
            return -1000
            
    if len(st)<2:
        return icount
    
    
    pz1=shengdict[st[1]]
    if rhy!=2:
        if len(pz1)==1:
            if rhy!=pz1[0]:
                if print_reason:
                    print(st,'第2字平仄')
                return -1000
    if rhy==2:
        if len(pz1)==1:
            rhy=pz1[0]
    #guping1
   
            # only apply for 
        
    if len(st)>=4:
        pz2=shengdict[st[3]]
        if rhy!=2:
            if len(pz2)==1:
                if pz2[0]+rhy!=1:
                    if print_reason:
                        print(st,'第4字平仄')
                    return -1000
        if rhy==2:
            if len(pz2)==1:
                rhy=1-pz2[0]

    #rhy: 0: 010  1:101  2: not decided
    #  endrhy:0/1, 2:undecided
  
    if len(st)>max_length-3:
    #regulate the 3rd word
        if endrhy!=2:
            
            wrhy=rhy
            if max_length==5:
                wrhy=1-rhy
            if endrhy==wrhy:
                pz=shengdict[st[max_length-3]]
                if len(pz)==1:
                    if pz[0]==wrhy:
                        if print_reason:
                            print(st,'三连')
                        return -1000
                        
    if len(st)>=6:
        
        pz3=shengdict[st[5]]
        if rhy!=2:
            if len(pz3)==1:
                if rhy!=pz3[0]:
                    if print_reason:
                        print(st,'第6字平仄')
                    return -1000
    
    if endrhy==0:
        #guping
        if rhy==0:
            if len(st)>=3:
                pz1=shengdict[st[0]]
                pz3=shengdict[st[2]]
                if len(pz1)+len(pz3)==2:
                    if pz1[0]+pz3[0]==2:
                        if print_reason:
                            print(st,'第2字孤平')
                        return -1000
        if rhy==1:
            if len(st)>=5:
                pz1=shengdict[st[2]]
                pz3=shengdict[st[4]]
                if len(pz1)+len(pz3)==2:
                    if pz1[0]+pz3[0]==2:
                        if print_reason:
                            print(st,'第4字孤平')
                        return -1000



    if fullst:
        if st[-1] in ['不']:
            return -1000
        pz11=shengdict[st[-3]]
        pz12=shengdict[st[-2]]
        pz13=shengdict[st[-1]]
        if len(pz11)+len(pz12)+len(pz13)==3:
            if pz11[0]+pz12[0]+pz13[0]==0:
                if print_reason:
                    print(st,'三连平')
                return -1000
            if pz11[0]+pz12[0]+pz13[0]==3:
                if print_reason:
                    print(st,'三连仄')
                return -1000
    
        if endrhy!=2:
            if len(pz13)==1:
                
                if endrhy!=pz13[0]:
                    if print_reason:
                        print(st,'仄起平收')
                    return -1000
        if endrhy==2:
            if len(pz13)==1:
                endrhy=pz13[0]
            
        if (len(yayun)>0 and endrhy==0):
            final1=wdic[st[-1]]
            final2=[]
            for i in yayun:
                final2.append(wdic[i])
            #print(st,yayun,final1,final2)
            doc=0
            for i in final1:
                doc=1
                for td in final2:
                    if not(i in td):
                        doc=0
                if doc==1:
                    break
            if doc==0:
                if print_reason:
                    print(st,'押韵')
                return -1000
        #print(decode_tokens)
        if with_:
            return 1000

    return 0
        
    
def verify_rhy(sentence,ids,shengdict,yayun,rhy):
    #print(sentence)
    sentence=sentence.replace('[gMASK]','')
    st=get_last_piece(sentence)
    
    st=st[:-1]
    length=len(st)
    pz=shengdict[st[-1]]
    needyy=0
    if (ids==0):
        if len(pz)==1:
            if pz[0]==0:
                needyy=1
    if ids%2==1:
        needyy=1
    if needyy==1:
        yayun=yayun+st[-1]
    
    #poem rhy shall refer to how the 1st sentence is behaving.
    if rhy!=2:
        return yayun,rhy,length

    #ids=0,3,4,7 are one group, 1,2,5,6 are another group.
    
    pz1=shengdict[st[1]]
    if len(pz1)==1:
        rhy=pz1[0]
        if ids in [1,2,5,6]:
            rhy=1-rhy
        return yayun,rhy,length
    pz2=shengdict[st[3]]
    if len(pz2)==1:
        rhy=1-pz2[0]
        if ids in [1,2,5,6]:
            rhy=1-rhy
        return yayun,rhy,length
    if len(st)>5:
        pz3=shengdict[st[5]]
        if len(pz3)==1:
            rhy=pz3[0]
        if ids in [1,2,5,6]:
            rhy=1-rhy

    return yayun,rhy,length
    
def get_pron():
    file='cipai.txt'
    f=open(file,'r')
    lines=f.readlines()
    cp={}
    alllist=[]
    for line in lines:
        linsp=line.split(':')
        if len(linsp)>1:
        #shuangdiao
            cp[linsp[0]]=linsp[1].replace('\n','')
            alllist.append(linsp[0])
    return cp,alllist
