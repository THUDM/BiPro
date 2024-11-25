import os
import json
import copy
import numpy as np
q=os.listdir("app_configs")
analysis_term='202401'
order_distribute={}
for item in q:
    tl=json.load(open("app_configs/"+item,'r',encoding='utf-8'))
    if analysis_term in tl["app_name"]:
        for i in range(len(tl["tasks"])):
            for j in range(len(tl["tasks"][i]["pages"])):

                grps=tl["tasks"][i]["pages"][j]["groups"]
                for k in range(len(grps)):
                    order_distribute[i*1000+j*10+k]=grps[k]["tags"]

w=os.listdir("app_records")

people_answers=[]
for item in w:
    addin=False
    tl=json.load(open("app_records/"+item,'r',encoding='utf-8'))
    if analysis_term in tl["app_name"]:
        addin=True
        for i in range(len(tl["tasks"])):
            for j in range(len(tl["tasks"][i]["pages"])):

                grps=tl["tasks"][i]["pages"][j]["groups"]
                for k in range(len(grps)):
                    answer=grps[k]["answers"]
                    for itm in answer:
                        if itm is None:
                            addin=False
    if addin:
        people_answers.append(copy.deepcopy(tl))
        #print(tl)
removed_pairs=[]
#removed_pairs=[[0,4]]
sumscore_for_ai={}
all_avg_score={}
for i in range(6):
    for j in range(7):
        if not([i,j] in  removed_pairs):
            for k in range(6):
                ai=order_distribute[i*1000+j*10+k][0]
                sum_score=np.zeros(6)
                for ppl in people_answers:
                    score=ppl["tasks"][i]["pages"][j]["groups"][k]["answers"]
                    sum_score=sum_score+np.array(score)
                avg_score=sum_score/len(people_answers)
                all_avg_score[i*1000+j*10+k]=avg_score[4]
                #print(ai)

                if ai in sumscore_for_ai:
                    sumscore_for_ai[ai]+=avg_score/42
                else:
                    sumscore_for_ai[ai]=avg_score/42

print(sumscore_for_ai)
human_performance={}
sumscore_second={}
added_items=0
iter_list=[]

    
player_score={}
for i in range(6):
    for j in range(7):
        if not([i,j] in  removed_pairs):
            added_items+=1
            for k in range(6):
                

                ai=order_distribute[i*1000+j*10+k][0]

               
                # iterate 
                all_scores=np.zeros((11,11))
                for ppl in people_answers:
                    score=ppl["tasks"][i]["pages"][j]["groups"][k]["answers"]
                    all_scores[int(score[4]+0.5),int(score[5]+0.5)]+=1
                point=0
                current_best_score=0
                best_candidates=[]
                #print(all_scores)
                for w in range(1,10):
                    for bonus in [0.25,0.75]:
                        s=w+bonus
                        alls=0
                        act=False
                        
                        for i2 in range(1,11):
                            for j2 in range(1,11):
                                if all_scores[i2,j2]>0.5:
                                    if np.abs(i2-s)<=np.abs(j2-s):
                                        if np.abs(i2-s)<3:
                                            alls+=all_scores[i2,j2]*all_scores[i2,j2]
                                            if np.abs(i2-s)<1:
                                                #print(s,i2,j2)
                                                act=True
                        if act:
                            if alls==current_best_score:
                                best_candidates.append(s)
                            if alls>current_best_score:
                                current_best_score=alls
                                best_candidates=[s]
                        
                if len(best_candidates)<20:
                    avg_score=np.average(best_candidates)
                else:
                    avg_score=all_avg_score[i*1000+j*10+k]
                    
                print(i,j,k,best_candidates,avg_score)
                for ppl in people_answers:
                    score=ppl["tasks"][i]["pages"][j]["groups"][k]["answers"][4]
                    for w in best_candidates:
                        if np.abs(w-score)<0.5:
                            if not(ppl['username'] in player_score):
                                player_score[ppl["username"]]=0
                            player_score[ppl["username"]]+=1
                if ai in sumscore_second:
                    sumscore_second[ai]+=avg_score/42
                else:
                    sumscore_second[ai]=avg_score/42
print(added_items)
print(sumscore_second,human_performance)
print(player_score)
