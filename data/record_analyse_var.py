import os
import json
import copy
import numpy as np
q=os.listdir("app_configs")
analysis_term='202401'
order_distribute={}
all_ais=[]
for item in q:
    tl=json.load(open("app_configs/"+item,'r',encoding='utf-8'))
    if analysis_term in tl["app_name"]:
        for i in range(len(tl["tasks"])):
            for j in range(len(tl["tasks"][i]["pages"])):

                grps=tl["tasks"][i]["pages"][j]["groups"]
                for k in range(len(grps)):
                    order_distribute[i*1000+j*10+k]=grps[k]["tags"]
                    if not(grps[k]['tags']) in all_ais:
                        all_ais.append(grps[k]['tags'])
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
'''
the [0,4] poem "山行" is controversial in scoring as direct generation generates a poem exactly the same as a famous human
ancient poem with the same title, removed. 
'''
#removed_pairs=[[0,4]]
sumscore_for_ai={}
all_avg_score={}
ppl_scores=[]
for i in range(len(people_answers)):
    ppl_scores.append({})

for i in range(6):
    for j in range(7):
        if not([i,j] in  removed_pairs):
            for k in range(6):
                ai=order_distribute[i*1000+j*10+k][0]
                sum_score=np.zeros(6)
                for w in range(len(people_answers)):
                    ppl=people_answers[w]
                    score=ppl["tasks"][i]["pages"][j]["groups"][k]["answers"]
                    sum_score=sum_score+np.array(score)
                    if not(ai in ppl_scores[w]):
                        ppl_scores[w][ai]=np.array(score)
                    else:
                        ppl_scores[w][ai]+=np.array(score)

                avg_score=sum_score/len(people_answers)
                all_avg_score[i*1000+j*10+k]=avg_score[4]
                #print(ai)

                if ai in sumscore_for_ai:
                    sumscore_for_ai[ai]+=avg_score/42
                else:
                    sumscore_for_ai[ai]=avg_score/42

print(sumscore_for_ai)
print(ppl_scores)
print(ppl_scores[0]['gpt4'][0])
print(all_ais)
for ai in all_ais:
    for i in range(5):
        listin=[]
        for j in range(len(people_answers)):
            listin.append(ppl_scores[j][ai[0]][i]/42)
        var=np.std(listin)
        print(ai[0],i,var)
human_performance={}
sumscore_second={}
added_items=0
iter_list=[]

  