# BiPro
``Bipro`` directory contains code for bipro using GLM-10B-Chinese.
GLM-10B-Chinese model download:
``https://github.com/THUDM/GLM``

Packages required:
``SwissArmyTransformer``

Please change the ``WEIGHT_PATH`` in ``config/glm_10b_chinese.sh `` to your own weight path.

Script for Bipro Generation:
``bash run_bipro_poem.sh config/glm_10b_chinese.sh $GPU  ``  
(will start generating poems according to titles in ``titles2.txt``)

Script for Direct Generation:
``bash run_direct_poem.sh config/glm_10b_chinese.sh $GPU ``

``data`` directory contains data and analysis code for 2 human experiments.

``python record_analyse_challenge1.py`` analyses the result for challenge 1.
``python record_analyse_challenge2.py`` analyses the result for challenge 2.

``poem_gpt4.ipynb`` is the code for getting few shot GPT-4 generations through GPT-4 API. Results of other baselines are achieved manually through webpage.

Online Inference:
https://chatglm.cn/main/gdetail/672c837c8ba8cf3453de646c?lang=zh

This paper is accepted to ACL 2025 main. 

