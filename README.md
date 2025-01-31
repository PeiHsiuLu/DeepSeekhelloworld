# How to use DeepSeek with HuggingFace Transformer?
## Reference Video && DeepSeek Chat
[DeepSeek-R1 Crash Course](https://www.youtube.com/watch?v=_CXwZ5xyFno&t=3711s)
[DeepSeek](https://chat.deepseek.com/)
## Operaion Steps
OS Environment: Windows 10/11.
### 1. Create Environment on Visual Studio Code
Open your Visual Studio Code and connect to WSL, and then find the left sidebar to open your folder.

If you haven't create the WSL in your Windows, you can see this article which I have written before: [如何輕鬆解決 WSL 問題並重新安裝？](https://hackmd.io/TMXCFf3ASCiQlCFw72Detw?view)

![image](https://hackmd.io/_uploads/BkDw8Pdd1l.png)

Find your folder location and click "OK".
![image](https://hackmd.io/_uploads/HJXkvDuOJg.png)

Go to this github page: [GenAI-Essentials](https://github.com/ExamProCo/GenAI-Essentials/tree/main) and clone the web URL:
```bash=
https://github.com/ExamProCo/GenAI-Essentials.git
```

Input the command line in your terminal:
```wsl=
 git clone https://github.com/ExamProCo/GenAI-Essentials.git
```
And then you will see the "GenAI-Essentials" folder shown on the left sidebar.
![image](https://hackmd.io/_uploads/Sy1_YvuOke.png)

### 2. Create the python file (ipynb file)
Create the folder "deepseek" and the inner folder "r1-transformer", and then crete the ipynb file "deepseeker.ipynb" on the left sidebar. (The name can be created by what you want.)
![image](https://hackmd.io/_uploads/B1kKZqu_kl.png)

Open the ipynb file's integrated terminal.
![image](https://hackmd.io/_uploads/r1Olf5dOyx.png)

### 3. Create the Conda virtual environment

Operate the ipynb program in the conda environment. You can see the folloing article to know how to create the conda environment: [Install Conda in WSL (Guide to setting up Miniconda or Anaconda in WSL)](https://hackmd.io/@0weGKPooQo2ngm39PfuTAA/SJnXAsu_kg)

Now we can crete a conda new virtual environment "deepseeker", and then copy the following code: 
```bash=
conda create -n deepseeker python=3.10.16 -y
conda activate deepseeker
```
And then paste the following code in WSL:
```WSL=
conda install -c conda-forge ipykernel
```
Input "y" when you see this line, and it will install some package you need it.
![image](https://hackmd.io/_uploads/ByqGJ6_uJg.png)

### 4. Create transformer environment and jupyternotebook in WSL
You need to install the transformer package in your ipynb file before using the transformer code. 

And Browse marketplace for kernel extension (Jupyter Notebook)
![image](https://hackmd.io/_uploads/Bka-4TddJx.png)

Find Jupyter Notebook and install in WSL.
![image](https://hackmd.io/_uploads/HJks46_dye.png)

Select python kernel environment, and choose "deepseeker".
![image](https://hackmd.io/_uploads/ryr7HauO1e.png)

Now, you can execute the ipynb file line by line.

### 5. Install some package in your python code and WSL
You can reference the code in [the hugging face website(deepseek-r1)(click to navigate to the website)](https://huggingface.co/deepseek-ai/DeepSeek-R1)

Click the transformer you will see the reference code.
![image](https://hackmd.io/_uploads/Hy-Ud6ud1g.png)


Install the "transformer"
```python=
%pip install -q transformers python-dotenv
%pip install --upgrade transformers
```
When it completely executes, there will be a line of text, "Note: you may need to restart the kernel to use updated packages."

And follow the text hint, restart the kernel.

Next, you need to install pytorch, paste this line in your WSL:
```bash=
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install sympy=1.13.1
```


**Why use pytorch not tensorflow?**
It depends on python version. Since TensorFlow and PyTorch do not support all Python versions, you need to check if your Python version is compatible.

You can chack these websites to help you know whether your python version corresponds to your deep learning framework.

- [Pytorch wesite](https://pytorch.org/get-started/locally/)
- [Tensorflow website](https://www.tensorflow.org/install) 

If needed, you can install accelerate in your python virtual environment.
```python=
%pip install accelerate
```

You can install "ipywidgets" package and update it.
```python=
%pip install ipywidgets --upgrade
```

**Check GPU version**
You can see my article which I wrote before: [Check Your GPU in WSL and Use Python to Check It Out](https://hackmd.io/eb4gTtlbSwaPbIKsAFo9vA?view)


### 6. Create tokens and two additional files
Create two files on the left sidebar:
```file=
.gitignore
env.txt
```
![image](https://hackmd.io/_uploads/B1Dzpw5d1x.png)

Log in your huggingface account and click your profile picture and click "Access Tokens". 
![image](https://hackmd.io/_uploads/SJz8Aw9OJe.png)

After input your password, you will be navigated to the token page.
![image](https://hackmd.io/_uploads/By_aCwqdke.png)

Click "Create new token", and choose "read" function.
![image](https://hackmd.io/_uploads/rkkHyd5_yx.png)

After creating new token, copy and paste the tokens in the "env.txt" file.
```file name=
varible name = "tokens"
```
![image](https://hackmd.io/_uploads/HyEqkdq_1x.png)


### 7. Paste the python code in your ipynb file 
transformer:
```python=
# Use a pipeline as a high-level helper
from transformers import pipeline
```
dotenv:
```python=
from dotenv import load_dotenv
import os

#specify the path to env.txt
load_dotenv("env.txt")
```
example messages:
```python=
messages = [
    {"role": "user", "content": "Nice to meet you!"},
]
```
pipeline:
```python=
pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
```
pipe(messages):
```python=
pipe(messages)
```

## Conclusion
After Following the above steps, you will successfully use deepseek model in WSL!
