import torch
import random
from utils import get_one_hop_neighbors, get_top_k_neighbor_simcse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import torch
from conversation import conv_v1,conv_v2,conv_v3
import utils
import json
from call_api import call_api
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

def prompt_cluster_sum(data_obj,confi_mask,dataset_name="cora",memory_limit = 10000000):
    if len(data_obj.raw_texts) < memory_limit:
        raw_texts = np.array(data_obj.raw_texts)
    else:
        raw_texts = data_obj.raw_texts
    prompts = []
    typ = "diabetes" if dataset_name == "pubmed" else "computer science"
    for key in confi_mask.keys():
        prompt = f"Below are the high-confidence samples related to {typ} from a specific cluster. Analyze the commonalities and core content of these samples and provide a concise summary of the cluster's theme. Output the theme as a short name without adding any extra explanations."
        prompt += f"\n The cluster has the following high confidence samples:\n"
        prompt += f"High confidence sample in cluster{int(key)+1}:"+" {{\n"

        for idx in confi_mask[key]:
            prompt += f"{raw_texts[idx]}"+"}}\n"
        prompt += "Please comprehensively consider the commonalities between these samples and then conclude the topic of the cluster concisely. Output the newly generated topic name as a dictionary, with key \"answer\" and \"reason\"."
        prompts.append(prompt)
    return prompts

            
def prompt_neighbor_generate(data_obj,sampled_node_idxs,g_feat,topic_clusters,hop=2,sample_num=2,dataset_name="cora",memory_limit = 10000000):
    if len(data_obj.raw_texts) < memory_limit:
        raw_texts = np.array(data_obj.raw_texts)
    else:
        raw_texts = data_obj.raw_texts
    prompts = []    
    neighbor_dict = get_top_k_neighbor_simcse(data_obj,sampled_node_idxs,g_feat,sample_num,hop)
    typ = "diabetes" if dataset_name == "pubmed" else "computer science"
    
    for idx,sam_idx in enumerate(sampled_node_idxs):           
        #form = infer_text_form(raw_texts[sam_idx])    
        prompt = f"Given a target article related to {typ}: \n {raw_texts[sam_idx]}."
        prompt += f"\n The topic of this article may fall under one of the following clusters: {topic_clusters}."
        if len(neighbor_dict[sam_idx])>0:
            prompt += f"\n It has following important neighbors which has citation relationship to this {typ}, from most related to least related:\n"
            prompt += f"Neighbors of {typ} {idx}:"+" {\n"
            for nei_idx,nei in enumerate(neighbor_dict[sam_idx]):
                prompt += f"{raw_texts[nei]}"+"}\n"                
                
            if dataset_name == "wikics":
                prompt += f"Please consider the information from the target article and its neighbors, and generate a novel article similar to the given one, with no more than 350 words. Output the newly generated article as a dictionary, with key \"answer\" and \"reason\"."
            else:
                prompt += f"Please consider the information from the target article and its neighbors, and generate a novel article similar to the given one, with a title of no more than 15 words and an abstract limited to 300 words.Output the newly generated article as a dictionary, with key \"answer\" and \"reason\"."                
        else:
            if dataset_name == "wikics":
                prompt += f"Please consider the information from the target article, and generate a novel article similar to the given one, with no more than 350 words. Output the newly generated article as a dictionary, with key \"answer\" and \"reason\"." 
            else:
                prompt += f"Please consider the information from the target article, and generate a novel article similar to the given one, with a title of no more than 15 words and an abstract limited to 300 words. Output the newly generated article as a dictionary, with key \"answer\" and \"reason\"."                                 
        prompts.append(prompt)
    return prompts


def prompt_aug_classifier(data_obj, sampled_node_idxs, aug_node_texts, topic_clusters, dataset_name="cora", memory_limit = 1000000):
    if len(data_obj.raw_texts) < memory_limit:
        raw_texts = np.array(data_obj.raw_texts)
    else:
        raw_texts = data_obj.raw_texts
    prompts1, prompts2 = [], []
    n_clusters = data_obj.y.max().item()+1
    typ = "diabetes" if dataset_name == "pubmed" else "computer science"
    #anchor nodes
    for idx,sam_idx in enumerate(sampled_node_idxs):
        prompt = f"Given a target article related to {typ}: \n {raw_texts[sam_idx]}."
        prompt += f"\n Please determine which cluster this {typ} most likely belongs to only from {n_clusters} clusters.\n These optional clusters are: "
        contexts = []
        for num_cluster in range(n_clusters):
            this_context = {f"Cluster {num_cluster+1}": topic_clusters[num_cluster]}
            contexts.append(this_context)
        prompt += str(contexts)
        prompt += f"\n Please comprehensively consider which cluster this article most likely belongs to, only answer the cluster number directly as a dictionary, with key \"answer\" and \"reason\"."
        prompt += f"\n The cluster number should be a interger and range from 1 to {n_clusters}."
        prompts1.append(prompt)

    #augmentation nodes    
    for text in aug_node_texts:
        prompt = f"Given a target article related to {typ}: \n {text}."
        prompt += f"\n Please determine which cluster this article most likely belongs to only from {n_clusters} clusters.\n These optional clusters are: "
        contexts = []
        for num_cluster in range(n_clusters):
            this_context = {f"Cluster {num_cluster+1}": topic_clusters[num_cluster]}
            contexts.append(this_context)
        prompt += str(contexts)
        prompt += f"\n Please comprehensively consider which cluster this article most likely belongs to, only answer the cluster number directly as a dictionary, with key \"answer\" and \"reason\"."
        prompt += f"\n The cluster number should be a interger and range from 1 to {n_clusters}."
        prompts2.append(prompt)
        
    return prompts1,prompts2

    #return prompts1
        
def prompt_aug_classifier_equ(data_obj,sampled_node_idxs,aug_node_texts,topic_clusters,dataset_name="cora",memory_limit = 1000000):
    if len(data_obj.raw_texts) < memory_limit:
        raw_texts = np.array(data_obj.raw_texts)
    else:
        raw_texts = data_obj.raw_texts
    prompts = []
    n_clusters = data_obj.y.max().item()+1
    typ = "diabetes" if dataset_name == "pubmed" else "computer science"
    for idx,sam_idx in enumerate(sampled_node_idxs):
        prompt = f"Given two target articles related to {typ}.\n The first article is:{raw_texts[sam_idx]}.\n The second article is {aug_node_texts[idx]}."
        prompt += f"\n The topic of these articles may fall under one of the following clusters: {topic_clusters}."
        prompt += f"\n Please comprehensively consider the two articles above and the cluster they are most likely to belong to separately, and then determine whether they belong to the same cluster. Output 1 if they belong to the same cluster, or 0 if they belong to different clusters. Please answer as a dictionary, with key \"answer\" and \"reason\"."
        prompts.append(prompt)
    return prompts    


def efficient_gpt_text_ind(input_text):    
    gpt_result = []

    prompts = []
    for idx, text in enumerate(input_text):
        conv = conv_v1.copy()
        conv.append_message(conv.roles[0], text)
        final_prompt = conv.get_prompt()
        prompts.append({'idx': idx, "prompt": final_prompt})

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(call_api, prompt, temperature=0, max_tokens=300): prompt['idx'] for prompt in prompts}
        
        for future in tqdm(as_completed(futures), total=len(prompts)):
            idx = futures[future]
            try:
                result = future.result()
                gpt_result.append([result, idx])
            except Exception as e:
                print(f"Error occurred for prompt index {idx}: {e}")
    
    gpt_result = sorted(gpt_result, key=lambda x: x[-1])
    final_result = []
    print(gpt_result)
    for i in range(len(gpt_result)):
        result = None
        try:
            data = json.loads(gpt_result[i][0])
            result = data.get("answer")
        except:
            try:
                match = re.search(r'"answer"\s*:\s*"(.*?)"', gpt_result[i][0], re.DOTALL)
                if match:
                    result = match.group(1)
            except ValueError:
                pass
        if result is None:
            final_result.append("Error occured")
        else:
            final_result.append(str(result))
        
    return final_result, gpt_result

def efficient_gpt_text_ge(input_text):    
    gpt_result = []

    prompts = []
    for idx, text in enumerate(input_text):
        conv = conv_v2.copy()
        conv.append_message(conv.roles[0], text)
        final_prompt = conv.get_prompt()
        prompts.append({'idx': idx, "prompt": final_prompt})

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(call_api, prompt, temperature=0, max_tokens=1000): prompt['idx'] for prompt in prompts}
        
        for future in tqdm(as_completed(futures), total=len(prompts)):
            idx = futures[future] 
            try:
                result = future.result() 
                gpt_result.append([result, idx]) 
            except Exception as e:
                print(f"Error occurred for prompt index {idx}: {e}")

    #breakpoint()
    gpt_result = sorted(gpt_result, key=lambda x: x[-1])
    #print(gpt_result)
    final_result = []
    for i in range(len(gpt_result)):
        result = None
        try:
            data = json.loads(gpt_result[i][0])
            result = data.get("answer")
        except:
            try:
                match = re.search(r'"answer":\s*(\{.*?\})', gpt_result[i][0],re.DOTALL)
                #print(match)
                if match:
                    result = match.group(1)
            except ValueError:
                pass
        if result is None:
            final_result.append("Error occured")
        else:
            final_result.append(str(result))
            
    return final_result, gpt_result

def efficient_gpt_text_cls(input_text, n_clusters):
    gpt_result = []
    prompts = []
    classes = np.arange(n_clusters)
    probs = [1 / len(classes)] * len(classes)

    for idx, text in enumerate(input_text):
        conv = conv_v3.copy()
        conv.append_message(conv.roles[0], text)
        final_prompt = conv.get_prompt()
        prompts.append({'idx': idx, "prompt": final_prompt})

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(call_api, prompt, temperature=0, max_tokens=100): prompt['idx'] for prompt in prompts}
        
        for future in tqdm(as_completed(futures), total=len(prompts)):
            idx = futures[future]
            try:
                result = future.result()
                gpt_result.append([result, idx])
            except Exception as e:
                print(f"Error occurred for prompt index {idx}: {e}")
                gpt_result.append([{"answer":str(random.choices(classes, probs)[0]+1),"reason":""}, idx])

    gpt_result = sorted(gpt_result, key=lambda x: x[-1]) 
    result_cls = []
    for i in range(len(gpt_result)):
        result = None
        try:
            match = re.search(r'"answer"\s*:\s*"?(\d+)"?', gpt_result[i][0])
            if match:
                result = int(match.group(1))
        except ValueError:
            pass
        if result is None:
            result = random.choices(classes, probs)[0] + 1
        if int(result - 1) in range(n_clusters):
            result_cls.append(int(result) - 1)
        else:
            result_cls.append(random.choices(classes, probs)[0])

    return result_cls, gpt_result

def efficient_gpt_text_cls_equ(input_text):
    gpt_result = []
    prompts = []

    for idx, text in enumerate(input_text):
        conv = conv_v3.copy()
        conv.append_message(conv.roles[0], text)
        final_prompt = conv.get_prompt()
        prompts.append({'idx': idx, "prompt": final_prompt})
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(call_api, prompt, temperature=0, max_tokens=100): prompt['idx'] for prompt in prompts}
        
        for future in tqdm(as_completed(futures), total=len(prompts)):
            idx = futures[future]
            try:
                result = future.result() 
                gpt_result.append((result, idx))
            except Exception as e:
                print(f"Error occurred for prompt index {idx}: {e}")
                gpt_result.append([{"answer":str(random.choice([0, 1])),"reason":""}, idx]) 
    gpt_result = sorted(gpt_result, key=lambda x: x[-1]) 
    result_cls = []
    for i in range(len(gpt_result)):
        try:
            match = re.search(r'"answer"\s*:\s*"?(\d+)"?', gpt_result[i][0])
            if match:
                result = int(match.group(1))
        except ValueError:
            pass                
        if result is None:
            result = random.choice([0,1])
        if result in [0,1]:
            result_cls.append(result)
        else:
            result_cls.append(random.choice([0,1]))
    #breakpoint()
    return result_cls, gpt_result

def prompt_similar_generate(data_obj, sampled_node_idxs, dataset_name="cora", memory_limit=10000000):
    if len(data_obj.raw_texts) < memory_limit:
        raw_texts = np.array(data_obj.raw_texts)
    else:
        raw_texts = data_obj.raw_texts

    prompts = []
    typ = "product" if dataset_name == "product" else "paper"

    for idx, sam_idx in enumerate(sampled_node_idxs):
        text = raw_texts[sam_idx]
        form = infer_text_form(text)
        if form == "title":
            prompt = f"Given {typ} title: '{text}', please generate a novel {typ} title similar to the given title."        
        elif form == "abstract":
            prompt = f"Given {typ} abstract: '{text}', please generate a novel {typ} abstract similar to the given abstract."
            
        elif form == "title_abstract":
            title, abstract = text.split(":", 1)
            title = title.strip()
            abstract = abstract.strip()
            prompt = f"Given {typ} title: '{title}' and abstract: '{abstract}', please generate a novel {typ} title and abstract similar to the given ones."
        
        else:
            prompt = f"Given {typ} main text: '{text}', please generate a novel {typ} main text similar to the given text."
        prompts.append(prompt)

    return prompts

'''
def prompt_aug_classfier(data_obj,sampled_node_idxs,aug_node_texts,confi_node_idxs,dataset_name="cora",memory_limit = 1000000):
    if len(data_obj.raw_texts) < memory_limit:
        raw_texts = np.array(data_obj.raw_texts)
    else:
        raw_texts = data_obj.raw_texts
    prompts1,prompts2 = [],[]
    n_clusters = data_obj.y.max().item()+1
    typ = "product" if dataset_name == "product" else "paper"
    for idx,sam_idx in enumerate(sampled_node_idxs):
        prompt = f"Given {typ} description: {raw_texts[sam_idx]} and there exists {n_clusters} clusters."
        prompt += f"Below are typical samples for each cluster:\n. Please determine which cluster this {typ} most likely belongs to only from {n_clusters} clusters."
        contexts = []
        for num_cluster in range(n_clusters):
            this_context = {f"Cluster {num_cluster+1}": raw_texts[confi_node_idxs[num_cluster]]}
            contexts.append(this_context)
        prompt += str(contexts)
                                
        #prompt += f"Please determine which cluster this {typ} most likely belongs to, only output the cluster number as a single integer(e.g.,0,1,2)."
        prompt += f"Please determine which cluster this {typ} most likely belongs to, only output the cluster number as a single integer(e.g.,1,2)."
        prompt += f"\n The cluster number should be a interger and range from 1 and {n_clusters}."
        prompts1.append(prompt)
        
        
    for idx in range(len(aug_node_texts)):
        prompt = f"Given {typ} description: {aug_node_texts[idx]} and there exists {n_clusters} clusters."
        prompt += f"Below are typical samples for each cluster:\n. Please determine which cluster this {typ} most likely belongs to only from {n_clusters} clusters."
        contexts = []
        for num_cluster in range(n_clusters):
            this_context = {f"Cluster {num_cluster+1}": raw_texts[confi_node_idxs[num_cluster]]}
            contexts.append(this_context)
        prompt += str(contexts)
        prompt += f"Please determine which cluster this {typ} most likely belongs to, only output the cluster number as a single integer(e.g.,1,2)."
        prompt += f"\n The cluster number should be a interger and range from 1 and {n_clusters}."
        prompts2.append(prompt)
    return prompts1,prompts2 #unconfident,augmentation                           
'''
def parse_llama_output(output_text):
    import re
    match = re.search(r'\b\d+\b', output_text)
    if match:
        return int(match.group(0))
    else:
        return -1

@torch.inference_mode()
def efficient_llama_text_cls(input_text,device,model_name="meta-llama/Llama-3.1-70B-Instruct/", rewrite=True):
    tokenizer, model = load_llama_model(model_name,device)
    llama_result = []
    for idx,text in enumerate(input_text):
        conv = conv_v1.copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()
        result = llama_generate(final_prompt,tokenizer,model,device,100)
        #print(result)
        result = parse_llama_output(result)        
        llama_result.append((result,idx))
    llama_result = sorted(llama_result, key=lambda x:x[-1])
    llama_result = [item[0]-1 for item in llama_result]

    return llama_result 
cached_tokenizer = None
cached_model = None

@torch.inference_mode()
def load_llama_model(model_name,device):
    global cached_tokenizer, cached_model
    if cached_tokenizer is None or cached_model is None:
        cached_tokenizer = AutoTokenizer.from_pretrained(model_name)
        cached_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device}, 
            torch_dtype=torch.float16,
            load_in_4bit=True)
        #cached_model = cached_model.float16()
    return cached_tokenizer, cached_model

@torch.inference_mode()
def llama_generate(input_text, tokenizer, model,device,max_new=1000):
    with torch.no_grad():
        inputs = tokenizer([input_text], truncation=True, max_length=100000, return_tensors="pt").to(device)
        outputs = model.generate(inputs["input_ids"],
                                 do_sample=False,
                                 attention_mask=inputs["attention_mask"],
                                 max_new_tokens=max_new,
                                 num_return_sequences=1)
        outputs = outputs[0][len(inputs["input_ids"][0]):]
    generated_text = tokenizer.decode(outputs,skip_special_tokens=True,trucation=True).strip()
    return generated_text

@torch.inference_mode()
def efficient_llama_text(input_text,device,model_name="meta-llama/Llama-3.1-70B-Instruct/", rewrite=True):
    tokenizer, model = load_llama_model(model_name,device)
    llama_result = []
    for idx,text in enumerate(input_text):
        conv = conv_v1.copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()
        result = llama_generate(final_prompt,tokenizer,model,device)
        result = extract_use_infor(result)
        llama_result.append((result,idx))
    llama_result = sorted(llama_result, key=lambda x:x[-1])
    llama_result = [item[0] for item in llama_result]

    return llama_result
import re

def is_sent(text):
    return text.strip().endswith(".")

def split_title_abs(text,dataset_name="cora"):
    text_split = {}
    if dataset_name == "cora":
        colon_positions = [m.start() for m in re.finditer(":", text)]
        
    
        if len(colon_positions) == 0:
            text_split['title'] = text
    
        for i, colon_pos in enumerate(colon_positions):
            before_colon = text[:colon_pos].strip()

            if is_sent(before_colon):
                title = text[:colon_pos].strip()
                text_split['title'] = title
                abstract = text[colon_pos + 1:].strip()
                text_split['abs'] = abstract
                return text_split
        if 'abs' not in text.keys():
            text_split = ''
    elif dataset_name == "pubmed":
        
        title_match = re.search(r"Title:\s*(.*?)(?=\n|$)", text)
        abs_match = re.search(r"Abstract:\s*(.*?)(?=\n|$)", text)
        if title_match and abs_match:
            title = title_match.group(1).strip()
            abstract = abs_match.group(1).strip()
            text_split['title'] = title
            text_split['abs'] = abstract
    #more to continue            
        
              
    return text_split



def extract_use_infor(text):
    # extract title and abstract
    title_match = re.search(r"\*\*Title:\*\* '(.*?)'", text)
    abstract_match = re.search(r"\*\*Abstract:\*\* '(.*?)'", text)
    
    # if matching
    if title_match and abstract_match:
        title = title_match.group(1).strip()
        abstract = abstract_match.group(1).strip()

        new_string = f"{title}: {abstract}"
        return new_string
    elif title_match:
        title = title_match.group(1).strip()
        new_string = f"{title}"
        return new_string
    elif abstract_match:
        abstract = abstract_match.group(1).strip()
        new_string = f"{abstract}"
        return new_string
    else:
        return text
