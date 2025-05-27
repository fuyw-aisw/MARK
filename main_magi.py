import torch
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from utils import set_seed_config,get_modul_mask
from network import Graph_embed_dmon, Graph_embed_magi
from data import get_sentence_embeddings
from metrics import modularity, conductance, calculate_accuracy_and_f1,true_map_cluster
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
from utils import log, top_k_samples
from torch_geometric.utils import to_dense_adj
from call_api import call_api
import prompt
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
#from sklearn.cluster import KMeans
import json


def train(args,data,seeds):
    for seed in [1]:
        print("Random seed",seed)
        set_seed_config(seed)
        model = Graph_embed_magi(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.wd)
        if args.normalize:
            data.x = F.normalize(data.x, dim = -1)
        N = data.x.shape[0]
        data_obj = Data(x = data.x, edge_index = data.edge_index)
        data_obj = data_obj.to(args.device)
        n_clusters = data.y.max().item()+1
        train_loss = {'warm_up':[],'llm_train':[]}
        y_true = data.y.numpy()

        #warm up
        for epoch in range(args.warm_up_epochs):
            model.train()
            optimizer.zero_grad() 

            g_feat1,g_feat2,data_aug1,data_aug2 = model(data_obj)
            #mask_aug1 = get_modul_mask(data_aug1.cpu(),args)
            mask_aug1 = get_modul_mask(data_aug1.cpu(),args)
            mask_aug2 = get_modul_mask(data_aug2.cpu(),args)
            magi_loss = model.magi_loss(g_feat1,mask_aug1) + model.magi_loss(g_feat2,mask_aug2)
            cl_loss1 = model.cl_loss1(g_feat1,g_feat2,args.tau1)
            loss = magi_loss + args.alpha * cl_loss1 
            log(f"{epoch+1} pretrain_epoch ||| Loss: {round(float(loss),3)} ||| magi_loss: {round(float(magi_loss),3)} ||| cl_loss1: {round(float(cl_loss1),3)}")
            train_loss['warm_up'].append(loss.clone().detach().cpu().numpy())
                              
            loss.backward()            
            optimizer.step()
            
            if (epoch+1) % 100 == 0:
                
                model.eval()    
                conduct_list,modul_list,acc_list,nmi_list,ari_list,f1_list = [],[],[],[],[],[]
                with torch.no_grad():                
                    g_feat1,g_feat2,_,_ = model(data_obj)
                    #y_pred,_,mis_mask = model.cluster(g_feat1,g_feat2)
                    for seed in seeds:
                        y_pred,mis_mask = model.cluster(g_feat1,g_feat2,seed) 
                        
                        y_assignments = y_pred
                        adj = to_scipy_sparse_matrix(data.edge_index).tocsr()
                        conduct = conductance(adj, y_assignments)
                        modul = modularity(adj, y_assignments)                    
                        nmi = normalized_mutual_info_score(y_true,y_assignments)
                        ari = adjusted_rand_score(y_true, y_assignments)
                        acc,f1 = calculate_accuracy_and_f1(y_true,y_assignments)
                        conduct_list.append(conduct)
                        modul_list.append(modul)
                        acc_list.append(acc)
                        nmi_list.append(nmi)
                        ari_list.append(ari)
                        f1_list.append(f1) 
                      
                        log(f"{epoch+1} {seed} pretrain_val_epoch ||| conductance: {round(float(conduct),3)} ||| modularity: {round(float(modul),3)} ||| accuracy: {round(float(acc),3)} ||| nmi: {round(float(nmi),3)} |||ari: {round(float(ari),3)}|||f1 score : {round(float(f1),3)}")
                    
                    log(f"{epoch+1} pretrain_val_epoch ||| conductance mean: {round(float(np.mean(conduct_list)),3)}  ||| conductance std: {round(float(np.std(conduct_list)),3)}||| modularity mean : {round(float(np.mean(modul_list)),3)} ||| modularity std : {round(float(np.std(modul_list)),3)} ||| accuracy mean: {round(float(np.mean(acc_list)),3)}||| accuracy std: {round(float(np.std(acc_list)),3)} ||| nmi mean: {round(float(np.mean(nmi_list)),3)} ||| nmi std: {round(float(np.std(nmi_list)),3)}|||ari mean: {round(float(np.mean(ari_list)),3)}|||ari std: {round(float(np.std(ari_list)),3)}|||f1 score mean: {round(float(np.mean(f1_list)),3)}|||f1 score std: {round(float(np.std(f1_list)),3)}") 
            
        #finetuning
        
        
        #torch.save(model.state_dict(), f"models/magi_{args.dataset}_we_{args.alpha}_tau_{args.tau1}_pretrain_dis.pt")


        #model.load_state_dict(torch.load(f"models/magi_{args.dataset}_we_{args.alpha}_tau_{args.tau1}_pretrain_dis.pt",map_location=args.device))
        #breakpoint()
        texts_llm_ge = {}
        labels_llm_ge = {}
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            g_feat1,g_feat2,data_aug1,data_aug2 = model(data_obj)
            mask_aug1 = get_modul_mask(data_aug1.cpu(),args)
            mask_aug2 = get_modul_mask(data_aug2.cpu(),args)
            magi_loss = model.magi_loss(g_feat1,mask_aug1) + model.magi_loss(g_feat2,mask_aug2)
            #torch.save(g_feat1.detach().cpu(),f"models/g_feat_{args.dataset}.pt")
            cl_loss1 = model.cl_loss1(g_feat1,g_feat2,args.tau1)
            if epoch % 100 == 0:
                y_assignments,mis_mask = model.cluster(g_feat1,g_feat2)
                uncertain_mask = mis_mask.copy()
                #uncertain_mask = np.array([0,1])
                #y_assignments = torch.argmax(y_pred.cpu(), dim=-1).numpy()
                g_feat_confi = torch.stack([g_feat1[y_assignments == i].detach().mean(axis=0) for i in range(args.num_classes)]).to(args.device)
                confi_mask = {}
                for i in range(args.num_classes):
                    idx_in_cluster = np.nonzero(y_assignments == i)[0]
                    dist = torch.norm(g_feat1[y_assignments == i] - g_feat_confi[i], p=2, dim=1)  
                    num_top = min(len(dist),args.top)
                    print(num_top)
                    _, idx = torch.topk(dist, num_top, largest=False)
                    confi_mask[i] = idx_in_cluster[idx.cpu().numpy()].tolist()
                mapping = true_map_cluster(y_true,y_assignments)
                print(len(uncertain_mask))
                print(mapping)
                #print(confident_node_idx)
                print('----')
                print(uncertain_mask)
                print(confi_mask)

                prompts_topics = prompt.prompt_cluster_sum(data_obj=data, confi_mask=confi_mask, dataset_name=args.dataset)
                #print(prompts_topics)

                topics_generate, topics_reason = prompt.efficient_gpt_text_ind(prompts_topics)
                print(topics_generate)
                with open(f'jsons/magi_topics_{args.dataset}_we_{args.alpha}_{args.beta}_tau_{args.tau1}_{args.tau2}{args.suff}.json', 'w', encoding='utf-8') as f:
                    json.dump(topics_generate, f, ensure_ascii=False, indent=4)
                with open(f'jsons/magi_topics_reason_{args.dataset}_we_{args.alpha}_{args.beta}_tau_{args.tau1}_{args.tau2}{args.suff}.json', 'w', encoding='utf-8') as f:
                    json.dump(topics_reason, f, ensure_ascii=False, indent=4)    
                prompts_generate = prompt.prompt_neighbor_generate(data_obj=data, sampled_node_idxs=uncertain_mask, dataset_name=args.dataset, g_feat=g_feat1.detach().cpu().numpy(), topic_clusters=topics_generate, hop=args.hop, sample_num=args.k)
                texts_generate, texts_reason = prompt.efficient_gpt_text_ge(prompts_generate) #texts
                #print(texts_generate)
                with open(f'jsons/magi_texts_{args.dataset}_we_{args.alpha}_{args.beta}_tau_{args.tau1}_{args.tau2}{args.suff}.json', 'w', encoding='utf-8') as f:
                    json.dump(texts_generate, f, ensure_ascii=False, indent=4) 
                with open(f'jsons/magi_texts_reason_{args.dataset}_we_{args.alpha}_{args.beta}_tau_{args.tau1}_{args.tau2}{args.suff}.json', 'w', encoding='utf-8') as f:
                    json.dump(texts_reason, f, ensure_ascii=False, indent=4)
                    
                #print(len(mis_mask),len(texts_generate))
                assert len(uncertain_mask) == len(texts_generate)

                prompts_classify1, prompts_classify2 = prompt.prompt_aug_classifier(data_obj=data, sampled_node_idxs=uncertain_mask, aug_node_texts=texts_generate, topic_clusters=topics_generate, dataset_name=args.dataset)
                
                label_classify1, label1_reason = prompt.efficient_gpt_text_cls(prompts_classify1, n_clusters)
                with open(f'jsons/magi_label1_reason_{args.dataset}_we_{args.alpha}_{args.beta}_tau_{args.tau1}_{args.tau2}{args.suff}.json', 'w', encoding='utf-8') as f:
                    json.dump(label1_reason, f, ensure_ascii=False, indent=4)
               
                label_classify2, label2_reason = prompt.efficient_gpt_text_cls(prompts_classify2, n_clusters)
                with open(f'jsons/magi_label2_reason_{args.dataset}_we_{args.alpha}_{args.beta}_tau_{args.tau1}_{args.tau2}{args.suff}.json', 'w', encoding='utf-8') as f:
                    json.dump(label2_reason, f, ensure_ascii=False, indent=4)                 
                for idx in range(len(uncertain_mask)):
                    labels_llm_ge[int(uncertain_mask[idx])] = [int(label_classify1[idx]),int(label_classify2[idx])] # first label ; second consistence
                with open(f'jsons/magi_labels_{args.dataset}_we_{args.alpha}_{args.beta}_tau_{args.tau1}_{args.tau2}{args.suff}.json', 'w', encoding='utf-8') as f:
                    json.dump(labels_llm_ge, f, ensure_ascii=False, indent=4)
                '''
                with open(f'jsons/magi_texts_{args.dataset}_we_{args.alpha}_{args.beta}_tau_{args.tau1}_{args.tau2}{args.suff}.json', 'r', encoding='utf-8') as f:
                    texts_generate = json.load(f) 
                with open(f'jsons/magi_labels_{args.dataset}_we_{args.alpha}_{args.beta}_tau_{args.tau1}_{args.tau2}{args.suff}.json', 'r', encoding='utf-8') as f:
                    labels_llm_ge = json.load(f)

                uncertain_mask, label_classify1, label_classify2 = [],[],[]
                for key, value in labels_llm_ge.items():
                    uncertain_mask.append(int(key))
                    label_classify1.append(int(value[0]))
                    label_classify2.append(int(value[1]))
                uncertain_mask = np.array(uncertain_mask)
                '''

                ground_truth = np.array([mapping[int(label)] for label in data.y[uncertain_mask]])
                llm_idx = np.array([i for i in range(len(label_classify1)) if label_classify1[i] == label_classify2[i]])
                llm_labels = np.array([label_classify1[i] for i in range(len(label_classify1)) if label_classify1[i] == label_classify2[i]])
                #print(uncertain_mask)
                #print(len(llm_labels), len(uncertain_mask), len(llm_labels)/len(uncertain_mask))
                #print("label_classify1")
                #print(label_classify1)
                #print("label_classify2")
                #print(label_classify2)
                #print("Ground truth")
                #print(ground_truth)
                acc_gen1,acc_gen2 = np.array(label_classify1==ground_truth).mean(), np.array(label_classify2==ground_truth).mean()
                print(acc_gen1,acc_gen2)
                #update features
                texts_generate_embed = get_sentence_embeddings(texts_generate, embed_type=args.embed_type, device=args.device).to(args.device)
                for idx in llm_idx:
                    #data.raw_texts[uncertain_mask[idx]] = data.raw_texts[uncertain_mask[idx]] + texts_generate[idx]
                    data_obj.x[uncertain_mask[idx]] = (data_obj.x[uncertain_mask[idx]] + texts_generate_embed[idx])/2
            assert g_feat1[uncertain_mask[llm_idx]].shape[0] == len(llm_labels)
            cl_loss2 = model.cl_loss2(g_feat1[uncertain_mask[llm_idx]], g_feat_confi, torch.tensor(llm_labels,dtype=torch.long).to(args.device), args.tau2)
            #criterion = nn.BCEWithLogitsLoss()
            #mask = torch.cat([torch.ones(len(uncertain_mask)), torch.zeros(len(uncertain_mask))])
            #mask = mask.view(-1,1).to(args.device)
            #y_dis = model.readout(torch.cat([data_obj.x[uncertain_mask], texts_generate_embed], dim=0))
            #dis_loss = criterion(y_dis, mask)

            #print(uncertain_mask)
            #loss = args.beta * cl_loss1 + (1 - args.beta) * cl_loss2 + dis_loss
            if args.dataset in ['cora','wikics']:
                loss = args.beta * cl_loss1 + (1-args.beta)*cl_loss2          
                log(f"{epoch+1} finetune_epoch ||| Loss: {round(float(loss),3)}||| cl_loss1: {round(float(cl_loss1),3)}  ||| cl_loss2: {round(float(cl_loss2),3)}")
            else:
                loss = args.beta * magi_loss + (1-args.beta)*cl_loss2          
                log(f"{epoch+1} finetune_epoch ||| Loss: {round(float(loss),3)}||| magi_loss: {round(float(magi_loss),3)}  ||| cl_loss2: {round(float(cl_loss2),3)}")
            train_loss['llm_train'].append(loss.clone().detach().cpu().numpy())
            
            loss.backward()
            optimizer.step() 
            
            model.eval()
            conduct_list,modul_list,acc_list,nmi_list,ari_list,f1_list = [],[],[],[],[],[]
            
            with torch.no_grad():
                g_feat1,g_feat2,_,_ = model(data_obj)
                #y_pred,_,mis_mask = model.cluster(g_feat1,g_feat2)
                if (epoch + 1) % 10 == 0:
                    #torch.save([g_feat1.cpu(),g_feat2.cpu()],"models/magi_embeds_{args.dataset}_1_{args.tau1}_{args.tau2}_{epoch}.pt")
                    for seed in seeds:
                        y_pred, mis_mask = model.cluster(g_feat1,g_feat2,seed)     
                        y_assignments = y_pred
                        adj = to_scipy_sparse_matrix(data.edge_index).tocsr()
                        conduct = conductance(adj, y_assignments)
                        modul = modularity(adj, y_assignments)                    
                        nmi = normalized_mutual_info_score(y_true,y_assignments)
                        ari = adjusted_rand_score(y_true, y_assignments)
                        acc,f1 = calculate_accuracy_and_f1(y_true,y_assignments)
                        #g_feat = g_feat1.cpu().detach().numpy()
                        conduct_list.append(conduct)
                        modul_list.append(modul)
                        acc_list.append(acc)
                        nmi_list.append(nmi)
                        ari_list.append(ari)
                        f1_list.append(f1)                         
                        log(f"{epoch+1} {seed} finetune_val_epoch ||| conductance: {round(float(conduct),3)} ||| modularity: {round(float(modul),3)} ||| accuracy: {round(float(acc),3)} ||| nmi: {round(float(nmi),3)} |||ari: {round(float(ari),3)}|||f1 score : {round(float(f1),3)}")
                    
                    log(f"{epoch+1} finetune_val_epoch ||| conductance mean: {round(float(np.mean(conduct_list)),3)}  ||| conductance std: {round(float(np.std(conduct_list)),3)}||| modularity mean : {round(float(np.mean(modul_list)),3)} ||| modularity std : {round(float(np.std(modul_list)),3)} ||| accuracy mean: {round(float(np.mean(acc_list)),3)}||| accuracy std: {round(float(np.std(acc_list)),3)} ||| nmi mean: {round(float(np.mean(nmi_list)),3)} ||| nmi std: {round(float(np.std(nmi_list)),3)}|||ari mean: {round(float(np.mean(ari_list)),3)}|||ari std: {round(float(np.std(ari_list)),3)}|||f1 score mean: {round(float(np.mean(f1_list)),3)}|||f1 score std: {round(float(np.std(f1_list)),3)}")      
                    
                

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Graph clustering enhanced by LLM")
    parser.add_argument('--dataset',default='cora',type=str)
    parser.add_argument('--embed_type',default='sbert',type=str)
    parser.add_argument('--epochs', type=int, default=200) 
    parser.add_argument('--warm_up_epochs', type=int, default=100)
    parser.add_argument('--seed_num', type=int, default=5)
    parser.add_argument('--optim',type=str, default='adam')
    parser.add_argument('--alpha',help='hyper-parameter to control the weights',type=float, default=0.1)
    parser.add_argument('--beta',help='hyper-parameter to control the weights',type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--norm', type=str, default=None)
    parser.add_argument('--k', type=int, default=2,help='hyper-parameter for neighbor selection')
    parser.add_argument('--hop', type=int, default=2,help='hyper-parameter for the number of hop')
    parser.add_argument('--top', type=int, default=50,help='hyper-parameter for the number of top high confidence samples')
    parser.add_argument('--tau1', type=float, default=0.5, help='temperature for cl1')
    parser.add_argument('--tau2', type=float, default=0.01, help='temperature for cl2')
    #parser.add_argument('--batch_size', type=int, default=1024)  
    parser.add_argument('--hidden', type=str, default='512', help='GNN encoder')
    parser.add_argument('--projection', type=str, default='', help='Projection')

    # sample para
    parser.add_argument('--wt', type=int, default=100,
                    help='number of random walks')
    parser.add_argument('--wl', type=int, default=2, help='depth of random walks')
    parser.add_argument('--tau', type=float, default=0.3, help='temperature')

    # learning para

    
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-3, help='weight decay')
    #parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--ns', type=float, default=0.5, help='') 
    parser.add_argument('--suff', type=str, default='')
    
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}"
    data = torch.load(f"preprocessed_data/{args.dataset}_{args.embed_type}.pt")
    #data = torch.load(f"../Graph-LLM-master/preprocessed_data/new/{args.dataset}_random_{args.embed_type}.pt")
    args.input_dim = data.x.shape[1]
    args.num_classes = data.y.max().item()+1
    seeds = [i for i in range(1,args.seed_num+1)]
    print(args)
    train(args,data,seeds)
        
      
