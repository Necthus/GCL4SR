# -*- coding: utf-8 -*-
import numpy as np
import tqdm
import torch

from utils import recall_at_k, ndcg_k, get_metric,Timer


class Trainer:
    def __init__(self, model, args):

        self.args = args
        self.model = model
        if self.model.cuda_condition:
            self.model.to(self.model.device)

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    
    def get_sample_score(self, epoch, answers, pred_list):
        
        
        import math
        
        num_users = len(pred_list)
        
        assert len(pred_list) == len(answers)
        
        
        metrics = {}
        metrics['Epoch']=epoch
        metrics['X'] = {}
        metrics['Y'] = {}
        for domain in ['X','Y']:
            for m in ['MRR','NDCG5','NDCG10','HR1','HR5','HR10','NUM']:
                metrics[domain][m]=0
        
        
        for i in range(num_users):
            
            # if i % 100 == 0:
            #     print(i,flush=True)
            
        
            
            true_item = answers[i][0]
            current_pred_list = pred_list[i]
            if true_item<self.args.x_domain_size:
                current_domain = 'X'
                metrics['X']['NUM']+=1
                new_pred_list = np.array([x for x in current_pred_list if x<self.args.x_domain_size])
            else:
                current_domain = 'Y'
                metrics['Y']['NUM']+=1
                new_pred_list = np.array([y for y in current_pred_list if y>=self.args.x_domain_size])
                
            gt_pos = np.argwhere(new_pred_list==true_item)[0][0]
            # print(gt_pos)
            slices = np.random.choice(len(new_pred_list),self.args.eval_size,replace=False)
            # print(new_pred_list.shape)
            if gt_pos not in slices:
                slices[-1]=gt_pos

            sorted_slices = np.sort(slices)
            new_pred_list = new_pred_list[sorted_slices]    
                
            new_pred_list = list(new_pred_list)
            
                
            if true_item in new_pred_list:
                first_appear = new_pred_list.index(true_item)+1
                metrics[current_domain]['MRR']+=1/first_appear
            
            if true_item in new_pred_list[:1]:
                metrics[current_domain]['HR1']+=1
                
            if true_item in new_pred_list[:5]:
                metrics[current_domain]['HR5']+=1
                first_appear = new_pred_list.index(true_item)+1
                metrics[current_domain]['NDCG5']+=1/math.log(first_appear+1,2)
                
            if true_item in new_pred_list[:10]:
                metrics[current_domain]['HR10']+=1
                first_appear = new_pred_list.index(true_item)+1
                metrics[current_domain]['NDCG10']+=1/math.log(first_appear+1,2)
                
        for domain in ['X','Y']:
            for m in ['MRR','NDCG5','NDCG10','HR1','HR5','HR10']:
                metrics[domain][m]/=metrics[domain]['NUM']
                metrics[domain][m]=round(metrics[domain][m],4)
        
        
        
        print(metrics)
        with open(self.args.log_file, 'a') as f:
            f.write(str(metrics) + '\n')    
        
        return [(metrics['X']['HR10']+metrics['Y']['HR10'])/2],str(metrics)     
    
    
    def get_full_sort_score(self, epoch, answers, pred_list):
        
        
        import math
        
        num_users = len(pred_list)
        
        assert len(pred_list) == len(answers)
        
        
        metrics = {}
        metrics['Epoch']=epoch
        metrics['X'] = {}
        metrics['Y'] = {}
        for domain in ['X','Y']:
            for m in ['MRR','NDCG5','NDCG10','HR1','HR5','HR10','NUM']:
                metrics[domain][m]=0
        
        
        for i in range(num_users):
            true_item = answers[i][0]
            current_pred_list = pred_list[i]
            if true_item<self.args.x_domain_size:
                current_domain = 'X'
                metrics['X']['NUM']+=1
                new_pred_list = [x for x in current_pred_list if x<self.args.x_domain_size]
            else:
                current_domain = 'Y'
                metrics['Y']['NUM']+=1
                new_pred_list = [y for y in current_pred_list if y>=self.args.x_domain_size]
                
            if true_item in new_pred_list:
                first_appear = new_pred_list.index(true_item)+1
                metrics[current_domain]['MRR']+=1/first_appear
            
            if true_item in new_pred_list[:1]:
                metrics[current_domain]['HR1']+=1
                
            if true_item in new_pred_list[:5]:
                metrics[current_domain]['HR5']+=1
                first_appear = new_pred_list.index(true_item)+1
                metrics[current_domain]['NDCG5']+=1/math.log(first_appear+1)
                
            if true_item in new_pred_list[:10]:
                metrics[current_domain]['HR10']+=1
                first_appear = new_pred_list.index(true_item)+1
                metrics[current_domain]['NDCG10']+=1/math.log(first_appear+1)
                
        for domain in ['X','Y']:
            for m in ['MRR','NDCG5','NDCG10','HR1','HR5','HR10']:
                metrics[domain][m]/=metrics[domain]['NUM']
                metrics[domain][m]=round(metrics[domain][m],4)
        
        
        
        print(metrics)
        with open(self.args.log_file, 'a') as f:
            f.write(str(metrics) + '\n')    
        
        return [(metrics['X']['HR10']+metrics['Y']['HR10'])/2],str(metrics)        
        
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.model.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class GCL4SR_Train(Trainer):
    def __init__(self, model, args):
        super(GCL4SR_Train, self).__init__(
            model,
            args
        )

    def train_stage(self, epoch, train_dataloader):

        
        
        desc = f'n_sample-{self.args.sample_size}-' \
               f'hidden_size-{self.args.hidden_size}'

        train_data_iter = tqdm.tqdm(enumerate(train_dataloader),
                                       desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
                                       total=len(train_dataloader),
                                       bar_format="{l_bar}{r_bar}")

        self.model.train()
        joint_loss_avg = 0.0
        main_loss_avg = 0.0
        cl_loss_avg = 0.0
        mmd_loss_avg = 0.0

        for i, batch in train_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            
            # print(i,'Step begin')
            # timer.start()
            
            
            batch = tuple(t.to(self.model.device) for t in batch)
            # timer.record('Batch prepared')
            
            joint_loss, main_loss, cl_loss, mmd_loss = self.model.train_stage(batch)
            # timer.record('Forward Done')

            self.model.optimizer.zero_grad()
            joint_loss.backward()
            # timer.record('Backward Done')
            
            self.model.optimizer.step()
            # timer.record('Optimize Done')

            joint_loss_avg += joint_loss.item()
            main_loss_avg += main_loss.item()
            cl_loss_avg += cl_loss.item()
            mmd_loss_avg += mmd_loss.item()
            
            # timer.record('Step Done')
            
        self.model.scheduler.step()
        post_fix = {
            "epoch": epoch,
            "joint_loss_avg": '{:.4f}'.format(joint_loss_avg / len(train_data_iter)),
            "main_loss_avg": '{:.4f}'.format(main_loss_avg / len(train_data_iter)),
            "gcl_loss_avg": '{:.4f}'.format(cl_loss_avg / len(train_data_iter)),
            "mmd_loss_avg": '{:.4f}'.format(mmd_loss_avg / len(train_data_iter)),
        }
        print(desc)
        print(str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(desc) + '\n')
            f.write(str(post_fix) + '\n')

    def eval_stage(self, epoch, dataloader, full_sort=False, test=True):

        str_code = "test" if test else "eval"
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        self.model.eval()

        pred_list = None

        if full_sort:
            answer_list = None

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.model.device) for t in batch)
                user_ids = batch[0]
                answers = batch[2]
                recommend_output = self.model.eval_stage(batch)
                answers = answers.view(-1, 1)
                
                # 推荐的结果
                rating_pred = self.predict_full(recommend_output)
                
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                
                # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
                # 加负号"-"表示取大的值
                
                n = self.args.item_size
                
                ind = np.argpartition(rating_pred, -n)[:, -n:]
                # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                # 对子表进行排序 得到从大到小的顺序
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                # 再取一次 从ind中取回 原来的下标
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
            return self.get_sample_score(epoch, answer_list, pred_list)

        else:
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.model.device) for t in batch)
                user_ids, inputs, target_seq, target_neg, target_nxt, mask_node_sequence, pos_node, sample_negs = batch

                recommend_output = self.model.eval_stage(batch)
                test_neg_items = torch.cat((target_nxt.view(-1, 1), sample_negs), -1)
                recommend_output = recommend_output[:, -1, :]

                test_logits = self.predict_sample(recommend_output, test_neg_items)
                test_logits = test_logits.cpu().detach().numpy().copy()
                if i == 0:
                    pred_list = test_logits
                else:
                    pred_list = np.append(pred_list, test_logits, axis=0)

            return self.get_sample_scores(epoch, pred_list)

