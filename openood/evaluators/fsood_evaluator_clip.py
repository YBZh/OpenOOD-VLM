import csv
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from tqdm import tqdm
import openood.utils.comm as comm
import torch.nn.functional as F

from openood.postprocessors import BasePostprocessor

from .ood_evaluator import OODEvaluator
from .metrics import compute_all_metrics
import pdb

class FSOODEvaluatorClip(OODEvaluator):
    def eval_csid_acc(self, net: nn.Module,
                      csid_loaders: Dict[str, Dict[str, DataLoader]], postprocessor):
        # ensure the networks in eval mode
        net.eval()
        # pdb.set_trace()

        for dataset_name, csid_dl in csid_loaders.items():
            print(f'Computing accuracy on {dataset_name} dataset...')
            correct = 0
            with torch.no_grad():
                for batch in csid_dl:
                    data = batch['data'].cuda()
                    target = batch['label'].cuda()
                    # forward
                    # output = net(data)
                    pred, _ = postprocessor.postprocess(net, data)
                    # # accuracy
                    # pred = output.data.max(1)[1]
                    correct += pred.eq(target.data).sum().item()
            acc = correct / len(csid_dl.dataset)
            if self.config.recorder.save_csv:
                self._save_acc_results(acc, dataset_name)
        print(u'\u2500' * 70, flush=True)

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        net.eval()

        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True,
                              disable=not comm.is_main_process()):
                # prepare data
                data = batch['data'].cuda()
                target = batch['label'].cuda()

                # forward
                # output = net(data)
                pred, _ = postprocessor.postprocess(net, data)

                # loss = F.cross_entropy(output, target)

                # # accuracy
                # pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                # loss_avg += float(loss.data)

        loss = loss_avg / len(data_loader)
        acc = correct / len(data_loader.dataset)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss)
        metrics['acc'] = self.save_metrics(acc)
        return metrics

    def _save_acc_results(self, acc, dataset_name):
        write_content = {
            'dataset': dataset_name,
            'FPR@95': '-',
            'AUROC': '-',
            'AUPR_IN': '-',
            'AUPR_OUT': '-',
            'ACC': '{:.2f}'.format(100 * acc),
        }
        fieldnames = list(write_content.keys())
        # print csid metric results
        print('CSID[{}] accuracy: {:.2f}%'.format(dataset_name, 100 * acc),
              flush=True)
        csv_path = os.path.join(self.config.output_dir, 'csid.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def eval_ood(self, net: nn.Module, id_data_loader: List[DataLoader],
                 ood_data_loaders: List[DataLoader],
                 postprocessor: BasePostprocessor):
        # ensure the networks in eval mode
        net.eval()
        # load training in-distribution data
        assert 'test' in id_data_loader, \
            'id_data_loaders should have the key: test!'
        dataset_name = self.config.dataset.name
        print(f'Performing inference on {dataset_name} dataset...', flush=True)
        id_pred, id_conf, id_gt = postprocessor.inference(
            net, id_data_loader['test'])
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        # load csid data and compute confidence
        for dataset_name, csid_dl in ood_data_loaders['csid'].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            csid_pred, csid_conf, csid_gt = postprocessor.inference(net, csid_dl)
            if self.config.recorder.save_scores:
                self._save_scores(csid_pred, csid_conf, csid_gt, dataset_name)
            id_pred = np.concatenate([id_pred, csid_pred])
            id_conf = np.concatenate([id_conf, csid_conf])
            id_gt = np.concatenate([id_gt, csid_gt])

        # compute accuracy on csid
        print(u'\u2500' * 70, flush=True)
        self.eval_csid_acc(net, ood_data_loaders['csid'], postprocessor)

        # load nearood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='nearood')

        # load farood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='farood')


class OODEvaluatorClip(OODEvaluator):
    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        net.eval()

        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True,
                              disable=not comm.is_main_process()):
                # prepare data
                data = batch['data'].cuda()
                target = batch['label'].cuda()

                # forward
                # output = net(data)
                pred, _ = postprocessor.postprocess(net, data)

                # loss = F.cross_entropy(output, target)

                # # accuracy
                # pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                # loss_avg += float(loss.data)

        loss = loss_avg / len(data_loader)
        acc = correct / len(data_loader.dataset)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss)
        metrics['acc'] = self.save_metrics(acc)
        return metrics

    def eval_ood(self, net: nn.Module, id_data_loaders: List[DataLoader],
                 ood_data_loaders: List[DataLoader],
                 postprocessor: BasePostprocessor,  fsood: bool = False):
        # ensure the networks in eval mode
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        # load training in-distribution data
        assert 'test' in id_data_loaders, \
            'id_data_loaders should have the key: test!'
        dataset_name = self.config.dataset.name
        if self.config.postprocessor.APS_mode:
            assert 'val' in id_data_loaders
            assert 'val' in ood_data_loaders
            self.hyperparam_search(net, id_data_loaders['val'],
                                   ood_data_loaders['val'], postprocessor)
        print(f'Performing inference on {dataset_name} dataset...', flush=True)

        id_pred, id_conf, id_gt = postprocessor.inference(
            net, id_data_loaders['test'])
        # id_pred: 50k array, [0, 1k]
        # id_conf: 50k array, [float]
        # id_gt: 50k array, gt of id_pred!! note this should be random order in TTA method. The order is fixed in default. 
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        # if fsood:
        #     # load csid data and compute confidence
        #     for dataset_name, csid_dl in ood_data_loaders['csid'].items():
        #         print(f'Performing inference on {dataset_name} dataset...',
        #               flush=True)
        #         csid_pred, csid_conf, csid_gt = postprocessor.inference(
        #             net, csid_dl)
        #         if self.config.recorder.save_scores:
        #             self._save_scores(csid_pred, csid_conf, csid_gt,
        #                               dataset_name)
        #         id_pred = np.concatenate([id_pred, csid_pred])
        #         id_conf = np.concatenate([id_conf, csid_conf])
        #         id_gt = np.concatenate([id_gt, csid_gt])
        # load nearood data and compute ood metrics
        # pdb.set_trace()
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='nearood')

        # load farood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='farood')

class OODEvaluatorClipTTA(OODEvaluator):
    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1, fsood=True,
                 csid_data_loaders=None):
        postprocessor.reset_memory()  ## reset the memory for the ID evaluation.
        net.eval()

        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True,
                              disable=not comm.is_main_process()):
                # prepare data
                data = batch['data'].cuda()
                target = batch['label'].cuda()

                # forward
                # output = net(data)
                pred, _ = postprocessor.postprocess(net, data)

                # loss = F.cross_entropy(output, target)

                # # accuracy
                # pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                # loss_avg += float(loss.data)

        loss = loss_avg / len(data_loader)
        acc = correct / len(data_loader.dataset)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss)
        metrics['acc'] = self.save_metrics(acc)
        return metrics

    def eval_ood(self, net: nn.Module, id_data_loaders: List[DataLoader],
                 ood_data_loaders: List[DataLoader],
                 postprocessor: BasePostprocessor,  fsood: bool = False):
        # ensure the networks in eval mode
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        # load training in-distribution data
        assert 'test' in id_data_loaders, \
            'id_data_loaders should have the key: test!'
        dataset_name = self.config.dataset.name
        if self.config.postprocessor.APS_mode:
            assert 'val' in id_data_loaders
            assert 'val' in ood_data_loaders
            self.hyperparam_search(net, id_data_loaders['val'],
                                   ood_data_loaders['val'], postprocessor)
        print(f'Performing inference on {dataset_name} dataset...', flush=True)

        # id_pred, id_conf, id_gt = postprocessor.inference(
        #     net, id_data_loaders['test'])
        # id_pred: 50k array, [0, 1k]
        # id_conf: 50k array, [float]
        # id_gt: 50k array, gt of id_pred!! note this should be random order in TTA method. The order is fixed in default. 
        # if self.config.recorder.save_scores:
        #     self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        # if fsood:
        #     # load csid data and compute confidence
        #     for dataset_name, csid_dl in ood_data_loaders['csid'].items():
        #         print(f'Performing inference on {dataset_name} dataset...',
        #               flush=True)
        #         csid_pred, csid_conf, csid_gt = postprocessor.inference(
        #             net, csid_dl)
        #         if self.config.recorder.save_scores:
        #             self._save_scores(csid_pred, csid_conf, csid_gt,
        #                               dataset_name)
        #         id_pred = np.concatenate([id_pred, csid_pred])
        #         id_conf = np.concatenate([id_conf, csid_conf])
        #         id_gt = np.concatenate([id_gt, csid_gt])
        # load nearood data and compute ood metrics
        # pdb.set_trace()
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, id_data_loaders['test'],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='nearood', fsood=fsood)

        # load farood data and compute ood metrics
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, id_data_loaders['test'],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='farood', fsood=fsood)

    ## calculate the id pred/conf/gt online, not offline as the default setting. Therefore, different data order may lead to different results. 
    def _eval_ood(self,
                  net: nn.Module,
                  id_loader: DataLoader,
                  ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                  postprocessor: BasePostprocessor,
                  ood_split: str = 'nearood', fsood=False):
        print(f'Processing {ood_split}...', flush=True)
        # [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        # postprocessor.reset_memory()  ## here, we inherit the memory with the same near/far OOD group; using more information, not fair
        for dataset_name, ood_dl in ood_data_loaders[ood_split].items():
            postprocessor.reset_memory()  ## here, we reset the memory for each OOD datasets.
            print(f'Performing inference on {dataset_name} dataset...', flush=True)
            # merging the id dataloader and ood dataloader! 
            # pdb.set_trace()
            combined_dataset = ConcatDataset([id_loader.dataset, ood_dl.dataset])
            if fsood and 'csid' in ood_data_loaders.keys():
                print(f'concating ID, CSID, and OOD dataset', flush=True)
                for dataset_name_csid, csid in ood_data_loaders['csid'].items():
                    combined_dataset = ConcatDataset([combined_dataset, csid.dataset])
            print(f'Generating combined dataset with ID and OOD dataset of {dataset_name}, total size {len(combined_dataset)}')
            # pdb.set_trace()
            # Create a new DataLoader from the combined dataset. The shuffle operation is verified
            combined_dataloader = DataLoader(combined_dataset, batch_size=id_loader.batch_size, num_workers=id_loader.num_workers, shuffle=True)
            pred, conf, label = postprocessor.inference(net, combined_dataloader) 
            # mydict = {'conf': conf, 'label': label}
            # torch.save(mydict, 'ssbhard_conf.pth')
            # pdb.set_trace()
            # ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood?
            # if self.config.recorder.save_scores:
            #     self._save_scores(ood_pred, ood_conf, ood_gt, dataset_name)

            # pred = np.concatenate([id_pred, ood_pred])
            # conf = np.concatenate([id_conf, ood_conf])
            # label = np.concatenate([id_gt, ood_gt])

            print(f'Computing metrics on {dataset_name} dataset...')

            ood_metrics = compute_all_metrics(conf, label, pred)
            if self.config.recorder.save_csv:
                self._save_csv(ood_metrics, dataset_name=dataset_name)
            metrics_list.append(ood_metrics)
            ############################# for visualization of SUN dataset.
            # pdb.set_trace()
            # save1 = '/home/notebook/code/personal/S9052995/syn_pro/OpenOOD/imagenet_id_ood_textfeat.pth'
            # save2 = '/home/notebook/code/personal/S9052995/syn_pro/OpenOOD/sun_textfeat.pth'
            # save3 = '/home/notebook/code/personal/S9052995/syn_pro/OpenOOD/adaneg_feat.pth'
            # sa_text_features = postprocessor.feat_memory.mean(1)
            # sa_text_features /= sa_text_features.norm(dim=-1, keepdim=True)  ## renorm
            # # net.text_features
            # # net.sun_features
            # torch.save(sa_text_features.cpu(), save3)
            # torch.save(net.text_features.cpu(), save1)
            # torch.save(net.sun_features.cpu(), save2)


        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        if self.config.recorder.save_csv:
            self._save_csv(metrics_mean, dataset_name=ood_split)
