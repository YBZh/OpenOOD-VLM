from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger
import torch
import pdb

class TrainFSOODPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)

        train_loader, val_loader = loader_dict['train'], loader_dict['val']
        test_loader = loader_dict['test']

        # init network
        net = get_network(self.config.network)

        # init trainer and evaluator
        trainer = get_trainer(net, train_loader, val_loader, self.config)
        evaluator = get_evaluator(self.config)

       # init ood postprocessor
        postprocessor = get_postprocessor(self.config)
        # setup for distance-based methods
        postprocessor.setup(net, loader_dict, ood_loader_dict)
        print('\n', flush=True)
        print(u'\u2500' * 70, flush=True)

        # init recorder
        recorder = get_recorder(self.config)

        # trainer setup
        trainer.setup()
        print('\n' + u'\u2500' * 70, flush=True)

        print('Start training...', flush=True)
        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            # train and eval the model
            net, train_metrics = trainer.train_epoch(epoch_idx)
            # val_metrics = evaluator.eval_acc(net, test_loader, postprocessor, epoch_idx, fsood=True, csid_data_loaders=ood_data_loaders['csid'])
            # val_metrics = evaluator.eval_acc(net, val_loader, postprocessor, epoch_idx) ## not calculate the in-domain acc.
            val_metrics = evaluator.eval_ood_val_accname(net, loader_dict, ood_loader_dict, postprocessor, epoch_idx) ## not calculate the in-domain acc.
            # ood_metrics = evaluator.eval_ood(net, loader_dict, ood_loader_dict, postprocessor) ## 
            # save model and report the result
            # pdb.set_trace()
            recorder.save_model(net, val_metrics)
            recorder.report(train_metrics, val_metrics)
           
        recorder.summary()
        print(u'\u2500' * 70, flush=True)
        # load the checkpoint with the best val metrics, it typicall leads to better results than the last ckpt.
        best_ckpt = self.config.output_dir + '/best.ckpt'
        ckpt = torch.load(best_ckpt)
        net.load_state_dict(torch.load(best_ckpt), strict=False)
        print('Model Loading {} Completed!'.format(best_ckpt))
        
        # evaluate on test set
        print('Start testing...', flush=True)
        test_metrics = evaluator.eval_acc(net, test_loader, postprocessor)
        print('\nComplete Evaluation, accuracy {:.2f}'.format(
            100.0 * test_metrics['acc']),
              flush=True)
        
        # start evaluating ood detection methods
        evaluator.eval_ood(net, loader_dict, ood_loader_dict, postprocessor)
        print('Completed!', flush=True)
