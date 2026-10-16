import os
import re
import time
import random
import argparse
import json
import numpy as np
from omegaconf import OmegaConf
from thop import profile

import torch
import torch.optim as optim

from toolkit.utils.loss import *
from toolkit.utils.metric import *
from toolkit.utils.functions import *
from toolkit.models import get_models
from toolkit.dataloader import get_dataloaders


def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', str(filename))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_balanced_class_weight(train_loader, num_classes):
    dataset = train_loader.dataset
    if hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    labels = [item['emo'] for item in dataset.labels]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    safe_counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * safe_counts)
    print(f'class counts: {counts.astype(int).tolist()}')
    print(f'class weights: {[round(float(item), 4) for item in weights.tolist()]}')
    return torch.FloatTensor(weights)


def train_or_eval_model(args, model, reg_loss, cls_loss, dataloader, epoch, optimizer=None, train=False):
    
    vidnames = []
    val_preds, val_labels = [], []
    emo_probs, emo_labels = [], []
    losses = []

    assert not train or optimizer!=None
    config.train = train # 将 train 设置为全局变量影响后面 dataloader 信息
    if train:
        model.train()
    else:
        model.eval()
    
    for iter, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()
        
        # read data + cuda
        batch, emos, vals, bnames = data
        vidnames += bnames
        for key in batch: batch[key] = batch[key].cuda()
        emos = emos.cuda()
        vals = vals.cuda()

        # forward process
        # start_time = time.time()
        features, emos_out, vals_out, interloss = model(batch)
        # duration = time.time() - start_time
        # macs, params = profile(model, inputs=(batch, ))
        # print(f"MACs: {macs}, Parameters: {params}, Duration: {duration}; bsize: {len(bnames)}")

        # loss calculation
        loss = interloss
        if args.output_dim1 != 0:
            loss = loss + cls_loss(emos_out, emos)
            emo_probs.append(emos_out.data.cpu().numpy())
            emo_labels.append(emos.data.cpu().numpy())
        if args.output_dim2 != 0: 
            loss = loss + reg_loss(vals_out, vals)
            val_preds.append(vals_out.data.cpu().numpy())
            val_labels.append(vals.data.cpu().numpy())
        losses.append(loss.data.cpu().numpy())
        
        # optimize params
        if train:
            loss.backward()
            if model.model.grad_clip != -1:
                torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], model.model.grad_clip)
            optimizer.step()
        
        # print
        if (iter+1) % args.print_iters == 0:
            print (f'process on {iter+1}|{len(dataloader)}, meanloss: {np.mean(losses)}')

        ## whether save models [only train process consider save models]
        if train and args.savemodel and (iter+1) % args.save_iters == 0: # [time consuming]
            save_name = sanitize_filename(f"{prefix_name}_epoch:{'%02d' %(epoch)}_iter:{'%06d' %(iter+1)}_meanloss:{np.mean(losses)}_{name_time}")
            save_path = os.path.join(save_modelroot, save_name)
            model.model.model.save_pretrained(save_path)

    if emo_probs  != []: emo_probs  = np.concatenate(emo_probs)
    if emo_labels != []: emo_labels = np.concatenate(emo_labels)
    if val_preds  != []: val_preds  = np.concatenate(val_preds)
    if val_labels != []: val_labels = np.concatenate(val_labels)
    results, _ = dataloader_class.calculate_results(emo_probs, emo_labels, val_preds, val_labels)
    save_results = dict(
        names = vidnames,
        loss  = np.mean(losses),
        **results,
    )
    return save_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Params for datasets
    parser.add_argument('--dataset', type=str, default=None, help='dataset')
    parser.add_argument('--train_dataset', type=str, default=None, help='train dataset') # for cross-corpus test  
    parser.add_argument('--test_dataset',  type=str, default=None, help='test dataset')  # for cross-corpus test
    parser.add_argument('--save_root', type=str, default='./saved', help='save prediction results and models')
    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--savemodel', action='store_true', default=False, help='whether to save model, default: False')
    parser.add_argument('--save_iters', type=int, default=1e8, help='save models per iters')

    # Params for feature inputs
    parser.add_argument('--audio_feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text_feature',  type=str, default=None, help='text feature name')
    parser.add_argument('--video_feature', type=str, default=None, help='video feature name')
    parser.add_argument('--feat_type',  type=str, default=None, help='feature type [utt, frm_align, frm_unalign]')
    parser.add_argument('--feat_scale', type=int, default=None, help='pre-compress input from [seqlen, dim] -> [seqlen/scale, dim]')
    # Params for raw inputs
    parser.add_argument('--e2e_name', type=str, default=None, help='e2e pretrained model names')
    parser.add_argument('--e2e_dim',  type=int, default=None, help='e2e pretrained model hidden size')

    # Params for model
    parser.add_argument('--n_classes', type=int, default=None, help='number of classes [defined by args.label_path]')
    parser.add_argument('--hyper_path', type=str, default=None, help='whether choose fixed hyper-params [default use hyperparam tuning]')
    parser.add_argument('--model', type=str, default=None, help='model name for training [mlp, attention, and others from MMSA toolkits]')

    # Params for training
    parser.add_argument('--lr', type=float, default=None, metavar='lr', help='set lr rate') # 如果是None, lr 作为了 hyper-params 通过 model-tune.yaml 调节
    parser.add_argument('--lr_adjust', type=str, default='case1', help='[case1, case2]. case1: uniform lr; case2: pretrain lr = 1/10 fc lr')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='batch size [deal with OOM]')
    parser.add_argument('--num_workers', type=int, default=0, metavar='nw', help='number of workers')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--print_iters', type=int, default=1e8, help='print per-iteartion')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('--seed', default=2026, type=int, help='random seed')
    parser.add_argument('--class_weight', type=str, default='none', choices=['none', 'balanced'], help='classification class weighting')
    parser.add_argument('--drop_modalities', type=str, default='', help='comma-separated modalities to zero: audio,text,visual')
    parser.add_argument('--start_fold', type=int, default=1, help='1-based fold to start training; earlier folds are restored from --resume_root')
    parser.add_argument('--resume_root', type=str, default=None, help='completed run directory containing model/fold*_best.pt for folds before start_fold')
    args = parser.parse_args()
    args.drop_modalities_list = [
        item.strip().lower()
        for item in args.drop_modalities.split(',')
        if item.strip()
    ]
    invalid_modalities = set(args.drop_modalities_list) - {'audio', 'text', 'visual', 'video'}
    if invalid_modalities:
        parser.error(f'unsupported drop modalities: {sorted(invalid_modalities)}')
    set_random_seed(args.seed)
    torch.cuda.set_device(args.gpu)


    print ('====== Params Pre-analysis =======')
    ## accelate: pre-compress for ['frm_align', 'frm_unalign']
    if args.feat_type == 'utt':
        args.feat_scale = 1
    elif args.feat_type == 'frm_align':
        assert args.audio_feature.endswith('FRA')
        assert args.text_feature.endswith('FRA')
        assert args.video_feature.endswith('FRA')
        args.feat_scale = 6
    elif args.feat_type == 'frm_unalign':
        assert args.audio_feature.endswith('FRA')
        assert args.text_feature.endswith('FRA')
        assert args.video_feature.endswith('FRA')
        args.feat_scale = 12

    ## define store folder
    if args.train_dataset is not None:
        args.save_root = f'{args.save_root}-cross'
    whole_features = [args.audio_feature, args.text_feature, args.video_feature]
    whole_features = [item for item in whole_features if item is not None]
    if len(set(whole_features)) == 0:
        args.save_root = f'{args.save_root}-others'
    elif len(set(whole_features)) == 1:
        args.save_root = f'{args.save_root}-unimodal'
    elif len(set(whole_features)) == 2:
        args.save_root = f'{args.save_root}-bimodal'
    elif len(set(whole_features)) == 3:
        args.save_root = f'{args.save_root}-trimodal'

    ## generate model_config
    if args.hyper_path is None:
        model_config = OmegaConf.load('toolkit/model-tune.yaml')[args.model]
        model_config = func_random_select(model_config)
    else:
        model_config = OmegaConf.load(args.hyper_path)[args.model]
    config.dataset = args.dataset
    args = merge_args_config(args, model_config) # merge params
    print('args: ', args)

    ## save root
    save_resroot  = os.path.join(args.save_root, 'result')
    save_modelroot  = os.path.join(args.save_root, 'model')
    if not os.path.exists(save_resroot):  os.makedirs(save_resroot)
    if not os.path.exists(save_modelroot): os.makedirs(save_modelroot)
    config_path = os.path.join(args.save_root, 'run_config.json')
    with open(config_path, 'w', encoding='utf-8') as config_file:
        json.dump(vars(args), config_file, ensure_ascii=False, indent=2, default=str)
    print (f'save run config in {config_path}')
    # gain prefix_name
    feature_name = "+".join(sorted(list(set(whole_features)))) # sort to avoid random order
    model_name = f'{args.model}+{args.feat_type}+{args.e2e_name}'
    prefix_name = f'features:{feature_name}_dataset:{args.dataset}_model:{model_name}'
    if args.train_dataset is not None:
        assert args.test_dataset is not None
        prefix_name += f'_train:{args.train_dataset}_test:{args.test_dataset}'
    if args.class_weight != 'none':
        prefix_name += f'_classweight:{args.class_weight}'
    if args.drop_modalities_list:
        prefix_name += f"_drop:{'+'.join(args.drop_modalities_list)}"
    safe_prefix_name = sanitize_filename(prefix_name)


    print ('====== Reading Data =======')
    dataloader_class = get_dataloaders(args) # (MER2023 + e2e + e2e_name)
    train_loaders, eval_loaders, test_loaders = dataloader_class.get_loaders()
    assert len(train_loaders) == len(eval_loaders)
    print (f'train&val folder:{len(train_loaders)}; test sets:{len(test_loaders)}')
    if args.start_fold < 1 or args.start_fold > len(train_loaders):
        raise ValueError(f'start_fold must be in [1, {len(train_loaders)}], got {args.start_fold}')
    resume_modelroot = None
    if args.start_fold > 1:
        if args.resume_root is None:
            raise ValueError('resume_root is required when start_fold > 1')
        resume_modelroot = os.path.join(args.resume_root, 'model')
        if not os.path.isdir(resume_modelroot):
            raise FileNotFoundError(f'resume model directory does not exist: {resume_modelroot}')
    args.audio_dim, args.text_dim, args.video_dim = train_loaders[0].dataset.get_featdim()
    class_weight = None
    if args.class_weight == 'balanced':
        class_weight = build_balanced_class_weight(train_loaders[0], args.output_dim1)
        args.class_weight_values = [float(item) for item in class_weight.numpy().tolist()]


    print ('====== Training and Evaluation =======')
    folder_save = [] # store best results for each folder
    folder_duration = []
    for ii in range(len(train_loaders)):
        print (f'>>>>> Cross-validation: training on the {ii+1} folder >>>>>')
        train_loader = train_loaders[ii]
        eval_loader  = eval_loaders[ii]

        # Reconstruct completed folds from their best checkpoints so an interrupted
        # long-running CV job can continue without repeating earlier folds.
        if ii < args.start_fold - 1:
            checkpoint_path = os.path.join(
                resume_modelroot,
                sanitize_filename(f'fold{ii+1}_{safe_prefix_name}_best.pt'),
            )
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f'missing checkpoint for restored fold {ii+1}: {checkpoint_path}')
            print (f'Step1: restore completed fold from {checkpoint_path}')
            model = get_models(args).cuda()
            checkpoint = torch.load(checkpoint_path, map_location='cuda')
            model.load_state_dict(checkpoint['model_state_dict'])
            reg_loss = MSELoss().cuda()
            cls_loss = CELoss(weight=class_weight).cuda()
            fold_store = {}
            eval_results = train_or_eval_model(args, model, reg_loss, cls_loss, eval_loader, epoch=checkpoint.get('epoch', 0), optimizer=None, train=False)
            func_update_storage(inputs=eval_results, prefix='eval', outputs=fold_store)
            for jj, test_loader in enumerate(test_loaders):
                test_results = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, epoch=checkpoint.get('epoch', 0), optimizer=None, train=False)
                func_update_storage(inputs=test_results, prefix=f'test{jj+1}', outputs=fold_store)
            fold_store['best_model_path'] = checkpoint_path
            folder_save.append(fold_store)
            folder_duration.append(0.0)
            del model
            torch.cuda.empty_cache()
            print (f'>>>>> Restored fold {ii+1}; checkpoint epoch: {checkpoint.get("epoch", "unknown")}, metric: {checkpoint.get("metric", "unknown")} >>>>>')
            continue

        start_time = name_time = time.time()
        best_metric_so_far = -np.inf
        best_model_path = os.path.join(save_modelroot, sanitize_filename(f'fold{ii+1}_{safe_prefix_name}_best.pt'))

        print (f'Step1: build model (each folder has its own model)')
        model = get_models(args).cuda()
        reg_loss = MSELoss().cuda()
        cls_loss = CELoss(weight=class_weight).cuda()

        if args.lr_adjust == 'case1':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        elif args.lr_adjust == 'case2':
            assert args.model == 'e2e_model', 'lr_adjust=case2 only support for e2e_model'
            print ('set different learning rates for different layers')
            optimizer = optim.Adam([{'params': model.model.pretrain_model.parameters(), 'lr': args.lr/10},
                                    {'params': model.model.encoder.parameters(),        'lr': args.lr},
                                    {'params': model.model.fc_out_1.parameters(),       'lr': args.lr},
                                    {'params': model.model.fc_out_2.parameters(),       'lr': args.lr},
                                    ], lr=args.lr, weight_decay=args.l2)
       

        print (f'Step2: training (multiple epoches)')
        whole_store = []
        whole_metrics = []
        for epoch in range(args.epochs):

            epoch_store = {}

            ## training and validation
            train_results = train_or_eval_model(args, model, reg_loss, cls_loss, train_loader, epoch=epoch, optimizer=optimizer, train=True )
            eval_results  = train_or_eval_model(args, model, reg_loss, cls_loss, eval_loader,  epoch=epoch, optimizer=None,      train=False)
            func_update_storage(inputs=eval_results, prefix='eval', outputs=epoch_store)

            ## use args.metric_name to determine best_index
            train_metric = gain_metric_from_results(train_results, args.metric_name)
            eval_metric  = gain_metric_from_results(eval_results,  args.metric_name)
            whole_metrics.append(eval_metric)
            print ('epoch:%d; metric:%s; train results:%.4f; eval results:%.4f' %(epoch+1, args.metric_name, train_metric, eval_metric))
            if eval_metric > best_metric_so_far:
                best_metric_so_far = eval_metric
                torch.save({
                    'fold': ii + 1,
                    'epoch': epoch + 1,
                    'metric_name': args.metric_name,
                    'metric': float(eval_metric),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args),
                }, best_model_path)

            ## testing and saving
            for jj, test_loader in enumerate(test_loaders):
                test_results = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, epoch=epoch, optimizer=None, train=False)
                func_update_storage(inputs=test_results, prefix=f'test{jj+1}', outputs=epoch_store)
            
            ## saving
            whole_store.append(epoch_store)


        print (f'Step3: saving and testing on the {ii+1} folder')
        best_index = np.argmax(np.array(whole_metrics))
        whole_store[best_index]['best_model_path'] = best_model_path
        folder_save.append(whole_store[best_index])
        end_time = time.time()
        duration = end_time - start_time
        folder_duration.append(duration)
        print (f'>>>>> Finish: training on the {ii+1}-th folder, best_index: {best_index}, duration: {duration} >>>>>')
        # clear memory
        del model
        del optimizer
        torch.cuda.empty_cache()


    print ('====== Prediction and Saving =======')
    args.duration = np.sum(folder_duration) # store duration
    cv_result = gain_cv_results(folder_save)
    save_path = os.path.join(save_resroot, sanitize_filename(f'cv_{safe_prefix_name}_{cv_result}_{name_time}.npz'))
    print (f'save results in {save_path}')
    np.savez_compressed(save_path,
                        args=np.array(args, dtype=object),
                        fold_results=np.array(folder_save, dtype=object),
                        fold_durations=np.array(folder_duration))

    ## store test1|test2|test3 results
    for jj in range(len(test_loaders)):
        emo_labels, emo_probs = average_folder_for_emos(folder_save, f'test{jj+1}')
        val_labels, val_preds = average_folder_for_vals(folder_save, f'test{jj+1}')
        _, test_result = dataloader_class.calculate_results(emo_probs, emo_labels, val_preds, val_labels)
        save_path = os.path.join(save_resroot, sanitize_filename(f'test{jj+1}_{safe_prefix_name}_{test_result}_{name_time}.npz'))
        print (f'save results in {save_path}')
        emo_labels = np.asarray(emo_labels)
        emo_probs = np.asarray(emo_probs)
        emo_preds = np.argmax(emo_probs, axis=1) if len(emo_probs) else np.asarray([])
        val_labels = np.asarray(val_labels)
        val_preds = np.asarray(val_preds)
        np.savez_compressed(save_path,
                            args=np.array(args, dtype=object),
                            emolabels=emo_labels,
                            emoprobs=emo_probs,
                            emopreds=emo_preds,
                            vallabels=val_labels,
                            valpreds=val_preds)
