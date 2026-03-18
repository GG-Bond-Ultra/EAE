'''Train DYNN from checkpoint of trained backbone'''
import itertools
import os
import torch
# import mlflow
from timm.models import *
from timm.models import create_model
from collect_metric_iter import aggregate_metrics, process_things
from conformal_eedn import compute_conf_threshold
from data_loading.data_loader_helper import get_path_to_project_root, split_dataloader_in_n,  get_abs_path
from learning_helper import LearningHelper
from log_helper import aggregate_metrics_mlflow
from classifier_training_helper import LossContributionMode
from utils import aggregate_dicts, progress_bar
from early_exit_utils import switch_training_phase
from gfnet_dynn import TrainingPhase
import numpy as np
import pickle as pk
import utils
from timm.utils import accuracy
def display_progress_bar(prefix_logger, training_phase, step, total, log_dict):
    loss = log_dict[prefix_logger+'/loss']
    if training_phase == "warm_up":
        progress_bar(step, total,'Loss: %.3f | Warmup' % (loss))
    elif training_phase == "classifier":
        gated_acc = log_dict[prefix_logger+'/gated_acc']
        progress_bar(
                step, total,
                'Cls_Loss: %.3f | Cls_Acc: %.3f%%' %
                (loss, gated_acc))
    elif training_phase == "gate":
        progress_bar(step, total, 'Gate Loss: %.3f ' % (loss)) 

def evaluate(best_acc, args, helper: LearningHelper, device, init_loader, epoch, mode: str, experiment_name: str, store_results=False):
    helper.net.eval()
    num_layers = len(helper.net.module.blocks)
    gate_positions = helper.net.module.intermediate_head_positions
    metrics_dict = {}
    if mode == 'test': # we should split the data and combine at the end
        loaders = split_dataloader_in_n(init_loader, n=10)
    else:
        loaders = [init_loader]
    metrics_dicts = []
    log_dicts_of_trials = {}
    average_trials_log_dict = {}
    for loader in loaders:
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = targets.size(0)
            loss, things_of_interest = helper.get_surrogate_loss(inputs, targets)
            
            # obtain the metrics associated with the batch
            metrics_of_batch = process_things(things_of_interest, gate_positions=gate_positions,
                                              targets=targets, batch_size=batch_size,
                                              cost_per_exit=helper.net.module.mult_add_at_exits, num_layers=num_layers)
            metrics_of_batch['loss'] = (loss.item(), batch_size)
            

            # keep track of the average metrics
            metrics_dict = aggregate_metrics(metrics_of_batch, metrics_dict, gate_positions=gate_positions)

            # format the metric ready to be displayed
            log_dict = aggregate_metrics_mlflow(
                    prefix_logger=mode,
                    metrics_dict=metrics_dict, gate_positions=gate_positions)
            display_progress_bar(prefix_logger=mode, training_phase="classifier", step=batch_idx, total=len(loader), log_dict=log_dict)

            if args.barely_train:
                    if batch_idx > 50:
                        print(
                            '++++++++++++++WARNING++++++++++++++ you are barely testing to test some things'
                        )
                        break
        metrics_dicts.append(metrics_dict)
        for k, v in log_dict.items():
            aggregate_dicts(log_dicts_of_trials, k, v)
    for k,v in log_dicts_of_trials.items():
        average_trials_log_dict[k] = np.mean(v)
    
    gated_acc = average_trials_log_dict[mode+'/gated_acc']
    average_trials_log_dict[mode+'/test_acc']= gated_acc
    # mlflow.log_metrics(average_trials_log_dict, step=epoch)
    # Save checkpoint.
    print("gated_acc:",gated_acc)
    # if gated_acc > best_acc and mode == 'val' and store_results:
    if gated_acc > best_acc and mode == 'val':
        print('Saving..')
        state = {
            'model': helper.net.module.state_dict(),
            'acc': gated_acc,
            'epoch': epoch,
        }
        checkpoint_path = os.path.join(get_path_to_project_root(), 'checkpoint')
        this_run_checkpoint_path = os.path.join(checkpoint_path, f'checkpoint_{args.dataset}_{args.arch}_confEE')
        if not os.path.isdir(this_run_checkpoint_path):
            os.mkdir(this_run_checkpoint_path)
        torch.save(
            state,
            os.path.join(this_run_checkpoint_path,f'ckpt_{args.ce_ic_tradeoff}_{gated_acc}.pth')
        )
        best_acc = gated_acc
        
    
    elif mode == 'test' and store_results:
        print('storing results....')
        with open(experiment_name+'_'+args.dataset+"_"+args.arch+"_"+str(args.ce_ic_tradeoff)+'_results.pk', 'wb') as file:
            pk.dump(log_dicts_of_trials, file)
    return metrics_dict, best_acc, log_dicts_of_trials

# Any action based on the validation set
def set_from_validation(learning_helper, val_metrics_dict, freeze_classifier_with_val=False, alpha_conf = 0.04):
   
    # we fix the 1/0 ratios of of gate tasks based on the optimal percent exit in the validation sets
    
    exit_count_optimal_gate = val_metrics_dict['exit_count_optimal_gate'] # ({0: 0, 1: 0, 2: 0, 3: 0, 4: 6, 5: 72}, 128)
    total = exit_count_optimal_gate[1]
    pos_weights = []
    pos_weights_previous = []
    for gate, count in exit_count_optimal_gate[0].items():
        count = max(count, 0.1)
        pos_weight = (total-count) / count # #0/#1
        pos_weight = min(pos_weight, 5) # clip for stability
        pos_weights.append(pos_weight)



    learning_helper.gate_training_helper.set_ratios(pos_weights)
    
    

    ## compute the quantiles for the conformal intervals
    
    mixed_score, n = val_metrics_dict['gated_score']
    scores_per_gate, n = val_metrics_dict['score_per_gate']
    score_per_final_gate, n = val_metrics_dict['score_per_final_gate']

    all_score_per_gates, n = val_metrics_dict['all_score_per_gate']
    all_final_score, n = val_metrics_dict['all_final_score']

    alpha_qhat_dict = compute_conf_threshold(mixed_score, scores_per_gate+[score_per_final_gate], all_score_per_gates+[all_final_score])
    

    learning_helper.classifier_training_helper.set_conf_thresholds(alpha_qhat_dict)
   


def perform_test(model, test_loader, threshold, flops, device):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()
    n_blocks = len(model.module.blocks)
    
    n_stage = n_blocks
    exp = torch.zeros(n_stage)
    exp_correct = torch.zeros(n_stage)
    model.eval()
    # test_meter.iter_tic()
    num = 0
    acc = 0
    header = 'Val:'
    for images, target in metric_logger.log_every(test_loader, 500, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        num = num + len(target)
        # test_meter.data_toc()

        # Perform the forward pass.
        
        with torch.cuda.amp.autocast():
            final_logits, intermediate_logits, Gates = model.module.forward_for_inference(images)
            intermediate_logits.append(final_logits)
        
        Gates = torch.cat(Gates, dim=1)
        actual_exits_binary = torch.nn.functional.sigmoid(Gates)
        s = actual_exits_binary.sum(dim=0)
        output = intermediate_logits
        for i in range(final_logits.size(0)):
            for j in range(n_stage):
                    if j < n_stage - 1:
                        if actual_exits_binary[i][j] >= threshold: #确定退出
                            pred = output[j]
                            pred = pred[i, :]
                            max_preds, argmax_preds = pred.max(dim=0, keepdim=False)
                            if argmax_preds == target[i]:
                                acc += 1
                                exp_correct[j] += 1
                            exp[j] += 1
                            break
                    else:
                        pred = output[j]
                        pred = pred[i, :]
                        max_preds, argmax_preds = pred.max(dim=0, keepdim=False)

                        if argmax_preds == target[i]:
                            acc += 1
                            exp_correct[j] += 1
                        exp[j] += 1
        
    exit_rate = [0] * n_stage
    exit_correct_rate = [0] * n_stage
    print(num)
    print(acc)
    print(f"****************{threshold}****************")
    expected_GFLOPs = 0
    for k in range(n_stage):
        exit_rate[k] = exp[k] * 100.0 / num
        if exp[k] == 0:
            exit_correct_rate[k] = 0
        else:
            exit_correct_rate[k] = exp_correct[k] * 100.0 / exp[k]
        print(f"Exiting Layer{k}:[{exit_rate[k]}/{exit_correct_rate[k]}]")
            # _t = 1.0 * exp[k] / n_sample
        expected_GFLOPs += exit_rate[k] * flops[k]
    acc_correct = acc / num * 100
    expected_GFLOPs = expected_GFLOPs / 100
    print(f"acc_val={acc_correct},total_GFLOPs={expected_GFLOPs}G")
    print(f"****************{threshold}****************")
    return acc_correct, expected_GFLOPs