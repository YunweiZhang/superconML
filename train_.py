import random
from utils_ import interleave,SemiLoss,prob,validate,get_conductor, ConductorNet, AverageMeter, result_out
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim


def main():

    acc_list = []
    prob_val_list = []
    f_final_list_1 = []
    f_final_list_2 = []


    for random_state in range(200):
        train_labeled_set, train_unlabeled_set_1, train_unlabeled_set_2, test_set, name_list_1,name_list_2 = get_conductor(random_state)
        model = ConductorNet()
        model = model.cpu()

        train_criterion = SemiLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        test_accs = []
        f_list_1 = []
        f_list_2 = []

        prob_mat = []
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        model.train()
        for batch_idx in range(1,2501):

            batch_size = 25

            batch_l = random.sample(range(1, np.shape(train_labeled_set['targets'])[0]), batch_size)
            batch_u = random.sample(range(1, np.shape(train_unlabeled_set_1['targets'])[0]), batch_size)

            inputs_x = torch.from_numpy(train_labeled_set['data'][batch_l, :]).float()
            targets_xx = torch.from_numpy(train_labeled_set['targets'][batch_l]).long()

            inputs_u = torch.from_numpy(train_unlabeled_set_1['data'][batch_u, :]).float()


            targets_xx = torch.zeros(batch_size, 2).scatter_(1, targets_xx.view(-1, 1), 1)

            with torch.no_grad():
                #compute guessed labels of unlabel samples
                outputs_u = model(inputs_u)



            #mixup
            all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
            all_targets = torch.cat([targets_xx, outputs_u.detach()], dim=0)

            l = np.random.beta(0.1, 0.1)

            l = max(l, 1 - l)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            logits = [model(mixed_input[0])]
            for input in mixed_input[1:]:
                logits.append(model(input))

            # put interleaved samples back
            logits = interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            Lx, Lu = train_criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:])

            loss = Lx + 0.1 * Lu

            # record loss
            losses.update(loss.item(), inputs_x.size(0))
            losses_x.update(Lx.item(), inputs_x.size(0))
            losses_u.update(Lu.item(), inputs_x.size(0))


            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # measure elapsed time

            if batch_idx%100==0 and batch_idx>=2000:

                test_acc1, prob_val = validate(test_set, model)

                test_accs.append(test_acc1)
                prob_mat.append(prob_val)

                prob_list_1 = prob(train_unlabeled_set_1, model)
                prob_list_1 = np.squeeze(prob_list_1)[:, 1]
                f_list_1.append(prob_list_1)

                prob_list_2 = prob(train_unlabeled_set_2, model)
                prob_list_2 = np.squeeze(prob_list_2)[:, 1]
                f_list_2.append(prob_list_2)


        prob_ave = np.mean(prob_mat,axis=0)[:,1]
        prob_val_list.append([prob_ave, np.squeeze(test_set['targets'])])
        acc = np.squeeze(test_set['targets']) == (prob_ave>0.5)
        acc_list.append(acc)

        result_1, f_final_list_1 = result_out(f_list_1, f_final_list_1, name_list_1)
        result_2, f_final_list_2 = result_out(f_list_2, f_final_list_2, name_list_2)

        result_1.to_csv('counts_binary.csv', index=False,header=['Name','Code', 'Prob','Prob_std'])
        result_2.to_csv('counts_ternary.csv', index=False, header=['Name', 'Code', 'Prob', 'Prob_std'])

        import pickle
        pickle.dump(prob_val_list, open('prob.csv', "wb"))

        print(random_state, 'ave. acc:', np.mean(acc_list))

        del model



if __name__ == '__main__':
    main()
