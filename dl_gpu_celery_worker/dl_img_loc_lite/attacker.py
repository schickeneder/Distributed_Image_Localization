import numpy as np
import os
import pickle
import torch
from locconfig import LocConfig
from torch.utils.data import DataLoader, TensorDataset


def batch_wrapper(batch_tensors, func, batch_size, return_length=1, kwargs={}, shuffle=False, limit_size=True):
    dataset = TensorDataset(*batch_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    result_dict = {k:[] for k in range(return_length)}
    for inputs in dataloader:
        result = func(*inputs, **kwargs)
        if hasattr(result, 'detach'):
            result = result.detach()
        if limit_size:
            for i, res in enumerate(result):
                if i >= return_length: 
                    break
                result_dict[i].append(res)
        else:
            result_dict[0].append(result)
    for key in result_dict:
        result_dict[key] = torch.concat(result_dict[key])
    return result_dict.values()


def get_eta_vec_from_grad(x_vecs, eta, device=None):
    assert len(x_vecs.shape) == 3
    batch_size = eta.shape[0]
    eta_imgs = eta / abs(eta).amax(dim=(1,2,3)).view((-1,1,1,1))
    coords = x_vecs[:,:,1:3].round().long()
    num_sensors = coords.shape[1]
    eta_sensors = eta_imgs[
        torch.arange(batch_size).repeat_interleave(num_sensors),
        1,
        coords[:,:,1].flatten(),
        coords[:,:,0].flatten()
        ].reshape(batch_size, num_sensors) 
    eta_sensors *= (x_vecs[:,:,0] != 0)
    if device is not None:
        return eta_sensors.to(device), eta_imgs.to(device)
    else:
        return eta_sensors, eta_imgs

def naive_random_attack(x_vecs, lower_bound, upper_bound, img_shape):
    adv_x = x_vecs.clone()
    val = np.random.uniform(lower_bound, upper_bound)
    coords = np.array([np.random.randint(1, high=img_shape[-1]-1), np.random.randint(1, high=img_shape[-2]-1)])
    coords = torch.Tensor(coords).float()
    batch_size = img_shape[0]
    for ind in range(batch_size):
        zero_ind = torch.where(x_vecs[ind,:,0] == 0)[0][-1]
        adv_x[ind, zero_ind] = torch.Tensor([val, coords[0], coords[1], zero_ind.float(), 0])
        match_inds = torch.where((coords.cuda() == x_vecs[ind,:,1:3].round()).prod(axis=1))[0]
        adv_x[ind, match_inds,0] = val
    
    return adv_x

def universal_attack(x_vec, rss_vec, img_shape):
    num_pixels = np.prod(img_shape)
    adv_x = x_vec.unsqueeze(0).clone()
    adv_x = adv_x.expand(len(rss_vec), num_pixels,-1,-1).clone()
    coords = torch.stack(torch.meshgrid(torch.arange(img_shape[-1]), torch.arange(img_shape[-2]), indexing='ij')).reshape(2,-1).T
    zero_ind = torch.where(x_vec[:,0] == 0)[0][-1]
    adv_x[:,:,zero_ind,0] = rss_vec[:,None]
    adv_x[:,:,zero_ind, 1:3] = coords
    adv_x[:,:,zero_ind,3] = zero_ind
    return adv_x

def worst_case_attack(x_vecs, y_vecs, dlloc, img_shape, return_many_preds=False, num_return=20):
    num_images = len(x_vecs)
    preds = torch.zeros((2, len(x_vecs), 2)).to(dlloc.device)
    many_preds = [[], []]
    for ind, x_vec in enumerate(x_vecs):
        #print(f"{ind}/{num_images}     ", end='\r')
        adv_x = universal_attack(x_vec, dlloc.rss_tensor, img_shape)
        for i, rss in enumerate(dlloc.rss_tensor):
            pred,  = batch_wrapper( (adv_x[i],), dlloc.predict_img_many, 128, limit_size=False)
            err = torch.linalg.norm(pred - y_vecs[ind,0,1:3], axis=1)
            preds[i,ind] = pred[err.argmax()]
            if return_many_preds:
                top_err, inds = err.topk(num_return)
                many_preds[i].append([pred[inds], adv_x[i][inds], top_err])
    if return_many_preds:
        return preds, many_preds
    return preds

def top_percent_attack_vec(x_vecs, eta_vecs, epsilon, percent, sign=True):
    adv_x = x_vecs.clone()
    for i, eta_vec in enumerate(eta_vecs):
        abs_eta = abs(eta_vec)
        top_mask = abs_eta >= abs_eta[abs_eta > 0].quantile(1-percent)
        adv_x[i,top_mask,0] += eta_vec[top_mask].sign() * epsilon if sign else eta_vec[top_mask] * epsilon
    adv_x[:,:,0] = adv_x[:,:,0].clamp(-0.03, 0.97)
    return adv_x


def dropout_percent_attack_vec(x_vecs, eta_vecs, percent):
    adv_x = x_vecs.clone()
    for i, eta_vec in enumerate(eta_vecs):
        abs_eta = abs(eta_vec)
        top_mask = abs_eta >= abs_eta[abs_eta > 0].quantile(1-percent)
        adv_x[i,top_mask] = 0
    adv_x[:,:,0] = adv_x[:,:,0].clamp(-0.03, 0.97)
    return adv_x


def high_low_eta_attack_vec(eta_imgs, x_vecs, epsilon, num_high=1, num_low=1, sign=False):
    assert len(eta_imgs.shape) == 4
    assert len(x_vecs.shape) == 3
    eta_imgs = eta_imgs[:,1].clone()
    batch_size = eta_imgs.shape[0]
    new_eta = torch.zeros_like(x_vecs)
    high_values, high_inds = eta_imgs.view(batch_size, -1).topk(num_high)
    low_values, low_inds = eta_imgs.view(batch_size, -1).topk(num_low,largest=False)
    high_coords = np.array(np.unravel_index(high_inds.cpu(), eta_imgs.shape[1:]))
    high_coords = torch.Tensor(high_coords).to(new_eta.device)
    low_coords = np.array(np.unravel_index(low_inds.cpu(), eta_imgs.shape[1:]))
    low_coords = torch.Tensor(low_coords).to(new_eta.device)
    for ind in range(batch_size):
        zero_inds = torch.where(x_vecs[ind,:,1] == 0)[0]
        if num_high+num_low > len(zero_inds):
            num_low = len(zero_inds) - num_high
        new_eta[ind,zero_inds[:num_high],0] = epsilon if sign else high_values[ind] * epsilon
        new_eta[ind,zero_inds[:num_high],1] = high_coords[1,ind]
        new_eta[ind,zero_inds[:num_high],2] = high_coords[0,ind]
        new_eta[ind,zero_inds[num_high:num_high+num_low],0] = -epsilon if sign else low_values[ind,:num_low] * epsilon
        new_eta[ind,zero_inds[num_high:num_high+num_low],1] = low_coords[1,ind,:num_low]
        new_eta[ind,zero_inds[num_high:num_high+num_low],2] = low_coords[0,ind,:num_low]

        new_eta[ind,zero_inds[:num_high+num_low],3] = zero_inds[:num_high+num_low].float()
        new_eta[ind,zero_inds[:num_high+num_low],4] = 0 # Each fake sensor is a bus?
    return new_eta + x_vecs


def get_random_attack_vec(x_vecs, grad, rss_low, rss_high, epsilon=0.5):
    eta_vecs, eta_imgs = get_eta_vec_from_grad(x_vecs, grad)
    rand_select = np.random.randint(9)
    rand_epsilon = np.random.uniform(0,epsilon)
    hilo_epsilon = np.random.uniform(0.3,0.7)
    if rand_select == 0:
        return top_percent_attack_vec(x_vecs, eta_vecs, rand_epsilon, 1)
    elif rand_select == 1:
        return top_percent_attack_vec(x_vecs, eta_vecs, rand_epsilon, np.random.random()/2)
    elif rand_select == 2:
        return dropout_percent_attack_vec(x_vecs, eta_vecs, np.random.random()/3)
    elif rand_select == 3:
        return high_low_eta_attack_vec(eta_imgs, x_vecs, hilo_epsilon, np.random.randint(1,6), 0)
    elif rand_select == 4:
        return high_low_eta_attack_vec(eta_imgs, x_vecs, hilo_epsilon, 0, np.random.randint(1,6))
    elif rand_select == 5:
        return high_low_eta_attack_vec(eta_imgs, x_vecs, hilo_epsilon, np.random.randint(1,6), np.random.randint(1,6))
    elif rand_select == 6:
        adv_x = top_percent_attack_vec(x_vecs, eta_vecs, rand_epsilon, np.random.random()/2)
        return high_low_eta_attack_vec(eta_imgs, adv_x, hilo_epsilon, np.random.randint(1,6), np.random.randint(1,6))
    elif rand_select == 7:
        adv_x = top_percent_attack_vec(x_vecs, eta_vecs, rand_epsilon, 0.01)
        return high_low_eta_attack_vec(eta_imgs, adv_x, hilo_epsilon, np.random.randint(1,6), np.random.randint(1,6))
    elif rand_select == 8:
        return naive_random_attack(x_vecs, rss_low, rss_high, eta_imgs.shape)

def get_all_attack_preds_without_grad(x_vecs, y_vecs, dlloc, img_shape, include_worst_case=True):
    pred_img, x_img, y_img = dlloc.model((x_vecs, y_vecs))
    if isinstance(dlloc.loss_func, torch.nn.MSELoss):
        loss = dlloc.loss_func(pred_img, y_img)
    else:
        loss = dlloc.loss_func(pred_img, y_img, y_vecs)
    loss.backward()
    grad = x_img.grad.data.clone()
    eta_vecs, eta_imgs = get_eta_vec_from_grad(x_vecs, grad)
    top20_x = top_percent_attack_vec(x_vecs, eta_vecs, 0.2, percent=0.2)
    top20_y = dlloc.predict_img_many(top20_x)
    drop20_x = dropout_percent_attack_vec(x_vecs, eta_vecs, percent=0.2)
    drop20_y = dlloc.predict_img_many(drop20_x)
    hilo_x = high_low_eta_attack_vec(eta_imgs, x_vecs, epsilon=0.5, num_high=1, num_low=5)
    hilo_y = dlloc.predict_img_many(hilo_x)
    adv_x = top_percent_attack_vec(x_vecs, eta_vecs, 0.5, percent=0.01)
    hilotop1_x = high_low_eta_attack_vec(eta_imgs, adv_x, 0.5, 1, 5)
    hilotop1_y = dlloc.predict_img_many(hilotop1_x)
    if include_worst_case:
        worst = worst_case_attack(x_vecs, y_vecs, dlloc, img_shape)
        worst_lo, worst_hi = worst
        return worst_lo.detach(), worst_hi.detach(), top20_y.detach(), drop20_y.detach(), hilo_y.detach(), hilotop1_y.detach()
    return top20_y.detach(), drop20_y.detach(), hilo_y.detach(), hilotop1_y.detach()


def main():
    raise NotImplementedError
    from localization import DLLocalization
    from dataset import RSSLocDataset
    train_set_repeats = 1
    min_sensors = 6
    arch = 'unet' # Model architecture
    # Data augmentation: Randomly select only a subset of receivers for evaluation
    sensor_dropout = False
    should_augment = False
    force_num_tx = 1

    # Change 'cuda' to 'cpu' if no gpu available. If many gpu, I think you set to 'cuda:0' 
    device = torch.device('cuda')

    # The random split of train/train_val. For now, we can just use an initial split of 0
    #for random_state in range(0,num_training_repeats):
    random_state = 0
    # The split is how we are separating our train/test data. 'random' is just a random shuffle. 
    # 'gridX' where x is some integer, divides campus into an X by X grid and assigns transmitters
    # occuring in each grid to either train or test. This is used to artificially separate our 
    # train and test distributions, in order to show generalization.
    split = 'random'
    # meter_scale is the image resolution, or number of meters per pixel. Total campus area is about 2300x2400 meters,
    meter_scale = 60
    # Provide elevation map as an additional input channel (seems to hurt performance)
    include_elevation_map = False
    # Learn some device-specific linear parameters (y=ax+b, learn a and b) that are applied to the input RSS values
    device_multiplication = False 
    # Learn some category-specific linear parameters (y=ax+b, learn a and b) that are applied to the input RSS values. There are 4 radio categories: Buses, Rooftop CBRS nodes, Dense Deployment nodes, and Fixed endpoints on side of buildings. See POWDER webside for more details.
    category_multiplication = True

    batch_size = 32 # You can probably increase the batch size, depending on meter_scale and GPU memory
    # See Params.py for an extensive list of dataset params, some of which may not be helpful.
    params = LocConfig(include_elevation_map=include_elevation_map, random_state=random_state, batch_size=batch_size, augment_data=should_augment, force_num_tx=force_num_tx, data_split=split, arch=arch, device_multiplication=device_multiplication, category_multiplication=category_multiplication, sensor_dropout=sensor_dropout, meter_scale=meter_scale, device=device)
    linear_params =  LocConfig(include_elevation_map=include_elevation_map, random_state=random_state, batch_size=batch_size, augment_data=should_augment, force_num_tx=force_num_tx, data_split=split, arch='unet_notarget_linear', device_multiplication=device_multiplication, category_multiplication=category_multiplication, sensor_dropout=sensor_dropout, meter_scale=meter_scale, device=device)
    

    # rldataset is the dataset of sensors. You can look at the definitions, but relevant members are:
    # rldataset.data[key].tx_vecs : A (N,k,3) shape numpy array of transmitter locations, where N is the number of unique samples, k is the number of active transmitters (usually 1). Last dimension has [1, xcoord, ycoord]
    # rldataset.data[key].rx_vecs : An array of N numpy arrays, (k,5) in size, of receiver data, where N is the number of unique samples, k is the number of active receivers (varies from 10 to 25). Last dimension has [uncalibrated_rss_value, xcoord, ycoord, device_id (0-33), category_id(1-4)].
    # If you're looking to edit the inputs like an adversary might, you could edit rss values or location info in rldataset.data[test_key].rx_vecs.

    # rldataset.set_default_keys(train_key, test_keys=test_keys) will make pytorch tensors for each dataset key provided. 
    # rldataset.data[key].ordered_dataset.tensors : The ordered Rx and Tx tensors. Some entries will be zero, since not every sample has the same number of Tx or Rx.
    rldataset = RSSLocDataset(params, random_state=random_state)
    train_key, test_key = rldataset.make_datasets(make_val=False)
    train_data, test_data = rldataset.data[train_key], rldataset.data[test_key]
    #rldataset.print_dataset_stats()

    # dlloc is the model object that contains the training/eval functions, etc.
    dlloc = DLLocalization(rldataset)
    dlloc_linear = DLLocalization(RSSLocDataset(linear_params))
    dlloc_linear.load_model('final_models/unet_notarget_linear_60m_test_fit.pt')
    for model_path in ['final_models/final_model_60m.pt', 'final_models/unet_adversarial_training_60m.pt']:
        dlloc.load_model(model_path)

        train_loss, train_res, train_images = dlloc.get_results(train_data, force_ordered=True, save_images=True)
        test_loss, test_res, test_images = dlloc.get_results(test_data, force_ordered=True, save_images=True)
        ### train_res and test_res are dicts with all the results. You can look at test_res['mean_error'] or test_res['error'] for more concrete results.
        dlloc.best_threshold = train_res['thresh']
        dlloc.best_size = train_res['size']
        tr_truth = np.array(train_res['truth'])
        tr_preds = np.array(train_res['preds'])
        tr_err = np.array(train_res['error']).flatten()
        tr_max = train_images[0].reshape(len(train_images[0]), -1).max(axis=1)
        te_truth = np.array(test_res['truth']) # (N,X,2) np array of actual transmitter locations, where X is the number of transmitters (usually 1)
        te_preds = np.array(test_res['preds']) # (N,X,3) np array of prediction transmitter locations, plus max predicted values per image
        te_err = np.array(test_res['error']).flatten() # Overall localization error
        te_max = te_preds[:,:,2] #" Confidence,", or  Max predicted value per image, 

        linear_train_loss, linear_train_res = dlloc_linear.get_results(train_data, force_ordered=True)
        linear_test_loss, linear_test_res = dlloc_linear.get_results(test_data, force_ordered=True)
        ### train_res and test_res are dicts with all the results. You can look at test_res['mean_error'] or test_res['error'] for more concrete results.
        linear_tr_truth = np.array(linear_train_res['truth'])
        linear_tr_preds = np.array(linear_train_res['preds'])
        linear_tr_err = np.array(linear_train_res['error']).flatten()
        linear_te_truth = np.array(linear_test_res['truth']) # (N,X,2) np array of actual transmitter locations, where X is the number of transmitters (usually 1)
        linear_te_preds = np.array(linear_test_res['preds']) # (N,X,3) np array of prediction transmitter locations, plus max predicted values per image
        linear_te_err = np.array(linear_test_res['error']).flatten() # Overall localization error

        #make_error_line_plot(test_res)

        all_simple_vecs = [dlloc_linear.make_simple_tensors(x_vec, y_vec) for x_vec, y_vec in zip(test_data.rx_vecs, test_data.tx_vecs)]
        all_simple_x_vecs = torch.concat([vec[0] for vec in all_simple_vecs])
        all_simple_y_vecs = torch.concat([vec[1] for vec in all_simple_vecs])
        all_imgs = [dlloc_linear.vec2im((all_simple_x_vecs, all_simple_y_vecs))]
        all_x_imgs = torch.concat([img[0] for img in all_imgs])
        all_y_imgs = torch.concat([img[1] for img in all_imgs])

        def predict_on_image(x_img):
            pred_image = dlloc.model(x_img).detach().squeeze()
            pred = np.stack([np.array(np.unravel_index(torch.argmax(ten).item(), ten.shape))  for ten in pred_image])
            return pred # in img frame (simple vec)

        def get_gradient_vecs():
            batch_base = 0
            batch_size = 64
            all_eta_vecs = []
            while batch_base < len(all_x_imgs):
                x_imgs = all_x_imgs[batch_base:batch_base+batch_size].detach()
                x_imgs.requires_grad = True
                x_imgs.retain_grad()
                pred_img, pred_vec = dlloc_linear.model(x_imgs)
                loss = nn.functional.mse_loss(all_simple_y_vecs[batch_base:batch_base+batch_size,:,1:], pred_vec[:,:,1:])
                dlloc_linear.model.zero_grad()
                loss.backward()
                eta = x_imgs.grad.data.clone()
                all_eta_vecs.append(eta)
                batch_base += batch_size
            return torch.concat(all_eta_vecs)

        rss_inds = train_data.ordered_dataloader.dataset.tensors[0][:,:,0] != 0
        all_rss = train_data.ordered_dataloader.dataset.tensors[0][:,:,0][rss_inds]
        lower_bound, upper_bound = np.quantile(all_rss.cpu(), [0.1, 0.9])


        for attack in ['naive', 'worst_case']:
            error_dict = {}
            data_file = f'{attack}_attack2.pkl' if 'adversarial' not in model_path else f'{attack}_attack2_adversarial.pkl'
            error_dict['test_err'] = te_err
            random_samples = 600
            errors = np.zeros((len(te_err), random_samples))

            if os.path.exists(data_file):
                with open(data_file, 'rb') as infile:
                    error_dict = pickle.load(infile)
            else:
                if attack == 'naive':
                    for ind in range(random_samples):
                        print(f"{ind}/{random_samples}     ", end='\r')
                        adv_x = naive_random_attack(all_simple_x_vecs, lower_bound, upper_bound, all_x_imgs.shape)
                        pred, _, _ = batch_wrapper( (adv_x,), dlloc.predict_img_many, 64, return_length=3, kwargs={"simple_vecs":True})
                        error = np.linalg.norm(pred.cpu() - all_simple_y_vecs[:,0,1:3].cpu(), axis=1)
                        errors[:, ind] = error
                    with open(data_file, 'wb') as outfile:
                        error_dict['naive'] = errors
                        pickle.dump(error_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)

                elif attack == 'worst_case':
                    rss_vec = [lower_bound, upper_bound]
                    pred_dict = worst_case_attack(all_simple_x_vecs, rss_vec, dlloc, all_x_imgs.shape)
                    for rss in rss_vec:
                        error_dict[rss] = np.linalg.norm(pred_dict[rss] - all_simple_y_vecs[:,:,1:3].cpu().numpy(), axis=2)

                    with open(data_file, 'wb') as outfile:
                        pickle.dump(error_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
                
                else:
                    error("Not naive or worst_case attack!")


        def plot_imgs(ind=0):
            fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
            axs.flat[0].imshow(all_x_imgs[ind,1].detach().cpu(), origin='lower')
            axs.flat[1].imshow(all_y_imgs[ind,0].detach().cpu(), origin='lower')
            axs.flat[2].imshow(all_eta_imgs[ind,1].detach().cpu(), origin='lower')
            axs.flat[3].imshow((all_eta_imgs[ind,0]*(all_x_imgs[ind,0]!=0)).detach().cpu(), origin='lower')
            plt.show()




if __name__ == '__main__':
    main()
