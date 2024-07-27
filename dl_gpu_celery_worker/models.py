import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from locconfig import LocConfig
import ot


class SlicedEarthMoversDistance(nn.Module):
    def __init__(self, num_projections=100, reduction='mean', scaling=1.0, p=1, normalize=True, device='cuda') -> None:
        super().__init__()
        if reduction == 'mean':
            self.reduction = torch.mean
        elif reduction == 'none':
            self.reduction = torch.nn.Identity()
        elif reduction == 'sum':
            self.reduction = torch.sum
        self.num_proj = num_projections
        self.eps = 1e-6
        self.scaling = scaling
        self.p = p
        self.normalize = normalize

    def forward(self, X, Y, *args):
        batch_tuple = X.shape[:-2]
        flat_X = X.reshape(batch_tuple + (-1,))

        # If max is 0, add epsilon
        max_vals, max_inds = flat_X.max(dim=-1)
        should_max = max_vals[:,0] < self.eps
        flat_X[should_max,0,max_inds[should_max,0]] = self.eps
        X = torch.mean(X, dim=1, keepdim=True)

        x = X[0,0]
        y = Y[0,0]
        x_coords = torch.nonzero(x > 0).float() / self.scaling
        y_coords = torch.nonzero(y > 0).float() / self.scaling
        dists = []
        if self.normalize:
            loss, projections = ot.sliced_wasserstein_distance(x_coords, y_coords, x[x>0]/x.sum(), y[y>0]/y.sum(),p=self.p, n_projections=self.num_proj, log=True)
        else:
            loss, projections = ot.sliced_wasserstein_distance(x_coords, y_coords, x[x>0], y[y>0],p=self.p, n_projections=self.num_proj, log=True)
        projections = projections['projections']
        for x, y in zip(X[1:],Y[1:]):
            x = x[0]
            y = y[0]
            x_coords = torch.nonzero(x > 0).float() / self.scaling
            y_coords = torch.nonzero(y > 0).float() / self.scaling
            if self.normalize:
                loss += ot.sliced_wasserstein_distance(x_coords, y_coords, x[x>0]/x.sum(), y[y>0]/y.sum(),p=self.p, projections=projections)
            else:
                loss += ot.sliced_wasserstein_distance(x_coords, y_coords, x[x>0], y[y>0],p=self.p, projections=projections)
        return loss


class CoMLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, pred_img, truth_img, tx_truth_coords):
        mean_pred = pred_img.mean(axis=1)
        centers_of_mass = get_centers_of_mass(mean_pred)
        error = torch.linalg.norm(tx_truth_coords[:,0,1:] - centers_of_mass, axis=1)
        return error.mean()


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, dropout=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.padding = kernel_size // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=self.padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=self.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels=None, res_channels=None, bilinear=True, kernel_size=3, dropout=0):
        super().__init__()

        if res_channels is None:
            res_channels = out_channels

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels+res_channels, out_channels, mid_channels=mid_channels, kernel_size=kernel_size, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=mid_channels, kernel_size=kernel_size, dropout=dropout)
        self.use_res = res_channels != 0

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if self.use_res:
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)
        else:
            return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)


class Vec2Im(nn.Module):
    def __init__(self, config: LocConfig,
                img_size: np.ndarray, 
                device,
                force_random_power_on_eval=False,
                elevation_map: torch.FloatTensor=None,
                max_num_rx=34,
                num_rx_categories=4,
                force_dropout_on_eval=False
    ):
        super(Vec2Im, self).__init__()
        self.img_size = torch.tensor(img_size).to(device)
        self.device = device
        self.n_channels = 3 if config.include_elevation_map else 2
        self.config = config

        self.force_random_power_on_eval = force_random_power_on_eval
        self.elevation_map = elevation_map
        self.force_dropout_on_eval = force_dropout_on_eval

        self.rand_generator = torch.Generator(device=self.device)
        self.rand_generator.manual_seed(config.random_state)
        self.device_weights = nn.Parameter(torch.ones(max_num_rx))
        self.device_bias = nn.Parameter(torch.zeros(max_num_rx))
        self.category_weights = nn.Parameter(torch.ones(num_rx_categories+1))
        self.category_bias = nn.Parameter(torch.zeros(num_rx_categories+1))

    
    def dropout(self, rx_vec):
        num_sensors = (rx_vec[:,:,0] != 0).sum(axis=1)
        for i, num in enumerate(num_sensors):
            nonzeros = rx_vec[i,:,0].nonzero()
            if num > self.config.min_dropout_inputs:
                num_indices = torch.randint(self.config.min_dropout_inputs, num, size=(1,)).long().item()
                to_remove = torch.randperm(num)[num_indices:]
                rx_vec[i,nonzeros[to_remove]] = 0
        return rx_vec

    
    def forward(self, x):
        rect_ids = None
        y_vecs = None
        if isinstance(x,tuple) and len(x) == 3:
            x_vecs, rect_ids, y_vecs = x
        elif isinstance(x,tuple) and len(x) == 2:
            x_vecs, y_vecs = x
        else:
            x_vecs = x
        if len(x_vecs.shape) < 3:
            batch_size = 1
            x_vecs = x_vecs.unsqueeze(0)
        batch_size = x_vecs.shape[0]

        # Make x_img
        x_img = torch.zeros((batch_size, self.n_channels, self.img_size[0], self.img_size[1]), device=self.device)

        with torch.no_grad():
            if self.config.sensor_dropout and (self.training or self.force_dropout_on_eval):
                x_vecs = self.dropout(x_vecs)
        all_powers = x_vecs[:,:,0].clone()
        power_inds =  all_powers != 0
        if self.config.device_multiplication:
            all_dbias = (all_powers != 0)*self.device_bias
            all_powers = all_powers*self.device_weights
            all_powers += all_dbias
        if self.config.category_multiplication:
            all_device_categories = x_vecs[:,:,-1]
            all_cweights = torch.take(self.category_weights, all_device_categories.long())
            all_cbias = (power_inds)*torch.take(self.category_bias, all_device_categories.long())
            all_powers = all_powers * all_cweights
            all_powers += all_cbias
        coords = x_vecs[:,:,1:3].round().long().cpu().numpy()
        if self.config.apply_rss_noise and (self.training or self.force_random_power_on_eval): 
            #We should only set random power in train mode.
            all_powers[power_inds] = all_powers[power_inds] + ((torch.rand(all_powers[power_inds].shape, generator=self.rand_generator, device=self.device) - 0.5) * 2*self.config.power_limit)
        if self.config.apply_power_scaling and (self.training or self.force_random_power_on_eval): 
            all_powers[power_inds] += (torch.rand(1, generator=self.rand_generator, device=self.device) - 0.5) * self.config.scale_limit*2
        x_img[torch.arange(batch_size).repeat_interleave(all_powers.shape[-1]), 0, coords[:,:,1].flatten(), coords[:,:,0].flatten()] = all_powers.flatten()
        ### TODO: This should be including the inputs with RSS noise and power scaling, but without category/device multiplication.
        x_img[torch.arange(batch_size).repeat_interleave(all_powers.shape[-1]), 1, coords[:,:,1].flatten(), coords[:,:,0].flatten()] = x_vecs[:,:,0].flatten()

        # Make y_img
        tx_marker_size = 3
        tx_marker_value = self.config.tx_marker_value
        if y_vecs is not None:
            y_img = torch.zeros((batch_size, 1, self.img_size[0], self.img_size[1]), device=self.device)
            y_vecs = y_vecs.clone()
            pad = tx_marker_size // 2
            if isinstance(tx_marker_value, float):
                marker_value = tx_marker_value
            else:
                x_grid,y_grid = np.meshgrid( np.linspace(-(tx_marker_size//2), tx_marker_size//2, tx_marker_size),  np.linspace(-(tx_marker_size//2), tx_marker_size//2, tx_marker_size) )
                dst = np.sqrt(x_grid*x_grid + y_grid*y_grid)
                marker_value = np.exp(-( (dst)**2 / (2.0*tx_marker_value[1]**2)))

            ind0, ind1 = torch.where(y_vecs[:,:,0])
            coords = y_vecs[ind0, ind1, 1:3].round().long()
            y_img[ind0, 0, coords[:,1], coords[:,0]] = 1.0 - 8*marker_value
            pads = [
                [-1,-1],
                [-1, 0],
                [-1, 1],
                [ 0,-1],
                [ 0, 1],
                [ 1,-1],
                [ 1, 0],
                [ 1, 1],
            ]
            for shift in pads:
                edge_coords = coords.clone() + torch.Tensor(shift).long().to(self.device)
                valid_inds = (edge_coords.min(axis=1)[0] >= 0) * (edge_coords[:,0] < self.img_size[1].cpu()) * (edge_coords[:,1] < self.img_size[0].cpu())
                y_img[ind0[valid_inds], 0, edge_coords[valid_inds,1], edge_coords[valid_inds,0]] = marker_value

        if self.config.include_elevation_map and self.elevation_map is not None:
            size = self.elevation_map.shape
            x_img[:,-1,1:1+size[0],1:1+size[1]] = self.elevation_map
        if self.config.adv_train:
            x_img = x_img.detach()
            x_img.requires_grad = True
            x_img.retain_grad()
        if y_vecs is not None:
            return x_img, y_img, y_vecs
        else:
            return x_img



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=1, channel_scale=64, kernel_size=3, out_kernel_size=1, bilinear=True, depth=4, use_residual=True, dropout=0):
        super(UNet, self).__init__()
        self.n_channels = n_channels + 1
        self.n_classes = n_classes
        self.use_residual = int(use_residual)

        mult = [1,2,4,8,16,32,64,128,256]
        if depth > 7:
            raise NotImplementedError
        
        self.inc = DoubleConv(self.n_channels, channel_scale, kernel_size=kernel_size, dropout=dropout)
        factor = 2 if bilinear else 1
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(depth):
            if i < depth-1:
                self.downs.append(Down(channel_scale*mult[i], channel_scale*mult[i+1], kernel_size=kernel_size, dropout=dropout))
            else:
                self.downs.append(Down(channel_scale*mult[i], channel_scale*mult[i+1] // factor, kernel_size=kernel_size, dropout=dropout))
        for i in range(depth):
            if i < depth-1:
                self.ups.append(Up(channel_scale*mult[depth-i] // factor, channel_scale*mult[depth-1-i] // factor, bilinear=bilinear, res_channels=self.use_residual*channel_scale*mult[depth-i] // factor, kernel_size=kernel_size, dropout=dropout))
            else:
                self.ups.append(Up(channel_scale*mult[depth-i] // factor, channel_scale*mult[depth-1-i], bilinear=bilinear, res_channels=self.use_residual*channel_scale*mult[depth-i] // factor, kernel_size=kernel_size, dropout=dropout))
        self.outc = nn.Sequential(
            OutConv(channel_scale, n_classes, kernel_size=out_kernel_size),
            nn.Sigmoid(),
            nn.ReLU())

    def forward(self, x):
        x1 = self.inc(x)
        x_values = [x1]
        for down in self.downs:
            x_values.append( down(x_values[-1]) )
        x = x_values[-1]
        ind = -2
        for up in self.ups:
            x = up(x, x_values[ind])
            ind += -1
        logits = self.outc(x)
        return logits


class MLPLocalization(nn.Module):
    def __init__(self, in_features, out_features=2, channel_scale=256, device=torch.device('cuda')):
        super().__init__()
        num_features = [
            in_features,
            channel_scale*1,
            channel_scale*8,
            channel_scale*8,
            out_features
                        ]
        self.layers = nn.Sequential(
            nn.Linear(num_features[0], num_features[1]),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], num_features[2]),
            nn.LeakyReLU(),
            nn.Linear(num_features[2], num_features[3]),
            nn.LeakyReLU(),
            nn.Linear(num_features[3], num_features[4]),
            nn.LeakyReLU(),
        )
        self.to(device)

    def forward(self, x):
        rss = x[:,:,0]
        return self.layers(rss)
    

class EnsembleLocalization(nn.Module):
    def __init__(self, config, n_channels, n_classes, img_shape, device, elevation_map=None, num_models=20, channel_scale=32, kernel_size=3, out_kernel_size=1, scales=None, bilinear=True, depth=3, use_residual=True, input_resolution=5, single_model_training=True):
        super(EnsembleLocalization, self).__init__()
        self.vec2im = Vec2Im(config, img_shape, device, elevation_map=elevation_map)
        self.single_model_training = single_model_training
        self.output_shape = img_shape
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.return_preds = False
        models = []
        for i in range(num_models):
            mod = nn.Sequential( 
                #nn.MaxPool2d(img_scale//input_resolution),
                UNet(n_channels, n_classes, channel_scale, kernel_size, out_kernel_size, bilinear, depth, use_residual),
                #nn.ConvTranspose2d(1, 1, kernel_size=img_scale//input_resolution, stride=img_scale//input_resolution),
                nn.Upsample((self.output_shape[0], self.output_shape[1]), mode='bilinear')
                )
            models.append(mod)
        #else:
        #    for i in range(num_models):
        #        mod = UNet(n_channels, n_classes, channel_scale, kernel_size, out_kernel_size, bilinear, depth, use_residual)
        #        models.append(mod)
        self.models = nn.ModuleList(models)

    def forward(self, x):
        if isinstance(x, tuple):
            x_img, y_img, y_vecs = self.vec2im(x)
        else:
            x_img = self.vec2im(x)

        if self.training and self.single_model_training:
            random_ind = torch.randint(len(self.models), (1,))
            preds = self.models[random_ind](x_img)
        else:
            preds = []
            for model in self.models:
                preds.append(model(x_img))
            preds = torch.cat(preds, dim=1)
            # preds = preds.mean(dim=1, keepdim=True)
            # avg = avg / avg.view(*avg.size()[:2], -1).sum(dim=2, keepdims=True).unsqueeze(-1)
        if isinstance(x, tuple):
            return preds, x_img, y_img
        else:
            return preds
        #soft_avg = self.softmax(avg.view(*avg.size()[:2], -1)).view_as(avg)
    
    def predict(self, x, input_is_pred_img=False):
        if input_is_pred_img:
            preds = x
        else:
            single_model_training = self.single_model_training
            self.single_model_training = False
            preds = self.forward(x)
            self.single_model_training = single_model_training
        
        peaks, peak_locs = torch.max( preds.reshape((preds.shape[0], len(self.models), -1)), dim=-1)
        peak_locs = unravel_indices(peak_locs, preds.shape[2:])
        weighted_preds = (peak_locs * peaks.unsqueeze(-1)).sum(dim=1) / peaks.sum(dim=1, keepdim=True)
        weighted_preds = torch.hstack((torch.fliplr(weighted_preds), peaks.mean(dim=1, keepdim=True)))
        return weighted_preds

    
    def com_predict(self, x, input_is_pred_img=False):
        if input_is_pred_img:
            preds = x
        else:
            single_model_training = self.single_model_training
            self.single_model_training = False
            preds = self.forward(x)
            self.single_model_training = single_model_training
        
        peaks, peak_locs = torch.max( preds.reshape((preds.shape[0], len(self.models), -1)), dim=-1)
        mean_pred = preds.mean(axis=1)
        centers_of_mass = get_centers_of_mass(mean_pred)
        new_preds = torch.hstack((centers_of_mass, peaks.mean(dim=1, keepdim=True)))
        return new_preds 
        

def unravel_indices(indices: torch.LongTensor, shape, ) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = torch.div(indices, dim, rounding_mode='floor')
    coord = torch.stack(coord[::-1], dim=-1)
    return coord


def get_centers_of_mass(tensor):
#taken from:https://gitlab.liu.se/emibr12/wasp-secc/blob/cb02839115da475c2ad593064e3b9daf2531cac3/utils/tensor_utils.py    
    """
    Args:
        tensor (Tensor): Size (*,height,width)
    Returns:
        Tuple (Tensor): Tuple of two tensors of sizes (*)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    width = tensor.size(-1)
    height = tensor.size(-2)
    
    x_coord_im = torch.linspace(0,width,width).repeat(height,1).to(device)
    y_coord_im = torch.linspace(0,width,height).unsqueeze(0).transpose(0,1).repeat(1,width).to(device)
    
    x_mean = torch.mul(tensor,x_coord_im).sum(-1).sum(-1)/torch.add(tensor.sum(-1).sum(-1),1e-10)
    y_mean = torch.mul(tensor,y_coord_im).sum(-1).sum(-1)/torch.add(tensor.sum(-1).sum(-1),1e-10)
    
    return torch.stack((y_mean, x_mean)).T


class TiremMLP(torch.nn.Module):
    def __init__(self, num_features=[14,200], device='cuda', dropout=0.01, input_dropout=0.1) -> None:
        super().__init__()
        self.tirem_bias = nn.Parameter(torch.ones(1))
        self.layers = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(num_features[0], num_features[1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], num_features[1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], num_features[1]),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(num_features[1], 1),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.to(device)

    def forward(self, x, tirem_pred):
        #tirem_bounded = nn.functional.relu(tirem_pred + self.tirem_bias, inplace=True)
        return self.layers(x)# + tirem_bounded[:,None]