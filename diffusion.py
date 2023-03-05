import torch.nn.functional as F
from torch import nn
import torch
import sklearn.metrics as skm
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import scipy
from sklearn.metrics import roc_auc_score, average_precision_score

def low_density_anomalies(test_log_probs, num_anomalies):
    #anomaly_indices = np.argpartition(scores, -num_anomalies)[-num_anomalies:]
    anomaly_indices = np.argpartition(test_log_probs, num_anomalies-1)[:num_anomalies]
    preds = np.zeros(len(test_log_probs))
    preds[anomaly_indices] = 1
    return preds
  
def quantile_loss(y_true, y_pred, quantile):
    """
    Computes the quantile loss for a given quantile level.

    Args:
        y_true (torch.Tensor): Ground truth target values.
        y_pred (torch.Tensor): Predicted target values.
        quantile (float): Quantile level, a number between 0 and 1.

    Returns:
        torch.Tensor: The quantile loss.
    """
    residual = y_true - y_pred
    return torch.mean(torch.max((quantile - 1) * residual, quantile * residual))
  
def binning(t, num_bins = 30, device = 'cpu'):
    return torch.maximum(torch.minimum(torch.floor(t*num_bins/timesteps).to(device), torch.tensor(num_bins-1).to(device)), torch.tensor(0).to(device)).long()

def linear_beta_schedule(timesteps, start=0.0001, end=0.01):
    return torch.linspace(start, end, timesteps)

num_bins = 7
# Define beta schedule
T = 300
timesteps = T
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def forward_noise(x_0, t, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    noise.requires_grad_()
    sqrt_alphas_cumprod_t = torch.take(sqrt_alphas_cumprod, t.cpu()).to(device).unsqueeze(1)
    sqrt_one_minus_alphas_cumprod_t = torch.take(sqrt_one_minus_alphas_cumprod, t.cpu()).to(device).unsqueeze(1)
    # mean + variance
    #return (sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    #+ sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)).to(torch.float32), noise.to(device).to(torch.float32)
    return (x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)).to(torch.float32), noise.to(device).to(torch.float32)


def get_loss(model, x_0, t, device = 'cpu'):
    x_noisy, noise = forward_noise(x_0, t, device)

    t_pred = model(x_noisy)
    loss_fn = nn.HuberLoss(delta=5) #\nn.MSELoss()
    #loss_fn = nn.MSELoss()

    loss = loss_fn(t.float(), t_pred.squeeze())
    #loss = quantile_loss(t.float(), t_pred.squeeze(), 0.1)
    return loss

def get_loss_bins(model, x_0, t, device = 'cpu'):
    x_noisy, noise = forward_noise(x_0, t, device)

    t_pred = model(x_noisy)

    bin = binning(t, device = device,  num_bins = num_bins)

    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.MultiMarginLoss(2, 0.5) #nn.CrossEntropyLoss()

    loss = loss_fn(t_pred, bin)

    return loss

class Diffusion(nn.Module):
    def __init__(self, output_sizes, binning = True):
        super().__init__()
        self.output_sizes = output_sizes
        self.activation = nn.ReLU()
        self.binning = binning
        
        layers = []
        for i in range(1, len(self.output_sizes)):
            layers.append(nn.Linear(output_sizes[i-1], output_sizes[i]))
        
        if binning:
            self.loss_fn = get_loss_bins
            layers.append(nn.Linear(output_sizes[-1], num_bins))
            self.softmax = nn.Softmax(dim = 1)  
        else:
            self.loss_fn = get_loss
            layers.append(nn.Linear(output_sizes[-1], 1))
            self.softmax = lambda x : x
              
        self.layers = nn.ModuleList(layers)
        self.output_size = num_bins

        self.drop = torch.nn.Dropout(p=0.5, inplace=False)
          
    
    def forward(self, x):
        x = self.activation(self.layers[0](x))
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))
            x = self.drop(x)
 
        return self.softmax(self.layers[-1](x))
    
    def fit(self, X_train, y_train, X_test, Y_test, epochs = 100, batch_size = 64, lr = 1e-4, weight_decay = 5e-4, device = 'cpu'):
      optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

      train_loader = DataLoader(torch.from_numpy(X_train).float(),
                                          batch_size=batch_size, shuffle=True, drop_last=False)
      train_losses = []
      val_losses = []
      for epoch in range(epochs):
          self.train()
          loss_ = []
          for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            t = torch.randint(0, T, (x.shape[0],), device=device).long()

            loss = self.loss_fn(self, x, t, device=device)
            loss.backward()
            optimizer.step()
            loss_.append(loss.item())
            
            # optimizer.zero_grad()
            # t = torch.zeros((x.shape[0],), device=device).long()

            # loss = self.loss_fn(self, x, t, device=device)
            # loss.backward()
            # optimizer.step()
            # loss_.append(loss.item())
        
          train_losses.append(np.mean(np.array(loss_)))
          #val_losses.append(np.mean(validation(train_dl)))
          if epoch % 1 == 0:
            print(self.evaluate(X_test, Y_test, device=device))
            print(roc_auc_score(y_true=Y_test, y_score=self.predict_score(X_test, device=device)))
            print(f"Epoch {epoch} Train Loss: {train_losses[len(train_losses)-1]}")
          #if epoch > 20:
              #if train_losses[len(train_losses)-1] > train_losses[len(train_losses)-19]:
                  #break
        
      return self, self.evaluate(X_test, Y_test, device=device)

    def predict_score(self, X, device = 'cpu'):
      test_loader = DataLoader(torch.from_numpy(X).float(),
                                          batch_size=100, shuffle=False, drop_last=False)
      preds = []
      self.eval()
      for x in test_loader:
        pred_t = self(x.to(device).to(torch.float32))
        preds.append(pred_t.squeeze().cpu().detach().numpy())
      
      sig = lambda x : 1/(1+np.exp(-0.05*(x-150)))
      preds = np.concatenate(preds, axis=0)
      
      if self.binning:
        #preds = np.argmax(preds, axis=1)
        preds = np.matmul(preds, np.arange(0, preds.shape[-1]))

      return sig(preds)
  
    def evaluate(self, X, y, device = 'cpu'):
      test_loader = DataLoader(torch.from_numpy(X).float(),
                                          batch_size=100, shuffle=False, drop_last=False)
      preds = []
      self.eval()
      for x in test_loader:
        pred_t = self(x.to(device).to(torch.float32))
        preds.append(pred_t.squeeze().cpu().detach().numpy())
      
      preds = np.concatenate(preds, axis=0)
      
      if self.binning:
        preds = np.matmul(preds, np.arange(0, preds.shape[-1]))
        #preds = np.argmax(preds, axis=1)
      #score = self.predict_score(X)
      score = preds
      indices = np.arange(len(y))
      p = low_density_anomalies(-score, len(indices[y==1]))
      #p = level_set_count_anomalies(-preds_train, -preds, len(labels_indices))
      old_f1 = skm.f1_score(y, p)
      
      return old_f1
  
  
class DiffusionBagging():
    def __init__(self, seed = 0 , model_name = "Diffusion"):
        self.seed = seed
      
        self.faster_version='no'
      
        self.models = []
        self.perms = []
    
    def fit(self, X_train, y_train, X_test, Y_test, epochs = 400, device = "cpu"):
        d = X_train.shape[1]
        n = X_train.shape[0]
        if self.faster_version=='yes':
            num_permutations = min(int(np.floor(100 / (np.log(n) + d)) + 1),2)
        else:
            num_permutations=int(np.floor(100/(np.log(n)+d))+1)
        print("going to run for: ", num_permutations, ' permutations')
        for permutations in range(num_permutations):
            if num_permutations > 1:
                random_idx = torch.randperm(d)
                self.perms.append(random_idx)
                X_train = X_train[:, random_idx]
                X_test = X_test[:, random_idx]                 
       
            model = Diffusion([d, 512, 1024, 512], binning=True).to(device)
            self.models.append(model)
            
            model.fit(X_train=X_train, y_train=y_train, X_test = X_test, Y_test = Y_test, epochs=epochs, device=device)
        
        return self
      
    def predict_score(self, X, device = 'cpu'):
        total = np.zeros(X.shape[0],dtype='f')
        for i, model in enumerate(self.models):
            if len(self.perms) > 0:
                X = X[:, self.perms[i]]
            total += model.predict_score(X, device)
        return total
            