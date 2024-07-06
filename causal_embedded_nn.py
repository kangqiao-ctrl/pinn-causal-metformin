import nni
import torch
import torch.optim as optim
import sys
import logging
import argparse
import csv
import set_random_seed

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from functorch import vmap, jacfwd, hessian

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt


logger = logging.getLogger()
device = torch.device('cuda') 

def map_act_func(af_name):
    if af_name == "ReLU":
        act_func = torch.nn.ReLU()
    elif af_name == "LeakyReLU":
        act_func = torch.nn.LeakyReLU()
    elif af_name == "Sigmoid":
        act_func = torch.nn.Sigmoid()
    elif af_name == "Tanh":
        act_func = torch.nn.Tanh()
    elif af_name == "Softplus":
        act_func = torch.nn.Softplus()
    else:
        sys.exit("Invalid activation function")
    return act_func

def map_optimizer(opt_name, net_params, lr):
    if opt_name == "SGD":
        opt = optim.SGD(net_params, lr=lr)
    elif opt_name == "Adam":
        opt = optim.Adam(net_params, lr=lr)
    else:
        sys.exit("Invalid optimizer")
    return opt

def calculate_ace(model, data_mean, data_cov, num_interventions=10):
    """
    Calculate the Average Causal Effect (ACE) for a given treatment variable t.

    Args:
        model (torch.nn.Module): the trained PyTorch model
        data_mean (ndarray): pre-calculated mean vector of the input data
        data_cov (ndarray): pre-calculated covariance matrix of the input data
        t (int): the index of the treatment variable
        num_interventions (int): number of interventions to perform

    Returns:
        list: a list of ACE values for each intervention
    """
    res = []

    low, high = 0., 1.
    n = num_interventions

    
    for t in range(len(data_mean)):
        cov = data_cov.clone()
        cov[t, :] = 0
        cov[:, t] = 0

        IE = []
        GRAD = []
        HESSIAN = []


        for a in np.linspace(low, high, n):

            miu = data_mean.clone()
            miu[t] = a
            x = torch.tensor(miu, requires_grad=True).float().to(device)
            y = model(x)
            if a == 0:
                prev = y


            grad_y = torch.autograd.grad(y, x, create_graph=True)[0].to(device)
            GRAD.append(grad_y[t].cpu().item())  
            hessian_y = torch.zeros((13, 13)).to(device)
            for i in range(13):
                hessian_y[i] = torch.autograd.grad(grad_y[i], x, retain_graph=True)[0]
            HESSIAN.append(torch.trace(hessian_y).cpu().item())

            ACE_correction  = (1/2) * torch.trace(torch.matmul(hessian_y, cov))
        
            IE.append(num_interventions * (y + ACE_correction - prev).cpu().item())
            
            prev = y
        
        res.append([IE,GRAD,HESSIAN])

    
    return res

# Define the MLP model
class MyNet(torch.nn.Module):
    def __init__(self, input_size, params):
        super(MyNet, self).__init__()

        self.input_size = input_size
        self.layer_size = params['layer_size']
        hidden_layer_numer = params['hidden_layer_number']

        self.first_layer = torch.nn.Linear(input_size, self.layer_size)
        self.act_func = map_act_func(params['act_func'])
        
        if params['modify_weights']:
            self.first_layer.weight.data = self.weight_dist_gen(params['ate_weight'])
            self.first_layer.bias.data.fill_(0.0)

        if hidden_layer_numer >= 1:
            self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(self.layer_size, self.layer_size) for _ in range(hidden_layer_numer)])
        self.final_layer = torch.nn.Linear(self.layer_size, 1)


        
    def forward(self, x):
        x = self.act_func(self.first_layer(x))
        if hasattr(self, 'hidden_layers'):
            for hidden_layer in self.hidden_layers:
                x = self.act_func(hidden_layer(x))
        x = self.final_layer(x)
        return x
    
    def weight_dist_gen(self, ate_vals):
        random_weights = np.zeros((self.input_size, self.layer_size))
        for i in range(self.input_size):
            random_weights[i, :] = np.random.normal(ate_vals[i], 0.1, params['layer_size'])
        return torch.from_numpy(random_weights.astype('float32')).T


   
def main(params):
    df = pd.read_parquet("df_nn.parquet")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_str = 'layer_' + str(params['hidden_layer_number']) + 'node_' + str(params['layer_size']) + '_' +   params['act_func'] + '_LR_'+ str(params['learning_rate'])
    if params['modify_weights']:
        param_str += "weightmod_1"
    else:
        param_str += "weightmod_0"

    param_str += 'lbd_'+str(params['lambda_1'])

    experiment_id = nni.get_experiment_id()


    loss_file = experiment_id + f'_loss_{param_str}.csv'

    causal_file = experiment_id+ f'_causal_effect_{param_str}.csv'
    grad_file = experiment_id+ f'_grad_{param_str}.csv'
    hessian_file = experiment_id+ f'_hessian_{param_str}.csv'

    shap_file = experiment_id+ f'_shap_{param_str}.csv'
    shap_png = experiment_id+ f'_shap_{param_str}.png'    
 

    train_loss = []
    test_loss = []

    # Define the model, loss function, and optimizer
    input_size = X_train.shape[1]
    model = MyNet(input_size,params)
    optimizer = map_optimizer(params['optimizer'], model.parameters(), params['learning_rate'])

    model.to(device)

    # Train the model
    epochs = 1000
    last_results = []
    inputs = torch.tensor(torch.from_numpy(X_train.astype('float32'))).to(device)
    
    cov_mat = torch.cov(inputs.T)
    var_mean = torch.mean(inputs,0)

    for epoch in range(1,epochs+1):
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                data = torch.from_numpy(X_test.astype('float32')).to(device)
                label = torch.from_numpy(y_test.astype('float32')).to(device)
                outputs = model(data)
                rmse = mean_squared_error(label.cpu().numpy(), outputs.cpu().numpy(), squared=False)
                r2 = r2_score(label.cpu().numpy(), outputs.cpu().numpy())
                print("epoch ", str(epoch), " | RMSE: ", str(rmse), " | R2: ", str(r2))
                test_loss.append(rmse)
            nni.report_intermediate_result(rmse)


        model.train()
        optimizer.zero_grad()
        targets = torch.from_numpy(y_train.astype('float32')).to(device)
        outputs = model(inputs)
        if params['lambda_1'] > 0:
            function_derivative = torch.tensor([params['ate_weight']]).to(device)
            compute_batch_jacobian = vmap(jacfwd(model, argnums=0), in_dims=0)
            fwd_jacobian = compute_batch_jacobian(inputs).to(device)
            error_term = fwd_jacobian - function_derivative
            L1 = torch.norm(error_term, p=1,dim=2)
            loss = torch.nn.MSELoss()(outputs, targets) + params['lambda_1'] * torch.mean(torch.max(L1 - 0.01, torch.zeros_like(L1)))
        else:
            loss = torch.nn.MSELoss()(outputs, targets)
        if epoch % 5 == 0:
            train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if epoch >= 700:
            last_results.append(rmse)
    
    causal_list = calculate_ace(model, var_mean, cov_mat, num_interventions=10)    

    pd.DataFrame([causal_list[i][0] for i in range(13)]).T.to_csv(causal_file)
    pd.DataFrame([causal_list[i][1] for i in range(13)]).T.to_csv(grad_file)    
    pd.DataFrame([causal_list[i][2] for i in range(13)]).T.to_csv(hessian_file)

    model.cpu()

    sample_train_idx = np.random.randint(X_train.shape[0], size=1000)
    sample_test_idx = np.random.randint(X_test.shape[0], size=1000)
    sampled_inputs = X_train[sample_train_idx,:]
    background = torch.from_numpy(sampled_inputs.astype('float32'))
    explainer = shap.DeepExplainer(model, background)
    shap_data = torch.from_numpy(X_test[sample_test_idx,:].astype('float32'))
    shap_values = explainer.shap_values(shap_data)


    shap_df = pd.DataFrame(shap_values, columns=[f"feature_{i}" for i in range(shap_values.shape[1])])
    shap_df.to_csv(shap_file, index=False)

    shap.summary_plot(shap_values, shap_data.detach().numpy(),show=False)
    plt.savefig(shap_png, dpi=300, bbox_inches='tight')


    
    with open(loss_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Epoch', 'Train Loss','Test Loss'])
        for i, (train_loss, test_loss) in enumerate(zip(train_loss,test_loss)):
            writer.writerow([i*5, train_loss, test_loss])
            
    state_dict = model.state_dict()
    for key in state_dict.keys():
        if 'weight' in key:
            weights_df = pd.DataFrame(state_dict[key].numpy()).T
            weights_df.to_csv(experiment_id+ '_' + key + f'_{param_str}.csv')


    nni.report_final_result(min(last_results)) # use min of last results since results fluctuates a lot sometimes


def generate_default_params():
    '''
    Generate default parameters for mnist network.
    '''
    params = {
        'hidden_layer_number': 2,
        'layer_size': 32,
        'act_func': 'Tanh',
        'learning_rate': 0.05,
        'optimizer': 'Adam',
        # ATEs from SCM regressions
        'ate_weight':[-0.001219236, -0.035074218,0, 0, 0.033900008, 0.000465403,-0.042283258,-1.064407829, -0.144739628,-0.025415903,-0.019302039,0,0.162546474], 
        'lambda_1':1,
        'modify_weights':1}
    return params

if __name__ == '__main__':
    print(torch.cuda.is_available())
    set_random_seed.set_random_seed()
    parser = argparse.ArgumentParser(description='NNI tunner')
    args = parser.parse_args()
    try:
        # get parameters from tuner
        updated_params = nni.get_next_parameter()
        logger.debug(updated_params)
        # run a NNI session
        params = generate_default_params()
        params.update(updated_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
