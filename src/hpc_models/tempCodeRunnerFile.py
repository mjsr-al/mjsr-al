def uncertainity(probs, weights):
    lis = []
    lis_output = []
    for i in range(hyperparameters['num_tasks']):
        attr_output = probs[i]
        w = weights[:,i]
        k = -1* np.sum(attr_output*np.log(attr_output),axis=1)
        lis_output.append(k)
        lis.append(w*k)
    
    variance = np.var(np.array(lis),axis=0)
    return np.array(lis).sum(axis=0), variance

def getIndices(output, hyperparameters ,pretrain=False):
    if pretrain == True:
        count =  hyperparameters['train_initial_batches']*hyperparameters['batch_size']
        if ((output<=0.5).sum())>=count:
            sort = np.argwhere(output<=0.5)[:,0]
            return sort
        else:
            selection = (int((hyperparameters['train_initial_batches']*hyperparameters['batch_size'])/1000)+1)*1000
            sort = np.argpartition((output)[:,0], selection)
            return sort[:selection]
    else:
        count = hyperparameters['num_uncertain_elements']
        if ((output<=0.5).sum())>=count:
            sort = np.argwhere(output<=0.5)[:,0]
            return sort
        else:
            selection = (int(hyperparameters['num_uncertain_elements']/1000)+1)*1000
            sort = np.argpartition((output)[:,0], selection)
            return sort[:selection]
        
def divide_data(train, initial = False):
    num_samples = train.values.shape[0]
    
    if initial:
        idx = random.sample(list(np.arange(num_samples)), ((int(hyperparameters['initial_percent_val']*num_samples)//hyperparameters['batch_size'])*hyperparameters['batch_size']))
    else:
        idx = random.sample(list(np.arange(num_samples)), ((int(hyperparameters['initial_percent']*num_samples)//hyperparameters['batch_size'])*hyperparameters['batch_size']))

    return pd.DataFrame(train.values[idx,:], columns=train.columns), idx

attr = pd.read_csv('../input/celeba-dataset/list_attr_celeba.csv')
eval_partition = pd.read_csv('../input/celeba-dataset/list_eval_partition.csv')

break_point_ep = {'3': 5e-4,'6': 5e-4,'10': 1e-5}
splits = [0.1,0.15,0.2,0.25,0.3,0.35,0.40]
