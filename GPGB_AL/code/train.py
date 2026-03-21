from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel, RationalQuadratic, Matern, \
    ExpSineSquared
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import copy, random, sklearn, math, sys, torch, json, argparse, configparser, pickle, os
from read_data import DataProcessor, AADataset
from sklearn import metrics
from sklearn.metrics import r2_score
from datetime import datetime
import xgboost as xgb
from gplearn import genetic
import xgboost as xgb
import polars as pl


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# NN regressor
class NNRegressor(torch.nn.Module):
    def __init__(self, dims, activation=torch.nn.ReLU(), last_act=None):
        super(NNRegressor, self).__init__()
        self.network = torch.nn.ModuleList([])
        for i in range(len(dims)-1):
            self.network.append(torch.nn.Linear(dims[i], dims[i+1]))
            if i != len(dims)-2:
                self.network.append(activation)
        if not last_act is None:
            self.network.append(last_act)

    def forward(self, x):
        for f in self.network:
            x = f(x)
        return x


def train_ml_model(train_dataset, test_dataset, new_dataset=None, model=None):
    model.fit(train_dataset['data_x'], train_dataset['data_y'])
    train_predictions=model.predict(train_dataset['data_x'])
    k_train =list(zip(train_dataset['data_y'],train_predictions))
    k_train_p =list(zip( np.power(10, train_dataset['data_y']), np.power(10, model.predict(train_dataset['data_x']))))

    mse_train = np.mean(np.square(np.abs([item[0] - item[1] for item in k_train])))
    mse_train_p = np.mean(np.square(np.abs([item[0] - item[1] for item in k_train_p])))
    names_train=[]
    for i in train_dataset['names']:
        names_train.append(int(i[0]))
    data_y_train=[]
    for i in train_dataset['data_y']:
        data_y_train.append(i[0])
    train_pred=[]
    for i in train_predictions:
        # train_pred.append(i[0])
        train_pred.append(i)

    train_pred_p=np.power(10,train_pred)


    test_predictions=model.predict(test_dataset['data_x'])
    test_predictions_p = np.power(10, test_predictions)
    k_test =list(zip(test_dataset['data_y'],test_predictions))
    k_test_p =list(zip( np.power(10, test_dataset['data_y']), np.power(10, model.predict(test_dataset['data_x']))))
    mse_test = np.mean(np.square(np.abs([item[0] - item[1] for item in k_test])))
    mse_test_p = np.mean(np.square(np.abs([item[0] - item[1] for item in k_test_p])))

    names_test=[]
    for i in test_dataset['names']:
        names_test.append(int(i[0]))
    data_y=[]
    for i in test_dataset['data_y']:
        data_y.append(i[0])
    test_pred=[]
    for i in test_predictions:
        # test_pred.append(i[0])
        test_pred.append(i)

    test_pred_p=[]
    for i in test_predictions_p:
        test_pred_p.append(i)
        # test_pred_p.append(i[0])
    results = {
        'mse_train': mse_train,
        'mse_train_p': mse_train_p,

        'mse_test': mse_test,
        'mse_test_p': mse_test_p,
        'train_preds': list(zip(names_train, data_y_train, train_pred)),
        'test_preds': list(zip(names_test, data_y, test_pred))
    }

    train_final_list =[]
    train_y=[]
    train_predict_y=[]
    for i in range(len(train_dataset['data_x'])):
        meta=[]
        meta.append(train_dataset['names'][i][0])
        meta.append(train_pred[i])
        meta.append(train_dataset['data_y'][i][0])
        train_y.append(train_dataset['data_y'][i][0])
        train_predict_y.append(train_pred[i])
        train_final_list.append(meta)

    test_final_list =[]
    test_y=[]
    test_predict_y=[]
    for i in range(len(test_dataset['data_x'])):
        meta=[]
        meta.append(test_dataset['names'][i][0])
        meta.append(test_pred[i])
        meta.append(test_dataset['data_y'][i][0])
        test_y.append(test_dataset['data_y'][i][0])
        test_predict_y.append(test_pred[i])
        test_final_list.append(meta)
    print("Train MAE:",metrics.mean_absolute_error(train_y, train_predict_y))
    print("Train MSE:",metrics.mean_squared_error(train_y, train_predict_y))
    print("Train R-square:",r2_score(train_y, train_predict_y))
    print("Test MAE:",metrics.mean_absolute_error(test_y, test_predict_y))
    print("Test MSE:",metrics.mean_squared_error(test_y, test_predict_y))
    print("Test R-square:",r2_score(test_y, test_predict_y))

    if True:
        import pandas
        train_df = pandas.DataFrame(train_final_list,
                                    columns=['Order', 'Predict Value', 'Exp Value'])
        test_df = pandas.DataFrame(test_final_list,
                                   columns=['Order', 'Predict Value', 'Exp Value'])
        with pandas.ExcelWriter('0115/base_model/gpr_Best_Model.xlsx', engine='openpyxl') as writer:
            train_df.to_excel(writer, sheet_name='train_result', index=False)
            test_df.to_excel(writer, sheet_name='test_result', index=False)
    # print(results)
    return results, model


def grid_search_linear(train_dataset_all, test_dataset):
    records = {}
    for l1_weight in [0, 0.01, 0.2, 0.5, 1.0, 2]:
        for l2_weight in [0, 0.2, 0.5, 1.0, 2]:
            model_name = None
            setting = 'l1:{}-l2:{}'.format(l1_weight, l2_weight)
            if l1_weight == 0 and l2_weight == 0:
                linear_model = LinearRegression()
                model_name = 'ols'
            elif l1_weight == 0 and l2_weight > 0:
                linear_model = Ridge(alpha=l2_weight)
                model_name = 'ridge-l1:{}-l2:{}'.format(l1_weight, l2_weight)
            elif l1_weight > 0 and l2_weight == 0:
                linear_model = Lasso(alpha=l1_weight)
                model_name = 'lasso-l1:{}-l2:{}'.format(l1_weight, l2_weight)
            else:
                alpha = l1_weight+2*l2_weight
                l1_ratio = l1_weight/alpha
                linear_model = ElasticNet(
                    alpha=alpha, l1_ratio=l1_ratio)
                model_name = 'es_net-l1:{}-l2:{}-alpha:{}-l1ratio:{}'.format(
                    l1_weight, l2_weight, alpha, l1_ratio)

            records[model_name] = train_ml_model(
                train_dataset_all, test_dataset, model=linear_model)

    output_performance(records)


def grid_search_gpr(train_dataset_all, test_dataset):
    records = {}
    kernels = []
    kernel_names = []
    for constant_value in [0, 0.5, 1, 10]:
        for noise_level in [0, 0.05, 0.5, 1.0]:
            # Matern Kernal (=RBF when nu=0.5)
            for nu in [0.5, 1, 5]:
                for length_scale in [0.5, 1,  5]:
                    kernels.append(WhiteKernel(
                        noise_level)+ConstantKernel(constant_value)+Matern(length_scale=length_scale, nu=nu))
                    kernel_names.append('WhiteKernel({})+ConstantKernel({})+MaternKernel({}, {})'.format(
                        noise_level, constant_value, length_scale, nu))
            # DocProduct Kernel
            for sigma in [0.5, 1.0, 5]:
                kernels.append(WhiteKernel(
                    noise_level)+ConstantKernel(constant_value)+DotProduct(sigma))
                kernel_names.append('WhiteKernel({})+ConstantKernel({})+DotProduct({})'.format(
                    noise_level, constant_value, sigma))
            # RationalQuadratic Kernel
            for length_scale in [0.5, 1.0, 2, 5]:
                kernels.append(WhiteKernel(
                    noise_level)+ConstantKernel(constant_value)+RationalQuadratic(length_scale=length_scale))
                kernel_names.append('WhiteKernel({})+ConstantKernel({})+RationalQuadratic({})'.format(
                    noise_level, constant_value, length_scale))
            # ExpSineSquared Kernel
            for length_scale in [0.5, 1., 2., 5]:
                for periodicity in [0.5, 1, 2, 5]:
                    kernels.append(WhiteKernel(noise_level)+ConstantKernel(constant_value) +
                                   ExpSineSquared(length_scale=length_scale, periodicity=periodicity))
                    kernel_names.append('WhiteKernel({})+ConstantKernel({})+ExpSineSquared({}, {})'.format(
                        noise_level, constant_value, length_scale, periodicity))

    for i in range(len(kernels)):
        model = GaussianProcessRegressor(kernel=kernels[i], random_state=0)
        try:
            records[kernel_names[i]] = train_ml_model(
                train_dataset_all, test_dataset, model)
        except Exception:
            continue
    output_performance(records)


def output_performance(records):
    best_model_val = sorted(records.items(), key=lambda x: x[1]['mse_val_p'])
    best_model_test = sorted(records.items(), key=lambda x: x[1]['mse_test_p'])
    best_model_train = sorted(
        records.items(), key=lambda x: x[1]['mse_train_p'])
    for item in best_model_val:
        print('model_name: {}, mse_train: {}, mse_train_p: {}, mse_val: {}, mse_val_p: {}, mse_test: {}, mse_test_p:{}, mse_val_std: {}, mse_val_std_p: {}'.format(
            item[0],
            item[1]['mse_train'],
            item[1]['mse_train_p'],
            item[1]['mse_val'],
            item[1]['mse_val_p'],
            item[1]['mse_test'],
            item[1]['mse_test_p'],
            item[1]['mse_val_std'],
            item[1]['mse_val_std_p']
        )),


def grid_search_rf(train_dataset_all, test_dataset):
    records = {}
    for n_estimators in [10, 50, 100]:
        for max_depth in [5, 10, 15, None]:
            rf_model = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=3)
            model_name = 'n:{}-depth:{}'.format(n_estimators, max_depth)
            records[model_name] = train_ml_model(
                train_dataset_all, test_dataset, model=rf_model)
    output_performance(records)


def grid_search_svr(train_dataset_all, test_dataset):
    records = {}
    for kernel in ['poly', 'rbf', 'sigmoid']:
        for c in [0.1, 1, 5]:
            for gamma in [0.1, 1, 0.01]:
                if kernel == 'rbf':
                    model = SVR(kernel=kernel, C=c, gamma=gamma)
                else:
                    model = SVR(kernel=kernel)
                model_name = '{}-{}-{}'.format(kernel, c, gamma)
                records[model_name] = train_ml_model(
                    train_dataset_all, test_dataset, model=model)
    output_performance(records)

def train_ann(dims, lr, batch_size, activation, epoch, train_dataset_all, seed):
    print("torch.cuda.is_available():",torch.cuda.is_available())
    results = {
        'train_mse': 0,
        'train_mse_p': 0,
        'train_preds': [],
        'val_mse': 0,
        'val_mse_p': 0,
        'val_mse_std': 0,
        'val_mse_std_p': 0,
    }
    model = NNRegressor(dims=dims, activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_dataset, test_dataset = DataProcessor.split(train_dataset_all, args.train_data_num)
    print("train_dataset:", train_dataset['size'])
    print("test_dataset:", test_dataset['size'])
    train_loader = DataLoader(AADataset(train_dataset), batch_size=1)
    val_loader = DataLoader(AADataset(test_dataset), batch_size=1)

    print('epoch:',epoch)
    for epoch_ in range(epoch):
        shuffle_train_dataset= DataProcessor.shuffle(copy.deepcopy(train_dataset))
        shuffle_train_loader= DataLoader(AADataset(shuffle_train_dataset), batch_size=batch_size)
        for batch_id, (data_x, data_y, name) in enumerate(shuffle_train_loader):

            optimizer.zero_grad()
            y_ = model(data_x.float())
            regular_loss = 0.0
            for para in model.parameters():
                regular_loss += torch.sum(para ** 2.0)
            loss = criterion(y_, data_y.float())

            loss.backward()
            optimizer.step()
            # if last_para[0] ==optimizer.param_groups[0]['params'][0]:
            #     print("same parameter!!!!!!!!!!!!")
        # for z in  optimizer.param_groups[0]['params']:
        #     print(z.grad )
        #     break

        train_predictions = []
        train_predictions_p = []
        val_mse = []
        val_mse_p = []
        TRAIN_SHOW =[]
        train_y=[]
        train_predict_y=[]
        for batch_id, (data_x, data_y, name) in enumerate(train_loader):
            y_pred = model(data_x.float()).detach().numpy()[0][0]
            y_pred_p = np.power(10, y_pred)
            y_true = data_y.detach().numpy()[0][0],
            y_true_p = np.power(10, y_true)
            train_predictions.append([
                y_pred,
                y_true,
            ])
            train_y.append(y_true)
            train_predict_y.append(y_pred)
            train_predictions_p.append([
                y_pred_p,
                y_true_p
            ])
            TRAIN_SHOW.append([
                int(name[0][0]),
                y_pred,
                y_true[0],
            ])
        xx_val_mse=np.mean(np.square(np.abs([item[0]-item[1]
                       for item in train_predictions])))
        val_mse.append(xx_val_mse)
        xx_val_mse_p = np.mean(np.square(np.abs([item[0]-item[1] for item in train_predictions_p])))
        val_mse_p.append(xx_val_mse_p)

        val_predictions = []
        val_predictions_p = []
        VAL_SHOW =[]
        test_y = []
        test_predict_y = []
        for batch_id, (data_x, data_y, name) in enumerate(val_loader):
            y_pred = model(data_x.float()).detach().numpy()[0][0]
            y_pred_p = np.power(10, y_pred)
            y_true = data_y.detach().numpy()[0][0],
            y_true_p = np.power(10, y_true)
            val_predictions.append([
                y_pred,
                y_true,
            ])
            test_y.append(y_true)
            test_predict_y.append(y_pred)
            VAL_SHOW.append([
                int(name[0][0]),
                y_pred,
                y_true[0],
            ])
            val_predictions_p.append([
                y_pred_p,
                y_true_p
            ])
        yy_val_mse=np.mean(np.square(np.abs([item[0]-item[1]
                       for item in val_predictions])))
        val_mse.append(yy_val_mse)
        yy_val_mse_p = np.mean(np.square(np.abs([item[0]-item[1] for item in val_predictions_p])))
        val_mse_p.append(yy_val_mse_p)
        if(epoch_ %500==0):
            # print("TRAIN_SHOW:", TRAIN_SHOW)
            # print("VAL_SHOW:",VAL_SHOW)
            print("Train MAE:", metrics.mean_absolute_error(train_y, train_predict_y))
            print("Train MSE:", metrics.mean_squared_error(train_y, train_predict_y))
            print("Train R-square:", r2_score(train_y, train_predict_y))
            print("Test MAE:", metrics.mean_absolute_error(test_y, test_predict_y))
            print("Test MSE:", metrics.mean_squared_error(test_y, test_predict_y))
            print("Test R-square:", r2_score(test_y, test_predict_y))
            print(" epoch,lr,train_mse,test_mse,",
                  epoch_,optimizer.state_dict()['param_groups'][0]['lr'],xx_val_mse,yy_val_mse)

        results['val_mse'] = np.mean(val_mse)
        results['val_mse_p'] = np.mean(val_mse_p)
        results['val_mse_std'] = np.std(val_mse)
        results['val_mse_std_p'] = np.std(val_mse_p)
    return results, model


def grid_search_gbdt(train_dataset_all, test_dataset):
    records = {}
    for n_estimators in [10, 50, 100]:
        for max_depth in [3, 5, 10, 50]:
            for learning_rate in [1e-2, 1e-3, 1e-1]:
                gbdt_model = GradientBoostingRegressor(
                    loss='squared_error', n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
                model_name = 'n:{}-depth:{}-lr:{}'.format(
                    n_estimators, max_depth, learning_rate)
                records[model_name] = train_ml_model(
                    train_dataset_all, test_dataset, gbdt_model)

    output_performance(records)

def xgboost(train_x, test_x,best_parameters,random_seed):
    train_dataset=train_x.__dict__
    test_dataset=test_x.__dict__

    modify_train_x=train_dataset['data_x']
    modify_test_x=test_dataset['data_x']
    setup_seed(random_seed)

    model = xgb.XGBRegressor(max_depth=best_parameters['max_depth'], n_estimators=best_parameters['n_estimators'], learning_rate=best_parameters['learning_rate']*0.0001)
    model.fit(np.array(modify_train_x), np.array(train_dataset['data_y']))
    # train
    x_train_predicted = model.predict(np.array(modify_train_x))
    x_train_data =[]
    train_compare = []
    train_show = []
    # print(modify_train_x[0])
    for i in range(len(modify_train_x)):
        train_compare.append([x_train_predicted[i], train_dataset['data_y'][i]])
        train_show.append([
            int(train_dataset['names'][i][0]), x_train_predicted[i], train_dataset['data_y'][i][0]])
        x_train_data.append(list(modify_train_x[i]))
    train_mse = np.mean(np.square(np.abs([item[0] - item[1] for item in train_compare])))

    # test
    x_test_predicted = model.predict(np.array(modify_test_x))

    x_test_data =[]
    test_compare = []
    test_show = []
    for i in range(len(modify_test_x)):
        test_compare.append([x_test_predicted[i], test_dataset['data_y'][i]])
        test_show.append([
            int(test_dataset['names'][i][0]), x_test_predicted[i], test_dataset['data_y'][i][0]])
        x_test_data.append(list(modify_test_x[i]))
    test_mse = np.mean(np.square(np.abs([item[0] - item[1]
                                         for item in test_compare])))
    # print("x_train_data:",x_train_data)
    train_final_list=[]
    for i in range(len(x_train_data)):
        meta_train =train_show[i]+x_train_data[i]
        train_final_list.append(meta_train)

    test_final_list=[]
    for i in range(len(x_test_data)):
        meta_test =test_show[i]+x_test_data[i]
        test_final_list.append(meta_test)

    y_train=[]
    y_train_predict=[]
    y_test=[]
    y_test_predict=[]
    for i in range(len(train_final_list)):
        y_train.append(train_final_list[i][2])
        y_train_predict.append(train_final_list[i][1])

    for i in range(len(test_final_list)):
        y_test.append(test_final_list[i][2])
        y_test_predict.append(test_final_list[i][1])

    train_r2=r2_score(y_train, y_train_predict)
    test_r2=r2_score(y_test, y_test_predict)
    if True:
        print("R2 train_score,test_score: ", train_r2, test_r2)
        print("MAE  train_score,test_score:", metrics.mean_absolute_error(y_train, y_train_predict),
              metrics.mean_absolute_error(y_test, y_test_predict))
        print("MSE train_score,test_score:", metrics.mean_squared_error(y_train, y_train_predict),
              metrics.mean_squared_error(y_test, y_test_predict))
        # print("xgboost特征重要性：", model.feature_importances_)
        print("......................")
        if False:
            import pandas
            train_df = pandas.DataFrame(train_final_list,columns=['Order','Predict Value','Exp Value','X0','X1', 'X2', 'X3','X4','X5','X6','X7','X8','X9','X10','X11', 'X12', 'X13','X14','X15','X16','X17','X18','X19'])
            test_df = pandas.DataFrame(test_final_list,columns=['Order','Predict Value','Exp Value','X0','X1', 'X2', 'X3','X4','X5','X6','X7','X8','X9','X10','X11', 'X12', 'X13','X14','X15','X16','X17','X18','X19'])
            with pandas.ExcelWriter('0115/base_model/raw_xgboost_Best_Model.xlsx', engine='openpyxl') as writer:
                train_df.to_excel(writer,sheet_name='train_result', index=False)
                test_df.to_excel(writer,sheet_name='test_result', index=False)
            print('train_final_list :', len(train_final_list))
            print('test_final_list :', len(test_final_list))


    return r2_score(y_train, y_train_predict),r2_score(y_test, y_test_predict)

def gpgb(train_x, test_x,exp_x_name,exp_x,best_parameters,n_component,gp_adjust_random):

    train_dataset=train_x.__dict__
    test_dataset=test_x.__dict__
    new_train_data=np.array(list(train_dataset['data_x'])+list(test_dataset['data_x']))
    new_label_data=np.array(list(train_dataset['data_y'])+list(test_dataset['data_y']))
    gp_seed =gp_adjust_random #best_parameters['gp_random_state']
    setup_seed(gp_seed)
    #build trainsform model
    modify_model = genetic.SymbolicTransformer(generations=4, parsimony_coefficient=0.005,n_components=n_component,function_set=('add', 'sub', 'mul', 'div','sqrt'))
    modify_model.fit(new_train_data,new_label_data )
    #transform x
    modify_train_x=modify_model.transform(train_dataset['data_x'])
    # print("modify_train_x.shape:",modify_train_x.shape)
    modify_test_x=modify_model.transform(test_dataset['data_x'])
    if args.search_optimal_validation_data or args.enable_active_learning:
        exp_x = np.array(exp_x)
        modify_exp_predict = modify_model.transform(exp_x)
    model = xgb.XGBRegressor(max_depth=best_parameters['max_depth'], n_estimators=best_parameters['n_estimators'], learning_rate=best_parameters['learning_rate']*0.0001)
    model.fit(np.array(modify_train_x), np.array(train_dataset['data_y']))
    #train and test
    x_train_predicted = model.predict(np.array(modify_train_x))
    x_test_predicted = model.predict(np.array(modify_test_x))
    # compute metric
    train_show = []
    test_show = []
    y_train = []
    y_train_predict = []
    y_test = []
    y_test_predict = []
    for i in range(len(modify_train_x)):
        train_show.append([
            int(train_dataset['names'][i][0]), x_train_predicted[i], train_dataset['data_y'][i][0]])
    for i in range(len(modify_test_x)):
        test_show.append([
            int(test_dataset['names'][i][0]), x_test_predicted[i], test_dataset['data_y'][i][0]])
    for i in range(len(train_show)):
        y_train.append(train_show[i][2])
        y_train_predict.append(train_show[i][1])
    for i in range(len(test_show)):
        y_test.append(test_show[i][2])
        y_test_predict.append(test_show[i][1])

    train_final_list = []
    for i in range(len(modify_train_x)):
        meta_train = list(train_show[i]) + list(modify_train_x[i])
        train_final_list.append(meta_train)
    test_final_list = []
    for i in range(len(modify_test_x)):
        meta_test = list(test_show[i]) + list(modify_test_x[i])
        test_final_list.append(meta_test)
    train_r2 = r2_score(y_train, y_train_predict)
    test_r2 = r2_score(y_test, y_test_predict)

    if args.enable_active_learning:
        # all decision tree
        pred_contribs = model.get_booster().predict(xgb.DMatrix(np.array(modify_exp_predict)), pred_contribs=True)
        # 计算每个样本的最终预测（每行是一个样本，每列是一个树的贡献，除最后一列外，它是偏置项）
        final_predictions = np.sum(pred_contribs, axis=1)
        # 计算所有树的预测结果（不包括偏置项的贡献）
        predictions_per_tree = pred_contribs[:, :-1]
        # 计算方差
        variances = np.var(predictions_per_tree, axis=1)
        # std_devs = np.sqrt(variances)
        # print("每个样本的预测方差:", len(variances),variances[:5])
        # print("每个样本的预测标准差:", len(std_devs),std_devs[:5])
        if  (0.84<test_r2 ) :
        #     print("特征重要性：",model.feature_importances_)
            print("Train R-square:", train_r2)
            print("Test R-square:", test_r2)
            print("Train MAE:", metrics.mean_absolute_error(y_train, y_train_predict))
            print("Train MSE:", metrics.mean_squared_error(y_train, y_train_predict))
            print("Test MAE:",metrics.mean_absolute_error(y_test, y_test_predict))
            print("Test MSE:",metrics.mean_squared_error(y_test, y_test_predict))
            print('....................')

    #exp_predict
    if  args.search_optimal_validation_data:
        exp_predict = model.predict(modify_exp_predict)
        # higner_rank_name = []
        if   test_r2 > 0.85:
            print("R2 train_score,test_score: ", train_r2, test_r2)
            print("MAE  train_score,test_score:", metrics.mean_absolute_error(y_train, y_train_predict), metrics.mean_absolute_error(y_test, y_test_predict))
            print("MSE train_score,test_score:", metrics.mean_squared_error(y_train, y_train_predict),metrics.mean_squared_error(y_test, y_test_predict))
            # print("gplearning model:", modify_model)
            # print("gpgb特征重要性：", model.feature_importances_
            import pandas
            # predict data
            if True:
                exp_front = []
                for i in range(len(exp_predict)):
                    exp_front.append([exp_x_name[i][0],exp_x_name[i][1], float(exp_predict[i])]+list(modify_exp_predict[i].reshape(11)))
                exp_front.sort(key=lambda sl: -sl[2], )
                print(len(exp_front),exp_front[:5])
                predict_value = pandas.DataFrame(exp_front,columns=['Name','Order', 'Predict Value','Z0','Z1', 'Z2', 'Z3','Z4','Z5','Z6','Z7','Z8','Z9','Z10'])
                predict_value.to_excel('Predict_model_doubleZnCe_add_Ce_0308_rank_Z0_Z10.xlsx',sheet_name='predict_result', index=False)

                #only_for DAC
                # exp_front = []
                # for i in range(len(exp_predict)):
                #     exp_front.append([exp_x_name[i][0],exp_x_name[i][1], float(exp_predict[i])]+list(modify_exp_predict[i].reshape(11)))
                # exp_front.sort(key=lambda sl: -sl[2], )
                # print(len(exp_front),exp_front[:5])
                # predict_value = pandas.DataFrame(exp_front,columns=['Name','Order', 'Predict Value','Z0','Z1', 'Z2', 'Z3','Z4','Z5','Z6','Z7','Z8','Z9','Z10'])
                # predict_value.to_excel('DAC_data_predict_rank.xlsx',sheet_name='predict_result', index=False)
            #get model train/test distribution
            if False:
                train_df = pandas.DataFrame(train_final_list,columns=['Order', 'Predict Value', 'Exp Value', 'Z0','Z1', 'Z2', 'Z3','Z4','Z5','Z6','Z7','Z8','Z9','Z10'])
                test_df = pandas.DataFrame(test_final_list,
                                           columns=['Order', 'Predict Value', 'Exp Value','Z0','Z1', 'Z2', 'Z3','Z4','Z5','Z6','Z7','Z8','Z9','Z10'])
                with pandas.ExcelWriter('GPGB_Raw_Model.xlsx', engine='openpyxl') as writer:
                    train_df.to_excel(writer, sheet_name='train_result', index=False)
                    test_df.to_excel(writer, sheet_name='test_result', index=False)
            print('....................')
            #get final model  x1-x20, z1-z10 distribution
            if False:
                import polars as pl
                data_array = pl.read_excel(
                    '/home/a/PKU/PKU240918/code_data/task_data/base_model_data_Ni1_Mn1_Ir1_Ni2_Ti1_Fe1_Cu1_Ni3.xlsx', sheet_name='DATA')
                data_lie_all =[]
                data_lib_value =[]
                for row_index in range(1, data_array.shape[0]):
                    # print(row_index)
                    data = data_array[row_index].to_numpy()[0]
                    # if row_index <3:
                    #     print(data[0:6])
                    #     print(data[6:26])
                    z_data=modify_model.transform(np.array(data[6:26]).reshape(-1,20))
                    z_predict = model.predict(z_data)
                    # print("z_data: ",z_data)
                    # print(z_predict)
                    data_lie_all.append([z_predict[0]]+list(data[2:26])+list(z_data[0]))
                # print("data_lie_all:",data_lie_all)
                all_value = pandas.DataFrame(data_lie_all,columns=['Predict Value','model序号','中心金属','数据库编号' ,'EXP Value','X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19',
                                                                   'Z0','Z1', 'Z2', 'Z3','Z4','Z5','Z6','Z7','Z8','Z9','Z10'])
                all_value.to_excel('Feature_Importance.xlsx',sheet_name='data', index=False)


    if args.search_optimal_validation_data or args.enable_active_learning:
        return train_r2, test_r2,[]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='gpgb')  # [ols, lasso, ridge, svr, rf, gpr, ann_1, ann_2, ann_3,xgboost,gpgb]
    parser.add_argument('--output_dir', type=str, default='data/results')
    parser.add_argument('--model_params', type=str, default='data/model_params.json')
    parser.add_argument('--train_data_num', type=int, default=228)
    parser.add_argument('--data_lib_path', type=str, default='/home/a/PKU/PKU240918/code/data/20260308/model_doubleZnCe_add_Ce.xlsx')
    parser.add_argument('--search_optimal_validation_data', type=bool, default=True)
    parser.add_argument('--enable_active_learning', type=bool, default=False)


    args = parser.parse_args()

    with open(args.model_params, 'r') as fp:
        best_parameters = json.loads(''.join(fp.readlines()))

    reverse_y = 'log'

    processor = DataProcessor()
    train_dataset_all = processor.get_dataset()
    test_dataset = processor.get_dataset()
    full_dataset = processor.get_dataset()
    new_dataset = processor.get_dataset()

    gpr_kernels = {
        'DotProduct': DotProduct,
        'RationalQuadratic': RationalQuadratic,
        'ExpSineSquared': ExpSineSquared,
        'Matern': Matern,
    }
    nn_activations = {
        'ReLU': torch.nn.ReLU(),
        'Tanh': torch.nn.Tanh(),
        'LeakyReLU': torch.nn.LeakyReLU(),
        'Sigmoid':torch.nn.Sigmoid(),
        'Softmax': torch.nn.Softmax(),
    }

    # load models
    model_map = {
        'ols': LinearRegression(),
        'lasso': Lasso(alpha=best_parameters['lasso']['alpha']),
        'ridge': Ridge(alpha=best_parameters['ridge']['alpha']),
        'svr': SVR(**best_parameters['svr']),
        'rf': RandomForestRegressor(
            n_estimators=best_parameters['rf']['n_estimators'],
            max_depth=best_parameters['rf']['max_depth'],
            random_state=best_parameters['rf']['random_state']
        ),
        'gpr': GaussianProcessRegressor(WhiteKernel(best_parameters['gpr']['WhiteKernel'])
                                        +ConstantKernel(best_parameters['gpr']['ConstantKernel'])+
                                        gpr_kernels[best_parameters['gpr']['Kernel']](**best_parameters['gpr']['kernel_params'])),
    }
    if  args.model =='xgboost' :
        setup_seed(best_parameters[args.model]['random_state'])
        processor = DataProcessor(seed=best_parameters[args.model]['random_state'])
        train_dataset_all = processor.get_dataset()
        train_dataset_all = DataProcessor.shuffle(train_dataset_all)
        train_dataset, test_dataset = DataProcessor.split(train_dataset_all, args.train_data_num)
        train_loader = DataLoader(AADataset(train_dataset), batch_size=1)
        val_loader = DataLoader(AADataset(test_dataset), batch_size=1)
        train_r2,test_r2= xgboost(train_loader.dataset,val_loader.dataset,best_parameters[args.model],best_parameters[args.model]['random_state'])
    elif args.model == 'gpgb':
        data_lib_value =[]
        data_lib_name=[]
        if args.search_optimal_validation_data or args.enable_active_learning:
            data_array = pl.read_excel(
                args.data_lib_path, sheet_name='DATA')
            # 2. 过滤掉包含任何空值的行（核心操作）
            # drop_nulls() 默认删除包含任意空值的行，等价于how="any"
            data_array = data_array.drop_nulls()
            for row_index in range(0,data_array.shape[0]):
                data=data_array[row_index].to_numpy()[0]
                # if row_index <3:
                    # print(data[0:2])
                    # print(data[2:22])
                data_lib_name.append(data[0:2])
                data_lib_value.append(data[2:22])

        setup_seed(best_parameters[args.model]['random_state'])
        processor = DataProcessor(seed=best_parameters[args.model]['random_state'])
        train_dataset_all = processor.get_dataset()
        train_dataset_all = DataProcessor.shuffle(train_dataset_all)
        train_dataset, test_dataset = DataProcessor.split(train_dataset_all, args.train_data_num)
        train_loader = DataLoader(AADataset(train_dataset), batch_size=1)
        val_loader = DataLoader(AADataset(test_dataset), batch_size=1)
        train_r2,test_r2,exp_score= gpgb(train_loader.dataset,val_loader.dataset,
                                                 data_lib_name,data_lib_value,best_parameters[args.model],best_parameters[args.model]['n_components'],best_parameters[args.model]['gp_random_state'])

    elif args.model in ['ols', 'lasso', 'ridge', 'rf', 'svr', 'gpr']:
        train_loader, val_loader = DataProcessor.split(train_dataset_all, args.train_data_num)
        train_result, model = train_ml_model(
            train_loader,
            val_loader,
            model=model_map[args.model]
        )
        with open(os.path.join(args.output_dir, '{}.md'.format(args.model)), 'wb') as fp:
            pickle.dump(model, fp)
    else:
        setup_seed(best_parameters[args.model]['random_state'])
        processor = DataProcessor(seed=best_parameters[args.model]['random_state'])
        train_dataset_all = processor.get_dataset()
        train_dataset_all = DataProcessor.shuffle(train_dataset_all)
        print("train_dataset_all:", train_dataset_all['size'])
        train_result, model = train_ann(
            dims=best_parameters[args.model]['dims'],
            lr=best_parameters[args.model]['learning_rate'],
            batch_size=best_parameters[args.model]['batch_size'],
            activation=nn_activations[best_parameters[args.model]['activation']],
            epoch=best_parameters[args.model]['epoch'],
            train_dataset_all=train_dataset_all,
            seed=best_parameters[args.model]['random_state']
        )
        torch.save(model.state_dict(), os.path.join(args.output_dir, '{}.md'.format(args.model)))
    # with open(os.path.join(args.output_dir, '{}_train_result.json'.format(args.model)), 'w') as fp:
    #     fp.write(json.dumps(train_result))


