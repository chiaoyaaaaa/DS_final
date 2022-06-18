import numpy as np
import torch
import torch.nn.functional as F
import pickle

from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from utils.plot_utils import plot_curve, plot_sample
from utils.utils import build_optimizer, objectview, get_known_mask, mask_edge

def train_gnn_y(data, args, log_path, device=torch.device('cpu')):
    model = get_gnn(data, args).to(device)

    if args.impute_hiddens == '':
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int, args.impute_hiddens.split('_')))
    if args.concat_states:
        input_dim = args.node_dim * len(model.convs) * 2
    else:
        input_dim = args.node_dim * 2
    # add
    if hasattr(args,'impute_ce') and args.impute_ce:
        output_dim = data.class_values
    else:
        output_dim = 1
    # add
    impute_model = MLPNet(input_dim, output_dim,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            dropout=args.dropout).to(device)

    if args.predict_hiddens == '':
        predict_hiddens = []
    else:
        predict_hiddens = list(map(int, args.predict_hiddens.split('_')))

    # add
    if args.data_type == 'cls':
        output_dim = data.class_y
    else:
        output_dim = 1
    n_row, n_col = data.df_X.shape
    predict_model = MLPNet(n_col, output_dim,
                           hidden_layer_sizes=predict_hiddens,
                           dropout=args.dropout).to(device)

    trainable_parameters = list(model.parameters()) \
                           + list(impute_model.parameters()) \
                           + list(predict_model.parameters())
    print("total trainable_parameters: ",len(trainable_parameters))

    # build optimizer
    scheduler, opt = build_optimizer(args, trainable_parameters)

    # train
    # add
    Train_loss = []
    Test_loss = []
    if hasattr(args,'impute_ce') and args.impute_ce:
        Impute_accuracy = []
    else:
        Impute_rmse = []
        Impute_l1 = []
    if args.data_type == 'cls':
        Predict_accuracy = []
    else: 
        Predict_rmse = []
        Predict_l1 = []
    Lr = []

    x = data.x.clone().detach().to(device)
    y = data.y.clone().detach().to(device)
    edge_index = data.edge_index.clone().detach().to(device)
    train_edge_index = data.train_edge_index.clone().detach().to(device)
    train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    train_labels = data.train_labels.clone().detach().to(device)
    all_train_y_mask = data.train_y_mask.clone().detach().to(device)
    test_y_mask = data.test_y_mask.clone().detach().to(device)
    test_edge_index = data.test_edge_index.clone().detach().to(device)
    test_edge_attr = data.test_edge_attr.clone().detach().to(device)
    test_labels = data.test_labels.clone().detach().to(device)

    # add
    if hasattr(data,'class_values'):
        class_values = data.class_values

    if args.valid > 0.:
        valid_mask = get_known_mask(args.valid, all_train_y_mask.shape[0]).to(device)
        valid_mask = valid_mask*all_train_y_mask
        train_y_mask = all_train_y_mask.clone().detach()
        train_y_mask[valid_mask] = False
        valid_y_mask = all_train_y_mask.clone().detach()
        valid_y_mask[~valid_mask] = False
        print("all y num is {}, train num is {}, valid num is {}, test num is {}"\
                .format(
                all_train_y_mask.shape[0],torch.sum(train_y_mask),
                torch.sum(valid_y_mask),torch.sum(test_y_mask)))
        Valid_rmse = []
        Valid_l1 = []
        best_valid_rmse = np.inf
        best_valid_rmse_epoch = 0
        best_valid_l1 = np.inf
        best_valid_l1_epoch = 0
    else:
        train_y_mask = all_train_y_mask.clone().detach()
        print("all y num is {}, train num is {}, test num is {}"\
                .format(
                all_train_y_mask.shape[0],torch.sum(train_y_mask),
                torch.sum(test_y_mask)))

    for epoch in range(args.epochs):
        model.train()
        impute_model.train()
        predict_model.train()

        known_mask = get_known_mask(args.known, int(train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)

        ## ---------- Imputation ----------
        # original (all data)
        X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
        X = torch.reshape(X, [n_row, n_col])

        # add (train data)
        pred = impute_model([x_embd[train_edge_index[0]], x_embd[train_edge_index[1]]])

        # add
        if hasattr(args,'impute_ce') and args.impute_ce:
            pred_train = pred[:int(train_edge_attr.shape[0] / 2)]
        else:
            pred_train = pred[:int(train_edge_attr.shape[0] / 2), 0]
        label_train = train_labels

        # add
        if hasattr(args,'impute_ce') and args.impute_ce:
            impute_loss = F.cross_entropy(pred_train, label_train.type(torch.LongTensor).to(device))
        else:
            impute_loss = F.mse_loss(pred_train, label_train)

        ## ---------- Prediction ----------
        # add
        if args.data_type == 'cls':
            pred = predict_model(X)
        else:
            pred = predict_model(X)[:, 0]
        pred_train = pred[train_y_mask]
        label_train = y[train_y_mask]

        # add
        if args.data_type == 'cls':
            predict_loss = F.cross_entropy(pred_train, label_train.type(torch.LongTensor).to(device))
        else:
            predict_loss = F.mse_loss(pred_train, label_train)

        ## ---------- Joint loss ----------
        # add
        loss = args.impute_loss_weight * impute_loss + args.predict_loss_weight * predict_loss
        loss.backward()
        opt.step()
        train_loss = loss.item()
        if scheduler is not None:
            scheduler.step(epoch)
        for param_group in opt.param_groups:
            Lr.append(param_group['lr'])

        model.eval()
        impute_model.eval()
        predict_model.eval()
        with torch.no_grad():
            if args.valid > 0.:
                x_embd = model(x, train_edge_attr, train_edge_index)
                X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
                X = torch.reshape(X, [n_row, n_col])
                pred = predict_model(X)[:, 0]
                pred_valid = pred[valid_y_mask]
                label_valid = y[valid_y_mask]
                mse = F.mse_loss(pred_valid, label_valid)
                valid_rmse = np.sqrt(mse.item())
                l1 = F.l1_loss(pred_valid, label_valid)
                valid_l1 = l1.item()
                if valid_l1 < best_valid_l1:
                    best_valid_l1 = valid_l1
                    best_valid_l1_epoch = epoch
                    torch.save(model, log_path + 'model_best_valid_l1.pt')
                    torch.save(impute_model, log_path + 'impute_model_best_valid_l1.pt')
                    torch.save(predict_model, log_path + 'predict_model_best_valid_l1.pt')
                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_valid_rmse_epoch = epoch
                    torch.save(model, log_path + 'model_best_valid_rmse.pt')
                    torch.save(impute_model, log_path + 'impute_model_best_valid_rmse.pt')
                    torch.save(predict_model, log_path + 'predict_model_best_valid_rmse.pt')
                Valid_rmse.append(valid_rmse)
                Valid_l1.append(valid_l1)

            x_embd = model(x, train_edge_attr, train_edge_index)

            ## ---------- Imputation ----------
            # original (all data)
            X = impute_model([x_embd[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
            X = torch.reshape(X, [n_row, n_col])

            # add (test data)
            pred = impute_model([x_embd[test_edge_index[0]], x_embd[test_edge_index[1]]])

            # add
            if hasattr(args,'impute_ce') and args.impute_ce:
                pred_test = pred[:int(test_edge_attr.shape[0] / 2)]
            else:
                pred_test = pred[:int(test_edge_attr.shape[0] / 2),0]
            label_test = test_labels

            # add
            if len(test_labels) == 0:
                impute_loss, impute_acc, impute_rmse, impute_l1 = 0, np.nan, np.nan, np.nan
            elif hasattr(args,'impute_ce') and args.impute_ce:
                impute_loss = F.cross_entropy(pred_test, label_test.type(torch.LongTensor).to(device)).item()
                impute_acc = (torch.sum(pred_test.argmax(dim = 1) == label_test) / label_test.shape[0]).item()
            else:
                impute_loss = F.mse_loss(pred_test, label_test).item()
                impute_rmse = np.sqrt(impute_loss)
                l1 = F.l1_loss(pred_test, label_test)
                impute_l1 = l1.item()
            
            ## ---------- Prediction ----------
            # add
            if args.data_type == 'cls':
                pred = predict_model(X)
            else:
                pred = predict_model(X)[:, 0]
            pred_test = pred[test_y_mask]
            label_test = y[test_y_mask]

            # add
            if args.data_type == 'cls':
                predict_loss = F.cross_entropy(pred_test, label_test.type(torch.LongTensor).to(device)).item()
                predict_acc = (torch.sum(pred_test.argmax(dim = 1) == label_test) / label_test.shape[0]).item()
            else:
                predict_loss = F.mse_loss(pred_test, label_test).item()
                predict_rmse = np.sqrt(predict_loss)
                l1 = F.l1_loss(pred_test, label_test)
                predict_l1 = l1.item()

            ## ---------- Joint loss ----------
            # add
            test_loss = args.impute_loss_weight * impute_loss + args.predict_loss_weight * predict_loss
            
            # add
            Train_loss.append(train_loss)
            Test_loss.append(test_loss)
            print('epoch: ', epoch)
            print('train loss: ', train_loss)
            print('test loss: ', test_loss)
            print('loss weight: imputation {}, prediction {}'.format(args.impute_loss_weight, args.predict_loss_weight))

            # add
            if hasattr(args,'impute_ce') and args.impute_ce:
                Impute_accuracy.append(impute_acc)
                print('impute accuracy: ', impute_acc)
            else:
                Impute_rmse.append(impute_rmse)
                Impute_l1.append(impute_l1)
                print('impute rmse: ', impute_rmse)
                print('impute l1: ', impute_l1)

            # add
            if args.data_type == 'cls':
                Predict_accuracy.append(predict_acc)
                print('predict accuracy: ', predict_acc)
            else:
                Predict_rmse.append(predict_rmse)
                Predict_l1.append(predict_l1)
                print('predict rmse: ', predict_rmse)
                print('predict l1: ', predict_l1)

            if args.valid > 0.:
                print('valid rmse: ', valid_rmse)
                print('valid l1: ', valid_l1)

    pred_train = pred_train.detach().cpu().numpy()
    label_train = label_train.detach().cpu().numpy()
    pred_test = pred_test.detach().cpu().numpy()
    label_test = label_test.detach().cpu().numpy()

    obj = dict()
    obj['args'] = args
    obj['curves'] = dict()
    obj['curves']['train_loss'] = Train_loss
    obj['curves']['test_loss'] = Test_loss
    if args.valid > 0.:
        obj['curves']['valid_rmse'] = Valid_rmse
        obj['curves']['valid_l1'] = Valid_l1

    obj['curves']['impute_rmse'] = Impute_rmse
    obj['curves']['impute_l1'] = Impute_l1
    if args.data_type == 'cls':
        obj['curves']['predict_accuracy'] = Predict_accuracy
    else:
        obj['curves']['predict_rmse'] = Predict_rmse
        obj['curves']['predict_l1'] = Predict_l1
    obj['lr'] = Lr

    obj['outputs'] = dict()
    obj['outputs']['pred_train'] = pred_train
    obj['outputs']['label_train'] = label_train
    obj['outputs']['pred_test'] = pred_test
    obj['outputs']['label_test'] = label_test

    obj['data'] = dict()
    obj['data']['data'] = data
    obj['data']['train'] = train_y_mask
    obj['data']['test'] = test_y_mask
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

    torch.save(model, log_path + 'model.pt')
    torch.save(impute_model, log_path + 'impute_model.pt')
    torch.save(predict_model, log_path + 'predict_model.pt')

    # obj = objectview(obj)
    plot_curve(obj['curves'], log_path+'curves.png',keys=None, 
                clip=True, label_min=True, label_end=True)
    plot_curve(obj, log_path+'lr.png',keys=['lr'], 
                clip=False, label_min=False, label_end=False)
    plot_sample(obj['outputs'], log_path+'outputs.png', 
                groups=[['pred_train','label_train'],
                        ['pred_test','label_test']
                        ], 
                num_points=20)
    if args.valid > 0.:
        print("best valid rmse is {:.3g} at epoch {}".format(best_valid_rmse,best_valid_rmse_epoch))
        print("best valid l1 is {:.3g} at epoch {}".format(best_valid_l1,best_valid_l1_epoch))