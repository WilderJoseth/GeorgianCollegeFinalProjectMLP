# Georgian College Final Project MLP
Georgian College Final Project Machine Learning Frameworks - AI

This project is based on https://github.com/tusharsarkar3/XBNet repository from Tushar Sarkar.

Link paper: https://paperswithcode.com/paper/xbnet-an-extremely-boosted-neural-network

The purpose of this project was to test XBNet changing the number of trees (n_estimators) on XGBClassifier. To achieve the goal, I changed the hyperparameter on model constructor and base_tree method for classification taks.

    def __init__(self, X_values, y_values, num_layers, num_layers_boosted=1,
                 input_through_cmd = False,inputs_for_gui=None):
        super(XBNETClassifier, self).__init__()
        self.name = "Classification"
        self.layers = OrderedDict()
        self.boosted_layers = {}
        self.num_layers = num_layers
        self.num_layers_boosted = num_layers_boosted
        self.X = X_values
        self.y = y_values
        self.gui = input_through_cmd
        self.inputs_layers_gui = inputs_for_gui

        self.take_layers_dim()
        self.base_tree()

        self.layers[str(0)].weight = torch.nn.Parameter(torch.from_numpy(self.temp.T))

        print('AIDI 1002 Final project modification, number of n_estimators:', 150)
        self.xg = XGBClassifier(n_estimators=150)

        self.sequential = Seq(self.layers)
        self.sequential.give(self.xg, self.num_layers_boosted)
        self.feature_importances_ = None

    def base_tree(self):
        '''
        Instantiates and trains a XGBRegressor on the first layer of the neural network to set its feature importances
         as the weights of the layer
        '''
        print('AIDI 1002 Final project modification, number of n_estimators:', 150)
        self.temp1 = XGBClassifier(n_estimators=150).fit(self.X, self.y,eval_metric="mlogloss").feature_importances_
        self.temp = self.temp1
        for i in range(1, self.input_out_dim):
            self.temp = np.column_stack((self.temp, self.temp1))

After that, import new version from directory.

    from XBNet2 import models, Seq
    from XBNet2.run import run_XBNET
    import torch
    
    importlib.reload(models)
    importlib.reload(Seq)
