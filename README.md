USI Lugano Graph Deep Learning (Spring 2022) project

#
Team members: Joonsu Gha ([**@arahosu**](https://github.com/arahosu)), Rahil Doshi ([**@rahildoshi97**](https://github.com/rahildoshi97)), Naga Venkata Sai Jitin Jami ([**@jitinjami**](https://github.com/jitinjami))

#Installation
'tsl' is compatible with Python>=3.7. We recomend installation from the source to get the latest version:
    git clone hhtps://github.com/TorchSpatiotemporal/tsl.git
    cd tsl
    pip install .

Alternatively, you can install the library from pip
    pip install torch-spatiotemporal

#Execution
The code is exicuted with "main.py" with the following flags:
'--dataset-name' takes the name of the dataset of 'type=str' with arguments 'la' or 'bay' for the 'MetrLA()' or 'PemsBay()' dataset respectively.

    '--model' takes the name of the model with arguments 'transformer' or 'graphormer' for the 'TransformerModel' or 'GraphormerModel' respectively.

    '--horizon' forecasts the upcoming time steps. Defaults argument is 12 with 'typr=int'.

    '--window' takes in the window size of 'type=int' for slicing the time series with a default value of 12.

    '--batch_size' takes in an argument of 'type=int', 'default=32'.

    '--n_epochs' is the number of epochs, 'type=int', 'default=40'.

    '--val_len' is the validation length ratio of 'type=float' with 'default=0.1'.

    '--test_len' is the test length ratio of 'type=float' with 'default=0.2'.

    '--hidden_size' is the number of hidden nodes in every layer of 'type=int' with 'default=64'.

    '--ff_size' is the size of the feed forward neural network of 'type=int' with 'default=64'.

    '--n_head' is the number of heards in multihead attention of 'type=int' with 'default=2'.

    '--n_layers' is the number of transformer or graphormer layers of 'type=int' with 'default=2'.

    '--dropout' is the probability of the nodes (neurons) dropping out, 'type=float', 'default=0.0'.

    '--axis' takes input argument of 'step' or 'both'.

    '--activation' takes the activation function of 'type=str' with arguments 'elu' or 'relu'.

    '--sge' activates the spatial graph embeding with imput argument 'True', 'default=False'.

    '--ce' enables the centrality encoding with imput argument 'True', 'default=False'.