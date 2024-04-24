import pika
from pika.exchange_type import ExchangeType
import tensorflow as tf
import json
import time
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import BatchNormalization

class MessageConsumer:
    def __init__(self):

        self.data = np.load('al-al.npy') # train data
        self.test_data = np.load('cu-al.npy') # test data
        self.Clinet_name = 5
        self.Client_num = 2
        self.num_of_clients = 3
        self.routing_key = "Material_6"
        self.queue = f'{self.routing_key}_{self.Clinet_name}'
        self.exchange = 'Fed_weights_6'
        self.server_exchange = 'server_exch_6'
        self.server_queue ='server_6'
        self.user = 'FL'
        self.password = '12345'
        self.IP = '*************'
        self.Vhost = 'FL_vhost'
        self.Client_epoch = -1
        self.BN_size = 0
        self.layers= []
        self.Test_class_name = ''
        self.l2_regularization = 0 # regularization of hidden layer
        self.lr_rate = -1 
        self.seed_num = 10
        self.input_shape= 2
        self.hidden_layer = 0
        self.optimizer = None
        self.proximal_term_weight = 0
        self.reg_layer = [0,0,0]

    def consume(self):
        credentials = pika.PlainCredentials(self.user, self.password)
        parameters = pika.ConnectionParameters(
            host=self.IP,
            port=5672,
            virtual_host=self.Vhost,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300
        )

        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        # Declare client side queues and exchange

        channel.exchange_declare(exchange=self.exchange, exchange_type=ExchangeType.direct, durable=True)

        # Declare the durable queue
        channel.queue_declare(queue=self.queue, auto_delete=True)

        # Bind the queue to the exchange
        channel.queue_bind(exchange=self.exchange, queue=self.queue, routing_key=self.routing_key)

        # Declare server side queues and exchange

        # Declare the durable direct exchange
        channel.exchange_declare(exchange=self.server_exchange, exchange_type=ExchangeType.direct, durable=True)

        # Declare the durable queue
        channel.queue_declare(queue=self.server_queue, durable=True)

        # Bind the queue to the exchange with a routing key
        channel.queue_bind(exchange=self.server_exchange, queue=self.server_queue, routing_key='')

        # consume 
        channel.basic_consume(queue=self.queue, auto_ack=True, on_message_callback=self.on_message_received)
        print("Starting Consuming")
        channel.start_consuming()


    def load_data(self):
        np.random.seed(self.seed_num)
        np.random.shuffle(self.data)

        x_train1 = self.data[:, 1:]  # Input features
        y_train1 = keras.utils.to_categorical(self.data[:, 0], num_classes=self.layers[-1])  # Labels (one-hot encoding)
        x_test1 = self.test_data[:, 1:]  # Input features
        y_test = keras.utils.to_categorical(self.test_data[:, 0], num_classes=self.layers[-1])  # Labels (one-hot encoding)


        #number of data in this client
        data_per_client = len(x_train1) // self.num_of_clients
        x1 = x_train1[self.Client_num * data_per_client : (self.Client_num + 1) * data_per_client,:]  # Input features
        y_train = y_train1[self.Client_num * data_per_client : (self.Client_num + 1) * data_per_client] # Labels (one-hot encoding)
        # standardising the data
        sc = StandardScaler()
        sc.fit(x_train1)
        x_test = sc.transform(x_test1)
        x_train = sc.transform(x1)
        return x_train, y_train, x_test, y_test


    def loss_fn(self, y_true, y_pred):
        # Define the loss function
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    def set_optimizer(self):
        # Set the optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_rate)

    def create_model(self):

        # Load the data
        x_train, y_train, _, _ = self.load_data()

        # Define the input shape based on the data
        self.input_shape = x_train.shape[1]

        self.set_optimizer()
        model = keras.Sequential()

        model.add(keras.layers.Dense(self.layers[0], activation='relu', input_shape=(self.input_shape,),
                                kernel_regularizer=regularizers.l2(self.reg_layer[0])))

        # Add the hidden layers with L2 regularization and BatchNormalization
        for index, neurons in enumerate(self.layers[1:-1]):
            model.add(keras.layers.Dense(neurons, activation='relu',
                                kernel_regularizer=regularizers.l2(self.reg_layer[index+1])))

        # Add the output layer
        model.add(keras.layers.Dense(self.layers[-1], activation='softmax'))

        model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=['accuracy'])

        return model

    def on_message_received(self, ch, method, properties, body):
        A = json.loads(body.decode('utf-8'))
        topic = A["topic"]
        print("Received", A["topic"])
        weights = A['weights']
        # print(weights[0][0][0])
        time.sleep(0.0001)
        if topic == "initial_weight":
            self.layers= A["model"]
            self.BN_size = A["Batch"]
            self.Client_epoch = A["epoch"]
            self.Test_class_name = A["test_class"]
            self.l2_regularization = A["regularization"]
            self.lr_rate = A["lr"]
            self.seed_num = A["seed_num"]
            self.hidden_layer = A["hidden_layer"]
            self.proximal_term_weight =  A["prox_term"]
            self.reg_layer = A["reg_layer"]
        else:
            pass


        if (len(weights) > 0 and topic == "updated_weight") or (len(weights) > 0 and topic == "initial_weight"):
            # print(f"recieved {topic}")
            for i, k in enumerate(weights):
                weights[i] = np.array(weights[i])

            initial_global_weights = weights 
            # Split data into train and test sets
            x_train, y_train , x_test, y_test = self.load_data()

            # Define input and output sizes
            input_shape = x_train.shape[1]
            self.set_optimizer()

            model = self.create_model()

            model.set_weights(weights)
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)


            fold_test_acc = []
            fold_test_loss = []


            kfold = KFold(n_splits=5, shuffle=True,random_state = self.seed_num + 10 )

            for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train)):
                print(f"Fold {fold + 1}")
                x_train_fold = x_train[train_idx]
                y_train_fold = y_train[train_idx]
                x_val_fold = x_train[val_idx]
                y_val_fold = y_train[val_idx]
               # Train the model on this fold
                num_batches = len(x_train_fold) // self.BN_size

                for epoch in range(self.Client_epoch):
                    for batch in range(num_batches):
                        start = batch * self.BN_size
                        end = start + self.BN_size

                        with tf.GradientTape(persistent=True) as tape:
                            intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.layers[self.hidden_layer].output)
                            z = intermediate_model(x_train_fold[start:end])
                            reg = tf.reduce_mean(tf.norm(z))

                            # Compute the logits and loss
                            logits = model(x_train_fold[start:end])
                            loss = self.loss_fn(y_train_fold[start:end], logits)
                            loss += reg * self.l2_regularization
                            # Add proximal term to the loss
                            proximal_term = 0.0
                            for weight_client, weight_global in zip(model.trainable_variables, initial_global_weights):
                                proximal_term += tf.reduce_sum(tf.square(weight_client - weight_global))
                                
                            loss += self.proximal_term_weight * proximal_term

                        gradients = tape.gradient(loss, model.trainable_variables)
                        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                val_loss, val_acc = model.evaluate(x_val_fold, y_val_fold, verbose=0)
                fold_test_loss.append(val_loss)
                fold_test_acc.append(val_acc)

            avg_test_loss = np.mean(fold_test_loss)
            avg_test_acc = np.mean(fold_test_acc)

            # Get the updated weights to send back to the server
            wei = model.get_weights()

            for i, h in enumerate(wei):
                wei[i] = wei[i].tolist()
                
            T = proximal_term.numpy()
            prox = T.tolist()
            D = dict()

            # print(f"{self.queue} updated weights: {wei[0][0][0]}")
            D = {"topic": self.queue, "weights": wei  , "data_point" : x_train.shape[0] ,"Test_acc" : test_acc , "Test_loss" : test_loss , "proximal_term": prox}
            D1 = json.dumps(D)

            time.sleep(0.0001)

            ch.basic_publish(
                exchange = self.server_exchange,
                routing_key='',
                body=D1
            )
        else:
            print("it is not weights")
# data = np.load('al-cu.npy')

message_consumer = MessageConsumer()
message_consumer.consume()
