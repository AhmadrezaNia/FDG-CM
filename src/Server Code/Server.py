import pika
import time
import os
import numpy as np
import json
from pika.exchange_type import ExchangeType
from tensorflow import keras
import requests
import statistics
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras import regularizers

class Server:
    def __init__(self):
        self.test_name_S = 'clean' # Surface 
        self.test_name_M = 'cu-al' # Material 
        self.data_S = np.load('All.npy') #load Surface test set 
        self.seed_num = 135
        self.hidden_layer = 2
        self.test_class = [0,3,6]  # test classes for Surface condition 
        self.data_M = np.load(f'{self.test_name_M}.npy') # load Material test set 
        self.BN_S = 8
        self.Client_epoch_S = 1
        self.BN_M = 8
        self.Client_epoch_M = 1
        self.layers_S = [175, 125, 50, 3] # model Surface
        self.layers_M = [175, 125, 50, 4] # model Material 
        self.regular_S = 0.0001 # regularization of hidden layer 
        self.regular_M = 0.0001 # regularization of hidden layer
        self.reg_layers_S = [0,0,0]
        self.reg_layers_M = [0,0,0]
        self.lr_S = 0.0005  
        self.lr_M = 0.0005  
        self.prox_term = 1

        self.Personalized_layer = 2
        # mapping for Surface 
        self.label_mapping = {
            0: 0, 1: 0, 2: 0,
            3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2
        }

        self.plot_interval = 5
        self.routing_key_S = "Surface_6"
        self.routing_key_M = "Material_6"
        self.server_exchange = 'server_exch_6'
        self.server_queue = 'server_6'
        self.epoch = 0
        self.weights_exchange = 'Fed_weights_6'
        self.user = 'FL'
        self.password = '12345'
        self.IP = '**********'
        self.Vhost = 'FL_vhost'
        self.received_weights = {}
        self.recieved_test_Acc = {}
        self.recieved_test_Loss = {}
        self.total_accuracy_S= []
        self.total_accuracy_M= []
        self.total_loss_S = []
        self.total_loss_M= []
        self.data_points= {}
        self.prox_terms = {}
        # Define the list of dataset names
        dataset_names = ["al-cu", "cu-al", "cu-cu", "al-al"]

        # Remove the test dataset name from the list
        dataset_names.remove(self.test_name_M)

        # Initialize an empty list to store the training data
        self.training_data = []

        # Read and concatenate the training datasets
        for dataset_name in dataset_names:
            dataset = np.load(f'{dataset_name}.npy')
            self.training_data.append(dataset)

        # Concatenate the training datasets along the first axis
        self.training_data = np.concatenate(self.training_data)


        # define the Averaging algoritm 
    def calculate_average(self, weights_list, data_points_list):
        # Ensure at least two weight sets are provided
        if len(weights_list) < 2:
            raise ValueError('At least two weight sets must be provided')

        # Get the length of the weight sets
        length = len(weights_list[0])

        # Check if all weight sets have the same length
        if not all(len(weights) == length for weights in weights_list):
            raise ValueError('Weight sets must have the same length')

        # Create an empty list to store the results
        result = []

        # Loop through the weight sets and calculate the weighted average of each pair of elements
        for i in range(length):
            if isinstance(weights_list[0][i], list):
                # If the element is a list, calculate the weighted average of sublists
                sublist = self.calculate_average([weights[i] for weights in weights_list], data_points_list)
                result.append(sublist)
            else:
                # If the element is a number, calculate the weighted average of numbers
                weighted_average = sum(weights[i] * data_points_list[index] for index, weights in enumerate(weights_list)) / sum(data_points_list)
                result.append(weighted_average)

        # Return the resulting list
        return result
    
    # inpu the standardized test data and the layers to the following model
    def calculate_model_accuracy(self, weights, test_data, test_labels, layers, test_names ,L2_regular):
        # Load the test data
        
        for i, k in enumerate(weights):
            weights[i] = np.array(weights[i])

        input_shape = test_data.shape[1]

        model1 = keras.Sequential()

        model1.add(keras.layers.Dense(layers[0], activation='relu', input_shape=(input_shape,),
                                    kernel_regularizer=regularizers.l2(L2_regular[0])))


        # Add the hidden layers with L2 regularization and BatchNormalization
        for index , neurons in enumerate(layers[1:-1]):
            model1.add(keras.layers.Dense(neurons, activation='relu',
                                        kernel_regularizer=regularizers.l2(L2_regular[index+1])))

        # Add the output layer
        model1.add(keras.layers.Dense(layers[-1], activation='softmax'))


        model1.set_weights(weights)

        model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Evaluate the model on the test data
        loss, accuracy = model1.evaluate(test_data, test_labels, verbose=0)
        print( f"server test accuracy for {test_names} is : {accuracy} and loss is :{loss}")

        return accuracy , loss


    def plot_metrics(self):
        epochs = range(0, self.epoch)

        # Surface model metrics
        accuracy_surface = np.array(self.total_accuracy_S)
        loss_surface = np.array(self.total_loss_S)

        # Material model metrics
        accuracy_material = np.array(self.total_accuracy_M)
        loss_material = np.array(self.total_loss_M)

        folder_name = f'tests/{self.test_name_M}_{self.test_name_S}/Bns{self.BN_S}_BN_M{self.BN_M}_{self.layers_S[0]}_epoch_{self.Client_epoch_S}_prox_term{self.prox_term}_regular{self.regular_S}_seed{self.seed_num}_lr{self.lr_S}'
        if not os.path.exists(folder_name):
            # Create the "tests" folder
            os.makedirs(folder_name)

        # Plot and save surface model loss
        plt.plot(epochs, loss_surface, label='Surface Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.test_name_S} Surface Model Loss')
        plt.legend()
        plt.savefig(f"{folder_name}/{self.test_name_S}_surface_loss_plot{self.epoch}.png")
        plt.close()

        # Plot and save surface model accuracy
        plt.plot(epochs, accuracy_surface, label='Surface Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{self.test_name_S} Surface Model Accuracy')
        plt.legend()
        plt.savefig(f"{folder_name}/{self.test_name_S}_surface_accuracy_plot{self.epoch}.png")
        plt.close()

        # Plot and save material model loss
        plt.plot(epochs, loss_material, label='Material Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.test_name_M} Material Model Loss')
        plt.legend()
        plt.savefig(f"{folder_name}/{self.test_name_M}_material_loss_plot{self.epoch}.png")
        plt.close()

        # Plot and save material model accuracy
        plt.plot(epochs, accuracy_material, label='Material Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'{self.test_name_M} Material Model Accuracy')
        plt.legend()
        plt.savefig(f"{folder_name}/{self.test_name_M}_material_accuracy_plot{self.epoch}.png")
        plt.close()


    
    def multiply_scalar_to_nested_list(self , nested_list, scalar):
        if isinstance(nested_list, list):
            return [self.multiply_scalar_to_nested_list(item, scalar) for item in nested_list]
        else:
            return scalar * nested_list
    

    def on_request_message_received(self, ch, method, properties, body):
        A = json.loads(body.decode('utf-8'))
        weights = A["weights"]
        topic = A["topic"]
    
        # Store the received weights in the dictionary
        self.received_weights[topic] = weights
        self.recieved_test_Acc[topic] = A["Test_acc"]
        self.recieved_test_Loss[topic] = A["Test_loss"]
        self.data_points[topic] = A["data_point"]
        self.prox_terms[topic] = A["proximal_term"]
        print(f"Received weights for topic: {topic}")

        # Check if weights are received from all queues
        if len(self.received_weights) == len(self.bound_queues):
            # Separate weights by domain
            N = 2 * self.Personalized_layer  # Specify the number of layers for personalized weights

            surface_weights = []
            material_weights = []
            surface_data_points = []
            material_data_points = []
            surface_test_acc = []
            surface_test_loss = []
            material_test_acc = []
            material_test_loss = []
            for topic, weights in self.received_weights.items():
                if topic.startswith(self.routing_key_S):
                    surface_weights.append(weights)
                    surface_data_points.append(self.data_points[topic])
                    surface_test_acc.append(self.recieved_test_Acc[topic])
                    surface_test_loss.append(self.recieved_test_Loss[topic])

                elif topic.startswith(self.routing_key_M):
                    material_weights.append(weights)
                    material_data_points.append(self.data_points[topic])
                    material_test_acc.append(self.recieved_test_Acc[topic])
                    material_test_loss.append(self.recieved_test_Loss[topic])


            # Calculate the average of personalized layers for each domain using weighted averaging
            personalized_weights_S = self.calculate_average([weights[-N:] for weights in surface_weights], surface_data_points)
            personalized_weights_M = self.calculate_average([weights[-N:] for weights in material_weights], material_data_points)

            # Calculate the average of common layers for both domains using weighted averaging
            common_weights = self.calculate_average([weights[:-N] for weights in surface_weights] + [weights[:-N] for weights in material_weights],
                                                    surface_data_points + material_data_points)

            # Concatenate the personalized weights and common layers for each domain to build models
            surface_model_weights = common_weights+ personalized_weights_S
            material_model_weights = common_weights+personalized_weights_M 
            # Publish the models to the corresponding clients based on domain
            surface_model = {"topic": 'updated_weight', "weights": surface_model_weights}
            surface_model_json = json.dumps(surface_model)
            ch.basic_publish(exchange=self.weights_exchange, routing_key=self.routing_key_S, body=surface_model_json)

            print(f"Updated surface model weights sent to clients")

            material_model = {"topic": 'updated_weight', "weights": material_model_weights}
            material_model_json = json.dumps(material_model)
            ch.basic_publish(exchange=self.weights_exchange, routing_key=self.routing_key_M, body=material_model_json)
            print(f"Updated material model weights sent to clients")
            print(f"prox terms are :{statistics.mean(self.prox_terms.values())}")
            self.prox_terms ={}
            self.epoch += 1
            self.received_weights = {}
            print(f"epoch: {self.epoch}")

            # calculate Surface Accuracy 
            x_S = self.data_S[:, 1:]  # Input features
            y_S = self.data_S[:, 0]  # Labels
            mapped_labels = np.array([self.label_mapping[label] for label in y_S])
            # Find the indices of labels that belong to the test classes
            indices_set_test = np.where(np.isin(y_S, self.test_class))[0]
            indices_set_train = np.where(~np.isin(y_S, self.test_class))[0]  # Indices for training classes

            # Convert labels to one-hot encoded format
            test_labels_S = keras.utils.to_categorical(mapped_labels[indices_set_test], num_classes=self.layers_S[-1])

            # Separate the training and test sets
            x_test1_S = x_S[indices_set_test]
            x_train_S = x_S[indices_set_train]
            # Convert training labels to one-hot encoded format
            sc = StandardScaler()
            sc.fit(x_train_S)
            test_data_S = sc.transform(x_test1_S)
            print(f"Surface Clients mean test accuracy is {np.mean(surface_test_acc)}")
            print(f"Surface Clients mean test loss is {np.mean(surface_test_loss)}")

            print(f"Material Clients mean test accuracy is {np.mean(material_test_acc)}")
            print(f"Material Clients mean test loss is {np.mean(material_test_loss)}")
            Acc_S, lss_S = self.calculate_model_accuracy(surface_model_weights, test_data_S, test_labels_S, self.layers_S, self.test_name_S ,L2_regular = self.reg_layers_S)
            self.total_accuracy_S.append([Acc_S])

            self.total_loss_S.append([lss_S])

            # calculate material Accuracy 
            x_test1_M = self.data_M[:, 1:]  # Input features
            y_test1_M = self.data_M[:, 0]  # Labels
            # Define the list of dataset names
            x_train1_M = self.training_data[:, 1:]   
            sc_M = StandardScaler()
            sc_M.fit(x_train1_M)
            test_data_M = sc_M.transform(x_test1_M)
            test_labels_M = keras.utils.to_categorical(y_test1_M, num_classes=self.layers_M[-1])
            
            Acc_M, lss_M = self.calculate_model_accuracy(material_model_weights, test_data_M, test_labels_M, self.layers_M, self.test_name_M, L2_regular = self.reg_layers_S)
            self.total_accuracy_M.append([Acc_M])

            self.total_loss_M.append([lss_M])

            if self.epoch % self.plot_interval == 0:
                self.plot_metrics()
            


    def get_initial_weights(self, data, layers):
        #getting the training data input size ( -1 (lables))
        input_shape = data.shape[1] - 1 
        
        model = keras.Sequential()

        #Add the input layer with Kaming Normal initialization
        model.add(keras.layers.Dense(layers[0], activation='relu', input_shape=(input_shape,),
                                    kernel_initializer=keras.initializers.he_normal()))

        # Add the hidden layers with Kaming Normal initialization
        for neurons in layers[1:-1]:
            model.add(keras.layers.Dense(neurons, activation='relu',
                                        kernel_initializer=keras.initializers.he_normal()))

        # Add the output layer with Kaming Normal initialization
        model.add(keras.layers.Dense(layers[-1], activation='softmax',
                                    kernel_initializer=keras.initializers.he_normal()))

        # Retrieve the initial weights of the model

        layer_weights = model.get_weights()
        for i, h in enumerate(layer_weights):
            layer_weights[i] = layer_weights[i].tolist()

        return layer_weights
    

    def get_queues_bound_to_exchange(self):
        api_url = f"http://{self.IP}:15672/api"
        url = f"{api_url}/exchanges/{self.Vhost}/{self.weights_exchange}/bindings/source"

        response = requests.get(url, auth=(self.user, self.password))

        if response.status_code == 200:
            queues = [binding['destination'] for binding in response.json()]
            return queues
        else:
            print(f"Failed to retrieve queues: {response.status_code} {response.reason}")
            return []
        

    def run(self):
        credentials = pika.PlainCredentials(self.user, self.password)
        parameters = pika.ConnectionParameters(self.IP,
                                               5672,
                                               self.Vhost,
                                               credentials, heartbeat=600,
                                               blocked_connection_timeout=300
                                                                                )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        # Declare the durable server exchange
        channel.exchange_declare(exchange=self.server_exchange, exchange_type=ExchangeType.direct, durable=True)

        # Declare the durable server queue
        channel.queue_declare(queue=self.server_queue, durable=True)

        # Bind the queue to the exchange with a routing key
        channel.queue_bind(exchange=self.server_exchange, queue=self.server_queue, routing_key='')

        # Declare the exchange for clients
        channel.exchange_declare(exchange=self.weights_exchange, exchange_type=ExchangeType.direct, durable=True)

        # Get the list of bound queue names
        self.bound_queues = self.get_queues_bound_to_exchange()
        
        # start Consuming 
        channel.basic_consume(queue=self.server_queue, auto_ack=True,
                              on_message_callback=self.on_request_message_received)

  
        # send initial weights for Surface and Material 
        initial_weights_S = self.get_initial_weights(self.data_S, self.layers_S)
        initial_weights_M = self.get_initial_weights(self.data_M, self.layers_M)
        
        #Surface
        initial_weights_json_S = json.dumps({"topic": 'initial_weight', "weights": initial_weights_S, "Batch":self.BN_S , "epoch": self.Client_epoch_S , 
                                             "model": self.layers_S , "test_class" : self.test_class , "label_mapping":self.label_mapping,
                                               "regularization" : self.regular_S, "lr":self.lr_S , "seed_num" : self.seed_num, "hidden_layer": self.hidden_layer, 
                                               "prox_term":self.prox_term, "reg_layer":self.reg_layers_S})
        # Material 
        initial_weights_json_M = json.dumps({"topic": 'initial_weight', "weights": initial_weights_M, "Batch":self.BN_M , "epoch": self.Client_epoch_M ,
                                              "model": self.layers_M , "test_class" : self.test_name_M ,"regularization" : self.regular_M , "lr":self.lr_M , 
                                              "seed_num" : self.seed_num, "hidden_layer": self.hidden_layer,"prox_term":self.prox_term, "reg_layer":self.reg_layers_M})

        channel.basic_publish(exchange=self.weights_exchange, routing_key=self.routing_key_S, body=initial_weights_json_S)
        channel.basic_publish(exchange=self.weights_exchange, routing_key=self.routing_key_M , body=initial_weights_json_M)

        print("Initial weights sent")

        channel.start_consuming()


# Create an instance of the Server class and run it
# data = np.load('cu-cu.npy')
server = Server()
server.run()