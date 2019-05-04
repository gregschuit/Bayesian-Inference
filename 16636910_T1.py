import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from time import time

from sklearn.model_selection import train_test_split


class NaiveBayes:
    def __init__(self):
        # Para calcular p(x_i == d | y = c)
        self.feature_frecuencies = dict()  # Probabilidades de cada feature dada una clase. (dict de dicts de dicts)
        # self.feature_frecuencies[class][feature][valor_feature] = frecuencia

        # Para calcular p(y == c)
        self.class_frecuencies = dict()  # Cantidad de datos de cada clase (dict)

        self.total_datos = 0  # Cantidad de datos en total (int)

    def fit(self, X, y):

        # Posibles clases y features
        self.clases = list(set(y))
        self.features = list(X)

        # Inicializamos valores segun la data
        print("Iniciando...")
        for clase in self.clases:
            self.class_frecuencies[clase] = 0
            self.feature_frecuencies[clase] = dict()
            for feature in self.features:
                self.feature_frecuencies[clase][feature] = dict()

        self.total_datos = X.shape[0]

        # Para tomar el tiempo
        t = time()
        i = 1

        # Contamos todas las frecuencias que necesitamos, tiempo esperado unos 2 minutos
        for index, row in X.iterrows():
            if i % 1000 == 0:
                print('Time remaining {:.3f} minutes...'.format(((time() - t) / (i) * (self.total_datos - i)) / 60),
                      end='\r')
            i += 1

            # Contamos las clases
            clase = y[index]
            self.class_frecuencies[clase] += 1

            # Para cada feature dada la clase, calculamos la distribucion de frecuencias
            for feature in self.features:
                entrada = row[feature]
                if entrada in self.feature_frecuencies[clase][feature]:
                    self.feature_frecuencies[clase][feature][entrada] += 1
                else:
                    self.feature_frecuencies[clase][feature][entrada] = 1

    def predict(self, x, y):
        """ x: Vector de features del dato a predecir, de la clase row de data.iterrows()
            return p(y|x): probabilidad no normalizada de que la clase sea "y" dada las features "x"
        """

        # probabilidad de la clase p(y = clase)
        p_clase = self.class_frecuencies[clase] / self.total_datos

        # Cantidad de datos en la clase
        total_given_class = self.class_frecuencies[clase]

        # Probabilidad de la clase dado los datos. Se calculará multiplicando de manera acumulativa
        p_y = p_clase

        # Calculamos p(features | clase) asumiendo independencia entre las features
        for feature in self.features:
            # Almacenamos el valor de la feature
            dato = x[feature]

            # Calculamos cuantas veces aparece el valor dada la clase
            frecuencia = self.feature_frecuencies[clase][feature][dato] \
                if dato in self.feature_frecuencies[clase][feature] else 0

            # p(x_i | y)
            p_dato_given_class = frecuencia / total_given_class

            # Se acumula la probabilidad calculada multiplicándola
            p_y *= p_dato_given_class

        return p_y


class BayesianNaiveBayes(NaiveBayes):
    def __init__(self):
        super().__init__()
        self.alpha = dict()  # self.alpha[clase] = int
        self.alpha_sum = 0
        self.beta = dict()  # self.beta[clase][feature][value] = int
        self.beta_sum = dict()  # self.beta_sum[clase][feature] = int

    def fit(self, X, y):
        super().fit(X, y)

        # Almacenamos todos los posibles valores de cada feature.
        self.posible_feature_values = dict()
        for feature in self.features:
            self.posible_feature_values[feature] = set(X[feature])

        self.change_priors(10000, 1000)

    def predict(self, x, y):
        """ x: Vector de features del dato a predecir, de la clase row de data.iterrows()
            return p(y|x): probabilidad no normalizada de que la clase sea "y" dada las features "x"
        """

        # Hacemos lo mismo que en Non-Bayesian Naive Bayes pero
        # aplicamos conteos imaginarios aquí...
        p_clase = (self.class_frecuencies[clase] + self.alpha[clase]) / (self.total_datos + self.alpha_sum)

        p_y = p_clase

        total_given_class = self.class_frecuencies[clase]

        for feature in self.features:
            dato = x[feature]

            frecuencia = self.feature_frecuencies[clase][feature][dato] \
                if dato in self.feature_frecuencies[clase][feature] else 0

            # ...y aquí.
            p_dato_given_class = (frecuencia + self.beta[clase][feature][dato]) / (
            total_given_class + self.beta_sum[clase][feature])

            p_y *= p_dato_given_class

        return p_y

    def change_priors(self, alpha=1000, beta=10):

        # Creamos los valores para alpha y beta, suponiendo igual cantidad para todos.
        for clase in self.clases:
            self.alpha[clase] = alpha  # alpha_c
            self.alpha_sum += beta  # alpha_0
            self.beta[clase] = dict()
            self.beta_sum[clase] = dict()

            for feature in self.features:
                self.beta[clase][feature] = dict()
                self.beta_sum[clase][feature] = 0

                for value in self.posible_feature_values[feature]:
                    self.beta[clase][feature][value] = 1  # Beta_jc
                    self.beta_sum[clase][feature] += 1  # Beta_0


def confusion_matrix(model, X_test, y_test):
    # Inicializamos la matriz
    matrix = [[0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]]

    # Para tomar el tiempo
    i = 1
    t = time()
    test_size = X_test.shape[0]

    # Iteramos sobre los datos para testear el modelo
    for index, row in X_test.iterrows():

        true_class = y_test[index]
        predicted_class = 4
        probability = 0

        # Seleccionamos la clase con mayor probabilidad según la predicción
        for clase in range(6):
            p = model.predict(row, clase)

            if p > probability:
                probability = p
                predicted_class = clase

        # Registramos la prediccion en la matriz
        matrix[true_class][predicted_class] += 1

        # Informamos el tiempo cada 1000 iteraciones
        if i % 1000 == 0:
            print('Time remaining {:.3f} minutes...'.format(((time() - t) / (i) * (test_size - i)) / 60), end='\r')
        i += 1

    return matrix


if __name__ == '__main__':

    # Leemos los datos
    data = pd.read_csv('FATS_OGLE_bin.dat', sep="\t")
    print(data.shape)
    # data.head()

    # Observamos las clases existentes y sus frecuencias
    class_counts = data['Class'].value_counts()

    # Realizamos muestreo estratificado
    X = data[['Amplitude', 'Std', 'Period', 'Mean', 'MaxSlope', 'Meanvariance', 'LinearTrend']]
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123, stratify=data['Class'])

    # Instanciamos al modelo y lo entrenamos
    NB = NaiveBayes()
    NB.fit(X_train, y_train)

    # Instanciamos al modelo y lo entrenamos
    BayesianNB = BayesianNaiveBayes()
    BayesianNB.fit(X_train, y_train)


    #Tests
    for CLASE in range(6):

        sum_probabilities = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        sum_bayesian = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        BayesianNB.change_priors(alpha=100000, beta=10000)
        for index, row in X_test[y_test == CLASE].iterrows():

            for clase in sum_probabilities:
                sum_probabilities[clase] += NB.predict(row, clase)
                sum_bayesian[clase] += BayesianNB.predict(row, clase)

        frec_clase = y_test.value_counts()[CLASE]
        plt.figure(figsize=(15, 6))
        plt.scatter(sum_probabilities.keys(), [np.log(d / frec_clase) for d in sum_probabilities.values()],
                    label='Traditional')
        plt.scatter(sum_bayesian.keys(), [np.log(d / frec_clase) for d in sum_bayesian.values()], color='orange',
                    label='Bayesian')

        plt.title(f'Suma de log(p(y|x)) para datos de la clase {CLASE}')
        plt.xlabel('Candidato a la clase y')
        plt.ylabel('Suma de log(p(y|x))')
        plt.legend()
        plt.show()

    aciertos = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    i = 1
    t = time()
    test_size = X_test.shape[0]

    for index, row in X_test.iterrows():

        true_class = y_test[index]
        predicted_class = 0
        probability = 0

        for clase in aciertos:
            p = NB.predict(row, clase)

            if p > probability:
                probability = p
                predicted_class = clase

        if true_class == predicted_class:
            aciertos[true_class] += 1

        if i % 1000 == 0:
            print('Time remaining {:.3f} minutes...'.format(((time() - t) / (i) * (test_size - i)) / 60), end='\r')
        i += 1

    for key in aciertos:
        aciertos[key] /= y_test.value_counts()[key]

    test_size = X_test.shape[0]

    alphas = [1000, 10000, 30000]  # , 40000, 60000, 80000, 100000]
    colors = {0: 'yellow', 1: 'orange', 2: 'red', 3: 'purple', 4: 'blue', 5: 'green'}

    plt.figure(figsize=(15, 6))
    linx = np.linspace(min(alphas), max(alphas), 1000)
    for clase in aciertos:
        plt.plot(linx, aciertos[clase] * np.ones(len(linx)), color=colors[clase], linestyle='--')

    for j in range(len(alphas)):

        # Informamos la iteracion
        print(f"iteracion {j}... ")

        # Cambiamos los priors sin tener que volver a entrenar el modelo
        BayesianNB.change_priors(alpha=alphas[j])

        # Registramos metricas
        true_positives = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        false_positives = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        false_negatives = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        # Para tomar el tiempo
        i = 1
        t = time()

        for index, row in X_test.iterrows():

            true_class = y_test[index]
            predicted_class = 0
            probability = 0

            # Seleccionamos la clase con mayor probabilidad según la predicción
            for clase in range(6):
                p = BayesianNB.predict(row, clase)

                if p > probability:
                    probability = p
                    predicted_class = clase

            # Registramos resultados
            if true_class == predicted_class:
                true_positives[true_class] += 1
            else:
                false_negatives[true_class] += 1
                false_positives[predicted_class] += 1

            # Informamos el tiempo cada 1000 iteraciones
            if i % 1000 == 0:
                print('Time remaining {:.3f} minutes...'.format(((time() - t) / (i) * (test_size - i)) / 60), end='\r')
            i += 1

        recall_bayesian = dict()
        precision_bayesian = dict()
        for key in range(6):
            recall_bayesian[key] = true_positives[key] / (true_positives[key] + false_negatives[key])
            precision_bayesian[key] = true_positives[key] / (true_positives[key] + false_positives[key])

            plt.scatter(alphas[j], recall_bayesian[key], color=colors[key], label=f"class: {key}")

        print("recall:", recall_bayesian)
        print("precision:", precision_bayesian)

    plt.xlabel("Alpha")
    plt.ylabel("Recall")
    plt.show()
