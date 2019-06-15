import numpy as np
import glob


class Net:
    class Layer:
        def __init__(self, size, function, bias=0., weight=[]):
            self.bias = bias
            self.weight = weight
            self.size = size
            if self.weight == []:
                for output in range(size[1]):
                    self.weight.append(np.random.random(size[0])*10-5)
            self.function = function

    def __init__(self, layers):
        self.layers = layers
        self.value = []

    def predict(self, X):
        self.value = []
        for layer in self.layers:
            X = layer.function(self, X, layer)
            self.value.append(X)
        return X

    def sumator(self, X, W):
        suma = []
        suma_V = 0
        for out_neuron in range(len(W)):
            for wage_or_value in range(len(X)):
                suma_V += X[wage_or_value] * W[out_neuron][wage_or_value]
            suma.append(suma_V)
            suma_V = 0
        return suma

    def unipolar_descrete(self, x, layer):
        net = np.array(self.sumator(x, layer.weight)) + layer.bias
        for n in range(len(net)):
            if net[n] >= 0:
                net[n] = 1
            else:
                net[n] = 0

        x = net
        return x

    def bipolar_descrete(self, x, layer):
        net = np.array(self.sumator(x, layer.weight)) + layer.bias
        for n in range(len(net)):
            if net[n] >= 0:
                net[n] = 1
            else:
                net[n] = -1

        x = net
        return x

    def unipolar_continuous(self, x, layer):
        #1/(1+e^(-x))

        net = np.array(self.sumator(x, layer.weight)) + layer.bias
        for n in range(len(net)):
            net[n] = 1/(1+np.exp(-net[n]))

        x = net
        return x

    def bipolar_continuous(self, x, layer):
        #2/(1+e^(-2x))-1

        net = np.array(self.sumator(x, layer.weight)) + layer.bias
        for n in range(len(net)):
            net[n] = 2 / (1 + np.exp(-2*net[n])) - 1

        x = net
        return x

    def test(self, tests):
        results = []
        for sample in tests:
            target = sample[-1]
            y = self.predict(sample[0:-1])
            results.append(np.power(target - y, 2))
        results = np.array(results)
        return results

    def print_net(self):
        for layer in self.layers:
            print(layer.weight, ' ', layer.bias)


class GeneticAlgorithm:
    def __init__(self, population, X, num_elite, tournament_population_size, mutation_rate, max_gen):
        self.num_elite = num_elite
        self.population = population
        self.tests = X
        self.gen_num = 0
        self.mutation_rate = mutation_rate
        self.net_size = []
        self.functions = []
        self.sizes = []
        self.max_gen = max_gen
        for layer in self.population[0].layers:
            self.net_size.append(layer.size[1] * layer.size[0] + 1)
            self.sizes.append(layer.size)
            self.functions.append(layer.function)
        self.tournament_population_size = tournament_population_size

    def predict(self, results):
        p = []
        for result in results:
            if result<0.5:
                p.append(0)
            else:
                p.append(1)
        p = np.array(p)
        return p.mean()

    def train(self):
        t = 0
        while self.predict(self.population[0].test(self.tests)) != 0 and t < self.max_gen:
            self.population = self.evolve(self.population)
            self.population.sort(key=lambda x: self.predict(x.test(tests)), reverse=False)
            self.gen_num += 1
            if t % 10 == 0:
                print(t)
            t+=1

            # print(self.gen_num)
            # for net in self.population:
            #     net.print_net()
            #     print(net.test(tests))

    def evolve(self, population):
        return self.mutate_population(self.crossover_population(population))

    def crossover_population(self, population):
        crossover_pop = set_population(0, self.sizes, self.functions)
        for i in range(self.num_elite):
            crossover_pop.append(population[i])
        i = self.num_elite
        while i < len(self.population):
            net1 = self.select_tournament_population(population)[0]
            net2 = self.select_tournament_population(population)[0]
            crossover_pop.append(self.crossover_net(net1, net2))
            i += 1
        return crossover_pop

    def mutate_population(self, population):
        for i in range(self.num_elite, len(self.population)):
            self.mutate_net(population[i])
        return population

    def crossover_net(self, net1, net2):
        crossover_n = self.create_net()
        j = -1
        for layer in self.population[0].layers:
            j += 1
            for i in range(layer.size[1]):
                for k in range(layer.size[0]):
                    if np.random.random() < self.mutation_rate:
                        if np.random.random() < 0.5:
                            crossover_n.layers[j].weight[i][k] = net1.layers[j].weight[i][k]
                        else:
                            crossover_n.layers[j].weight[i][k] = net2.layers[j].weight[i][k]
            if np.random.random() < self.mutation_rate:
                if np.random.random() < 0.5:
                    crossover_n.layers[j].bias = net1.layers[j].bias
                else:
                    crossover_n.layers[j].bias = net2.layers[j].bias
        return crossover_n

    def mutate_net(self, net):
        j = -1
        for layer in self.population[0].layers:
            j += 1
            for i in range(layer.size[1]):
                for k in range(layer.size[0]):
                    if np.random.random() < self.mutation_rate:
                        net.layers[j].weight[i][k] = np.random.random() * 10 - 5
            if np.random.random() < self.mutation_rate:
                net.layers[j].bias = np.random.random() * 10 - 5

    def select_tournament_population(self, population):
        j = -1
        for layer in self.population[0].layers:
            j += 1
            tournament_pop = set_population(0, self.population[0].layers[j].size, self.population[0].layers[j].function)
            i = 0
            while i < self.tournament_population_size:
                tournament_pop.append(population[np.random.randint(0, len(self.population))])
                i += 1
            tournament_pop.sort(key=lambda x: self.predict(x.test(tests)), reverse=False)
        return tournament_pop

    def create_net(self):
        j = -1
        layers = []
        for layer in self.population[0].layers:
            j += 1
            weight = []
            for output in range(self.population[0].layers[j].size[1]):
                weight.append(np.random.random(self.population[0].layers[j].size[0]) * 10 - 5)
            layers.append(Net.Layer(self.population[0].layers[j].size, self.population[0].layers[j].function, np.random.random() * 10 - 5, weight=weight))
        return Net(layers)


def set_population(num_pop, size, function):
    population = []
    for num in range(num_pop):
        layers = []
        i = 0
        for layer in size:
            weight = []
            for output in range(layer[1]):
                weight.append(np.random.random(layer[0]) * 10 - 5)
            layers.append(Net.Layer(size[i], function[i], np.random.random()*10-5, weight))
            i += 1
        population.append(Net(layers))
    return population


def file_convert_to_test(open_file):
    test = []
    content = []
    while content != [0]:
        content = open_file.readline()
        content = content.split(',')
        for input in range(len(content)-1):
            content[input] = float(content[input])
        if content[-1] == 'Iris-virginica\n':
            content[-1] = 1
        else:
            content[-1] = 0

        test.append(content)
    test.pop()
    return test


if __name__ == '__main__':
    population = set_population(10, [[4, 1]], [Net.unipolar_continuous])

    # files = glob.glob("tests\*.txt")
    #
    # for path in files:
    path = "tests\\iris.data"
    file = open(path, "r")
    tests = file_convert_to_test(file)
    train_method = GeneticAlgorithm(population, tests, 1, 4, 0.1, 100)
    train_method.train()
    population = train_method.population
    print('generation number = ', train_method.gen_num, 'file: ', path)
    for net in population:
        net.print_net()
        print(train_method.predict(net.test(tests)))

    file.close()

    path = "tests\\iris.data"
    file = open(path, "r")
    tests = file_convert_to_test(file)
    res = population[0].test(tests)
    file.close()




