import yaml
import matplotlib.pyplot as plt
import argparse
import math
import numpy as np
import csv
import csv as pd


class SSA:
    """
    Stochastic Simulation Algorithm Class. time and time steps are set to 0.0
    and 0.0001 respectively. Change it according to your requirement.
    """

    t = 0.0
    dt = 0.0001

    def __init__(self, population, k, stoichiometry_matrix, steps, species):
        self.population = population
        self.k = k
        self.stoichiometry_matrix = stoichiometry_matrix
        self.steps = steps
        self.species = species

    def propensity(self, rxn_id):
        reactant_index = []
        reactant_stoichiometry = []
        for j in range(len(self.population)):
            if self.stoichiometry_matrix[rxn_id][j] < 0:
                reactant_index.append(j)
                reactant_stoichiometry.append(self.stoichiometry_matrix[rxn_id][j])
        order = -sum(reactant_stoichiometry)
        if order == 1:
            return self.k[rxn_id] * self.population[reactant_index[0]] * self.dt
        elif order == 2 and len(reactant_index) == 1:
            return (
                    0.5
                    * self.k[rxn_id]
                    * self.population[reactant_index[0]]
                    * (self.population[reactant_index[0]] - 1)
                    * self.dt
            )
        elif order == 2 and len(reactant_index) == 2:
            return (
                    self.k[rxn_id]
                    * self.population[reactant_index[0]]
                    * self.population[reactant_index[1]]
                    * self.dt
            )
        elif order == 3 and len(reactant_index) == 2:
            return (
                    0.5
                    * 0.5
                    * self.k[rxn_id]
                    * self.population[reactant_index[0]]
                    * self.population[reactant_index[1]]
                    * self.population[reactant_index[1]]
                    * self.dt
            )
        else:
            print(f"Propensity of the reaction {rxn_id} is unknown")
            return None

    def gillespie(self):
        header_line = ["time", *self.species]
        data_file = []
        for _ in range(self.steps):
            a = [self.propensity(j) for j in range(len(self.k))]
            a0 = sum(a)
            if a0 == 0:
                print(a)
                print("propensity sum can not be zero")
                break
            r1 = np.random.random_sample()
            r2 = np.random.random_sample()
            tau = (1.0 / a0) * (np.log(1.0 / r1))
            if tau < 0:
                raise ValueError("tau can not be zero")
            s = 0
            for l in range(len(self.stoichiometry_matrix)):
                s += a[l]
                if s > r2 * a0:
                    j = l
                    break
            self.t = self.t + tau
            self.population = self.population + self.stoichiometry_matrix[j]
            tmp_lst = [str(self.t)]
            for length in range(len(self.population)):
                self.population[length] = max(self.population[length], 0)
                tmp_lst.append(str(self.population[length]))
            data_file.append(tmp_lst)

        with open("output.csv", "w") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(header_line)
            for i in data_file:
                writer.writerow(i)
        print("Final population is ", self.population)
        return self.population


def e_act_to_rate(e_act, temp):
    """
    convert activation energy from kcal/mol ---> rate
    """
    e_act_joules = e_act * 4184
    k_B = 1.3806503 * pow(10, -23)
    R = 8.31446
    h = 6.62607015 * pow(10, -34)
    res = (-1 * e_act_joules) / (R * temp)
    res = math.exp(res)
    res *= k_B / h
    res *= temp
    return res


def parse_data(yml_file):
    with open(yml_file, "r") as ssa_conf:
        data = yaml.safe_load(ssa_conf)
    temperature = data.get("Temp")
    steps = data.get("Steps")
    initial_setup = data.get("Initial_pop")
    stoichiometry_matrix = data.get("Stoichiometry")
    species_name = []
    initial_population = []
    gibbs_lst = []
    state_vector = []
    for i in initial_setup:
        species_name.append(i)
        initial_population.append((initial_setup[i]))

    for i in stoichiometry_matrix:
        gibbs_lst.append(i[0][0])
        state_vector.append(i[1])

    return temperature, steps, species_name, initial_population, gibbs_lst, state_vector


def plotter(csv_file):
    df = pd.read_csv(csv_file, index_col=0, dtype=float)
    for i in df.columns:
        plt.plot(df.index, df[i], label=i)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.savefig("output.jpg")
    plt.show()


def header():
    print(''' 
================================================================================
                               ____ ____    _    
                              / ___/ ___|  / \   
                              \___ \___ \ / _ \  
                               ___) |__) / ___ \ 
                              |____/____/_/   \_\
 

               ---   Gillespie Stochastic Simulation Algorithm ---
                     http://www.chemistry.iitkgp.ac.in/~anoop/
                              License: GNU GPL V3
================================================================================

With contribution from:
    Saikat Roy
    Debankur Bhattacharyya
    Prof. Anakuthil Anoop
''')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gillespie stochastic simulation code")
    parser.add_argument(
        "-f",
        "--yaml_file",
        type=str,
        required=True,
        help="Simulation configuration file",
    )
    args = parser.parse_args()
    yaml_conf = args.yaml_file
    (
        temperature,
        steps,
        species_name,
        initial_population,
        gibbs_lst,
        state_vector,
    ) = parse_data(yaml_conf)
    k = [e_act_to_rate(i, temperature) for i in gibbs_lst]
    header()
    print(f'{"-" * 19} Starting Gillespie Stochastic Simulation {"-" * 20}')
    print(f"species names are {species_name}")
    print("Initial Population: ")
    print(initial_population)
    print("-" * 80)
    print("Stoichiometric Matrix: ")
    print(state_vector)
    print("-" * 80)
    print("Gibbs Free Energy of Activations: ")
    print(gibbs_lst)
    print("-" * 80)
    print(f"Temperature is set to {temperature}K ")
    print(f"Rate Constant at {temperature}K : ")
    print(k)
    print("-" * 80)
    print(f"Number of Monte carlo steps: {steps}")
    ssa_obj = SSA(
        np.array(initial_population),
        np.array(k),
        np.array(state_vector),
        steps,
        species_name,
    )
    ssa_obj.gillespie()
    plotter("output.csv")
    print(f'{"-" * 29} Finishing Simulation {"-" * 30}')
