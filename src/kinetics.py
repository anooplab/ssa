import yaml
import matplotlib.pyplot as plt
import argparse
import math
import numpy as np
import csv
import pandas as pd
import datetime
from pprint import pprint
import sys
import os


class SSA:
    """
    Stochastic Simulation Algorithm Class. time and time n_steps are
    set to 0.0 and 0.0001 respectively. Change it according to your
    requirement.
    """

    t = 0.0
    dt = 0.0001
    time_now = str(datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    output_csv = f"output_{time_now}.csv"

    def __init__(
            self, population, rate_constant, stoichiometry_matrix, n_steps,
            species
    ):
        self.population = population
        self.k = rate_constant
        self.stoichiometry_matrix = stoichiometry_matrix
        self.steps = n_steps
        self.species = species

    def propensity(self, rxn_id):
        reactant_index = []
        reactant_stoichiometry = []
        for i, stoichiometry in enumerate(self.stoichiometry_matrix[rxn_id]):
            if stoichiometry < 0:
                reactant_index.append(i)
                reactant_stoichiometry.append(stoichiometry)
        order = - sum(reactant_stoichiometry)
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
        block = int(self.steps / 10)
        for step_number in range(1, self.steps + 1):
            a = [self.propensity(j) for j in range(len(self.k))]
            a0 = sum(a)
            if a0 == 0:
                print(a)
                print("propensity sum can not be zero")
                break
            r1, r2 = np.random.random_sample(2)
            tau = np.log(1.0 / r1) / a0
            if tau < 0:
                raise ValueError("tau can not be zero")
            s = 0
            for i, _ in enumerate(self.stoichiometry_matrix):
                s += a[i]
                if s > r2 * a0:
                    self.population += self.stoichiometry_matrix[i]
                    break
            self.t += tau

#           tmp_lst = [str(self.t)]
#           for length in range(len(self.population)):
#               self.population[length] = max(self.population[length], 0)
#               tmp_lst.append(str(self.population[length]))
#           data_file.append(tmp_lst)

            tmp_lst = [str(self.t)] +  [str(c) for c in self.population]
            # Dumping data in each loop
            if step_number % block == 0:
                with open('CheckPoint.txt', 'w') as chk_file:
                    print(f'Dumping data to CheckPoint File after '
                          f'{step_number} steps')
                    chk_file.writelines(str(data_file) + '\n')

        # Create Final CSV from tmp file

        with open(self.output_csv, 'w') as fp:
            tmp_data = yaml.safe_load(
                open('CheckPoint.txt', 'r').readlines()[-1])
            wr = csv.writer(fp)
            wr.writerow(header_line)
            for j in tmp_data:
                wr.writerow(j)
        print('Removing the CheckPoint File')
        os.remove('CheckPoint.txt')
        print("Final population is ", self.population)
        return self.population, self.species


def e_act_to_rate(e_act, temperature):
    """
    convert activation energy from kcal/mol ---> rate
    """
    e_act_joules = e_act * 4184
    k_b = 1.3806503 * pow(10, -23)
    gas_constant = 8.31446
    planks_constant = 6.62607015 * pow(10, -34)
    res = math.exp((-e_act_joules) / (gas_constant * temperature))
    res *= temperature * k_b / planks_constant
    return res


def parse_data(yml_file):
    with open(yml_file, "r") as ssa_conf:
        data = yaml.safe_load(ssa_conf)
    the_temperature = data.get("Temp")
    number_of_steps = data.get("Steps")
    initial_setup = data.get("Initial_pop")
    stoichiometry_matrix = data.get("Stoichiometry")
    name_of_the_species = []
    population_at_start = []
    gibbs_energies = []
    the_state_vector = []
    for i in initial_setup:
        name_of_the_species.append(i)
        population_at_start.append((initial_setup[i]))

    for i in stoichiometry_matrix:
        gibbs_energies.append(i[0][0])
        the_state_vector.append(i[1])

    return (
        the_temperature,
        number_of_steps,
        name_of_the_species,
        population_at_start,
        gibbs_energies,
        the_state_vector,
    )


def plotter(csv_file):
    df = pd.read_csv(csv_file, index_col=0, dtype=float)
    for i in df.columns:
        plt.plot(df.index, df[i], label=i)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Population")
    # plt.savefig("output.jpg")
    plt.show()


def analyze(csv_file):
    df = pd.read_csv(csv_file, index_col=0, dtype=float)
    last = df.tail(1).T
    last.columns = ['Population']
    last['percentage'] = 100*last/last.Population.sum()
    pprint(last)
    last.plot.pie(y='Population').get_figure().savefig('output.png')


def calculate_percentage(final_population, species_name):
    sum_of_population = sum(final_population)
    percent_lst = []
    for i in final_population:
        percent_lst.append(str(i*100/sum_of_population)+' %')
    percent_dict = dict(zip(species_name, percent_lst))
    print('Final Population percentage of each species')
    pprint(percent_dict)


def merge_csvs(csv1, csv2):
    time_now = str(datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    output_csv = f"combined_{time_now}.csv"
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    df2['time'] = df2['time'] + df1['time'].iloc[-1]
    frame = [df1, df2]
    result = pd.concat(frame)
    result.set_index('time', inplace=True)
    result.to_csv(output_csv, index=True)
    print(f'Merged two csv {csv1} and {csv2} to {output_csv}')


def regenerate_yaml(yaml_file, output_csv):
    time_now = str(datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    restart_yaml = f"restart_conf_{time_now}.yaml"
    with open(yaml_file, "r") as ssa_conf:
        data = yaml.safe_load(ssa_conf)
    keys_to_extract_from_old_csv = ["Temp", "Steps"]
    old_data_dict = {key: data[key] for key in keys_to_extract_from_old_csv}
    old_run_final_population_dict = data["Initial_pop"]
    with open(output_csv, "r") as old_csv:
        last_line = old_csv.readlines()[-1]
    old_final_pop_lst = [int(i) for i in last_line.strip().split(",")[1:]]
    species_name = list(old_run_final_population_dict)
    new_init_pop = dict(zip(species_name, old_final_pop_lst))
    old_data_dict["Initial_pop"] = new_init_pop
    with open(restart_yaml, "w") as new_yaml_file:
        yaml.dump(
            old_data_dict, new_yaml_file, default_flow_style=False,
            sort_keys=False
        )
        new_yaml_file.writelines('Stoichiometry:' + '\n')
        for i in data['Stoichiometry']:
            new_yaml_file.writelines('  - ' + str(i) + '\n')


def get_sample_yaml(sample_conf_file):
    sample_file = """
# Sample configuration file for SSA code. Created from script
Temp: 273  # Set temperature in kelvin
Steps: 100000  # No of monte carlo n_steps
Initial_pop:
    A0: 1000
    A1: 4000
    A2: 0
    A5: 0
    A6: 0
    A10: 0
    A12: 0
    A15: 0
    A17: 0
    A18: 0

# [[Gibbs free energy], [stoichiometry of the elementary reaction]]
Stoichiometry:  
    - [[7.52], [-1, -1, 1, 0, 0, 0, 0, 0, 0, 0]]
    - [[10.11], [1, 1, -1, 0, 0, 0, 0, 0, 0, 0]]
    - [[5.34], [0, -1, -1, 1, 0, 0, 0, 0, 0, 0]]
    - [[16.73], [0, 1, 1, -1, 0, 0, 0, 0, 0, 0]]
    - [[4.82], [0, -1, -1, 0, 1, 0, 0, 0, 0, 0]]
    - [[17.47], [0, 1, 1, 0, -1, 0, 0, 0, 0, 0]]
    - [[17.40], [0, -1, 0, -1, 0, 1, 0, 0, 0, 0]]
    - [[4.39], [0, 1, 0, 1, 0, -1, 0, 0, 0, 0]]
    - [[17.49], [0, -1, 0, 0, -1, 0, 1, 0, 0, 0]]
    - [[2.29], [0, 1, 0, 0, 1, 0, -1, 0, 0, 0]]
    - [[11.04], [0, -1, 0, 0, 0, -1, 0, 1, 0, 0]]
    - [[11.96], [0, 1, 0, 0, 0, 1, 0, -1, 0, 0]]
    - [[0.27], [0, -1, 0, 0, 0, 0, -1, 0, 1, 0]]
    - [[9.83], [0, 1, 0, 0, 0, 0, 1, 0, -1, 0]]
    - [[7.00], [0, -1, 0, 0, 0, 0, -1, 0, 0, 1]]
    - [[17.06], [0, 1, 0, 0, 0, 0, 1, 0, 0, -1]]
    """
    with open(sample_conf_file, "w") as sample_yaml_file:
        sample_yaml_file.writelines(sample_file)


def header():
    print(
        r""" 
================================================================================
                               ____ ____    _    
                              / ___/ ___|  / \   
                              \___ \___ \ / _ \  
                               ___) |__) / ___ \ 
                              |____/____/_/   \_\ 
 

               --- Gillespie Stochastic Simulation Algorithm ---
                     http://www.chemistry.iitkgp.ac.in/~anoop/
                              License: GNU GPL V3
================================================================================

With contribution from:
    Saikat Roy
    Debankur Bhattacharyya
    Anakuthil Anoop
"""
    )


def run_ssa():
    epilog_msg = """
For quick start: 

    $ python kinetics.py --sample_yaml conf.yaml 

Run after editing conf.yaml to suit your system: 

    $ python kinetics.py -s -f conf.yaml
    Output is in csv format containing time evolution of the population.

Plot the output populations:

    $ python kinetics.py -p output.csv

To continue the simulation from the last point, generate a new yaml file:
    $ python kinetics.py -r output.csv -f continue.yaml

To merge outfile from two simulations:
    $ python kinetics.py --merge output_1.csv output_2.csv

Enjoy!
"""
    parser = argparse.ArgumentParser(
        description="Gillespie stochastic simulation code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog_msg
    )
    parser.add_argument(
        "-f",
        "--yaml_file",
        type=str,
        required=False,
        help="Simulation configuration file",
    )
    parser.add_argument(
        "-s",
        "--ssa",
        action="store_true",
        help="Run Gillespie's Stochastic Simulation Algorithm"
    )
    parser.add_argument(
        "-p",
        "--plot",
        metavar="csv_file",
        required=False,
        type=str,
        help="Plot all the population after simulation from the csv file"
    )
    parser.add_argument(
        "-a",
        "--analyze",
        metavar="csv_file",
        required=False,
        type=str,
        help="Analyze the population from the csv file"
    )
    parser.add_argument(
        "-r", "--restart",
        metavar="csv_file",
        type=str, required=False, help="provide the \
            csv file from the previous run"
    )
    parser.add_argument(
        "--sample_yaml",
        type=str,
        required=False,
        help="This will generate a sample configuration yaml file for this "
             "simulation "
    )
    parser.add_argument(
        "-m", '--merge',
        metavar=('output_1.csv', 'output_2.csv'),
        type=str, required=False, nargs=2,
        help="Merge two csv file, Order Matters!!"
    )
    args = parser.parse_args()
    if args.sample_yaml:
        print("Generating a sample yaml configuration file for the simulation")
        get_sample_yaml(args.sample_yaml)
        sys.exit()
    if args.restart:
        if not args.yaml_file:
            sys.exit(
                "yaml_file of the previous run is required. Provide with -f "
                "argument")
        print("-" * 80)
        print("!!! ATTENTION !!!")
        print("Generating a new yaml configuration file from the previous run")
        print("Check the restart file carefully before running")
        print("-" * 80)
        regenerate_yaml(args.yaml_file, args.restart)
        sys.exit()
    if args.merge:
        merge_csvs(args.merge[0], args.merge[1])
    if args.ssa:
        header()
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
        start_time = datetime.datetime.now()
        print(f'{"-" * 19} Starting Gillespie Stochastic Simulation {"-" * 20}')
        pprint(f"species names are {species_name}")
        print("Initial Population: ")
        pprint(initial_population)
        print("-" * 80)
        print("Stoichiometric Matrix: ")
        pprint(state_vector)
        print("-" * 80)
        print("Gibbs Free Energy of Activations: ")
        pprint(gibbs_lst)
        print("-" * 80)
        print(f"Temperature is set to {temperature}K ")
        print(f"Rate Constant at {temperature}K : ")
        pprint(k)
        print("-" * 80)
        print(f"Number of Monte carlo n_steps: {steps}")

        ssa_obj = SSA(
            np.array(initial_population),
            np.array(k),
            np.array(state_vector),
            steps,
            species_name,
        )
        final_population, species = ssa_obj.gillespie()
        calculate_percentage(final_population, species)
        end_time = datetime.datetime.now()
        print(
            f'{"-" * 29} Finished Simulation in '
            f'{end_time - start_time} {"-" * 20}')
    if args.analyze:
        analyze(args.analyze)
    if args.plot:
        plotter(args.plot)


if __name__ == "__main__":
    run_ssa()
    print("Done")
    # x = e_act_to_rate(16, 273)
    # print(x)
