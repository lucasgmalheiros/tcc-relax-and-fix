import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def read_instancia(file_path):
    """"Função para a leitura das instâncias geradas"""
    name = file_path.split('/')[-1].replace('.dat', '')
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 1. Lendo quantidade de itens e períodos
    items, periods = map(int, lines[0].split())

    # 2. Lendo número de plantas
    num_plants = int(lines[1].strip())

    # 3. Lendo capacidades das plantas
    capacities = [int(lines[i + 2].strip()) for i in range(num_plants)]
    capacities = np.tile(capacities, (periods, 1)).T  # Repete as capacidades ao longo dos períodos (deixar na forma j, t)

    # 4. Lendo a matriz de produção (tempo de produção, tempo de setup, custo de setup, custo de produção)
    production_problem_data = []
    start_line = 2 + num_plants
    production_time = np.zeros((items, num_plants))  # Inicializar listas para armazenar separadamente os tempos e custos
    setup_time = np.zeros((items, num_plants))
    setup_cost = np.zeros((items, num_plants))
    production_cost = np.zeros((items, num_plants))
    for i in range(num_plants * items):  # Preencher as matrizes com os dados lidos
        plant = i // items  # Determina a planta
        item = i % items    # Determina o item
        # Extrair os dados de cada linha
        prod_time, set_time, set_cost, prod_cost = map(float, lines[start_line + i].split())
        production_time[item, plant] = prod_time  # Preencher as respectivas matrizes
        setup_time[item, plant] = set_time
        setup_cost[item, plant] = set_cost
        production_cost[item, plant] = prod_cost

    # 5. Lendo os custos de inventário
    inventory_costs_line = start_line + num_plants * items
    inventory_costs = list(map(float, lines[inventory_costs_line].split()))  # Lê todos os valores de inventory_costs como uma única lista
    inventory_costs = np.array(inventory_costs).reshape(num_plants, -1)  # Divide a lista de custos de inventário por planta
    inventory_costs = inventory_costs.T  # Deixa na forma (i, j)

    # 6. Lendo a matriz de demanda (12 linhas)
    demand_matrix = []
    demand_start_line = inventory_costs_line + 1
    
    # Leitura inicial das demandas
    for i in range(periods):  # Lê as linhas de demandas para os períodos
        demands = list(map(int, lines[demand_start_line + i].split()))
        demand_matrix.append(demands)
    
    # Agora vamos dividir os valores de cada linha combinada entre as plantas
    final_demand_matrix = []
    for demands in demand_matrix:
        period_demand = []
        for j in range(num_plants):
            # Divide a demanda combinada por planta, assumindo que cada planta tem o mesmo número de itens
            plant_demand = demands[j*items:(j+1)*items]
            period_demand.append(plant_demand)
        final_demand_matrix.append(period_demand)
    
    # Transpor a matriz de demanda para o formato correto (itens, plantas, períodos)
    final_demand_matrix = np.array(final_demand_matrix)
    final_demand_matrix = np.transpose(final_demand_matrix, (2, 1, 0))  # Converte para o formato (itens, plantas, períodos)

    # 7. Reading transfer costs directly from the document as a matrix
    transfer_cost_matrix = []
    transfer_cost_line = demand_start_line + periods

    # Read the matrix of transfer costs line by line
    while transfer_cost_line < len(lines):
        line = lines[transfer_cost_line].strip()
        if line:
            # Split the line into individual cost values and convert them to float
            row = [float(value) for value in line.split()]
            transfer_cost_matrix.append(row)
        transfer_cost_line += 1

    # Convert to a numpy array (optional, if you want to work with numpy for matrix operations)
    transfer_costs = np.array(transfer_cost_matrix)

    return {"instance": name,
            "items": items,
            "periods": periods,
            "num_plants": num_plants,
            "capacities": capacities,
            "production_time": production_time,
            "setup_time": setup_time,
            "setup_cost": setup_cost,  
            "production_cost": production_cost,
            "inventory_costs": inventory_costs,
            "demand_matrix": final_demand_matrix,
            "transfer_costs": transfer_costs}


def extract_features(problem_data):
    # Extract structured data from the problem
    I = np.array([_ for _ in range(problem_data['items'])])  # Products
    J = np.array([_ for _ in range(problem_data['num_plants'])])  # Plants
    T = np.array([_ for _ in range(problem_data['periods'])])  # Periods
    d = np.array(problem_data['demand_matrix'])  # Demand (i, j, t)
    cap = np.array(problem_data['capacities'])  # Capacity (j, t)
    b = np.array(problem_data['production_time'])  # Production time (i, j)
    f = np.array(problem_data['setup_time'])  # Setup time (i, j)
    c = np.array(problem_data['production_cost'])  # Production cost (i, j)
    s = np.array(problem_data['setup_cost'])  # Setup cost (i, j)
    r = np.array(problem_data['transfer_costs'])  # Transportation costs (j, k)
    h = np.array(problem_data['inventory_costs'])  # Inventory costs (i, j)

    # Calculate instance features
    instance_features = {
        # Basic instance-level features
        'num_products': len(I),
        'num_plants': len(J),
        'num_periods': len(T),
        'binary_vars': len(I) * len(J) * len(T),

        # Demand statistics
        'total_demand': np.sum(d),
        'avg_demand_per_product': np.mean(d),
        'variance_demand_per_product': np.var(d),
        'std_demand_per_product': np.std(d),

        # Capacity and utilization statistics
        'mean_utilization': np.mean([
            np.sum(f[:, j] + b[:, j] * d[:, j, t]) / cap[j, t]
            for t in T for j in J
        ]),
        'max_utilization': np.max([
            np.sum(f[:, j] + b[:, j] * d[:, j, t]) / cap[j, t]
            for t in T for j in J
        ]),
        'min_utilization': np.min([
            np.sum(f[:, j] + b[:, j] * d[:, j, t]) / cap[j, t]
            for t in T for j in J
        ]),
        'std_utilization': np.std([
            np.sum(f[:, j] + b[:, j] * d[:, j, t]) / cap[j, t]
            for t in T for j in J
        ]),

        # Cost statistics
        'avg_setup_cost': np.mean(s),
        'variance_setup_cost': np.var(s),
        'std_setup_cost': np.std(s),
        'avg_production_cost': np.mean(c),
        'variance_production_cost': np.var(c),
        'std_production_cost': np.std(c),
        'total_transportation_cost': np.sum(r),
        'avg_inventory_cost': np.mean(h),
        'variance_inventory_cost': np.var(h),

        # Relationships between features
        'demand_to_capacity_ratio': np.sum(d) / np.sum(cap),
        'setup_to_production_cost_ratio': np.sum(s) / np.sum(c),
        'avg_demand_to_setup_cost_ratio': np.mean(d) / np.mean(s),
        'total_cost_to_demand_ratio': (np.sum(s) + np.sum(c)) / np.sum(d),

        # **Skewness and Kurtosis for More Features**
        'skew_demand': skew(d.flatten()),  # Skewness of the demand distribution
        'kurt_demand': kurtosis(d.flatten()),  # Kurtosis of the demand distribution
        'skew_setup_cost': skew(s.flatten()),  # Skewness of setup cost
        'kurt_setup_cost': kurtosis(s.flatten()),  # Kurtosis of setup cost
        'skew_production_cost': skew(c.flatten()),  # Skewness of production cost
        'kurt_production_cost': kurtosis(c.flatten()),  # Kurtosis of production cost
        'skew_utilization': skew([np.sum(f[:, j] + b[:, j] * d[:, j, t]) / cap[j, t] for t in T for j in J]),
        'kurt_utilization': kurtosis([np.sum(f[:, j] + b[:, j] * d[:, j, t]) / cap[j, t] for t in T for j in J]),

        # **Additional Features: Skewness and Kurtosis for other features**
        'skew_capacity': skew(cap.flatten()),  # Skewness of capacity
        'kurt_capacity': kurtosis(cap.flatten()),  # Kurtosis of capacity
        'skew_production_time': skew(b.flatten()),  # Skewness of production time
        'kurt_production_time': kurtosis(b.flatten()),  # Kurtosis of production time
        'skew_setup_time': skew(f.flatten()),  # Skewness of setup time
        'kurt_setup_time': kurtosis(f.flatten()),  # Kurtosis of setup time
        'skew_transportation_cost': skew(r.flatten()),  # Skewness of transportation costs
        'kurt_transportation_cost': kurtosis(r.flatten()),  # Kurtosis of transportation costs
        'skew_inventory_cost': skew(h.flatten()),  # Skewness of inventory costs
        'kurt_inventory_cost': kurtosis(h.flatten()),  # Kurtosis of inventory costs

        # **Percentiles and IQR**
        'p25_demand': np.percentile(d.flatten(), 25),
        'p50_demand': np.percentile(d.flatten(), 50),
        'p75_demand': np.percentile(d.flatten(), 75),
        'iqr_demand': np.percentile(d.flatten(), 75) - np.percentile(d.flatten(), 25),

        # **Coefficient of Variation (CV) for Demand**
        'cv_demand': np.std(d.flatten()) / np.mean(d.flatten()) if np.mean(d.flatten()) != 0 else 0,

        # **Ratios for setup and production costs**
        'setup_to_production_time_ratio': np.sum(f) / np.sum(b),
        'capacity_utilization_efficiency': np.sum(f) / np.sum(cap),

        # **Interactions between features**
        'demand_to_capacity_interaction': np.sum(d) * np.sum(cap),
        'demand_to_cost_interaction': np.sum(d) * (np.sum(s) + np.sum(c)),

        # **Additional statistics: time per unit of cost**
        'time_per_unit_of_cost': np.sum(b + f) / (np.sum(s) + np.sum(c)) if (np.sum(s) + np.sum(c)) != 0 else 0
    }

    return instance_features


# Create dataset from all instances in a folder
def create_dataset(instances_folder):
    dataset = []
    for file_name in os.listdir(instances_folder):
        if file_name.endswith('.dat'):
            # Read instance data
            problem_data = read_instancia(os.path.join(instances_folder, file_name))
            
            # Extract features
            features = extract_features(problem_data)
            
            # Add instance name ID
            features['instance'] = problem_data['instance']

            # Append to the dataset
            dataset.append(features)
    
    # Convert to a pandas DataFrame
    df = pd.DataFrame(dataset)
    return df


if __name__ == '__main__':
    # Example usage
    instances_folder = '../instancias/multi_plant_instances/'
    df = create_dataset(instances_folder)

    # Save the dataset to a CSV file
    df.to_csv('multi_plant_instance_features.csv', index=False)
