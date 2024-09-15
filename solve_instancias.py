import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
from natsort import natsorted
import csv


# Leitura de instâncias
def read_dat_file(file_path):
    """"Função para a leitura das instâncias de Mariá"""
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
    production_data = []
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

    # 6. Lendo a matriz de demanda (12 linhas, 12 colunas)
    demand_matrix = []
    demand_start_line = inventory_costs_line + 1
    
    for i in range(periods):  # Lê a linha de demandas para o período
        demands = list(map(int, lines[demand_start_line + i].split()))
        period_demand = []  # Divide as demandas entre as plantas
        for j in range(num_plants):
            # Extraindo demandas por planta (assumindo que cada planta tem um número igual de itens)
            plant_demand = demands[j*items:(j+1)*items]
            period_demand.append(plant_demand)

        demand_matrix.append(period_demand)
    demand_matrix = np.array(demand_matrix)
    demand_matrix = np.transpose(demand_matrix, (2, 1, 0))

    # 7. Lendo os custos de transferência
    transfer_costs = []
    transfer_cost_line = demand_start_line + periods
    while transfer_cost_line < len(lines):
        line = lines[transfer_cost_line].strip()  # Verificar se a linha não está vazia antes de tentar ler
        if line:
            transfer_costs.append(float(line))
        transfer_cost_line += 1

    def create_transfer_cost_matrix(transfer_costs, num_plants):  # Criar a matriz de custos de transferência (simétrica)
        transfer_cost_matrix = np.zeros((num_plants, num_plants))  # Inicializar a matriz de zeros
        if len(transfer_costs) == 1:
            transfer_cost = transfer_costs[0]  # Se houver apenas um custo de transferência, aplicar para todos os pares de plantas
            for j in range(num_plants):
                for k in range(j + 1, num_plants):
                    transfer_cost_matrix[j, k] = transfer_cost
                    transfer_cost_matrix[k, j] = transfer_cost
        else:
            idx = 0  # Se houver múltiplos custos, aplicar entre pares de plantas
            for j in range(num_plants):
                for k in range(j + 1, num_plants):
                    transfer_cost_matrix[j, k] = transfer_costs[idx]
                    transfer_cost_matrix[k, j] = transfer_costs[idx]
                    idx += 1
        return transfer_cost_matrix

    transfer_costs = create_transfer_cost_matrix(transfer_costs, num_plants)

    return {"items": items,
            "periods": periods,
            "num_plants": num_plants,
            "capacities": capacities,
            "production_time": production_time,
            "setup_time": setup_time,
            "setup_cost": setup_cost,  
            "production_cost": production_cost,
            "inventory_costs": inventory_costs,
            "demand_matrix": demand_matrix,
            "transfer_costs": transfer_costs}


# Lê a primeira coluna do CSV de resultados para ver instâncias já resolvidas
try:
    instancias_resolvidas = [row[0] for row in csv.reader(open('results.csv'))][1:]  # Exclui o cabeçalho
except IndexError:
    instancias_resolvidas = []

folder_path = 'instancias/maria_desiree/'
# Instâncias a serem resolvidas
instancias = natsorted([f for f in os.listdir(folder_path) if f.endswith('.dat') and f.replace('.dat', '') not in instancias_resolvidas])

print(instancias)

for file in instancias:
    # Exemplo de uso
    data = read_dat_file(folder_path + file)
    
    # Modelo
    m = gp.Model(file.replace('.dat', ''))
    
    # Conjuntos
    # Produtos (i)
    I = np.array([_ for _ in range(data['items'])])
    # Plantas (j)
    J = np.array([_ for _ in range(data['num_plants'])])
    # Períodos (t)
    T = np.array([_ for _ in range(data['periods'])])
    
    # Parâmetros
    # Demanda (i, j, t)
    d = np.array(data['demand_matrix'])
    # Capacidade (j, t)
    cap = np.array(data['capacities'])
    # Tempo de produção (i, j)
    b = np.array(data['production_time'])
    # Tempo de setup (i, j)
    f = np.array(data['setup_time'])
    # Custo de produção (i, j)
    c = np.array(data['production_cost'])
    # Custo de setup (i, j)
    s = np.array(data['setup_cost'])
    # Custo de transporte (j, k)
    r = np.array(data['transfer_costs'])
    # Custo de estoque (i, j)
    h = np.array(data['inventory_costs'])
    
    # Variáveis de decisão
    # Quantidade produzida (i, j, t)
    X = m.addVars(I, J, T, vtype=GRB.CONTINUOUS, name='X')
    # Quantidade estocada (i, j, t)
    Q = m.addVars(I, J, T, vtype=GRB.CONTINUOUS, name='Q')
    # # Quantidade transportada (i, j, k(um outro j), t)
    W = m.addVars(I, J, J, T, vtype=GRB.CONTINUOUS, name='W')
    # # Variável de setup (binária) (i, j, t)
    Z = m.addVars(I, J, T, vtype=GRB.BINARY, name='Z')
    
    # Função objetivo
    expr_objetivo = sum(sum(sum(c[i, j] * X[i, j, t] + h[i, j] * Q[i, j, t] + s[i, j] * Z[i, j, t] + 
                            sum(r[j, k] * W[i, j, k, t] for k in J if k != j) for t in T) for j in J) for i in I)
    m.setObjective(expr_objetivo, sense=GRB.MINIMIZE)
    
    # Restrições
    # Balanço de estoque (revisar comportamento)
        # Período inicial
    m.addConstrs((Q[i, j, t] == X[i, j, t] - sum(W[i, j, k, t] for k in J if k != j) + sum(W[i, l, j, t] for l in J if l != j) - d[i, j, t] for i in I for j in J for t in T if t == 0),
                name='restricao_balanco_estoque')
        # Demais períodos
    m.addConstrs((Q[i, j, t] == Q[i, j, t-1] + X[i, j, t] - sum(W[i, j, k, t] for k in J if k != j) + sum(W[i, l, j, t] for l in J if l != j) - d[i, j, t] for i in I for j in J for t in T if t > 0),
                name='restricao_balanco_estoque')
    
    # Restrição que obriga setup (validar o range do r)
    m.addConstrs((X[i, j, t] <= min((cap[j, t] - f[i, j]) / b[i, j], sum(sum(d[i, k, r] for r in range(t, T[-1] + 1)) for k in J)) * Z[i, j, t] for i in I for j in J for t in T)
                , name='restricao_setup')
    
    # Restrição de capacidade
    m.addConstrs((sum(b[i, j] * X[i, j, t] + f[i, j] * Z[i, j, t] for i in I) <= cap[j, t] for j in J for t in T)
             , name='restricao_capacidade')
    
    # Resolução
    # Critérios de parada
    m.Params.timelimit = 3600  # Tempo (default = inf)
    # m.Params.MIPgap = 0.01  # Gap de otimalidade (default = 0.0001 = 0.01%)
    # Otimização
    m.optimize()
    
    # Parâmetros armazenáveis da resolução
    inst = m.ModelName
    plantas = len(J)
    produtos = len(I)
    periodos = len(T)
    lb = round(m.ObjBound, 2)
    ub = round(m.ObjVal, 2)
    gap = round(m.MIPgap, 2)
    time = round(m.Runtime, 2)
    status = m.Status

    # Verificação de status de resolução se resultado deve ser guardado:
    # if status not in [11]:  # Não escreve caso seja interrompido pelo usuário 
        # Escrever ou adicionar resultados no CSV
    with open('results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)

        # Escrever o cabeçalho apenas na primeira vez
        if file.tell() == 0:  # Se o arquivo estiver vazio, escrever o cabeçalho
            writer.writerow(['Instância', 'Plantas', 'Produtos', 'Períodos', 
                            'Lower Bound (LB)', 'Upper Bound (UB)', 'Gap', 
                            'Tempo (s)', 'Status'])

        # Adicionar os resultados da instância ao CSV
        writer.writerow([inst, plantas, produtos, periodos, lb, ub, gap, time, status])
