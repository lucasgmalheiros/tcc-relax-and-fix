import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
from natsort import natsorted
import csv


def read_dat_file(file_path):
    """"Função para a leitura das instâncias geradas"""
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

    return {"items": items,
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


def lista_instancias(folder_path, results_file):
    """Verifica instancias a serem resolvidas"""
    # Lê a primeira coluna do CSV de resultados para ver instâncias já resolvidas
    try:
        instancias_resolvidas = [row[0] for row in csv.reader(open(results_file))][1:]  # Exclui o cabeçalho
    except IndexError:
        instancias_resolvidas = []
    # Lista das instâncias na pasta que ainda não foram resolvidas
    instancias = natsorted([f for f in os.listdir(folder_path) if f.endswith('.dat') and f.replace('.dat', '') not in instancias_resolvidas])
    return instancias


def model_sy_relaxation(folder_path, instancia):
    """"Modelo como proposto por Sambasivan & Yahya (2005) preparado para resolução com relax-and-fix"""
    # Dados da instância
    data = read_dat_file(folder_path + instancia)

    # Modelo
    m = gp.Model(instancia.replace('.dat', ''))

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
    # # Variável de setup (binária -> relaxada) (i, j, t)
    Z = m.addVars(I, J, T, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='Z')
    
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
    m.update()
    
    return m, I, J, T, Z


def main_solve_hybrid_matheuristic (folder_path, results_file, time_limit, neighboorhod_size, betta):
    for file in lista_instancias(folder_path, results_file):
        # Criação do modelo
        print(file)
        model = model_sy_relaxation(folder_path, file)
        m, I, J, T, Z = model[0], model[1], model[2], model[3], model[4]

        m.params.OutputFlag = 0  # Desativa a exibição de logs
        m.params.SoftMemLimit = 8  # Limite de 8GB de RAM
        m.Params.timelimit = time_limit / len(T)  # Tempo limite por subproblema (s)
        runtime = 0
        status_list = []

        # Auxiliares
        k = 1  # Iterações
        N = T[-1]  # Último período do horizonte de planejamento

        # Main loop
        while k <= N:
            # Variáveis na janela tornam-se binárias
            for i in I:
                for j in J:
                    Z[i, j, k - 1].setAttr('VType', GRB.BINARY)

            # Controle da restrição de local branching
            if k > 1:
                if k > 2:
                    m.remove(m.getConstrByName('local_branching'))
                # Adicionar a restrição Δ(x, x*) ≤ K
                delta_expr = sum(Z for Z in Z0) + sum(1 - Z for Z in Z1)
                m.addConstr(delta_expr <= neighboorhod_size, 'local_branching')

            # Resolução do subproblema
            m.optimize()
            runtime += m.Runtime
            status_list.append(m.Status)
            if m.Status == GRB.INFEASIBLE or m.SolCount == 0:
                break

            # Atualização dos auxiliares
            auxz0, auxz1 = [], []
            for i in I:
                for j in J:
                    for t in range(k):
                        if Z[i, j, t].VType == 'B':
                            if Z[i, j, t].X == 0:
                                auxz0.append(Z[i, j, t])
                            elif Z[i, j, t].X == 1:
                                auxz1.append(Z[i, j, t])
            Z0, Z1 = auxz0, auxz1
            k += 1
            
            # Fixação das soluções obtidas após betta períodos
            if k > betta:
                for i in I:
                    for j in J:
                        Z[i, j, k - 1 - betta].setAttr('LB', Z[i, j, k - 1 - betta].X)  # Lower bound fixado para a solução encontrada
                        Z[i, j, k - 1 - betta].setAttr('UB', Z[i, j, k - 1 - betta].X)  # Upper bound fixado para a solução encontrada

        # Última iteração
        if m.Status != GRB.INFEASIBLE and m.SolCount > 0:
            # Variáveis na janela tornam-se binárias
            for i in I:
                for j in J:
                    Z[i, j, k - 1].setAttr('VType', GRB.BINARY)
            m.remove(m.getConstrByName('local_branching'))
            # Adicionar a restrição Δ(x, x*) ≤ K
            delta_expr = sum(Z for Z in Z0) + sum(1 - Z for Z in Z1)
            m.addConstr(delta_expr <= neighboorhod_size, 'local_branching')
            # Última resolução
            m.optimize()
            runtime += m.Runtime
            status_list.append(m.Status)

        # Parâmetros armazenáveis da resolução
        inst = m.ModelName
        lb = round(m.ObjBound, 4)
        try:
            ub = round(m.ObjVal, 4)
        except AttributeError:
            ub = 'inf'
        gap = round(m.MIPgap, 4)
        time = round(runtime, 4)
        if 3 in status_list:
            status = 3
        elif m.SolCount == 0:
            status = 99
        else:
            status = max(status_list)

        # Verificação de status de resolução se resultado deve ser guardado:
        if status not in [11]:  # Não escreve caso seja interrompido pelo usuário 
            # Escrever ou adicionar resultados no CSV
            with open(results_file, mode='a', newline='') as file:
                writer = csv.writer(file)

                # Escrever o cabeçalho apenas na primeira vez
                if file.tell() == 0:  # Se o arquivo estiver vazio, escrever o cabeçalho
                    writer.writerow(['Instancia', 'K', 'Betta', 'Lower Bound (LB)',
                                        'Upper Bound (UB)', 'Gap', 'Tempo (s)', 'Status'])

                # Adicionar os resultados da instância ao CSV
                writer.writerow([inst, neighboorhod_size, betta, lb, ub, gap, time, status])
