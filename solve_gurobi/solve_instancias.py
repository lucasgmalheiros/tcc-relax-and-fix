from functions_model_sy import lista_instancias, model_sy
import csv

# Arquivo de salvamento
results_csv = 'resultados1.csv'
# Pasta com instâncias a serem resolvidas
folder_instancias = '../instancias/maria_desiree/'
# Instâncias
instancias = lista_instancias(folder_instancias, results_csv)

for file in instancias:
    model = model_sy(folder_instancias, file)
    m, I, J, T = model[0], model[1], model[2], model[3]
    
    # Resolução
    # Critérios de parada
    m.Params.timelimit = 10  # Tempo (s) (default = inf)
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
    if status not in [11]:  # Não escreve caso seja interrompido pelo usuário 
        # Escrever ou adicionar resultados no CSV
        with open(results_csv, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Escrever o cabeçalho apenas na primeira vez
            if file.tell() == 0:  # Se o arquivo estiver vazio, escrever o cabeçalho
                writer.writerow(['Instancia', 'Plantas', 'Produtos', 'Periodos', 
                                'Lower Bound (LB)', 'Upper Bound (UB)', 'Gap', 
                                'Tempo (s)', 'Status'])

            # Adicionar os resultados da instância ao CSV
            writer.writerow([inst, plantas, produtos, periodos, lb, ub, gap, time, status])
