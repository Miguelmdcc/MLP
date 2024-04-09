import numpy as np

#Padrões
x = np.array([[1,0.5,-1],
              [0,0.5,1],
              [1,-0.5,-1]])

#Alfa(taxa de aprendizagem)
alfa  = 0.01

#Quantidade de neuronios na camada intermediaria
neuronios = 2

#Primeira camada
#Pesos sinapticos aleatorios entre -0,5 e 0,5
# v = np.random.uniform(-0.5, 0.5, size=(x.shape[0],neuronios))
# print(v)
#Pesos do exemplo dos slides
v = np.array([[0.12, -0.03],
              [-0.04, 0.15],
              [0.31, -0.41]])

#Bias aleatorios entre -0.5 e 0.5
# v0 = np.random.uniform(-0.5, 0.5, size=neuronios)
# print(v0)
v0 = np.array([-0.09,0.18])

#Segunda camada
#Pesos sinapticos aleatorios entre -0,5 e 0,5
# w = np.random.uniform(-0.5, 0.5, size=(neuronios,targ.shape[0]))
# print(w)
#Pesos do exemplo dos slides
w = np.array([[-0.05, -0.34,0.21],
              [0.19,-0.09,0.26]])

#Bias aleatorios entre -0.5 e 0.5
# w0 = np.random.uniform(-0.5, 0.5, size=targ.shape[0])
# print(w0)
w0 = np.array([0.18,-0.27,-0.12])

#Vamos usar a função de ativação tangente hiperbolica
# np.tanh()

#definindo targets em one of classes
targ = np.array([[1,-1,-1],
                [-1,1,-1],
                [-1,-1,1]])

#calculo de zin(1x2) = x(3x3)*v(3x2) + v0(1x2)
zin = 0
y = 0
z = 0
errototal = 1
quadratico = 0
deltak = 0
deltaw = 0
deltaw0 = 0
deltain = 0
delta = 0
deltav = 0
deltav0 = 0
ciclo = 0
errotolerado = 0.01

while(errototal > errotolerado):
    errototal = 0
    for i in range(x.shape[0]):
        # Cálculo da primeira camada
        zin = np.dot(x[i], v) + v0
        z = np.tanh(zin)
        
        # Cálculo da segunda camada
        yin = np.dot(z, w) + w0
        y = np.tanh(yin)
        
        quadratico = np.sum((targ[i] - y) ** 2)
        errototal += 0.5 * quadratico
        
        # Adquirindo a expressão para correção dos pesos da segunda camada
        deltak = (targ[:][i] - y) * (1 - np.square(np.tanh(yin)))
        # Correção dos pesos sinápticos da segunda camada
        deltaw = alfa * np.outer(deltak, z)
        deltaw0 = alfa * deltak
        
        # Propagando o erro de volta para a primeira camada
        deltain = np.dot(deltak, np.transpose(w))

        # Adquirindo a expressão para correção dos pesos da primeira camada
        delta = deltain * (1 - np.square(np.tanh(zin)))
        #Correção dos pesos sinápticos da primeira camada
        deltav = alfa*(np.outer(delta,x[i]))
        deltav0 = alfa*delta

        # Atualização dos pesos da camada de saída
        w += np.transpose(deltaw)
        w0 += np.transpose(deltaw0)
        # Atualização dos pesos da camada intermediária
        v += np.transpose(deltav)
        v0 += np.transpose(deltav0)

    ciclo += 1
    print("Ciclo:", ciclo, "Erro total:", errototal)

# Novos padrões de entrada para teste
novos_padroes = np.array([[1, 0.5, -1],
                          [1, 0, -1]])

# Realizar a propagação direta para obter as previsões
for novo_padrao in x:
    # Cálculo da primeira camada
    zin = np.dot(novo_padrao, v) + v0
    z = np.tanh(zin)

    # Cálculo da segunda camada
    yin = np.dot(z, w) + w0
    y = np.tanh(yin)
    
    print("Padrão de entrada:", novo_padrao)
    print("Saída prevista:", y)