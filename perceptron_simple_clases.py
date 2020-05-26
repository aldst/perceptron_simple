import matplotlib
import random
from numpy import array
import matplotlib.pyplot as plt

class ConjuntoEntrenamiento:

    def __init__(self, x1, x2, esperado):
        self.x1 = x1
        self.x2 = x2
        self.esperado = esperado

class Etapa:

    def __init__(self, w1, w2, theta):
        self.w1 = w1
        self.w2 = w2
        self.theta = theta
        self.conj_esp = []

    def salida_red(self,conjunto):

        aux = self.w1 * conjunto.x1 + self.w2 * conjunto.x2 + self.theta
        
        if aux > 0:
            return 1
        else:
            return -1
    
    def agregar_conj_esp(self, conjuntos):

        for conjunto in conjuntos:
            self.conj_esp.append(conjunto)

    def modifica_pesos(self, fila, error):

        self.w1 = self.w1 + error * fila.x1
        self.w2 = self.w2 + error * fila.x2
        self.theta = self.theta + error


def algoritmo(conjuntos_entrenamiento):
    salida = [0,0,0,0]

    # se crea valores aletoria paraa los pesos y lumbral
    w1 = random.randint(0,1)
    w2 = random.randint(0,1)
    umbral = random.randint(0,10) / 10
    epoca = 0

    while encuentra_hiperplano(salida,conjuntos_entrenamiento):

        #selecciona un valor de x1,x2 y umbral       
        etapa = Etapa(w1,w2,umbral)
        etapa.agregar_conj_esp(conjuntos_entrenamiento)
        
        for conjunto in etapa.conj_esp: 
            # se calcula la red de salida y = f(x1w1 + w2x2 + ... + theta)          
            y = etapa.salida_red(conjunto)
            
            # si la clasificacion es incorrecta, se modifican los pesos y el umbral
            if y != conjunto.esperado:
                error = conjunto.esperado - y
                etapa.modifica_pesos(conjunto, error)

            salida[epoca] = y
            epoca+=1

        w1 = etapa.w1
        w2 = etapa.w2
        umbral = etapa.theta
        epoca = 0
    
        return w1,w2,umbral

def obtener_pesos(conjuntos_entrenamiento, col):
    pesos = array([0,0,0,0])
    i = 0
    for fila in conjuntos_entrenamiento:
        if col == 0:
            pesos[i] = fila.x1
        elif col == 1:
            pesos[i] = fila.x2
        i+=1

    return pesos

def graficar(conjuntos_entrenamiento, w1,w2,umbral):
    x = obtener_pesos(conjuntos_entrenamiento,0)
    
    y = obtener_pesos(conjuntos_entrenamiento,1)
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, 'bo')
    plt.axvline(x=0, ymin=-1, ymax=1)
    plt.axhline(y=0, xmin=-1, xmax=1)
    lx = [-1, 1]
    ly = []
    ly.append((-w1 * -1) / w2 - (umbral / w2))
    ly.append((-w1 * 1) / w2 - (umbral / w2))
    plt.plot(lx, ly, 'r')
    plt.show()

def encuentra_hiperplano(salida,conjuntos_entrenamiento):
    
    i= 0
    for con_entren in conjuntos_entrenamiento:
        if salida[i] != con_entren.esperado:
            return True
        i+=1
    return False

def inicializar_input():

    conjuntos_entrenamiento = []
         
    conjuntos_entrenamiento.append(ConjuntoEntrenamiento(-1,-1,-1))
    conjuntos_entrenamiento.append(ConjuntoEntrenamiento(1,-1,-1))
    conjuntos_entrenamiento.append(ConjuntoEntrenamiento(-1,1,-1))
    conjuntos_entrenamiento.append(ConjuntoEntrenamiento(1,1,1))

    return conjuntos_entrenamiento

def main():
    conjuntos_entrenamiento = inicializar_input()

    w1,w2,umbral = algoritmo(conjuntos_entrenamiento)
    graficar(conjuntos_entrenamiento, w1,w2,umbral)

if __name__ == "__main__":
    main()