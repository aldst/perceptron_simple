import random
import math
import numpy as np
import matplotlib.pyplot as plt


class ConjuntoEntrenamiento:

    def __init__(self, x1, x2,x3, esperado):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.esperado = esperado

class Etapa:

    def __init__(self, conjunto):
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
        self.error = 0
        self.conjunto = conjunto

    def inicializar_pesos(self, w1, w2, w3):
      self.w1 = w1
      self.w2 = w2
      self.w3 = w3

    def salida_red(self):
        return round(self.w1 * self.conjunto.x1 + self.w2 * self.conjunto.x2 + self.w3 * self.conjunto.x3,3)
        
    def hallar_error(self, y):
      self.error = round(self.conjunto.esperado - y,3)

    def modifica_pesos(self,alpha):
        self.w1 = round( self.w1 + alpha * self.error * self.conjunto.x1 , 3 )
        self.w2 = round( self.w2 + alpha * self.error * self.conjunto.x2 , 3 )
        self.w3 = round( self.w3 + alpha * self.error * self.conjunto.x3 , 3 )

def algoritmo(conjuntos_entrenamiento):

    # se crea valores aletoria paraa los pesos y lumbral
    w1 = 0.84  #random.randint(0,100) / 100
    w2 = 0.394 #random.randint(0,100) / 100
    w3 = 0.783 #random.randint(0,100) / 100

    alpha = 0.3
    parada = True
    while parada:       
        pesos_etapa = []
        for conjunto in conjuntos_entrenamiento:
            
            
            #selecciona un valor de x1,x2 y umbral       
            etapa = Etapa(conjunto)
            etapa.inicializar_pesos(w1,w2,w3)
            
            # se calcula la red de salida y = f(x1w1 + w2x2 + ... + w3x3)          
            y = etapa.salida_red()
            etapa.hallar_error(y)
            etapa.modifica_pesos(alpha)

            w1 = etapa.w1
            w2 = etapa.w2
            w3 = etapa.w3
            pesos_etapa.append(etapa.error)


        parada = verificar_parada(pesos_etapa)

    return w1,w2,w3

def encontrar_valores(conjuntos_entrenamiento, w1, w2, w3):
    total_pesos = []
    for conjunto in conjuntos_entrenamiento:
        resultado = w1 * conjunto.x1 + w2 * conjunto.x2 + w3 * conjunto.x3
        pesos = np.array(['{}, {}, {}'.format(conjunto.x1,conjunto.x2,conjunto.x3),resultado])
        total_pesos.append(pesos)

    total_pesos = np.array(total_pesos)
    return total_pesos

def graficar(valores):

    #entrada = ['0,0,1','0,1,0','0,1,1','1,0,0','1,0,1','1,1,0','1,1,1']
    entrada = valores[:,0]
    deseado = valores[:,1]

    plt.figure(figsize=(20,10))
    plt.plot(entrada,deseado, marker="o")

    plt.yticks(deseado, rotation='vertical')
    plt.xticks(entrada)
    plt.title("Entradas vs. Esperados")
    plt.xlabel('Entradas')
    plt.ylabel('Esperadas')
    
    plt.show()

def convertir_2_decimales(peso):

  dec, unid = math.modf(peso)
  dec = (math.floor(dec * 10 **2))/10**2
  return round(unid + dec , 2)

def verificar_parada(pesos_etapas):
  suma = 0
  for peso in pesos_etapas:
    suma += peso
  suma /= 2

  if suma < 0:
    return False
  else:
    return True

def inicializar_input():

    conjuntos_entrenamiento = []
         
    conjuntos_entrenamiento.append(ConjuntoEntrenamiento(0,0,1,1))
    conjuntos_entrenamiento.append(ConjuntoEntrenamiento(0,1,0,2))
    conjuntos_entrenamiento.append(ConjuntoEntrenamiento(0,1,1,3))
    conjuntos_entrenamiento.append(ConjuntoEntrenamiento(1,0,0,4))
    conjuntos_entrenamiento.append(ConjuntoEntrenamiento(1,0,1,5))
    conjuntos_entrenamiento.append(ConjuntoEntrenamiento(1,1,0,6))
    conjuntos_entrenamiento.append(ConjuntoEntrenamiento(1,1,1,7))

    return conjuntos_entrenamiento

def main():

    conjuntos_entrenamiento = inicializar_input()
    w1, w2, w3 = algoritmo(conjuntos_entrenamiento)
    w1 = convertir_2_decimales(w1)
    w2 = convertir_2_decimales(w2)
    w3 = convertir_2_decimales(w3)
    entradas = encontrar_valores(conjuntos_entrenamiento,w1, w2, w3)
    graficar(entradas)

if __name__ == "__main__":
    main()