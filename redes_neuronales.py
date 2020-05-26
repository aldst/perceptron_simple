import matplotlib
from numpy import array, array_equal, dot, where
import matplotlib.pyplot as plt

def entrenamiento(datosEntrada, datosSalida, pesos):

    salida = array([[0.0, 0.0, 0.0, 0.0]]).T
    epocas= 1
    #graficar(datosEntrada, pesos)
    while (not (array_equal(salida, datosSalida))):
        salidaEpoca = []
        numeroFila = 0
        print('###################EPOCA ', epocas, '###################')
        for fila_epoca in datosEntrada:
            print('Entrada', fila_epoca)
            real = prediccion(fila_epoca, pesos)
            print('real: ', real)
            deseada = datosSalida[numeroFila, :]
            print('deseada: ', deseada)
            error = deseada - real
            print('error: ', error)
            if (error[0] != [0]):
                pesos = modificaPesos(pesos, fila_epoca, error)
            salidaEpoca.append(real[0])
            numeroFila = numeroFila + 1
            print('Peso:\n', pesos, '\n')
            print('------------------------------')
            #graficar(datosEntrada, pesos)
        salida = array([salidaEpoca]).T
        epocas = epocas + 1
        print('Salida:\n', salida, '\n')
    print('new_pesos:\n',pesos,'\n')
    graficar(datosEntrada, pesos)
    return pesos

def prediccion(fila, pesos):
    return where(dot(fila, pesos) >= 0, 1, -1)
    
def modificaPesos(weight, fila, e):
    print("peso inicial ", e * (array([fila]).T))
    return weight + (e * (array([fila]).T))

def graficar(data,pesos):

    x = data[:, 0]
    print(type(data[:, 0]))
    y = data[:, 1]
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, 'bo')
    plt.axvline(x=0, ymin=-1, ymax=1)
    plt.axhline(y=0, xmin=-1, xmax=1)
    lx = [-1, 1]
    ly = []
    ly.append((-pesos[0, 0] * -1) / pesos[1, 0] - pesos[2, 0] / pesos[1, 0])
    ly.append((-pesos[0, 0] * 1) / pesos[1, 0] - pesos[2, 0] / pesos[1, 0])
    print(lx)
    print(ly)
    plt.plot(lx, ly, 'r')
    plt.show()

def main():
    print('#######################INICIO#########################')
    BIAS = 1
    datosEntrenamiento = array([
        #x1, x2, const
        [-1, -1, BIAS],
        [1, -1, BIAS],
        [-1, 1, BIAS],
        [1, 1, BIAS]])
    salidaEsperada = array([
        [-1, -1, -1, 1]]).T
    pesos = array([[1, 1, 0.5]]).T

    print('Training_data_input:\n', datosEntrenamiento, '\n')
    print('Training_data_output:\n', salidaEsperada,'\n')
    print('Init_weight:\n', pesos, '\n')
    pesos = entrenamiento(datosEntrenamiento, salidaEsperada, pesos)
    print('new_pesos:\n',pesos,'\n')

main();