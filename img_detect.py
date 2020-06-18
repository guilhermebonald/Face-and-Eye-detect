import cv2

classificador = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')
imagem = cv2.imread('pessoas\pessoas1.jpg')
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


# scaleFactor deve ser maior que 1. Auxilia na detecção de diferentes tamanhos de faces.
# minNeighbors define a distancia detectada pelos quadrados. Quanto maior o valor menor a distancia permitida.
f_detect = classificador.detectMultiScale(imagem_cinza, scaleFactor=1.07, minNeighbors=8, minSize=(30, 30)) 
#print(len(faces_detect)) // Retorna o número de fazer detectadas.


for (x, y, l, a) in f_detect:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)


cv2.imshow('Imagens Detectadas', imagem)
print(f'{len(f_detect)} faces detectadas.')
cv2.waitKey(0)
