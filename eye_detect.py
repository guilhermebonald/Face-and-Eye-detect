import cv2

classificador = cv2.CascadeClassifier('cascades\haarcascade_eye.xml')

imagem = cv2.imread('pessoas\\pai.jpg')
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

e_detect = classificador.detectMultiScale(imagem_cinza)

for (x, y, l, a) in e_detect:
    cv2.rectangle(imagem, (x, y), (x + y, y + a), (0, 0, 255), 2)

cv2.imshow('Olhos Detectados', imagem)
cv2.waitKey()

