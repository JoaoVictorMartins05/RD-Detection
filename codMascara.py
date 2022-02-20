import cv2
import numpy as np
import glob
from PIL import Image


imagensMild = [] #array das imagens doentes
imagensSaudaveis = []
imagensModerate = []
imagensProliferate = []
imagensSevere = []


nomeMascaraSaudaveis = []
nomeMascaraMild = []
nomeMascaraModerate = []
nomeMascaraProliferate = []
nomeMascaraSevere = []


def trackbar_change(pos):
    pos = pos/100 # get a scaling factor from trackbar pos
    image = cv2.imread('image.png') # read image
    h = int(image.shape[0]*pos) # scale h
    w = int(image.shape[1]*pos) # scale w
    rsz_image = cv2.resize(image, (w, h)) # resize image
    cv2.resizeWindow('image_window', w, h) # resize window
    cv2.imshow('image_window', rsz_image) # display resized image
    return


# Ler a imagem


def listarArrayImagensSaudaveis(): #pegar as imagens da pasta e colocar em um array    
    for f in glob.glob('./database/imgSaudavel/*.png'): #lista todos os arquivos da pasta especifica       
        print(f)    #conferindo se pegou todas as imagens    
        nomeMascaraSaudaveis.append(f)
        imagensSaudaveis.append(cv2.imread(f,0)) #lendo as imagens da pasta para o array Imagens
        #(thresh, imagensDoentes[f]) = cv2.threshold(imagensDoentes[f], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #thresh = 127
        #imagensDoentes[f] = cv2.threshold(imagensDoentes[f], thresh, 255, cv2.THRESH_BINARY)[1]
        
        
def listarArrayImagensMild(): #pegar as imagens da pasta e colocar em um array    
    for f in glob.glob('./database/imgMild/*.png'): #lista todos os arquivos da pasta especifica       
        print(f)    #conferindo se pegou todas as imagens    
        nomeMascaraMild.append(f)
        imagensMild.append(cv2.imread(f, 0)) #lendo as imagens da pasta para o array Imagens
        #(thresh, imagensDoentes[f]) = cv2.threshold(imagensDoentes[f], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #thresh = 127
        #imagensDoentes[f] = cv2.threshold(imagensDoentes[f], thresh, 255, cv2.THRESH_BINARY)[1        


def listarArrayImagensModerate(): #pegar as imagens da pasta e colocar em um array    
    for f in glob.glob('./database/imgModerate/*.png'): #lista todos os arquivos da pasta especifica       
        print(f)    #conferindo se pegou todas as imagens    
        nomeMascaraModerate.append(f)
        imagensModerate.append(cv2.imread(f, 0)) #lendo as imagens da pasta para o array Imagens
        #(thresh, imagensDoentes[f]) = cv2.threshold(imagensDoentes[f], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #thresh = 127
        #imagensDoentes[f] = cv2.threshold(imagensDoentes[f], thresh, 255, cv2.THRESH_BINARY)[1   
        
def listarArrayImagensProliferate(): #pegar as imagens da pasta e colocar em um array    
    for f in glob.glob('./database/imgProliferate/*.png'): #lista todos os arquivos da pasta especifica       
        print(f)    #conferindo se pegou todas as imagens    
        nomeMascaraProliferate.append(f)
        imagensProliferate.append(cv2.imread(f, 0)) #lendo as imagens da pasta para o array Imagens
        #(thresh, imagensDoentes[f]) = cv2.threshold(imagensDoentes[f], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #thresh = 127
        #imagensDoentes[f] = cv2.threshold(imagensDoentes[f], thresh, 255, cv2.THRESH_BINARY)[1  
        

def listarArrayImagensSevere(): #pegar as imagens da pasta e colocar em um array    
    for f in glob.glob('./database/imgSevere/*.png'): #lista todos os arquivos da pasta especifica       
        print(f)    #conferindo se pegou todas as imagens    
        nomeMascaraSevere.append(f)
        imagensSevere.append(cv2.imread(f, 0)) #lendo as imagens da pasta para o array Imagens
        #(thresh, imagensDoentes[f]) = cv2.threshold(imagensDoentes[f], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #thresh = 127
        #imagensDoentes[f] = cv2.threshold(imagensDoentes[f], thresh, 255, cv2.THRESH_BINARY)[1  




def maskSaudavel():
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    for i in range (len(imagensSaudaveis)):
        img = clahe.apply(imagensSaudaveis[i])
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(blur,160,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((12,12),np.uint8)
        closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
        nome = nomeMascaraSaudaveis[i][23:]
        print(nome)
        nome = nome[:-3] + "jpeg"
        cv2.imwrite("./database_mask/imgSaudavel/" + nome, closing)
        
        
def maskMild():
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    for i in range (len(imagensMild)):
        img = clahe.apply(imagensMild[i])
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(blur,160,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((12,12),np.uint8)
        closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
        nome = nomeMascaraMild[i][19:]
        print(nome)
        nome = nome[:-3] + "jpeg"
        cv2.imwrite("./database_mask/imgMild/" + nome, closing)
        

def maskModerate():
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    for i in range (len(imagensModerate)):
        img = clahe.apply(imagensModerate[i])
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(blur,160,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((12,12),np.uint8)
        closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
        nome = nomeMascaraModerate[i][23:]
        print(nome)
        nome = nome[:-3] + "jpeg"
        cv2.imwrite("./database_mask/imgModerate/" + nome, closing)
        
        
def maskProliferate():
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    for i in range (len(imagensProliferate)):
        img = clahe.apply(imagensProliferate[i])
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(blur,160,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((12,12),np.uint8)
        closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
        nome = nomeMascaraProliferate[i][26:]
        print(nome)
        nome = nome[:-3] + "jpeg"
        cv2.imwrite("./database_mask/imgProliferate/" + nome, closing)
    
    
def maskSevere():
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    for i in range (len(imagensSevere)):
        img = clahe.apply(imagensSevere[i])
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(blur,160,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((12,12),np.uint8)
        closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
        nome = nomeMascaraSevere[i][21:]
        print(nome)
        nome = nome[:-3] + "jpeg"
        cv2.imwrite("./database_mask/imgSevere/" + nome, closing)



        
listarArrayImagensSaudaveis() 
maskSaudavel()  
    
listarArrayImagensMild()
maskMild()

listarArrayImagensModerate()
maskModerate()

listarArrayImagensProliferate()
maskProliferate()

listarArrayImagensSevere()
maskSevere()

        


