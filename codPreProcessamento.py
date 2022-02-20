import numpy
import cv2
import os
import numpy
import glob



def pyCLAHE(inputImage):
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        outputImage = clahe.apply(inputImage)
        return outputImage
    except Exception as e:
        print('ERROR (pyCLAHE): ' + str(e))  


def pyGreen(inputImage):
    try:
        imageGreenBRG=inputImage[:,:,1]
        rgb = cv2.cvtColor(imageGreenBRG, cv2.COLOR_BGR2RGB)
        outputImage=rgb[:,:,1]
        return outputImage
    except Exception as e:
        print('ERROR (pyGreen): ' + str(e)) 


    
    
def listarArrayImagensSaudaveis(): #pegar as imagens da pasta e colocar em um array    
 #   i = 0    
    for f in glob.glob('./database/imgSaudavel/*.png'): #lista todos os arquivos da pasta especifica       
        img = cv2.imread(f)
        print('PROCESSANDO CAMADA GREEN: ' + f)
        outputImage = pyGreen(img)
        print('PROCESSANDO EQUALIZACAO CLAHE: ' + f)
        outputImage = pyCLAHE(outputImage)
        print('CALCULANDO NEGATIVO DE: ' + f)
        outputImage = cv2.bitwise_not(outputImage)
        #median = cv2.medianBlur(outputImage,3)
        gray = cv2.cvtColor(outputImage, cv2.COLOR_BGR2RGB)       
        caminho = "./database_Processada/imgSaudavel/img_"+ f[26:]
        cv2.imwrite(caminho ,gray)
       # i+=1
        
def listarArrayImagensMild(): #pegar as imagens da pasta e colocar em um array    
 #   i = 0    
    for f in glob.glob('./database/imgMild/*.png'): #lista todos os arquivos da pasta especifica       
        print(f)    #conferindo se pegou todas as imagens    
        img = cv2.imread(f)
        print('PROCESSANDO CAMADA GREEN: ' + f)
        outputImage = pyGreen(img)
        print('PROCESSANDO EQUALIZACAO CLAHE: ' + f)
        outputImage = pyCLAHE(outputImage)
        print('CALCULANDO NEGATIVO DE: ' + f)
        outputImage = cv2.bitwise_not(outputImage)
        #median = cv2.medianBlur(outputImage,3)
        gray = cv2.cvtColor(outputImage, cv2.COLOR_BGR2RGB)        
        caminho = "./database_Processada/imgMild/img_"+ f[22:]
        cv2.imwrite(caminho ,gray) 
        #i+=1
        
def listarArrayImagensModerate(): #pegar as imagens da pasta e colocar em um array    
 #   i = 0    
    for f in glob.glob('./database/imgModerate/*.png'): #lista todos os arquivos da pasta especifica       
        print(f)    #conferindo se pegou todas as imagens    
        img = cv2.imread(f)
        print('PROCESSANDO CAMADA GREEN: ' + f)
        outputImage = pyGreen(img)
        print('PROCESSANDO EQUALIZACAO CLAHE: ' + f)
        outputImage = pyCLAHE(outputImage)
        print('CALCULANDO NEGATIVO DE: ' + f)
        outputImage = cv2.bitwise_not(outputImage)
        median = cv2.medianBlur(outputImage,3)
        gray = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)        
        caminho = "./database_Processada/imgModerate/img_"+ f[26:]
        cv2.imwrite(caminho ,gray) 
        #i+=1
    

def listarArrayImagensProliferate(): #pegar as imagens da pasta e colocar em um array    
 #   i = 0    
    for f in glob.glob('./database/imgProliferate/*.png'): #lista todos os arquivos da pasta especifica       
        print(f)    #conferindo se pegou todas as imagens    
        img = cv2.imread(f)
        print('PROCESSANDO CAMADA GREEN: ' + f)
        outputImage = pyGreen(img)
        print('PROCESSANDO EQUALIZACAO CLAHE: ' + f)
        outputImage = pyCLAHE(outputImage)
        print('CALCULANDO NEGATIVO DE: ' + f)
        outputImage = cv2.bitwise_not(outputImage)
        median = cv2.medianBlur(outputImage,3)
        gray = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)        
        caminho = "./database_Processada/imgProliferate/img_"+ f[29:]
        cv2.imwrite(caminho ,gray) 

    
def listarArrayImagensSeveve(): #pegar as imagens da pasta e colocar em um array    
 #   i = 0    
    for f in glob.glob('./database/imgSevere/*.png'): #lista todos os arquivos da pasta especifica       
        print(f)    #conferindo se pegou todas as imagens    
        img = cv2.imread(f)
        print('PROCESSANDO CAMADA GREEN: ' + f)
        outputImage = pyGreen(img)
        print('PROCESSANDO EQUALIZACAO CLAHE: ' + f)
        outputImage = pyCLAHE(outputImage)
        print('CALCULANDO NEGATIVO DE: ' + f)
        outputImage = cv2.bitwise_not(outputImage)
        median = cv2.medianBlur(outputImage,3)
        gray = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)        
        caminho = "./database_Processada/imgSevere/img_"+ f[24:]
        cv2.imwrite(caminho ,gray) 
    
    
    
listarArrayImagensSaudaveis()
listarArrayImagensMild()
listarArrayImagensModerate()
listarArrayImagensProliferate()
listarArrayImagensSeveve()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    