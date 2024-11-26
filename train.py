import cv2 
import os 
import numpy as np

dataPath = "C:/Users/Enmanuel/Downloads/samuel/Arbuino/Dataset_faces"
dir_list = os.listdir(dataPath)
print("lista archivos",dir_list)

labels = []
faceData = []
label = 0 

for name_dir in dir_list:
    dir_path = dataPath + "/" + name_dir
    
    for file_name in os.listdir(dir_path):
        imagen_path = dir_path + "/" + file_name
        print(imagen_path)
        imagen = cv2.imread(imagen_path,0)
        # cv2.imshow("Imagen",imagen)
        # cv2.waitKey(10)

        faceData.append(imagen)
        labels.append(label)
    label += 1 

print("etiqueta 0 ", np.count_nonzero(np.array(labels)== 0))
print("etiqueta 1 ", np.count_nonzero(np.array(labels)== 1))

face_mask = cv2.face.LBPHFaceRecognizer_create()

print("entrenando ....")
face_mask.train(faceData,np.array(labels))

face_mask.write("face_mask_model.xml")
print("modelo almacenado")