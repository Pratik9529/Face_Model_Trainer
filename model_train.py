import tkinter as tk
from tkinter import  * 
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import shutil

def save_content_to_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        print(f"File saved successfully to {file_path}")
    except Exception as e:
        print(f"Error: {e}")

def addDataset():
    global new_dataset
    new_dataset=dataset_dir+"_preprocessed"
    shutil.copytree(src=dataset_dir,dst=new_dataset)
    cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    for folder in os.listdir(new_dataset):
        if os.path.isdir(new_dataset+'/'+folder):
            for file in os.listdir(new_dataset+'/'+folder):
                img=cv2.imread(new_dataset+'/'+folder+'/'+file)
                img_pil=Image.open(new_dataset+'/'+folder+'/'+file)
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces=cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                face=img_pil.crop((x,y,x+w,y+h))
                face.save(new_dataset+'/'+folder+'/'+file)

def rmdataset():
    shutil.rmtree(new_dataset)

root=tk.Tk()
root.state('zoomed')
root_Label=Label(root,text="Model Trainer")
root_Label.pack()
browse_label=Label(root,text="Select Dataset Folder")
browse_label.place(x=100,y=20)

def browse():
    global dataset_dir
    dataset_dir=filedialog.askdirectory(initialdir='/')
    dataset_textBox.insert("end",dataset_dir)

browse_button=Button(text="Select",command=browse)
browse_button.place(x=100,y=45)

dataset_label=Label(text="Dataset Location")
dataset_label.place(x=100,y=90)

dataset_textBox=Text(root,height=1)
dataset_textBox.place(x=100,y=120)

model_file_label=Label(text="Trained Model Path")
model_file_label.place(x=100,y=180)

model_file_path=Text(height=1)
model_file_path.place(x=100,y=210)

def get_dataset_partisions_tf(ds,train_split=0.8,test_split=0.2,shuffle_size=10000,shuffle=True):
    if shuffle:
        ds=ds.shuffle(shuffle_size,seed=12)
    train_size=int(train_split* len(ds))
    test_size=int(test_split* len(ds))
    train_ds=ds.take(train_size)
    test_ds=ds.skip(train_size)
    val_ds=test_ds.skip(test_size)
    test_ds=test_ds.take(test_size)
    return train_ds,test_ds,val_ds

def train():
    addDataset()
    dataset=tf.keras.preprocessing.image_dataset_from_directory(dataset_dir, shuffle=True,batch_size=1,image_size=(299,299),)
    labels=dataset.class_names
    save_content_to_file(dataset_dir+'/'+'labels.txt',labels)
    for image_batch,labels_batch in dataset.take(1):
        print(image_batch.shape)
        print(labels_batch.numpy())
        break
    train_size=int(0.8*len(dataset))
    test_size=int(0.2 * len(dataset))
    train_ds , test_ds, val_ds=get_dataset_partisions_tf(dataset)
    resize_and_rescale=tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Resizing(299,299),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    ])
    base_model=tf.keras.applications.InceptionV3(
    weights='imagenet',
    input_shape=(299,299,3),
    include_top=False,
    pooling='avg',
    classifier_activation='softmax',
    classes=len(labels)
    )
    base_model.trainable=False
    inputs=tf.keras.Input(shape=(299,299,3))
    x=resize_and_rescale(inputs)
    x=base_model(x,training=False)
    x=tf.keras.layers.Dense(128,activation='relu')(x)
    x=tf.keras.layers.Dropout(0.2)(x)
    outputs=tf.keras.layers.Dense(len(labels),activation='softmax')(x)
    model=tf.keras.Model(inputs,outputs)
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()
    history=model.fit(train_ds,validation_data=val_ds,batch_size=1,epochs=20)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model_Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','val'],loc='upper left')
    plt.show()
    model.save(dataset_dir+'.h5',)
    rmdataset()

train_button=Button(text="Train Data Model",command=train)
train_button.place(x=100,y=150)
root.mainloop()