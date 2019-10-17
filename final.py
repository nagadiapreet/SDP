import Tkinter as tk
from Tkinter import *
from PIL import ImageTk, Image
import os
import tkFont

def showEmoji():
	if InputText.get() == "sad":
		panel.configure(image = SadEmotionImage, bg="black")
	elif InputText.get() == "happy":
		panel.configure(image = HappyEmotionImage, bg="black")
	elif InputText.get() == "worry":
		panel.configure(image = WorryEmotionImage, bg="black")
	elif InputText.get() == "angry":
		panel.configure(image = AngryEmotionImage, bg="black")
	else:
		panel.configure(image = NeutralEmotionImage, bg="black")
	panel.place(x=330, y=150)

master = tk.Tk()
master.geometry("800x500")

InputTextLabel = Label(master, text = "Enter Message: ", bg="black", fg="white").place(x = 30,y = 25)  
InputText = Entry(master, width="70")
InputText.place(x = 150, y = 25)  
checkEmotion = Button(master, text = "Check Emotion",activebackground = "black", activeforeground = "white", bg="white", command=showEmoji).place(x = 330, y = 65)

panel = tk.Label(master)
SadImagePath = "/home/nagadiapreet/Desktop/SDP/FrontEnd/emoji/sad.png"
SadEmotionImage = Image.open(SadImagePath)
SadEmotionImage = SadEmotionImage.resize((150, 150), Image.BILINEAR)
SadEmotionImage = ImageTk.PhotoImage(SadEmotionImage)

HappyImagePath = "/home/nagadiapreet/Desktop/SDP/FrontEnd/emoji/happy.png"
HappyEmotionImage = Image.open(HappyImagePath)
HappyEmotionImage = HappyEmotionImage.resize((150, 150), Image.BILINEAR)
HappyEmotionImage = ImageTk.PhotoImage(HappyEmotionImage)

AngryImagePath = "/home/nagadiapreet/Desktop/SDP/FrontEnd/emoji/Angry.png"
AngryEmotionImage = Image.open(AngryImagePath)
AngryEmotionImage = AngryEmotionImage.resize((150, 150), Image.BILINEAR)
AngryEmotionImage = ImageTk.PhotoImage(AngryEmotionImage)

NeutralImagePath = "/home/nagadiapreet/Desktop/SDP/FrontEnd/emoji/neutral.png"
NeutralEmotionImage = Image.open(NeutralImagePath)
NeutralEmotionImage = NeutralEmotionImage.resize((150, 150), Image.BILINEAR)
NeutralEmotionImage = ImageTk.PhotoImage(NeutralEmotionImage)

WorryImagePath = "/home/nagadiapreet/Desktop/SDP/FrontEnd/emoji/worry.png"
WorryEmotionImage = Image.open(WorryImagePath)
WorryEmotionImage = WorryEmotionImage.resize((150, 150), Image.BILINEAR)
WorryEmotionImage = ImageTk.PhotoImage(WorryEmotionImage)

master.title("AI Therapist")
master.configure(background='black')

master.mainloop()