from tkinter import *
import os
import tkinter.messagebox
import tkinter.filedialog
import math
import numpy
from final_code import predict
import cv2
from skimage.transform import resize
from PIL import ImageTk, Image
 #return [[],[]]

class ImageWindow:
    def __init__(self, master):

        #defining master window
        self.master=master
        master.geometry("1000x800")
        master.title("Mythodex")

        #defining variables
        self.images=[]
        self.predictions=[]
        self.current_image=None
        
        #defining widgets for browsing window
        self.background_image=ImageTk.PhotoImage(file="D:\\MIL\\background1.png",master=root)
        self.background_canvas=Canvas(master, height=800, width=1000, bg="cornsilk2")
        self.background_canvas.create_image(250,300,image=self.background_image, anchor=CENTER)
        self.browse_button=Button(self.background_canvas, text="Browse", activebackground="cornsilk3", bg="cornsilk2", command=self.browsefunc,state=NORMAL)
        self.location_entry=Entry(self.background_canvas, width=70, text="Enter image path here.")
        self.analyse_image_button=Button(self.background_canvas, text="Analyse Image", activebackground="cornsilk3", bg="cornsilk2", command=self.next_buttons, state=NORMAL)
        #self.welctext=Label(self.background_canvas, fg="")
        
        #placing widgets for resuly window
        self.image_canvas=Canvas(self.background_canvas, height=400, width=500, bg="cornsilk2")
        self.next_button=Button(self.background_canvas, text="Next", activebackground="cornsilk3", bg="cornsilk2", command=self.get_next_image)
        self.new_image=Button(self.background_canvas, text="New Image", command=self.prev_buttons)
        self.close_button=Button(self.background_canvas, text="Close", width=10, activebackground="cornsilk3", bg="cornsilk2", command=root.destroy)
        self.prediction_label=Message(self.background_canvas, text="Character: ", bg="cornsilk2")
        
        #placing widgets for 
        self.background_canvas.pack()
        self.location_entry.place(relx=0.55, rely=0.2)
        self.browse_button.place(relx=0.9, rely=0.23)
        self.analyse_image_button.place(relx=0.8, rely=0.9)
    
    def prev_buttons(self):
        self.image_canvas.delete("all")
        self.analyse_image_button.config(state=ACTIVE)
        self.browse_button.config(state="normal")
        self.location_entry.place(relx=0.1, rely=0.1)
        self.browse_button.place(relx=0.5, rely=0.1)
        self.analyse_image_button.place(relx=0.7, rely=0.9)

        self.image_canvas.place_forget()
        self.next_button.place_forget()
        self.new_image.place_forget()
        self.close_button.place_forget()
        self.prediction_label.place_forget()

    def browsefunc(self):
        self.browser = tkinter.filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        self.location_entry.delete(0,END)
        self.location_entry.insert(0,self.browser)

    def next_buttons(self):
        if not os.path.isfile(self.location_entry.get()):
            tkinter.messagebox.showinfo("Invalid directory.","Please enter a valid file path.")
            return
        self.analyse_image_button.config(state=DISABLED)
        self.browse_button.config(state=DISABLED)
        self.model_output=predict(self.location_entry.get())
        self.i=0
        print(self.model_output[-1])
        self.images=[x[0] for x in self.model_output[:-1]]
        self.predictions=[self.clean_predictions(x[1]) for x in self.model_output[:-1]]
        self.final_predictions=self.model_output[-1]

        self.location_entry.place_forget()
        self.browse_button.place_forget()
        self.analyse_image_button.place_forget()

        #placing widgets
        self.background_canvas.pack()
        self.image_canvas.place(anchor=CENTER, relx=0.5, rely=0.47)
        self.change_image(self.images[0])
        self.set_label_texts()
        self.next_button.place(anchor=CENTER, relx=0.95, rely=0.03)
        self.new_image.place(anchor=CENTER, relx=0.09, rely=0.03)
        self.close_button.place(anchor=CENTER, relx=0.9, rely=0.9)
        self.prediction_label.place(anchor=CENTER, relx=0.5, rely=0.85)

    def change_image(self, image):
        #function to set the image canvas to another image in the form of a numpy array 
        pil_image=Image.fromarray(image.astype("uint8"))
        scale = max(pil_image.size[0]/500, pil_image.size[1]/400)
        self.current_image = ImageTk.PhotoImage(image=pil_image.resize(tuple([int(x/scale) for x in pil_image.size])))
        self.image_canvas.create_image((250,250),image=self.current_image,anchor=CENTER)

    def set_label_texts(self):
        #sets label texts based on what labels are applicable
        self.prediction_label.config(text=self.predictions[self.i])

    def get_next_image(self):
        self.i=(self.i+1)%(3+len(self.images))
        if self.i-len(self.images) in range(3):
            self.image_canvas.delete("all")
            predict_texts=[open("D:\\ML\\Scenes\\"+x).read() for x in self.final_predictions]
            colors=["green", "orange", "red"]
            self.image_canvas.create_text(250,200, text=predict_texts[self.i-len(self.images)], fill=colors[self.i-len(self.images)], width=500, anchor=CENTER)
        else:
            self.image_canvas.delete("all")
            self.change_image(self.images[self.i])
            self.set_label_texts()
    
    def clean_predictions(self, predict_array):
        s=[]
        for entry in predict_array:
            curr=""
            for items in entry:
                #print(entry,"\n")
                if type(items)==type("S"):
                    curr+=items+": "
                else:
                    curr+=("Maybe " if items[1] else "")+items[0]
            s+=[curr]
        #print("here:",s)
        return "\n".join(s)

root=Tk()
root.resizable(False, False)
gui=ImageWindow(root)
root.mainloop()
#comment
#comment2
