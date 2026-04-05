import os
from tkinter import filedialog
import customtkinter as ctk
import pyautogui
import pygetwindow
from PIL import ImageTk, Image

from predictions import predict

# Global variables for storing project paths
# Get the directory where this script is located
project_folder = os.path.dirname(os.path.abspath(__file__))
# Define the path to the images folder
folder_path = project_folder + '/images/'

# Used to store the selected image filename
filename = ""


class App(ctk.CTk):
    """
    Main application class for the Bone Fracture Detection GUI.
    Inherits from customtkinter.CTk to create a modern-looking window.
    """
    def __init__(self):
        super().__init__()

        # Set up the main window properties
        self.title("Bone Fracture Detection")
        self.geometry(f"{500}x{740}")
        
        # Create the top frame for the title and info button
        self.head_frame = ctk.CTkFrame(master=self)
        self.head_frame.pack(pady=20, padx=60, fill="both", expand=True)
        
        # Create the main content frame
        self.main_frame = ctk.CTkFrame(master=self)
        self.main_frame.pack(pady=20, padx=60, fill="both", expand=True)
        
        # Application title label
        self.head_label = ctk.CTkLabel(master=self.head_frame, text="Bone Fracture Detection",
                                       font=(ctk.CTkFont("Roboto"), 28))
        self.head_label.pack(pady=20, padx=10, anchor="nw", side="left")
        
        # Load and set the info icon
        img1 = ctk.CTkImage(Image.open(folder_path + "info.png"))
        self.img_label = ctk.CTkButton(master=self.head_frame, text="", image=img1, command=self.open_image_window,
                                       width=40, height=40)
        self.img_label.pack(pady=10, padx=10, anchor="nw", side="right")

        # Instruction label
        self.info_label = ctk.CTkLabel(master=self.main_frame,
                                       text="Bone fracture detection system, upload an x-ray image for fracture detection.",
                                       wraplength=300, font=(ctk.CTkFont("Roboto"), 18))
        self.info_label.pack(pady=10, padx=10)

        # Upload image button
        self.upload_btn = ctk.CTkButton(master=self.main_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=0, padx=1)

        # Frame for displaying the selected image
        self.frame2 = ctk.CTkFrame(master=self.main_frame, fg_color="transparent", width=256, height=256)
        self.frame2.pack(pady=10, padx=1)

        # Load a default placeholder image
        img = Image.open(folder_path + "Question_Mark.jpg")
        img_resized = img.resize((int(256 / img.height * img.width), 256))  # Maintain aspect ratio
        img = ImageTk.PhotoImage(img_resized)

        # Label to show the image
        self.img_label = ctk.CTkLabel(master=self.frame2, text="", image=img)
        self.img_label.pack(pady=1, padx=10)

        # Predict button to run the model
        self.predict_btn = ctk.CTkButton(master=self.main_frame, text="Predict", command=self.predict_gui)
        self.predict_btn.pack(pady=0, padx=1)

        # Frame to display prediction results
        self.result_frame = ctk.CTkFrame(master=self.main_frame, fg_color="transparent", width=200, height=100)
        self.result_frame.pack(pady=5, padx=5)

        # Loader label (can be used for an animated spinner while predicting)
        self.loader_label = ctk.CTkLabel(master=self.main_frame, width=100, height=100, text="")
        self.loader_label.pack(pady=3, padx=3)

        # Label for the predicted body part (bone type)
        self.res1_label = ctk.CTkLabel(master=self.result_frame, text="")
        self.res1_label.pack(pady=5, padx=20)

        # Label for the prediction result (Fractured / Normal)
        self.res2_label = ctk.CTkLabel(master=self.result_frame, text="")
        self.res2_label.pack(pady=5, padx=20)

        # Button to save the result screenshot (hidden by default)
        self.save_btn = ctk.CTkButton(master=self.result_frame, text="Save Result", command=self.save_result)

        # Label showing save status (e.g., "Saved!")
        self.save_label = ctk.CTkLabel(master=self.result_frame, text="")

        # Project Creator Footer
        self.footer_frame = ctk.CTkFrame(master=self, fg_color="transparent")
        self.footer_frame.pack(side="bottom", pady=15, fill="x")
        self.footer_label = ctk.CTkLabel(master=self.footer_frame, text="Project is created by Manjunath kotabagi", text_color="gray", font=(ctk.CTkFont("Roboto"), 14, "italic"))
        self.footer_label.pack()



    def upload_image(self):
        """
        Opens a file dialog to allow the user to select an X-ray image.
        Updates the GUI to display the selected image and resets results.
        """
        global filename
        f_types = [("All Files", "*.*")]
        # Open file dialog
        filename = filedialog.askopenfilename(filetypes=f_types, initialdir=project_folder+'/test/Wrist/')
        
        # Reset labels from previous predictions
        self.save_label.configure(text="")
        self.res2_label.configure(text="")
        self.res1_label.configure(text="")
        self.img_label.configure(self.frame2, text="", image="")
        
        # Load and resize the selected image
        img = Image.open(filename)
        img_resized = img.resize((int(256 / img.height * img.width), 256))  # new width & height
        img = ImageTk.PhotoImage(img_resized)
        
        # Display the loaded image
        self.img_label.configure(self.frame2, image=img, text="")
        self.img_label.image = img
        
        # Hide the save button since there are no new predictions yet
        self.save_btn.pack_forget()
        self.save_label.pack_forget()

    def predict_gui(self):
        """
        Runs the prediction pipeline on the uploaded image.
        Updates the GUI labels with the predicted bone type and fracture status.
        """
        global filename
        
        # First, predict the bone type or use the general model
        bone_type_result = predict(filename)
        # Second, predict the specific result (normal/fractured) based on the bone type
        result = predict(filename, bone_type_result)
        print(result)
        
        # Display the result (Fractured vs Normal) with colored text
        if result == 'fractured':
            self.res2_label.configure(text_color="RED", text="Result: Fractured", font=(ctk.CTkFont("Roboto"), 24))
        else:
            self.res2_label.configure(text_color="GREEN", text="Result: Normal", font=(ctk.CTkFont("Roboto"), 24))
            
        # Third, predict specific bone parts
        bone_type_result = predict(filename, "Parts")
        self.res1_label.configure(text="Type: " + bone_type_result, font=(ctk.CTkFont("Roboto"), 24))
        print(bone_type_result)
        
        # Show the save result button
        self.save_btn.pack(pady=10, padx=1)
        self.save_label.pack(pady=5, padx=20)

    def save_result(self):
        """
        Takes a screenshot of the application window to save the prediction results.
        Prompts the user to select the save location.
        """
        # Ask for save path
        tempdir = filedialog.asksaveasfilename(parent=self, initialdir=project_folder + '/PredictResults/',
                                               title='Please select a directory and filename', defaultextension=".png")
        screenshots_dir = tempdir
        
        # Find the application window to capture
        window = pygetwindow.getWindowsWithTitle('Bone Fracture Detection')[0]
        left, top = window.topleft
        right, bottom = window.bottomright
        
        # Take full screen screenshot
        pyautogui.screenshot(screenshots_dir)
        
        # Crop the screenshot to only include the app window (adjusting for borders)
        im = Image.open(screenshots_dir)
        im = im.crop((left + 10, top + 35, right - 10, bottom - 10))
        im.save(screenshots_dir)
        
        # Indicate that the save was successful
        self.save_label.configure(text_color="WHITE", text="Saved!", font=(ctk.CTkFont("Roboto"), 16))

    def open_image_window(self):
        """
        Opens a separate window to display the rules/instructions image.
        """
        im = Image.open(folder_path + "rules.jpeg")
        im = im.resize((700, 700))
        im.show()


if __name__ == "__main__":
    app = App()
    app.mainloop()
