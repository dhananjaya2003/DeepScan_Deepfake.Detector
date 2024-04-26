from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton,MDIconButton,MDRoundFlatButton,MDRectangleFlatButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.dialog import MDDialog
from kivymd.uix.screen import Screen
from kivymd.uix.label import MDLabel
from tkinter import filedialog
from kivy.uix.screenmanager import ScreenManager, Screen
import webbrowser
import kivy
from kivy.config import Config
Config.set('kivy', 'log_level', 'debug')



KV = '''
ScreenManager:
    LoginScreen:
    HomeScreen:
    ResultScreen:
    

<LoginScreen>:
    name: "login"
    
    MDTopAppBar:
        title:"Deepfake Detection"
        pos_hint:{'top':1}

    MDTextField:
        id: user_id
        icon_left:"clipboard-account"
        hint_text: "User ID or Email"
        pos_hint: {'center_x': 0.5, 'center_y': 0.7}
        size_hint_x: None
        width: 300
    
    MDTextField:
        id: password
        icon_left:"account-key"
        hint_text: "Password"
        password: True
        pos_hint: {'center_x': 0.5, 'center_y': 0.6}
        size_hint_x: None
        width: 300
    
    MDRectangleFlatButton:
        text: "Login"
        pos_hint: {'center_x': 0.5, 'center_y': 0.5}
        on_press: root.goto_home_screen()

    MDBottomAppBar:
        right_action_items: [["alert-circle-outline"]]
        pos_hint:{'bottom':1}
        



<HomeScreen>
    name:'home'
    Screen:
        MDNavigationLayout:

            ScreenManager:

                Screen:

                    MDBoxLayout:
                        orientation: 'vertical'
                        spacing:10

                        MDTopAppBar:
                            
                            title: "Home"
                            left_action_items: [["menu", lambda x: nav_drawer.set_state('open')]]
                            right_action_items: [["account-circle", lambda x: app.show_user_info()]]

                        AnchorLayout:
                            size_hint_y: None
                            height: dp(200)  
                            padding: dp(16)

                            AsyncImage:
                                source: 'logo.jpg'
                                size_hint: None, None
                                size: self.parent.size  # Set size to match parent size
                                allow_stretch: True  # Allow stretching the image to fit the size
                                keep_ratio: True  # Maintain the aspect ratio of the image
                                pos_hint: {'center_x': 0.5,'center_y': 0.5}  # Center the image

                        MDLabel:
                            text: "[b]Check for REAL or FAKE[/b]"
                            halign: "center"
                            pos_hint:{'center_x': 0.5,'center_y': 0.5}
                            multiline: True
                            markup:True


                        BoxLayout:
                            size_hint_y: None
                            height: self.minimum_height
                            padding: dp(8)
                            canvas.before:
                                Color:
                                    rgba: app.theme_cls.divider_color
                                RoundedRectangle:
                                    pos: self.pos
                                    size: self.size
                                    radius: [10]
                            MDLabel:
                                id: note_label
                                text: "To check image for 'Real' or 'Fake' Please Upload the image..."
                                halign: "center"
                                multiline: True
                                font_style: "Body1"
                                size_hint_y: None
                                height: self.texture_size[1]  
                                theme_text_color: "Secondary"
                                font_size: "12sp"  # Adjust the font size here


                        MDRectangleFlatButton:
                            text: "Upload File"
                            pos_hint: {'center_x': 0.5,'center_y': 0.5}
                            on_release: app.check()

                       

                        MDBottomAppBar:
                            pos_hint:{'bottom':1}

                            

            MDNavigationDrawer:
                id: nav_drawer

                BoxLayout:
                    orientation: 'vertical'

                    MDTopAppBar:
                        title: "Menu"
                        width:150

                    ScrollView:

                        MDList:
                            OneLineIconListItem:
                                text: "Github"
                                on_release: app.show_github_link()

                                IconLeftWidget:
                                    icon: "github"

                            OneLineIconListItem:
                                text: "Team"
                                on_release: app.show_team_members()

                                IconLeftWidget:
                                    icon: "account-group"

                            OneLineIconListItem:
                                text: "Contact"
                                on_release: app.show_contact()

                                IconLeftWidget:
                                    icon: "email"

                            OneLineIconListItem:
                                text: "Settings"
                                on_release: app.show_setting()

                                IconLeftWidget:
                                    icon: "cog"

<ResultScreen>:
    name: "result"
    
    BoxLayout:
        orientation: 'vertical'
        spacing:10
    

        MDTopAppBar:
            title:"Result"
            pos_hint:{'top':1}

        MDLabel:
            id: result_label
            text: "Result"
            halign: "center"
            font_style: "H6"
            bold:True

        BoxLayout:
            size_hint_y: None
            height: self.minimum_height
            padding: dp(8)
            canvas.before:
                Color:
                    rgba: app.theme_cls.divider_color
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [10]
            MDLabel:
                id: note_label
                text: "Note : Our deepfake detector is under development (~62% accuracy). We're aiming for 80%+ soon! For now, results may be unreliable. Double-check suspicious images using fact-checkers or reverse image search. Thanks for your patience!"
                halign: "center"
                multiline: True
                font_style: "Body1"
                size_hint_y: None
                height: self.texture_size[1]  # Adjust height to fit content
                theme_text_color: "Secondary"
                font_size: "12sp"  # Adjust the font size here
        
        MDRectangleFlatButton:
            text:"Back <--"
            left_icon:"page-previous"
            pos_hint: {'center_x': 0.5,'center_y':0.5}
            on_press: root.manager.current = "home"

        MDBottomAppBar:
            left_action_items:[["page-previous-outine"]]
            pos_hint:{'bottom':1}

        

'''

## Model Loading

import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import PIL
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()


model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()



## Interface
class LoginScreen(Screen):
    def goto_home_screen(self):
        user_id = self.ids.user_id.text
        password = self.ids.password.text
        
        if user_id and password:  
            self.manager.current = "home"
        else:
            self.show_warning_dialog()

    def show_warning_dialog(self):
        dialog = MDDialog(
            title="Warning",
            text="Please enter User ID and Password.",
            size_hint=(None, None),
            size=(300, 200), 
            buttons=[
                MDRectangleFlatButton(
                    text="OK", on_release=lambda *args: dialog.dismiss()
                )
            ],
        )
        dialog.open()

class HomeScreen(Screen):
    pass

class ResultScreen(Screen):
    pass


class Deepfake_DetecationApp(MDApp):
    def build(self):
        return Builder.load_string(KV)
    
    def predict(self,input_path:str):
        input_image = Image.open(input_path).convert('RGB')
        face = mtcnn(input_image)
        if face is None:
            return 'No face detected'
        face = face.unsqueeze(0)
        face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

        prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
        prev_face = prev_face.astype('uint8')

        face = face.to(DEVICE)
        face = face.to(torch.float32)
        face = face / 255.0
        face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

        target_layers = [model.block8.branch1[-1]]
        use_cuda = True if torch.cuda.is_available() else False

        with torch.no_grad():
            output = torch.sigmoid(model(face).squeeze(0))
            prediction = "Real" if output.item() < 0.1 else "Fake"

            real_prediction = 1 - output.item()
            fake_prediction = output.item()

            confidences = {
                'real': real_prediction,
                'fake': fake_prediction
            }
        return prediction
    
    def open_file_manager(self):
        filepath = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        return filepath

    def check(self):
        filepath = self.open_file_manager()
        
        preds = self.predict(filepath)
        self.root.current = "result"
        self.root.get_screen("result").ids.result_label.text = preds


    def show_github_link(self):
        github_link = "https://github.com/dhananjaya2003/Deepfake_detection"
        webbrowser.open_new(github_link)

    def show_team_members(self):
        team_members = ["Dhananjay Ramrao Ambatwar", "Aryan Shahaji Tapase", "Manasvi Harihar Mude", "Shreya Ramchandra Jadhav"]
        members_text = "\n".join(team_members)
        
        dialog = MDDialog(
            title="Team Members",
            text=members_text,
            buttons=[
                MDFlatButton(
                    text="Close", on_release=lambda *args: dialog.dismiss()
                )
            ],
        )
        dialog.open()

    def show_setting(self):
        dialog = MDDialog(
            title="Settings",
            text="You can't change settings due to copyrights...!",
            buttons=[
                MDFlatButton(
                    text="Close", on_release=lambda *args: dialog.dismiss()
                )
            ],
        )
        dialog.open()

    def show_user_info(self):
        user_id = self.root.get_screen("login").ids.user_id.text
        dialog = MDDialog(
            text="User ID: {}".format(user_id),
            buttons=[MDLabel(text="Close", on_touch_down=lambda x: dialog.dismiss())],
        )
        dialog.open()

    def show_contact(self):
        team_members = ["Help Desk", "   Email - support_team.deepfakedetection@gmail.com"]
        members_text = "\n".join(team_members)
        
        dialog = MDDialog(
            title="Contact",
            text=members_text,
            buttons=[
                MDFlatButton(
                    text="Close", on_release=lambda *args: dialog.dismiss()
                )
            ],
        )
        dialog.open()


    
    


Deepfake_DetecationApp().run()
