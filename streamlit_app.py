import streamlit as st

## Load Model
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import webbrowser 

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def load_model():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(select_largest=False, post_process=False, device=device).eval()
    model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=device).eval()

    checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return mtcnn, model

# Prediction function
def predict(input_path:str, mtcnn, model):
    input_image = Image.open(input_path).convert('RGB')
    face = mtcnn(input_image)
    if face is None:
        return 'No face detected'

    face = face.unsqueeze(0)
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "Real" if output.item() < 0.1 else "Fake"

    return prediction


           


def home(mtcnn,model):
    st.title("Welcome to :violet[DeepScan]🔎 ")
    st.markdown('**Check for :green[REAL] or :red[FAKE]**')

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        input_path = "temp_image.jpg"
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        input_image = Image.open(uploaded_file).convert('RGB')
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(input_image, caption='Uploaded Image.', width=150)  # Adjust width as needed
            
        pred = None  # Initialize pred outside the if statement
        with col2:
            if st.button('Check'):
                pred = predict(input_path, mtcnn, model)
        if pred is not None:
            result(pred)
        
def result(pred):
    st.markdown("<h3><b>Result</b></h3>", unsafe_allow_html=True) 
    if pred == "Real":
                st.success(f"**{pred}**")  # Display "Real" prediction in green color
    else:
                st.error(f"**{pred}**")
    st.info("Note: Our deepfake detector is under development (~62% accuracy). We're aiming for 80%+ soon! For now, results may be unreliable. Double-check suspicious images using fact-checkers or reverse image search. Thanks for your patience!")

def contact():
    st.markdown(
        """
        <style>
        body {
            background-color: #121212; /* Set the background color to dark */
            color: white; /* Set the text color to white */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Contact us :green[✉]")

    # Use columns layout to display buttons in a row
    col1, col2, col3 = st.columns(3)

    
    if col1.button("🛠️ Developers"):
        st.write("🔗 Dhananjay Ambatwar")
        st.write("🔗 Aryan Tapase")
        st.write("🔗 Manasvi Mude")
        st.write("🔗 Shreya Jadhav")

    
    if col2.button("📩 Help Desk"):
        st.info("You can reach us via email at supportteam_deepscan@gmail.com.")
        st.write("For any inquiries, feel free to send us a message!")


    if col3.button("🌐 GitHub"):
        github_link = "https://github.com/dhananjaya2003/Deepfake_detection"
        webbrowser.open_new(github_link)



    
         



def main():
  st.markdown(
      """
      <script>
      document.addEventListener('DOMContentLoaded', function() {
          const sidebarToggle = document.getElementsByClassName('sidebar-toggle')[0];
          if (sidebarToggle) {
              sidebarToggle.click();
          }
      });
      </script>
      """,
      unsafe_allow_html=True
  )

  st.markdown(
      """
      <style>
      .sidebar-content {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
      }
      </style>
      """,
      unsafe_allow_html=True
  )

  
  col1, col2 = st.columns([1, 2])
  with col1:
      st.sidebar.image("DeepScanLogo.png", use_column_width=True)
      st.sidebar.markdown(
          """
          <div class="sidebar-content">
              <h1 style="font-weight: bold;">Deepfake Detector</h1>
          </div>
          """,
          unsafe_allow_html=True
      )
      st.sidebar.markdown('<br>', unsafe_allow_html=True)
      selected_page = st.sidebar.radio('**Select Page 📲**', ['🏠 Home', '📞 Contact us'])
  
  mtcnn, model = load_model()
  
  if selected_page == '🏠 Home':
          home(mtcnn, model)
  elif selected_page == '📞 Contact us':
          contact()

if __name__ == '__main__':
  main()

